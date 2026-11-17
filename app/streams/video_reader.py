# app/streams/video_reader.py
"""
Camera / video-file reader for a 24/7 service.

Live sources (RTSP/RTMP/HTTP(S)/UDP/device index):
  * a background thread always keeps only the NEWEST frame (the pipeline never processes stale video);
  * RTSP over TCP by default (UDP packet loss produces smeared grey frames that ruin detection);
  * automatic reconnect with exponential back-off, logged; the frame size is kept constant after a reconnect
    (all geometry - calibration, zones, lines - is in stream pixels);
  * read() waits at most `read_timeout_sec` and never blocks forever: the caller decides what to do when the
    camera is down (status() tells it).
Files: sequential read with the presentation timestamp of each frame; `ended` becomes True at the end.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger("FactoryReID")

LIVE_PREFIXES = ("rtsp://", "rtsps://", "rtmp://", "http://", "https://", "udp://", "tcp://")


def is_live_source(source: Union[str, int]) -> bool:
    return isinstance(source, int) or str(source).isdigit() or str(source).lower().startswith(LIVE_PREFIXES)


class VariableFPSVideoReader:
    def __init__(
        self,
        source: Union[str, int],
        drop_frames_on_lag: bool = True,
        reconnect_delay_sec: float = 3.0,
        max_reconnect_delay_sec: float = 30.0,
        rtsp_transport: str = "tcp",
        read_timeout_sec: float = 5.0,
    ):
        self.source = int(source) if str(source).isdigit() else source
        self.is_live = is_live_source(self.source)
        self.drop_frames_on_lag = bool(drop_frames_on_lag) and self.is_live
        self.reconnect_delay_sec = float(reconnect_delay_sec)
        self.max_reconnect_delay_sec = float(max_reconnect_delay_sec)
        self.read_timeout_sec = float(read_timeout_sec)
        if (self.is_live and isinstance(self.source, str) and self.source.lower().startswith(("rtsp", "rtsps"))
                and rtsp_transport and "OPENCV_FFMPEG_CAPTURE_OPTIONS" not in os.environ):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{rtsp_transport}"

        self.cap = self._open()
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self._safe_name()}")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        raw_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.nominal_fps = raw_fps if 1.0 <= raw_fps <= 120.0 else 30.0

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_ts = 0.0
        self._frame_ready = False
        self._stopped = False
        self._ended = False
        self._frame_idx = 0
        self.frames_received = 0
        self.reconnects = 0
        self.last_frame_wall = time.time()
        self.connected = True
        self._resize_warned = False

        if self.drop_frames_on_lag:
            self._thread = threading.Thread(target=self._update_loop, name="video-grabber", daemon=True)
            self._thread.start()

    def _safe_name(self) -> str:
        s = str(self.source)
        if "@" in s and "://" in s:  # hide credentials in logs
            scheme, rest = s.split("://", 1)
            return f"{scheme}://***@{rest.split('@', 1)[1]}"
        return s

    def _open(self) -> cv2.VideoCapture:
        if isinstance(self.source, str) and self.is_live:
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(self.source)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    def _fit(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if (w, h) != (self.width, self.height):
            if not self._resize_warned:
                logger.warning("Stream resolution changed to %dx%d: frames are resized to the original %dx%d",
                               w, h, self.width, self.height)
                self._resize_warned = True
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame

    def _update_loop(self) -> None:
        delay = self.reconnect_delay_sec
        while not self._stopped:
            ret, frame = self.cap.read()
            now_ts = time.monotonic()
            if not ret or frame is None:
                self.connected = False
                logger.warning("Camera %s: no frame - reconnecting in %.0f s", self._safe_name(), delay)
                time.sleep(delay)
                if self._stopped:
                    break
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = self._open()
                self.reconnects += 1
                if self.cap.isOpened():
                    logger.info("Camera %s: reconnected (#%d)", self._safe_name(), self.reconnects)
                delay = min(self.max_reconnect_delay_sec, delay * 2.0)
                continue
            delay = self.reconnect_delay_sec
            self.connected = True
            frame = self._fit(frame)
            with self._cond:
                self._latest_frame = frame
                self._latest_ts = now_ts
                self._frame_ready = True
                self.frames_received += 1
                self.last_frame_wall = time.time()
                self._cond.notify_all()

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """(success, frame, timestamp_sec). Live: waits at most read_timeout_sec for a NEW frame."""
        if self.drop_frames_on_lag:
            deadline = time.monotonic() + self.read_timeout_sec
            with self._cond:
                while not self._frame_ready and not self._stopped:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False, None, time.monotonic()
                    self._cond.wait(timeout=min(remaining, 0.5))
                if self._stopped or self._latest_frame is None:
                    return False, None, time.monotonic()
                self._frame_ready = False
                return True, self._latest_frame, self._latest_ts
        ret, frame = self.cap.read()
        if not ret or frame is None:
            if self.is_live:
                return False, None, time.monotonic()
            self._ended = True
            return False, None, 0.0
        self._frame_idx += 1
        self.frames_received += 1
        self.last_frame_wall = time.time()
        if self.is_live:
            return True, self._fit(frame), time.monotonic()
        pts_msec = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        ts = (pts_msec / 1000.0) if pts_msec > 0 else (self._frame_idx / self.nominal_fps)
        return True, frame, float(ts)

    @property
    def ended(self) -> bool:
        """True when a FILE reached its end or the reader was released (a live camera never 'ends')."""
        return self._ended or self._stopped

    def status(self) -> Dict[str, Any]:
        return {
            "source": self._safe_name(),
            "live": self.is_live,
            "connected": bool(self.connected),
            "resolution": [self.width, self.height],
            "nominal_fps": round(self.nominal_fps, 2),
            "frames_received": self.frames_received,
            "reconnects": self.reconnects,
            "seconds_since_last_frame": round(time.time() - self.last_frame_wall, 1),
        }

    def release(self) -> None:
        self._stopped = True
        with self._cond:
            self._cond.notify_all()
        if self.drop_frames_on_lag and hasattr(self, "_thread"):
            self._thread.join(timeout=2.0)
        try:
            if self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass
