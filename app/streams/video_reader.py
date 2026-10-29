# app/streams/video_reader.py
from __future__ import annotations
import threading
import time
from typing import Optional, Tuple, Union
import cv2
import numpy as np


class VariableFPSVideoReader:
    """
    Camera-agnostic RTSP / Video File Reader.
    - For live RTSP/webcam streams: drains the internal OpenCV buffer on a background thread
      so the pipeline always processes the newest frame with its true wall-clock timestamp.
    - For recorded video files: reads sequentially and computes exact PTS timestamps.
    """

    def __init__(
        self,
        source: Union[str, int],
        drop_frames_on_lag: bool = True,
        reconnect_delay_sec: float = 3.0,
    ):
        self.source = int(source) if str(source).isdigit() else source
        self.is_live = isinstance(self.source, int) or str(self.source).startswith(("rtsp://", "rtmp://", "http://"))
        self.drop_frames_on_lag = drop_frames_on_lag and self.is_live
        self.reconnect_delay_sec = reconnect_delay_sec

        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.source}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        raw_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.nominal_fps = raw_fps if 1.0 <= raw_fps <= 120.0 else 30.0

        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_ts: float = 0.0
        self._frame_ready = False
        self._stopped = False
        self._frame_idx = 0

        if self.drop_frames_on_lag:
            self._thread = threading.Thread(target=self._update_loop, daemon=True)
            self._thread.start()

    def _update_loop(self) -> None:
        while not self._stopped:
            ret, frame = self.cap.read()
            now_ts = time.monotonic()
            if not ret or frame is None:
                time.sleep(self.reconnect_delay_sec)
                self.cap.release()
                self.cap = cv2.VideoCapture(self.source)
                continue

            with self._lock:
                self._latest_frame = frame
                self._latest_ts = now_ts
                self._frame_ready = True

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """Returns (success, frame, monotonic_timestamp_sec)."""
        if self.drop_frames_on_lag:
            while not self._stopped:
                with self._lock:
                    if self._frame_ready and self._latest_frame is not None:
                        frame = self._latest_frame.copy()
                        ts = self._latest_ts
                        self._frame_ready = False
                        return True, frame, ts
                time.sleep(0.002)
            return False, None, 0.0
        else:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                return False, None, 0.0
            self._frame_idx += 1
            # Use presentation timestamp (PTS) if available, otherwise fallback to frame_idx / nominal_fps
            pts_msec = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            ts = (pts_msec / 1000.0) if pts_msec > 0 else (self._frame_idx / self.nominal_fps)
            return True, frame, float(ts)

    def release(self) -> None:
        self._stopped = True
        if self.drop_frames_on_lag and hasattr(self, "_thread"):
            self._thread.join(timeout=1.0)
        if self.cap.isOpened():
            self.cap.release()