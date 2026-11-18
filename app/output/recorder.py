# app/output/recorder.py
"""
Annotated video output.

segment_minutes = 0  -> one file (offline processing of a clip; the old behaviour)
segment_minutes > 0  -> <stem>_YYYYMMDD_HHMMSS<ext> files of N minutes, keeping the newest `max_segments`.
A 24/7 service must not write one ever-growing MP4: it is unreadable until closed (a crash loses everything)
and eventually fills the disk.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("FactoryReID")


def resolve_output_path(value, source, camera_id: str = "cam1") -> str:
    """recording.output_video_path -> a real file path ('' = no video).

    "auto": a processed video FILE is saved as outputs/<video name>_annotated.mp4, so every clip keeps its own result
    (the old fixed name was overwritten by the next run); a live camera goes to outputs/<camera_id>_live.mp4.
    """
    value = "" if value is None else str(value).strip()
    if value.lower() != "auto":
        return value
    src = str(source)
    if src.isdigit() or "://" in src:
        return str(Path("outputs") / f"{camera_id}_live.mp4")
    stem = Path(src).stem or camera_id
    return str(Path("outputs") / f"{stem}_annotated.mp4")


class SegmentedVideoWriter:
    def __init__(self, output_path: str, fps: float, frame_wh: Tuple[int, int], segment_minutes: float = 0.0,
                 max_segments: int = 96, fourcc: str = "mp4v"):
        self.base = Path(output_path)
        self.base.parent.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps) if fps and fps > 0 else 25.0
        self.wh = (int(frame_wh[0]), int(frame_wh[1]))
        self.segment_sec = float(segment_minutes) * 60.0
        self.max_segments = int(max_segments)
        self.fourcc = cv2.VideoWriter_fourcc(*fourcc)
        self._writer: Optional[cv2.VideoWriter] = None
        self._segment_start: Optional[float] = None
        self._segments: List[Path] = []
        self.current_path: Optional[Path] = None

    def _open(self, wall_time: float) -> None:
        if self.segment_sec > 0:
            stamp = datetime.fromtimestamp(wall_time).strftime("%Y%m%d_%H%M%S")
            path = self.base.with_name(f"{self.base.stem}_{stamp}{self.base.suffix or '.mp4'}")
        else:
            path = self.base
        self._writer = cv2.VideoWriter(str(path), self.fourcc, self.fps, self.wh)
        if not self._writer.isOpened():
            logger.error("Cannot open video writer %s", path)
        self.current_path = path
        self._segment_start = wall_time
        if self.segment_sec > 0:
            self._segments.append(path)
            while len(self._segments) > self.max_segments:
                old = self._segments.pop(0)
                try:
                    old.unlink()
                except OSError:
                    pass

    def write(self, frame: np.ndarray, wall_time: float) -> None:
        if self._writer is None:
            self._open(wall_time)
        elif self.segment_sec > 0 and wall_time - (self._segment_start or wall_time) >= self.segment_sec:
            self._writer.release()
            self._open(wall_time)
        if (frame.shape[1], frame.shape[0]) != self.wh:
            frame = cv2.resize(frame, self.wh)
        self._writer.write(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
