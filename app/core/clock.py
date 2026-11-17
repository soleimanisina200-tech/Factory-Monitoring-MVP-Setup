# app/core/clock.py
"""
One clock for every module.

All per-frame logic runs on the STREAM clock (`frame_ts`, seconds):
  * live camera -> time.monotonic() at capture
  * video file  -> presentation time of the frame inside the file
Reports and audit records need WALL-CLOCK time. StreamClock converts between the two, so an event raised while a
recorded clip is processed 3x faster than real time still carries the moment it happened in the clip.
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Optional


class StreamClock:
    def __init__(self, is_live: bool, file_start_wall: Optional[float] = None):
        self.is_live = bool(is_live)
        if self.is_live:
            # wall = monotonic + offset (offset measured once: immune to later NTP jumps inside the run)
            self._offset = time.time() - time.monotonic()
        else:
            # wall = start of the recording + position inside the file
            self._offset = float(file_start_wall) if file_start_wall is not None else time.time()

    def wall(self, ts: float) -> float:
        return self._offset + float(ts)

    def from_wall(self, wall_epoch: float) -> float:
        return float(wall_epoch) - self._offset

    def iso(self, ts: float) -> str:
        return datetime.fromtimestamp(self.wall(ts)).isoformat(sep=" ", timespec="milliseconds")

    @staticmethod
    def parse_start_time(text: Optional[str]) -> Optional[float]:
        """'2026-11-17 08:00:00' (local time) -> epoch seconds; None/'' -> None."""
        if not text:
            return None
        return datetime.fromisoformat(str(text).strip()).timestamp()
