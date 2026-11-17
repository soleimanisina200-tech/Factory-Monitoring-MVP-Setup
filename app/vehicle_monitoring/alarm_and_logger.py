import csv
import logging
import queue
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np

logger = logging.getLogger("FactoryReID")

# Event types that sound the alarm and are rate-limited per vehicle.
ALARM_EVENT_TYPES = ("SAFE_DISTANCE_VIOLATION_ALARM", "DRIVER_NOT_VISIBLE_ALARM")


class SafetyAlarmAndLogger:
    """
    Provides non-blocking audible/visual alarm activation and persistent event logging
    (SQLite + CSV + Annotated Video Snapshots) for all driver-vehicle events.

    Disk I/O (JPEG snapshot, SQLite, CSV) runs on a background thread so the real-time video loop is never
    blocked by storage latency. `get_recent_events()` is updated immediately.
    """

    def __init__(self, config: dict):
        alarm_cfg = config.get("alarms", {})
        log_cfg = config.get("logging", {})

        self.enable_audio: bool = bool(alarm_cfg.get("enable_audio_alarm", True))
        self.enable_visual: bool = bool(alarm_cfg.get("enable_visual_alarm", True))
        self.cooldown_sec: float = float(alarm_cfg.get("alarm_cooldown_sec", 5.0))
        self.freq_hz: int = int(alarm_cfg.get("audio_frequency_hz", 2200))
        self.duration_ms: int = int(alarm_cfg.get("audio_duration_ms", 400))

        self.db_path = Path(log_cfg.get("db_path", "data/vehicle_safety_events.db"))
        self.csv_path = Path(log_cfg.get("csv_log_path", "data/vehicle_safety_events.csv"))
        self.snapshot_dir = Path(log_cfg.get("snapshot_dir", "data/snapshots/vehicle_safety"))

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        self._last_alarm_ts: Dict[str, float] = {}
        self._recent_events: List[Dict] = []
        self._init_storage()

        self._queue: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=512)
        self._worker = threading.Thread(target=self._writer_loop, name="safety-log-writer", daemon=True)
        self._worker.start()

    def _init_storage(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vehicle_safety_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_iso TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    vehicle_id TEXT NOT NULL,
                    driver_id TEXT,
                    distance_m REAL,
                    threshold_m REAL,
                    snapshot_path TEXT
                )
            """)
            conn.commit()

        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp_iso", "event_type", "vehicle_id",
                    "driver_id", "distance_m", "threshold_m", "snapshot_path"
                ])

    def _trigger_audible_beep_async(self):
        if not self.enable_audio:
            return

        def _sound_worker():
            try:
                import winsound
                winsound.Beep(self.freq_hz, self.duration_ms)
            except ImportError:
                # Cross-platform terminal bell fallback on Linux/WSL
                print("\a", end="", flush=True)

        threading.Thread(target=_sound_worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Background persistence
    # ------------------------------------------------------------------
    def _persist(self, record: Dict, frame: np.ndarray) -> None:
        cv2.imwrite(record["snapshot_path"], frame)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO vehicle_safety_events
                (timestamp_iso, event_type, vehicle_id, driver_id, distance_m, threshold_m, snapshot_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                record["timestamp_iso"],
                record["event_type"],
                record["vehicle_id"],
                record["driver_id"],
                record["distance_m"],
                record["threshold_m"],
                record["snapshot_path"],
            ))
            conn.commit()

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                record["timestamp_iso"],
                record["event_type"],
                record["vehicle_id"],
                record["driver_id"],
                record["distance_m"],
                record["threshold_m"],
                record["snapshot_path"],
            ])

    def _writer_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            record, frame = item
            try:
                self._persist(record, frame)
            except Exception:
                logger.exception("Failed to persist safety event %s", record.get("event_type"))
            finally:
                self._queue.task_done()

    def flush(self, timeout: float = 5.0) -> None:
        """Block until queued events are written (used on shutdown and in tests)."""
        deadline = time.time() + timeout
        while self._queue.unfinished_tasks and time.time() < deadline:
            time.sleep(0.01)

    def close(self) -> None:
        self.flush()
        self._queue.put(None)

    # ------------------------------------------------------------------
    def process_events(self, events: List[Dict], annotated_frame: np.ndarray):
        now = time.time()
        for ev in events:
            vid = str(ev.get("vehicle_id", "unknown"))
            ev_type = ev.get("event_type", "UNKNOWN")

            # Enforce cooldown for repeat alarms on the same vehicle
            if ev_type in ALARM_EVENT_TYPES:
                last_ts = self._last_alarm_ts.get(vid, 0.0)
                if (now - last_ts) < self.cooldown_sec:
                    continue
                self._last_alarm_ts[vid] = now
                self._trigger_audible_beep_async()

            ts_iso = datetime.fromtimestamp(ev.get("timestamp", now)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            snap_filename = f"{ev_type}_V{vid}_D{ev.get('driver_id')}_{int(now * 1000)}.jpg"
            snap_full_path = self.snapshot_dir / snap_filename

            record = {
                "timestamp_iso": ts_iso,
                "event_type": ev_type,
                "vehicle_id": vid,
                "driver_id": str(ev.get("driver_id", "")),
                "distance_m": ev.get("distance_m"),
                "threshold_m": ev.get("threshold_m"),
                "snapshot_path": str(snap_full_path),
            }

            try:
                self._queue.put_nowait((record, annotated_frame.copy()))
            except queue.Full:
                # Never lose an audit record: fall back to writing synchronously.
                logger.warning("Safety log queue full; writing event synchronously")
                try:
                    self._persist(record, annotated_frame)
                except Exception:
                    logger.exception("Failed to persist safety event %s", ev_type)

            self._recent_events.insert(0, record)
            self._recent_events = self._recent_events[:50]

    def get_recent_events(self, limit: int = 20) -> List[Dict]:
        return self._recent_events[:limit]
