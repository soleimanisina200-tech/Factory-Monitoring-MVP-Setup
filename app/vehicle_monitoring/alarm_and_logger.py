import csv
import time
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np


class SafetyAlarmAndLogger:
    """
    Provides non-blocking audible/visual alarm activation and persistent event logging
    (SQLite + CSV + Annotated Video Snapshots) for all driver-vehicle events.
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

    def process_events(self, events: List[Dict], annotated_frame: np.ndarray):
        now = time.time()
        for ev in events:
            vid = str(ev.get("vehicle_id", "unknown"))
            ev_type = ev.get("event_type", "UNKNOWN")

            # Enforce cooldown for repeat violation alarms on the same vehicle
            if ev_type == "SAFE_DISTANCE_VIOLATION_ALARM":
                last_ts = self._last_alarm_ts.get(vid, 0.0)
                if (now - last_ts) < self.cooldown_sec:
                    continue
                self._last_alarm_ts[vid] = now
                self._trigger_audible_beep_async()

            ts_iso = datetime.fromtimestamp(ev.get("timestamp", now)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            snap_filename = f"{ev_type}_V{vid}_D{ev.get('driver_id')}_{int(now * 1000)}.jpg"
            snap_full_path = self.snapshot_dir / snap_filename

            cv2.imwrite(str(snap_full_path), annotated_frame)

            record = {
                "timestamp_iso": ts_iso,
                "event_type": ev_type,
                "vehicle_id": vid,
                "driver_id": str(ev.get("driver_id", "")),
                "distance_m": ev.get("distance_m"),
                "threshold_m": ev.get("threshold_m"),
                "snapshot_path": str(snap_full_path),
            }

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
                writer = csv.writer(f)
                writer.writerow([
                    record["timestamp_iso"],
                    record["event_type"],
                    record["vehicle_id"],
                    record["driver_id"],
                    record["distance_m"],
                    record["threshold_m"],
                    record["snapshot_path"],
                ])

            self._recent_events.insert(0, record)
            self._recent_events = self._recent_events[:50]

    def get_recent_events(self, limit: int = 20) -> List[Dict]:
        return self._recent_events[:limit]