# app/vehicles/alarm_and_logger.py
import csv
import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np

from app.vehicles.association import SafetyEvent, VehicleSafetyState


class AlarmAndEventLogger:
    """
    Handles non-blocking audible alarms, isolated SQLite event storage,
    JSONL/CSV structured audit logging, and annotated video snapshot capture
    without modifying any existing project database or logging files.
    """

    def __init__(self, config: Dict):
        # Support both 'alarms_and_logging' and split 'alarms'/'logging' config sections
        al_cfg = config.get("alarms_and_logging", {})
        alarm_cfg = config.get("alarms", {})
        log_cfg = config.get("logging", {})

        self.camera_id: str = str(config.get("camera_id", "CAM_01_WAREHOUSE"))
        self.audio_enabled: bool = bool(
            al_cfg.get("audio_enabled", alarm_cfg.get("enable_audio_alarm", True))
        )
        self.alarm_cooldown_sec: float = float(
            al_cfg.get("alarm_cooldown_sec", alarm_cfg.get("alarm_cooldown_sec", 5.0))
        )
        self.audio_freq_hz: int = int(alarm_cfg.get("audio_frequency_hz", 2200))
        self.audio_duration_ms: int = int(alarm_cfg.get("audio_duration_ms", 450))

        self.db_path = Path(
            al_cfg.get("db_path", log_cfg.get("db_path", "data/vehicle_events.db"))
        )
        self.jsonl_path = Path(
            al_cfg.get("jsonl_path", "logs/vehicle_events.jsonl")
        )
        self.csv_path = Path(
            log_cfg.get("csv_log_path", "logs/vehicle_events.csv")
        )
        self.snapshot_dir = Path(
            al_cfg.get(
                "snapshot_dir",
                log_cfg.get("snapshot_dir", "logs/snapshots"),
            )
        )

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        self._last_alarm_time: Dict[int, float] = {}
        self._recent_events: List[Dict] = []
        self._db_lock = threading.Lock()

        self._init_storage()

    def _init_storage(self) -> None:
        """Initializes the isolated SQLite table and CSV header if they do not exist."""
        with self._db_lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vehicle_safety_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_utc TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    forklift_id INTEGER NOT NULL,
                    driver_id INTEGER,
                    state TEXT NOT NULL,
                    distance_m REAL,
                    threshold_m REAL NOT NULL,
                    uncertainty_reason TEXT,
                    vehicle_bbox TEXT,
                    driver_bbox TEXT,
                    snapshot_path TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_veh_events_ts ON vehicle_safety_events(timestamp_utc)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_veh_events_type ON vehicle_safety_events(event_type)"
            )
            conn.commit()

        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp_utc",
                        "camera_id",
                        "event_type",
                        "forklift_id",
                        "driver_id",
                        "state",
                        "distance_m",
                        "threshold_m",
                        "uncertainty_reason",
                        "snapshot_path",
                    ]
                )

    def trigger_audible_alarm(self, forklift_id: int) -> None:
        """Fires a non-blocking audible alert respecting per-forklift cooldown."""
        if not self.audio_enabled:
            return

        now = time.monotonic()
        last_t = self._last_alarm_time.get(forklift_id, 0.0)
        if (now - last_t) < self.alarm_cooldown_sec:
            return
        self._last_alarm_time[forklift_id] = now

        def _beep_worker() -> None:
            try:
                import winsound
                # Double high-visibility industrial safety beep on Windows
                winsound.Beep(self.audio_freq_hz, self.audio_duration_ms)
                time.sleep(0.08)
                winsound.Beep(self.audio_freq_hz + 300, self.audio_duration_ms)
            except Exception:
                # Cross-platform console bell fallback for Linux / WSL
                print("\a", end="", flush=True)

        threading.Thread(target=_beep_worker, daemon=True).start()

    def _save_event_snapshot(
        self, frame: np.ndarray, event: SafetyEvent, ts_slug: str
    ) -> str:
        """Saves an annotated snapshot highlighting the forklift, driver, and violation details."""
        snap = frame.copy()
        vx1, vy1, vx2, vy2 = map(int, event.vehicle_bbox)
        is_alarm = event.event_type == "SAFE_DISTANCE_VIOLATION"
        box_color = (0, 0, 255) if is_alarm else (0, 200, 255)

        cv2.rectangle(snap, (vx1, vy1), (vx2, vy2), box_color, 3)
        cv2.putText(
            snap,
            f"Forklift #{event.forklift_id}",
            (vx1, max(25, vy1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            box_color,
            2,
        )

        if event.driver_bbox is not None:
            dx1, dy1, dx2, dy2 = map(int, event.driver_bbox)
            cv2.rectangle(snap, (dx1, dy1), (dx2, dy2), box_color, 2)
            cv2.putText(
                snap,
                f"Driver #{event.driver_id}",
                (dx1, max(25, dy1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                box_color,
                2,
            )
            # Draw line connecting driver ground point and forklift base center
            drv_foot = (int((dx1 + dx2) / 2), dy2)
            veh_base = (int((vx1 + vx2) / 2), vy2)
            cv2.line(snap, drv_foot, veh_base, box_color, 2, cv2.LINE_AA)

        dist_str = (
            f"{event.distance_m:.2f}m"
            if event.distance_m is not None
            else f"UNCERTAIN ({event.uncertainty_reason})"
        )
        banner = (
            f"{event.event_type} | Forklift #{event.forklift_id} | "
            f"Driver #{event.driver_id} | Dist: {dist_str} (Max: {event.threshold_m:.1f}m)"
        )
        cv2.rectangle(snap, (0, 0), (snap.shape[1], 42), (15, 15, 15), -1)
        cv2.putText(
            snap,
            banner,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        filename = (
            f"{ts_slug}_{event.event_type}_F{event.forklift_id}_D{event.driver_id}.jpg"
        )
        full_path = self.snapshot_dir / filename
        cv2.imwrite(str(full_path), snap)
        return str(full_path)

    def process_events(
        self, events: List[SafetyEvent], frame: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Processes all safety events emitted on the current frame:
        triggers audible alarms on violations, saves snapshots, and logs to SQLite, JSONL, and CSV.
        """
        logged_records: List[Dict] = []

        for ev in events:
            now_dt = datetime.now(timezone.utc)
            ts_iso = now_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            ts_slug = now_dt.strftime("%Y%m%d_%H%M%S_%f")[:-3]

            if ev.event_type == "SAFE_DISTANCE_VIOLATION":
                self.trigger_audible_alarm(ev.forklift_id)

            snapshot_path = ""
            if frame is not None and ev.event_type in (
                "DRIVER_EXITED",
                "SAFE_DISTANCE_VIOLATION",
                "DISTANCE_UNCERTAIN",
            ):
                snapshot_path = self._save_event_snapshot(frame, ev, ts_slug)

            record = {
                "timestamp_utc": ts_iso,
                "camera_id": self.camera_id,
                "event_type": ev.event_type,
                "forklift_id": ev.forklift_id,
                "driver_id": ev.driver_id,
                "state": ev.state.value if isinstance(ev.state, VehicleSafetyState) else str(ev.state),
                "distance_m": ev.distance_m,
                "threshold_m": ev.threshold_m,
                "uncertainty_reason": ev.uncertainty_reason,
                "vehicle_bbox": [round(x, 1) for x in ev.vehicle_bbox],
                "driver_bbox": [round(x, 1) for x in ev.driver_bbox] if ev.driver_bbox else None,
                "snapshot_path": snapshot_path,
            }

            # 1. Persist to SQLite
            with self._db_lock, sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO vehicle_safety_events (
                        timestamp_utc, camera_id, event_type, forklift_id, driver_id,
                        state, distance_m, threshold_m, uncertainty_reason,
                        vehicle_bbox, driver_bbox, snapshot_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record["timestamp_utc"],
                        record["camera_id"],
                        record["event_type"],
                        record["forklift_id"],
                        record["driver_id"],
                        record["state"],
                        record["distance_m"],
                        record["threshold_m"],
                        record["uncertainty_reason"],
                        json.dumps(record["vehicle_bbox"]),
                        json.dumps(record["driver_bbox"]),
                        record["snapshot_path"],
                    ),
                )
                conn.commit()

            # 2. Append to JSONL audit log
            with open(self.jsonl_path, "a", encoding="utf-8") as jf:
                jf.write(json.dumps(record) + "\n")

            # 3. Append to CSV report
            with open(self.csv_path, "a", newline="", encoding="utf-8") as cf:
                writer = csv.writer(cf)
                writer.writerow(
                    [
                        record["timestamp_utc"],
                        record["camera_id"],
                        record["event_type"],
                        record["forklift_id"],
                        record["driver_id"],
                        record["state"],
                        record["distance_m"],
                        record["threshold_m"],
                        record["uncertainty_reason"],
                        record["snapshot_path"],
                    ]
                )

            self._recent_events.insert(0, record)
            self._recent_events = self._recent_events[:50]
            logged_records.append(record)

        return logged_records

    def get_recent_events(self, limit: int = 10) -> List[Dict]:
        """Returns the most recent logged events for the real-time HUD and dashboard."""
        return self._recent_events[:limit]