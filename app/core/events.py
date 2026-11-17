# app/core/events.py
"""
Single audit trail for every module (identity, zones, lines/regions, forklifts, alarms).

Every event is a plain dict built with `make_event` and written by ONE background thread to
  * SQLite (WAL mode: the report script can read while the service runs)   -> data/factory_events.db
  * a daily CSV mirror per camera                                          -> data/events/events_<cam>_YYYYMMDD.csv
  * a JPEG snapshot for alarms (and configured event types)                -> data/snapshots/
Disk latency never blocks the video loop; if the queue overflows the record is written synchronously instead
of being dropped (an audit trail must not lose alarms).

Identity merges are stored in `identity_aliases` (old_id -> new_id) so reports attribute every event of a
temporary ID to the worker it was merged into.
"""
from __future__ import annotations

import csv
import json
import logging
import queue
import sqlite3
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Set

import cv2
import numpy as np

from app.core.clock import StreamClock

logger = logging.getLogger("FactoryReID")

INFO, WARNING, ALARM = "info", "warning", "alarm"
SEVERITY_RANK = {INFO: 0, WARNING: 1, ALARM: 2}

CSV_COLUMNS = (
    "wall_time", "stream_ts", "camera_id", "type", "severity", "worker_id", "worker_label", "track_id",
    "vehicle_id", "subject", "direction", "value", "details", "snapshot_path",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    wall_time     TEXT NOT NULL,
    wall_epoch    REAL NOT NULL,
    stream_ts     REAL,
    camera_id     TEXT,
    type          TEXT NOT NULL,
    severity      TEXT,
    worker_id     INTEGER,
    worker_label  TEXT,
    track_id      INTEGER,
    vehicle_id    TEXT,
    subject       TEXT,
    direction     TEXT,
    value         REAL,
    details       TEXT,
    snapshot_path TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_type   ON events(type);
CREATE INDEX IF NOT EXISTS idx_events_worker ON events(worker_id);
CREATE INDEX IF NOT EXISTS idx_events_time   ON events(wall_epoch);
CREATE TABLE IF NOT EXISTS identity_aliases (
    camera_id  TEXT NOT NULL,
    old_id     INTEGER NOT NULL,
    new_id     INTEGER NOT NULL,
    wall_time  TEXT,
    wall_epoch REAL,
    PRIMARY KEY (camera_id, old_id)
);
CREATE TABLE IF NOT EXISTS worker_labels (
    camera_id  TEXT NOT NULL,
    worker_id  INTEGER NOT NULL,
    label      TEXT NOT NULL,
    wall_epoch REAL,
    PRIMARY KEY (camera_id, worker_id)
);
"""


def make_event(
    event_type: str,
    ts: float,
    *,
    severity: str = INFO,
    worker_id: Optional[int] = None,
    track_id: Optional[int] = None,
    vehicle_id: Optional[str] = None,
    subject: Optional[str] = None,
    direction: Optional[str] = None,
    value: Optional[float] = None,
    **details: Any,
) -> Dict[str, Any]:
    """`ts` is the STREAM time of the moment the event refers to (see app/core/clock.py)."""
    return {
        "type": event_type,
        "ts": float(ts),
        "severity": severity,
        "worker_id": None if worker_id is None else int(worker_id),
        "track_id": None if track_id is None else int(track_id),
        "vehicle_id": None if vehicle_id is None else str(vehicle_id),
        "subject": subject,
        "direction": direction,
        "value": None if value is None else float(value),
        "details": details,
    }


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


class EventStore:
    def __init__(
        self,
        db_path: Optional[str] = "data/factory_events.db",
        csv_dir: Optional[str] = "data/events",
        snapshot_dir: Optional[str] = "data/snapshots",
        clock: Optional[StreamClock] = None,
        camera_id: str = "cam1",
        snapshot_types: Optional[Iterable[str]] = None,
        snapshot_max_width: int = 1920,
        retention_days: float = 0.0,
        queue_size: int = 4096,
        label_lookup: Optional[Callable[[int], Optional[str]]] = None,
    ):
        self.clock = clock or StreamClock(is_live=True)
        self.camera_id = str(camera_id)
        self.db_path = Path(db_path) if db_path else None
        self.csv_dir = Path(csv_dir) if csv_dir else None
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else None
        self.snapshot_types: Set[str] = set(snapshot_types or [])
        self.snapshot_max_width = int(snapshot_max_width)
        self.retention_days = float(retention_days)
        self.label_lookup = label_lookup

        for d in (self.db_path.parent if self.db_path else None, self.csv_dir, self.snapshot_dir):
            if d is not None:
                d.mkdir(parents=True, exist_ok=True)

        self._listeners: List[Callable[[Dict[str, Any]], None]] = []
        self._recent: Deque[Dict[str, Any]] = deque(maxlen=300)
        self._recent_lock = threading.Lock()
        self._queue: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=max(16, int(queue_size)))
        self._sync_conn: Optional[sqlite3.Connection] = None  # only for the queue-overflow fallback
        self._sync_lock = threading.Lock()
        self._closed = False
        self.n_written = 0
        self.n_failed = 0

        if self.db_path is not None:
            conn = self._connect()
            conn.close()
        self._worker = threading.Thread(target=self._writer_loop, name="event-store-writer", daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------
    # Public API (called from the video loop)
    # ------------------------------------------------------------------
    def subscribe(self, fn: Callable[[Dict[str, Any]], None]) -> None:
        """`fn(event)` is called synchronously for every emitted event (alarm dispatch, telemetry...)."""
        self._listeners.append(fn)

    def emit(self, event: Dict[str, Any], frame: Optional[np.ndarray] = None) -> Dict[str, Any]:
        ev = dict(event)
        ev.setdefault("details", {})
        ev.setdefault("severity", INFO)
        ts = float(ev.get("ts", 0.0))
        wall = self.clock.wall(ts)
        ev["wall_epoch"] = wall
        ev["wall_time"] = datetime.fromtimestamp(wall).isoformat(sep=" ", timespec="milliseconds")
        ev["camera_id"] = self.camera_id
        if ev.get("worker_id") is not None and self.label_lookup is not None and not ev.get("worker_label"):
            try:
                ev["worker_label"] = self.label_lookup(int(ev["worker_id"]))
            except Exception:
                ev["worker_label"] = None

        snap_frame = None
        if frame is not None and self.snapshot_dir is not None and (
            ev["severity"] == ALARM or ev["type"] in self.snapshot_types
        ):
            stamp = datetime.fromtimestamp(wall).strftime("%Y%m%d_%H%M%S_%f")[:-3]
            who = ev.get("worker_id") if ev.get("worker_id") is not None else (ev.get("vehicle_id") or "na")
            day_dir = self.snapshot_dir / datetime.fromtimestamp(wall).strftime("%Y%m%d")
            ev["snapshot_path"] = str(day_dir / f"{stamp}_{self.camera_id}_{ev['type']}_{who}.jpg")
            snap_frame = frame.copy()

        self._enqueue(("event", ev, snap_frame))
        with self._recent_lock:
            self._recent.append(ev)
        for fn in self._listeners:
            try:
                fn(ev)
            except Exception:
                logger.exception("Event listener failed for %s", ev.get("type"))
        return ev

    def emit_many(self, events: Iterable[Dict[str, Any]], frame: Optional[np.ndarray] = None) -> None:
        for ev in events:
            self.emit(ev, frame)

    def add_alias(self, old_id: int, new_id: int, ts: float) -> None:
        """Temporary ID `old_id` was merged into `new_id`."""
        self._enqueue(("alias", int(old_id), int(new_id), self.clock.wall(ts)))

    def set_label(self, worker_id: int, label: str, ts: float) -> None:
        self._enqueue(("label", int(worker_id), str(label), self.clock.wall(ts)))

    def recent(self, limit: int = 50, min_severity: str = INFO) -> List[Dict[str, Any]]:
        rank = SEVERITY_RANK.get(min_severity, 0)
        with self._recent_lock:
            items = [e for e in self._recent if SEVERITY_RANK.get(e.get("severity", INFO), 0) >= rank]
        return [dict(e) for e in items[-limit:][::-1]]

    def flush(self, timeout: float = 10.0) -> bool:
        deadline = time.time() + timeout
        while self._queue.unfinished_tasks and time.time() < deadline:
            time.sleep(0.01)
        return self._queue.unfinished_tasks == 0

    def close(self, timeout: float = 10.0) -> None:
        if self._closed:
            return
        self._closed = True
        self.flush(timeout)
        try:
            self._queue.put(None, timeout=1.0)
        except queue.Full:
            pass
        self._worker.join(timeout=timeout)
        if self._sync_conn is not None:
            self._sync_conn.close()
            self._sync_conn = None

    # ------------------------------------------------------------------
    # Writer thread
    # ------------------------------------------------------------------
    def _enqueue(self, item: tuple) -> None:
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            logger.warning("Event queue full: writing %s synchronously", item[0])
            with self._sync_lock:
                if self._sync_conn is None and self.db_path is not None:
                    self._sync_conn = self._connect()
                try:
                    self._write_item(self._sync_conn, item)
                    if self._sync_conn is not None:
                        self._sync_conn.commit()
                except Exception:
                    self.n_failed += 1
                    logger.exception("Synchronous event write failed")

    def _connect(self) -> Optional[sqlite3.Connection]:
        if self.db_path is None:
            return None
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(_SCHEMA)
        conn.commit()
        return conn

    def _writer_loop(self) -> None:
        conn = None
        try:
            conn = self._connect()
        except Exception:
            logger.exception("Cannot open event database %s: events go to CSV only", self.db_path)
        last_cleanup = 0.0
        stop = False
        while not stop:
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                item = "idle"
            batch = []
            if item is None:
                stop = True
                self._queue.task_done()
            elif item != "idle":
                batch.append(item)
                while len(batch) < 200:
                    try:
                        nxt = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if nxt is None:
                        stop = True
                        self._queue.task_done()
                        break
                    batch.append(nxt)
            for it in batch:
                try:
                    self._write_item(conn, it)
                except Exception:
                    self.n_failed += 1
                    logger.exception("Failed to persist %s", it[0])
            if batch and conn is not None:
                try:
                    conn.commit()
                except Exception:
                    logger.exception("Event database commit failed")
            for _ in batch:
                self._queue.task_done()
            if self.retention_days > 0 and time.time() - last_cleanup > 6 * 3600:
                last_cleanup = time.time()
                self._cleanup_old_files()
        if conn is not None:
            conn.close()

    def _write_item(self, conn: Optional[sqlite3.Connection], item: tuple) -> None:
        kind = item[0]
        if kind == "event":
            _, ev, frame = item
            if frame is not None and ev.get("snapshot_path"):
                self._write_snapshot(ev["snapshot_path"], frame)
            details = json.dumps(ev.get("details") or {}, default=_json_default, ensure_ascii=False)
            row = (
                ev["wall_time"], ev["wall_epoch"], ev.get("ts"), ev["camera_id"], ev["type"], ev.get("severity"),
                ev.get("worker_id"), ev.get("worker_label"), ev.get("track_id"), ev.get("vehicle_id"),
                ev.get("subject"), ev.get("direction"), ev.get("value"), details, ev.get("snapshot_path"),
            )
            if conn is not None:
                conn.execute(
                    "INSERT INTO events (wall_time, wall_epoch, stream_ts, camera_id, type, severity, worker_id, "
                    "worker_label, track_id, vehicle_id, subject, direction, value, details, snapshot_path) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    row,
                )
            self._write_csv(ev, details)
            self.n_written += 1
        elif kind == "alias" and conn is not None:
            _, old_id, new_id, wall = item
            conn.execute(
                "INSERT OR REPLACE INTO identity_aliases (camera_id, old_id, new_id, wall_time, wall_epoch) "
                "VALUES (?,?,?,?,?)",
                (self.camera_id, old_id, new_id, datetime.fromtimestamp(wall).isoformat(sep=" "), wall),
            )
        elif kind == "label" and conn is not None:
            _, wid, label, wall = item
            conn.execute(
                "INSERT OR REPLACE INTO worker_labels (camera_id, worker_id, label, wall_epoch) VALUES (?,?,?,?)",
                (self.camera_id, wid, label, wall),
            )

    def _write_csv(self, ev: Dict[str, Any], details_json: str) -> None:
        if self.csv_dir is None:
            return
        day = datetime.fromtimestamp(ev["wall_epoch"]).strftime("%Y%m%d")
        # one file per camera: several camera processes share the folder and must not interleave lines
        path = self.csv_dir / f"events_{self.camera_id}_{day}.csv"
        new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8-sig" if new else "utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(CSV_COLUMNS)
            w.writerow([
                ev["wall_time"], round(float(ev.get("ts", 0.0)), 3), ev["camera_id"], ev["type"], ev.get("severity"),
                ev.get("worker_id"), ev.get("worker_label"), ev.get("track_id"), ev.get("vehicle_id"),
                ev.get("subject"), ev.get("direction"),
                None if ev.get("value") is None else round(float(ev["value"]), 3), details_json,
                ev.get("snapshot_path"),
            ])

    def _write_snapshot(self, path: str, frame: np.ndarray) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        h, w = frame.shape[:2]
        if self.snapshot_max_width > 0 and w > self.snapshot_max_width:
            frame = cv2.resize(frame, (self.snapshot_max_width, int(h * self.snapshot_max_width / w)),
                               interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(p), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

    def _cleanup_old_files(self) -> None:
        """Retention for snapshots and CSV mirrors (the SQLite audit trail is kept)."""
        cutoff = time.time() - self.retention_days * 86400.0
        removed = 0
        for root, pattern in ((self.snapshot_dir, "**/*.jpg"), (self.csv_dir, "events_*.csv")):
            if root is None or not root.exists():
                continue
            for f in root.glob(pattern):
                try:
                    if f.stat().st_mtime < cutoff:
                        f.unlink()
                        removed += 1
                except OSError:
                    pass
        if removed:
            logger.info("Retention: removed %d files older than %.0f days", removed, self.retention_days)
