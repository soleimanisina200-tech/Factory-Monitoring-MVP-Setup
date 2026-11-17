# app/analytics/entities.py
"""
Analytics keys and per-worker event attribution.

A person on screen is either a verified worker (global ID, int) or a provisional track that has no ID yet
("T<track_id>"). Analytics follow BOTH, so a worker who crosses the gate line in his first second on camera is not
missed. Events of provisional tracks are held in the AttributionBuffer until the track receives its worker ID
(or until a timeout), then written with the right worker_id.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("FactoryReID")

EntityKey = Union[int, str]


def entity_key(global_id: Optional[int], track_id: int) -> EntityKey:
    return int(global_id) if global_id is not None and int(global_id) >= 0 else f"T{int(track_id)}"


def split_entity(key: EntityKey) -> Tuple[Optional[int], Optional[int]]:
    """-> (worker_id, track_id)"""
    if isinstance(key, str):
        if key.startswith("T") and key[1:].lstrip("-").isdigit():
            return None, int(key[1:])
        if key.isdigit():
            return int(key), None
        return None, None
    return int(key), None


def tag_event(ev: Dict[str, Any], key: EntityKey) -> Dict[str, Any]:
    wid, tid = split_entity(key)
    ev["worker_id"], ev["track_id"] = wid, tid
    return ev


class AttributionBuffer:
    """
    Holds events of provisional tracks ("T12") until the identity manager binds the track to a worker.

      submit(ev)            -> forwarded at once when it already has a worker_id
      resolve(tid, gid)     -> releases the held events of that track with worker_id = gid
      tick(now)             -> releases (as unidentified) or drops events older than `timeout_sec`
    """

    def __init__(self, sink: Callable[[Dict[str, Any]], None], timeout_sec: float = 20.0,
                 keep_unidentified: bool = True, max_held: int = 5000):
        self.sink = sink
        self.timeout_sec = float(timeout_sec)
        self.keep_unidentified = bool(keep_unidentified)
        self.max_held = int(max_held)
        self._held: Dict[int, List[Tuple[float, Dict[str, Any]]]] = {}
        self._n_held = 0

    def submit(self, ev: Dict[str, Any], now: Optional[float] = None) -> None:
        if ev.get("worker_id") is not None or ev.get("track_id") is None:
            self.sink(ev)
            return
        if self._n_held >= self.max_held:
            self._release_oldest()
        self._held.setdefault(int(ev["track_id"]), []).append((float(now if now is not None else ev["ts"]), ev))
        self._n_held += 1

    def resolve(self, track_id: int, worker_id: int) -> int:
        items = self._held.pop(int(track_id), [])
        self._n_held -= len(items)
        for _, ev in items:
            ev["worker_id"] = int(worker_id)
            ev.setdefault("details", {})["attributed_late"] = True
            self.sink(ev)
        return len(items)

    def pending_tracks(self) -> List[int]:
        return list(self._held)

    def tick(self, now: float) -> None:
        for tid in [t for t, items in self._held.items() if items and now - items[0][0] > self.timeout_sec]:
            self._expire(tid)

    def flush_all(self) -> None:
        for tid in list(self._held):
            self._expire(tid)

    def _expire(self, tid: int) -> None:
        items = self._held.pop(tid, [])
        self._n_held -= len(items)
        if not self.keep_unidentified:
            return
        for _, ev in items:
            ev.setdefault("details", {})["unidentified"] = True
            self.sink(ev)

    def _release_oldest(self) -> None:
        oldest = min(self._held, key=lambda t: self._held[t][0][0] if self._held[t] else float("inf"))
        logger.warning("Attribution buffer full: releasing events of track %s unidentified", oldest)
        self._expire(oldest)
