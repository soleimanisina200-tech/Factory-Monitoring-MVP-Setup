# app/analytics/zones.py
"""
Zone presence analytics: who is in which zone, since when, and for how long in total.

Per (person, zone) a time-based hysteresis state machine (frame-rate independent):
    outside --(inside for entry_confirm_sec)--> inside  : ZONE_ENTER  (time = first inside moment)
    inside  --(outside for exit_confirm_sec)--> outside : ZONE_EXIT   (dwell = first outside moment - entry)
    inside  --(not visible for lost_timeout_sec)------> : ZONE_EXIT reason LOST (dwell up to last sighting)
    inside  --(identity manager: left the camera)-----> : ZONE_EXIT reason LEFT_CAMERA
Every visit is added to per-worker totals (visits, total seconds), which the old state machine overwrote.

Zone rules (scene.yaml):
    type: restricted        -> ZONE_INTRUSION alarm on entry (+ ZONE_INTRUSION_CLEARED on exit)
    max_dwell_sec: N        -> ZONE_DWELL_EXCEEDED warning once per visit
    max_occupancy: N        -> ZONE_OVER_CAPACITY warning while more than N people are inside (+ _CLEARED)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.analytics.entities import EntityKey, tag_event
from app.core.events import ALARM, INFO, WARNING, make_event


@dataclass
class ZoneSpec:
    name: str
    polygon: np.ndarray
    kind: str = "presence"
    max_dwell_sec: float = 0.0
    max_occupancy: int = 0
    color: Tuple[int, int, int] = (200, 160, 0)


@dataclass
class _VisitState:
    inside: bool = False
    enter_pending_since: Optional[float] = None
    exit_pending_since: Optional[float] = None
    visit_start: float = 0.0
    last_seen: float = 0.0
    last_inside_ts: float = 0.0
    dwell_alerted: bool = False


@dataclass
class ZoneTotals:
    visits: int = 0
    total_sec: float = 0.0
    first_enter_ts: Optional[float] = None
    last_exit_ts: Optional[float] = None

    def add(self, start: float, end: float) -> None:
        self.visits += 1
        self.total_sec += max(0.0, end - start)
        self.first_enter_ts = start if self.first_enter_ts is None else min(self.first_enter_ts, start)
        self.last_exit_ts = end if self.last_exit_ts is None else max(self.last_exit_ts, end)


class ZoneAnalytics:
    def __init__(
        self,
        zones: Dict[str, Dict[str, Any]],
        entry_confirm_sec: float = 0.5,
        exit_confirm_sec: float = 1.0,
        lost_timeout_sec: float = 15.0,
    ):
        self.entry_confirm_sec = float(entry_confirm_sec)
        self.exit_confirm_sec = float(exit_confirm_sec)
        self.lost_timeout_sec = float(lost_timeout_sec)
        self.zones: Dict[str, ZoneSpec] = {}
        for name, spec in (zones or {}).items():
            self.zones[name] = ZoneSpec(
                name=name,
                polygon=np.asarray(spec["polygon"], dtype=np.float32).reshape(-1, 2),
                kind=str(spec.get("type", "presence")),
                max_dwell_sec=float(spec.get("max_dwell_sec") or 0.0),
                max_occupancy=int(spec.get("max_occupancy") or 0),
                color=tuple(int(c) for c in spec.get("color", (0, 0, 220) if spec.get("type") == "restricted"
                                                       else (200, 160, 0))),
            )
        self.states: Dict[str, Dict[EntityKey, _VisitState]] = {z: {} for z in self.zones}
        self.totals: Dict[EntityKey, Dict[str, ZoneTotals]] = {}
        self._over_capacity: Dict[str, bool] = {z: False for z in self.zones}

    # ------------------------------------------------------------------
    def _contains(self, zone: ZoneSpec, pt: Tuple[float, float]) -> bool:
        return cv2.pointPolygonTest(zone.polygon, (float(pt[0]), float(pt[1])), False) >= 0

    def _close_visit(self, zone: ZoneSpec, key: EntityKey, st: _VisitState, end_ts: float, reason: str,
                     events: List[Dict[str, Any]]) -> None:
        end_ts = max(end_ts, st.visit_start)
        dwell = end_ts - st.visit_start
        self.totals.setdefault(key, {}).setdefault(zone.name, ZoneTotals()).add(st.visit_start, end_ts)
        st.inside = False
        st.exit_pending_since = None
        st.dwell_alerted = False
        events.append(tag_event(make_event(
            "ZONE_EXIT", end_ts, subject=zone.name, value=round(dwell, 2), reason=reason,
            visit_start_ts=round(st.visit_start, 3), zone_type=zone.kind,
        ), key))
        if zone.kind == "restricted":
            events.append(tag_event(make_event("ZONE_INTRUSION_CLEARED", end_ts, subject=zone.name), key))

    def _open_visit(self, zone: ZoneSpec, key: EntityKey, st: _VisitState, start_ts: float,
                    events: List[Dict[str, Any]]) -> None:
        st.inside = True
        st.visit_start = start_ts
        st.enter_pending_since = None
        st.last_inside_ts = start_ts
        events.append(tag_event(make_event("ZONE_ENTER", start_ts, subject=zone.name, zone_type=zone.kind), key))
        if zone.kind == "restricted":
            events.append(tag_event(make_event("ZONE_INTRUSION", start_ts, severity=ALARM, subject=zone.name), key))

    # ------------------------------------------------------------------
    def update(self, points: Dict[EntityKey, Tuple[float, float]], ts: float) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        for zname, zone in self.zones.items():
            zstates = self.states[zname]
            for key, pt in points.items():
                st = zstates.get(key)
                if st is None:
                    st = zstates[key] = _VisitState(last_seen=ts)
                st.last_seen = ts
                if self._contains(zone, pt):
                    st.exit_pending_since = None
                    st.last_inside_ts = ts
                    if not st.inside:
                        if st.enter_pending_since is None:
                            st.enter_pending_since = ts
                        if ts - st.enter_pending_since >= self.entry_confirm_sec:
                            self._open_visit(zone, key, st, st.enter_pending_since, events)
                    elif zone.max_dwell_sec > 0 and not st.dwell_alerted and ts - st.visit_start >= zone.max_dwell_sec:
                        st.dwell_alerted = True
                        events.append(tag_event(make_event(
                            "ZONE_DWELL_EXCEEDED", ts, severity=WARNING, subject=zone.name,
                            value=round(ts - st.visit_start, 1), limit_sec=zone.max_dwell_sec,
                        ), key))
                else:
                    st.enter_pending_since = None
                    if st.inside:
                        if st.exit_pending_since is None:
                            st.exit_pending_since = ts
                        if ts - st.exit_pending_since >= self.exit_confirm_sec:
                            self._close_visit(zone, key, st, st.exit_pending_since, "WALKED_OUT", events)

            # people not visible in this frame
            for key in [k for k in zstates if k not in points]:
                st = zstates[key]
                st.enter_pending_since = None
                unseen = ts - st.last_seen
                if st.inside and unseen >= self.lost_timeout_sec:
                    self._close_visit(zone, key, st, st.last_inside_ts, "LOST", events)
                if not st.inside and unseen >= 2.0 * self.lost_timeout_sec:
                    del zstates[key]

            if zone.max_occupancy > 0:
                count = sum(1 for s in zstates.values() if s.inside)
                if count > zone.max_occupancy and not self._over_capacity[zname]:
                    self._over_capacity[zname] = True
                    events.append(make_event("ZONE_OVER_CAPACITY", ts, severity=WARNING, subject=zname,
                                             value=count, limit=zone.max_occupancy))
                elif count <= zone.max_occupancy and self._over_capacity[zname]:
                    self._over_capacity[zname] = False
                    events.append(make_event("ZONE_OVER_CAPACITY_CLEARED", ts, severity=INFO, subject=zname,
                                             value=count, limit=zone.max_occupancy))
        return events

    def close_entity(self, key: EntityKey, ts: float, reason: str = "LEFT_CAMERA") -> List[Dict[str, Any]]:
        """The person left the camera view: close open visits at the last time they were seen inside."""
        events: List[Dict[str, Any]] = []
        for zname, zone in self.zones.items():
            st = self.states[zname].pop(key, None)
            if st is not None and st.inside:
                self._close_visit(zone, key, st, st.last_inside_ts, reason, events)
        return events

    def close_all(self, ts: float, reason: str = "SHUTDOWN") -> List[Dict[str, Any]]:
        """Service stop: record every visit in progress up to the last time the person was seen inside."""
        events: List[Dict[str, Any]] = []
        for zname, zone in self.zones.items():
            for key, st in list(self.states[zname].items()):
                if st.inside:
                    self._close_visit(zone, key, st, min(ts, max(st.last_inside_ts, st.visit_start)), reason, events)
        return events

    def rename(self, old: EntityKey, new: EntityKey) -> None:
        """`old` and `new` are the same person (track got its ID, or two IDs were merged)."""
        if old == new:
            return
        for zstates in self.states.values():
            st_old = zstates.pop(old, None)
            if st_old is None:
                continue
            st_new = zstates.get(new)
            if st_new is None or (st_old.inside and not st_new.inside):
                zstates[new] = st_old
            elif st_old.inside and st_new.inside:
                st_new.visit_start = min(st_new.visit_start, st_old.visit_start)
                st_new.last_seen = max(st_new.last_seen, st_old.last_seen)
        old_totals = self.totals.pop(old, None)
        if old_totals:
            dst = self.totals.setdefault(new, {})
            for zname, t in old_totals.items():
                d = dst.setdefault(zname, ZoneTotals())
                d.visits += t.visits
                d.total_sec += t.total_sec
                for attr, fn in (("first_enter_ts", min), ("last_exit_ts", max)):
                    a, b = getattr(d, attr), getattr(t, attr)
                    setattr(d, attr, b if a is None else (a if b is None else fn(a, b)))

    def promote(self, track_id: int, global_id: int) -> None:
        self.rename(f"T{int(track_id)}", int(global_id))

    # ------------------------------------------------------------------
    # Read-out
    # ------------------------------------------------------------------
    def occupants(self, zone: str) -> List[EntityKey]:
        return [k for k, s in self.states.get(zone, {}).items() if s.inside]

    def zones_of(self, key: EntityKey) -> List[str]:
        return [z for z, states in self.states.items() if key in states and states[key].inside]

    def worker_totals(self, now: Optional[float] = None) -> Dict[EntityKey, Dict[str, Dict[str, float]]]:
        """Totals per person and zone, including the visit in progress."""
        out: Dict[EntityKey, Dict[str, Dict[str, float]]] = {}
        for key, per_zone in self.totals.items():
            for zname, t in per_zone.items():
                out.setdefault(key, {})[zname] = {"visits": t.visits, "total_sec": round(t.total_sec, 1)}
        if now is not None:
            for zname, states in self.states.items():
                for key, st in states.items():
                    if st.inside:
                        d = out.setdefault(key, {}).setdefault(zname, {"visits": 0, "total_sec": 0.0})
                        d["visits"] += 1
                        d["total_sec"] = round(d["total_sec"] + max(0.0, now - st.visit_start), 1)
        return out

    def snapshot(self, now: float) -> Dict[str, Any]:
        out = {}
        for zname, zone in self.zones.items():
            occ = [(k, s) for k, s in self.states[zname].items() if s.inside]
            out[zname] = {
                "type": zone.kind,
                "count": len(occ),
                "occupants": [{"id": str(k), "dwell_sec": round(now - s.visit_start, 1)} for k, s in occ],
                "over_capacity": self._over_capacity[zname],
            }
        return out

    # ------------------------------------------------------------------
    def draw(self, frame: np.ndarray, now: Optional[float] = None) -> None:
        if not self.zones:
            return
        h, w = frame.shape[:2]
        all_pts = np.vstack([z.polygon for z in self.zones.values()]).astype(np.int32)
        x1, y1 = (int(max(0, v)) for v in all_pts.min(axis=0))
        x2, y2 = int(min(w, all_pts[:, 0].max() + 1)), int(min(h, all_pts[:, 1].max() + 1))
        if x2 > x1 and y2 > y1:
            roi = frame[y1:y2, x1:x2]
            overlay = roi.copy()
            for zname, zone in self.zones.items():
                occupied = any(s.inside for s in self.states[zname].values())
                color = (0, 0, 255) if (occupied and zone.kind == "restricted") else (
                    (0, 180, 0) if occupied else zone.color)
                cv2.fillPoly(overlay, [zone.polygon.astype(np.int32) - np.array([x1, y1], dtype=np.int32)], color)
            cv2.addWeighted(overlay, 0.18, roi, 0.82, 0, roi)
        for zname, zone in self.zones.items():
            occ = [(k, s) for k, s in self.states[zname].items() if s.inside]
            color = (0, 0, 255) if (occ and zone.kind == "restricted") else ((0, 180, 0) if occ else zone.color)
            poly = zone.polygon.astype(np.int32)
            cv2.polylines(frame, [poly], True, color, 2, cv2.LINE_AA)
            top = poly[np.argmin(poly[:, 1])]
            parts = [f"{k}:{(now - s.visit_start):.0f}s" if now is not None else str(k) for k, s in occ[:8]]
            label = f"{zname} [{len(occ)}] " + " ".join(parts)
            cv2.putText(frame, label, (int(top[0]), max(25, int(top[1]) - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2, cv2.LINE_AA)
