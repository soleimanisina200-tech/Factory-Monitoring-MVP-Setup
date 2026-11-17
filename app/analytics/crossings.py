# app/analytics/crossings.py
"""
Directional analytics: which worker crossed which line / passed which region, in which direction.

Lines (tripwires) use a SIDE STATE with a dead band instead of raw segment intersection:
    signed distance d of the ground point to the line;  d > +dead_band -> side +1,  d < -dead_band -> side -1
    a crossing is a change of the stable side (-1 -> +1 = labels[0] "IN", +1 -> -1 = labels[1] "OUT"; `invert`
    swaps them), counted only where the path actually passes between the end points (segment_margin) and
    within max_gap_sec (a worker hidden for a moment while crossing is still counted).
A worker standing ON the line (foot point jitter of a few pixels) can no longer produce IN/OUT/IN/OUT.

Directional regions (gates): entering the polygon stores the entry point; leaving it compares the displacement with
the region's entry_vector:  >= +min_displacement_px -> IN,  <= -min_displacement_px -> OUT.  A worker who vanishes
INSIDE the region (walks through a door that lies inside it) is evaluated at the last point seen - the old tracker
never counted those.

allowed_direction: IN|OUT on a line/region -> a WRONG_WAY event for the other direction (one-way corridors).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.analytics.entities import EntityKey, tag_event
from app.core.events import INFO, WARNING, make_event

Point = Tuple[float, float]


@dataclass
class LineSpec:
    name: str
    a: np.ndarray
    b: np.ndarray
    invert: bool = False
    labels: Tuple[str, str] = ("IN", "OUT")
    dead_band_px: float = 18.0
    cooldown_sec: float = 1.0
    max_gap_sec: float = 4.0
    segment_margin: float = 0.10
    allowed_direction: Optional[str] = None
    wrong_way_severity: str = WARNING
    color: Tuple[int, int, int] = (0, 255, 255)

    def signed_distance(self, p: Point) -> float:
        ab = self.b - self.a
        n = float(np.hypot(ab[0], ab[1])) or 1.0
        return float(ab[0] * (p[1] - self.a[1]) - ab[1] * (p[0] - self.a[0])) / n

    def crossing_on_segment(self, p0: Point, d0: float, p1: Point, d1: float) -> bool:
        if d0 == d1:
            return False
        s = d0 / (d0 - d1)
        c = np.array([p0[0] + s * (p1[0] - p0[0]), p0[1] + s * (p1[1] - p0[1])])
        ab = self.b - self.a
        t = float(np.dot(c - self.a, ab) / (float(np.dot(ab, ab)) or 1.0))
        return -self.segment_margin <= t <= 1.0 + self.segment_margin


@dataclass
class RegionSpec:
    name: str
    polygon: np.ndarray
    entry_vector: np.ndarray
    min_displacement_px: float = 40.0
    labels: Tuple[str, str] = ("IN", "OUT")
    allowed_direction: Optional[str] = None
    wrong_way_severity: str = WARNING
    color: Tuple[int, int, int] = (255, 165, 0)


@dataclass
class _LineState:
    side: int = 0
    side_ts: float = 0.0
    side_pt: Optional[Point] = None
    side_d: float = 0.0
    last_cross: Dict[str, float] = field(default_factory=dict)


@dataclass
class _RegionState:
    inside: bool = False
    enter_pt: Optional[Point] = None
    enter_ts: float = 0.0
    last_pt: Optional[Point] = None
    last_inside_ts: float = 0.0
    outside_since: Optional[float] = None
    prev_outside_pt: Optional[Point] = None
    prev_outside_ts: float = -1e18


class CrossingAnalytics:
    def __init__(
        self,
        lines: Dict[str, Dict[str, Any]],
        regions: Dict[str, Dict[str, Any]],
        dead_band_px: float = 18.0,
        cooldown_sec: float = 1.0,
        max_gap_sec: float = 4.0,
        segment_margin: float = 0.10,
        region_exit_confirm_sec: float = 0.3,
        region_lost_sec: float = 3.0,
        state_ttl_sec: float = 120.0,
        draw_trails: bool = True,
    ):
        self.lines: Dict[str, LineSpec] = {}
        for name, s in (lines or {}).items():
            labels = tuple(s.get("labels") or ("IN", "OUT"))
            self.lines[name] = LineSpec(
                name=name,
                a=np.asarray(s["a"], dtype=np.float64),
                b=np.asarray(s["b"], dtype=np.float64),
                invert=bool(s.get("invert", False)),
                labels=(str(labels[0]), str(labels[1])),
                dead_band_px=float(s.get("dead_band_px", dead_band_px)),
                cooldown_sec=float(s.get("cooldown_sec", cooldown_sec)),
                max_gap_sec=float(s.get("max_gap_sec", max_gap_sec)),
                segment_margin=float(s.get("segment_margin", segment_margin)),
                allowed_direction=s.get("allowed_direction") or None,
                wrong_way_severity=str(s.get("wrong_way_severity", WARNING)),
                color=tuple(int(c) for c in s.get("color", (0, 255, 255))),
            )
        self.regions: Dict[str, RegionSpec] = {}
        for name, s in (regions or {}).items():
            v = np.asarray(s.get("entry_vector", (0.0, -1.0)), dtype=np.float64)
            v = v / (float(np.linalg.norm(v)) or 1.0)
            labels = tuple(s.get("labels") or ("IN", "OUT"))
            self.regions[name] = RegionSpec(
                name=name,
                polygon=np.asarray(s["polygon"], dtype=np.float32).reshape(-1, 2),
                entry_vector=v,
                min_displacement_px=float(s.get("min_displacement_px", 40.0)),
                labels=(str(labels[0]), str(labels[1])),
                allowed_direction=s.get("allowed_direction") or None,
                wrong_way_severity=str(s.get("wrong_way_severity", WARNING)),
                color=tuple(int(c) for c in s.get("color", (255, 165, 0))),
            )
        self.region_exit_confirm_sec = float(region_exit_confirm_sec)
        self.region_lost_sec = float(region_lost_sec)
        self.state_ttl_sec = float(state_ttl_sec)
        self.draw_trails = bool(draw_trails)

        self.line_states: Dict[str, Dict[EntityKey, _LineState]] = {n: {} for n in self.lines}
        self.region_states: Dict[str, Dict[EntityKey, _RegionState]] = {n: {} for n in self.regions}
        self.counts: Dict[str, Dict[str, int]] = {
            **{n: {l.labels[0]: 0, l.labels[1]: 0} for n, l in self.lines.items()},
            **{n: {r.labels[0]: 0, r.labels[1]: 0} for n, r in self.regions.items()},
        }
        self.trails: Dict[EntityKey, Deque[Tuple[int, int]]] = {}
        self.last_seen: Dict[EntityKey, float] = {}
        self.last_label: Dict[EntityKey, str] = {}
        self._last_prune = -1e18

    @property
    def enabled(self) -> bool:
        return bool(self.lines or self.regions)

    # ------------------------------------------------------------------
    def _emit_pass(self, kind: str, name: str, direction: str, allowed: Optional[str], severity: str, key: EntityKey,
                   ts: float, events: List[Dict[str, Any]], **details: Any) -> None:
        self.counts[name][direction] = self.counts[name].get(direction, 0) + 1
        self.last_label[key] = f"{name}:{direction}"
        events.append(tag_event(make_event(kind, ts, subject=name, direction=direction, **details), key))
        if allowed and direction != allowed:
            events.append(tag_event(make_event("WRONG_WAY", ts, severity=severity, subject=name,
                                               direction=direction, allowed_direction=allowed), key))

    def _update_line(self, line: LineSpec, key: EntityKey, pt: Point, ts: float,
                     events: List[Dict[str, Any]]) -> None:
        d = line.signed_distance(pt)
        side = 1 if d > line.dead_band_px else (-1 if d < -line.dead_band_px else 0)
        st = self.line_states[line.name].setdefault(key, _LineState())
        if side == 0:
            return
        if st.side == 0 or side == st.side or st.side_pt is None:
            st.side, st.side_ts, st.side_pt, st.side_d = side, ts, pt, d
            return
        # stable side changed
        if ts - st.side_ts <= line.max_gap_sec and line.crossing_on_segment(st.side_pt, st.side_d, pt, d):
            to_positive = st.side < 0 < side
            direction = line.labels[1] if (to_positive == line.invert) else line.labels[0]
            if ts - st.last_cross.get(direction, -1e18) >= line.cooldown_sec:
                st.last_cross[direction] = ts
                self._emit_pass("LINE_CROSS", line.name, direction, line.allowed_direction,
                                line.wrong_way_severity, key, ts, events)
        st.side, st.side_ts, st.side_pt, st.side_d = side, ts, pt, d

    def _evaluate_region(self, region: RegionSpec, key: EntityKey, st: _RegionState, exit_pt: Point, ts: float,
                         reason: str, events: List[Dict[str, Any]]) -> None:
        st.inside = False
        st.outside_since = None
        if st.enter_pt is None:
            return
        disp = np.array([exit_pt[0] - st.enter_pt[0], exit_pt[1] - st.enter_pt[1]], dtype=np.float64)
        proj = float(np.dot(disp, region.entry_vector))
        if abs(proj) < region.min_displacement_px:
            return
        direction = region.labels[0] if proj > 0 else region.labels[1]
        self._emit_pass("REGION_PASS", region.name, direction, region.allowed_direction,
                        region.wrong_way_severity, key, ts, events, reason=reason,
                        displacement_px=round(proj, 1), entered_ts=round(st.enter_ts, 3))

    def _update_region(self, region: RegionSpec, key: EntityKey, pt: Point, ts: float,
                       events: List[Dict[str, Any]]) -> None:
        st = self.region_states[region.name].setdefault(key, _RegionState())
        inside = cv2.pointPolygonTest(region.polygon, (float(pt[0]), float(pt[1])), False) >= 0
        if inside:
            st.outside_since = None
            if not st.inside:
                st.inside = True
                # entry point = last point seen OUTSIDE when it is recent (more displacement, better direction)
                recent = st.prev_outside_pt is not None and ts - st.prev_outside_ts <= 1.0
                st.enter_pt = st.prev_outside_pt if recent else pt
                st.enter_ts = ts
            st.last_pt = pt
            st.last_inside_ts = ts
        else:
            if st.inside:
                if st.outside_since is None:
                    st.outside_since = ts
                if ts - st.outside_since >= self.region_exit_confirm_sec:
                    self._evaluate_region(region, key, st, pt, ts, "WALKED_THROUGH", events)
            st.prev_outside_pt = pt
            st.prev_outside_ts = ts

    # ------------------------------------------------------------------
    def update(self, points: Dict[EntityKey, Point], ts: float) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        for key, pt in points.items():
            pt = (float(pt[0]), float(pt[1]))
            self.last_seen[key] = ts
            if self.draw_trails:
                self.trails.setdefault(key, deque(maxlen=30)).append((int(pt[0]), int(pt[1])))
            for line in self.lines.values():
                self._update_line(line, key, pt, ts, events)
            for region in self.regions.values():
                self._update_region(region, key, pt, ts, events)
        events.extend(self.tick(ts))
        return events

    def tick(self, ts: float) -> List[Dict[str, Any]]:
        """Regions: a worker who vanished INSIDE a region for region_lost_sec is evaluated at his last point."""
        events: List[Dict[str, Any]] = []
        for rname, region in self.regions.items():
            for key, st in self.region_states[rname].items():
                if st.inside and ts - st.last_inside_ts >= self.region_lost_sec and st.last_pt is not None:
                    self._evaluate_region(region, key, st, st.last_pt, st.last_inside_ts, "VANISHED_INSIDE", events)
        if ts - self._last_prune >= 5.0:
            self._last_prune = ts
            for key in [k for k, t in self.last_seen.items() if ts - t > self.state_ttl_sec]:
                self._forget(key)
        return events

    def close_entity(self, key: EntityKey, ts: float) -> List[Dict[str, Any]]:
        """The identity manager reports that this worker left the camera."""
        events: List[Dict[str, Any]] = []
        for rname, region in self.regions.items():
            st = self.region_states[rname].get(key)
            if st is not None and st.inside and st.last_pt is not None:
                self._evaluate_region(region, key, st, st.last_pt, st.last_inside_ts, "LEFT_CAMERA", events)
        return events

    def _forget(self, key: EntityKey) -> None:
        for states in list(self.line_states.values()) + list(self.region_states.values()):
            states.pop(key, None)
        self.trails.pop(key, None)
        self.last_seen.pop(key, None)
        self.last_label.pop(key, None)

    def rename(self, old: EntityKey, new: EntityKey) -> None:
        if old == new:
            return
        for states in self.line_states.values():
            s_old = states.pop(old, None)
            if s_old is not None and (new not in states or states[new].side_ts < s_old.side_ts):
                states[new] = s_old
        for states in self.region_states.values():
            s_old = states.pop(old, None)
            if s_old is not None and (new not in states or (s_old.inside and not states[new].inside)):
                states[new] = s_old
        if old in self.trails:
            trail = self.trails.pop(old)
            if new not in self.trails:
                self.trails[new] = trail
        if old in self.last_seen:
            self.last_seen[new] = max(self.last_seen.pop(old), self.last_seen.get(new, -1e18))
        if old in self.last_label:
            self.last_label.setdefault(new, self.last_label.pop(old))

    def promote(self, track_id: int, global_id: int) -> None:
        self.rename(f"T{int(track_id)}", int(global_id))

    def snapshot(self) -> Dict[str, Dict[str, int]]:
        return {k: dict(v) for k, v in self.counts.items()}

    # ------------------------------------------------------------------
    def draw(self, frame: np.ndarray, active_keys: Optional[set] = None) -> None:
        if self.regions:
            # blend only the bounding box of the regions, not the whole (multi-megapixel) frame
            h, w = frame.shape[:2]
            allp = np.vstack([r.polygon for r in self.regions.values()]).astype(np.int32)
            x1, y1 = int(max(0, allp[:, 0].min())), int(max(0, allp[:, 1].min()))
            x2, y2 = int(min(w, allp[:, 0].max() + 1)), int(min(h, allp[:, 1].max() + 1))
            if x2 > x1 and y2 > y1:
                roi = frame[y1:y2, x1:x2]
                overlay = roi.copy()
                for region in self.regions.values():
                    cv2.fillPoly(overlay, [region.polygon.astype(np.int32) - np.array([x1, y1], np.int32)],
                                 region.color)
                cv2.addWeighted(overlay, 0.12, roi, 0.88, 0, roi)
            for rname, region in self.regions.items():
                poly = region.polygon.astype(np.int32)
                cv2.polylines(frame, [poly], True, region.color, 2, cv2.LINE_AA)
                c = poly.mean(axis=0).astype(int)
                tip = (int(c[0] + 50 * region.entry_vector[0]), int(c[1] + 50 * region.entry_vector[1]))
                cv2.arrowedLine(frame, (int(c[0]), int(c[1])), tip, (0, 255, 0), 2, cv2.LINE_AA, tipLength=0.3)
                cnt = self.counts[rname]
                label = f"{rname} | " + "  ".join(f"{k}: {v}" for k, v in cnt.items())
                cv2.putText(frame, label, (int(poly[0][0]), max(25, int(poly[0][1]) - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, region.color, 2, cv2.LINE_AA)
        for lname, line in self.lines.items():
            a = (int(line.a[0]), int(line.a[1]))
            b = (int(line.b[0]), int(line.b[1]))
            cv2.line(frame, a, b, line.color, 3, cv2.LINE_AA)
            mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
            dx, dy = b[0] - a[0], b[1] - a[1]
            n = max(1.0, float(np.hypot(dx, dy)))
            sign = -1.0 if line.invert else 1.0
            tip = (int(mid[0] - sign * dy / n * 40), int(mid[1] + sign * dx / n * 40))
            cv2.arrowedLine(frame, mid, tip, (0, 255, 0), 2, cv2.LINE_AA, tipLength=0.3)
            cv2.putText(frame, line.labels[0], tip, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
            cnt = self.counts[lname]
            label = f"{lname} | " + "  ".join(f"{k}: {v}" for k, v in cnt.items())
            cv2.putText(frame, label, (a[0], max(25, a[1] - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, line.color, 2,
                        cv2.LINE_AA)
        if self.draw_trails:
            for key, trail in self.trails.items():
                if active_keys is not None and key not in active_keys:
                    continue
                if len(trail) >= 2:
                    cv2.polylines(frame, [np.array(trail, dtype=np.int32)], False, (0, 255, 255), 2, cv2.LINE_AA)
