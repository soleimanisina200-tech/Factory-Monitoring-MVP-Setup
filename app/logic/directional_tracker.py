from __future__ import annotations
import time
from collections import deque
from typing import Dict, Any, Tuple, List, Optional
import cv2
import numpy as np

Point = Tuple[int, int]
BBox = Tuple[int, int, int, int]


class DirectionalTracker:
    """
    Standalone CCTV Directional Tracking & Border Exit/Re-Entry Guard Module.
    Supports:
      1. Line-Crossing Detection (Point A -> Point B virtual tripwires)
      2. Entrance/Exit Polygon Gate Zone Traversal Detection
      3. Seamless Track-ID to Global-ID Trajectory Promotion (so distant walkers are never missed)
    """

    def __init__(self, cfg: Dict[str, Any], log_file: str = "attendance_events.log"):
        self.cfg = cfg or {}
        self.log_file = log_file

        dir_cfg = self.cfg.get("directional_tracking", {})
        self.lines: Dict[str, Dict[str, Any]] = dir_cfg.get("lines", {}) or {}
        self.entrance_zones: Dict[str, Dict[str, Any]] = dir_cfg.get("entrance_zones", {}) or {}
        self.draw_trails: bool = bool(dir_cfg.get("draw_trails", True))
        self.border_margin_px: int = int(dir_cfg.get("border_exit_margin_px", 120))

        # Per-person trajectory & state keyed by entity_key (gid if verified, else "T_<tid>")
        self.trajectories: Dict[Any, deque] = {}
        self.last_seen_ts: Dict[Any, float] = {}
        self.last_line_cross_ts: Dict[Tuple[Any, str], float] = {}
        self.active_gate_zones: Dict[Any, set] = {}
        self.gate_entry_point: Dict[Tuple[Any, str], Point] = {}
        self.last_event_label: Dict[Any, str] = {}
        self.exited_camera_ids: set[Any] = set()

        # Live Counters
        self.line_counts: Dict[str, Dict[str, int]] = {
            name: {"ENTRY": 0, "EXIT": 0} for name in self.lines
        }
        self.zone_counts: Dict[str, Dict[str, int]] = {
            name: {"ENTRY": 0, "EXIT": 0} for name in self.entrance_zones
        }

    @property
    def total_entries(self) -> int:
        l_in = sum(c["ENTRY"] for c in self.line_counts.values())
        z_in = sum(c["ENTRY"] for c in self.zone_counts.values())
        return l_in + z_in

    @property
    def total_exits(self) -> int:
        l_out = sum(c["EXIT"] for c in self.line_counts.values())
        z_out = sum(c["EXIT"] for c in self.zone_counts.values())
        return l_out + z_out

    def get_person_status(self, person_id: Any) -> Optional[str]:
        return self.last_event_label.get(person_id)

    def is_near_frame_border(self, pt: Point, width: int, height: int) -> bool:
        m = self.border_margin_px
        x, y = pt
        return x <= m or x >= (width - m) or y <= m or y >= (height - m)

    @staticmethod
    def get_foot_point(bbox: BBox) -> Point:
        x1, y1, x2, y2 = map(int, bbox[:4])
        return int((x1 + x2) / 2), int(y2)

    @staticmethod
    def _cross_product(a: Point, b: Point, p: Point) -> float:
        return float((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]))

    def _segments_intersect(self, p1: Point, p2: Point, a: Point, b: Point) -> bool:
        d1 = self._cross_product(a, b, p1)
        d2 = self._cross_product(a, b, p2)
        d3 = self._cross_product(p1, p2, a)
        d4 = self._cross_product(p1, p2, b)
        return ((d1 >= 0 and d2 < 0) or (d1 <= 0 and d2 > 0)) and (
            (d3 >= 0 and d4 < 0) or (d3 <= 0 and d4 > 0)
        )

    def _check_line_crossing(
        self, prev_pt: Point, curr_pt: Point, pt_a: Point, pt_b: Point, invert: bool = False
    ) -> Optional[str]:
        if not self._segments_intersect(prev_pt, curr_pt, pt_a, pt_b):
            return None
        prev_side = self._cross_product(pt_a, pt_b, prev_pt)
        curr_side = self._cross_product(pt_a, pt_b, curr_pt)
        if prev_side <= 0 and curr_side > 0:
            return "EXIT" if invert else "ENTRY"
        if prev_side >= 0 and curr_side < 0:
            return "ENTRY" if invert else "EXIT"
        return None

    @staticmethod
    def _point_in_poly(pt: Point, polygon: List[List[int]]) -> bool:
        if not polygon or len(polygon) < 3:
            return False
        pts = np.array(polygon, dtype=np.int32)
        return cv2.pointPolygonTest(pts, (float(pt[0]), float(pt[1])), False) >= 0

    def promote_track_to_global(self, track_id: int, global_id: int) -> None:
        """Transfers trajectory history when an initializing track_id receives its global_id."""
        t_key = f"T_{track_id}"
        if t_key in self.trajectories:
            t_traj = self.trajectories.pop(t_key)
            if global_id not in self.trajectories:
                self.trajectories[global_id] = t_traj
                self.active_gate_zones[global_id] = self.active_gate_zones.pop(t_key, set())
            if t_key in self.last_event_label and global_id not in self.last_event_label:
                self.last_event_label[global_id] = self.last_event_label.pop(t_key)

    def update_all(
        self,
        active_ground_points: Dict[Any, Tuple[float, float]],
        frame_ts: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Batch-updates all active tracks (both verified Global IDs and initializing Track IDs).
        """
        now = frame_ts if frame_ts is not None else time.monotonic()
        events: List[Dict[str, Any]] = []

        for entity_id, ground_uv in active_ground_points.items():
            foot_pt: Point = (int(ground_uv[0]), int(ground_uv[1]))
            self.last_seen_ts[entity_id] = now
            if entity_id in self.exited_camera_ids:
                self.exited_camera_ids.discard(entity_id)
            person_events = self._update_point(entity_id, foot_pt, now)
            events.extend(person_events)

        return events

    def update_person(
        self, person_id: Any, bbox: BBox, timestamp: Optional[float] = None
    ) -> Optional[str]:
        now = timestamp if timestamp is not None else time.monotonic()
        foot_pt = self.get_foot_point(bbox)
        self.last_seen_ts[person_id] = now
        self._update_point(person_id, foot_pt, now)
        return self.last_event_label.get(person_id)

    def _update_point(
        self, person_id: Any, foot_pt: Point, now: float
    ) -> List[Dict[str, Any]]:
        triggered_events: List[Dict[str, Any]] = []

        if person_id not in self.trajectories:
            self.trajectories[person_id] = deque(maxlen=40)
            self.active_gate_zones[person_id] = set()

        traj = self.trajectories[person_id]
        prev_pt = traj[-1] if len(traj) > 0 else None
        traj.append(foot_pt)

        # 1. Evaluate CCTV Line-Crossing (Point A -> Point B)
        if prev_pt is not None:
            # Check both immediate segment and 3-frame lookback segment to prevent fast-walk skips
            lookback_pt = traj[-4] if len(traj) >= 4 else prev_pt
            for lname, lcfg in self.lines.items():
                pt_a = (int(lcfg["point_a"][0]), int(lcfg["point_a"][1]))
                pt_b = (int(lcfg["point_b"][0]), int(lcfg["point_b"][1]))
                invert = bool(lcfg.get("invert_direction", False))
                cooldown = float(lcfg.get("cooldown_seconds", 2.0))

                direction = self._check_line_crossing(prev_pt, foot_pt, pt_a, pt_b, invert)
                if direction is None and lookback_pt != prev_pt:
                    direction = self._check_line_crossing(lookback_pt, foot_pt, pt_a, pt_b, invert)

                if direction is not None:
                    last_t = self.last_line_cross_ts.get((person_id, lname), -1e9)
                    if (now - last_t) >= cooldown:
                        self.last_line_cross_ts[(person_id, lname)] = now
                        self.line_counts[lname][direction] += 1
                        self.last_event_label[person_id] = f"{lname}:{direction}"
                        ev = self._record_event(person_id, f"LINE_{direction}", lname)
                        triggered_events.append(ev)

        # 2. Evaluate Entrance/Exit Gate Polygon Transitions
        curr_zones = set()
        for zname, zcfg in self.entrance_zones.items():
            if self._point_in_poly(foot_pt, zcfg.get("polygon", [])):
                curr_zones.add(zname)

        prev_zones = self.active_gate_zones[person_id]

        for zname in curr_zones - prev_zones:
            self.gate_entry_point[(person_id, zname)] = foot_pt

        for zname in prev_zones - curr_zones:
            enter_pt = self.gate_entry_point.pop((person_id, zname), foot_pt)
            zcfg = self.entrance_zones.get(zname, {})
            ref_vec = zcfg.get("entry_direction_vector", [0, -1])
            min_disp = float(zcfg.get("min_displacement", 20.0))

            dx = float(foot_pt[0] - enter_pt[0])
            dy = float(foot_pt[1] - enter_pt[1])
            if np.hypot(dx, dy) >= min_disp:
                dot = dx * float(ref_vec[0]) + dy * float(ref_vec[1])
                direction = "ENTRY" if dot >= 0 else "EXIT"
                self.zone_counts[zname][direction] += 1
                self.last_event_label[person_id] = f"{zname}:{direction}"
                ev = self._record_event(person_id, f"ZONE_{direction}", zname)
                triggered_events.append(ev)

        self.active_gate_zones[person_id] = curr_zones
        return triggered_events

    def draw_overlays(
        self, frame: np.ndarray, active_gids: Optional[set] = None
    ) -> np.ndarray:
        """Draws entrance gate zones, Point A -> Point B tripwires, direction arrows, and active trails."""
        if self.entrance_zones:
            overlay = frame.copy()
            for zname, zcfg in self.entrance_zones.items():
                poly = zcfg.get("polygon", [])
                if len(poly) < 3:
                    continue
                pts = np.array(poly, dtype=np.int32)
                color = tuple(zcfg.get("color", [255, 165, 0]))
                cv2.polylines(frame, [pts], True, color, 2, cv2.LINE_AA)
                cv2.fillPoly(overlay, [pts], color)

                c = self.zone_counts[zname]
                label = f"{zname} | ENTRY: {c['ENTRY']} EXIT: {c['EXIT']}"
                cv2.putText(
                    frame,
                    label,
                    (pts[0][0], max(25, pts[0][1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

        for lname, lcfg in self.lines.items():
            pt_a = (int(lcfg["point_a"][0]), int(lcfg["point_a"][1]))
            pt_b = (int(lcfg["point_b"][0]), int(lcfg["point_b"][1]))
            color = tuple(lcfg.get("color", [0, 255, 255]))
            invert = bool(lcfg.get("invert_direction", False))

            cv2.line(frame, pt_a, pt_b, color, 3, cv2.LINE_AA)
            cv2.circle(frame, pt_a, 7, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(frame, pt_b, 7, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(frame, "A", (pt_a[0] - 18, pt_a[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "B", (pt_b[0] + 10, pt_b[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

            mid_x = int((pt_a[0] + pt_b[0]) / 2)
            mid_y = int((pt_a[1] + pt_b[1]) / 2)
            dx, dy = pt_b[0] - pt_a[0], pt_b[1] - pt_a[1]
            length = max(1.0, float(np.hypot(dx, dy)))
            sign = -1.0 if invert else 1.0
            nx = int(-sign * (dy / length) * 40)
            ny = int(sign * (dx / length) * 40)
            arrow_tip = (mid_x + nx, mid_y + ny)
            cv2.arrowedLine(frame, (mid_x, mid_y), arrow_tip, (0, 255, 0), 2, cv2.LINE_AA, tipLength=0.3)
            cv2.putText(frame, "ENTRY", arrow_tip, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

            c = self.line_counts[lname]
            text = f"{lname} | ENTRY: {c['ENTRY']}  EXIT: {c['EXIT']}"
            cv2.putText(
                frame,
                text,
                (pt_a[0], max(25, pt_a[1] - 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )

        if self.draw_trails:
            for gid, traj in self.trajectories.items():
                if active_gids is not None and gid not in active_gids:
                    continue
                if len(traj) >= 2:
                    for i in range(1, len(traj)):
                        cv2.line(frame, traj[i - 1], traj[i], (0, 255, 255), 2, cv2.LINE_AA)

        return frame

    def _record_event(self, person_id: Any, event_type: str, target: str) -> Dict[str, Any]:
        wall_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        ev = {
            "timestamp": wall_str,
            "global_id": person_id,
            "event": event_type,
            "target": target,
        }
        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"[{wall_str}] ID:{person_id} | {event_type} | {target}\n")
        return ev