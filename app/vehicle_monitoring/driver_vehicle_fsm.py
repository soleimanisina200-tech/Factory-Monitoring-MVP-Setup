# app/vehicle_monitoring/driver_vehicle_fsm.py
"""
Forklift <-> driver monitoring.

Everything is time based (seconds on the stream clock), so behaviour does not change with the processing FPS.

WHO DRIVES (driver binding)
  1. Visible in the cab   : a person whose skeleton/box sits in the cab region for board_confirm_sec (parked
                            vehicle, slow person), seated_board_confirm_sec (seated pose), or moving_board_confirm_sec
                            while travelling WITH a moving vehicle (co-motion). One person -> one vehicle.
  2. Vanished into the cab: ROOFED cabs often hide the driver completely. A person last seen at/inside the cab
                            who disappears, followed by the vehicle starting to move (or staying hidden inside the
                            cab region for hidden_board_confirm_sec), becomes the driver ("hidden onboard").
  3. Emerged from the cab : a NEW person appearing right next to a parked vehicle whose driver was hidden (or whose
                            occupant was unknown) is the driver climbing out.
  4. Moving, nobody known : OCCUPIED_UNKNOWN_DRIVER (someone must be driving - hidden by the roof).
  A different person confirmed in the cab while the bound driver is out -> driver change (old session closed).
  Driver boarding another vehicle -> his old session closes.

WHERE IS THE DRIVER
  in cab (visible, or hidden under the roof)        -> DRIVER_ONBOARD (onboard time accumulates)
  out of the cab for dismount_confirm_sec           -> DRIVER_EXITED (distance monitoring)
  distance > safe threshold for alarm_confirm_sec   -> SAFE_DISTANCE_VIOLATION_ALARM (cleared with hysteresis)
  dismounted and walked out of the camera           -> DRIVER_LEFT_VIEW_ALARM (alarm_on_driver_left_view)
  dismounted and invisible next to the vehicle      -> DRIVER_NOT_VISIBLE (+ optional alarm, driver_lost_alarm_sec)

Uncertain distances (feet hidden) never clear an alarm and raise one only when the distance exceeds the threshold
by `uncertain_margin_m`. The old weak appearance fallback (cosine 0.68 - true for most people in the same uniform)
is gone: the caller keeps person IDs consistent (identity manager) and reports merges via `rename_person`.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from app.core.events import ALARM, INFO, WARNING, make_event
from app.vehicle_monitoring.distance_estimator import (
    DistanceMeasurement,
    RealWorldDistanceEstimator,
    point_polygon_distance,
)


class VehicleSafetyState(str, Enum):
    UNASSIGNED = "UNASSIGNED"
    OCCUPIED_UNKNOWN_DRIVER = "OCCUPIED_UNKNOWN_DRIVER"
    DRIVER_ONBOARD = "DRIVER_ONBOARD"
    DRIVER_EXITED_SAFE = "DRIVER_EXITED_SAFE"
    DRIVER_EXITED_WARNING = "DRIVER_EXITED_WARNING"
    DISTANCE_VIOLATION_ALARM = "DISTANCE_VIOLATION_ALARM"
    DISTANCE_UNCERTAIN = "DISTANCE_UNCERTAIN"
    DRIVER_NOT_VISIBLE = "DRIVER_NOT_VISIBLE"
    DRIVER_NOT_VISIBLE_ALARM = "DRIVER_NOT_VISIBLE_ALARM"
    DRIVER_LEFT_VIEW_ALARM = "DRIVER_LEFT_VIEW_ALARM"


ALARM_EVENT = {
    "DISTANCE": ("SAFE_DISTANCE_VIOLATION_ALARM", VehicleSafetyState.DISTANCE_VIOLATION_ALARM),
    "NOT_VISIBLE": ("DRIVER_NOT_VISIBLE_ALARM", VehicleSafetyState.DRIVER_NOT_VISIBLE_ALARM),
    "LEFT_VIEW": ("DRIVER_LEFT_VIEW_ALARM", VehicleSafetyState.DRIVER_LEFT_VIEW_ALARM),
}


def _ids_of(pid: Optional[str]) -> Dict[str, Optional[int]]:
    """Person id ("7" verified worker, "T12" provisional track) -> worker_id / track_id event fields."""
    if pid is None:
        return {"worker_id": None, "track_id": None}
    s = str(pid)
    if s.startswith("T") and s[1:].isdigit():
        return {"worker_id": None, "track_id": int(s[1:])}
    if s.isdigit():
        return {"worker_id": int(s), "track_id": None}
    return {"worker_id": None, "track_id": None}


@dataclass
class PersonTrackHistory:
    person_id: str
    first_seen_ts: float
    spawn_px: Tuple[float, float]
    spawn_world_xy: Tuple[float, float]
    current_world_xy: Tuple[float, float]
    last_seen_ts: float = 0.0
    total_frames_seen: int = 0
    last_bbox: Optional[List[float]] = None
    world_history: Deque[Tuple[float, float, float]] = field(default_factory=lambda: deque(maxlen=90))
    px_history: Deque[Tuple[float, float, float]] = field(default_factory=lambda: deque(maxlen=90))
    left_view: bool = False
    vanish_ts: Optional[float] = None
    vanish_near: Dict[str, float] = field(default_factory=dict)      # vehicle -> metres to footprint
    vanish_cab: Dict[str, float] = field(default_factory=dict)       # vehicle -> cab score when last seen
    last_cab: Dict[str, float] = field(default_factory=dict)

    def update_position(self, world_xy: Tuple[float, float], ts: float,
                        ground_px: Optional[Tuple[float, float]] = None) -> None:
        self.current_world_xy = world_xy
        self.world_history.append((float(world_xy[0]), float(world_xy[1]), ts))
        if ground_px is not None:
            self.px_history.append((float(ground_px[0]), float(ground_px[1]), ts))
        self.total_frames_seen += 1
        self.last_seen_ts = ts

    def get_speed_mps(self, window_sec: float = 1.0) -> float:
        if len(self.world_history) < 3:
            return 0.0
        t_end = self.world_history[-1][2]
        old = next((h for h in self.world_history if t_end - h[2] <= window_sec), self.world_history[0])
        dt = t_end - old[2]
        if dt < 0.2:
            return 0.0
        last = self.world_history[-1]
        return float(np.hypot(last[0] - old[0], last[1] - old[1]) / dt)

    def max_distance_from_spawn_m(self) -> float:
        return float(np.hypot(self.current_world_xy[0] - self.spawn_world_xy[0],
                              self.current_world_xy[1] - self.spawn_world_xy[1]))


@dataclass
class VehicleSessionRecord:
    vehicle_id: str
    bbox: List[float]
    state: VehicleSafetyState = VehicleSafetyState.UNASSIGNED
    first_seen_ts: float = 0.0
    last_seen_ts: float = 0.0
    is_visible_in_current_frame: bool = True
    contour: Optional[np.ndarray] = None
    footprint_world: Optional[np.ndarray] = None
    # motion
    world_pos_history: Deque[Tuple[float, float, float]] = field(default_factory=lambda: deque(maxlen=120))
    px_history: Deque[Tuple[float, float, float]] = field(default_factory=lambda: deque(maxlen=120))
    is_currently_moving: bool = False
    has_ever_moved: bool = False
    last_moved_ts: float = -1e18
    moving_since: Optional[float] = None
    still_since: Optional[float] = None
    # driver
    associated_driver_id: Optional[str] = None
    driver_bbox: Optional[List[float]] = None
    driver_reid_embedding: Optional[np.ndarray] = None
    driver_visible: bool = False
    driver_hidden_onboard: bool = False
    driver_inferred: str = ""
    onboard: bool = False
    out_of_cab_since: Optional[float] = None
    driver_last_seen_ts: float = 0.0
    driver_last_seen_in_cab: bool = False
    driver_last_world_xy: Optional[Tuple[float, float]] = None
    exit_timestamp: Optional[float] = None
    lost_since: Optional[float] = None
    # session / seat time
    session_start_ts: Optional[float] = None
    stint_start_ts: Optional[float] = None
    onboard_closed_sec: float = 0.0
    unknown_occupied: bool = False
    unknown_since: Optional[float] = None
    # boarding candidate
    cand_pid: Optional[str] = None
    cand_time: float = 0.0
    comotion_fail_since: Optional[float] = None
    reboard_since: Optional[float] = None
    # distance & alarms
    last_measurement: Optional[DistanceMeasurement] = None
    violation_since: Optional[float] = None
    clear_since: Optional[float] = None
    alarm_active: bool = False
    alarm_type: str = ""
    alarm_since: Optional[float] = None

    def onboard_seconds(self, now: float) -> float:
        cur = (now - self.stint_start_ts) if (self.onboard and self.stint_start_ts is not None) else 0.0
        return float(self.onboard_closed_sec + max(0.0, cur))


class DriverVehicleMonitorFSM:
    def __init__(self, config: dict, distance_estimator: RealWorldDistanceEstimator):
        self.cfg = config
        self.estimator = distance_estimator
        p = config.get("pairing", {}) or {}
        d = config.get("distance", {}) or {}

        self.cab_shrink = float(p.get("cab_shrink", 0.08))
        self.cab_thresh = float(p.get("cab_score_threshold", p.get("entry_iou_threshold", 0.5)))
        self.board_confirm_sec = float(p.get("board_confirm_sec", 1.5))
        self.seated_board_confirm_sec = float(p.get("seated_board_confirm_sec", 0.6))
        self.moving_board_confirm_sec = float(p.get("moving_board_confirm_sec", 0.4))
        self.dismount_confirm_sec = float(p.get("dismount_confirm_sec", 0.6))
        self.max_boarding_walk_speed_mps = float(p.get("max_boarding_walk_speed_mps", 0.6))
        self.min_kpt_conf = float(p.get("min_keypoint_confidence", 0.30))

        self.vehicle_motion_thresh_m = float(p.get("vehicle_motion_threshold_m", 0.40))
        self.motion_window_sec = float(p.get("motion_window_sec", 1.0))
        self.stop_confirm_sec = float(p.get("stop_confirm_sec", 1.5))
        self.min_comotion_cos = float(p.get("comotion_min_cosine", 0.6))
        self.comotion_speed_ratio = (float(p.get("comotion_min_speed_ratio", 0.4)),
                                     float(p.get("comotion_max_speed_ratio", 2.0)))

        # roofed cabs / hidden drivers
        self.board_vanish_radius_m = float(p.get("board_vanish_radius_m", 1.2))
        self.vanish_board_window_sec = float(p.get("vanish_board_window_sec", 30.0))
        self.hidden_board_confirm_sec = float(p.get("hidden_board_confirm_sec", 2.0))
        self.dismount_spawn_radius_m = float(p.get("dismount_spawn_radius_m", 1.5))
        self.emergence_window_sec = float(p.get("emergence_window_sec", 1.5))
        self.recent_activity_sec = float(p.get("recent_activity_sec", 120.0))
        self.require_vehicle_activity_for_spawn = bool(p.get("require_vehicle_activity_for_spawn", True))
        self.allow_provisional_spawn = bool(p.get("allow_provisional_ids_for_spawn", False))
        self.spawn_border_margin_px = float(p.get("spawn_border_margin_px", 40))
        self.unknown_occupancy_timeout_sec = float(p.get("unknown_occupancy_timeout_sec", 120.0))

        self.handover_iou_thresh = float(p.get("driver_handover_iou", 0.5))
        self.handover_max_gap_sec = float(p.get("driver_handover_max_gap_sec", 1.0))
        self.unassigned_ghost_timeout_sec = float(p.get("unassigned_ghost_timeout_sec", 3.0))
        self.assigned_ghost_timeout_sec = float(p.get("assigned_ghost_timeout_sec", 30.0))
        self.person_history_ttl_sec = float(p.get("person_history_ttl_sec", 60.0))
        self.session_timeout_sec = float(p.get("session_timeout_sec", 900.0))
        # alarm policy (confirm with the site safety officer)
        self.driver_lost_alarm_sec = float(p.get("driver_lost_alarm_sec", 0.0))
        self.alarm_on_driver_left_view = bool(p.get("alarm_on_driver_left_view", True))

        self.safe_thresh_m = float(d.get("safe_distance_threshold_m", 3.0))
        self.warn_thresh_m = float(d.get("warning_distance_threshold_m", 2.0))
        self.alarm_confirm_sec = float(d.get("alarm_confirm_sec", 1.0))
        self.alarm_clear_sec = float(d.get("alarm_clear_sec", 2.0))
        self.clear_hysteresis_m = float(d.get("clear_hysteresis_m", 0.3))
        self.uncertain_margin_m = float(d.get("uncertain_margin_m", 1.5))

        self.vehicles: Dict[str, VehicleSessionRecord] = {}
        self.person_histories: Dict[str, PersonTrackHistory] = {}
        self.driver_of: Dict[str, str] = {}
        self._last_now: Optional[float] = None
        self._frame_shape: Tuple[int, int] = (1080, 1920)

    # ==================================================================
    # Geometry helpers
    # ==================================================================
    @staticmethod
    def _bbox_iou(a: List[float], b: List[float]) -> float:
        ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0.0:
            return 0.0
        aa = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
        ab = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
        return float(inter / (aa + ab - inter))

    def _cab_rect(self, vb: List[float]) -> Tuple[float, float, float, float]:
        vw, vh = vb[2] - vb[0], vb[3] - vb[1]
        s = self.cab_shrink
        return vb[0] + s * vw, vb[1] + 0.04 * vh, vb[2] - s * vw, vb[3] - 0.04 * vh

    def cab_score(self, person: Dict[str, Any], vbbox: List[float]) -> Tuple[float, bool]:
        """
        0..1 evidence that `person` sits in the cab of the vehicle, and whether the pose is CLEARLY seated
        (legs visible and folded). Works with partial visibility (roof / mast / side panels hide part of the body).
        Hidden legs raise the score a little but do not count as seated: someone standing BEHIND a parked forklift
        also has hidden legs, so that case needs the full board_confirm_sec (and is undone if he does not travel
        with the vehicle once it moves).
        """
        pb = person["bbox"]
        cx1, cy1, cx2, cy2 = self._cab_rect(vbbox)
        ix1, iy1, ix2, iy2 = max(pb[0], cx1), max(pb[1], cy1), min(pb[2], cx2), min(pb[3], cy2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area = max(1.0, (pb[2] - pb[0]) * (pb[3] - pb[1]))
        containment = inter / area
        if containment < 0.30:
            return 0.0, False
        vh = vbbox[3] - vbbox[1]
        kp = person.get("keypoints")
        k = np.asarray(kp) if kp is not None else None
        if k is None or k.ndim != 2 or len(k) < 17:
            foot = person.get("foot_px")
            if foot is not None and not person.get("ground_uncertain", False) and foot[1] > vbbox[3] - 0.03 * vh:
                return 0.15 * containment, False  # feet on the floor in front of the vehicle
            return float(containment), False

        conf = k[:, 2] if k.shape[1] >= 3 else np.ones(len(k))
        vis = (conf >= self.min_kpt_conf) & (k[:, 0] > 1) & (k[:, 1] > 1)
        ankles = [i for i in (15, 16) if vis[i]]
        if ankles:
            ankle_y = float(np.mean([k[i][1] for i in ankles]))
            if ankle_y > vbbox[3] - 0.04 * vh:
                return 0.15 * containment, False  # standing on the floor in front of / beside the vehicle
        inside = vis & (k[:, 0] >= cx1) & (k[:, 0] <= cx2) & (k[:, 1] >= cy1) & (k[:, 1] <= cy2)
        n_vis = int(vis.sum())
        inside_frac = float(inside.sum()) / n_vis if n_vis >= 3 else containment
        seated = legs_hidden = False
        sh = [k[i][1] for i in (5, 6) if vis[i]]
        hp = [k[i][1] for i in (11, 12) if vis[i]]
        if sh and hp:
            torso_h = max(1.0, float(np.mean(hp) - np.mean(sh)))
            if not ankles:
                legs_hidden = True  # torso in the cab, legs hidden by the chassis / door (or by the vehicle)
            else:
                leg_h = max(0.0, float(np.mean([k[i][1] for i in ankles])) - float(np.mean(hp)))
                seated = (leg_h / torso_h) < 1.25
        score = 0.5 * containment + 0.5 * inside_frac + (0.15 if seated else (0.05 if legs_hidden else 0.0))
        return float(min(1.0, score)), seated

    def _footprint(self, rec: VehicleSessionRecord) -> np.ndarray:
        if rec.footprint_world is None:
            rec.footprint_world, _ = self.estimator.vehicle_footprint(rec.bbox, self._frame_shape, rec.contour)
        return rec.footprint_world

    def _dist_to_vehicle(self, rec: VehicleSessionRecord, world_xy: Tuple[float, float]) -> float:
        return point_polygon_distance(world_xy, self._footprint(rec))[0]

    def _update_vehicle_motion(self, rec: VehicleSessionRecord, now: float) -> None:
        x1, y1, x2, y2 = rec.bbox
        base = (0.5 * (x1 + x2), float(y2))
        wx, wy = self.estimator.pixel_to_world(base)
        rec.world_pos_history.append((wx, wy, now))
        rec.px_history.append((base[0], base[1], now))
        old = next((h for h in rec.world_pos_history if now - h[2] <= self.motion_window_sec),
                   rec.world_pos_history[0])
        if now - old[2] < 0.5 * self.motion_window_sec:
            return
        disp = float(np.hypot(wx - old[0], wy - old[1]))
        if disp >= self.vehicle_motion_thresh_m:
            if not rec.is_currently_moving:
                rec.moving_since = now
            rec.is_currently_moving = True
            rec.has_ever_moved = True
            rec.last_moved_ts = now
            rec.still_since = None
        elif disp < 0.5 * self.vehicle_motion_thresh_m:
            if rec.still_since is None:
                rec.still_since = now
            if rec.is_currently_moving and now - rec.still_since >= self.stop_confirm_sec:
                rec.is_currently_moving = False
                rec.moving_since = None

    def _co_motion(self, rec: VehicleSessionRecord, hist: PersonTrackHistory, window: float = 1.0) -> Optional[bool]:
        """Does the person travel WITH the vehicle (same direction, similar speed)? None = cannot judge yet."""
        if len(rec.px_history) < 3 or len(hist.px_history) < 3:
            return None
        t_end = min(rec.px_history[-1][2], hist.px_history[-1][2])

        def disp(h):
            pts = [x for x in h if t_end - window <= x[2] <= t_end]
            if len(pts) < 3 or pts[-1][2] - pts[0][2] < 0.3:
                return None
            return np.array([pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]], dtype=np.float64)

        v, p = disp(rec.px_history), disp(hist.px_history)
        if v is None or p is None:
            return None
        v_len, p_len = float(np.linalg.norm(v)), float(np.linalg.norm(p))
        if v_len < 3.0:
            return None
        cos = float(np.dot(v, p) / (v_len * p_len + 1e-8)) if p_len > 1e-6 else 0.0
        lo, hi = self.comotion_speed_ratio
        return bool(cos >= self.min_comotion_cos and lo <= p_len / v_len <= hi)

    def _deduplicate_vehicle_tracks(self, detected_vehicles: List[Dict], now: float) -> None:
        """A tracker id switch on the same vehicle keeps the session (IoU with a just-lost record)."""
        active_ids = {str(v["id"]) for v in detected_vehicles}
        for v in detected_vehicles:
            new_id = str(v["id"])
            if new_id in self.vehicles:
                continue
            for old_id, old in list(self.vehicles.items()):
                if old_id in active_ids or now - old.last_seen_ts >= 3.0:
                    continue
                if self._bbox_iou(v["bbox"], old.bbox) > 0.6:
                    old.vehicle_id = new_id
                    self.vehicles[new_id] = old
                    del self.vehicles[old_id]
                    if old.associated_driver_id is not None:
                        self.driver_of[old.associated_driver_id] = new_id
                    break

    # ==================================================================
    # Session bookkeeping
    # ==================================================================
    def _event(self, etype: str, ts: float, rec: VehicleSessionRecord, pid: Optional[str], severity: str = INFO,
               value: Optional[float] = None, **details: Any) -> Dict[str, Any]:
        return make_event(etype, ts, severity=severity, vehicle_id=rec.vehicle_id, subject=f"vehicle_{rec.vehicle_id}",
                          value=value, driver_id=pid, **_ids_of(pid), **details)

    def _raise_alarm(self, rec: VehicleSessionRecord, kind: str, now: float, events: List[Dict],
                     value: Optional[float] = None, **details: Any) -> None:
        etype, state = ALARM_EVENT[kind]
        if rec.alarm_active and rec.alarm_type == kind:
            rec.state = state
            return
        rec.alarm_active, rec.alarm_type, rec.alarm_since = True, kind, now
        rec.clear_since = None
        rec.state = state
        events.append(self._event(etype, now, rec, rec.associated_driver_id, ALARM, value,
                                  threshold_m=self.safe_thresh_m if kind == "DISTANCE" else None, **details))

    def _clear_alarm(self, rec: VehicleSessionRecord, now: float, events: List[Dict], reason: str) -> None:
        if not rec.alarm_active:
            return
        events.append(self._event("DRIVER_ALARM_CLEARED", now, rec, rec.associated_driver_id, INFO,
                                  alarm_type=rec.alarm_type, reason=reason,
                                  alarm_duration_sec=round(now - (rec.alarm_since or now), 1)))
        rec.alarm_active, rec.alarm_type, rec.alarm_since = False, "", None
        rec.violation_since = rec.clear_since = None

    def _board(self, rec: VehicleSessionRecord, pid: str, ts: float, events: List[Dict], how: str,
               person: Optional[Dict[str, Any]] = None) -> None:
        if rec.associated_driver_id not in (None, pid):
            self._end_session(rec, ts, events, "DRIVER_CHANGED")
        other_vid = self.driver_of.get(pid)
        if other_vid is not None and other_vid != rec.vehicle_id and other_vid in self.vehicles:
            self._end_session(self.vehicles[other_vid], ts, events, "DRIVER_MOVED_TO_OTHER_VEHICLE")
        if rec.alarm_active:
            self._clear_alarm(rec, ts, events, "DRIVER_RETURNED" if rec.associated_driver_id == pid else
                              "OTHER_DRIVER_BOARDED")
        new_session = rec.associated_driver_id != pid or rec.session_start_ts is None
        if new_session:
            start = ts
            if rec.unknown_occupied and rec.unknown_since is not None:
                start = min(ts, rec.unknown_since)  # it was (hidden) occupied since then
            rec.session_start_ts = start
            rec.onboard_closed_sec = max(0.0, ts - start)
        rec.associated_driver_id = pid
        self.driver_of[pid] = rec.vehicle_id
        rec.onboard = True
        rec.stint_start_ts = ts
        rec.out_of_cab_since = None
        rec.reboard_since = None
        rec.exit_timestamp = None
        rec.lost_since = None
        rec.unknown_occupied, rec.unknown_since = False, None
        rec.driver_inferred = how
        rec.driver_hidden_onboard = how in ("VANISHED_INTO_CAB", "VEHICLE_MOVED")
        rec.driver_last_seen_in_cab = True
        rec.last_measurement = None
        rec.violation_since = rec.clear_since = None
        rec.driver_last_seen_ts = max(rec.driver_last_seen_ts, ts)
        if person is not None:
            rec.driver_bbox = list(person["bbox"])
            rec.driver_visible = True
            rec.driver_last_seen_ts = self._last_now if self._last_now is not None else ts
            if person.get("embedding") is not None:
                rec.driver_reid_embedding = np.asarray(person["embedding"])
        rec.state = VehicleSafetyState.DRIVER_ONBOARD
        events.append(self._event("DRIVER_BOARDED", ts, rec, pid, INFO, how=how, reboard=not new_session))

    def _dismount(self, rec: VehicleSessionRecord, ts: float, events: List[Dict], how: str) -> None:
        if not rec.onboard:
            return
        stint = max(0.0, ts - (rec.stint_start_ts if rec.stint_start_ts is not None else ts))
        rec.onboard_closed_sec += stint
        rec.onboard = False
        rec.stint_start_ts = None
        rec.driver_hidden_onboard = False
        rec.exit_timestamp = ts
        rec.out_of_cab_since = None
        rec.state = VehicleSafetyState.DRIVER_EXITED_SAFE
        events.append(self._event("DRIVER_EXITED", ts, rec, rec.associated_driver_id, INFO, value=round(stint, 1),
                                  how=how, stint_onboard_sec=round(stint, 1)))

    def _end_session(self, rec: VehicleSessionRecord, ts: float, events: List[Dict], reason: str) -> None:
        pid = rec.associated_driver_id
        if pid is None:
            rec.unknown_occupied, rec.unknown_since = False, None
            return
        if rec.onboard and rec.stint_start_ts is not None:
            rec.onboard_closed_sec += max(0.0, ts - rec.stint_start_ts)
        if rec.alarm_active:
            self._clear_alarm(rec, ts, events, reason)
        events.append(self._event(
            "DRIVER_SESSION_END", ts, rec, pid, INFO, value=round(rec.onboard_closed_sec, 1), reason=reason,
            session_start_ts=None if rec.session_start_ts is None else round(rec.session_start_ts, 3),
            session_sec=None if rec.session_start_ts is None else round(ts - rec.session_start_ts, 1),
            onboard_sec=round(rec.onboard_closed_sec, 1),
        ))
        if self.driver_of.get(pid) == rec.vehicle_id:
            del self.driver_of[pid]
        rec.associated_driver_id = None
        rec.driver_bbox = None
        rec.driver_reid_embedding = None
        rec.driver_visible = rec.driver_hidden_onboard = rec.onboard = False
        rec.driver_inferred = ""
        rec.session_start_ts = rec.stint_start_ts = rec.exit_timestamp = rec.lost_since = None
        rec.out_of_cab_since = rec.reboard_since = rec.comotion_fail_since = None
        rec.onboard_closed_sec = 0.0
        rec.last_measurement = None
        rec.violation_since = rec.clear_since = None
        rec.state = VehicleSafetyState.UNASSIGNED

    def _set_unknown_occupied(self, rec: VehicleSessionRecord, now: float, events: List[Dict], **details: Any) -> None:
        if not rec.unknown_occupied:
            rec.unknown_occupied, rec.unknown_since = True, now
            events.append(self._event("VEHICLE_OCCUPIED_UNKNOWN_DRIVER", now, rec, None, INFO, **details))
        rec.state = VehicleSafetyState.OCCUPIED_UNKNOWN_DRIVER

    def close_all(self, ts: Optional[float] = None, reason: str = "SHUTDOWN") -> List[Dict]:
        """Service stop: close every driver session so its seat time is recorded."""
        ts = self._last_now if ts is None else ts
        events: List[Dict] = []
        if ts is None:
            return events
        for rec in self.vehicles.values():
            if rec.associated_driver_id is not None:
                self._end_session(rec, ts, events, reason)
        return events

    def rename_person(self, old: str, new: str) -> None:
        """The identity layer decided `old` and `new` are the same person (track verified / IDs merged)."""
        old, new = str(old), str(new)
        if old == new:
            return
        h = self.person_histories.pop(old, None)
        if h is not None:
            h.person_id = new
            existing = self.person_histories.get(new)
            if existing is None or existing.last_seen_ts < h.last_seen_ts:
                self.person_histories[new] = h
        vid = self.driver_of.pop(old, None)
        if vid is not None and vid in self.vehicles:
            rec = self.vehicles[vid]
            if new not in self.driver_of:
                rec.associated_driver_id = new
                self.driver_of[new] = vid
        for rec in self.vehicles.values():
            if rec.cand_pid == old:
                rec.cand_pid = new
        self.estimator.forget(f"driver_{old}")

    # ==================================================================
    # Main update
    # ==================================================================
    def update(
        self,
        detected_vehicles: List[Dict],
        detected_persons: List[Dict],
        frame_shape: Tuple[int, ...],
        now: Optional[float] = None,
        left_view_ids: Optional[Iterable[str]] = None,
    ) -> Tuple[Dict[str, VehicleSessionRecord], List[Dict]]:
        """
        `now` = stream clock (seconds). `left_view_ids` = person ids the identity manager reports as having
        walked out of the camera this frame.
        """
        now = time.time() if now is None else float(now)
        dt = 0.0 if self._last_now is None else float(np.clip(now - self._last_now, 0.0, 1.0))
        self._last_now = now
        self._frame_shape = (int(frame_shape[0]), int(frame_shape[1]))
        H, W = self._frame_shape
        events: List[Dict] = []

        # ---- 1. persons --------------------------------------------------------------------------------------
        person_map: Dict[str, Dict] = {}
        for p in detected_persons:
            pid = str(p["id"])
            person_map[pid] = p
            x1, y1, x2, y2 = p["bbox"]
            ground = p.get("foot_px") or (0.5 * (x1 + x2), float(y2))
            bev = p.get("bev_xy")
            world = (float(bev[0]), float(bev[1])) if bev is not None else self.estimator.pixel_to_world(ground)
            hist = self.person_histories.get(pid)
            if hist is None:
                hist = self.person_histories[pid] = PersonTrackHistory(
                    person_id=pid, first_seen_ts=now, spawn_px=(0.5 * (x1 + x2), 0.5 * (y1 + y2)),
                    spawn_world_xy=world, current_world_xy=world,
                )
            hist.update_position(world, now, (float(ground[0]), float(ground[1])))
            hist.last_bbox = list(p["bbox"])
            hist.left_view = False
            hist.vanish_ts = None
            hist.vanish_near.clear()
            hist.vanish_cab.clear()
        for pid in left_view_ids or ():
            h = self.person_histories.get(str(pid))
            if h is not None:
                h.left_view = True

        # ---- 2. vehicles -------------------------------------------------------------------------------------
        for rec in self.vehicles.values():
            rec.is_visible_in_current_frame = False
        self._deduplicate_vehicle_tracks(detected_vehicles, now)
        visible: List[VehicleSessionRecord] = []
        for v in detected_vehicles:
            vid = str(v["id"])
            rec = self.vehicles.get(vid)
            if rec is None:
                rec = self.vehicles[vid] = VehicleSessionRecord(vehicle_id=vid, bbox=list(v["bbox"]),
                                                                first_seen_ts=now, last_seen_ts=now)
            rec.bbox = list(v["bbox"])
            rec.contour = v.get("contour")
            rec.footprint_world = None
            rec.last_seen_ts = now
            rec.is_visible_in_current_frame = True
            self._update_vehicle_motion(rec, now)
            visible.append(rec)

        # ---- 3. cab evidence (person x vehicle) --------------------------------------------------------------
        scores: Dict[Tuple[str, str], Tuple[float, bool]] = {}
        for rec in visible:
            for pid, p in person_map.items():
                if self._bbox_iou(p["bbox"], rec.bbox) <= 0.0 and not _box_inside(p["bbox"], rec.bbox):
                    continue
                s = self.cab_score(p, rec.bbox)
                if s[0] > 0:
                    scores[(rec.vehicle_id, pid)] = s
        for pid in person_map:
            self.person_histories[pid].last_cab = {vid: s[0] for (vid, q), s in scores.items() if q == pid}

        # ---- 4. who just vanished, and next to which vehicle? -------------------------------------------------
        for pid, h in self.person_histories.items():
            if pid in person_map or h.vanish_ts is not None:
                continue
            h.vanish_ts = h.last_seen_ts
            for rec in visible:
                dist = self._dist_to_vehicle(rec, h.current_world_xy)
                cab = h.last_cab.get(rec.vehicle_id, 0.0)
                if dist <= self.board_vanish_radius_m or cab >= 0.3:
                    h.vanish_near[rec.vehicle_id] = dist
                    h.vanish_cab[rec.vehicle_id] = cab

        # ---- 5. vehicles that have a driver -------------------------------------------------------------------
        for rec in visible:
            if rec.associated_driver_id is not None:
                self._process_driver(rec, person_map, scores, now, events)

        # ---- 6. boarding / driver change / roof inference ----------------------------------------------------
        self._visible_boarding(visible, person_map, scores, dt, now, events)
        for rec in visible:
            self._hidden_inference(rec, person_map, now, events)
        self._emergence(visible, person_map, now, W, H, events)
        for rec in visible:
            if rec.associated_driver_id is None and not rec.unknown_occupied:
                rec.state = VehicleSafetyState.UNASSIGNED
            elif rec.associated_driver_id is None and rec.unknown_occupied:
                rec.state = VehicleSafetyState.OCCUPIED_UNKNOWN_DRIVER
                if not rec.is_currently_moving and rec.still_since is not None and \
                        now - rec.still_since >= self.unknown_occupancy_timeout_sec:
                    rec.unknown_occupied, rec.unknown_since = False, None
                    rec.state = VehicleSafetyState.UNASSIGNED

        # ---- 7. stale vehicles, session timeouts, memory bounds -----------------------------------------------
        for vid, rec in list(self.vehicles.items()):
            age = now - rec.last_seen_ts
            occupied = rec.associated_driver_id is not None or rec.unknown_occupied
            if age > (self.assigned_ghost_timeout_sec if occupied else self.unassigned_ghost_timeout_sec):
                self._end_session(rec, rec.last_seen_ts, events, "VEHICLE_LEFT_VIEW")
                self.vehicles.pop(vid, None)
                self.estimator.forget(f"veh_{vid}_")
                continue
            if (rec.associated_driver_id is not None and not rec.onboard and rec.exit_timestamp is not None
                    and now - rec.exit_timestamp >= self.session_timeout_sec):
                self._end_session(rec, now, events, "TIMEOUT_DRIVER_AWAY")
        bound = set(self.driver_of)
        for pid in [pid for pid, h in self.person_histories.items()
                    if pid not in person_map and pid not in bound and now - h.last_seen_ts > self.person_history_ttl_sec]:
            del self.person_histories[pid]
            self.estimator.forget(f"driver_{pid}")
        return self.vehicles, events

    # ------------------------------------------------------------------
    def _find_handover(self, rec: VehicleSessionRecord, person_map: Dict[str, Dict], now: float) -> Optional[str]:
        """Driver's id changed (tracker switch) but the same body is still where the driver was."""
        if rec.driver_bbox is None or now - rec.driver_last_seen_ts > self.handover_max_gap_sec:
            return None
        best_iou, best = 0.0, None
        for pid, p in person_map.items():
            if pid in self.driver_of and self.driver_of[pid] != rec.vehicle_id:
                continue
            iou = self._bbox_iou(p["bbox"], rec.driver_bbox)
            if iou < self.handover_iou_thresh or iou <= best_iou:
                continue
            if self._clearly_different(p.get("embedding"), rec.driver_reid_embedding):
                continue  # clearly a different person standing where the driver was
            best_iou, best = iou, pid
        return best

    @staticmethod
    def _clearly_different(emb_a: Optional[np.ndarray], emb_b: Optional[np.ndarray], max_cos: float = 0.35) -> bool:
        """Veto only (never a positive match: in uniforms most people look alike)."""
        if emb_a is None or emb_b is None:
            return False
        a, b = np.asarray(emb_a).ravel(), np.asarray(emb_b).ravel()
        if a.shape != b.shape:
            return False
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)) < max_cos

    def _process_driver(self, rec: VehicleSessionRecord, person_map: Dict[str, Dict],
                        scores: Dict[Tuple[str, str], Tuple[float, bool]], now: float, events: List[Dict]) -> None:
        pid = rec.associated_driver_id
        p = person_map.get(pid)
        if p is None:
            new_pid = self._find_handover(rec, person_map, now)
            if new_pid is not None:
                self.driver_of.pop(pid, None)
                rec.associated_driver_id = pid = new_pid
                self.driver_of[pid] = rec.vehicle_id
                p = person_map[pid]

        if p is not None:
            hist = self.person_histories[pid]
            rec.driver_visible = True
            rec.driver_last_seen_ts = now
            rec.driver_bbox = list(p["bbox"])
            rec.driver_last_world_xy = hist.current_world_xy
            rec.lost_since = None
            if p.get("embedding") is not None:
                rec.driver_reid_embedding = np.asarray(p["embedding"])
            score, seated = scores.get((rec.vehicle_id, pid), (0.0, False))
            in_cab = score >= self.cab_thresh
            rec.driver_last_seen_in_cab = in_cab
            if in_cab and not rec.onboard:
                # The DISMOUNTED driver overlaps the cab again: climbing back in - or only walking past the
                # forklift in the image. Re-boarding needs the same confirmation as boarding; until then he is
                # still monitored as dismounted (an instant re-board used to mask the distance alarm).
                if rec.is_currently_moving:
                    ok, required = self._co_motion(rec, hist) is True, self.moving_board_confirm_sec
                else:
                    ok = hist.get_speed_mps() <= self.max_boarding_walk_speed_mps
                    required = self.seated_board_confirm_sec if seated else self.board_confirm_sec
                rec.reboard_since = (rec.reboard_since or now) if ok else None
                if rec.reboard_since is not None and now - rec.reboard_since >= required:
                    start, rec.reboard_since = rec.reboard_since, None
                    self._board(rec, pid, start, events, "VISIBLE_IN_CAB", p)
                    rec.state = VehicleSafetyState.DRIVER_ONBOARD
                    return
                in_cab = False
            elif not in_cab:
                rec.reboard_since = None
            if in_cab:
                rec.out_of_cab_since = None
                rec.driver_hidden_onboard = False
                if rec.is_currently_moving:
                    # self-correction: an operator travels WITH the vehicle. Someone who only overlapped the cab
                    # in the image (standing behind a parked forklift) stays put when it drives off.
                    cm = self._co_motion(rec, hist)
                    if cm is False:
                        rec.comotion_fail_since = rec.comotion_fail_since or now
                        if now - rec.comotion_fail_since >= 1.0:
                            rec.comotion_fail_since = None
                            self._end_session(rec, now, events, "NOT_MOVING_WITH_VEHICLE")
                            self._set_unknown_occupied(rec, now, events, reason="bound person did not move with it")
                            return
                    elif cm is True:
                        rec.comotion_fail_since = None
                rec.state = VehicleSafetyState.DRIVER_ONBOARD
                return
            if rec.onboard:
                if rec.out_of_cab_since is None:
                    rec.out_of_cab_since = now
                if now - rec.out_of_cab_since < self.dismount_confirm_sec:
                    rec.state = VehicleSafetyState.DRIVER_ONBOARD
                    return
                self._dismount(rec, rec.out_of_cab_since, events, "VISIBLE")
            # dismounted: is somebody else driving the vehicle away?
            if rec.is_currently_moving and rec.moving_since is not None and now - rec.moving_since >= 1.0:
                self._end_session(rec, now, events, "VEHICLE_MOVED_WITHOUT_DRIVER")
                self._set_unknown_occupied(rec, now, events, reason="moved while the bound driver was outside")
                return
            meas = self.estimator.estimate_driver_vehicle_distance(
                driver_id=pid, driver_bbox=p["bbox"], vehicle_id=rec.vehicle_id, vehicle_bbox=rec.bbox,
                frame_shape=self._frame_shape, keypoints=p.get("keypoints"), foot_px=p.get("foot_px"),
                ground_uncertain=p.get("ground_uncertain"), vehicle_contour=rec.contour,
            )
            rec.last_measurement = meas
            self._apply_distance(rec, meas, now, events)
            return

        # ---- driver not visible ----
        rec.driver_visible = False
        if rec.onboard:
            rec.driver_hidden_onboard = True     # roof / mast / cab panels hide the seated driver
            rec.state = VehicleSafetyState.DRIVER_ONBOARD
            return
        hist = self.person_histories.get(pid)
        last_xy = rec.driver_last_world_xy or (hist.current_world_xy if hist else None)
        near_cab = last_xy is not None and self._dist_to_vehicle(rec, last_xy) <= self.board_vanish_radius_m
        if rec.is_currently_moving:
            if near_cab:  # he climbed back in and drove off under the roof
                self._board(rec, pid, rec.moving_since or now, events, "VEHICLE_MOVED")
            else:
                self._end_session(rec, now, events, "VEHICLE_MOVED_WITHOUT_DRIVER")
                self._set_unknown_occupied(rec, now, events, reason="moved while the bound driver was away")
            return
        if hist is not None and hist.left_view:
            if self.alarm_on_driver_left_view:
                self._raise_alarm(rec, "LEFT_VIEW", now, events, last_distance_m=(
                    None if rec.last_measurement is None else rec.last_measurement.distance_m))
            else:
                rec.state = VehicleSafetyState.DRIVER_NOT_VISIBLE
            return
        if rec.alarm_active:
            return  # last seen beyond the limit: still away until seen back near the vehicle
        if rec.lost_since is None:
            rec.lost_since = rec.driver_last_seen_ts
        rec.state = VehicleSafetyState.DRIVER_NOT_VISIBLE
        if (near_cab and rec.driver_last_seen_in_cab is False and hist is not None
                and hist.vanish_cab.get(rec.vehicle_id, 0.0) >= 0.3
                and now - rec.driver_last_seen_ts >= self.hidden_board_confirm_sec):
            self._board(rec, pid, rec.driver_last_seen_ts, events, "VANISHED_INTO_CAB")
            return
        if self.driver_lost_alarm_sec > 0 and now - rec.lost_since >= self.driver_lost_alarm_sec:
            self._raise_alarm(rec, "NOT_VISIBLE", now, events, lost_sec=round(now - rec.lost_since, 1))

    def _apply_distance(self, rec: VehicleSessionRecord, meas: DistanceMeasurement, now: float,
                        events: List[Dict]) -> None:
        d, unc = meas.distance_m, meas.is_uncertain
        far = d is not None and ((not unc and d > self.safe_thresh_m) or
                                 (unc and d > self.safe_thresh_m + self.uncertain_margin_m))
        near = d is not None and not unc and d < self.safe_thresh_m - self.clear_hysteresis_m
        rec.violation_since = (rec.violation_since or now) if far else None

        if rec.alarm_active and rec.alarm_type in ("LEFT_VIEW", "NOT_VISIBLE"):
            # the driver is visible again: continue as a distance alarm, or clear when he is back near
            if far:
                rec.alarm_type = "DISTANCE"
            elif near:
                self._clear_alarm(rec, now, events, "DRIVER_RETURNED")
        if not rec.alarm_active and far and now - rec.violation_since >= self.alarm_confirm_sec:
            self._raise_alarm(rec, "DISTANCE", now, events, value=d, uncertain=unc, reason=meas.uncertainty_reason)
        elif rec.alarm_active:
            if near:
                rec.clear_since = rec.clear_since or now
                if now - rec.clear_since >= self.alarm_clear_sec:
                    self._clear_alarm(rec, now, events, "DRIVER_RETURNED")
            else:
                rec.clear_since = None

        if rec.alarm_active:
            rec.state = ALARM_EVENT[rec.alarm_type][1]
        elif d is None or unc:
            rec.state = VehicleSafetyState.DISTANCE_UNCERTAIN
        elif d > self.warn_thresh_m:
            rec.state = VehicleSafetyState.DRIVER_EXITED_WARNING
        else:
            rec.state = VehicleSafetyState.DRIVER_EXITED_SAFE

    def _visible_boarding(self, visible: List[VehicleSessionRecord], person_map: Dict[str, Dict],
                          scores: Dict[Tuple[str, str], Tuple[float, bool]], dt: float, now: float,
                          events: List[Dict]) -> None:
        pairs = []
        for rec in visible:
            if rec.associated_driver_id is not None and rec.onboard and not rec.driver_hidden_onboard:
                continue  # the bound driver is visibly in the cab
            for (vid, pid), (score, seated) in scores.items():
                if vid != rec.vehicle_id or score < self.cab_thresh or pid == rec.associated_driver_id:
                    continue
                other = self.driver_of.get(pid)
                if other is not None and other in self.vehicles and self.vehicles[other].onboard:
                    continue  # onboard another vehicle
                pairs.append((score, rec.vehicle_id, pid, seated))
        pairs.sort(key=lambda t: t[0], reverse=True)
        chosen: Dict[str, Tuple[str, bool]] = {}
        taken: Set[str] = set()
        for score, vid, pid, seated in pairs:
            if vid not in chosen and pid not in taken:
                chosen[vid] = (pid, seated)
                taken.add(pid)

        for rec in visible:
            pick = chosen.get(rec.vehicle_id)
            if pick is None:
                rec.cand_pid, rec.cand_time = None, 0.0
                continue
            pid, seated = pick
            if pid != rec.cand_pid:
                rec.cand_pid, rec.cand_time = pid, 0.0
            hist = self.person_histories[pid]
            if rec.is_currently_moving:
                cm = self._co_motion(rec, hist)
                if cm is None:
                    continue
                if cm is False:
                    rec.cand_time = 0.0
                    continue
                required = self.moving_board_confirm_sec
            elif hist.get_speed_mps() <= self.max_boarding_walk_speed_mps:
                required = self.seated_board_confirm_sec if seated else self.board_confirm_sec
            else:
                rec.cand_time = 0.0
                continue
            rec.cand_time += dt
            if rec.cand_time + 1e-9 < required:
                continue
            start = now - rec.cand_time
            if rec.associated_driver_id is not None and rec.onboard and rec.driver_hidden_onboard:
                # someone is now SEEN in the cab where the known driver was hidden: same person under a new id
                # (the roof broke the identity) unless he is an established other worker
                if (now - hist.first_seen_ts < 5.0 or pid.startswith("T")) and not self._clearly_different(
                        person_map[pid].get("embedding"), rec.driver_reid_embedding):
                    old = rec.associated_driver_id
                    self.driver_of.pop(old, None)
                    rec.associated_driver_id = pid
                    self.driver_of[pid] = rec.vehicle_id
                    rec.driver_hidden_onboard = False
                    rec.driver_inferred = "VISIBLE_IN_CAB"
                    events.append(self._event("DRIVER_REBOUND", now, rec, pid, INFO, previous_driver=old))
                    rec.cand_pid, rec.cand_time = None, 0.0
                    continue
            self._board(rec, pid, start, events, "VISIBLE_IN_CAB", person_map[pid])
            rec.cand_pid, rec.cand_time = None, 0.0

    def _hidden_inference(self, rec: VehicleSessionRecord, person_map: Dict[str, Dict], now: float,
                          events: List[Dict]) -> None:
        """Roofed cabs: the driver vanished INTO the cab (seen last at/inside it) -> hidden onboard."""
        if rec.associated_driver_id is not None:
            return
        window = self.vanish_board_window_sec
        cands = []
        for pid, h in self.person_histories.items():
            if pid in person_map or h.vanish_ts is None or h.left_view or rec.vehicle_id not in h.vanish_near:
                continue
            if now - h.vanish_ts > window or pid in self.driver_of:
                continue
            cands.append(h)
        if rec.is_currently_moving:
            if rec.cand_pid is not None:
                return  # a visible person in the cab is being confirmed right now
            # moving and nobody known aboard: whoever vanished at this vehicle shortly before is driving it
            if len(cands) == 1:
                self._board(rec, cands[0].person_id, cands[0].vanish_ts, events, "VANISHED_INTO_CAB")
            elif len(cands) > 1:
                cands.sort(key=lambda h: (h.vanish_near[rec.vehicle_id], -h.vanish_ts))
                gap = cands[1].vanish_near[rec.vehicle_id] - cands[0].vanish_near[rec.vehicle_id]
                if gap >= 0.8:
                    self._board(rec, cands[0].person_id, cands[0].vanish_ts, events, "VANISHED_INTO_CAB")
                else:
                    self._set_unknown_occupied(rec, now, events, candidates=[h.person_id for h in cands])
            else:
                self._set_unknown_occupied(rec, now, events, reason="moving, driver hidden")
            return
        inside = [h for h in cands if h.vanish_cab.get(rec.vehicle_id, 0.0) >= self.cab_thresh
                  and now - h.vanish_ts >= self.hidden_board_confirm_sec]
        if len(inside) == 1:
            self._board(rec, inside[0].person_id, inside[0].vanish_ts, events, "VANISHED_INTO_CAB")

    def _emergence(self, visible: List[VehicleSessionRecord], person_map: Dict[str, Dict], now: float,
                   W: int, H: int, events: List[Dict]) -> None:
        """A NEW person appearing right next to a parked vehicle whose occupant was hidden = the driver climbing out."""
        m = self.spawn_border_margin_px
        for pid, p in person_map.items():
            h = self.person_histories[pid]
            if now - h.first_seen_ts > self.emergence_window_sec or pid in self.driver_of:
                continue
            if pid.startswith("T") and not self.allow_provisional_spawn:
                continue
            sx, sy = h.spawn_px
            if sx <= m or sy <= m or sx >= W - m or sy >= H - m:
                continue  # walked in from outside the view
            if h.max_distance_from_spawn_m() >= 1.1:
                continue
            best, best_d = None, float("inf")
            for rec in visible:
                if rec.is_currently_moving or h.last_cab.get(rec.vehicle_id, 0.0) >= self.cab_thresh:
                    continue  # moving vehicles do not shed drivers; a person IN the cab is handled by boarding
                hidden_driver = rec.associated_driver_id is not None and rec.onboard and rec.driver_hidden_onboard
                if hidden_driver and self._clearly_different(p.get("embedding"), rec.driver_reid_embedding):
                    continue
                recently_active = now - rec.last_moved_ts <= self.recent_activity_sec
                eligible = hidden_driver or rec.unknown_occupied or (
                    rec.associated_driver_id is None
                    and (recently_active or not self.require_vehicle_activity_for_spawn))
                if not eligible:
                    continue
                d = self._dist_to_vehicle(rec, h.spawn_world_xy)
                if d <= self.dismount_spawn_radius_m and d < best_d:
                    best, best_d = rec, d
            if best is None:
                continue
            if best.associated_driver_id is not None:
                old = best.associated_driver_id
                self.driver_of.pop(old, None)
                best.associated_driver_id = pid
                self.driver_of[pid] = best.vehicle_id
                events.append(self._event("DRIVER_REBOUND", now, best, pid, INFO, previous_driver=old,
                                          how="EMERGED_FROM_CAB"))
            else:
                self._board(best, pid, h.first_seen_ts, events, "EMERGED_FROM_CAB", p)
            best.driver_visible = True
            best.driver_last_seen_ts = now
            best.driver_bbox = list(p["bbox"])
            best.driver_last_world_xy = h.current_world_xy
            self._dismount(best, h.first_seen_ts, events, "EMERGED_FROM_CAB")


def _box_inside(inner: List[float], outer: List[float]) -> bool:
    return inner[0] >= outer[0] and inner[1] >= outer[1] and inner[2] <= outer[2] and inner[3] <= outer[3]
