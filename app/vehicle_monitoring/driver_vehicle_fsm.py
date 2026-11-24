# app/vehicle_monitoring/driver_vehicle_fsm.py
"""
Forklift <-> driver monitoring.

Everything is time based (seconds on the stream clock), so behaviour does not change with the processing FPS.

WHO IS IN THE CAB (cab_score)
  Overlapping boxes are not enough: seen from an oblique camera the box of a forklift also covers people who stand
  metres behind or in front of it. The FLOOR decides:
    feet visible, standing pose              -> he stands on the floor: not an operator
    feet hidden + seated pose                -> proof (thighs level with the hips)
    feet hidden + body inside the footprint  -> proof (hips + a standing leg length, projected on the floor, against
                                                the vehicle's floor footprint = a rectangle of the real vehicle size
                                                fitted to the lower outline of its mask)
    upper body hidden, hips + knees visible  -> seated under the overhead guard (steep camera): proof
    feet hidden + body beyond the footprint  -> standing behind the vehicle: no
    feet hidden, no proof                    -> candidate only

WHO DRIVES (driver binding)
  1. Proven in the cab    : proof adds up for board_confirm_sec (seated pose: seated_board_confirm_sec) on a parked
                            vehicle. A candidate WITHOUT proof is never bound to a parked vehicle
                            (ambiguous_board_confirm_sec = 0), but:
  2. Travels with it      : on a moving vehicle, whoever stays inside its footprint while it travels (and is not
                            standing still) for moving_board_confirm_sec is the driver - back-dated to the moment he
                            became the candidate. Turning on the spot is not travelling: a pivoting forklift does not
                            lose its driver. One person -> one vehicle.
  3. Vanished into the cab: ROOFED / CLOSED cabs often hide the driver completely. A person who disappears (really:
                            vanish_min_sec, not one missed frame) while standing AT the cab in front of / beside the
                            vehicle - where nothing else can hide him - is inside after hidden_board_confirm_sec. A
                            person who disappears BEHIND it (or just near it) is the driver only if the vehicle then
                            starts to move and he is the only candidate.
  4. Emerged from the cab : a NEW person appearing right next to a parked vehicle whose driver was hidden (or whose
                            occupant was unknown) is the driver climbing out.
  5. Moving, nobody known : OCCUPIED_UNKNOWN_DRIVER (someone must be driving - hidden by the roof).
  A different person proven in the cab (or vanished into it) while the bound driver is out -> driver change.
  Driver boarding another vehicle -> his old session closes.
  Measured against simulated ground truth: scripts/benchmark_driver.py (app/vehicle_monitoring/simulator.py).

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
from typing import Any, Deque, Dict, Iterable, List, NamedTuple, Optional, Set, Tuple

import numpy as np

from app.core.events import ALARM, INFO, WARNING, make_event
from app.vehicle_monitoring.distance_estimator import (
    DistanceMeasurement,
    RealWorldDistanceEstimator,
    point_polygon_distance,
    signed_gap,
)


class CabEvidence(NamedTuple):
    """Is this person IN the cab of this vehicle? (indexable like the old (score, seated) pair)"""
    score: float            # 0..1 after the floor checks; >= cab_score_threshold = "in the cab"
    seated: bool            # clearly seated pose (folded legs, or thighs level with the hips)
    raw: float = 0.0        # image evidence alone (before the floor checks)
    on_floor: bool = False  # feet visibly on the floor in a standing pose: NOT an operator
    strong: bool = False    # positive proof of being inside: seated pose or body well inside the footprint
    gap: Optional[float] = None   # metres from the body's floor point to the footprint (negative = inside)


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
    # vehicle -> he can only have gone INTO it: last seen with proof of being in the cab, or standing at it on
    # the camera's side (nothing else there to hide behind). Somebody vanishing on the FAR side may stand behind it.
    vanish_into: Dict[str, bool] = field(default_factory=dict)
    last_cab: Dict[str, float] = field(default_factory=dict)
    last_strong: Dict[str, bool] = field(default_factory=dict)
    last_ground_certain: bool = False
    # last time he was seen STANDING ON THE FLOOR (feet visible, standing pose): where he really was. The frames
    # while he climbs in / is half hidden give floor points that are off by a metre.
    floor_history: Deque[Tuple[float, float, float]] = field(default_factory=lambda: deque(maxlen=40))
    # vehicle -> recent (time, metres from his floor point to that vehicle's footprint; negative = inside)
    gap_history: Dict[str, Deque[Tuple[float, float]]] = field(default_factory=dict)
    has_pose: bool = False             # the last detection carried keypoints

    def floor_before(self, ts: float, lead_sec: float, memory_sec: float) -> Optional[Tuple[float, float]]:
        """Where he stood on the floor shortly BEFORE `ts` (at least lead_sec earlier: not the climb itself)."""
        for x, y, t in reversed(self.floor_history):
            if t <= ts - lead_sec:
                return (x, y) if ts - t <= memory_sec else None
        return None

    def update_position(self, world_xy: Tuple[float, float], ts: float,
                        ground_px: Optional[Tuple[float, float]] = None) -> None:
        self.current_world_xy = world_xy
        self.world_history.append((float(world_xy[0]), float(world_xy[1]), ts))
        if ground_px is not None:
            self.px_history.append((float(ground_px[0]), float(ground_px[1]), ts))
        self.total_frames_seen += 1
        self.last_seen_ts = ts

    def get_speed_mps(self, window_sec: float = 1.2) -> float:
        """
        Walking speed on the floor, robust against jitter: the floor point of somebody whose legs are hidden (a
        seated driver!) jumps by decimetres from frame to frame, which a plain first-to-last difference reads as
        "walking". Here the median position of the first half of the window is compared with that of the second.
        """
        if len(self.world_history) < 4:
            return 0.0
        t_end = self.world_history[-1][2]
        pts = [h for h in self.world_history if t_end - h[2] <= window_sec]
        if len(pts) < 4 or pts[-1][2] - pts[0][2] < 0.3:
            return 0.0
        half = len(pts) // 2
        a, b = np.asarray(pts[:half]), np.asarray(pts[half:])
        dt = float(np.median(b[:, 2]) - np.median(a[:, 2]))
        if dt < 0.1:
            return 0.0
        return float(np.hypot(np.median(b[:, 0]) - np.median(a[:, 0]), np.median(b[:, 1]) - np.median(a[:, 1])) / dt)

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
    exit_vehicle_xy: Optional[Tuple[float, float]] = None      # where the vehicle stood when the driver got out
    lost_since: Optional[float] = None
    # session / seat time
    session_start_ts: Optional[float] = None
    stint_start_ts: Optional[float] = None
    onboard_closed_sec: float = 0.0
    unknown_occupied: bool = False
    unknown_since: Optional[float] = None
    # boarding candidate
    cand_pid: Optional[str] = None
    cand_time: float = 0.0             # how long he has been the candidate (session start is back-dated by it)
    cand_proof: float = 0.0            # seconds of STRONG in-cab evidence (leaks away on frames without it)
    cand_move: float = 0.0             # seconds travelling with the moving vehicle
    cand_last_ts: float = -1e18
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
        # floor evidence (see cab_score)
        self.operator_stands = bool(p.get("operator_stands", False))          # stand-up trucks: no seated pose
        self.cab_outside_gap_m = float(p.get("cab_outside_gap_m", 0.45))      # body this far beyond the footprint
        self.cab_inside_gap_m = float(p.get("cab_inside_gap_m", 0.10))        # body this far inside = strong proof
        self.seated_knee_drop = float(p.get("seated_knee_drop", 0.45))        # knee below hip / torso length
        self.ambiguous_board_confirm_sec = float(p.get("ambiguous_board_confirm_sec", 0.0))   # 0 = never (parked)

        self.vehicle_motion_thresh_m = float(p.get("vehicle_motion_threshold_m", 0.40))
        self.motion_window_sec = float(p.get("motion_window_sec", 1.0))
        self.stop_confirm_sec = float(p.get("stop_confirm_sec", 1.5))
        self.comotion_min_travel_m = float(p.get("comotion_min_travel_m", 0.6))   # vehicle travel needed to judge
        self.comotion_inside_gap_m = float(p.get("comotion_inside_gap_m", 0.15))   # still inside the footprint
        self.comotion_outside_gap_m = float(p.get("comotion_outside_gap_m", 1.0))  # left behind
        self.moved_without_driver_m = float(p.get("moved_without_driver_m", 1.0))  # travel since the driver got out

        # roofed cabs / hidden drivers
        self.board_vanish_radius_m = float(p.get("board_vanish_radius_m", 1.2))
        self.vanish_board_window_sec = float(p.get("vanish_board_window_sec", 30.0))
        self.hidden_board_confirm_sec = float(p.get("hidden_board_confirm_sec", 2.0))
        self.vanish_min_sec = float(p.get("vanish_min_sec", 0.8))   # unseen this long = vanished (not a missed frame)
        self.vanish_floor_memory_sec = float(p.get("vanish_floor_memory_sec", 3.0))
        self.board_side_radius_m = float(p.get("board_side_radius_m", 0.75))
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

    def cab_score(self, person: Dict[str, Any], vbbox: List[float],
                  footprint: Optional[np.ndarray] = None) -> CabEvidence:
        """
        Evidence that `person` is IN the cab of the vehicle.

        Image overlap alone is not enough: from an oblique camera the vehicle's box also covers people who stand
        metres BEHIND it or in front of it. So the floor decides:
          * feet visible in a standing pose  -> he stands on the floor (in front of / behind / beside the
            vehicle): not an operator, however much the boxes overlap;
          * feet hidden -> seated in the cab, or standing right behind the vehicle. Told apart by
              - the pose: a seated person's thighs are level with the hips (knees visible near hip height); a
                standing person's knees hang a thigh-length below the hips;
              - the floor point of the body (hips + standing leg length, projected on the floor) against the
                vehicle's footprint: an operator is inside it, somebody behind the vehicle is beyond its far edge.
        `strong` = positive proof (seated pose, or well inside the footprint). Without it a person at a PARKED
        vehicle is not made its driver (see _visible_boarding); travelling with the moving vehicle still binds him.
        """
        pb = person["bbox"]
        cx1, cy1, cx2, cy2 = self._cab_rect(vbbox)
        ix1, iy1, ix2, iy2 = max(pb[0], cx1), max(pb[1], cy1), min(pb[2], cx2), min(pb[3], cy2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area = max(1.0, (pb[2] - pb[0]) * (pb[3] - pb[1]))
        containment = inter / area
        if containment < 0.30:
            return CabEvidence(0.0, False)
        vh = vbbox[3] - vbbox[1]
        kp = person.get("keypoints")
        k = np.asarray(kp) if kp is not None else None
        if k is None or k.ndim != 2 or len(k) < 17:
            # box-only detections (standalone runner): the legacy image rule
            foot = person.get("foot_px")
            if foot is not None and not person.get("ground_uncertain", False) and foot[1] > vbbox[3] - 0.03 * vh:
                return CabEvidence(0.15 * containment, False, float(containment), True)
            return CabEvidence(float(containment), False, float(containment), False, True)

        conf = k[:, 2] if k.shape[1] >= 3 else np.ones(len(k))
        vis = (conf >= self.min_kpt_conf) & (k[:, 0] > 1) & (k[:, 1] > 1)
        inside = vis & (k[:, 0] >= cx1) & (k[:, 0] <= cx2) & (k[:, 1] >= cy1) & (k[:, 1] <= cy2)
        n_vis = int(vis.sum())
        inside_frac = float(inside.sum()) / n_vis if n_vis >= 3 else containment
        ankles = [i for i in (15, 16) if vis[i]]
        knees = [i for i in (13, 14) if vis[i]]
        sh = [i for i in (5, 6) if vis[i]]
        hp = [i for i in (11, 12) if vis[i]]
        seated = legs_hidden = under_roof = False
        gap: Optional[float] = None
        if hp and knees and not sh and not ankles:
            # Hips and knees are seen but shoulders and head are not: the upper body is hidden while the lower body
            # is visible. Nothing does that to a person standing on the floor - it is the overhead guard / roof of
            # the cab seen from a steep camera. He sits under it.
            head_seen = bool(vis[:5].any())
            hip_mid, knee_mid = k[hp, :2].mean(axis=0), k[knees, :2].mean(axis=0)
            thigh_px = max(1.0, float(np.hypot(*(knee_mid - hip_mid))))
            legs_hidden = True
            # thighs roughly level (not hanging down like a standing person's) = seated
            level = (float(knee_mid[1] - hip_mid[1]) / thigh_px) < 0.6
            # both hips and both knees really seen (one stray joint of a half-hidden bystander is not enough)
            under_roof = (not head_seen and len(hp) == 2 and len(knees) == 2 and bool(inside[hp].all()) and level)
            seated = under_roof
        if sh and hp:
            sh_mid, hip_mid = k[sh, :2].mean(axis=0), k[hp, :2].mean(axis=0)
            torso_px = max(1.0, float(np.hypot(*(hip_mid - sh_mid))))
            torso_h = max(1.0, float(hip_mid[1] - sh_mid[1]))
            if ankles:
                leg_h = max(0.0, float(np.mean([k[i][1] for i in ankles])) - float(hip_mid[1]))
                seated = (leg_h / torso_h) < 1.25
            else:
                legs_hidden = True
                if knees:   # thighs level with the hips = sitting (a standing person's knees hang ~0.85 torso lower)
                    drop = (float(np.mean([k[i][1] for i in knees])) - float(hip_mid[1])) / torso_px
                    seated = drop < self.seated_knee_drop
            if footprint is not None and self.estimator.is_metric:
                # where this body would stand: hips + a standing leg length (1.86 x torso), on the floor
                body_floor = self.estimator.pixel_to_world((float(hip_mid[0]), float(hip_mid[1]) + 1.86 * torso_px))
                gap = signed_gap(body_floor, footprint)
        raw = float(min(1.0, 0.5 * containment + 0.5 * inside_frac + (0.15 if seated else
                                                                       (0.05 if legs_hidden else 0.0))))
        on_floor = bool(ankles) and not seated and not self.operator_stands
        if on_floor:
            # feet on the floor in a standing pose: beside / in front of / behind the vehicle
            return CabEvidence(0.15 * containment, False, raw, True, False, gap)
        if not seated and gap is not None and gap > self.cab_outside_gap_m:
            # the body is beyond the footprint: standing behind the vehicle (legs hidden by the chassis)
            return CabEvidence(min(raw, 0.8 * self.cab_thresh), False, raw, False, False, gap)
        strong = (seated or under_roof or (gap is not None and gap <= -self.cab_inside_gap_m)
                  or self.operator_stands)
        if gap is None and not seated and not under_roof:
            strong = not self.estimator.is_metric     # no calibration / no torso: nothing better than the image
        return CabEvidence(raw, seated, raw, False, bool(strong), gap)

    def _footprint(self, rec: VehicleSessionRecord) -> np.ndarray:
        if rec.footprint_world is None:
            rec.footprint_world, _ = self.estimator.vehicle_footprint(
                rec.bbox, self._frame_shape, rec.contour, key=f"veh_{rec.vehicle_id}_fp")
        return rec.footprint_world

    @staticmethod
    def _vehicle_xy(rec: VehicleSessionRecord) -> Optional[Tuple[float, float]]:
        if not rec.world_pos_history:
            return None
        return rec.world_pos_history[-1][0], rec.world_pos_history[-1][1]

    def _moved_since_exit(self, rec: VehicleSessionRecord) -> bool:
        """Has the vehicle really TRAVELLED since its driver got out? (not: is it still rolling to a stop /
        being nudged - the 'moving' flag lingers for a moment after every stop)"""
        now_xy = self._vehicle_xy(rec)
        if rec.exit_vehicle_xy is None or now_xy is None:
            return False
        return float(np.hypot(now_xy[0] - rec.exit_vehicle_xy[0], now_xy[1] - rec.exit_vehicle_xy[1])) \
            >= self.moved_without_driver_m

    def _torso_floor_point(self, person: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        """Floor position (metres) under a body whose feet are not visible: hips + 1.86 x torso length downwards."""
        if not self.estimator.is_metric:
            return None
        kp = person.get("keypoints")
        k = np.asarray(kp) if kp is not None else None
        if k is None or k.ndim != 2 or len(k) < 17:
            return None
        conf = k[:, 2] if k.shape[1] >= 3 else np.ones(len(k))
        sh = [i for i in (5, 6) if conf[i] >= self.min_kpt_conf]
        hp = [i for i in (11, 12) if conf[i] >= self.min_kpt_conf]
        if not (sh and hp):
            return None
        sh_mid, hip_mid = k[sh, :2].mean(axis=0), k[hp, :2].mean(axis=0)
        torso_px = max(1.0, float(np.hypot(hip_mid[0] - sh_mid[0], hip_mid[1] - sh_mid[1])))
        return self.estimator.pixel_to_world((float(hip_mid[0]), float(hip_mid[1]) + 1.86 * torso_px))

    def _stands_on_floor(self, person: Dict[str, Any]) -> bool:
        """Feet visible and the proportions of a standing body (not climbing / sitting): his floor point is real."""
        kp = person.get("keypoints")
        k = np.asarray(kp) if kp is not None else None
        if k is None or k.ndim != 2 or len(k) < 17:
            return person.get("ground_uncertain") is False
        conf = k[:, 2] if k.shape[1] >= 3 else np.ones(len(k))
        vis = conf >= self.min_kpt_conf
        ankles = [i for i in (15, 16) if vis[i]]
        sh = [i for i in (5, 6) if vis[i]]
        hp = [i for i in (11, 12) if vis[i]]
        if not (ankles and sh and hp):
            return False
        torso_h = max(1.0, float(k[hp, 1].mean() - k[sh, 1].mean()))
        return (float(k[ankles, 1].mean()) - float(k[hp, 1].mean())) / torso_h >= 1.45

    def _behind_vehicle(self, rec: VehicleSessionRecord, world_xy: Tuple[float, float],
                        margin: float = 0.15) -> bool:
        """
        Is this floor point in the "shadow" of the vehicle as seen from the camera (the line of sight to it passes
        over the footprint)? Only there a person can be hidden BEHIND the vehicle. Somebody who vanishes while
        standing in front of it or beside it has nowhere else to go than into the cab.
        """
        fp = self._footprint(rec)
        away = self.estimator._away_direction(rec.bbox)
        perp = np.array([-away[1], away[0]])
        pt = np.asarray(world_xy, dtype=np.float64)
        if signed_gap(pt, fp) <= 0:
            return True
        n = len(fp)
        for off in (-margin, 0.0, margin):
            o = pt + off * perp
            for i in range(n):                       # ray from the point TOWARDS the camera against every edge
                a, b = fp[i], fp[(i + 1) % n]
                e = b - a
                den = (-away[0]) * e[1] - (-away[1]) * e[0]
                if abs(den) < 1e-9:
                    continue
                w = a - o
                t = (w[0] * e[1] - w[1] * e[0]) / den
                u = (w[0] * (-away[1]) - w[1] * (-away[0])) / den
                if t >= 0.0 and 0.0 <= u <= 1.0:
                    return True
        return False

    def _dist_to_vehicle(self, rec: VehicleSessionRecord, world_xy: Tuple[float, float]) -> float:
        return point_polygon_distance(world_xy, self._footprint(rec))[0]

    def _update_vehicle_motion(self, rec: VehicleSessionRecord, now: float) -> None:
        x1, y1, x2, y2 = rec.bbox
        base = (0.5 * (x1 + x2), float(y2))
        # Position = centre of the floor footprint. (The bottom centre of the BOX moves when the vehicle merely
        # turns on the spot: the old code then saw "motion" the seated driver did not share and unbound him.)
        centre = self._footprint(rec).mean(axis=0)
        wx, wy = self.estimator._stabilize(f"veh_{rec.vehicle_id}_c", (float(centre[0]), float(centre[1])))
        rec.world_pos_history.append((wx, wy, now))
        rec.px_history.append((base[0], base[1], now))
        # displacement over motion_window_sec between the MEDIAN position now and then: a fitted footprint jitters
        # by a decimetre or two, and a parked vehicle must not look as if it had started to drive
        span = 0.6
        recent = [h for h in rec.world_pos_history if now - h[2] <= span]
        past = [h for h in rec.world_pos_history
                if self.motion_window_sec <= now - h[2] <= self.motion_window_sec + span]
        if len(recent) < 2 or len(past) < 2:
            return
        r, q = np.asarray(recent), np.asarray(past)
        disp = float(np.hypot(np.median(r[:, 0]) - np.median(q[:, 0]), np.median(r[:, 1]) - np.median(q[:, 1])))
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

    def _co_motion(self, rec: VehicleSessionRecord, hist: PersonTrackHistory, window: float = 1.5) -> Optional[bool]:
        """
        Does the person travel WITH the vehicle?  True / False / None (cannot judge: the vehicle hardly travelled,
        or too little data).

        With a skeleton: riding = while the vehicle travels, his floor point STAYS INSIDE its footprint and he is
        not standing still; left behind = the footprint moved away from him, or the vehicle travelled far while he
        stayed where he was. (Comparing the two displacements alone does not work: a forklift pivots, its centre
        and its seat then move along different arcs, and a seated driver looked "left behind" on every turn.)
        Without keypoints (box-only detections of the standalone runner) the floor point of a seated person is not
        known, and only the displacements are compared, as before.
        """
        vh = rec.world_pos_history
        if len(vh) < 3:
            return None
        t_end = vh[-1][2]
        past = [x for x in vh if t_end - window <= x[2]]
        if len(past) < 3:
            return None
        v = np.array([past[-1][0] - past[0][0], past[-1][1] - past[0][1]], dtype=np.float64)
        v_len = float(np.hypot(v[0], v[1]))
        if v_len < self.comotion_min_travel_m:
            return None
        if not hist.has_pose:
            pts = [x for x in hist.world_history if past[0][2] - 1e-6 <= x[2] <= t_end + 1e-6]
            if len(pts) < 3:
                return None
            residual = float(np.hypot(pts[-1][0] - pts[0][0] - v[0], pts[-1][1] - pts[0][1] - v[1]))
            if residual <= max(0.45 * v_len, 0.5):
                return True
            return False if residual >= max(0.70 * v_len, 0.8) else None
        gaps = [g for t, g in hist.gap_history.get(rec.vehicle_id, ()) if t_end - t <= 0.8]
        pts = np.asarray([x for x in hist.world_history if past[0][2] - 1e-6 <= x[2] <= t_end + 1e-6])
        if len(gaps) < 3 or len(pts) < 6:
            return None
        gap = float(np.median(gaps))        # medians: single frames with a wrong floor point do not decide
        k = max(2, len(pts) // 3)
        moved = float(np.hypot(np.median(pts[-k:, 0]) - np.median(pts[:k, 0]),
                               np.median(pts[-k:, 1]) - np.median(pts[:k, 1])))
        if gap >= self.comotion_outside_gap_m or (v_len >= 1.2 and moved <= 0.25 * v_len):
            return False                    # left behind / stood still while the vehicle drove on
        if gap <= self.comotion_inside_gap_m and moved >= 0.3 * v_len:
            return True
        return None

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
        rec.exit_vehicle_xy = self._vehicle_xy(rec)
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
            if p.get("ground_uncertain"):
                # Legs hidden (seated in a cab, behind a pallet ...): the bottom of the visible box is not on the
                # floor and projects metres away. The torso gives the place (hips + a standing leg length).
                torso = self._torso_floor_point(p)
                if torso is not None:
                    world = torso
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
            hist.vanish_into.clear()
            hist.last_ground_certain = p.get("ground_uncertain") is False
            hist.has_pose = p.get("keypoints") is not None
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
        scores: Dict[Tuple[str, str], CabEvidence] = {}
        for rec in visible:
            for pid, p in person_map.items():
                if self._bbox_iou(p["bbox"], rec.bbox) <= 0.0 and not _box_inside(p["bbox"], rec.bbox):
                    continue
                fp = self._footprint(rec)
                s = self.cab_score(p, rec.bbox, fp)
                if s[0] > 0 or s.raw > 0:
                    scores[(rec.vehicle_id, pid)] = s
                h = self.person_histories[pid]
                h.gap_history.setdefault(rec.vehicle_id, deque(maxlen=30)).append(
                    (now, signed_gap(h.current_world_xy, fp)))
        for pid in person_map:
            self.person_histories[pid].last_cab = {vid: s[0] for (vid, q), s in scores.items() if q == pid}
            h = self.person_histories[pid]
            h.last_strong = {
                vid: bool(s.strong and s[0] >= self.cab_thresh) for (vid, q), s in scores.items() if q == pid}
            if h.last_ground_certain and self._stands_on_floor(person_map[pid]):
                h.floor_history.append((h.current_world_xy[0], h.current_world_xy[1], now))

        # ---- 4. who just vanished, and next to which vehicle? -------------------------------------------------
        for pid, h in self.person_histories.items():
            if pid in person_map or h.vanish_ts is not None:
                continue
            h.vanish_ts = h.last_seen_ts
            # where did he stand just before? (the last frames of a climb give floor points that are a metre off)
            stood = h.floor_before(h.vanish_ts, 0.8, self.vanish_floor_memory_sec)
            for rec in visible:
                dist = self._dist_to_vehicle(rec, h.current_world_xy)
                dist_stood = self._dist_to_vehicle(rec, stood) if stood is not None else float("inf")
                cab = h.last_cab.get(rec.vehicle_id, 0.0)
                if min(dist, dist_stood) <= self.board_vanish_radius_m or cab >= 0.3:
                    h.vanish_near[rec.vehicle_id] = min(dist, dist_stood)
                    h.vanish_cab[rec.vehicle_id] = cab
                    # ... right AT the vehicle (a boarding driver stands within arm's reach of the cab)
                    in_sight = (dist_stood <= self.board_side_radius_m
                                and not self._behind_vehicle(rec, stood))
                    if in_sight and h.floor_history:
                        # ... unless his LAST step on the floor was already behind the vehicle and clear of it:
                        # he walked round it (a climbing driver's last floor point stays at / inside the footprint)
                        last = (h.floor_history[-1][0], h.floor_history[-1][1])
                        if signed_gap(last, self._footprint(rec)) > 0.3 and self._behind_vehicle(rec, last):
                            in_sight = False
                    h.vanish_into[rec.vehicle_id] = bool(h.last_strong.get(rec.vehicle_id, False) or in_sight)

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
            ev = scores.get((rec.vehicle_id, pid), CabEvidence(0.0, False))
            score, seated = ev.score, ev.seated
            in_cab = score >= self.cab_thresh
            if rec.onboard and not in_cab and ev.raw >= self.cab_thresh and not ev.on_floor:
                # An ONBOARD driver is only taken out of the cab by clear evidence (feet on the floor, or the body
                # well beyond the footprint): one noisy floor estimate must not "dismount" a seated operator.
                in_cab = ev.gap is None or ev.gap <= self.cab_outside_gap_m + 0.30
            if rec.is_currently_moving and not rec.onboard and ev.raw >= self.cab_thresh and not ev.on_floor:
                in_cab = True      # travelling in the cab region of the moving vehicle: co-motion decides below
            rec.driver_last_seen_in_cab = in_cab
            if in_cab and not rec.onboard:
                # The DISMOUNTED driver overlaps the cab again: climbing back in - or only walking past the
                # forklift in the image. Re-boarding needs the same confirmation as boarding; until then he is
                # still monitored as dismounted (an instant re-board used to mask the distance alarm).
                if rec.is_currently_moving:
                    ok, required = self._co_motion(rec, hist) is True, self.moving_board_confirm_sec
                else:
                    ok = seated or hist.get_speed_mps() <= self.max_boarding_walk_speed_mps
                    required = self.seated_board_confirm_sec if seated else self.board_confirm_sec
                    if not ev.strong:                # legs hidden, no proof of being inside (may stand behind it)
                        ok = ok and self.ambiguous_board_confirm_sec > 0
                        required = self.ambiguous_board_confirm_sec
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
            if rec.is_currently_moving and self._moved_since_exit(rec):
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
        if rec.is_currently_moving and self._moved_since_exit(rec):
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
                and hist.vanish_into.get(rec.vehicle_id, False)
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
            for (vid, pid), ev in scores.items():
                if vid != rec.vehicle_id or pid == rec.associated_driver_id:
                    continue
                # parked: the floor-checked score; moving: the image score (travelling with it decides), but
                # never somebody whose feet are on the floor
                score = ev.raw if (rec.is_currently_moving and not ev.on_floor) else ev.score
                if score < self.cab_thresh:
                    continue
                other = self.driver_of.get(pid)
                if other is not None and other in self.vehicles and self.vehicles[other].onboard:
                    continue  # onboard another vehicle
                pairs.append((score + (0.5 if ev.strong else 0.0), rec.vehicle_id, pid, ev))
        pairs.sort(key=lambda t: t[0], reverse=True)
        chosen: Dict[str, Tuple[str, CabEvidence]] = {}
        taken: Set[str] = set()
        for _, vid, pid, ev in pairs:
            if vid not in chosen and pid not in taken:
                chosen[vid] = (pid, ev)
                taken.add(pid)

        def reset(rec: VehicleSessionRecord, pid: Optional[str] = None) -> None:
            rec.cand_pid, rec.cand_time, rec.cand_proof, rec.cand_move = pid, 0.0, 0.0, 0.0

        for rec in visible:
            pick = chosen.get(rec.vehicle_id)
            if pick is None:
                if rec.cand_pid is not None and now - rec.cand_last_ts > 0.6:   # not just one missed detection
                    reset(rec)
                continue
            pid, ev = pick
            if pid != rec.cand_pid:
                reset(rec, pid)
            rec.cand_last_ts = now
            hist = self.person_histories[pid]
            rec.cand_time += dt
            if rec.is_currently_moving:
                cm = self._co_motion(rec, hist)
                if cm is False:
                    reset(rec, pid)                  # the vehicle went somewhere he did not
                    continue
                if cm is True:
                    rec.cand_move += dt
                confirmed = rec.cand_move + 1e-9 >= self.moving_board_confirm_sec
            else:
                rec.cand_move = 0.0
                if not (ev.seated or hist.get_speed_mps() <= self.max_boarding_walk_speed_mps):
                    reset(rec, pid)                  # walking past
                    continue
                # Proof of being INSIDE (seated pose / body within the footprint) must add up; single noisy frames
                # in either direction do not decide. Somebody with hidden legs and no such proof may be standing
                # behind the parked vehicle: he stays a candidate (and is bound, back-dated, the moment he travels
                # with the vehicle), but a parked vehicle does not get him as its driver.
                rec.cand_proof = rec.cand_proof + dt if ev.strong else max(0.0, rec.cand_proof - dt)
                required = self.seated_board_confirm_sec if ev.seated else self.board_confirm_sec
                confirmed = ev.strong and rec.cand_proof + 1e-9 >= required
                if not confirmed and self.ambiguous_board_confirm_sec > 0:
                    confirmed = rec.cand_time + 1e-9 >= self.ambiguous_board_confirm_sec
            if not confirmed:
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
                    reset(rec)
                    continue
            self._board(rec, pid, start, events, "VISIBLE_IN_CAB", person_map[pid])
            reset(rec)

    def _hidden_inference(self, rec: VehicleSessionRecord, person_map: Dict[str, Dict], now: float,
                          events: List[Dict]) -> None:
        """Roofed cabs: the driver vanished INTO the cab (seen last at/inside it) -> hidden onboard."""
        # also when the bound driver is OUT of the cab: somebody else who disappears into it is taking over
        bound_out = rec.associated_driver_id is not None and not rec.onboard
        if rec.associated_driver_id is not None and not bound_out:
            return
        window = self.vanish_board_window_sec
        cands = []
        for pid, h in self.person_histories.items():
            if pid in person_map or h.vanish_ts is None or h.left_view or rec.vehicle_id not in h.vanish_near:
                continue
            if now - h.vanish_ts > window or pid in self.driver_of:
                continue
            if now - h.vanish_ts < self.vanish_min_sec:
                continue        # one missed detection is not "disappeared into the cab"
            if bound_out and (rec.exit_timestamp is None or h.vanish_ts <= rec.exit_timestamp):
                continue        # he vanished before the current driver got out: not somebody taking over now
            cands.append(h)
        if rec.is_currently_moving:
            if bound_out or rec.cand_pid is not None:
                return  # (moving without its driver is handled there) / a visible person is being confirmed
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
        inside = [h for h in cands if h.vanish_into.get(rec.vehicle_id, False)
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
                old_hist = self.person_histories.get(old)
                if old_hist is not None:           # the same person under his previous id: no longer "hidden in the cab"
                    old_hist.vanish_near.pop(best.vehicle_id, None)
                    old_hist.vanish_into.pop(best.vehicle_id, None)
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
