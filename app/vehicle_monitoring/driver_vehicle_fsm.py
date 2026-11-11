# app/vehicle_monitoring/driver_vehicle_fsm.py
import inspect
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional, Set, Tuple
import numpy as np

from app.vehicle_monitoring.distance_estimator import (
    DistanceMeasurement,
    RealWorldDistanceEstimator,
)


class VehicleSafetyState(str, Enum):
    UNASSIGNED = "UNASSIGNED"
    DRIVER_ONBOARD = "DRIVER_ONBOARD"
    DRIVER_EXITED_SAFE = "DRIVER_EXITED_SAFE"
    DRIVER_EXITED_WARNING = "DRIVER_EXITED_WARNING"
    DISTANCE_VIOLATION_ALARM = "DISTANCE_VIOLATION_ALARM"
    DISTANCE_UNCERTAIN = "DISTANCE_UNCERTAIN"


@dataclass
class PersonTrackHistory:
    person_id: str
    first_seen_ts: float
    spawn_world_xy: Tuple[float, float]
    current_world_xy: Tuple[float, float]
    world_history: Deque[Tuple[float, float, float]] = field(
        default_factory=lambda: deque(maxlen=20)
    )
    total_frames_seen: int = 0

    def update_position(self, world_xy: Tuple[float, float], ts: float) -> None:
        self.current_world_xy = world_xy
        self.world_history.append((world_xy[0], world_xy[1], ts))
        self.total_frames_seen += 1

    def get_speed_mps(self) -> float:
        """Calculates recent walking speed in meters/second."""
        if len(self.world_history) < 4:
            return 0.0
        x0, y0, t0 = self.world_history[0]
        x1, y1, t1 = self.world_history[-1]
        dt = max(1e-3, t1 - t0)
        return float(np.hypot(x1 - x0, y1 - y0) / dt)

    def max_distance_from_spawn_m(self) -> float:
        return float(
            np.hypot(
                self.current_world_xy[0] - self.spawn_world_xy[0],
                self.current_world_xy[1] - self.spawn_world_xy[1],
            )
        )


@dataclass
class VehicleSessionRecord:
    vehicle_id: str
    bbox: List[float]
    associated_driver_id: Optional[str] = None
    driver_reid_embedding: Optional[np.ndarray] = None
    driver_bbox: Optional[List[float]] = None
    state: VehicleSafetyState = VehicleSafetyState.UNASSIGNED
    overlap_frames: Dict[str, int] = field(default_factory=dict)
    frames_outside_cab: int = 0
    last_measurement: Optional[DistanceMeasurement] = None
    last_seen_ts: float = field(default_factory=time.time)
    is_visible_in_current_frame: bool = True
    exit_timestamp: Optional[float] = None
    alarm_active: bool = False
    # Motion tracking to distinguish parked cars from active forklifts/vehicles
    world_pos_history: Deque[Tuple[float, float, float]] = field(
        default_factory=lambda: deque(maxlen=25)
    )
    is_currently_moving: bool = False
    has_ever_moved: bool = False
    last_moved_ts: float = 0.0


class DriverVehicleMonitorFSM:
    """
    Manages Forklift <-> Driver pairing using YOLO-Pose keypoints, vehicle motion verification,
    pass-by pedestrian rejection, cabin dismount emergence detection, 1-to-1 driver exclusivity,
    and real-world metric distance monitoring.
    """

    def __init__(self, config: dict, distance_estimator: RealWorldDistanceEstimator):
        self.cfg = config
        self.estimator = distance_estimator

        pair_cfg = config.get("pairing", {})
        dist_cfg = config.get("distance", {})

        self.entry_iou_thresh: float = float(pair_cfg.get("entry_iou_threshold", 0.30))
        self.entry_confirm_frames: int = int(pair_cfg.get("entry_confirmation_frames", 18))
        self.moving_entry_confirm_frames: int = int(
            pair_cfg.get("moving_entry_confirmation_frames", 4)
        )
        self.seated_pose_confirm_frames: int = int(
            pair_cfg.get("seated_pose_confirmation_frames", 6)
        )
        self.exit_confirm_frames: int = int(pair_cfg.get("exit_confirmation_frames", 6))
        self.reid_sim_thresh: float = float(pair_cfg.get("reid_similarity_threshold", 0.65))

        # Motion & Emergence thresholds
        self.vehicle_motion_thresh_m: float = float(
            pair_cfg.get("vehicle_motion_threshold_m", 0.40)
        )
        self.max_boarding_walk_speed_mps: float = float(
            pair_cfg.get("max_boarding_walk_speed_mps", 0.45)
        )
        self.dismount_spawn_radius_m: float = float(
            pair_cfg.get("dismount_spawn_radius_m", 1.35)
        )
        self.require_vehicle_activity_for_spawn: bool = bool(
            pair_cfg.get("require_vehicle_activity_for_spawn", False)
        )

        # Timeouts to prevent ghost boxes
        self.unassigned_ghost_timeout_sec: float = float(
            pair_cfg.get("unassigned_ghost_timeout_sec", 1.5)
        )
        self.assigned_ghost_timeout_sec: float = float(
            pair_cfg.get("assigned_ghost_timeout_sec", 15.0)
        )

        self.safe_thresh_m: float = float(dist_cfg.get("safe_distance_threshold_m", 3.0))
        self.warn_thresh_m: float = float(dist_cfg.get("warning_distance_threshold_m", 2.0))
        self.min_kpt_conf: float = float(dist_cfg.get("min_ankle_confidence", 0.35))

        self.vehicles: Dict[str, VehicleSessionRecord] = {}
        self.person_histories: Dict[str, PersonTrackHistory] = {}

        # Check if estimator supports keypoints and foot_px kwargs
        sig = inspect.signature(self.estimator.estimate_driver_vehicle_distance)
        self._estimator_accepts_pose = "keypoints" in sig.parameters

    @staticmethod
    def _bbox_iou(boxA: List[float], boxB: List[float]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter = max(0.0, xB - xA) * max(0.0, yB - yA)
        if inter <= 0.0:
            return 0.0
        areaA = max(1.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        areaB = max(1.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
        return float(inter / (areaA + areaB - inter))

    @staticmethod
    def _driver_cab_overlap_ratio(driver_bbox: List[float], vehicle_bbox: List[float]) -> float:
        """
        Computes overlap between the driver and the interior cabin zone of the vehicle
        (shrinking outer edges so pedestrians brushing past bumpers are ignored).
        """
        dx1, dy1, dx2, dy2 = driver_bbox
        vx1, vy1, vx2, vy2 = vehicle_bbox
        vw = vx2 - vx1
        vh = vy2 - vy1

        cx1 = vx1 + 0.08 * vw
        cy1 = vy1 + 0.05 * vh
        cx2 = vx2 - 0.08 * vw
        cy2 = vy2 - 0.05 * vh

        ix1, iy1 = max(dx1, cx1), max(dy1, cy1)
        ix2, iy2 = min(dx2, cx2), min(dy2, cy2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        inter_area = (ix2 - ix1) * (iy2 - iy1)
        driver_area = max(1.0, (dx2 - dx1) * (dy2 - dy1))
        return float(inter_area / driver_area)

    def _analyze_driver_pose_in_cabin(
        self,
        person_bbox: List[float],
        vehicle_bbox: List[float],
        keypoints: Optional[np.ndarray] = None,
        foot_px: Optional[Tuple[int, int]] = None,
    ) -> Tuple[bool, bool]:
        """
        Uses YOLO-Pose 17 keypoints to distinguish a true vehicle operator inside a cabin
        from a pedestrian walking past a parked car or standing in front of a vehicle.

        Returns:
            (is_valid_cabin_candidate, is_seated_or_chassis_occluded_operator)
        """
        vx1, vy1, vx2, vy2 = vehicle_bbox
        vw = max(1.0, vx2 - vx1)
        vh = max(1.0, vy2 - vy1)

        # Interior cabin bounds
        cx1, cy1 = vx1 + 0.08 * vw, vy1 + 0.04 * vh
        cx2, cy2 = vx2 - 0.08 * vw, vy2 - 0.04 * vh

        # Ground-level check: if a person's feet are below the bottom edge of the vehicle
        # (walking in front of the vehicle on the ground), they are NOT inside the cabin.
        if foot_px is not None:
            fx, fy = foot_px
            if fy > (vy2 - 0.03 * vh):
                return False, False

        if keypoints is None:
            ov = self._driver_cab_overlap_ratio(person_bbox, vehicle_bbox)
            return (ov >= self.entry_iou_thresh), False

        kpts = np.asarray(keypoints)
        if kpts.ndim != 2 or len(kpts) < 17:
            ov = self._driver_cab_overlap_ratio(person_bbox, vehicle_bbox)
            return (ov >= self.entry_iou_thresh), False

        def _valid(idx: int) -> bool:
            conf = float(kpts[idx][2]) if kpts.shape[1] >= 3 else 1.0
            return conf >= self.min_kpt_conf and kpts[idx][0] > 1.0 and kpts[idx][1] > 1.0

        def _in_cabin(idx: int) -> bool:
            return _valid(idx) and (cx1 <= kpts[idx][0] <= cx2) and (cy1 <= kpts[idx][1] <= cy2)

        # 1. Require torso keypoints (5=L_Shoulder, 6=R_Shoulder, 11=L_Hip, 12=R_Hip) inside cabin
        torso_in_cab = sum(1 for idx in (5, 6, 11, 12) if _in_cabin(idx))
        if torso_in_cab < 2:
            return False, False

        # 2. Check ankles (15=L_Ankle, 16=R_Ankle)
        valid_ankles = [kpts[i] for i in (15, 16) if _valid(i)]
        if not valid_ankles:
            # Torso is inside the cabin and ankles are hidden behind vehicle door/chassis -> Operator!
            return True, True

        avg_ankle_y = float(np.mean([a[1] for a in valid_ankles]))
        # If visible ankles are on the ground in front of the vehicle's bottom tire line,
        # the person is standing/walking in front of the vehicle, not inside the cabin.
        if avg_ankle_y > (vy2 - 0.04 * vh):
            return False, False

        # 3. Check seated posture (foreshortened leg-to-torso vertical ratio)
        shoulders_y = [kpts[i][1] for i in (5, 6) if _valid(i)]
        hips_y = [kpts[i][1] for i in (11, 12) if _valid(i)]
        if shoulders_y and hips_y:
            torso_h = max(1.0, float(np.mean(hips_y) - np.mean(shoulders_y)))
            leg_h = max(0.0, avg_ankle_y - float(np.mean(hips_y)))
            if (leg_h / torso_h) < 1.25:
                return True, True

        return True, False

    @staticmethod
    def _cosine_similarity(emb1: Optional[np.ndarray], emb2: Optional[np.ndarray]) -> float:
        if emb1 is None or emb2 is None:
            return 0.0
        v1 = emb1.flatten()
        v2 = emb2.flatten()
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-8
        return float(np.dot(v1, v2) / denom)

    def _update_vehicle_motion(self, rec: VehicleSessionRecord, now: float) -> None:
        vx1, vy1, vx2, vy2 = rec.bbox
        base_px = (0.5 * (vx1 + vx2), vy2)
        world_xy = self.estimator.pixel_to_world(base_px)
        rec.world_pos_history.append((world_xy[0], world_xy[1], now))

        if len(rec.world_pos_history) >= 6:
            x0, y0, _ = rec.world_pos_history[0]
            x1, y1, _ = rec.world_pos_history[-1]
            disp_m = float(np.hypot(x1 - x0, y1 - y0))
            if disp_m >= self.vehicle_motion_thresh_m:
                rec.is_currently_moving = True
                rec.has_ever_moved = True
                rec.last_moved_ts = now
            else:
                rec.is_currently_moving = False

    def _deduplicate_vehicle_tracks(self, detected_vehicles: List[Dict], now: float) -> None:
        """
        Migrates vehicle session if ByteTrack switches IDs on a stationary vehicle (IoU > 0.75).
        """
        active_ids = {str(v["id"]) for v in detected_vehicles}
        for v in detected_vehicles:
            new_id = str(v["id"])
            if new_id in self.vehicles:
                continue
            new_bbox = v["bbox"]
            for old_id, old_rec in list(self.vehicles.items()):
                if old_id not in active_ids and (now - old_rec.last_seen_ts) < 3.0:
                    if self._bbox_iou(new_bbox, old_rec.bbox) > 0.75:
                        old_rec.vehicle_id = new_id
                        self.vehicles[new_id] = old_rec
                        del self.vehicles[old_id]
                        break

    def _compute_distance(
        self,
        driver_id: str,
         person_data: Dict,
        vehicle_id: str,
        vehicle_bbox: List[float],
        frame_shape: Tuple[int, int],
        other_bboxes: Optional[List[List[float]]] = None,
    ) -> DistanceMeasurement:
        """Calls distance estimator with pose keypoints/foot_px when supported."""
        if self._estimator_accepts_pose:
            return self.estimator.estimate_driver_vehicle_distance(
                driver_id=driver_id,
                driver_bbox=person_data["bbox"],
                vehicle_id=vehicle_id,
                vehicle_bbox=vehicle_bbox,
                frame_shape=frame_shape,
                keypoints=person_data.get("keypoints"),
                foot_px=person_data.get("foot_px"),
                other_bboxes=other_bboxes,
            )
        return self.estimator.estimate_driver_vehicle_distance(
            driver_id=driver_id,
            driver_bbox=person_data["bbox"],
            vehicle_id=vehicle_id,
            vehicle_bbox=vehicle_bbox,
            frame_shape=frame_shape,
            other_bboxes=other_bboxes,
        )

    def update(
        self,
        detected_vehicles: List[Dict],
        detected_persons: List[Dict],
        frame_shape: Tuple[int, int],
    ) -> Tuple[Dict[str, VehicleSessionRecord], List[Dict]]:
        now = time.time()
        events: List[Dict] = []

        # 1. Update Person Trajectory Histories using YOLO-Pose ground_uv (foot_px) if available
        person_map: Dict[str, Dict] = {}
        for p in detected_persons:
            pid = str(p["id"])
            person_map[pid] = p
            px1, py1, px2, py2 = p["bbox"]
            foot_px = p.get("foot_px") or (int(0.5 * (px1 + px2)), int(py2))
            foot_world = p.get("bev_xy") or self.estimator.pixel_to_world(foot_px)
            if pid not in self.person_histories:
                self.person_histories[pid] = PersonTrackHistory(
                    person_id=pid,
                    first_seen_ts=now,
                    spawn_world_xy=foot_world,
                    current_world_xy=foot_world,
                )
            self.person_histories[pid].update_position(foot_world, now)

        for rec in self.vehicles.values():
            rec.is_visible_in_current_frame = False

        self._deduplicate_vehicle_tracks(detected_vehicles, now)

        # Enforce 1-to-1 driver exclusivity across all vehicles
        bound_driver_ids: Set[str] = {
            r.associated_driver_id
            for r in self.vehicles.values()
            if r.associated_driver_id is not None
        }

        for v in detected_vehicles:
            vid = str(v["id"])
            vbbox = v["bbox"]

            if vid not in self.vehicles:
                self.vehicles[vid] = VehicleSessionRecord(
                    vehicle_id=vid, bbox=vbbox, last_seen_ts=now
                )
            rec = self.vehicles[vid]
            rec.bbox = vbbox
            rec.last_seen_ts = now
            rec.is_visible_in_current_frame = True
            self._update_vehicle_motion(rec, now)

            # =========================================================================
            # STEP A: UNASSIGNED VEHICLE -> POSE-GUIDED DRIVER IDENTIFICATION
            # =========================================================================
            if rec.associated_driver_id is None:
                best_cab_pid: Optional[str] = None
                best_overlap: float = 0.0
                best_is_seated_pose: bool = False

                for pid, pdata in person_map.items():
                    if pid in bound_driver_ids:
                        continue

                    in_cab, is_seated_pose = self._analyze_driver_pose_in_cabin(
                        person_bbox=pdata["bbox"],
                        vehicle_bbox=vbbox,
                        keypoints=pdata.get("keypoints"),
                        foot_px=pdata.get("foot_px"),
                    )
                    if not in_cab:
                        continue

                    ov = self._driver_cab_overlap_ratio(pdata["bbox"], vbbox)
                    effective_score = max(ov, 0.60 if is_seated_pose else 0.35)
                    if effective_score > best_overlap:
                        best_overlap = effective_score
                        best_cab_pid = pid
                        best_is_seated_pose = is_seated_pose

                # Reset overlap counters for any person no longer inside the cabin
                for tracked_pid in list(rec.overlap_frames.keys()):
                    if tracked_pid != best_cab_pid or best_overlap < self.entry_iou_thresh:
                        rec.overlap_frames[tracked_pid] = 0

                # Path 1 & 2: Person is verified inside the cabin ROI via Pose + Overlap
                if best_cab_pid is not None and best_overlap >= self.entry_iou_thresh:
                    p_hist = self.person_histories[best_cab_pid]
                    person_speed = p_hist.get_speed_mps()

                    if rec.is_currently_moving:
                        rec.overlap_frames[best_cab_pid] = (
                            rec.overlap_frames.get(best_cab_pid, 0) + 1
                        )
                        req_frames = self.moving_entry_confirm_frames
                    elif best_is_seated_pose and person_speed <= self.max_boarding_walk_speed_mps:
                        rec.overlap_frames[best_cab_pid] = (
                            rec.overlap_frames.get(best_cab_pid, 0) + 1
                        )
                        req_frames = self.seated_pose_confirm_frames
                    elif person_speed <= self.max_boarding_walk_speed_mps:
                        rec.overlap_frames[best_cab_pid] = (
                            rec.overlap_frames.get(best_cab_pid, 0) + 1
                        )
                        req_frames = self.entry_confirm_frames
                    else:
                        rec.overlap_frames[best_cab_pid] = 0
                        req_frames = self.entry_confirm_frames

                    if rec.overlap_frames.get(best_cab_pid, 0) >= req_frames:
                        rec.associated_driver_id = best_cab_pid
                        bound_driver_ids.add(best_cab_pid)
                        rec.driver_bbox = person_map[best_cab_pid]["bbox"]
                        if person_map[best_cab_pid].get("embedding") is not None:
                            rec.driver_reid_embedding = person_map[best_cab_pid]["embedding"]
                        rec.state = VehicleSafetyState.DRIVER_ONBOARD
                        rec.frames_outside_cab = 0
                        rec.alarm_active = False
                        rec.last_measurement = None
                        events.append(
                            {
                                "event_type": "DRIVER_BOARDED",
                                "vehicle_id": vid,
                                "driver_id": best_cab_pid,
                                "distance_m": 0.0,
                                "timestamp": now,
                            }
                        )
                        continue

                # Path 3: Cabin Dismount Emergence (Driver steps out of an occluded forklift cab)
                if rec.associated_driver_id is None:
                    vehicle_active_ok = (
                        not self.require_vehicle_activity_for_spawn or rec.has_ever_moved
                    )
                    if vehicle_active_ok:
                        for pid, pdata in person_map.items():
                            if pid in bound_driver_ids:
                                continue
                            p_hist = self.person_histories[pid]
                            if 3 <= p_hist.total_frames_seen <= 10:
                                meas = self._compute_distance(
                                    driver_id=pid,
                                    person_data=pdata,
                                    vehicle_id=vid,
                                    vehicle_bbox=vbbox,
                                    frame_shape=frame_shape,
                                )
                                if (
                                    not meas.is_uncertain
                                    and meas.distance_m is not None
                                    and meas.distance_m <= self.dismount_spawn_radius_m
                                    and p_hist.max_distance_from_spawn_m() < 1.1
                                ):
                                    rec.associated_driver_id = pid
                                    bound_driver_ids.add(pid)
                                    rec.driver_bbox = pdata["bbox"]
                                    if pdata.get("embedding") is not None:
                                        rec.driver_reid_embedding = pdata["embedding"]
                                    rec.state = VehicleSafetyState.DRIVER_EXITED_SAFE
                                    rec.frames_outside_cab = self.exit_confirm_frames
                                    rec.exit_timestamp = now
                                    rec.last_measurement = meas
                                    events.append(
                                        {
                                            "event_type": "DRIVER_EXITED",
                                            "vehicle_id": vid,
                                            "driver_id": pid,
                                            "distance_m": meas.distance_m,
                                            "timestamp": now,
                                        }
                                    )
                                    break
                continue

            # =========================================================================
            # STEP B: VEHICLE HAS AN ASSOCIATED DRIVER -> MONITOR EXIT & DISTANCE
            # =========================================================================
            matched_person = person_map.get(rec.associated_driver_id)

            # Re-ID fallback if track ID changed during cabin exit
            if matched_person is None and rec.driver_reid_embedding is not None:
                best_sim = 0.0
                best_reid_person = None
                for pid, pdata in person_map.items():
                    if pid in bound_driver_ids and pid != rec.associated_driver_id:
                        continue
                    sim = self._cosine_similarity(
                        rec.driver_reid_embedding, pdata.get("embedding")
                    )
                    if sim > best_sim and sim >= self.reid_sim_thresh:
                        best_sim = sim
                        best_reid_person = pdata
                if best_reid_person is not None:
                    bound_driver_ids.discard(rec.associated_driver_id)
                    rec.associated_driver_id = str(best_reid_person["id"])
                    bound_driver_ids.add(rec.associated_driver_id)
                    matched_person = best_reid_person

            if matched_person is not None:
                rec.driver_bbox = matched_person["bbox"]
                if matched_person.get("embedding") is not None:
                    rec.driver_reid_embedding = matched_person["embedding"]

                in_cab, _ = self._analyze_driver_pose_in_cabin(
                    person_bbox=rec.driver_bbox,
                    vehicle_bbox=vbbox,
                    keypoints=matched_person.get("keypoints"),
                    foot_px=matched_person.get("foot_px"),
                )
                if in_cab:
                    rec.frames_outside_cab = 0
                    rec.state = VehicleSafetyState.DRIVER_ONBOARD
                    rec.alarm_active = False
                    rec.last_measurement = None
                    continue

                rec.frames_outside_cab += 1
                if rec.frames_outside_cab >= self.exit_confirm_frames:
                    was_onboard = rec.state == VehicleSafetyState.DRIVER_ONBOARD
                    if was_onboard:
                        rec.exit_timestamp = now

                    other_boxes = [
                        p["bbox"]
                        for pid, p in person_map.items()
                        if pid != rec.associated_driver_id
                    ]
                    measurement = self._compute_distance(
                        driver_id=rec.associated_driver_id,
                        person_data=matched_person,
                        vehicle_id=vid,
                        vehicle_bbox=vbbox,
                        frame_shape=frame_shape,
                        other_bboxes=other_boxes,
                    )
                    rec.last_measurement = measurement

                    if was_onboard:
                        events.append(
                            {
                                "event_type": "DRIVER_EXITED",
                                "vehicle_id": vid,
                                "driver_id": rec.associated_driver_id,
                                "distance_m": measurement.distance_m,
                                "timestamp": now,
                            }
                        )

                    if measurement.is_uncertain:
                        rec.state = VehicleSafetyState.DISTANCE_UNCERTAIN
                        rec.alarm_active = False
                    elif measurement.distance_m is not None:
                        if measurement.distance_m > self.safe_thresh_m:
                            prev_alarm = rec.alarm_active
                            rec.state = VehicleSafetyState.DISTANCE_VIOLATION_ALARM
                            rec.alarm_active = True
                            if not prev_alarm:
                                events.append(
                                    {
                                        "event_type": "SAFE_DISTANCE_VIOLATION_ALARM",
                                        "vehicle_id": vid,
                                        "driver_id": rec.associated_driver_id,
                                        "distance_m": measurement.distance_m,
                                        "threshold_m": self.safe_thresh_m,
                                        "timestamp": now,
                                    }
                                )
                        elif measurement.distance_m > self.warn_thresh_m:
                            rec.state = VehicleSafetyState.DRIVER_EXITED_WARNING
                            rec.alarm_active = False
                        else:
                            rec.state = VehicleSafetyState.DRIVER_EXITED_SAFE
                            rec.alarm_active = False
            else:
                if (
                    rec.is_currently_moving
                    and rec.state == VehicleSafetyState.DRIVER_ONBOARD
                ):
                    rec.alarm_active = False
                elif rec.state != VehicleSafetyState.UNASSIGNED:
                    rec.state = VehicleSafetyState.DISTANCE_UNCERTAIN
                    rec.alarm_active = False
                    rec.last_measurement = DistanceMeasurement(
                        distance_m=None,
                        driver_world_xy=None,
                        vehicle_world_xy=None,
                        driver_ground_px=(0, 0),
                        vehicle_nearest_px=(0, 0),
                        is_uncertain=True,
                        uncertainty_reason="DRIVER_NOT_VISIBLE_IN_FRAME",
                    )

        # Clean up stale ghost vehicles
        stale_ids = []
        for vid, r in self.vehicles.items():
            age = now - r.last_seen_ts
            if (
                r.state == VehicleSafetyState.UNASSIGNED
                and age > self.unassigned_ghost_timeout_sec
            ):
                stale_ids.append(vid)
            elif age > self.assigned_ghost_timeout_sec:
                stale_ids.append(vid)
        for vid in stale_ids:
            del self.vehicles[vid]

        return self.vehicles, events