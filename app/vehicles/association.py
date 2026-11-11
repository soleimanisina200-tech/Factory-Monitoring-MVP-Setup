# app/vehicles/association.py
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np

from app.vehicles.calibration import DistanceMeasurement, GroundPlaneProjector
from app.vehicles.detector import VehicleTrack


class VehicleSafetyState( str, Enum ):
    UNASSIGNED = "UNASSIGNED"
    OPERATED = "OPERATED"
    DRIVER_EXITED_SAFE = "DRIVER_EXITED_SAFE"
    WARNING_PROXIMITY = "WARNING_PROXIMITY"
    ALARM_VIOLATION = "ALARM_VIOLATION"
    UNCERTAIN_DISTANCE = "UNCERTAIN_DISTANCE"


@dataclass
class PersonObservation:
    """Normalized representation of a tracked person from the existing pipeline."""
    global_id: int
    bbox: Tuple[float, float, float, float]
    keypoints: Optional[np.ndarray] = None
    foot_px: Optional[Tuple[int, int]] = None
    reid_embedding: Optional[np.ndarray] = None


@dataclass
class ForkliftSession:
    forklift_id: int
    bbox: Tuple[float, float, float, float]
    state: VehicleSafetyState = VehicleSafetyState.UNASSIGNED
    driver_id: Optional[int] = None
    driver_embedding: Optional[np.ndarray] = None
    candidate_driver_id: Optional[int] = None
    candidate_overlap_frames: int = 0
    outside_cabin_frames: int = 0
    violation_frames: int = 0
    last_measurement: Optional[DistanceMeasurement] = None
    alarm_active: bool = False
    last_seen_frame: int = 0


@dataclass
class SafetyEvent:
    event_type: str  # DRIVER_ASSOCIATED, DRIVER_EXITED, SAFE_DISTANCE_VIOLATION, DISTANCE_UNCERTAIN, DRIVER_RETURNED
    forklift_id: int
    driver_id: Optional[int]
    state: VehicleSafetyState
    distance_m: Optional[float]
    threshold_m: float
    uncertainty_reason: str
    vehicle_bbox: Tuple[float, float, float, float]
    driver_bbox: Optional[Tuple[float, float, float, float]]


class DriverVehicleMonitor:
    """
    Manages the lifecycle of each forklift, pairs drivers via cabin overlap + OSNet Re-ID,
    detects driver exits, continuously monitors metric ground distance, and emits safety events.
    """

    def __init__(self, config: Dict, projector: GroundPlaneProjector):
        self.projector = projector
        assoc_cfg = config.get("association", {})
        dist_cfg = config.get("distance", {})

        self.cabin_iou_thresh = float(assoc_cfg.get("cabin_iou_threshold", 0.25))
        self.cabin_expansion = float(assoc_cfg.get("cabin_expansion_ratio", 0.15))
        self.pairing_confirm_frames = int(assoc_cfg.get("pairing_confirm_frames", 10))
        self.exit_confirm_frames = int(assoc_cfg.get("exit_confirm_frames", 12))
        self.max_reid_cosine_dist = float(assoc_cfg.get("max_reid_cosine_dist", 0.35))

        self.warn_threshold_m = float(dist_cfg.get("warning_threshold_m", 2.0))
        self.safe_threshold_m = float(dist_cfg.get("safe_threshold_m", 3.0))
        self.alarm_confirm_frames = int(dist_cfg.get("alarm_confirm_frames", 8))

        self.sessions: Dict[int, ForkliftSession] = {}
        self.frame_idx: int = 0

    def _compute_intersection_over_person(
        self,
        person_bbox: Tuple[float, float, float, float],
        vehicle_bbox: Tuple[float, float, float, float],
    ) -> float:
        """Calculates the fraction of the person's bounding box inside the expanded cabin envelope."""
        vx1, vy1, vx2, vy2 = vehicle_bbox
        vw, vh = vx2 - vx1, vy2 - vy1
        ex1 = vx1 - vw * self.cabin_expansion
        ey1 = vy1 - vh * self.cabin_expansion
        ex2 = vx2 + vw * self.cabin_expansion
        ey2 = vy2 + vh * self.cabin_expansion

        px1, py1, px2, py2 = person_bbox
        ix1, iy1 = max(px1, ex1), max(py1, ey1)
        ix2, iy2 = min(px2, ex2), min(py2, ey2)

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        inter_area = (ix2 - ix1) * (iy2 - iy1)
        person_area = max(1.0, (px2 - px1) * (py2 - py1))
        return float(inter_area / person_area)

    @staticmethod
    def _cosine_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
        n1 = np.linalg.norm(emb1)
        n2 = np.linalg.norm(emb2)
        if n1 == 0 or n2 == 0:
            return 1.0
        return float(1.0 - np.dot(emb1, emb2) / (n1 * n2))

    def _match_associated_driver(
        self, session: ForkliftSession, persons: Dict[int, PersonObservation]
    ) -> Optional[PersonObservation]:
        """Finds the associated driver by global ID, falling back to OSNet Re-ID cosine similarity."""
        if session.driver_id is None:
            return None

        if session.driver_id in persons:
            obs = persons[session.driver_id]
            if obs.reid_embedding is not None:
                session.driver_embedding = obs.reid_embedding
            return obs

        # Fallback: If ByteTrack switched IDs as the driver stepped out of the cabin,
        # match using the cached OSNet Re-ID embedding
        if session.driver_embedding is not None:
            best_obs = None
            best_dist = self.max_reid_cosine_dist
            for obs in persons.values():
                if obs.reid_embedding is not None:
                    c_dist = self._cosine_distance(
                        session.driver_embedding, obs.reid_embedding
                    )
                    if c_dist < best_dist:
                        best_dist = c_dist
                        best_obs = obs
            if best_obs is not None:
                session.driver_id = best_obs.global_id
                return best_obs

        return None

    def update(
        self,
        vehicle_tracks: List[VehicleTrack],
        person_observations: List[PersonObservation],
    ) -> Tuple[Dict[int, ForkliftSession], List[SafetyEvent]]:
        self.frame_idx += 1
        events: List[SafetyEvent] = []
        persons_by_id = {p.global_id: p for p in person_observations}

        for v_track in vehicle_tracks:
            fid = v_track.track_id
            if fid not in self.sessions:
                self.sessions[fid] = ForkliftSession(
                    forklift_id=fid,
                    bbox=v_track.bbox,
                    last_seen_frame=self.frame_idx,
                )
            session = self.sessions[fid]
            session.bbox = v_track.bbox
            session.last_seen_frame = self.frame_idx

            # 1. Check if a new driver is boarding/operating an unassigned forklift
            if session.driver_id is None:
                best_pid = None
                best_iop = 0.0
                for p in person_observations:
                    iop = self._compute_intersection_over_person(p.bbox, session.bbox)
                    if iop >= self.cabin_iou_thresh and iop > best_iop:
                        best_iop = iop
                        best_pid = p.global_id

                if best_pid is not None:
                    if session.candidate_driver_id == best_pid:
                        session.candidate_overlap_frames += 1
                    else:
                        session.candidate_driver_id = best_pid
                        session.candidate_overlap_frames = 1

                    if session.candidate_overlap_frames >= self.pairing_confirm_frames:
                        session.driver_id = best_pid
                        session.driver_embedding = persons_by_id[best_pid].reid_embedding
                        session.state = VehicleSafetyState.OPERATED
                        session.outside_cabin_frames = 0
                        session.violation_frames = 0
                        session.alarm_active = False
                        events.append(
                            SafetyEvent(
                                event_type="DRIVER_ASSOCIATED",
                                forklift_id=fid,
                                driver_id=best_pid,
                                state=session.state,
                                distance_m=0.0,
                                threshold_m=self.safe_threshold_m,
                                uncertainty_reason="OK",
                                vehicle_bbox=session.bbox,
                                driver_bbox=persons_by_id[best_pid].bbox,
                            )
                        )
                else:
                    session.candidate_driver_id = None
                    session.candidate_overlap_frames = 0
                continue

            # 2. Forklift has an assigned driver -> monitor cabin presence & distance
            driver_obs = self._match_associated_driver(session, persons_by_id)
            if driver_obs is None:
                # Driver is temporarily occluded inside cabin or behind mast
                continue

            iop = self._compute_intersection_over_person(driver_obs.bbox, session.bbox)

            # Check if driver is inside/boarding the forklift cabin
            if iop >= self.cabin_iou_thresh:
                if session.state not in (
                    VehicleSafetyState.OPERATED,
                    VehicleSafetyState.UNASSIGNED,
                ):
                    events.append(
                        SafetyEvent(
                            event_type="DRIVER_RETURNED",
                            forklift_id=fid,
                            driver_id=session.driver_id,
                            state=VehicleSafetyState.OPERATED,
                            distance_m=0.0,
                            threshold_m=self.safe_threshold_m,
                            uncertainty_reason="OK",
                            vehicle_bbox=session.bbox,
                            driver_bbox=driver_obs.bbox,
                        )
                    )
                session.state = VehicleSafetyState.OPERATED
                session.outside_cabin_frames = 0
                session.violation_frames = 0
                session.alarm_active = False
                session.last_measurement = None
                continue

            # Driver is outside the cabin envelope
            session.outside_cabin_frames += 1
            if session.outside_cabin_frames < self.exit_confirm_frames:
                # Still within exit debounce window
                continue

            prev_state = session.state

            # Compute real-world distance in meters
            measurement = self.projector.compute_driver_forklift_distance(
                driver_id=driver_obs.global_id,
                person_bbox=driver_obs.bbox,
                forklift_id=fid,
                vehicle_bbox=session.bbox,
                keypoints=driver_obs.keypoints,
                precomputed_foot_px=driver_obs.foot_px,
            )
            session.last_measurement = measurement

            # Emit DRIVER_EXITED event on initial confirmed exit transition
            if prev_state == VehicleSafetyState.OPERATED:
                events.append(
                    SafetyEvent(
                        event_type="DRIVER_EXITED",
                        forklift_id=fid,
                        driver_id=session.driver_id,
                        state=VehicleSafetyState.DRIVER_EXITED_SAFE,
                        distance_m=measurement.distance_m,
                        threshold_m=self.safe_threshold_m,
                        uncertainty_reason=measurement.uncertainty_reason,
                        vehicle_bbox=session.bbox,
                        driver_bbox=driver_obs.bbox,
                    )
                )

            # Handle Uncertain Distance (Occlusion / Uncalibrated / Out-of-bounds)
            if not measurement.is_certain or measurement.distance_m is None:
                session.state = VehicleSafetyState.UNCERTAIN_DISTANCE
                session.alarm_active = False
                if prev_state != VehicleSafetyState.UNCERTAIN_DISTANCE:
                    events.append(
                        SafetyEvent(
                            event_type="DISTANCE_UNCERTAIN",
                            forklift_id=fid,
                            driver_id=session.driver_id,
                            state=session.state,
                            distance_m=None,
                            threshold_m=self.safe_threshold_m,
                            uncertainty_reason=measurement.uncertainty_reason,
                            vehicle_bbox=session.bbox,
                            driver_bbox=driver_obs.bbox,
                        )
                    )
                continue

            # Evaluate physical distance against thresholds
            dist_m = measurement.distance_m
            if dist_m > self.safe_threshold_m:
                session.violation_frames += 1
                if session.violation_frames >= self.alarm_confirm_frames:
                    session.state = VehicleSafetyState.ALARM_VIOLATION
                    if not session.alarm_active:
                        session.alarm_active = True
                        events.append(
                            SafetyEvent(
                                event_type="SAFE_DISTANCE_VIOLATION",
                                forklift_id=fid,
                                driver_id=session.driver_id,
                                state=session.state,
                                distance_m=dist_m,
                                threshold_m=self.safe_threshold_m,
                                uncertainty_reason="OK",
                                vehicle_bbox=session.bbox,
                                driver_bbox=driver_obs.bbox,
                            )
                        )
            elif dist_m > self.warn_threshold_m:
                session.violation_frames = max(0, session.violation_frames - 1)
                session.state = VehicleSafetyState.WARNING_PROXIMITY
                session.alarm_active = False
            else:
                session.violation_frames = 0
                session.state = VehicleSafetyState.DRIVER_EXITED_SAFE
                session.alarm_active = False

        return self.sessions, events