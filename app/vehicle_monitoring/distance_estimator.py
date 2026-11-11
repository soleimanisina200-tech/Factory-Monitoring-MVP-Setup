# app/vehicle_monitoring/distance_estimator.py
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class DistanceMeasurement:
    distance_m: Optional[float]
    driver_world_xy: Optional[Tuple[float, float]]
    vehicle_world_xy: Optional[Tuple[float, float]]
    driver_ground_px: Tuple[int, int]
    vehicle_nearest_px: Tuple[int, int]
    is_uncertain: bool
    uncertainty_reason: str = ""


class RealWorldDistanceEstimator:
    """
    Computes perspective-corrected physical distance in meters between a driver's
    YOLO-Pose ankle keypoints and the nearest edge of the forklift's ground footprint.
    """

    def __init__(self, config: dict):
        dist_cfg = config.get("distance", {})
        cal_cfg = config.get("calibration", {})

        self.alpha: float = float(dist_cfg.get("ema_smoothing_alpha", 0.35))
        self.max_step_jump_m: float = float(dist_cfg.get("max_step_jump_m", 1.8))
        self.min_aspect_ratio: float = float(dist_cfg.get("min_bbox_aspect_ratio", 1.15))
        self.min_ankle_conf: float = float(dist_cfg.get("min_ankle_confidence", 0.40))
        self.border_margin_px: int = int(dist_cfg.get("border_margin_px", 12))
        self.fallback_scale: float = float(cal_cfg.get("fallback_meters_per_pixel", 0.012))

        img_pts = np.array(cal_cfg.get("image_points_px", []), dtype=np.float32)
        world_pts = np.array(cal_cfg.get("world_points_m", []), dtype=np.float32)

        self.homography_matrix: Optional[np.ndarray] = None
        self.inv_homography_matrix: Optional[np.ndarray] = None

        if len(img_pts) >= 4 and len(world_pts) >= 4:
            H, _ = cv2.findHomography(img_pts, world_pts, method=cv2.RANSAC)
            if H is not None:
                self.homography_matrix = H
                try:
                    self.inv_homography_matrix = np.linalg.inv(H)
                except np.linalg.LinAlgError:
                    self.inv_homography_matrix = None

        self._smoothed_world_coords: Dict[str, Tuple[float, float]] = {}

    def pixel_to_world(self, pt_px: Tuple[float, float]) -> Tuple[float, float]:
        if self.homography_matrix is not None:
            pts = np.array([[[float(pt_px[0]), float(pt_px[1])]]], dtype=np.float32)
            world_pt = cv2.perspectiveTransform(pts, self.homography_matrix)[0][0]
            return float(world_pt[0]), float(world_pt[1])
        return float(pt_px[0]) * self.fallback_scale, float(pt_px[1]) * self.fallback_scale

    def _stabilize_coord(self, key: str, raw_xy: Tuple[float, float]) -> Tuple[float, float]:
        if key not in self._smoothed_world_coords:
            self._smoothed_world_coords[key] = raw_xy
            return raw_xy

        prev_x, prev_y = self._smoothed_world_coords[key]
        dx, dy = raw_xy[0] - prev_x, raw_xy[1] - prev_y
        step_dist = float(np.hypot(dx, dy))

        if step_dist > self.max_step_jump_m and step_dist > 1e-5:
            scale = self.max_step_jump_m / step_dist
            raw_xy = (prev_x + dx * scale, prev_y + dy * scale)

        smooth_x = self.alpha * raw_xy[0] + (1.0 - self.alpha) * prev_x
        smooth_y = self.alpha * raw_xy[1] + (1.0 - self.alpha) * prev_y
        self._smoothed_world_coords[key] = (smooth_x, smooth_y)
        return smooth_x, smooth_y

    def extract_driver_ground_point(
        self,
        driver_bbox: List[float],
        frame_shape: Tuple[int, int],
        keypoints: Optional[np.ndarray] = None,
        foot_px: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Tuple[int, int], bool, str]:
        """
        Extracts the ground-contact pixel from YOLO-Pose ankles (indices 15, 16)
        or precomputed geometry.py foot_px, flagging uncertainty if ankles are occluded.
        """
        h_frame, w_frame = frame_shape[:2]
        x1, y1, x2, y2 = driver_bbox

        # 1. Use YOLO-Pose keypoints if available (COCO 15=left_ankle, 16=right_ankle)
        if keypoints is not None:
            kpts = np.asarray(keypoints)
            if kpts.ndim == 2 and len(kpts) >= 17:
                l_ank, r_ank = kpts[15], kpts[16]
                l_conf = float(l_ank[2]) if kpts.shape[1] >= 3 else 1.0
                r_conf = float(r_ank[2]) if kpts.shape[1] >= 3 else 1.0

                valid_pts = []
                if l_conf >= self.min_ankle_conf and l_ank[0] > 1 and l_ank[1] > 1:
                    valid_pts.append((float(l_ank[0]), float(l_ank[1])))
                if r_conf >= self.min_ankle_conf and r_ank[0] > 1 and r_ank[1] > 1:
                    valid_pts.append((float(r_ank[0]), float(r_ank[1])))

                if valid_pts:
                    gx = int(np.mean([p[0] for p in valid_pts]))
                    gy = int(np.mean([p[1] for p in valid_pts]))
                    if gy >= (h_frame - self.border_margin_px):
                        return (gx, gy), True, "ANKLES_CLIPPED_BY_BORDER"
                    return (gx, gy), False, ""

                # Both ankles have low confidence -> lower body is occluded behind obstacle/load
                fallback_pt = (int(0.5 * (x1 + x2)), int(y2))
                return fallback_pt, True, "ANKLES_OCCLUDED_LOW_POSE_CONF"

        # 2. Use precomputed foot_px from app/logic/geometry.py if provided
        if foot_px is not None:
            gx, gy = int(foot_px[0]), int(foot_px[1])
            aspect_ratio = max(1.0, y2 - y1) / max(1.0, x2 - x1)
            if aspect_ratio < self.min_aspect_ratio:
                return (gx, gy), True, "LOWER_BODY_OCCLUDED"
            return (gx, gy), False, ""

        # 3. Fallback to bounding-box bottom center
        gx, gy = int(0.5 * (x1 + x2)), int(y2)
        aspect_ratio = max(1.0, y2 - y1) / max(1.0, x2 - x1)
        if gy >= (h_frame - self.border_margin_px):
            return (gx, gy), True, "FEET_CLIPPED_BY_FRAME_BORDER"
        if aspect_ratio < self.min_aspect_ratio:
            return (gx, gy), True, "LOWER_BODY_OCCLUDED_TRUNCATED_BBOX"
        return (gx, gy), False, ""

    @staticmethod
    def _point_to_segment_nearest(
        px: float, py: float, ax: float, ay: float, bx: float, by: float
    ) -> Tuple[float, float, float]:
        abx, aby = bx - ax, by - ay
        apx, apy = px - ax, py - ay
        ab_len_sq = abx * abx + aby * aby
        if ab_len_sq <= 1e-8:
            return float(np.hypot(apx, apy)), ax, ay
        t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab_len_sq))
        nx, ny = ax + t * abx, ay + t * aby
        return float(np.hypot(px - nx, py - ny)), nx, ny

    def estimate_driver_vehicle_distance(
        self,
        driver_id: str,
        driver_bbox: List[float],
        vehicle_id: str,
        vehicle_bbox: List[float],
        frame_shape: Tuple[int, int],
        keypoints: Optional[np.ndarray] = None,
        foot_px: Optional[Tuple[int, int]] = None,
        other_bboxes: Optional[List[List[float]]] = None,
    ) -> DistanceMeasurement:
        vx1, vy1, vx2, vy2 = vehicle_bbox

        driver_foot_px, is_uncertain, reason = self.extract_driver_ground_point(
            driver_bbox=driver_bbox,
            frame_shape=frame_shape,
            keypoints=keypoints,
            foot_px=foot_px,
        )

        veh_bl_px = (int(vx1), int(vy2))
        veh_br_px = (int(vx2), int(vy2))
        veh_ml_px = (int(vx1), int(0.75 * vy2 + 0.25 * vy1))
        veh_mr_px = (int(vx2), int(0.75 * vy2 + 0.25 * vy1))

        raw_driver_xy = self.pixel_to_world(driver_foot_px)
        driver_xy = self._stabilize_coord(f"driver_{driver_id}", raw_driver_xy)

        v_bl = self._stabilize_coord(f"veh_{vehicle_id}_bl", self.pixel_to_world(veh_bl_px))
        v_br = self._stabilize_coord(f"veh_{vehicle_id}_br", self.pixel_to_world(veh_br_px))
        v_ml = self._stabilize_coord(f"veh_{vehicle_id}_ml", self.pixel_to_world(veh_ml_px))
        v_mr = self._stabilize_coord(f"veh_{vehicle_id}_mr", self.pixel_to_world(veh_mr_px))

        footprint_segments = [(v_bl, v_br), (v_bl, v_ml), (v_br, v_mr), (v_ml, v_mr)]
        min_dist_m = float("inf")
        nearest_world_xy = v_bl

        for (ax, ay), (bx, by) in footprint_segments:
            d, nx, ny = self._point_to_segment_nearest(
                driver_xy[0], driver_xy[1], ax, ay, bx, by
            )
            if d < min_dist_m:
                min_dist_m = d
                nearest_world_xy = (nx, ny)

        if self.inv_homography_matrix is not None:
            w_pt = np.array([[[nearest_world_xy[0], nearest_world_xy[1]]]], dtype=np.float32)
            px_pt = cv2.perspectiveTransform(w_pt, self.inv_homography_matrix)[0][0]
            veh_nearest_px = (int(px_pt[0]), int(px_pt[1]))
        else:
            veh_nearest_px = (int(0.5 * (vx1 + vx2)), int(vy2))

        return DistanceMeasurement(
            distance_m=round(min_dist_m, 2),
            driver_world_xy=driver_xy,
            vehicle_world_xy=nearest_world_xy,
            driver_ground_px=driver_foot_px,
            vehicle_nearest_px=veh_nearest_px,
            is_uncertain=is_uncertain,
            uncertainty_reason=reason,
        )