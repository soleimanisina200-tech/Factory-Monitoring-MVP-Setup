# app/vehicles/calibration.py
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class MetricPointResult:
    x_m: Optional[float]
    y_m: Optional[float]
    is_certain: bool
    uncertainty_reason: str = "OK"


@dataclass
class DistanceMeasurement:
    distance_m: Optional[float]
    is_certain: bool
    uncertainty_reason: str
    driver_world_m: Optional[Tuple[float, float]]
    vehicle_world_m: Optional[Tuple[float, float]]
    driver_ground_px: Tuple[int, int]
    vehicle_ground_px: Tuple[int, int]


class MetricKalmanFilter2D:
    """Stabilizes real-world (X_m, Y_m) ground coordinates against detection jitter."""

    def __init__(self, process_noise: float = 0.05, measurement_noise: float = 0.15):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
        )
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        self.initialized = False

    def update(self, x_m: float, y_m: float) -> Tuple[float, float]:
        measurement = np.array([[np.float32(x_m)], [np.float32(y_m)]])
        if not self.initialized:
            self.kf.statePre = np.array(
                [[np.float32(x_m)], [np.float32(y_m)], [0.0], [0.0]], dtype=np.float32
            )
            self.kf.statePost = self.kf.statePre.copy()
            self.initialized = True
            return x_m, y_m

        self.kf.predict()
        estimated = self.kf.correct(measurement)
        return float(estimated[0, 0]), float(estimated[1, 0])


class GroundPlaneProjector:
    """
    Converts 2D image ground-contact points into real-world metric coordinates (meters)
    using planar homography, lens undistortion, and explicit uncertainty validation.
    """

    def __init__(self, config: Dict):
        cal_cfg = config.get("calibration", {})
        dist_cfg = config.get("distance", {})

        self.is_calibrated = bool(cal_cfg.get("calibrated", False))
        self.min_ankle_conf = float(dist_cfg.get("min_ankle_confidence", 0.40))
        self.min_bbox_ar = float(dist_cfg.get("min_bbox_aspect_ratio", 1.15))
        self.max_extrapolation_m = float(dist_cfg.get("max_extrapolation_margin_m", 3.0))
        self.process_noise = float(dist_cfg.get("kalman_process_noise", 0.05))
        self.measurement_noise = float(dist_cfg.get("kalman_measurement_noise", 0.15))

        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        if cal_cfg.get("camera_matrix") and cal_cfg.get("dist_coeffs"):
            self.camera_matrix = np.array(cal_cfg["camera_matrix"], dtype=np.float64)
            self.dist_coeffs = np.array(cal_cfg["dist_coeffs"], dtype=np.float64)

        self.homography: Optional[np.ndarray] = None
        self.world_bounds_polygon: Optional[np.ndarray] = None
        self._filters: Dict[str, MetricKalmanFilter2D] = {}

        if self.is_calibrated:
            img_pts = np.array(cal_cfg.get("image_points_px", []), dtype=np.float32)
            world_pts = np.array(cal_cfg.get("world_points_m", []), dtype=np.float32)
            if len(img_pts) >= 4 and len(img_pts) == len(world_pts):
                if self.camera_matrix is not None and self.dist_coeffs is not None:
                    img_pts = self._undistort_px(img_pts)
                H, status = cv2.findHomography(img_pts, world_pts, cv2.RANSAC, 3.0)
                if H is not None:
                    self.homography = H
                    self.world_bounds_polygon = cv2.convexHull(world_pts)
                else:
                    self.is_calibrated = False
            else:
                self.is_calibrated = False

    def _undistort_px(self, pts: np.ndarray) -> np.ndarray:
        reshaped = pts.reshape(-1, 1, 2).astype(np.float64)
        undist = cv2.undistortPoints(
            reshaped, self.camera_matrix, self.dist_coeffs, P=self.camera_matrix
        )
        return undist.reshape(-1, 2).astype(np.float32)

    def project_pixel_to_meters(
        self, u: float, v: float, filter_key: Optional[str] = None
    ) -> MetricPointResult:
        if not self.is_calibrated or self.homography is None:
            return MetricPointResult(None, None, False, "CAMERA_UNCALIBRATED")

        pt = np.array([[u, v]], dtype=np.float32)
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            pt = self._undistort_px(pt)

        pts_homogeneous = pt.reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(pts_homogeneous, self.homography)
        x_m, y_m = float(projected[0, 0, 0]), float(projected[0, 0, 1])

        # Check if projected point is excessively outside calibrated ground region
        if self.world_bounds_polygon is not None:
            signed_dist = cv2.pointPolygonTest(
                self.world_bounds_polygon, (x_m, y_m), measureDist=True
            )
            if signed_dist < -self.max_extrapolation_m:
                return MetricPointResult(
                    x_m, y_m, False, "OUTSIDE_CALIBRATED_GROUND_PLANE"
                )

        if filter_key is not None:
            if filter_key not in self._filters:
                self._filters[filter_key] = MetricKalmanFilter2D(
                    self.process_noise, self.measurement_noise
                )
            x_m, y_m = self._filters[filter_key].update(x_m, y_m)

        return MetricPointResult(x_m, y_m, True, "OK")

    def extract_driver_ground_point(
        self,
        person_bbox: Tuple[float, float, float, float],
        keypoints: Optional[np.ndarray] = None,
        precomputed_foot_px: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Tuple[int, int], bool, str]:
        """
        Determines the driver's ground contact pixel and verifies whether the feet
        are visible or occluded (preventing perspective projection errors).
        """
        x1, y1, x2, y2 = person_bbox
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        aspect_ratio = h / w

        # 1. If COCO 17-keypoint array (shape: [17, 3] -> x, y, conf) is available:
        # Index 15 = left_ankle, Index 16 = right_ankle
        if keypoints is not None and len(keypoints) >= 17:
            l_ankle = keypoints[15]
            r_ankle = keypoints[16]
            l_conf = float(l_ankle[2]) if len(l_ankle) > 2 else 1.0
            r_conf = float(r_ankle[2]) if len(r_ankle) > 2 else 1.0

            valid_ankles = []
            if l_conf >= self.min_ankle_conf and l_ankle[0] > 0 and l_ankle[1] > 0:
                valid_ankles.append((float(l_ankle[0]), float(l_ankle[1])))
            if r_conf >= self.min_ankle_conf and r_ankle[0] > 0 and r_ankle[1] > 0:
                valid_ankles.append((float(r_ankle[0]), float(r_ankle[1])))

            if valid_ankles:
                avg_u = int(np.mean([p[0] for p in valid_ankles]))
                avg_v = int(np.mean([p[1] for p in valid_ankles]))
                return (avg_u, avg_v), True, "OK_ANKLE_KEYPOINTS"

            # Ankles not confident -> likely lower-body occlusion behind forklift/load
            fallback_px = (int((x1 + x2) / 2.0), int(y2))
            return fallback_px, False, "ANKLES_OCCLUDED_OR_LOW_CONF"

        # 2. If precomputed foot point from existing app/logic/geometry.py is provided
        if precomputed_foot_px is not None:
            if aspect_ratio < self.min_bbox_ar:
                return precomputed_foot_px, False, "LOWER_BODY_OCCLUDED_ASPECT_RATIO"
            return precomputed_foot_px, True, "OK_GEOMETRY_FOOT_POINT"

        # 3. Bounding-box bottom-center with aspect-ratio occlusion guard
        bottom_center = (int((x1 + x2) / 2.0), int(y2))
        if aspect_ratio < self.min_bbox_ar:
            return bottom_center, False, "LOWER_BODY_OCCLUDED_BBOX_CLIPPED"
        return bottom_center, True, "OK_BBOX_GROUND_CONTACT"

    def compute_driver_forklift_distance(
        self,
        driver_id: int,
        person_bbox: Tuple[float, float, float, float],
        forklift_id: int,
        vehicle_bbox: Tuple[float, float, float, float],
        keypoints: Optional[np.ndarray] = None,
        precomputed_foot_px: Optional[Tuple[int, int]] = None,
    ) -> DistanceMeasurement:
        """
        Calculates physical distance in meters between the driver's ground contact point
        and the nearest ground-contact point of the forklift's footprint.
        """
        driver_px, driver_vis_ok, vis_reason = self.extract_driver_ground_point(
            person_bbox, keypoints, precomputed_foot_px
        )

        vx1, vy1, vx2, vy2 = vehicle_bbox
        # Sample 3 ground-contact points along the base of the forklift (left tire, center, right tire)
        vehicle_base_samples_px = [
            (int(vx1 + 0.15 * (vx2 - vx1)), int(vy2)),
            (int((vx1 + vx2) / 2.0), int(vy2)),
            (int(vx2 - 0.15 * (vx2 - vx1)), int(vy2)),
        ]
        vehicle_center_px = vehicle_base_samples_px[1]

        driver_proj = self.project_pixel_to_meters(
            driver_px[0], driver_px[1], filter_key=f"drv_{driver_id}"
        )
        veh_center_proj = self.project_pixel_to_meters(
            vehicle_center_px[0], vehicle_center_px[1], filter_key=f"veh_{forklift_id}"
        )

        if not driver_proj.is_certain:
            return DistanceMeasurement(
                distance_m=None,
                is_certain=False,
                uncertainty_reason=driver_proj.uncertainty_reason,
                driver_world_m=None,
                vehicle_world_m=None,
                driver_ground_px=driver_px,
                vehicle_ground_px=vehicle_center_px,
            )

        if not veh_center_proj.is_certain:
            return DistanceMeasurement(
                distance_m=None,
                is_certain=False,
                uncertainty_reason=veh_center_proj.uncertainty_reason,
                driver_world_m=(driver_proj.x_m, driver_proj.y_m),
                vehicle_world_m=None,
                driver_ground_px=driver_px,
                vehicle_ground_px=vehicle_center_px,
            )

        # Compute minimum Euclidean distance in meters to the forklift's ground contact segment
        min_dist_m = float("inf")
        best_veh_world = (veh_center_proj.x_m, veh_center_proj.y_m)
        best_veh_px = vehicle_center_px

        for idx, (vu, vv) in enumerate(vehicle_base_samples_px):
            sample_proj = self.project_pixel_to_meters(
                vu, vv, filter_key=f"veh_{forklift_id}_pt{idx}"
            )
            if sample_proj.x_m is not None and sample_proj.y_m is not None:
                d = float(
                    np.hypot(
                        driver_proj.x_m - sample_proj.x_m,
                        driver_proj.y_m - sample_proj.y_m,
                    )
                )
                if d < min_dist_m:
                    min_dist_m = d
                    best_veh_world = (sample_proj.x_m, sample_proj.y_m)
                    best_veh_px = (vu, vv)

        is_overall_certain = driver_vis_ok and driver_proj.is_certain and veh_center_proj.is_certain
        reason = "OK" if is_overall_certain else vis_reason

        return DistanceMeasurement(
            distance_m=round(min_dist_m, 2) if is_overall_certain else None,
            is_certain=is_overall_certain,
            uncertainty_reason=reason,
            driver_world_m=(round(driver_proj.x_m, 2), round(driver_proj.y_m, 2)),
            vehicle_world_m=(round(best_veh_world[0], 2), round(best_veh_world[1], 2)),
            driver_ground_px=driver_px,
            vehicle_ground_px=best_veh_px,
        )