# app/vehicle_monitoring/distance_estimator.py
"""
Metric distance between a driver's ground contact point and the forklift's GROUND FOOTPRINT.

Only points that lie on the floor may go through the ground homography. The old estimator also projected points at
75 % of the vehicle box height (on the vehicle body, above the floor): the homography maps such points far behind
the vehicle, so the "footprint" was distorted and distances were wrong.

Footprint model:
  * near edge = the vehicle's lowest image points (bbox bottom edge, or the lower band of the segmentation mask when
    the vehicle model is a -seg model) -> these touch the floor and are projected exactly;
  * the footprint extends `vehicle_depth_m` away from the camera (direction obtained from the homography itself).
The result is a slightly generous polygon (bbox corners are not always on the vehicle), i.e. distances err on the
small side by a fraction of a metre; with a segmentation model the near edge follows the real chassis.

Driver ground point: visible ankle keypoints (certain) -> pose-based foot point from app/logic/geometry.py
(uncertain when it was extrapolated from the torso) -> bbox bottom (uncertain when the box looks truncated).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class DistanceMeasurement:
    distance_m: Optional[float]
    driver_world_xy: Optional[Tuple[float, float]]
    vehicle_world_xy: Optional[Tuple[float, float]]
    driver_ground_px: Tuple[int, int]
    vehicle_nearest_px: Tuple[int, int]
    is_uncertain: bool
    uncertainty_reason: str = ""


def point_polygon_distance(pt: Sequence[float], poly: np.ndarray) -> Tuple[float, Tuple[float, float]]:
    """Distance from `pt` to polygon `poly` (Kx2); 0 inside. Returns (distance, nearest point)."""
    p = np.asarray(pt, dtype=np.float64)
    poly = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    if len(poly) >= 3 and cv2.pointPolygonTest(poly.astype(np.float32), (float(p[0]), float(p[1])), False) >= 0:
        return 0.0, (float(p[0]), float(p[1]))
    best, best_pt = float("inf"), (float(poly[0][0]), float(poly[0][1]))
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ab = b - a
        denom = float(np.dot(ab, ab))
        t = 0.0 if denom <= 1e-12 else float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
        c = a + t * ab
        d = float(np.hypot(*(p - c)))
        if d < best:
            best, best_pt = d, (float(c[0]), float(c[1]))
    return best, best_pt


class RealWorldDistanceEstimator:
    def __init__(self, config: dict):
        dist_cfg = config.get("distance", {}) or {}
        cal_cfg = config.get("calibration", {}) or {}

        self.alpha = float(dist_cfg.get("ema_smoothing_alpha", 0.35))
        self.max_step_jump_m = float(dist_cfg.get("max_step_jump_m", 1.8))
        self.min_aspect_ratio = float(dist_cfg.get("min_bbox_aspect_ratio", 1.15))
        self.min_ankle_conf = float(dist_cfg.get("min_ankle_confidence", 0.35))
        self.border_margin_px = int(dist_cfg.get("border_margin_px", 12))
        self.vehicle_depth_m = float(dist_cfg.get("vehicle_depth_m", 2.5))
        self.fallback_scale = float(cal_cfg.get("fallback_meters_per_pixel", 0.012))

        self.homography_matrix: Optional[np.ndarray] = None
        self.inv_homography_matrix: Optional[np.ndarray] = None
        self._smoothed: Dict[str, Tuple[float, float]] = {}
        img_pts = np.array(cal_cfg.get("image_points_px", []) or [], dtype=np.float32)
        world_pts = np.array(cal_cfg.get("world_points_m", []) or [], dtype=np.float32)
        if len(img_pts) >= 4 and len(img_pts) == len(world_pts):
            H, _ = cv2.findHomography(img_pts, world_pts, method=0)
            self.set_homography(H)

    # ------------------------------------------------------------------
    def set_homography(self, H: Optional[np.ndarray]) -> None:
        """Single source of truth for pixel -> meters (main.py hands over the identity projector's matrix)."""
        if H is None:
            return
        self.homography_matrix = np.asarray(H, dtype=np.float64)
        try:
            self.inv_homography_matrix = np.linalg.inv(self.homography_matrix)
        except np.linalg.LinAlgError:
            self.inv_homography_matrix = None
        self._smoothed.clear()

    @property
    def is_metric(self) -> bool:
        return self.homography_matrix is not None

    def forget(self, key_prefix: str) -> None:
        for k in [k for k in self._smoothed if k.startswith(key_prefix)]:
            del self._smoothed[k]

    def pixel_to_world(self, pt_px: Sequence[float]) -> Tuple[float, float]:
        if self.homography_matrix is not None:
            pts = np.array([[[float(pt_px[0]), float(pt_px[1])]]], dtype=np.float64)
            w = cv2.perspectiveTransform(pts, self.homography_matrix)[0][0]
            return float(w[0]), float(w[1])
        return float(pt_px[0]) * self.fallback_scale, float(pt_px[1]) * self.fallback_scale

    def world_to_pixel(self, xy: Sequence[float]) -> Optional[Tuple[int, int]]:
        if self.inv_homography_matrix is None:
            if self.homography_matrix is None:
                return int(xy[0] / self.fallback_scale), int(xy[1] / self.fallback_scale)
            return None
        p = cv2.perspectiveTransform(np.array([[[float(xy[0]), float(xy[1])]]], np.float64),
                                     self.inv_homography_matrix)[0][0]
        return int(p[0]), int(p[1])

    def _stabilize(self, key: str, raw: Tuple[float, float]) -> Tuple[float, float]:
        prev = self._smoothed.get(key)
        if prev is None:
            self._smoothed[key] = raw
            return raw
        dx, dy = raw[0] - prev[0], raw[1] - prev[1]
        step = float(np.hypot(dx, dy))
        if step > self.max_step_jump_m > 0:
            s = self.max_step_jump_m / step
            raw = (prev[0] + dx * s, prev[1] + dy * s)
        out = (self.alpha * raw[0] + (1 - self.alpha) * prev[0], self.alpha * raw[1] + (1 - self.alpha) * prev[1])
        self._smoothed[key] = out
        return out

    # ------------------------------------------------------------------
    def vehicle_footprint(
        self, bbox: Sequence[float], frame_shape: Sequence[int], contour: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, bool]:
        """(ground footprint polygon in meters, near edge clipped by the frame border)."""
        x1, y1, x2, y2 = (float(v) for v in bbox[:4])
        h_frame = int(frame_shape[0])
        h = max(1.0, y2 - y1)
        near_px: List[Tuple[float, float]] = []
        if contour is not None and len(contour) >= 3:
            c = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
            band = c[c[:, 1] >= y2 - 0.12 * h]
            if len(band) >= 2:
                near_px = [tuple(p) for p in band]
        if not near_px:
            near_px = [(x1, y2), (0.5 * (x1 + x2), y2), (x2, y2)]
        near_w = np.array([self.pixel_to_world(p) for p in near_px], dtype=np.float64)

        cx = 0.5 * (x1 + x2)
        c0 = np.array(self.pixel_to_world((cx, y2)))
        c1 = np.array(self.pixel_to_world((cx, y2 - max(4.0, 0.05 * h))))
        away = c1 - c0
        n = float(np.hypot(*away))
        away = away / n if n > 1e-9 else np.array([0.0, -1.0])
        far_w = near_w + self.vehicle_depth_m * away
        pts = np.vstack([near_w, far_w]).astype(np.float32)
        hull = cv2.convexHull(pts).reshape(-1, 2).astype(np.float64)
        clipped = y2 >= h_frame - self.border_margin_px
        return hull, bool(clipped)

    def driver_ground_point(
        self,
        driver_bbox: Sequence[float],
        frame_shape: Sequence[int],
        keypoints: Optional[np.ndarray] = None,
        foot_px: Optional[Sequence[float]] = None,
        ground_uncertain: Optional[bool] = None,
    ) -> Tuple[Tuple[int, int], bool, str]:
        h_frame = int(frame_shape[0])
        x1, y1, x2, y2 = (float(v) for v in driver_bbox[:4])
        ankles_hidden = False
        if keypoints is not None:
            k = np.asarray(keypoints)
            if k.ndim == 2 and len(k) >= 17:
                pts = []
                for i in (15, 16):
                    conf = float(k[i][2]) if k.shape[1] >= 3 else 1.0
                    if conf >= self.min_ankle_conf and k[i][0] > 1 and k[i][1] > 1:
                        pts.append((float(k[i][0]), float(k[i][1])))
                if pts:
                    g = (int(np.mean([p[0] for p in pts])), int(max(p[1] for p in pts)))
                    if g[1] >= h_frame - self.border_margin_px:
                        return g, True, "ANKLES_CLIPPED_BY_BORDER"
                    return g, False, ""
                ankles_hidden = True
                if foot_px is None:
                    return (int(0.5 * (x1 + x2)), int(y2)), True, "ANKLES_OCCLUDED_LOW_POSE_CONF"
        aspect = max(1.0, y2 - y1) / max(1.0, x2 - x1)
        if foot_px is not None:
            g = (int(foot_px[0]), int(foot_px[1]))
            if ground_uncertain or ankles_hidden:
                return g, True, "FEET_HIDDEN_ESTIMATED_FROM_TORSO"
            if ground_uncertain is None and aspect < self.min_aspect_ratio:
                return g, True, "LOWER_BODY_OCCLUDED"
            if g[1] >= h_frame - self.border_margin_px:
                return g, True, "FEET_CLIPPED_BY_FRAME_BORDER"
            return g, False, ""
        g = (int(0.5 * (x1 + x2)), int(y2))
        if g[1] >= h_frame - self.border_margin_px:
            return g, True, "FEET_CLIPPED_BY_FRAME_BORDER"
        if aspect < self.min_aspect_ratio:
            return g, True, "LOWER_BODY_OCCLUDED_TRUNCATED_BBOX"
        return g, False, ""

    def estimate_driver_vehicle_distance(
        self,
        driver_id: str,
        driver_bbox: Sequence[float],
        vehicle_id: str,
        vehicle_bbox: Sequence[float],
        frame_shape: Sequence[int],
        keypoints: Optional[np.ndarray] = None,
        foot_px: Optional[Sequence[float]] = None,
        other_bboxes: Optional[List[List[float]]] = None,
        ground_uncertain: Optional[bool] = None,
        vehicle_contour: Optional[np.ndarray] = None,
    ) -> DistanceMeasurement:
        g_px, uncertain, reason = self.driver_ground_point(driver_bbox, frame_shape, keypoints, foot_px,
                                                           ground_uncertain)
        footprint, clipped = self.vehicle_footprint(vehicle_bbox, frame_shape, vehicle_contour)
        if clipped and not uncertain:
            uncertain, reason = True, "VEHICLE_CLIPPED_BY_BORDER"
        driver_xy = self._stabilize(f"driver_{driver_id}", self.pixel_to_world(g_px))
        dist, nearest = point_polygon_distance(driver_xy, footprint)
        nearest_px = self.world_to_pixel(nearest)
        if nearest_px is None:
            nearest_px = (int(0.5 * (vehicle_bbox[0] + vehicle_bbox[2])), int(vehicle_bbox[3]))
        return DistanceMeasurement(
            distance_m=round(float(dist), 2),
            driver_world_xy=(float(driver_xy[0]), float(driver_xy[1])),
            vehicle_world_xy=nearest,
            driver_ground_px=(int(g_px[0]), int(g_px[1])),
            vehicle_nearest_px=nearest_px,
            is_uncertain=bool(uncertain),
            uncertainty_reason=reason,
        )
