# app/vehicle_monitoring/distance_estimator.py
"""
Metric distance between a driver's ground contact point and the forklift's GROUND FOOTPRINT.

Only points that lie on the floor may go through the ground homography. The old estimator also projected points at
75 % of the vehicle box height (on the vehicle body, above the floor): the homography maps such points far behind
the vehicle, so the "footprint" was distorted and distances were wrong.

Footprint model:
  * with a segmentation mask (recommended, -seg vehicle model): the LOWER BOUNDARY of the mask is where the chassis
    touches the floor - the two bottom edges that face the camera. Projected to the floor they form an "L"; its
    bounding rectangle is the real footprint in true size (no assumed depth). Seen exactly from the side only one edge
    is visible: then the rectangle is completed away from the camera with the vehicle's width / length;
  * with a box only: near edge = bbox bottom edge (projected exactly), extended away from the camera by a depth
    chosen from the visible edge length and the vehicle size (a long visible side means the short dimension is the
    depth, and vice versa) - a generous polygon, distances err on the small side by a fraction of a metre.

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


def _inside_polygon(p: np.ndarray, poly: np.ndarray) -> bool:
    """Even-odd rule (pure numpy: this runs for every person x vehicle x frame)."""
    x, y = float(p[0]), float(p[1])
    x1, y1 = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    crosses = (y1 > y) != (y2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
    return bool(np.count_nonzero(crosses & (x < xi)) % 2)


def _nearest_on_boundary(p: np.ndarray, poly: np.ndarray) -> Tuple[float, Tuple[float, float]]:
    a = poly
    ab = np.roll(poly, -1, axis=0) - a
    denom = np.maximum((ab * ab).sum(axis=1), 1e-12)
    t = np.clip(((p - a) * ab).sum(axis=1) / denom, 0.0, 1.0)
    c = a + t[:, None] * ab
    d = np.hypot(p[0] - c[:, 0], p[1] - c[:, 1])
    i = int(np.argmin(d))
    return float(d[i]), (float(c[i, 0]), float(c[i, 1]))


def signed_gap(pt: Sequence[float], poly: np.ndarray) -> float:
    """Metres from `pt` to the polygon boundary: positive outside, negative inside."""
    poly = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    if len(poly) < 3:
        return float("inf")
    p = np.asarray(pt, dtype=np.float64)[:2]
    d, _ = _nearest_on_boundary(p, poly)
    return -d if _inside_polygon(p, poly) else d


def point_polygon_distance(pt: Sequence[float], poly: np.ndarray) -> Tuple[float, Tuple[float, float]]:
    """Distance from `pt` to polygon `poly` (Kx2); 0 inside. Returns (distance, nearest point)."""
    p = np.asarray(pt, dtype=np.float64)[:2]
    poly = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    if len(poly) >= 3 and _inside_polygon(p, poly):
        return 0.0, (float(p[0]), float(p[1]))
    return _nearest_on_boundary(p, poly)


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
        # real size of the site's forklifts (forks included in the length): completes the footprint when only one
        # side is visible, and bounds implausible mask outlines
        self.vehicle_length_m = float(dist_cfg.get("vehicle_length_m", 3.4))
        self.vehicle_width_m = float(dist_cfg.get("vehicle_width_m", 1.25))
        self.fallback_scale = float(cal_cfg.get("fallback_meters_per_pixel", 0.012))

        self.homography_matrix: Optional[np.ndarray] = None
        self.inv_homography_matrix: Optional[np.ndarray] = None
        self._smoothed: Dict[str, Tuple[float, float]] = {}
        self._fp_fit: Dict[str, Tuple[int, float]] = {}      # vehicle key -> (orientation deg, cost) of the last fit
        self._fp_last: Dict[str, Tuple[np.ndarray, Tuple[float, float], int]] = {}   # last fitted rectangle
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
        for k in [k for k in self._fp_fit if k.startswith(key_prefix)]:
            del self._fp_fit[k]
        for k in [k for k in self._fp_last if k.startswith(key_prefix)]:
            del self._fp_last[k]

    def pixel_to_world(self, pt_px: Sequence[float]) -> Tuple[float, float]:
        H = self.homography_matrix
        if H is not None:
            x, y = float(pt_px[0]), float(pt_px[1])
            w = H[2, 0] * x + H[2, 1] * y + H[2, 2]
            w = w if abs(w) > 1e-12 else 1e-12
            return (float((H[0, 0] * x + H[0, 1] * y + H[0, 2]) / w),
                    float((H[1, 0] * x + H[1, 1] * y + H[1, 2]) / w))
        return float(pt_px[0]) * self.fallback_scale, float(pt_px[1]) * self.fallback_scale

    def world_to_pixel(self, xy: Sequence[float]) -> Optional[Tuple[int, int]]:
        if self.inv_homography_matrix is None:
            if self.homography_matrix is None:
                return int(xy[0] / self.fallback_scale), int(xy[1] / self.fallback_scale)
            return None
        G = self.inv_homography_matrix
        x, y = float(xy[0]), float(xy[1])
        w = G[2, 0] * x + G[2, 1] * y + G[2, 2]
        w = w if abs(w) > 1e-12 else 1e-12
        return int((G[0, 0] * x + G[0, 1] * y + G[0, 2]) / w), int((G[1, 0] * x + G[1, 1] * y + G[1, 2]) / w)

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
    @staticmethod
    def _lower_boundary(contour: np.ndarray, max_bins: int = 48) -> np.ndarray:
        """
        Lowest outline point per image column band = the chassis / floor contact line. The outline is densified
        first and the bands are wide enough that EVERY band contains points of the lower side (otherwise a band
        with only top-edge points would put the "floor contact" on the roof).
        """
        c = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
        nxt = np.roll(c, -1, axis=0)
        parts = [c]
        for a, b in zip(c, nxt):
            n = int(np.hypot(*(b - a)) // 1.5)
            if n > 0:
                t = np.linspace(0.0, 1.0, n + 2)[1:-1, None]
                parts.append(a + t * (b - a))
        d = np.vstack(parts)
        x0, x1 = float(d[:, 0].min()), float(d[:, 0].max())
        if x1 - x0 < 12.0:
            return np.zeros((0, 2))
        n_bins = int(np.clip((x1 - x0) / 5.0, 6, max_bins))
        bins = np.clip(((d[:, 0] - x0) / (x1 - x0) * n_bins).astype(int), 0, n_bins - 1)
        low = []
        for b in range(n_bins):
            idx = np.flatnonzero(bins == b)
            if len(idx):
                low.append(d[idx[np.argmax(d[idx, 1])]])
        return np.asarray(low) if len(low) >= 5 else np.zeros((0, 2))

    def _away_direction(self, bbox: Sequence[float]) -> np.ndarray:
        """Unit vector on the floor pointing away from the camera at the vehicle."""
        x1, y1, x2, y2 = (float(v) for v in bbox[:4])
        cx, h = 0.5 * (x1 + x2), max(1.0, y2 - y1)
        c0 = np.array(self.pixel_to_world((cx, y2)))
        c1 = np.array(self.pixel_to_world((cx, y2 - max(4.0, 0.05 * h))))
        away = c1 - c0
        n = float(np.hypot(*away))
        return away / n if n > 1e-9 else np.array([0.0, -1.0])

    def _footprint_from_contour(self, contour: np.ndarray, away: np.ndarray, key: Optional[str] = None,
                                expected_centre: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Rectangle of the REAL vehicle size (vehicle_length_m x vehicle_width_m) fitted to the floor-contact line of
        the mask; None when the outline is unusable.

        The lower outline shows the bottom edges that face the camera (one or two sides of the rectangle). Its two
        ends run up the vertical edges of the body, and those points project metres away from the camera - so the
        fit is robust: for every orientation the rectangle is anchored at the points NEAREST to the camera (which are
        never such outliers) and scored by how many outline points lie on its two camera-facing sides.
        """
        low = self._lower_boundary(contour)
        if len(low) < 5:
            return None
        pts = np.array([self.pixel_to_world(p) for p in low], dtype=np.float64)
        L, W = self.vehicle_length_m, self.vehicle_width_m
        tau = 0.25
        best = None
        # a tracked vehicle turns a few degrees per frame: search around the previous orientation first
        prev = self._fp_fit.get(key) if key else None
        rounds = ([[(prev[0] + d) % 180 for d in range(-12, 13, 3)]] if prev else []) + [list(range(0, 180, 3))]
        for degs in rounds:
            best = self._fit_rectangle(pts, away, degs, L, W, tau, expected_centre)
            if best is not None and (prev is None or degs is rounds[-1]
                                     or best[0] <= max(0.2 * len(pts), 1.5 * prev[1] + 1.0)):
                break
        if best is None or best[0] > 0.6 * len(pts):
            if key:
                self._fp_fit.pop(key, None)
            return None                               # the outline does not look like this vehicle: box model
        cost, u, n, u0, n0, len_u, len_n, deg = best
        if key:
            self._fp_fit[key] = (deg, cost)
        o = u0 * u + n0 * n
        return np.array([o, o + len_u * u, o + len_u * u + len_n * n, o + len_n * n], dtype=np.float64)

    @staticmethod
    def _fit_rectangle(pts: np.ndarray, away: np.ndarray, degs, L: float, W: float, tau: float,
                       expected_centre: Optional[np.ndarray] = None):
        """Best (cost, u, n, u0, n0, len_u, len_n, deg). `expected_centre` (where the rectangle of the previous frame
        would be now) breaks ties between equally good rectangles: seen end-on, the outline does not say whether the
        vehicle extends this way or that, and a fit that flips between the two looks like a vehicle jumping."""
        best = None
        for deg in degs:
            t = np.radians(deg)
            u = np.array([np.cos(t), np.sin(t)])
            n = np.array([-u[1], u[0]])
            if u[0] * away[0] + u[1] * away[1] < 0:
                u = -u
            if n[0] * away[0] + n[1] * away[1] < 0:
                n = -n
            # (explicit 2-D products: this runs ~100 times per vehicle and frame)
            su = pts[:, 0] * u[0] + pts[:, 1] * u[1]
            sn = pts[:, 0] * n[0] + pts[:, 1] * n[1]
            u0, n0 = float(np.partition(su, 1)[1]), float(np.partition(sn, 1)[1])   # nearest (robust minimum)
            ru, rn = su - u0, sn - n0
            for len_u, len_n in ((L, W), (W, L)):
                d_a = np.hypot(rn, np.maximum(0.0, np.maximum(-ru, ru - len_u)))   # side along u (at n = n0)
                d_b = np.hypot(ru, np.maximum(0.0, np.maximum(-rn, rn - len_n)))   # side along n (at u = u0)
                on_a, on_b = d_a <= tau, d_b <= tau
                cost = float(np.minimum(np.minimum(d_a, d_b), tau).sum()) / tau
                # A visible side is seen along its WHOLE length: a short side must not be explained as a long one.
                # (A side counts as visible when points lie on it that are not just the shared corner.)
                if (on_a & ~on_b).sum() >= 2:
                    cost += 0.35 * len(pts) * max(0.0, 1.0 - float(ru[on_a].max()) / len_u)
                if (on_b & ~on_a).sum() >= 2:
                    cost += 0.35 * len(pts) * max(0.0, 1.0 - float(rn[on_b].max()) / len_n)
                if expected_centre is not None:
                    cu, cn = u0 + 0.5 * len_u, n0 + 0.5 * len_n
                    cx, cy = cu * u[0] + cn * n[0], cu * u[1] + cn * n[1]
                    jump = float(np.hypot(cx - expected_centre[0], cy - expected_centre[1]))
                    cost += 0.12 * len(pts) * min(1.5, max(0.0, jump - 0.15))
                if best is None or cost < best[0]:
                    best = (cost, u, n, u0, n0, len_u, len_n, int(deg))
        return best

    def vehicle_footprint(
        self, bbox: Sequence[float], frame_shape: Sequence[int], contour: Optional[np.ndarray] = None,
        key: Optional[str] = None,
    ) -> Tuple[np.ndarray, bool]:
        """(ground footprint polygon in meters, near edge clipped by the frame border). `key` = vehicle id (lets
        the fit start from the orientation found in the previous frame)."""
        x1, y1, x2, y2 = (float(v) for v in bbox[:4])
        h_frame = int(frame_shape[0])
        clipped = y2 >= h_frame - self.border_margin_px
        away = self._away_direction(bbox)
        if contour is not None and len(contour) >= 3 and self.homography_matrix is not None:
            base = self.pixel_to_world((0.5 * (x1 + x2), y2))
            last = self._fp_last.get(key) if key else None
            expected = None
            if last is not None:      # the previous rectangle, carried along with the box
                expected = last[0].mean(axis=0) + np.array([base[0] - last[1][0], base[1] - last[1][1]])
            fp = self._footprint_from_contour(contour, away, key, expected)
            if fp is not None:
                if key:
                    self._fp_last[key] = (fp, base, 0)
                return fp, bool(clipped)
            if last is not None and last[2] < 5:
                # one unusable outline (occluded / noisy mask): keep the rectangle of the previous frames, moved
                # with the box, instead of jumping to the coarser box model and back
                shift = np.array([base[0] - last[1][0], base[1] - last[1][1]])
                self._fp_last[key] = (last[0], last[1], last[2] + 1)
                return last[0] + shift, bool(clipped)
        near_px = [(x1, y2), (0.5 * (x1 + x2), y2), (x2, y2)]
        near_w = np.array([self.pixel_to_world(p) for p in near_px], dtype=np.float64)
        depth = self.vehicle_depth_m
        if self.homography_matrix is not None:
            near_len = float(np.hypot(*(near_w[-1] - near_w[0])))
            if near_len > 0.2:   # long visible side -> the depth is the short dimension, and vice versa
                depth = float(np.clip(self.vehicle_length_m * self.vehicle_width_m / near_len,
                                      self.vehicle_width_m, self.vehicle_length_m))
        far_w = near_w + depth * away
        hull = np.array([near_w[0], near_w[-1], far_w[-1], far_w[0]], dtype=np.float64)
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
