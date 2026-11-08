# app/logic/geometry.py
from __future__ import annotations
from typing import List, Optional, Tuple
import cv2
import numpy as np


class MetricBEVProjector:
    """
    Hybrid Homography + Pose-Corrected Anthropometric Metric Projector.
    Uses bone-length priors (shoulder-to-hip & limb lengths) when a worker crouches
    so bending over never distorts real-world metric distance calculations.
    """

    def __init__(
        self,
        src_image_pts: Optional[List[List[float]]] = None,
        dst_meter_pts: Optional[List[List[float]]] = None,
        default_px_per_meter: float = 110.0,
        stream_wh: Optional[Tuple[int, int]] = None,
        calib_ref_wh: Optional[Tuple[int, int]] = None,
    ):
        self.H: Optional[np.ndarray] = None
        self.default_scale = 1.0 / max(default_px_per_meter, 1.0)

        if src_image_pts is not None and dst_meter_pts is not None:
            src = np.array(src_image_pts, dtype=np.float32)
            dst = np.array(dst_meter_pts, dtype=np.float32)

            if stream_wh is not None and calib_ref_wh is not None:
                sw, sh = stream_wh
                rw, rh = calib_ref_wh
                if np.max(src[:, 0]) <= rw and sw > rw * 1.2:
                    src[:, 0] *= float(sw) / float(rw)
                    src[:, 1] *= float(sh) / float(rh)

            if src.shape == (4, 2) and dst.shape == (4, 2):
                self.H, _ = cv2.findHomography(src, dst, method=0)

    @staticmethod
    def estimate_standing_height_px(
        bbox_xyxy: np.ndarray,
        keypoints_xy: Optional[np.ndarray],
        keypoints_conf: Optional[np.ndarray],
        kpt_thresh: float = 0.30,
    ) -> float:
        """
        Estimates the worker's true upright standing height in pixels even when crouching or bending.
        Prevents bbox height shrinkage during crouching from corrupting metric scale.
        """
        x1, y1, x2, y2 = bbox_xyxy
        bw = float(x2 - x1)
        bh = max(20.0, float(y2 - y1))

        # If crouching/bending (wide aspect ratio), estimate standing height from skeleton bone lengths or max(bh, 1.8*bw)
        skeletal_est = bh
        if keypoints_xy is not None and keypoints_conf is not None and len(keypoints_conf) >= 17:
            sh_vis = [i for i in (5, 6) if keypoints_conf[i] >= kpt_thresh]
            hip_vis = [i for i in (11, 12) if keypoints_conf[i] >= kpt_thresh]
            if sh_vis and hip_vis:
                sh_pt = np.mean(keypoints_xy[sh_vis], axis=0)
                hip_pt = np.mean(keypoints_xy[hip_vis], axis=0)
                torso_px = float(np.linalg.norm(hip_pt - sh_pt))
                # Standing human height is ~3.3x shoulder-to-hip length
                skeletal_est = max(bh, torso_px * 3.3)

        return float(np.clip(max(bh, skeletal_est, bw * 1.65), bh, bh * 1.85))

    @staticmethod
    def estimate_ground_contact_point(
        bbox_xyxy: np.ndarray,
        keypoints_xy: Optional[np.ndarray],
        keypoints_conf: Optional[np.ndarray],
        kpt_thresh: float = 0.30,
    ) -> Tuple[float, float, bool]:
        x1, y1, x2, y2 = bbox_xyxy
        u_box = 0.5 * float(x1 + x2)
        bh = max(16.0, float(y2 - y1))

        if keypoints_xy is None or keypoints_conf is None or len(keypoints_conf) < 17:
            return u_box, float(y2), True

        ankle_vis = [i for i in (15, 16) if keypoints_conf[i] >= kpt_thresh and keypoints_xy[i][1] > 0]
        if ankle_vis:
            pts = keypoints_xy[ankle_vis]
            u_ank = float(np.mean(pts[:, 0]))
            v_ank = float(np.max(pts[:, 1]))
            u_stable = 0.60 * u_ank + 0.40 * u_box
            v_stable = float(np.clip(v_ank, float(y2) - 0.10 * bh, float(y2) + 0.04 * bh))
            return u_stable, v_stable, False

        sh_vis = [i for i in (5, 6) if keypoints_conf[i] >= kpt_thresh and keypoints_xy[i][1] > 0]
        hip_vis = [i for i in (11, 12) if keypoints_conf[i] >= kpt_thresh and keypoints_xy[i][1] > 0]
        if sh_vis and hip_vis:
            sh_mid = np.mean(keypoints_xy[sh_vis], axis=0)
            hip_mid = np.mean(keypoints_xy[hip_vis], axis=0)
            torso_len = float(np.linalg.norm(hip_mid - sh_mid))
            if torso_len > 6.0:
                extrapolated_v = float(hip_mid[1] + 1.60 * torso_len)
                v_clamped = float(np.clip(extrapolated_v, float(y2), float(y2) + 0.20 * bh))
                u_stable = 0.50 * float(hip_mid[0]) + 0.50 * u_box
                return u_stable, v_clamped, True

        return u_box, float(y2), True

    def to_bev_meters(self, u: float, v: float) -> np.ndarray:
        if self.H is not None:
            vec = np.array([u, v, 1.0], dtype=np.float64)
            proj = self.H @ vec
            if proj[2] > 0.08:
                bx, by = float(proj[0] / proj[2]), float(proj[1] / proj[2])
                if -40.0 <= bx <= 120.0 and -40.0 <= by <= 120.0:
                    return np.array([bx, by], dtype=np.float32)

        return np.array([u * self.default_scale, v * self.default_scale], dtype=np.float32)

    def compute_metric_distance(
        self,
        uv1: Tuple[float, float],
        bev1: np.ndarray,
        h1_px: float,
        uv2: Tuple[float, float],
        bev2: np.ndarray,
        h2_px: float,
    ) -> float:
        avg_h_px = max(28.0, 0.5 * (h1_px + h2_px))
        meters_per_px = 1.72 / avg_h_px
        du = uv1[0] - uv2[0]
        dv = (uv1[1] - uv2[1]) * 1.25
        anthro_dist = float(np.hypot(du, dv) * meters_per_px)

        bev_dist = float(np.linalg.norm(bev1 - bev2))
        if bev_dist > 2.0 * anthro_dist or anthro_dist > 2.0 * bev_dist:
            return anthro_dist
        return 0.35 * bev_dist + 0.65 * anthro_dist


def compute_single_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
    xA = max(float(boxA[0]), float(boxB[0]))
    yA = max(float(boxA[1]), float(boxB[1]))
    xB = min(float(boxA[2]), float(boxB[2]))
    yB = min(float(boxA[3]), float(boxB[3]))
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = max(1.0, float(boxA[2] - boxA[0]) * float(boxA[3] - boxA[1]))
    areaB = max(1.0, float(boxB[2] - boxB[0]) * float(boxB[3] - boxB[1]))
    return float(inter / (areaA + areaB - inter + 1e-6))


def compute_pairwise_iou_matrix(boxes_xyxy: np.ndarray) -> np.ndarray:
    N = len(boxes_xyxy)
    if N <= 1:
        return np.zeros((N, N), dtype=np.float32)

    x1 = boxes_xyxy[:, 0][:, None]
    y1 = boxes_xyxy[:, 1][:, None]
    x2 = boxes_xyxy[:, 2][:, None]
    y2 = boxes_xyxy[:, 3][:, None]

    areas = np.maximum(0.0, (x2 - x1)) * np.maximum(0.0, (y2 - y1))
    xx1 = np.maximum(x1, x1.T)
    yy1 = np.maximum(y1, y1.T)
    xx2 = np.minimum(x2, x2.T)
    yy2 = np.minimum(y2, y2.T)

    inter_w = np.maximum(0.0, xx2 - xx1)
    inter_h = np.maximum(0.0, yy2 - yy1)
    inter_area = inter_w * inter_h

    union_area = areas + areas.T - inter_area + 1e-6
    iou = inter_area / union_area
    np.fill_diagonal(iou, 0.0)
    return iou.astype(np.float32)


def compute_intersection_contamination(
    boxes_xyxy: np.ndarray,
    keypoints_xy: Optional[np.ndarray],
    keypoints_conf: Optional[np.ndarray],
    max_iou_thresh: float = 0.18,
    kpt_thresh: float = 0.40,
) -> np.ndarray:
    """
    Strictly flags any detection that overlaps with another worker (IoU > 0.18)
    or has another worker's skeleton joints intruding inside its bounding box.
    """
    N = len(boxes_xyxy)
    if N <= 1:
        return np.zeros(N, dtype=bool)

    iou_mat = compute_pairwise_iou_matrix(boxes_xyxy)
    contaminated = np.max(iou_mat, axis=1) > max_iou_thresh

    if keypoints_xy is not None and keypoints_conf is not None and len(keypoints_xy) == N:
        for i in range(N):
            if contaminated[i]:
                continue
            x1, y1, x2, y2 = boxes_xyxy[i]
            bw, bh = x2 - x1, y2 - y1
            cx1, cy1 = x1 + 0.10 * bw, y1 + 0.08 * bh
            cx2, cy2 = x2 - 0.10 * bw, y2 - 0.08 * bh

            for j in range(N):
                if i == j:
                    continue
                vis_idx = [k for k in range(17) if keypoints_conf[j, k] >= kpt_thresh]
                if not vis_idx:
                    continue
                pts = keypoints_xy[j, vis_idx]
                inside = (pts[:, 0] >= cx1) & (pts[:, 0] <= cx2) & (pts[:, 1] >= cy1) & (pts[:, 1] <= cy2)
                if np.any(inside):
                    contaminated[i] = True
                    break

    return contaminated