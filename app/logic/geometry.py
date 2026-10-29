# app/logic/geometry.py
from __future__ import annotations
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple


class MetricBEVProjector:
    """
    Projects 2D image coordinates into real-world Bird's-Eye View (BEV) metric coordinates (meters).
    Eliminates perspective distortion across cameras with different tilt angles and focal lengths.
    """

    def __init__(
        self,
        src_image_pts: Optional[List[List[float]]] = None,
        dst_meter_pts: Optional[List[List[float]]] = None,
        default_px_per_meter: float = 100.0,
    ):
        self.H: Optional[np.ndarray] = None
        self.H_inv: Optional[np.ndarray] = None
        self.default_scale = 1.0 / max(default_px_per_meter, 1.0)

        if src_image_pts is not None and dst_meter_pts is not None:
            src = np.array(src_image_pts, dtype=np.float32)
            dst = np.array(dst_meter_pts, dtype=np.float32)
            if src.shape == (4, 2) and dst.shape == (4, 2):
                self.H, _ = cv2.findHomography(src, dst, method=0)
                if self.H is not None:
                    self.H_inv = np.linalg.inv(self.H)

    @staticmethod
    def estimate_ground_contact_point(
        bbox_xyxy: np.ndarray,
        keypoints_xy: Optional[np.ndarray],
        keypoints_conf: Optional[np.ndarray],
        kpt_thresh: float = 0.45,
    ) -> Tuple[float, float, bool]:
        """
        Estimates the true floor contact point (u, v) even when legs are occluded by machinery.
        Returns: (u_ground, v_ground, is_extrapolated)
        COCO indices: 5:L_Shoulder, 6:R_Shoulder, 11:L_Hip, 12:R_Hip, 13:L_Knee, 14:R_Knee, 15:L_Ankle, 16:R_Ankle
        """
        x1, y1, x2, y2 = bbox_xyxy
        u_center = 0.5 * (x1 + x2)

        if keypoints_xy is None or keypoints_conf is None or len(keypoints_conf) < 17:
            return float(u_center), float(y2), True

        # 1. Best case: At least one ankle is visible
        ankle_vis = [i for i in (15, 16) if keypoints_conf[i] >= kpt_thresh and keypoints_xy[i][1] > 0]
        if ankle_vis:
            pts = keypoints_xy[ankle_vis]
            return float(np.mean(pts[:, 0])), float(np.max(pts[:, 1])), False

        # 2. Machinery occlusion case: Hips and shoulders are visible, but legs are hidden behind machinery
        sh_vis = [i for i in (5, 6) if keypoints_conf[i] >= kpt_thresh and keypoints_xy[i][1] > 0]
        hip_vis = [i for i in (11, 12) if keypoints_conf[i] >= kpt_thresh and keypoints_xy[i][1] > 0]
        if sh_vis and hip_vis:
            sh_mid = np.mean(keypoints_xy[sh_vis], axis=0)
            hip_mid = np.mean(keypoints_xy[hip_vis], axis=0)
            torso_len = np.linalg.norm(hip_mid - sh_mid)
            if torso_len > 8.0:
                # Anthropometric ratio: Hip-to-Ankle is ~1.75x Shoulder-to-Hip vertical distance
                extrapolated_v = float(hip_mid[1] + 1.75 * torso_len)
                return float(hip_mid[0]), max(extrapolated_v, float(y2)), True

        # 3. Fallback: Bottom-center of bounding box
        return float(u_center), float(y2), True

    def to_bev_meters(self, u: float, v: float) -> np.ndarray:
        """Converts image (u, v) ground contact point into BEV (X, Y) in meters."""
        if self.H is None:
            # Perspective-approximated fallback when 4-point calibration is absent
            return np.array([u * self.default_scale, v * self.default_scale], dtype=np.float32)

        pt = np.array([[[u, v]]], dtype=np.float32)
        bev_pt = cv2.perspectiveTransform(pt, self.H)[0, 0]
        return bev_pt.astype(np.float32)


def compute_pairwise_iou_matrix(boxes_xyxy: np.ndarray) -> np.ndarray:
    """
    Vectorized pairwise IoU matrix for N bounding boxes.
    boxes_xyxy: shape (N, 4)
    Returns: (N, N) matrix with zeroed diagonal.
    """
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
    max_iou_thresh: float = 0.15,
    kpt_thresh: float = 0.45,
) -> np.ndarray:
    """
    Determines which detections are contaminated by multi-person intersections.
    Flags a detection if:
      1. Its pairwise bbox IoU with ANY other person exceeds `max_iou_thresh`, OR
      2. Another person's confident keypoints fall inside its bounding box.
    Returns boolean array of shape (N,) where True = CONTAMINATED (Reject Gallery Update).
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
            for j in range(N):
                if i == j:
                    continue
                valid_kpts = keypoints_xy[j][keypoints_conf[j] >= kpt_thresh]
                if len(valid_kpts) == 0:
                    continue
                inside = (
                    (valid_kpts[:, 0] >= x1)
                    & (valid_kpts[:, 0] <= x2)
                    & (valid_kpts[:, 1] >= y1)
                    & (valid_kpts[:, 1] <= y2)
                )
                if np.any(inside):
                    contaminated[i] = True
                    break

    return contaminated