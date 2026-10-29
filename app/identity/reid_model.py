# app/identity/reid_model.py
from __future__ import annotations
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchreid

# Suppress PyTorch weights_only warning from third-party torchreid checkpoint loader
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torchreid")

logger = logging.getLogger("FactoryReID")

# Official torchreid MSMT17 weights for osnet_x1_0 (Google Drive & Direct Mirrors)
MSMT17_GDRIVE_ID = "1KxxvkhV_pnm34VUwtlum-JyvChL_2eP-"
MSMT17_DIRECT_URL = "https://raw.githubusercontent.com/KaiyangZhou/deep-person-reid/master/docs/MODEL_ZOO.md"

SKELETON_LIMBS = [
    (5, 7), (7, 9),    # Left arm
    (6, 8), (8, 10),   # Right arm
    (11, 13), (13, 15),# Left leg
    (12, 14), (14, 16),# Right leg
    (5, 6), (11, 12),  # Shoulder & hip bars
]

IMAGENET_MEAN_BGR = np.array([103.53, 116.28, 123.675], dtype=np.uint8)


@dataclass
class CropQualityReport:
    is_valid_for_reid: bool
    is_valid_for_gallery: bool
    sharpness_score: float
    visible_ratio: float
    pose_view_bin: str
    reject_reason: str = ""


def ensure_msmt17_weights(target_path: str = "weights/osnet_x1_0_msmt17.pt") -> str:
    """
    Guarantees that MSMT17 Person Re-ID metric weights are loaded,
    preventing catastrophic fallback to ImageNet classification weights.
    """
    path = Path(target_path)
    if path.exists() and path.stat().st_size > 5_000_000:
        return str(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading official MSMT17 OSNet Re-ID weights to {path}...")
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={MSMT17_GDRIVE_ID}"
        gdown.download(url, str(path), quiet=False)
    except Exception as e:
        logger.warning(
            f"gdown auto-download failed ({e}). Please manually download the MSMT17 checkpoint from "
            f"the torchreid model zoo (Google Drive ID: {MSMT17_GDRIVE_ID}) and save to {path}."
        )

    if not path.exists() or path.stat().st_size < 5_000_000:
        raise RuntimeError(
            f"CRITICAL ERROR: Failed to acquire MSMT17 weights at {path}. "
            f"Cannot run Zero-Data Person Re-ID with generic ImageNet weights!"
        )
    return str(path)


class PoseGuidedOSNetExtractor:
    """
    Zero-Data Industrial Re-ID Engine using OSNet (osnet_x1_0, MSMT17).
    """

    def __init__(
        self,
        model_name: str = "osnet_x1_0",
        weights_path: Optional[str] = None,
        device: str = "cuda:0",
        kpt_conf_thresh: float = 0.45,
        min_laplacian_var: float = 45.0,
        head_branch_weight: float = 0.45,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.kpt_conf_thresh = kpt_conf_thresh
        self.min_laplacian_var = min_laplacian_var
        self.head_weight = head_branch_weight
        self.body_weight = 1.0 - head_branch_weight

        # Enforce MSMT17 Re-ID checkpoint path
        valid_weights_path = weights_path if (weights_path and Path(weights_path).exists()) else ensure_msmt17_weights()
        logger.info(f"Loading true Person Re-ID MSMT17 weights from: {valid_weights_path}")

        self.extractor = torchreid.utils.FeatureExtractor(
            model_name=model_name,
            model_path=valid_weights_path,
            device=str(self.device),
        )

    @staticmethod
    def classify_pose_view(kpts_xy: np.ndarray, kpts_conf: np.ndarray, thresh: float = 0.45) -> str:
        if kpts_conf[5] >= thresh and kpts_conf[6] >= thresh:
            dx = kpts_xy[5, 0] - kpts_xy[6, 0]
            torso_w = abs(dx)
            face_vis = sum(1 for i in (0, 1, 2) if kpts_conf[i] >= thresh)

            if torso_w < 18.0:
                sh_mid_x = 0.5 * (kpts_xy[5, 0] + kpts_xy[6, 0])
                if kpts_conf[0] >= thresh:
                    return "RIGHT" if kpts_xy[0, 0] > sh_mid_x else "LEFT"
                return "LEFT" if kpts_conf[5] > kpts_conf[6] else "RIGHT"

            if dx > 0 and face_vis >= 1:
                return "FRONT"
            if dx < 0 and face_vis == 0:
                return "BACK"
            return "FRONT" if face_vis >= 2 else "BACK"
        return "FRONT"

    def build_pose_silhouette_mask(
        self,
        crop_shape: Tuple[int, int, int],
        local_kpts_xy: np.ndarray,
        kpts_conf: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        h, w = crop_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        th = self.kpt_conf_thresh

        limb_radius = max(int(w * 0.14), 8)
        head_radius = max(int(w * 0.20), 12)

        head_pts = [local_kpts_xy[i] for i in range(5) if kpts_conf[i] >= th]
        if head_pts:
            center = np.mean(head_pts, axis=0).astype(np.int32)
            cv2.circle(mask, tuple(center), head_radius, 255, -1)
        elif kpts_conf[5] >= th and kpts_conf[6] >= th:
            sh_mid = 0.5 * (local_kpts_xy[5] + local_kpts_xy[6])
            sh_dist = np.linalg.norm(local_kpts_xy[5] - local_kpts_xy[6])
            est_head = np.array([sh_mid[0], max(0, sh_mid[1] - 0.65 * sh_dist)], dtype=np.int32)
            cv2.circle(mask, tuple(est_head), head_radius, 255, -1)

        torso_indices = [i for i in (5, 6, 12, 11) if kpts_conf[i] >= th]
        if len(torso_indices) >= 3:
            torso_poly = local_kpts_xy[torso_indices].astype(np.int32)
            hull = cv2.convexHull(torso_poly)
            cv2.fillConvexPoly(mask, hull, 255)
            cv2.polylines(mask, [hull], isClosed=True, color=255, thickness=limb_radius)

        for p1, p2 in SKELETON_LIMBS:
            if kpts_conf[p1] >= th and kpts_conf[p2] >= th:
                pt1 = tuple(local_kpts_xy[p1].astype(np.int32))
                pt2 = tuple(local_kpts_xy[p2].astype(np.int32))
                cv2.line(mask, pt1, pt2, 255, thickness=limb_radius * 2, lineType=cv2.LINE_AA)

        mask_float = cv2.GaussianBlur(mask, (7, 7), 0).astype(np.float32) / 255.0
        visible_ratio = float(np.mean(mask > 127))
        return mask_float[:, :, None], visible_ratio

    def extract_head_shoulder_subcrop(
        self,
        masked_crop: np.ndarray,
        local_kpts_xy: np.ndarray,
        kpts_conf: np.ndarray,
    ) -> np.ndarray:
        h, w = masked_crop.shape[:2]
        th = self.kpt_conf_thresh
        upper_vis = [local_kpts_xy[i] for i in range(7) if kpts_conf[i] >= th]

        if len(upper_vis) >= 2:
            pts = np.array(upper_vis)
            y_max = int(min(h, np.max(pts[:, 1]) + 0.15 * h))
            y_min = int(max(0, np.min(pts[:, 1]) - 0.18 * h))
            if y_max - y_min >= 24:
                return masked_crop[y_min:y_max, :]

        return masked_crop[: max(24, int(h * 0.38)), :]

    def evaluate_and_extract(
        self,
        frame: np.ndarray,
        bbox_xyxy: np.ndarray,
        keypoints_xy: np.ndarray,
        keypoints_conf: np.ndarray,
        is_intersecting: bool,
    ) -> Tuple[Optional[np.ndarray], CropQualityReport]:
        H_img, W_img = frame.shape[:2]
        x1, y1, x2, y2 = np.clip(bbox_xyxy.astype(np.int32), [0, 0, 0, 0], [W_img, H_img, W_img, H_img])
        bw, bh = x2 - x1, y2 - y1

        if bw < 28 or bh < 56:
            return None, CropQualityReport(False, False, 0.0, 0.0, "FRONT", "BBox too small")

        th = self.kpt_conf_thresh
        shoulders_ok = (keypoints_conf[5] >= th) and (keypoints_conf[6] >= th)
        torso_anchor_ok = any(keypoints_conf[i] >= th for i in (0, 1, 2, 11, 12))

        if not (shoulders_ok and torso_anchor_ok):
            return None, CropQualityReport(False, False, 0.0, 0.0, "FRONT", "Heavy occlusion / Missing core skeleton")

        raw_crop = frame[y1:y2, x1:x2].copy()
        local_kpts = keypoints_xy.copy()
        local_kpts[:, 0] -= x1
        local_kpts[:, 1] -= y1

        gray = cv2.cvtColor(raw_crop, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if sharpness < self.min_laplacian_var:
            return None, CropQualityReport(False, False, sharpness, 0.0, "FRONT", f"Motion blur ({sharpness:.1f})")

        alpha_mask, vis_ratio = self.build_pose_silhouette_mask(raw_crop.shape, local_kpts, keypoints_conf)
        if vis_ratio < 0.18:
            return None, CropQualityReport(False, False, sharpness, vis_ratio, "FRONT", "Low silhouette coverage")

        bg_fill = np.full_like(raw_crop, IMAGENET_MEAN_BGR)
        masked_crop = (raw_crop.astype(np.float32) * alpha_mask + bg_fill.astype(np.float32) * (1.0 - alpha_mask)).astype(
            np.uint8
        )

        head_crop = self.extract_head_shoulder_subcrop(masked_crop, local_kpts, keypoints_conf)
        rgb_body = cv2.cvtColor(masked_crop, cv2.COLOR_BGR2RGB)
        rgb_head = cv2.cvtColor(head_crop, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            feats = self.extractor([rgb_body, rgb_head])
            feats = F.normalize(feats, p=2, dim=1)
            combined = self.body_weight * feats[0] + self.head_weight * feats[1]
            combined = F.normalize(combined, p=2, dim=0)
            embedding = combined.cpu().numpy().astype(np.float32)

        view_bin = self.classify_pose_view(keypoints_xy, keypoints_conf, th)
        valid_for_gallery = (not is_intersecting) and (sharpness >= self.min_laplacian_var * 1.25) and (vis_ratio >= 0.28)

        return embedding, CropQualityReport(
            is_valid_for_reid=True,
            is_valid_for_gallery=valid_for_gallery,
            sharpness_score=sharpness,
            visible_ratio=vis_ratio,
            pose_view_bin=view_bin,
            reject_reason="Intersecting" if is_intersecting else "OK",
        )