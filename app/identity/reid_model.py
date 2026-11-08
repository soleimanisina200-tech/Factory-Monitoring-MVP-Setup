# app/identity/reid_model.py
from __future__ import annotations
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchreid

warnings.filterwarnings("ignore", category=FutureWarning, module="torchreid")
logger = logging.getLogger("FactoryReID")

MSMT17_GDRIVE_ID = "1KxxvkhV_pnm34VUwtlum-JyvChL_2eP-"

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
        logger.warning(f"gdown auto-download failed ({e}).")

    if not path.exists() or path.stat().st_size < 5_000_000:
        raise RuntimeError(f"CRITICAL ERROR: Failed to acquire MSMT17 weights at {path}.")
    return str(path)


class PoseGuidedOSNetExtractor:
    """
    Posture-Invariant Zero-Data Re-ID Engine:
      - Eliminates grey-bar letterbox artifacts that broke matching between standing & crouching poses.
      - Uses Keypoint-Aligned Upper-Body Subcrops (tracks head & shoulders even when bent 90 degrees).
      - Uses Hellinger/Bhattacharyya Keypoint-Guided Chromatic Histograms for uniform/pant discrimination.
    """

    def __init__(
        self,
        model_name: str = "osnet_x1_0",
        weights_path: Optional[str] = None,
        device: str = "cuda:0",
        kpt_conf_thresh: float = 0.30,
        min_laplacian_var: float = 15.0,
        head_branch_weight: float = 0.32,
        color_branch_weight: float = 0.18,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.kpt_conf_thresh = kpt_conf_thresh
        self.min_laplacian_var = min_laplacian_var

        self.head_w = float(head_branch_weight)
        self.color_w = float(color_branch_weight)
        self.body_w = max(0.30, 1.0 - self.head_w - self.color_w)

        valid_weights_path = (
            weights_path if (weights_path and Path(weights_path).exists()) else ensure_msmt17_weights()
        )
        logger.info(f"Loading true Person Re-ID MSMT17 weights from: {valid_weights_path}")

        self.extractor = torchreid.utils.FeatureExtractor(
            model_name=model_name,
            model_path=valid_weights_path,
            device=str(self.device),
        )
        self.clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(4, 4))

    def normalize_illumination(self, bgr_crop: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        l_eq = self.clahe.apply(l_chan)
        merged = cv2.merge((l_eq, a_chan, b_chan))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    @staticmethod
    def resize_for_osnet(crop_bgr: np.ndarray, target_w: int = 128, target_h: int = 256) -> np.ndarray:
        """
        Resizes crop to OSNet's native (128, 256) resolution using replicate-border aspect clamping
        instead of flat grey bars, preventing artificial step-edges and vertical receptive-field shifts
        when workers crouch or bend over.
        """
        h, w = crop_bgr.shape[:2]
        if h <= 0 or w <= 0:
            return np.full((target_h, target_w, 3), IMAGENET_MEAN_BGR, dtype=np.uint8)

        desired_h = max(h, int(w * 1.55))
        if desired_h > h:
            pad_top = (desired_h - h) // 2
            pad_bot = desired_h - h - pad_top
            padded = cv2.copyMakeBorder(crop_bgr, pad_top, pad_bot, 0, 0, cv2.BORDER_REPLICATE)
            if pad_top > 0:
                padded[:pad_top] = (
                    0.35 * padded[:pad_top].astype(np.float32) + 0.65 * IMAGENET_MEAN_BGR
                ).astype(np.uint8)
            if pad_bot > 0:
                padded[-pad_bot:] = (
                    0.35 * padded[-pad_bot:].astype(np.float32) + 0.65 * IMAGENET_MEAN_BGR
                ).astype(np.uint8)
            crop_bgr = padded

        interp = cv2.INTER_AREA if (crop_bgr.shape[1] > target_w) else cv2.INTER_LINEAR
        return cv2.resize(crop_bgr, (target_w, target_h), interpolation=interp)

    @staticmethod
    def classify_pose_view(
        kpts_xy: np.ndarray,
        kpts_conf: np.ndarray,
        bbox_width: float,
        thresh: float = 0.30,
    ) -> str:
        if kpts_conf[5] >= thresh and kpts_conf[6] >= thresh:
            dx = float(kpts_xy[5, 0] - kpts_xy[6, 0])
            torso_w = abs(dx)
            face_vis = sum(1 for i in (0, 1, 2) if kpts_conf[i] >= thresh)

            if torso_w < max(8.0, 0.22 * bbox_width):
                sh_mid_x = 0.5 * float(kpts_xy[5, 0] + kpts_xy[6, 0])
                if kpts_conf[0] >= thresh:
                    return "RIGHT" if float(kpts_xy[0, 0]) > sh_mid_x else "LEFT"
                return "LEFT" if float(kpts_conf[5]) > float(kpts_conf[6]) else "RIGHT"

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

        cv2.ellipse(
            mask,
            (w // 2, int(h * 0.52)),
            (max(4, int(w * 0.36)), max(8, int(h * 0.44))),
            0,
            0,
            360,
            210,
            -1,
        )

        limb_radius = max(int(w * 0.16), 6)
        head_radius = max(int(w * 0.22), 9)

        head_pts = [local_kpts_xy[i] for i in range(5) if kpts_conf[i] >= th]
        if head_pts:
            center = np.mean(head_pts, axis=0).astype(np.int32)
            cv2.circle(mask, tuple(center), head_radius, 255, -1)

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

        ksize = 9 if min(h, w) >= 40 else 5
        mask_float = cv2.GaussianBlur(mask, (ksize, ksize), 0).astype(np.float32) / 255.0
        soft_alpha = 0.20 + 0.80 * mask_float
        visible_ratio = float(np.mean(mask > 127))
        return soft_alpha[:, :, None], visible_ratio

    def extract_anatomical_upper_subcrop(
        self,
        masked_crop: np.ndarray,
        local_kpts_xy: np.ndarray,
        kpts_conf: np.ndarray,
    ) -> np.ndarray:
        h, w = masked_crop.shape[:2]
        th = self.kpt_conf_thresh
        upper_indices = [i for i in range(7) if kpts_conf[i] >= th]

        if len(upper_indices) >= 2:
            pts = local_kpts_xy[upper_indices]
            pad_x = max(12.0, 0.28 * w)
            pad_y_top = max(14.0, 0.22 * h)
            pad_y_bot = max(14.0, 0.24 * h)

            ux1 = int(np.clip(np.min(pts[:, 0]) - pad_x, 0, max(0, w - 2)))
            ux2 = int(np.clip(np.max(pts[:, 0]) + pad_x, ux1 + 1, w))
            uy1 = int(np.clip(np.min(pts[:, 1]) - pad_y_top, 0, max(0, h - 2)))
            uy2 = int(np.clip(np.max(pts[:, 1]) + pad_y_bot, uy1 + 1, h))

            if (uy2 - uy1) >= 18 and (ux2 - ux1) >= 18:
                return masked_crop[uy1:uy2, ux1:ux2]

        return masked_crop[: max(18, int(h * 0.50)), :]

    @staticmethod
    def _hellinger_hist(lab_patch: np.ndarray, hsv_patch: np.ndarray, mask_patch: Optional[np.ndarray]) -> np.ndarray:
        if lab_patch.size == 0 or hsv_patch.size == 0:
            return np.zeros(64, dtype=np.float32)

        if mask_patch is not None and (mask_patch.shape[:2] != lab_patch.shape[:2] or np.count_nonzero(mask_patch) < 8):
            mask_patch = None

        h_a = cv2.calcHist([lab_patch], [1], mask_patch, [16], [0, 256]).flatten()
        h_b = cv2.calcHist([lab_patch], [2], mask_patch, [16], [0, 256]).flatten()
        h_h = cv2.calcHist([hsv_patch], [0], mask_patch, [16], [0, 180]).flatten()
        h_s = cv2.calcHist([hsv_patch], [1], mask_patch, [16], [0, 256]).flatten()

        raw = np.concatenate([h_a, h_b, h_h, h_s]).astype(np.float32)
        l1_norm = raw / (np.sum(raw) + 1e-6)
        hellinger = np.sqrt(l1_norm)
        return hellinger / (np.linalg.norm(hellinger) + 1e-6)

    def extract_chromatic_part_descriptor(
        self,
        norm_crop: np.ndarray,
        alpha_mask: np.ndarray,
        local_kpts_xy: np.ndarray,
        kpts_conf: np.ndarray,
    ) -> np.ndarray:
        h, w = norm_crop.shape[:2]
        binary_mask = (alpha_mask[:, :, 0] > 0.52).astype(np.uint8) * 255

        lab = cv2.cvtColor(norm_crop, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(norm_crop, cv2.COLOR_BGR2HSV)

        th = self.kpt_conf_thresh
        hip_vis = [local_kpts_xy[i, 1] for i in (11, 12) if kpts_conf[i] >= th]
        raw_split = int(np.mean(hip_vis)) if hip_vis else int(h * 0.52)
        split_y = int(np.clip(raw_split, max(1, int(h * 0.35)), max(2, min(h - 1, int(h * 0.70)))))

        upper_vec = self._hellinger_hist(lab[:split_y], hsv[:split_y], binary_mask[:split_y])
        lower_vec = self._hellinger_hist(lab[split_y:], hsv[split_y:], binary_mask[split_y:])
        full_vec = self._hellinger_hist(lab, hsv, binary_mask)

        color_vec = np.concatenate([upper_vec, lower_vec, full_vec]).astype(np.float32)
        return color_vec / (np.linalg.norm(color_vec) + 1e-6)

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
        bw, bh = int(x2 - x1), int(y2 - y1)

        if bw < 12 or bh < 24:
            return None, CropQualityReport(False, False, 0.0, 0.0, "FRONT", "BBox too small")

        raw_crop = frame[y1:y2, x1:x2].copy()
        local_kpts = keypoints_xy.copy()
        local_kpts[:, 0] -= x1
        local_kpts[:, 1] -= y1

        gray = cv2.cvtColor(raw_crop, cv2.COLOR_BGR2GRAY)
        if bw > 128:
            gray = cv2.resize(gray, (128, max(1, int(128 * bh / bw))), interpolation=cv2.INTER_AREA)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        scale_factor = min(1.0, (bh / 140.0) ** 1.5)
        effective_blur_thresh = max(6.0, self.min_laplacian_var * scale_factor)

        alpha_mask, vis_ratio = self.build_pose_silhouette_mask(raw_crop.shape, local_kpts, keypoints_conf)

        norm_crop = self.normalize_illumination(raw_crop)
        bg_fill = np.full_like(norm_crop, IMAGENET_MEAN_BGR)
        masked_crop = (
            norm_crop.astype(np.float32) * alpha_mask + bg_fill.astype(np.float32) * (1.0 - alpha_mask)
        ).astype(np.uint8)

        upper_crop = self.extract_anatomical_upper_subcrop(masked_crop, local_kpts, keypoints_conf)

        body_input = self.resize_for_osnet(masked_crop, 128, 256)
        upper_input = self.resize_for_osnet(upper_crop, 128, 256)

        rgb_body = cv2.cvtColor(body_input, cv2.COLOR_BGR2RGB)
        rgb_upper = cv2.cvtColor(upper_input, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            feats = self.extractor([rgb_body, rgb_upper])
            feats = F.normalize(feats, p=2, dim=1).cpu().numpy().astype(np.float32)

        color_feat = self.extract_chromatic_part_descriptor(norm_crop, alpha_mask, local_kpts, keypoints_conf)

        composite = np.concatenate(
            [
                np.sqrt(self.body_w) * feats[0],
                np.sqrt(self.head_w) * feats[1],
                np.sqrt(self.color_w) * color_feat,
            ]
        ).astype(np.float32)
        composite /= np.linalg.norm(composite) + 1e-8

        view_bin = self.classify_pose_view(keypoints_xy, keypoints_conf, float(bw), self.kpt_conf_thresh)
        valid_for_gallery = (not is_intersecting) and (sharpness >= effective_blur_thresh)

        return composite, CropQualityReport(
            is_valid_for_reid=True,
            is_valid_for_gallery=valid_for_gallery,
            sharpness_score=sharpness,
            visible_ratio=vis_ratio,
            pose_view_bin=view_bin,
            reject_reason="Intersecting" if is_intersecting else "OK",
        )