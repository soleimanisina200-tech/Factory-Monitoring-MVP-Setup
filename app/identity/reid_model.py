import logging
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchreid.utils import FeatureExtractor

from app.logic.geometry import (
    KP_L_SHOULDER,
    KP_R_SHOULDER,
    KP_L_HIP,
    KP_R_HIP,
    KP_L_KNEE,
    KP_R_KNEE,
    KP_L_ANKLE,
    KP_R_ANKLE,
    is_skeleton_complete,
)

logger = logging.getLogger(__name__)


class PoseGuidedOSNetExtractor:
    """
    استخراج‌کننده ویژگی مبتنی بر اسکلت و شبکه OSNet-AIN:
    - تقسیم بدن به ۳ ناحیه (سر/گردن، کفش/پا، تنه) با وزن‌دهی ضد لباس فرم یکسان
    - اجرای Batch روی GPU برای حذف گلوگاه همگام‌سازی تک‌تصویری CPU-GPU
    """

    def __init__(
        self,
        model_name: str = "osnet_ain_x1_0",
        device: str = "cuda",
        part_weights: dict | None = None,
        min_kp_conf: float = 0.42,
    ):
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.min_kp_conf = min_kp_conf

        weights_cfg = part_weights or {"head": 0.55, "shoes": 0.30, "torso": 0.15}
        total_w = float(sum(weights_cfg.values())) or 1.0
        self.w_head = weights_cfg.get("head", 0.55) / total_w
        self.w_shoes = weights_cfg.get("shoes", 0.30) / total_w
        self.w_torso = weights_cfg.get("torso", 0.15) / total_w

        logger.info(f"Loading Re-ID model '{model_name}' on {self.device}...")
        try:
            self.extractor = FeatureExtractor(model_name=model_name, device=self.device)
        except Exception as e:
            logger.warning(f"Could not load {model_name} ({e}), falling back to 'osnet_x1_0'...")
            self.extractor = FeatureExtractor(model_name="osnet_x1_0", device=self.device)

        logger.info(
            f"Re-ID Extractor ready (Part Weights -> Head: {self.w_head:.2f}, Shoes: {self.w_shoes:.2f}, Torso: {self.w_torso:.2f})"
        )

    def _safe_crop_rgb(self, frame_bgr: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray | None:
        h, w = frame_bgr.shape[:2]
        cx1 = max(0, min(w - 1, int(x1)))
        cy1 = max(0, min(h - 1, int(y1)))
        cx2 = max(0, min(w, int(x2)))
        cy2 = max(0, min(h, int(y2)))

        if (cx2 - cx1) < 12 or (cy2 - cy1) < 12:
            return None

        crop_bgr = frame_bgr[cy1:cy2, cx1:cx2]
        if crop_bgr.size == 0:
            return None
        return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

    def _build_anatomic_part_crops(
        self,
        frame_bgr: np.ndarray,
        box: list[int],
        keypoints: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """
        ساخت ۳ برش آناتومیک مجزا (سر/گردن، تنه، ساق/کفش) با استفاده از کی‌پوینت‌ها یا هندسه باکس.
        """
        x1, y1, x2, y2 = box
        bw = x2 - x1
        bh = y2 - y1
        if bw < 20 or bh < 45:
            return None

        # مقادیر پیش‌فرض هندسی در صورت نبودن برخی کی‌پوینت‌ها
        head_y1 = y1
        head_y2 = int(y1 + 0.28 * bh)
        head_x1 = int(x1 + 0.12 * bw)
        head_x2 = int(x2 - 0.12 * bw)

        torso_y1 = int(y1 + 0.22 * bh)
        torso_y2 = int(y1 + 0.62 * bh)
        torso_x1 = int(x1 + 0.08 * bw)
        torso_x2 = int(x2 - 0.08 * bw)

        shoes_y1 = int(y1 + 0.68 * bh)
        shoes_y2 = y2
        shoes_x1 = x1
        shoes_x2 = x2

        if keypoints is not None and len(keypoints) >= 17:
            sh_pts = [
                keypoints[idx]
                for idx in (KP_L_SHOULDER, KP_R_SHOULDER)
                if keypoints[idx][2] >= self.min_kp_conf
            ]
            hip_pts = [
                keypoints[idx]
                for idx in (KP_L_HIP, KP_R_HIP)
                if keypoints[idx][2] >= self.min_kp_conf
            ]
            knee_pts = [
                keypoints[idx]
                for idx in (KP_L_KNEE, KP_R_KNEE)
                if keypoints[idx][2] >= self.min_kp_conf
            ]
            ank_pts = [
                keypoints[idx]
                for idx in (KP_L_ANKLE, KP_R_ANKLE)
                if keypoints[idx][2] >= self.min_kp_conf
            ]

            if sh_pts:
                sh_y = float(np.mean([p[1] for p in sh_pts]))
                head_y2 = int(min(y2, max(y1 + 16, sh_y + 0.08 * bh)))
                torso_y1 = int(max(y1, sh_y - 0.04 * bh))

                if len(sh_pts) == 2:
                    min_sh_x = min(sh_pts[0][0], sh_pts[1][0])
                    max_sh_x = max(sh_pts[0][0], sh_pts[1][0])
                    pad_x = max(12.0, 0.20 * (max_sh_x - min_sh_x))
                    torso_x1 = int(max(x1, min_sh_x - pad_x))
                    torso_x2 = int(min(x2, max_sh_x + pad_x))

            if hip_pts:
                hip_y = float(np.mean([p[1] for p in hip_pts]))
                if hip_y > torso_y1 + 15:
                    torso_y2 = int(min(y2, hip_y + 0.05 * bh))

            if knee_pts or ank_pts:
                if knee_pts:
                    knee_y = float(np.mean([p[1] for p in knee_pts]))
                    shoes_y1 = int(max(y1 + 0.50 * bh, min(y2 - 16, knee_y - 0.04 * bh)))
                if ank_pts:
                    ank_y = float(np.max([p[1] for p in ank_pts]))
                    shoes_y2 = int(min(frame_bgr.shape[0], max(shoes_y1 + 16, ank_y + 0.06 * bh)))

        full_rgb = self._safe_crop_rgb(frame_bgr, x1, y1, x2, y2)
        if full_rgb is None:
            return None

        head_rgb = self._safe_crop_rgb(frame_bgr, head_x1, head_y1, head_x2, head_y2)
        torso_rgb = self._safe_crop_rgb(frame_bgr, torso_x1, torso_y1, torso_x2, torso_y2)
        shoes_rgb = self._safe_crop_rgb(frame_bgr, shoes_x1, shoes_y1, shoes_x2, shoes_y2)

        if head_rgb is None:
            head_rgb = full_rgb[: max(12, int(0.30 * full_rgb.shape[0])), :]
        if torso_rgb is None:
            torso_rgb = full_rgb[
                int(0.25 * full_rgb.shape[0]) : max(24, int(0.65 * full_rgb.shape[0])), :
            ]
        if shoes_rgb is None:
            shoes_rgb = full_rgb[int(0.65 * full_rgb.shape[0]) :, :]

        return head_rgb, torso_rgb, shoes_rgb

    def extract_batch(
        self,
        frame_bgr: np.ndarray,
        boxes: list[list[int]],
        keypoints_list: list[np.ndarray | None] | None = None,
        require_complete_skeleton: bool = False,
    ) -> list[np.ndarray | None]:
        """
        استخراج دسته‌ای (Batched GPU Inference) برای تمام کارگران حاضر در فریم طی یک مرحله Forward Pass.
        """
        num_targets = len(boxes)
        if num_targets == 0 or frame_bgr is None or frame_bgr.size == 0:
            return []

        h, w = frame_bgr.shape[:2]
        flat_crops = []
        valid_target_indices = []

        for i in range(num_targets):
            box = boxes[i]
            kpts = keypoints_list[i] if (keypoints_list and i < len(keypoints_list)) else None

            if require_complete_skeleton and kpts is not None:
                if not is_skeleton_complete(kpts, w, h, min_conf=self.min_kp_conf):
                    continue

            part_crops = self._build_anatomic_part_crops(frame_bgr, box, kpts)
            if part_crops is None:
                continue

            head_rgb, torso_rgb, shoes_rgb = part_crops
            flat_crops.extend([head_rgb, torso_rgb, shoes_rgb])
            valid_target_indices.append(i)

        results: list[np.ndarray | None] = [None] * num_targets
        if not flat_crops:
            return results

        with torch.no_grad():
            raw_features = self.extractor(flat_crops)
            if not isinstance(raw_features, torch.Tensor):
                raw_features = torch.tensor(raw_features, device=self.device, dtype=torch.float32)

            # نرمال‌سازی L2 روی GPU قبل از انتقال به CPU
            normed_features = F.normalize(raw_features, p=2, dim=1)
            # شکل‌دهی به (N_valid, 3, 512) برای ترکیب وزن‌دار سر، تنه و کفش روی GPU
            reshaped = normed_features.view(len(valid_target_indices), 3, -1)

            part_scales = torch.tensor(
                [np.sqrt(self.w_head), np.sqrt(self.w_torso), np.sqrt(self.w_shoes)],
                device=reshaped.device,
                dtype=reshaped.dtype,
            ).view(1, 3, 1)

            weighted_parts = reshaped * part_scales
            # ترکیب بردار وزن‌دار در فضای 512 بعدی + حفظ تمایز بخش‌ها
            fused_gpu = (
                self.w_head * reshaped[:, 0, :]
                + self.w_torso * reshaped[:, 1, :]
                + self.w_shoes * reshaped[:, 2, :]
            )
            # الحاق بردار بخش‌ها برای تفکیک حداکثری در ضرب داخلی کسینوسی
            concat_gpu = weighted_parts.reshape(len(valid_target_indices), -1)
            final_gpu = F.normalize(torch.cat([concat_gpu, fused_gpu], dim=1), p=2, dim=1)

            final_cpu = final_gpu.cpu().numpy()

        for idx_in_batch, target_idx in enumerate(valid_target_indices):
            results[target_idx] = final_cpu[idx_in_batch]

        return results

    def extract(self, cv2_image: np.ndarray, keypoints: np.ndarray | None = None) -> np.ndarray | None:
        """
        متد تک‌تصویری سازگار با فراخوانی‌های قدیمی.
        """
        if cv2_image is None or cv2_image.size == 0:
            return None

        h, w = cv2_image.shape[:2]
        batch_out = self.extract_batch(
            frame_bgr=cv2_image,
            boxes=[[0, 0, w, h]],
            keypoints_list=[keypoints],
            require_complete_skeleton=False,
        )
        return batch_out[0] if batch_out else None


OSNetExtractor = PoseGuidedOSNetExtractor