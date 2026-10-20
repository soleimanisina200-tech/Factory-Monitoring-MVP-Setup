import os
import logging
import cv2
import numpy as np
from torchreid.utils import FeatureExtractor

logger = logging.getLogger(__name__)

class OSNetExtractor:
    def __init__(self, device='cuda'):
        self.device = device
        weights_path = ""
        for fname in ["osnet_x1_0_msmt17.pt", "osnet_x1_0_msmt17.pth", "osnet_ain_x1_0_msmt17.pth"]:
            fpath = os.path.join("weights", fname)
            if os.path.exists(fpath):
                weights_path = fpath
                break

        self.extractor = FeatureExtractor(
            model_name='osnet_x1_0',
            model_path=weights_path,
            device=self.device
        )
        logger.info("OSNet Ready (Anti-Uniform Head-Focus Hack Enabled).")

    def _l2_norm(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-6 else vec

    def extract(self, person_crop: np.ndarray) -> np.ndarray:
        """تمرکز روی سر و صورت برای خنثی کردن اثر لباس‌های یکسان"""
        if person_crop is None or person_crop.size == 0:
            return None

        h, w = person_crop.shape[:2]
        if h < 40 or w < 20:
            return None

        rgb_full = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        
        # برش 30 درصد بالایی کادر (فقط سر، کلاه، چهره و شانه‌ها)
        head_h = max(15, int(h * 0.30))
        rgb_head = rgb_full[:head_h, :]

        # استخراج ویژگی از کل بدن و از سر به صورت همزمان
        tensors = self.extractor([rgb_full, rgb_head])
        vec_full = self._l2_norm(tensors[0].cpu().numpy().flatten())
        vec_head = self._l2_norm(tensors[1].cpu().numpy().flatten())

        # هک حیاتی: 75% اهمیت به سر و چهره + 25% اهمیت به قد و قواره بدن
        hybrid_vec = np.concatenate([
            vec_head * np.sqrt(0.75),
            vec_full * np.sqrt(0.25)
        ]).astype(np.float32)

        return self._l2_norm(hybrid_vec)