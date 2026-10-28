import os
import logging
import cv2
import numpy as np
import torch
from torchreid.utils import FeatureExtractor

logger = logging.getLogger(__name__)

class AdvancedReIDExtractor:
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # مغز تخصصی برای تشخیص لباس و جنسیت
        self.model_name = 'osnet_x1_0'
        
        weights_path = ""
        for fname in ["osnet_x1_0_msmt17.pt", "osnet_x1_0_msmt17.pth"]:
            fpath = os.path.join("weights", fname)
            if os.path.exists(fpath):
                weights_path = fpath
                break
                
        if not weights_path:
            logger.warning("CRITICAL: MSMT17 weights NOT FOUND in 'weights' folder!")
        
        logger.info(f"Initializing True Re-ID Engine: {self.model_name} on {self.device}...")
        
        try:
            self.extractor = FeatureExtractor(
                model_name=self.model_name,
                model_path=weights_path, 
                device=self.device
            )
            logger.info(f"True Re-ID Engine (MSMT17) Loaded from: {weights_path}")
        except Exception as e:
            logger.error(f"Failed to load OSNet: {e}")
            raise

    def extract(self, person_crop: np.ndarray) -> np.ndarray:
        if person_crop is None or person_crop.size == 0:
            return None

        h, w = person_crop.shape[:2]
        
        # فیلتر متناسب با ویدیوی 7 فریم و دوربین دور
        if h < 50 or w < 20: 
            return None

        rgb_full = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        features = self.extractor([rgb_full])[0].cpu().numpy().flatten()
        
        norm = np.linalg.norm(features)
        return features / norm if norm > 1e-6 else features