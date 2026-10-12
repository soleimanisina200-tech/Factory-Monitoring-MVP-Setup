import os
import urllib.request
import logging
import cv2
import numpy as np
from torchreid.utils import FeatureExtractor

logger = logging.getLogger(__name__)

class OSNetExtractor:
    def __init__(self, device='cuda'):
        self.device = device
        logger.info("Loading OSNet for Person Re-ID...")

        # ساخت پوشه برای ذخیره وزن‌های تخصصی
        weights_dir = "weights"
        os.makedirs(weights_dir, exist_ok=True)
        weights_path = os.path.join(weights_dir, "osnet_x1_0_msmt17.pt")

        # اگر فایل وزن‌های Re-ID وجود نداشت، آن را به صورت خودکار دانلود کن
        if not os.path.exists(weights_path):
            logger.info("Downloading OSNet MSMT17 Re-ID weights... (This may take a minute)")
            url = "https://github.com/mikel-brostrom/yolo_tracking/releases/download/v3.0/osnet_x1_0_msmt17.pt"
            try:
                # استفاده از User-Agent برای جلوگیری از مسدود شدن دانلود توسط گیت‌هاب
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response, open(weights_path, 'wb') as out_file:
                    out_file.write(response.read())
                logger.info("Download complete.")
            except Exception as e:
                logger.error(f"Failed to download weights automatically: {e}")
                logger.info(f"Please download manually from: {url}")
                logger.info(f"And save it exactly at: {os.path.abspath(weights_path)}")

        # --- بخش حیاتی: ارسال مسیر وزن‌های Re-ID به اکسترکتور ---
        self.extractor = FeatureExtractor(
            model_name='osnet_x1_0',
            model_path=weights_path if os.path.exists(weights_path) else '',
            device=self.device
        )
        logger.info("OSNet Feature Extractor loaded successfully with MSMT17 weights.")

    def extract(self, cv2_image):
        if cv2_image is None or cv2_image.size == 0:
            return None
            
        rgb_img = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        features_tensor = self.extractor([rgb_img])
        feature_vector = features_tensor[0].cpu().numpy().flatten()
        
        # نرمال‌سازی L2
        norm = np.linalg.norm(feature_vector)
        if norm > 0:
            feature_vector = feature_vector / norm
            
        return feature_vector