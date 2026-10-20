import logging
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class PersonDetector:
    def __init__(self, model_path="yolo11m.pt", conf_threshold=0.40, imgsz=1280):
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.use_half = torch.cuda.is_available()
        
        logger.info(f"Loading YOLO11 model ({model_path}) at imgsz={imgsz}, FP16={self.use_half}...")
        self.model = YOLO(model_path)
        logger.info("YOLO11 Model loaded successfully.")

    def track(self, frame):
        results = self.model.track(
            frame,
            persist=True,
            classes=[0],
            conf=self.conf_threshold,
            iou=0.55,
            imgsz=self.imgsz,
            half=self.use_half,
            tracker="configs/custom_botsort.yaml",
            verbose=False
        )
        return results[0]