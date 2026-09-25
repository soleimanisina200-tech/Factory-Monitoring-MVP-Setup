import logging
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class PersonDetector:
    def __init__(self, model_path: str = "yolo11s.pt", conf_threshold: float = 0.4):
        logger.info(f"Initializing YOLO model from {model_path}...")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        logger.info("YOLO model initialized successfully.")

    def detect(self, frame):
        # Run inference. 
        # classes=[0] ensures we only detect 'person' (COCO dataset id 0).
        # verbose=False prevents Ultralytics from printing logs for every single frame.
        results = self.model.predict(
            source=frame, 
            classes=[0], 
            conf=self.conf_threshold, 
            verbose=False
        )
        
        # We process one frame at a time, so we return the first result object
        return results[0]