import logging
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class PersonDetector:
    def __init__(self, model_path: str = "yolo11s.pt", conf_threshold: float = 0.4):
        logger.info(f"Initializing YOLO model from {model_path}...")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
        # Path to our custom tracker configuration
        self.tracker_config = "configs/custom_bytetrack.yaml"
        logger.info(f"Using custom tracker configuration from {self.tracker_config}")

    def track(self, frame):
        # Pass the custom tracker file instead of the default string
        results = self.model.track(
            source=frame, 
            classes=[0], 
            conf=self.conf_threshold,
            tracker=self.tracker_config, 
            persist=True,
            verbose=False
        )
        return results[0]