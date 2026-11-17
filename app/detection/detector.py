import logging
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class ObjectDetector:
    """
    تشخیص هوشمند انسان، مفاصل اسکلت (Pose Keypoints) و وسایل نقلیه (لیفتراک/خودرو) همراه با ردیابی ByteTrack.
    """

    def __init__(
        self,
        model_path: str = "yolo11m-pose.pt",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.50,
        tracker_config: str = "configs/custom_bytetrack.yaml",
        device: str = "cuda",
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.tracker_config = tracker_config
        self.device = device

        logger.info(f"Loading YOLO model from {self.model_path} on {self.device}...")
        self.model = YOLO(self.model_path)

    def track(self, frame: np.ndarray) -> list[dict]:
        if frame is None or frame.size == 0:
            return []

        results = self.model.track(
            source=frame,
            persist=True,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            tracker=self.tracker_config,
            device=self.device,
            verbose=False,
        )

        if not results or len(results) == 0:
            return []

        result = results[0]
        if result.boxes is None or result.boxes.id is None:
            return []

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.int().cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.int().cpu().numpy() if result.boxes.cls is not None else np.zeros(len(boxes_xyxy))

        has_keypoints = (
            hasattr(result, "keypoints")
            and result.keypoints is not None
            and result.keypoints.data is not None
            and len(result.keypoints.data) == len(boxes_xyxy)
        )
        keypoints_all = result.keypoints.data.cpu().numpy() if has_keypoints else None

        detections = []
        h, w = frame.shape[:2]

        for i in range(len(track_ids)):
            x1, y1, x2, y2 = boxes_xyxy[i]
            x1 = int(max(0, min(w - 1, x1)))
            y1 = int(max(0, min(h - 1, y1)))
            x2 = int(max(0, min(w - 1, x2)))
            y2 = int(max(0, min(h - 1, y2)))

            if x2 <= x1 or y2 <= y1:
                continue

            cls_id = int(classes[i])
            is_person = (cls_id == 0)

            kpts = keypoints_all[i] if (has_keypoints and is_person) else None
            detections.append(
                {
                    "track_id": int(track_ids[i]),
                    "box": [x1, y1, x2, y2],
                    "conf": float(confs[i]),
                    "class_id": cls_id,
                    "is_person": is_person,
                    "keypoints": kpts,
                }
            )

        return detections


Detector = ObjectDetector
PersonDetector = ObjectDetector