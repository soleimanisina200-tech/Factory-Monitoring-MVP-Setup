# app/vehicles/detector.py
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from ultralytics import YOLO


@dataclass
class VehicleTrack:
    track_id: int
    class_id: int
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)


class ForkliftDetector:
    """
    Dedicated YOLO11 + ByteTrack detector for forklifts and designated industrial vehicles.
    Runs independently so the existing IndustrialPoseDetector in app/detection/detector.py
    remains completely unmodified.
    """

    def __init__(self, config: Dict):
        det_cfg = config.get("detection", {})
        self.model_path = det_cfg.get("model_path", "yolo11m.pt")
        self.tracker_config = det_cfg.get(
            "tracker_config", "configs/custom_bytetrack.yaml"
        )
        self.conf_threshold = float(det_cfg.get("conf_threshold", 0.45))
        self.iou_threshold = float(det_cfg.get("iou_threshold", 0.50))
        self.vehicle_class_ids = list(det_cfg.get("vehicle_class_ids", [0, 7]))
        self.imgsz = int(det_cfg.get("imgsz", 640))

        self.model = YOLO(self.model_path)

    def track_vehicles(self, frame: np.ndarray) -> List[VehicleTrack]:
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_config,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.vehicle_class_ids,
            imgsz=self.imgsz,
            verbose=False,
        )

        tracks: List[VehicleTrack] = []
        if not results or results[0].boxes is None or results[0].boxes.id is None:
            return tracks

        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        cls_ids = results[0].boxes.cls.int().cpu().numpy()

        for bbox, tid, conf, cid in zip(boxes, track_ids, confs, cls_ids):
            tracks.append(
                VehicleTrack(
                    track_id=int(tid),
                    class_id=int(cid),
                    confidence=float(conf),
                    bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                )
            )
        return tracks