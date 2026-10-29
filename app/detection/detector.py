# app/detection/detector.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class PoseDetection:
    track_id: int
    bbox_xyxy: np.ndarray
    conf: float
    keypoints_xy: np.ndarray
    keypoints_conf: np.ndarray
    is_intersecting: bool = False
    ground_uv: Tuple[float, float] = (0.0, 0.0)
    bev_xy: Optional[np.ndarray] = None


class IndustrialPoseDetector:
    def __init__(
        self,
        model_weights: str = "yolo11m-pose.pt",
        tracker_config: str = "configs/custom_bytetrack.yaml",
        device: str = "cuda:0",
        conf_thres: float = 0.35,
        iou_thres: float = 0.50,
        imgsz: int = 640,
        half: bool = True,
    ):
        self.device = device
        self.tracker_config = tracker_config
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.imgsz = imgsz

        self.model = YOLO(model_weights)
        self.model.to(self.device)

        # Set FP16 directly on the PyTorch model to eliminate Ultralytics predict-time warnings
        if half and "cuda" in str(self.device) and torch.cuda.is_available():
            self.model.model.half()

    def track_frame(self, frame: np.ndarray) -> List[PoseDetection]:
        # Removed deprecated 'half=self.half' keyword argument
        results = self.model.track(
            source=frame,
            tracker=self.tracker_config,
            persist=True,
            classes=[0],
            conf=self.conf_thres,
            iou=self.iou_thres,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        if not results or results[0].boxes is None or results[0].boxes.id is None:
            return []

        res = results[0]
        boxes_xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float32)
        track_ids = res.boxes.id.int().cpu().numpy()
        confs = res.boxes.conf.cpu().numpy().astype(np.float32)

        if res.keypoints is not None and res.keypoints.xy is not None:
            kpts_xy = res.keypoints.xy.cpu().numpy().astype(np.float32)
            kpts_conf = (
                res.keypoints.conf.cpu().numpy().astype(np.float32)
                if res.keypoints.conf is not None
                else np.zeros((len(boxes_xyxy), 17), dtype=np.float32)
            )
        else:
            kpts_xy = np.zeros((len(boxes_xyxy), 17, 2), dtype=np.float32)
            kpts_conf = np.zeros((len(boxes_xyxy), 17), dtype=np.float32)

        detections: List[PoseDetection] = []
        for i in range(len(boxes_xyxy)):
            detections.append(
                PoseDetection(
                    track_id=int(track_ids[i]),
                    bbox_xyxy=boxes_xyxy[i],
                    conf=float(confs[i]),
                    keypoints_xy=kpts_xy[i],
                    keypoints_conf=kpts_conf[i],
                )
            )
        return detections