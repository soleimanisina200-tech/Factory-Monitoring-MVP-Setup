# app/vehicle_monitoring/pipeline.py
import yaml
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional
from ultralytics import YOLO

from app.vehicle_monitoring.distance_estimator import RealWorldDistanceEstimator
from app.vehicle_monitoring.driver_vehicle_fsm import DriverVehicleMonitorFSM
from app.vehicle_monitoring.alarm_and_logger import SafetyAlarmAndLogger
from app.vehicle_monitoring.dashboard import VehicleSafetyDashboard


class VehicleMonitoringPipeline:
    """
    Integrates directly with main.py's YOLO-Pose human detector (IndustrialPoseDetector)
    and IdentityManager. Detects forklifts/vehicles while consuming existing human
    pose keypoints, global IDs, and Re-ID embeddings.
    """

    def __init__(self, config_path: str = "configs/vehicle_monitoring.yaml", reid_extractor=None):
        with open(Path(config_path), "r", encoding="utf-8") as f:
            full_cfg = yaml.safe_load(f)
        self.cfg = full_cfg.get("vehicle_monitoring", {})

        weights = self.cfg.get("vehicle_detector_weights", "yolo11m.pt")
        self.model = YOLO(weights)
        self.person_cls_id: int = int(self.cfg.get("person_class_id", 0))
        self.vehicle_cls_ids: List[int] = [int(c) for c in self.cfg.get("vehicle_class_ids", [7])]
        self.v_conf: float = float(self.cfg.get("vehicle_conf_threshold", 0.45))
        self.p_conf: float = float(self.cfg.get("person_conf_threshold", 0.45))

        self.reid_extractor = reid_extractor
        self.estimator = RealWorldDistanceEstimator(self.cfg)
        self.fsm = DriverVehicleMonitorFSM(self.cfg, self.estimator)
        self.logger = SafetyAlarmAndLogger(self.cfg)
        self.dashboard = VehicleSafetyDashboard(self.cfg)

    def _extract_embedding(self, frame: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]

        if self.reid_extractor is not None:
            try:
                if hasattr(self.reid_extractor, "extract"):
                    return self.reid_extractor.extract(crop)
                elif hasattr(self.reid_extractor, "get_embedding"):
                    return self.reid_extractor.get_embedding(crop)
                elif callable(self.reid_extractor):
                    return self.reid_extractor(crop)
            except Exception:
                pass

        import cv2
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    def normalize_pose_persons(
        self, frame: np.ndarray, raw_persons: Any
    ) -> List[Dict]:
        """
        Converts human outputs from main.py (either Ultralytics YOLO-Pose Results,
        a list of track dicts, or custom track objects from IdentityManager) into
        standardized person dicts with 'id', 'bbox', 'keypoints', 'foot_px', and 'embedding'.
        """
        normalized: List[Dict] = []
        if raw_persons is None:
            return normalized

        # Case 1: Ultralytics YOLO-Pose Results object passed directly
        if hasattr(raw_persons, "boxes") or (
            isinstance(raw_persons, list) and len(raw_persons) > 0 and hasattr(raw_persons[0], "boxes")
        ):
            res = raw_persons[0] if isinstance(raw_persons, list) else raw_persons
            if res.boxes is None or res.boxes.id is None:
                return normalized
            boxes = res.boxes.xyxy.cpu().numpy()
            ids = res.boxes.id.int().cpu().tolist()
            kpts_all = (
                res.keypoints.data.cpu().numpy()
                if (hasattr(res, "keypoints") and res.keypoints is not None)
                else None
            )
            for idx, (bbox, tid) in enumerate(zip(boxes, ids)):
                kpts = kpts_all[idx] if kpts_all is not None and idx < len(kpts_all) else None
                b_list = bbox.tolist()
                normalized.append({
                    "id": str(tid),
                    "bbox": b_list,
                    "keypoints": kpts,
                    "foot_px": None,
                    "embedding": self._extract_embedding(frame, b_list),
                })
            return normalized

        # Case 2: List of dicts or track objects from main.py / IdentityManager
        for item in raw_persons:
            if isinstance(item, dict):
                pid = item.get("global_id", item.get("id", item.get("track_id")))
                bbox = item.get("bbox", item.get("box"))
                kpts = item.get("keypoints", item.get("pose"))
                foot_px = item.get("foot_px", item.get("foot_point", item.get("ground_point")))
                emb = item.get("embedding", item.get("reid_embedding", item.get("feature")))
            else:
                pid = getattr(item, "global_id", getattr(item, "id", getattr(item, "track_id", None)))
                bbox = getattr(item, "bbox", getattr(item, "box", None))
                kpts = getattr(item, "keypoints", getattr(item, "pose", None))
                foot_px = getattr(item, "foot_px", getattr(item, "foot_point", None))
                emb = getattr(item, "embedding", getattr(item, "reid_embedding", None))

            if pid is None or bbox is None:
                continue

            b_list = [float(x) for x in bbox]
            if emb is None:
                emb = self._extract_embedding(frame, b_list)

            normalized.append({
                "id": str(pid),
                "bbox": b_list,
                "keypoints": np.asarray(kpts) if kpts is not None else None,
                "foot_px": tuple(map(int, foot_px)) if foot_px is not None else None,
                "embedding": np.asarray(emb) if emb is not None else None,
            })

        return normalized

    def process_frame(
        self,
        frame: np.ndarray,
        external_persons: Optional[Any] = None,
        tracker_config: str = "configs/custom_bytetrack.yaml",
    ) -> np.ndarray:
        """
        If external_persons (from main.py's YOLO-Pose detector) is provided, only runs
        vehicle detection and fuses with the YOLO-Pose human tracks.
        """
        # Only detect vehicles if main.py already provides YOLO-Pose human detections
        target_classes = (
            self.vehicle_cls_ids
            if external_persons is not None
            else list(set([self.person_cls_id] + self.vehicle_cls_ids))
        )

        results = self.model.track(
            frame,
            persist=True,
            classes=target_classes,
            conf=self.v_conf if external_persons is not None else min(self.v_conf, self.p_conf),
            tracker=tracker_config if Path(tracker_config).exists() else "bytetrack.yaml",
            verbose=False,
        )

        detected_vehicles: List[Dict] = []
        detected_persons: List[Dict] = []

        if results and len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.int().cpu().tolist()
            confs = results[0].boxes.conf.cpu().tolist()

            for bbox, tid, cls_id, conf in zip(boxes, ids, clss, confs):
                if cls_id in self.vehicle_cls_ids and conf >= self.v_conf:
                    detected_vehicles.append({
                        "id": str(tid),
                        "bbox": bbox.tolist(),
                        "conf": float(conf),
                    })
                elif cls_id == self.person_cls_id and conf >= self.p_conf and external_persons is None:
                    detected_persons.append({
                        "id": str(tid),
                        "bbox": bbox.tolist(),
                        "conf": float(conf),
                        "keypoints": None,
                        "foot_px": None,
                        "embedding": self._extract_embedding(frame, bbox.tolist()),
                    })

        if external_persons is not None:
            detected_persons = self.normalize_pose_persons(frame, external_persons)

        vehicles_state, triggered_events = self.fsm.update(
            detected_vehicles=detected_vehicles,
            detected_persons=detected_persons,
            frame_shape=frame.shape,
        )

        annotated = self.dashboard.render_hud(
            frame=frame,
            vehicles=vehicles_state,
            recent_events=self.logger.get_recent_events(),
        )

        if triggered_events:
            self.logger.process_events(triggered_events, annotated)

        return annotated