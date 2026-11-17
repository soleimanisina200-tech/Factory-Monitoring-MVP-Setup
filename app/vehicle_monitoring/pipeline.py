# app/vehicle_monitoring/pipeline.py
"""
Forklift detection + driver monitoring.

In main.py the persons come from the pose detector + identity manager (stable worker IDs, keypoints, foot points);
this pipeline only detects/tracks vehicles and runs the driver state machine. Standalone
(run_vehicle_monitoring.py) it also detects persons with the vehicle model (no pose, tracker IDs only).

Vehicle classes are selected by NAME from the model (`vehicle_class_names: [forklift]`) or by id. A segmentation
model (e.g. a fine-tuned yolo11m-seg) is used automatically: its mask gives the real chassis edge on the floor.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import yaml

from app.logic.calibration import load_calibration
from app.vehicle_monitoring.dashboard import VehicleSafetyDashboard
from app.vehicle_monitoring.distance_estimator import RealWorldDistanceEstimator
from app.vehicle_monitoring.driver_vehicle_fsm import DriverVehicleMonitorFSM

logger = logging.getLogger("FactoryReID")


def resolve_class_ids(model_names: Dict[int, str], names: Optional[List[str]], ids: Optional[List[int]]) -> List[int]:
    """Class ids for the configured vehicle class NAMES (case-insensitive) plus explicit ids."""
    out = [int(i) for i in (ids or [])]
    if names:
        lookup = {str(n).strip().lower(): int(i) for i, n in (model_names or {}).items()}
        missing = []
        for n in names:
            key = str(n).strip().lower()
            if key in lookup:
                out.append(lookup[key])
            else:
                missing.append(n)
        if missing:
            logger.error("Vehicle classes %s not in the vehicle model (available: %s)", missing,
                         sorted(set((model_names or {}).values())))
    return sorted(set(out))


class VehicleMonitoringPipeline:
    def __init__(
        self,
        config_path: str = "configs/vehicle_monitoring.yaml",
        reid_extractor: Any = None,  # kept for backwards compatibility (driver Re-ID now comes from main.py)
        calibration_path: Optional[str] = None,
        event_sink: Optional[Callable[[Dict[str, Any], Optional[np.ndarray]], None]] = None,
        model: Any = None,
        device: Optional[str] = None,
    ):
        with open(Path(config_path), "r", encoding="utf-8") as f:
            full_cfg = yaml.safe_load(f) or {}
        self.cfg: Dict[str, Any] = full_cfg.get("vehicle_monitoring", {}) or {}

        if model is None:
            from ultralytics import YOLO

            model = YOLO(self.cfg.get("vehicle_detector_weights", "yolo11m.pt"))
        self.model = model
        self.device = device
        self.person_cls_id = int(self.cfg.get("person_class_id", 0))
        self.vehicle_cls_ids = resolve_class_ids(
            getattr(self.model, "names", {}) or {}, self.cfg.get("vehicle_class_names"),
            self.cfg.get("vehicle_class_ids", [] if self.cfg.get("vehicle_class_names") else [7]),
        )
        if not self.vehicle_cls_ids:
            raise ValueError("No vehicle classes selected: set vehicle_class_names or vehicle_class_ids")
        names = getattr(self.model, "names", {}) or {}
        logger.info("Vehicle classes: %s", [f"{i}:{names.get(i, '?')}" for i in self.vehicle_cls_ids])
        if not any("fork" in str(names.get(i, "")).lower() for i in self.vehicle_cls_ids):
            logger.warning("The vehicle model has no 'forklift' class selected: a COCO model detects forklifts only "
                           "by chance (as truck/car). Train/fine-tune a forklift model for production use.")
        self.v_conf = float(self.cfg.get("vehicle_conf_threshold", 0.40))
        self.p_conf = float(self.cfg.get("person_conf_threshold", 0.45))
        self.vehicle_tracker_cfg = str(self.cfg.get("vehicle_tracker_config", "bytetrack.yaml"))
        self.vehicle_imgsz = int(self.cfg.get("vehicle_imgsz", 1280))
        try:
            import torch

            cuda = torch.cuda.is_available()
        except Exception:
            cuda = False
        self.vehicle_half = bool(self.cfg.get("vehicle_half", True)) and cuda
        self.detect_every_n = max(1, int(self.cfg.get("vehicle_detect_every_n_frames", 1)))
        self._frame_idx = 0
        self._last_vehicles: List[Dict] = []

        self.estimator = RealWorldDistanceEstimator(self.cfg)
        self.fsm = DriverVehicleMonitorFSM(self.cfg, self.estimator)
        self.dashboard = VehicleSafetyDashboard(self.cfg, self.estimator)
        self.event_sink = event_sink
        self.last_events: List[Dict[str, Any]] = []
        self.reid_extractor = None

        self._calibration = load_calibration(calibration_path) if calibration_path else None
        self._calibration_applied = False

    # ------------------------------------------------------------------
    def normalize_pose_persons(self, frame: np.ndarray, raw_persons: Any) -> List[Dict]:
        """Person dicts from main.py -> the FSM format (ids stay strings: '7' worker, 'T12' provisional)."""
        out: List[Dict] = []
        for item in raw_persons or []:
            if not isinstance(item, dict):
                item = {k: getattr(item, k) for k in ("id", "global_id", "track_id", "bbox", "keypoints", "foot_px",
                                                      "embedding", "bev_xy") if hasattr(item, k)}
            pid = item.get("id")
            if pid is None:
                gid = item.get("global_id")
                pid = gid if gid not in (None, -1) else (
                    f"T{item['track_id']}" if item.get("track_id") is not None else None)
            bbox = item.get("bbox", item.get("box"))
            if pid is None or bbox is None:
                continue
            kpts = item.get("keypoints")
            foot = item.get("foot_px")
            emb = item.get("embedding")
            bev = item.get("bev_xy")
            out.append({
                "id": str(pid),
                "bbox": [float(x) for x in bbox],
                "keypoints": np.asarray(kpts) if kpts is not None else None,
                "foot_px": tuple(float(v) for v in foot) if foot is not None else None,
                "ground_uncertain": item.get("ground_uncertain"),
                "embedding": np.asarray(emb) if emb is not None else None,
                "bev_xy": tuple(float(v) for v in bev) if bev is not None else None,
            })
        return out

    def _resolve_tracker(self, override: Optional[str]) -> str:
        cfg = override or self.vehicle_tracker_cfg
        return cfg if (cfg in ("bytetrack.yaml", "botsort.yaml") or Path(cfg).exists()) else "bytetrack.yaml"

    def detect(self, frame: np.ndarray, with_persons: bool = False, tracker_config: Optional[str] = None):
        """-> (vehicles, persons). Vehicles carry an optional `contour` when the model is a segmentation model."""
        classes = self.vehicle_cls_ids + ([self.person_cls_id] if with_persons else [])
        kwargs = dict(persist=True, classes=sorted(set(classes)), conf=min(self.v_conf, self.p_conf) if with_persons
                      else self.v_conf, tracker=self._resolve_tracker(tracker_config), imgsz=self.vehicle_imgsz,
                      half=self.vehicle_half, verbose=False)
        if self.device:
            kwargs["device"] = self.device
        results = self.model.track(frame, **kwargs)
        vehicles: List[Dict] = []
        persons: List[Dict] = []
        if not results or results[0].boxes is None or results[0].boxes.id is None:
            return vehicles, persons
        r = results[0]
        boxes = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.int().cpu().tolist()
        clss = r.boxes.cls.int().cpu().tolist()
        confs = r.boxes.conf.cpu().tolist()
        masks = getattr(r, "masks", None)
        polys = list(masks.xy) if masks is not None and getattr(masks, "xy", None) is not None else None
        for i, (bbox, tid, cls_id, conf) in enumerate(zip(boxes, ids, clss, confs)):
            if cls_id in self.vehicle_cls_ids and conf >= self.v_conf:
                vehicles.append({"id": str(tid), "bbox": bbox.tolist(), "conf": float(conf), "class_id": cls_id,
                                 "contour": np.asarray(polys[i]) if polys is not None and i < len(polys) else None})
            elif with_persons and cls_id == self.person_cls_id and conf >= self.p_conf:
                persons.append({"id": str(tid), "bbox": bbox.tolist(), "conf": float(conf), "keypoints": None,
                                "foot_px": None, "embedding": None})
        return vehicles, persons

    def process_frame(
        self,
        frame: np.ndarray,
        external_persons: Optional[Any] = None,
        tracker_config: Optional[str] = None,
        frame_ts: Optional[float] = None,
        left_view_ids: Optional[List[str]] = None,
        draw: bool = True,
    ) -> np.ndarray:
        if self._calibration is not None and not self._calibration_applied:
            h, w = frame.shape[:2]
            self.estimator.set_homography(self._calibration.homography((w, h)))
            self._calibration_applied = True

        standalone = external_persons is None
        run = standalone or (self._frame_idx % self.detect_every_n == 0)
        self._frame_idx += 1
        if run:
            vehicles, persons = self.detect(frame, with_persons=standalone, tracker_config=tracker_config)
            self._last_vehicles = vehicles
        else:
            vehicles, persons = list(self._last_vehicles), []
        if not standalone:
            persons = self.normalize_pose_persons(frame, external_persons)

        vehicles_state, events = self.fsm.update(vehicles, persons, frame.shape, now=frame_ts,
                                                 left_view_ids=left_view_ids)
        self.last_events = events
        annotated = self.dashboard.render_hud(frame, vehicles_state, now=frame_ts) if draw else frame
        if self.event_sink is not None:
            for ev in events:
                self.event_sink(ev, annotated)
        return annotated

    def telemetry(self, now: float) -> List[Dict[str, Any]]:
        return self.dashboard.telemetry(self.fsm.vehicles, now)
