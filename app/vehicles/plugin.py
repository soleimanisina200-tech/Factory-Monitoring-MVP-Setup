# app/vehicles/plugin.py
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import yaml

from app.vehicles.alarm_and_logger import AlarmAndEventLogger
from app.vehicles.association import DriverVehicleMonitor, PersonObservation
from app.vehicles.calibration import GroundPlaneProjector
from app.vehicles.dashboard import VehicleDashboardRenderer
from app.vehicles.detector import ForkliftDetector


class VehicleMonitoringPlugin:
    """
    Isolated facade bridging existing person/pose/Re-ID tracks with the
    Forklift & Driver Monitoring Module.
    """

    def __init__(self, config_path: str = "configs/vehicle_monitoring.yaml"):
        with open(Path(config_path), "r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f)
        self.config: Dict = raw_cfg.get("vehicle_monitoring", raw_cfg)
        self.enabled: bool = bool(self.config.get("enabled", True))

        self.projector = GroundPlaneProjector(self.config)
        self.detector = ForkliftDetector(self.config)
        self.monitor = DriverVehicleMonitor(self.config, self.projector)
        self.alarm_logger = AlarmAndEventLogger(self.config)
        self.dashboard = VehicleDashboardRenderer(self.config)

    def _normalize_person_tracks(
        self, raw_person_tracks: Optional[List[Any]]
    ) -> List[PersonObservation]:
        """
        Adapts person tracks from the existing MVP (dicts or objects with
        id/global_id, bbox, keypoints, foot_point, embedding) into PersonObservation.
        """
        observations: List[PersonObservation] = []
        if not raw_person_tracks:
            return observations

        for item in raw_person_tracks:
            if isinstance(item, PersonObservation):
                observations.append(item)
            elif isinstance(item, dict):
                gid = int(item.get("global_id", item.get("id", item.get("track_id", -1))))
                bbox = tuple(float(x) for x in item["bbox"])
                kpts = item.get("keypoints")
                foot_px = item.get("foot_px", item.get("foot_point"))
                emb = item.get("reid_embedding", item.get("embedding"))
                observations.append(
                    PersonObservation(
                        global_id=gid,
                        bbox=bbox,
                        keypoints=np.array(kpts) if kpts is not None else None,
                        foot_px=tuple(map(int, foot_px)) if foot_px is not None else None,
                        reid_embedding=np.array(emb) if emb is not None else None,
                    )
                )
            else:
                gid = int(
                    getattr(item, "global_id", getattr(item, "track_id", getattr(item, "id", -1)))
                )
                bbox = tuple(float(x) for x in getattr(item, "bbox"))
                kpts = getattr(item, "keypoints", None)
                foot_px = getattr(item, "foot_px", getattr(item, "foot_point", None))
                emb = getattr(item, "reid_embedding", getattr(item, "embedding", None))
                observations.append(
                    PersonObservation(
                        global_id=gid,
                        bbox=bbox,
                        keypoints=np.array(kpts) if kpts is not None else None,
                        foot_px=tuple(map(int, foot_px)) if foot_px is not None else None,
                        reid_embedding=np.array(emb) if emb is not None else None,
                    )
                )
        return observations

    def process_frame(
        self,
        frame: np.ndarray,
        person_tracks: Optional[List[Any]] = None,
    ) -> np.ndarray:
        """
        Processes a single video frame:
        1. Detects and tracks forklifts
        2. Pairs drivers, detects exits, and calculates real-world distance in meters
        3. Triggers audible/visual alarms and persists SQLite/JSONL/CSV logs + snapshots
        4. Renders the monitoring HUD onto the frame
        """
        if not self.enabled:
            return frame

        vehicle_tracks = self.detector.track_vehicles(frame)
        normalized_persons = self._normalize_person_tracks(person_tracks)

        sessions, safety_events = self.monitor.update(
            vehicle_tracks=vehicle_tracks,
            person_observations=normalized_persons,
        )

        if safety_events:
            self.alarm_logger.process_events(safety_events, frame=frame)

        annotated_frame = self.dashboard.draw(
            frame=frame,
            sessions=sessions,
            persons=normalized_persons,
            recent_events=self.alarm_logger.get_recent_events(limit=5),
        )
        return annotated_frame