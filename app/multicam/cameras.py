# app/multicam/cameras.py
"""
Multi-camera configuration (configs/cameras.yaml): ONE base settings file + small per-camera overrides.

    base_settings: configs/settings.yaml
    vehicle_config: configs/vehicle_monitoring.yaml
    cross_camera:                      # shared by all cameras (see app/multicam/registry.py)
      registry_path: data/cross_camera.db
      topology:
        overlaps: [[cam1, cam2]]
        transitions: [{from: cam1, to: cam3, min_sec: 10, max_sec: 900}]
    cameras:
      cam1:
        source: rtsp://user:pass@10.0.0.11:554/stream1
        scene: configs/cameras/cam1/scene.yaml
        calibration: configs/cameras/cam1/calibration.yaml
        settings: {reid: {exit_edges: [left, right]}}      # any settings.yaml key
      cam2:
        source: rtsp://user:pass@10.0.0.12:554/stream1
        vehicle_config: ""                                  # no forklifts in this view

Every camera runs in its own process (run_multi_camera.py) with its own logs, heartbeat, decision log, recordings
and identity store; events of all cameras go to ONE database (column camera_id).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from app.core.config import ConfigError, deep_merge, load_settings, validate
from app.logic.calibration import DEFAULT_CALIBRATION_PATH

DEFAULT_CAMERAS_PATH = "configs/cameras.yaml"


@dataclass
class CameraSetup:
    camera_id: str
    settings: Dict[str, Any]
    vehicle_config: Optional[str]
    calibration: str


def load_cameras_file(path: str = DEFAULT_CAMERAS_PATH) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Camera list not found: {path} (copy configs/cameras.example.yaml)")
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data.get("cameras"), dict) or not data["cameras"]:
        raise ConfigError(f"{path}: 'cameras' must list at least one camera")
    return data


def camera_ids(path: str = DEFAULT_CAMERAS_PATH) -> List[str]:
    return [str(c) for c in load_cameras_file(path)["cameras"]]


def _in_camera_folder(path: Optional[str], camera_id: str) -> Optional[str]:
    if not path:
        return path
    p = Path(path)
    return str(p.parent / camera_id / p.name)


def scope_paths(cfg: Dict[str, Any], camera_id: str) -> Dict[str, Any]:
    """Per-camera output files so several camera processes never write to the same file."""
    cfg["system"]["log_dir"] = str(Path(cfg["system"].get("log_dir") or "logs") / camera_id)
    cfg["system"]["heartbeat_path"] = _in_camera_folder(cfg["system"].get("heartbeat_path"), camera_id)
    cfg["reid"]["decision_log_path"] = _in_camera_folder(cfg["reid"].get("decision_log_path"), camera_id)
    cfg["recording"]["output_video_path"] = _in_camera_folder(cfg["recording"].get("output_video_path"), camera_id)
    return cfg


def load_camera_setup(path: str, camera_id: str) -> CameraSetup:
    data = load_cameras_file(path)
    cams = data["cameras"]
    if camera_id not in cams:
        raise ConfigError(f"{path}: camera '{camera_id}' is not defined (known: {sorted(cams)})")
    cam = cams[camera_id] or {}
    cfg = load_settings(data.get("base_settings", "configs/settings.yaml"))

    overrides = cam.get("settings") or {}
    errors, warnings = validate(overrides)
    if errors:
        raise ConfigError(f"{path}: cameras.{camera_id}.settings: " + "; ".join(errors))
    cfg = deep_merge(cfg, overrides)

    cfg["system"]["camera_id"] = str(camera_id)
    if cam.get("source") is not None:
        cfg["stream"]["source"] = cam["source"]
    if "display_gui" in cam:
        cfg["system"]["display_gui"] = bool(cam["display_gui"])
    cfg["scene"]["path"] = cam.get("scene") or f"configs/cameras/{camera_id}/scene.yaml"
    cc = deep_merge(cfg["cross_camera"], data.get("cross_camera") or {})
    cc["enabled"] = bool((data.get("cross_camera") or {}).get("enabled", len(cams) > 1))
    cfg["cross_camera"] = cc
    if "port" not in (overrides.get("web") or {}):
        # one web API per camera process: base port + position of the camera in the list (8050, 8051, ...)
        cfg["web"]["port"] = int(cfg["web"]["port"]) + list(cams).index(camera_id)
    scope_paths(cfg, str(camera_id))

    vehicle = cam.get("vehicle_config", data.get("vehicle_config", "configs/vehicle_monitoring.yaml"))
    calibration = cam.get("calibration") or f"configs/cameras/{camera_id}/calibration.yaml"
    if not Path(calibration).exists() and len(cams) == 1 and Path(DEFAULT_CALIBRATION_PATH).exists():
        calibration = DEFAULT_CALIBRATION_PATH
    return CameraSetup(str(camera_id), cfg, vehicle or None, calibration)
