# app/core/config.py
"""
settings.yaml loading with defaults and validation.

A typo or a wrong type in a hand-edited YAML file used to be silently replaced by a hard-coded default deep inside
the code. Now every key has ONE default here, user values are merged on top, types are checked, and unknown keys
are reported at start-up (they are usually typos).
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

logger = logging.getLogger("FactoryReID")


class ConfigError(ValueError):
    pass


DEFAULT_SETTINGS: Dict[str, Any] = {
    "system": {
        "camera_id": "cam1",
        "device": "cuda:0",
        "half_precision": True,
        "display_gui": True,
        "log_dir": "logs",
        "log_level": "INFO",
        "heartbeat_path": "logs/heartbeat.json",
        "max_consecutive_errors": 100,
    },
    "stream": {
        "source": "data/input/test_video.mp4",
        "reconnect_delay_sec": 3.0,
        "max_reconnect_delay_sec": 30.0,
        "drop_frames_on_lag": True,
        "rtsp_transport": "tcp",
        "read_timeout_sec": 5.0,
        "max_stall_sec": 0.0,
        "recording_start_time": "",
    },
    "recording": {
        "output_video_path": "auto",   # auto = outputs/<video name>_annotated.mp4 | explicit path | "" = none
        "segment_minutes": 0.0,
        "max_segments": 96,
    },
    "detection": {
        "model_weights": "yolo11m-pose.pt",
        "tracker_config": "configs/custom_bytetrack.yaml",
        "conf_thres": 0.08,
        "iou_thres": 0.6,
        "imgsz": 1536,
    },
    "reid": {
        "model_name": "osnet_x1_0",
        "weights_path": "weights/osnet_x1_0_msmt17.pt",
        "kpt_conf_thresh": 0.2,
        "min_laplacian_var": 10.0,
        "head_branch_weight": 0.35,
        "color_branch_weight": 0.2,
        "cos_sim_thresh": 0.6,
        "spatial_recovery_sim_thresh": 0.48,
        "reentry_sim_thresh": 0.82,
        "reid_interval_sec": 0.35,
        "stitch_sim_thresh": 0.55,
        "stitch_margin": 0.04,
        "reentry_min_gap_sec": 2.0,
        "reentry_margin": 0.08,
        "reentry_min_samples": 3,
        "recover_strict_after_sec": 8.0,
        "exit_confirm_sec": 0.8,
        "exit_edges": ["left", "right", "top", "bottom"],
        "mismatch_min_span_sec": 1.0,
        "min_gallery_height_px": 72,
        "min_visible_kpts": 5,
        "anchor_min_height_px": 110,
        "track_state_ttl_sec": 10.0,
        "identity_retention_sec": 43200.0,   # 12 h: lunch and shift breaks
        "max_exited_identities": 500,
        "occlusion_max_sec": 20.0,
        "border_margin_ratio": 0.06,
        "pool_max_samples": 8,
        "pool_window_sec": 4.0,
        "track_gap_reset_sec": 1.5,
        "decision_log_path": "logs/identity_decisions.jsonl",
        "open_set": {
            "enabled": True,
            "quantile": 0.995,
            "margin": 0.03,
            "min_samples": 60,
            "max_threshold": 0.95,
            "sample_interval_sec": 1.0,
        },
        "persistence": {
            "mode": "auto",            # auto = on for live cameras, off for video files
            "dir": "data/identity_store",
            "save_interval_sec": 60.0,
        },
    },
    "markers": {
        "enabled": False,
        "registry_path": "configs/workers.yaml",
        "dictionary": "DICT_4X4_50",
        "accept_unregistered": False,
        "read_interval_sec": 0.5,
        "min_votes": 2,
        "vote_window_sec": 4.0,
        "min_person_height_px": 120,
    },
    "physics_bev": {
        "max_worker_speed_mps": 3.5,
        "max_internal_occlusion_m": 5.5,
        "lag_diffusion_coeff": 0.65,
        "base_pose_jitter_m": 0.7,
        "default_px_per_meter": 110.0,
        "calibration_ref_wh": None,
        "src_image_pts": None,
        "dst_meter_pts": None,
    },
    "scene": {"path": "configs/scene.yaml"},
    "zones": {
        "entry_confirm_sec": 0.5,
        "exit_confirm_sec": 1.0,
        "lost_timeout_sec": 15.0,
        "include_unverified": True,
        "definitions": {},
    },
    "crossings": {
        "dead_band_px": 18.0,
        "cooldown_sec": 1.0,
        "max_gap_sec": 4.0,
        "segment_margin": 0.10,
        "region_exit_confirm_sec": 0.3,
        "region_lost_sec": 3.0,
        "draw_trails": True,
    },
    "attribution": {
        "timeout_sec": 20.0,
        "record_unidentified": True,
    },
    "events": {
        "db_path": "data/factory_events.db",
        "csv_dir": "data/events",
        "snapshot_dir": "data/snapshots",
        "snapshot_event_types": [],
        "retention_days": 90.0,
    },
    "alarms": {
        "enable_audio_alarm": True,
        "alarm_cooldown_sec": 5.0,
        "audio_frequency_hz": 2200,
        "audio_duration_ms": 400,
        "forward_min_severity": "alarm",
        "webhook_urls": [],
        "webhook_headers": {},
        "webhook_timeout_sec": 3.0,
        "mqtt": {"enabled": False, "host": "127.0.0.1", "port": 1883, "topic": "factory/alarms",
                 "username": "", "password": ""},
    },
    "web": {
        "enabled": False,
        "host": "127.0.0.1",
        "port": 8050,
        "token": "",
        "stream_max_fps": 10.0,
        "stream_width": 1280,
    },
    # Same worker = same PERSON id in every camera (app/multicam/registry.py). Normally set in configs/cameras.yaml.
    "cross_camera": {
        "enabled": False,
        "registry_path": "data/cross_camera.db",
        "sync_interval_sec": 2.0,
        "link_threshold": 0.80,
        "margin": 0.06,
        "min_obs": 5,
        "decide_after_sec": 20.0,
        "late_merge_sec": 120.0,
        "retention_sec": 43200.0,
        "topology": {"overlaps": [], "transitions": [], "only_listed_transitions": False},
    },
    # legacy section (old draw_line.py output); read by app/core/scene.py
    "directional_tracking": {},
}

# Keys whose value type is free-form (e.g. camera index int OR url string; YAML reads a bare on/off as a boolean).
_FREE_TYPED = {("stream", "source"), ("directional_tracking",), ("zones", "definitions"),
               ("reid", "persistence", "mode")}


def persistence_mode(value: Any) -> str:
    """'auto' | 'on' | 'off' (YAML turns a bare `on` / `off` into True / False)."""
    if isinstance(value, bool):
        return "on" if value else "off"
    return str(value or "auto").strip().lower()


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _type_ok(default: Any, value: Any) -> bool:
    if default is None or value is None:
        return True
    if isinstance(default, bool):
        return isinstance(value, bool)
    if isinstance(default, (int, float)):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if isinstance(default, str):
        return isinstance(value, str)
    if isinstance(default, list):
        return isinstance(value, list)
    if isinstance(default, dict):
        return isinstance(value, dict)
    return True


def validate(user: Dict[str, Any], defaults: Dict[str, Any] = DEFAULT_SETTINGS,
             path: Tuple[str, ...] = ()) -> Tuple[List[str], List[str]]:
    """Returns (errors, warnings)."""
    errors: List[str] = []
    warnings: List[str] = []
    for k, v in (user or {}).items():
        p = path + (str(k),)
        if p in _FREE_TYPED or p[:1] in _FREE_TYPED:
            continue
        if k not in defaults:
            warnings.append(f"unknown key '{'.'.join(p)}' (typo?) - ignored")
            continue
        d = defaults[k]
        if not _type_ok(d, v):
            errors.append(f"'{'.'.join(p)}' must be {type(d).__name__}, got {type(v).__name__} ({v!r})")
            continue
        if isinstance(d, dict) and isinstance(v, dict) and d:
            e, w = validate(v, d, p)
            errors += e
            warnings += w
    return errors, warnings


def load_settings(path: str = "configs/settings.yaml") -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(p, "r", encoding="utf-8") as f:
        user = yaml.safe_load(f) or {}
    if not isinstance(user, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    # backwards compatibility: output_video_path used to live in `system`
    sys_cfg = user.get("system") or {}
    if "output_video_path" in sys_cfg:
        user.setdefault("recording", {}).setdefault("output_video_path", sys_cfg.pop("output_video_path"))
    errors, warnings = validate(user)
    for w in warnings:
        logger.warning("settings: %s", w)
    if errors:
        raise ConfigError(f"{path}: " + "; ".join(errors))
    return deep_merge(DEFAULT_SETTINGS, user)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
