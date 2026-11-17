"""
Installation and configuration check. Run it after installing and after every configuration change:

    python scripts/diagnostic.py
    python scripts/diagnostic.py --cameras configs/cameras.yaml       (multi-camera setup)

Exit code 0 = no blocking problem (warnings may remain), 1 = something must be fixed first.
"""
import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROBLEMS, WARNINGS = [], []


def ok(msg):
    print(f"  [ OK ] {msg}")


def warn(msg):
    WARNINGS.append(msg)
    print(f"  [WARN] {msg}")


def bad(msg):
    PROBLEMS.append(msg)
    print(f"  [FAIL] {msg}")


def check_packages():
    print("Packages")
    v = sys.version_info
    (ok if v >= (3, 10) else bad)(f"Python {v.major}.{v.minor}.{v.micro} (3.10+ required)")
    for name, required in (("torch", True), ("cv2", True), ("ultralytics", True), ("scipy", True), ("yaml", True),
                           ("torchreid", True), ("gdown", False), ("fastapi", False), ("uvicorn", False),
                           ("paho.mqtt.client", False)):
        try:
            mod = importlib.import_module(name)
            ok(f"{name} {getattr(mod, '__version__', '')}")
        except Exception as e:
            (bad if required else warn)(f"{name} not importable ({type(e).__name__}: {e})"
                                        + ("" if required else " - optional"))
    try:
        import torch

        if torch.cuda.is_available():
            ok(f"CUDA: {torch.cuda.get_device_name(0)} "
               f"({torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB)")
        else:
            warn(f"CUDA not available (torch {torch.__version__}, CUDA build {torch.version.cuda}): ~1 FPS on CPU. "
                 "Install the CUDA build from https://pytorch.org/get-started/locally/")
    except Exception:
        pass
    try:
        import cv2

        (ok if hasattr(cv2, "aruco") and hasattr(cv2.aruco, "ArucoDetector") else warn)(
            "OpenCV ArUco (needed only for helmet markers)")
    except Exception:
        pass


def check_camera(name, cfg, vehicle_config, calibration):
    from app.core.scene import load_scene
    from app.logic.calibration import load_calibration

    print(f"Camera '{name}'")
    src = str(cfg["stream"]["source"])
    if "://" in src or src.isdigit():
        ok("source: live camera")
    elif Path(src).exists():
        ok(f"source: {src}")
    else:
        bad(f"source file not found: {src}")
    try:
        cal = load_calibration(calibration)
        (ok if cal is not None else warn)(
            f"calibration {calibration}" + ("" if cal is not None else
                                            " missing: metric distances (forklift alarm!) are NOT valid"))
    except Exception as e:
        bad(f"calibration {calibration}: {e}")
    try:
        scene = load_scene(cfg["scene"]["path"], cfg)
        ok(f"scene {cfg['scene']['path']}: {len(scene.zones)} zones, {len(scene.lines)} lines, "
           f"{len(scene.regions)} regions, {len(scene.exit_regions)} exit doors"
           + ("" if Path(cfg["scene"]["path"]).exists() else " (file missing: nothing drawn yet)"))
    except Exception as e:
        bad(f"scene: {e}")
    reid = cfg["reid"]
    w = reid.get("weights_path")
    if w and Path(w).exists():
        ok(f"Re-ID weights {w}")
    elif reid["model_name"] == "osnet_x1_0":
        warn(f"Re-ID weights {w} not present: they are downloaded on the first run (internet needed)")
    else:
        bad(f"Re-ID weights {w} missing for {reid['model_name']} (manual download, see README)")
    if cfg["markers"]["enabled"] and not Path(cfg["markers"]["registry_path"]).exists():
        warn(f"markers enabled but {cfg['markers']['registry_path']} is missing")
    if vehicle_config and Path(vehicle_config).exists():
        import yaml

        v = (yaml.safe_load(Path(vehicle_config).read_text(encoding="utf-8")) or {}).get("vehicle_monitoring", {})
        if v.get("enabled", True):
            names = v.get("vehicle_class_names") or []
            if not any("fork" in str(n).lower() for n in names):
                warn("forklift module uses a generic model without a 'forklift' class: train/fine-tune one "
                     "(vehicle_detector_weights + vehicle_class_names: [forklift])")
            else:
                ok(f"forklift model {v.get('vehicle_detector_weights')} classes {names}")
            if cal is None:
                bad("forklift distance alarm needs a ground calibration for this camera")
    else:
        ok("forklift module off for this camera")


def main() -> int:
    ap = argparse.ArgumentParser(description="Installation / configuration check")
    ap.add_argument("--config", default="configs/settings.yaml")
    ap.add_argument("--vehicle-config", default="configs/vehicle_monitoring.yaml")
    ap.add_argument("--calibration", default="configs/calibration.yaml")
    ap.add_argument("--cameras", default=None, help="check every camera of a multi-camera file")
    args = ap.parse_args()

    check_packages()
    try:
        if args.cameras:
            from app.multicam.cameras import camera_ids, load_camera_setup

            for cam in camera_ids(args.cameras):
                s = load_camera_setup(args.cameras, cam)
                check_camera(cam, s.settings, s.vehicle_config, s.calibration)
        else:
            from app.core.config import load_settings

            cfg = load_settings(args.config)
            ok(f"settings {args.config} are valid")
            check_camera(cfg["system"]["camera_id"], cfg, args.vehicle_config, args.calibration)
    except Exception as e:
        bad(f"configuration: {type(e).__name__}: {e}")

    print(f"\n{len(PROBLEMS)} problem(s), {len(WARNINGS)} warning(s)")
    return 1 if PROBLEMS else 0


if __name__ == "__main__":
    raise SystemExit(main())
