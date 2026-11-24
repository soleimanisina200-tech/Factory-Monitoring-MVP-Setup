"""
Train the forklift detector on the dataset from scripts/prepare_vehicle_dataset.py and report how good it is.

    python scripts/train_vehicle_model.py --data data/forklift_dataset/data.yaml
    python scripts/train_vehicle_model.py --data data/forklift_dataset/data.yaml --model yolo11l-seg.pt --epochs 150

A segmentation model (-seg, chosen automatically for polygon labels) gives the real outline of the chassis on the
floor, which makes the driver-to-forklift distance more accurate. At the end it prints precision / recall / mAP on
the held-out videos and the lines for configs/vehicle_monitoring.yaml. Recall matters most for safety (a forklift
that is not detected cannot raise an alarm); check missed forklifts on the validation images before using it.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml


def detect_task(data_yaml: str) -> str:
    first = Path(data_yaml).read_text(encoding="utf-8").splitlines()[0] if Path(data_yaml).exists() else ""
    return "segment" if "segment" in first else "detect"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train / validate the forklift detector")
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default=None, help="start weights (default yolo11m-seg.pt or yolo11m.pt)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1280, help="large: forklifts far from the camera are small")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default=None, help="final weights (default weights/forklift_<model>.pt)")
    ap.add_argument("--project", default="runs/forklift")
    args = ap.parse_args(argv)

    from ultralytics import YOLO

    task = detect_task(args.data)
    base = args.model or ("yolo11m-seg.pt" if task == "segment" else "yolo11m.pt")
    if task == "segment" and "-seg" not in Path(base).name:
        print(f"warning: polygon labels but a box model ({base}); use a -seg model for the floor outline")
    names = yaml.safe_load(Path(args.data).read_text(encoding="utf-8")).get("names", {})
    names = list(names.values()) if isinstance(names, dict) else list(names)

    model = YOLO(base)
    # mosaic / scale augmentations help with far and partly hidden forklifts; no vertical flips (they never occur)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
                project=args.project, name=Path(base).stem, exist_ok=True, flipud=0.0, fliplr=0.5,
                patience=30, plots=True)
    best = Path(getattr(model.trainer, "best", "") or "")
    if not best.exists():
        print("training finished without a best.pt - see the log above")
        return 1
    final = YOLO(str(best))
    m = final.val(data=args.data, imgsz=args.imgsz, device=args.device, split="val")
    res = m.seg if task == "segment" and hasattr(m, "seg") else m.box
    out = Path(args.out or f"weights/forklift_{Path(base).stem}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out)
    print(f"\nheld-out videos: precision {res.mp:.3f}  recall {res.mr:.3f}  mAP50 {res.map50:.3f}  "
          f"mAP50-95 {res.map:.3f}")
    if res.mr < 0.9:
        print("recall below 0.90: forklifts are missed - add labelled images of the missed situations and retrain")
    print(f"saved {out}\nconfigs/vehicle_monitoring.yaml:\n  vehicle_detector_weights: {out.as_posix()}\n"
          f"  vehicle_class_names: {names}\n  vehicle_class_ids: []")
    return 0


if __name__ == "__main__":
    sys.exit(main())
