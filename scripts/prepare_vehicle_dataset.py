"""
Build a forklift (vehicle) dataset from the site's own videos for scripts/train_vehicle_model.py.

The public COCO model has NO forklift class, so driver monitoring needs a model trained on THIS site's forklifts
(with and without roof, loaded / empty, all camera angles, day and night).

Step 1 - extract frames worth labelling (near-duplicates of a static scene are skipped):
    python scripts/prepare_vehicle_dataset.py extract --source data/input/cam1.mp4 data/input/cam2.mp4 \
        --out data/forklift_dataset --every-sec 2
  optional pre-labels from any YOLO model (boxes to CORRECT, not to trust):
        --prelabel yolo11m.pt --prelabel-classes truck
  -> data/forklift_dataset/unlabeled/images/*.jpg (+ labels/*.txt in YOLO format when pre-labelled)

Step 2 - label them (CVAT, Label Studio, Roboflow, ...) in YOLO format, class 0 = forklift. Segmentation polygons
  (YOLO-seg format) give a better floor footprint and therefore better distances; boxes also work. An image without
  forklift needs an EMPTY .txt (negatives teach the model what is NOT a forklift: pallets, trucks, racks).
  Put the result in data/forklift_dataset/labeled/images and .../labeled/labels.

Step 3 - check and split (whole videos go to validation, never frames of a training video):
    python scripts/prepare_vehicle_dataset.py split --out data/forklift_dataset --names forklift
  -> data/forklift_dataset/data.yaml  (then: python scripts/train_vehicle_model.py --data data/forklift_dataset/data.yaml)
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


# ------------------------------------------------------------------------------------------------- extract
def _thumb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)


def _similar(a: np.ndarray, b: np.ndarray, threshold: float) -> bool:
    if float(np.abs(a - b).mean()) <= 4.0:          # practically the same picture (also dark / flat / noisy scenes)
        return True
    sa, sb = float(a.std()), float(b.std())
    if sa < 2.0 or sb < 2.0:
        return False
    return float((((a - a.mean()) / sa) * ((b - b.mean()) / sb)).mean()) >= threshold


def is_duplicate(thumb: np.ndarray, recent: List[np.ndarray], threshold: float) -> bool:
    """A parked forklift in an empty hall gives the same frame for hours: keep it once."""
    return any(_similar(thumb, r, threshold) for r in recent)


def extract(sources: Sequence[str], out: Path, every_sec: float, dedup: float, max_per_video: int,
            prelabeler=None) -> int:
    img_dir, lab_dir = out / "unlabeled" / "images", out / "unlabeled" / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for src in sources:
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            print(f"cannot open {src}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, int(round(every_sec * fps)))
        stem = Path(src).stem
        recent: List[np.ndarray] = []
        k = n = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if k % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    th = _thumb(frame)
                    if not is_duplicate(th, recent, dedup):
                        recent = (recent + [th])[-30:]
                        name = f"{stem}__{int(k / fps):06d}s"
                        cv2.imwrite(str(img_dir / f"{name}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        if prelabeler is not None:
                            lab_dir.mkdir(parents=True, exist_ok=True)
                            (lab_dir / f"{name}.txt").write_text(prelabeler(frame), encoding="utf-8")
                        n += 1
                        if max_per_video and n >= max_per_video:
                            break
            k += 1
        cap.release()
        print(f"{src}: {n} frames")
        total += n
    return total


def make_prelabeler(weights: str, class_names: Sequence[str], conf: float = 0.25):
    """YOLO boxes of the given classes -> YOLO label text with class 0 (to be corrected by a person)."""
    from ultralytics import YOLO

    model = YOLO(weights)
    names = {int(k): str(v) for k, v in model.names.items()}
    wanted = {i for i, n in names.items() if n in set(class_names)}
    if not wanted:
        raise SystemExit(f"none of {list(class_names)} in the model classes {sorted(names.values())}")

    def run(frame: np.ndarray) -> str:
        r = model.predict(frame, conf=conf, verbose=False)[0]
        h, w = frame.shape[:2]
        lines = []
        for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()):
            if int(c) in wanted:
                lines.append(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        return "\n".join(lines) + ("\n" if lines else "")

    return run


# ------------------------------------------------------------------------------------------------- split
def check_label_file(path: Path, n_classes: int) -> Tuple[int, str, Optional[str]]:
    """(number of objects, 'det'|'seg'|'empty', error or None)."""
    kinds, n = set(), 0
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        try:
            cls = int(parts[0])
            vals = [float(v) for v in parts[1:]]
        except ValueError:
            return n, "?", f"{path.name}:{i}: not numbers"
        if not 0 <= cls < n_classes:
            return n, "?", f"{path.name}:{i}: class {cls} but only {n_classes} class name(s) given"
        if len(vals) == 4:
            kinds.add("det")
        elif len(vals) >= 6 and len(vals) % 2 == 0:
            kinds.add("seg")
        else:
            return n, "?", f"{path.name}:{i}: {len(vals)} values (box = 4, polygon = even number >= 6)"
        if any(v < -0.01 or v > 1.01 for v in vals):
            return n, "?", f"{path.name}:{i}: coordinates must be normalised to 0..1"
        n += 1
    if len(kinds) > 1:
        return n, "?", f"{path.name}: mixes boxes and polygons"
    return n, (kinds.pop() if kinds else "empty"), None


def video_of(image_name: str) -> str:
    return image_name.split("__")[0]


def split(out: Path, names: Sequence[str], val_fraction: float, seed: int) -> Dict[str, object]:
    src_img, src_lab = out / "labeled" / "images", out / "labeled" / "labels"
    images = sorted(p for p in src_img.glob("*") if p.suffix.lower() in IMG_EXT)
    if not images:
        raise SystemExit(f"no images in {src_img}")
    errors, missing = [], []
    stats = defaultdict(int)
    kinds = set()
    for img in images:
        lab = src_lab / f"{img.stem}.txt"
        if not lab.exists():
            missing.append(img.name)
            continue
        n, kind, err = check_label_file(lab, len(names))
        if err:
            errors.append(err)
            continue
        if kind != "empty":
            kinds.add(kind)
        stats["objects"] += n
        stats["with_objects" if n else "negatives"] += 1
    if missing:
        errors.append(f"{len(missing)} image(s) without a label file (an image without forklift needs an EMPTY "
                      f".txt), e.g. {missing[:3]}")
    if len(kinds) > 1:
        errors.append("some files use boxes, others polygons: use one format for the whole dataset")
    if errors:
        raise SystemExit("label check failed:\n  " + "\n  ".join(errors[:20]))

    videos = sorted({video_of(p.stem) for p in images})
    rng = random.Random(seed)
    rng.shuffle(videos)
    if len(videos) > 1:
        n_val = max(1, int(round(val_fraction * len(videos))))
        val_videos = set(videos[:n_val])
        in_val = lambda p: video_of(p.stem) in val_videos                      # noqa: E731
    else:   # one video: hold out its last part in time (frames are named by second)
        order = sorted(images, key=lambda p: p.stem)
        cut = {p.name for p in order[int(len(order) * (1.0 - val_fraction)):]}
        in_val = lambda p: p.name in cut                                          # noqa: E731
    for sub in ("train", "val"):
        for kind in ("images", "labels"):
            d = out / kind / sub
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)
    counts = defaultdict(int)
    for img in images:
        sub = "val" if in_val(img) else "train"
        shutil.copy2(img, out / "images" / sub / img.name)
        shutil.copy2(src_lab / f"{img.stem}.txt", out / "labels" / sub / f"{img.stem}.txt")
        counts[sub] += 1
    task = "segment" if kinds == {"seg"} else "detect"
    yaml_lines = [f"# {task} dataset built by scripts/prepare_vehicle_dataset.py", f"path: {out.resolve().as_posix()}",
                  "train: images/train", "val: images/val", "names:"]
    yaml_lines += [f"  {i}: {n}" for i, n in enumerate(names)]
    (out / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    report = {"task": task, "train": counts["train"], "val": counts["val"], **stats}
    print(f"dataset: {report}")
    if stats["with_objects"] < 200:
        print("note: fewer than ~200 labelled forklift images usually gives an unreliable model; add more variety "
              "(other forklifts, roof / no roof, loaded, night, every camera).")
    if counts["val"] == 0:
        print("warning: empty validation set")
    print(f"next: python scripts/train_vehicle_model.py --data {(out / 'data.yaml').as_posix()}")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Forklift dataset: extract frames, check labels, split")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--source", nargs="+", required=True)
    e.add_argument("--out", default="data/forklift_dataset")
    e.add_argument("--every-sec", type=float, default=2.0)
    e.add_argument("--dedup", type=float, default=0.97, help="skip frames this similar to a recent one (0..1)")
    e.add_argument("--max-per-video", type=int, default=0)
    e.add_argument("--prelabel", default=None, help="YOLO weights for draft boxes, e.g. yolo11m.pt")
    e.add_argument("--prelabel-classes", nargs="*", default=["truck"])
    s = sub.add_parser("split")
    s.add_argument("--out", default="data/forklift_dataset")
    s.add_argument("--names", nargs="+", default=["forklift"])
    s.add_argument("--val-fraction", type=float, default=0.2)
    s.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "extract":
        pre = make_prelabeler(args.prelabel, args.prelabel_classes) if args.prelabel else None
        n = extract(args.source, Path(args.out), args.every_sec, args.dedup, args.max_per_video, pre)
        print(f"{n} frames in {Path(args.out) / 'unlabeled'} - label them, then run the 'split' step")
    else:
        split(Path(args.out), args.names, args.val_fraction, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
