"""
Collect Re-ID training data from the site's own videos (no hand labelling), for scripts/train_reid.py.

    python scripts/mine_reid_data.py --source data/input/shift1.mp4 data/input/shift2.mp4 --out data/reid_mining

Several hours from the real cameras (different days, shifts, light) give the best result. For every clean
segment of a track (one person, nobody overlapping) the exact images the Re-ID model sees in production (masked full
body + upper body, 256x128) are saved every --every-sec seconds, and which segments were visible at the same time
(= certainly different people, the hard negatives of a uniform site).

Output:  <out>/images/<segment>/<n>_body.jpg, <n>_upper.jpg
         <out>/samples.jsonl       one line per saved sample (segment, video, time, image paths)
         <out>/cooccurrence.jsonl  segments visible in the same frame
The folder contains images of employees: keep it as protected as the recordings themselves.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import load_settings  # noqa: E402
from app.identity.finetune import SegmentBuilder  # noqa: E402
from app.logic.geometry import compute_intersection_contamination  # noqa: E402


def mine_video(path: str, out: Path, detector, reid, every_sec: float, min_height: float, max_frames: int,
               samples_f, cooc_f, stride: int) -> int:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"cannot open {path}")
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stem = Path(path).stem
    builder = SegmentBuilder(stem, max_gap_sec=1.0)
    last_saved = {}
    counts = {}
    n_saved = 0
    k = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and k >= max_frames):
            break
        k += 1
        if (k - 1) % stride:
            continue
        ts = k / fps
        H, W = frame.shape[:2]
        dets = detector.track_frame(frame, set())
        if not dets:
            builder.update(ts, [])
            continue
        boxes = np.array([d.bbox_xyxy for d in dets], np.float32)
        kxy = np.array([d.keypoints_xy for d in dets], np.float32)
        kc = np.array([d.keypoints_conf for d in dets], np.float32)
        contaminated = compute_intersection_contamination(boxes, kxy, kc)
        clean = []
        for i, d in enumerate(dets):
            x1, y1, x2, y2 = (float(v) for v in d.bbox_xyxy)
            cut = x1 <= 2 or y1 <= 2 or x2 >= W - 2 or y2 >= H - 2
            clean.append((d.track_id, not contaminated[i] and not cut and (y2 - y1) >= min_height))
        seg_of = builder.update(ts, clean)
        visible = []
        for d in dets:
            seg = seg_of.get(d.track_id)
            if seg is None:
                continue
            visible.append(seg)
            if ts - last_saved.get(seg, -1e9) < every_sec:
                continue
            prep = reid.prepare_inputs(frame, d.bbox_xyxy, d.keypoints_xy, d.keypoints_conf, False)
            if prep is None or not prep[2].is_valid_for_gallery:
                continue
            body, upper, _ = prep
            n = counts.get(seg, 0)
            counts[seg] = n + 1
            folder = out / "images" / seg
            folder.mkdir(parents=True, exist_ok=True)
            rel_b = f"images/{seg}/{n:04d}_body.jpg"
            rel_u = f"images/{seg}/{n:04d}_upper.jpg"
            cv2.imwrite(str(out / rel_b), cv2.cvtColor(body, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(out / rel_u), cv2.cvtColor(upper, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            samples_f.write(json.dumps({"seg": seg, "video": stem, "ts": round(ts, 3), "body": rel_b,
                                        "upper": rel_u}) + "\n")
            last_saved[seg] = ts
            n_saved += 1
        if len(visible) >= 2:
            cooc_f.write(json.dumps({"video": stem, "ts": round(ts, 3), "segs": sorted(set(visible))}) + "\n")
        if k % int(fps * 30) == 0:
            print(f"  {stem}: {k / fps:7.0f} s, {n_saved} samples, {len(counts)} segments")
    cap.release()
    return n_saved


def main() -> int:
    ap = argparse.ArgumentParser(description="Mine self-labelled Re-ID data from site videos")
    ap.add_argument("--source", nargs="+", required=True, help="video files (recordings of the real cameras)")
    ap.add_argument("--out", default="data/reid_mining")
    ap.add_argument("--config", default="configs/settings.yaml")
    ap.add_argument("--every-sec", type=float, default=0.5, help="one sample per segment every N seconds")
    ap.add_argument("--min-height", type=float, default=100.0, help="smaller people are skipped (px)")
    ap.add_argument("--stride", type=int, default=1, help="process every N-th frame (faster, coarser tracking)")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from app.detection.detector import IndustrialPoseDetector
    from app.identity.reid_model import PoseGuidedOSNetExtractor

    cfg = load_settings(args.config)
    dev = args.device or cfg["system"]["device"]
    d, r = cfg["detection"], cfg["reid"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reid = PoseGuidedOSNetExtractor(r["model_name"], r["weights_path"], dev, r["kpt_conf_thresh"],
                                    r["min_laplacian_var"], r["head_branch_weight"], r["color_branch_weight"],
                                    r["min_gallery_height_px"], r["min_visible_kpts"])
    total = 0
    with open(out / "samples.jsonl", "a", encoding="utf-8") as sf, \
            open(out / "cooccurrence.jsonl", "a", encoding="utf-8") as cf:
        for src in args.source:
            # a fresh tracker per video (track ids must not continue across files)
            detector = IndustrialPoseDetector(d["model_weights"], d["tracker_config"], dev, d["conf_thres"],
                                              d["iou_thres"], d["imgsz"], bool(cfg["system"]["half_precision"]))
            print(f"mining {src}")
            total += mine_video(src, out, detector, reid, args.every_sec, args.min_height, args.max_frames, sf, cf,
                                max(1, args.stride))
    print(f"done: {total} samples in {out}  (next: python scripts/train_reid.py --data {out})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
