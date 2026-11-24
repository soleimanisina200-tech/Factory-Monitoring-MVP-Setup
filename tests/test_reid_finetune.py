"""Self-supervised Re-ID fine-tuning: segment mining, loss, metric, and a tiny end-to-end run (CPU)."""
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from app.detection.detector import PoseDetection
from app.identity.finetune import (
    MinedData,
    SegmentBuilder,
    batch_hard_triplet,
    sample_group_batch,
    separation_report,
)
from app.identity.reid_model import CropQualityReport

ROOT = Path(__file__).resolve().parent.parent


def test_segments_end_at_overlaps_and_gaps():
    b = SegmentBuilder("v", max_gap_sec=1.0)
    s1 = b.update(0.0, [(1, True), (2, True)])
    assert b.update(0.5, [(1, True), (2, True)]) == s1
    s2 = b.update(1.0, [(1, False), (2, True)])              # track 1 overlaps somebody: no segment
    assert 1 not in s2 and s2[2] == s1[2]
    s3 = b.update(1.5, [(1, True), (2, True)])
    assert s3[1] != s1[1]                                     # after the overlap: a NEW segment (tracker may swap)
    s4 = b.update(4.0, [(2, True)])
    assert s4[2] != s1[2]                                     # unseen too long: new segment


def test_batch_hard_triplet_is_zero_when_people_are_separated():
    import torch

    f = torch.tensor([[1, 0, 0], [0.99, 0.1, 0], [0, 1, 0], [0.1, 0.99, 0]], dtype=torch.float32)
    lab = torch.tensor([0, 0, 1, 1])
    assert float(batch_hard_triplet(f, lab, 0.3)) == 0.0
    mixed = torch.tensor([[1, 0, 0], [0, 1, 0], [0.99, 0.1, 0], [0.1, 0.99, 0]], dtype=torch.float32)
    assert float(batch_hard_triplet(mixed, lab, 0.3)) > 0.5


def test_separation_report_measures_what_passes_the_open_set_threshold():
    rng = np.random.default_rng(0)
    imp = rng.normal(0.80, 0.03, 2000)
    good = separation_report(rng.normal(0.95, 0.02, 500), imp)
    poor = separation_report(rng.normal(0.86, 0.04, 500), imp)
    assert 0.88 < good["threshold"] < 0.93
    assert good["genuine_accept"] > 0.95 > poor["genuine_accept"]
    assert good["eer"] < poor["eer"]


class FakeDetector:
    def __init__(self, script):
        self.script, self.k = script, 0

    def track_frame(self, frame, bound):
        dets = self.script(self.k)
        self.k += 1
        return dets


class FakePrep:
    def prepare_inputs(self, frame, bbox, kxy, kconf, inter):
        x = int(bbox[0])
        img = np.full((256, 128, 3), (x * 3) % 255, np.uint8)
        return img, img, CropQualityReport(True, True, 50.0, 1.0, "FRONT", "OK")


def _det(tid, x, y=100.0, h=200.0):
    return PoseDetection(track_id=tid, bbox_xyxy=np.array([x, y, x + 60, y + h], np.float32), conf=0.9,
                         keypoints_xy=np.zeros((17, 2), np.float32), keypoints_conf=np.zeros(17, np.float32))


def test_mining_saves_segments_and_who_was_visible_together(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import mine_reid_data

    video = tmp_path / "clip.mp4"
    vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (640, 480))
    for _ in range(40):
        vw.write(np.zeros((480, 640, 3), np.uint8))
    vw.release()
    out = tmp_path / "mined"
    det = FakeDetector(lambda k: [_det(1, 100.0), _det(2, 400.0)] if k < 20 else [_det(1, 100.0)])
    with open(out.mkdir() or out / "samples.jsonl", "w") as sf, open(out / "cooccurrence.jsonl", "w") as cf:
        n = mine_reid_data.mine_video(str(video), out, det, FakePrep(), 0.5, 100.0, 0, sf, cf, 1)
    assert n >= 8
    data = MinedData.load(str(out), min_samples=2)
    assert len(data.segments) == 2 and data.groups == [sorted(data.segments)]
    first = next(iter(data.segments.values()))[0]
    assert (out / first[0]).exists() and (out / first[1]).exists()


def _synthetic_mined(root: Path, n_videos=3, people=4, samples=8):
    """Each segment = a person with a distinct colour pattern; all people of a video visible together."""
    rng = np.random.default_rng(0)
    (root / "images").mkdir(parents=True)
    with open(root / "samples.jsonl", "w") as sf, open(root / "cooccurrence.jsonl", "w") as cf:
        for v in range(n_videos):
            segs = []
            for p in range(people):
                seg = f"v{v}_{p:06d}"
                segs.append(seg)
                (root / "images" / seg).mkdir()
                base = rng.integers(0, 255, 3)
                for k in range(samples):
                    img = np.clip(base + rng.normal(0, 12, (256, 128, 3)), 0, 255).astype(np.uint8)
                    for kind in ("body", "upper"):
                        cv2.imwrite(str(root / "images" / seg / f"{k:04d}_{kind}.jpg"), img)
                    sf.write(json.dumps({"seg": seg, "video": f"v{v}", "ts": k * 0.5,
                                         "body": f"images/{seg}/{k:04d}_body.jpg",
                                         "upper": f"images/{seg}/{k:04d}_upper.jpg"}) + "\n")
            cf.write(json.dumps({"video": f"v{v}", "ts": 1.0, "segs": segs}) + "\n")


def test_split_holds_out_whole_videos_and_sampler_respects_groups(tmp_path):
    _synthetic_mined(tmp_path)
    data = MinedData.load(str(tmp_path))
    tr, va = data.split_by_video(0.34, seed=1)
    assert set(tr.video_of.values()).isdisjoint(va.video_of.values()) and va.segments
    batch = sample_group_batch(tr, random.Random(0), n_groups=3, p=3, k=2)
    for group in batch:
        vids = {tr.video_of[s] for s, _ in group}
        assert len(vids) == 1 and all(len(idx) == 2 for _, idx in group)


def test_tiny_training_run_end_to_end(tmp_path):
    pytest.importorskip("torchreid")
    sys.path.insert(0, str(ROOT / "scripts"))
    import train_reid

    _synthetic_mined(tmp_path / "mined")
    args = SimpleNamespace(data=str(tmp_path / "mined"), config=str(ROOT / "configs" / "settings.yaml"),
                           model_name="osnet_x0_25", weights="none", out=str(tmp_path / "w.pt"), epochs=1, iters=2,
                           groups=2, p=2, k=2, lr=1e-4, margin=0.3, l2sp=1e-3, val_fraction=0.34, device="cpu",
                           seed=0, eval_only=False)
    rc = train_reid.train(args)
    assert rc in (0, 1)                       # 1 = "no improvement" is a valid, reported outcome
    if rc == 0:
        import torch
        assert "state_dict" in torch.load(tmp_path / "w.pt", map_location="cpu", weights_only=True)
