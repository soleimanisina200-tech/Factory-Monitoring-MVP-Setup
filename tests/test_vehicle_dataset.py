"""Forklift dataset tool: frame extraction with de-duplication, label checks, split by video, training wrapper."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_vehicle_dataset as pvd  # noqa: E402


def _video(path, frames):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240))
    for f in frames:
        vw.write(f)
    vw.release()


def test_static_scene_is_extracted_once_moving_scene_many_times(tmp_path):
    rng = np.random.default_rng(0)
    static = [np.full((240, 320, 3), 90, np.uint8)] * 100
    base = rng.integers(0, 255, (240, 320, 3)).astype(np.uint8)
    moving = []
    for k in range(100):
        f = (base // 3).copy()
        cv2.rectangle(f, (5 + 3 * k, 60), (70 + 3 * k, 160), (255, 255, 255), -1)   # a forklift crossing
        moving.append(f)
    _video(tmp_path / "parked.mp4", static)
    _video(tmp_path / "busy.mp4", moving)
    n = pvd.extract([str(tmp_path / "parked.mp4"), str(tmp_path / "busy.mp4")], tmp_path / "ds", every_sec=1.0,
                    dedup=0.97, max_per_video=0, prelabeler=lambda fr: "0 0.5 0.5 0.2 0.2\n")
    imgs = sorted((tmp_path / "ds" / "unlabeled" / "images").glob("*.jpg"))
    assert sum(1 for p in imgs if p.name.startswith("parked__")) == 1
    assert sum(1 for p in imgs if p.name.startswith("busy__")) >= 5
    assert n == len(imgs) and all((tmp_path / "ds" / "unlabeled" / "labels" / f"{p.stem}.txt").exists() for p in imgs)


def _labeled(root, items):
    (root / "labeled" / "images").mkdir(parents=True)
    (root / "labeled" / "labels").mkdir(parents=True)
    for name, text in items.items():
        cv2.imwrite(str(root / "labeled" / "images" / f"{name}.jpg"), np.zeros((32, 32, 3), np.uint8))
        if text is not None:
            (root / "labeled" / "labels" / f"{name}.txt").write_text(text)


def test_split_holds_out_whole_videos_and_writes_data_yaml(tmp_path):
    box = "0 0.5 0.5 0.3 0.2\n"
    _labeled(tmp_path, {**{f"camA__{i:06d}s": box for i in range(6)}, **{f"camB__{i:06d}s": box for i in range(4)},
                        **{f"camC__{i:06d}s": "" for i in range(3)}})
    rep = pvd.split(tmp_path, ["forklift"], 0.34, seed=3)
    train = {pvd.video_of(p.stem) for p in (tmp_path / "images" / "train").glob("*.jpg")}
    val = {pvd.video_of(p.stem) for p in (tmp_path / "images" / "val").glob("*.jpg")}
    assert train and val and train.isdisjoint(val)
    assert rep["task"] == "detect" and rep["negatives"] == 3 and rep["objects"] == 10
    text = (tmp_path / "data.yaml").read_text()
    assert "detect dataset" in text and "0: forklift" in text


def test_polygon_labels_make_a_segmentation_dataset(tmp_path):
    poly = "0 0.1 0.1 0.4 0.1 0.4 0.5 0.1 0.5\n"
    _labeled(tmp_path, {f"cam{c}__{i:06d}s": poly for c in "AB" for i in range(3)})
    assert pvd.split(tmp_path, ["forklift"], 0.5, seed=0)["task"] == "segment"


@pytest.mark.parametrize("bad, msg", [
    ("1 0.5 0.5 0.2 0.2\n", "class 1"),
    ("0 0.5 0.5 0.2\n", "values"),
    ("0 50 60 120 80\n", "normalised"),
    (None, "without a label file"),
])
def test_label_problems_are_reported_clearly(tmp_path, bad, msg):
    _labeled(tmp_path, {"camA__000001s": "0 0.5 0.5 0.2 0.2\n", "camB__000001s": bad})
    with pytest.raises(SystemExit) as e:
        pvd.split(tmp_path, ["forklift"], 0.5, seed=0)
    assert msg in str(e.value)


def test_training_wrapper_trains_validates_and_prints_the_config(tmp_path, monkeypatch, capsys):
    import types

    import train_vehicle_model as tvm

    data = tmp_path / "data.yaml"
    data.write_text("# segment dataset built by scripts/prepare_vehicle_dataset.py\npath: x\nnames:\n  0: forklift\n")
    best = tmp_path / "best.pt"
    best.write_bytes(b"weights")
    calls = {}

    class FakeYOLO:
        def __init__(self, w):
            calls.setdefault("init", []).append(w)
            self.trainer = types.SimpleNamespace(best=str(best))

        def train(self, **kw):
            calls["train"] = kw

        def val(self, **kw):
            r = types.SimpleNamespace(mp=0.93, mr=0.95, map50=0.96, map=0.71)
            return types.SimpleNamespace(seg=r, box=r)

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    out = tmp_path / "w" / "fk.pt"
    assert tvm.main(["--data", str(data), "--out", str(out), "--epochs", "1"]) == 0
    assert calls["init"][0] == "yolo11m-seg.pt" and calls["train"]["flipud"] == 0.0 and out.exists()
    printed = capsys.readouterr().out
    assert "recall 0.950" in printed and "vehicle_class_names: ['forklift']" in printed
