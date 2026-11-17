"""The factory tracker presets are complete, valid for the installed Ultralytics, and actually track."""
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
PRESETS = sorted((ROOT / "configs" / "trackers").glob("factory_*.yaml")) + [ROOT / "configs" / "custom_bytetrack.yaml"]

track_mod = pytest.importorskip("ultralytics.trackers.track")
from ultralytics.utils import IterableSimpleNamespace  # noqa: E402


def _builtin_defaults(tracker_type):
    import ultralytics

    p = Path(ultralytics.__file__).parent / "cfg" / "trackers" / f"{tracker_type}.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8"))


@pytest.mark.parametrize("preset", PRESETS, ids=lambda p: p.name)
def test_preset_has_every_key_of_the_builtin_tracker(preset):
    cfg = yaml.safe_load(preset.read_text(encoding="utf-8"))
    assert cfg["tracker_type"] in track_mod.TRACKER_MAP
    missing = set(_builtin_defaults(cfg["tracker_type"])) - set(cfg)
    assert not missing, f"{preset.name} lacks {sorted(missing)}"
    # Distant workers must be able to start a track: the detector keeps new people from conf 0.18.
    assert cfg["new_track_thresh"] <= 0.25


class _Boxes:
    """Minimal detections object with the attributes the Ultralytics trackers read."""

    def __init__(self, xyxy, conf):
        self.xyxy = np.asarray(xyxy, dtype=np.float32)
        self.conf = np.asarray(conf, dtype=np.float32)
        self.cls = np.zeros(len(conf), dtype=np.float32)
        x1, y1, x2, y2 = self.xyxy.T
        self.xywh = np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)
        self.data = np.concatenate([self.xyxy, self.conf[:, None], self.cls[:, None]], axis=1)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, idx):
        return _Boxes(self.xyxy[idx], self.conf[idx])


@pytest.mark.parametrize("preset", PRESETS, ids=lambda p: p.name)
def test_preset_keeps_one_id_for_a_walking_low_confidence_worker(preset):
    cfg = IterableSimpleNamespace(**yaml.safe_load(preset.read_text(encoding="utf-8")))
    try:
        tracker = track_mod.TRACKER_MAP[cfg.tracker_type](args=cfg)
    except TypeError:
        tracker = track_mod.TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=30)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    ids = set()
    for k in range(20):  # a distant worker (conf 0.3) walking right
        x = 100.0 + 6.0 * k
        out = tracker.update(_Boxes([[x, 200.0, x + 30.0, 280.0]], [0.3]), img)
        if len(out):
            ids.add(int(out[0][4]))
    assert len(ids) == 1, f"{preset.name}: ids {ids}"
