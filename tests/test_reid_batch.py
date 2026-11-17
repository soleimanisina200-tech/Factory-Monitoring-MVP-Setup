"""Batched Re-ID extraction and the gallery quality gate (no OSNet weights needed: fake backbone)."""
import cv2
import numpy as np
import torch

from app.identity.reid_model import PoseGuidedOSNetExtractor


class FakeBackbone:
    """Deterministic stand-in for torchreid's FeatureExtractor; counts forward passes."""

    def __init__(self):
        self.calls = 0

    def __call__(self, images):
        self.calls += 1
        feats = [np.resize(img.astype(np.float32).mean(axis=(0, 1)), 64) + np.arange(64) for img in images]
        return torch.tensor(np.stack(feats))


def make_extractor():
    ex = PoseGuidedOSNetExtractor.__new__(PoseGuidedOSNetExtractor)
    ex.kpt_conf_thresh = 0.3
    ex.min_laplacian_var = 0.0
    ex.head_w, ex.color_w = 0.35, 0.2
    ex.body_w = 0.45
    ex.min_gallery_height_px = 72.0
    ex.min_visible_kpts = 5
    ex.clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(4, 4))
    ex.extractor = FakeBackbone()
    return ex


def person_item(x1, y1, h, n_visible=17):
    bbox = np.array([x1, y1, x1 + h * 0.4, y1 + h], dtype=np.float32)
    kxy = np.stack([np.linspace(x1 + 5, x1 + h * 0.4 - 5, 17), np.linspace(y1 + 5, y1 + h - 5, 17)], 1)
    kconf = np.zeros(17, dtype=np.float32)
    kconf[:n_visible] = 0.9
    return bbox, kxy.astype(np.float32), kconf, False


def test_batch_equals_single_calls_and_uses_one_forward_pass():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    items = [person_item(50, 50, 200), person_item(300, 100, 160), person_item(500, 200, 120)]

    ex = make_extractor()
    batch = ex.evaluate_and_extract_batch(frame, items)
    assert ex.extractor.calls == 1

    for (emb_b, rep_b), item in zip(batch, items):
        emb_s, rep_s = make_extractor().evaluate_and_extract(frame, *item)
        assert np.allclose(emb_b, emb_s, atol=1e-5)
        assert rep_b.is_valid_for_gallery == rep_s.is_valid_for_gallery


def test_gallery_gate_rejects_tiny_crops_and_crops_with_few_joints():
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    ex = make_extractor()
    (_, ok), (_, small), (_, few) = ex.evaluate_and_extract_batch(
        frame, [person_item(50, 50, 200), person_item(300, 100, 50), person_item(450, 100, 200, n_visible=3)]
    )
    assert ok.is_valid_for_gallery and ok.reject_reason == "OK"
    assert not small.is_valid_for_gallery and small.reject_reason == "TooSmallForGallery"
    assert not few.is_valid_for_gallery and few.reject_reason == "TooFewKeypoints"


def test_rejected_crops_do_not_break_the_batch():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    ex = make_extractor()
    out = ex.evaluate_and_extract_batch(frame, [person_item(10, 10, 20), person_item(100, 100, 200)])
    assert out[0][0] is None and out[0][1].reject_reason == "BBox too small"
    assert out[1][0] is not None
