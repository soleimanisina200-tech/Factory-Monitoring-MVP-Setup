"""Tracklet-level appearance pooling: quality weighting, contamination filter, time window."""
import numpy as np

from app.detection.detector import PoseDetection
from app.identity.reid_model import CropQualityReport
from test_identity_reentry import Scenario, unit, with_similarity


def det(tid=1, h=200.0, isolated=True, intersecting=False):
    d = PoseDetection(
        track_id=tid,
        bbox_xyxy=np.array([100.0, 100.0, 160.0, 100.0 + h], dtype=np.float32),
        conf=0.9,
        keypoints_xy=np.zeros((17, 2), np.float32),
        keypoints_conf=np.zeros(17, np.float32),
    )
    d.is_isolated = isolated
    d.is_intersecting = intersecting
    return d


def report(valid=True, sharp=60.0, kpts=17, h=200.0, view="FRONT"):
    return CropQualityReport(True, valid, sharp, 1.0, view, "OK" if valid else "TooFewKeypoints", kpts, h)


def test_good_crops_dominate_the_pooled_embedding():
    mgr = Scenario().mgr
    good, bad = unit(0), unit(1)
    mgr._add_track_sample(det(h=220), good, report(sharp=80, kpts=17, h=220), 1.0)
    mgr._add_track_sample(det(h=40), bad, report(valid=False, sharp=8, kpts=3, h=40), 1.1)  # tiny, blurry
    pooled, view, n, _ = mgr._track_pooled_embedding(1, 1.2)
    assert n == 2 and view == "FRONT"
    assert float(np.dot(pooled, good)) > 0.99  # the bad crop barely moves the tracklet appearance


def test_a_single_outlier_frame_cannot_flip_the_tracklet_appearance():
    mgr = Scenario().mgr
    me = unit(0)
    for k in range(6):
        mgr._add_track_sample(det(), with_similarity(me, 2, 0.95), report(), 1.0 + 0.1 * k)
    mgr._add_track_sample(det(), unit(3), report(), 1.7)  # one frame that looks like someone else
    pooled, _, _, _ = mgr._track_pooled_embedding(1, 1.8)
    assert float(np.dot(pooled, me)) > 0.9


def test_crops_overlapping_another_person_are_never_pooled():
    mgr = Scenario().mgr
    mgr._add_track_sample(det(intersecting=True), unit(0), report(), 1.0)
    assert mgr._track_pooled_embedding(1, 1.0)[0] is None


def test_old_samples_leave_the_pool_and_strict_gate_counts_only_gallery_quality():
    mgr = Scenario(pool_window_sec=2.0).mgr
    mgr._add_track_sample(det(), unit(0), report(), 1.0)
    mgr._add_track_sample(det(), unit(0), report(valid=False), 2.5)
    assert mgr._track_pooled_embedding(1, 2.6)[2] == 2
    assert mgr._track_mean_embedding(1, 2.6)[1] == 1     # only the gallery-quality crop counts
    assert mgr._track_pooled_embedding(1, 3.4)[2] == 1   # the 1.0 s sample is outside the 2 s window


def test_unknown_quality_fields_get_neutral_weight():
    """Reports without keypoint/size info (older extractors, fakes) must not be penalised."""
    w = Scenario().mgr._sample_weight(CropQualityReport(True, True, 50.0, 1.0, "FRONT", "OK"), det(h=200))
    assert w == 1.0
