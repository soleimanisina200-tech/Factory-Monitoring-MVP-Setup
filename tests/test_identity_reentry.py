"""
Regression tests for the "exit camera -> next person from the same side inherits the ID" bug.

A fake Re-ID extractor is used so the scenarios are fully deterministic: each scenario
decides the embedding the extractor will return for the person that is on screen.
"""
import numpy as np

from app.detection.detector import PoseDetection
from app.identity.manager import EnterpriseIdentityManager
from app.identity.reid_model import CropQualityReport
from app.logic.geometry import MetricBEVProjector

W, H = 640, 480
FPS = 10.0
DIM = 16


def unit(i: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


def with_similarity(base: np.ndarray, other_axis: int, sim: float) -> np.ndarray:
    """Unit vector whose cosine similarity with `base` is exactly `sim`."""
    v = sim * base + np.sqrt(1.0 - sim * sim) * unit(other_axis)
    return (v / np.linalg.norm(v)).astype(np.float32)


class FakeReid:
    def __init__(self):
        self.current = unit(0)

    def evaluate_and_extract(self, frame, bbox, kxy, kconf, is_intersecting):
        report = CropQualityReport(True, True, 50.0, 1.0, "FRONT", "OK")
        return self.current.copy(), report


class Scenario:
    def __init__(self, **mgr_kwargs):
        self.reid = FakeReid()
        self.mgr = EnterpriseIdentityManager(
            reid_extractor=self.reid,
            bev_projector=MetricBEVProjector(),
            **mgr_kwargs,
        )
        self.frame = np.zeros((H, W, 3), dtype=np.uint8)
        self.ts = 100.0

    def step(self, track_id=None, x1=0.0, emb=None):
        """Advance one frame. track_id=None means nobody is visible."""
        self.ts += 1.0 / FPS
        if track_id is None:
            return self.mgr.process_frame(self.frame, [], self.ts)
        self.reid.current = emb
        det = PoseDetection(
            track_id=track_id,
            bbox_xyxy=np.array([x1, 200.0, x1 + 60.0, 360.0], dtype=np.float32),
            conf=0.9,
            keypoints_xy=np.zeros((17, 2), dtype=np.float32),
            keypoints_conf=np.zeros(17, dtype=np.float32),
        )
        return self.mgr.process_frame(self.frame, [det], self.ts)

    def idle(self, seconds):
        for _ in range(int(seconds * FPS)):
            self.step(None)

    def walk_out_right(self, track_id, emb, start_x=300.0, step=20.0):
        """Person walks to the right border and leaves the frame."""
        gid = None
        x = start_x
        while x + 60.0 < W:
            out = self.step(track_id, x, emb)
            gid = out.get(track_id, gid)
            x += step
        out = self.step(track_id, W - 60.0, emb)  # bbox touches the right border
        return out.get(track_id, gid)


def test_new_person_on_reused_track_id_does_not_inherit_exited_identity():
    """ByteTrack may reuse the old track id for a newcomer entering from the same border."""
    sc = Scenario()
    emb_a = unit(0)
    gid_a = sc.walk_out_right(track_id=1, emb=emb_a)
    assert gid_a is not None, "worker A should have been registered before leaving"

    sc.idle(1.0)  # nobody visible: A has left through the right border

    # Worker B looks fairly similar to A (same uniform) but is a different person.
    emb_b = with_similarity(emb_a, other_axis=1, sim=0.55)
    assigned = []
    for k in range(10):
        out = sc.step(1, W - 70.0 - 5.0 * k, emb_b)
        assigned.append(out.get(1))

    assert gid_a not in assigned, "newcomer inherited the identity of the person who left"
    assert assigned[-1] is not None and assigned[-1] != gid_a, "newcomer should get a fresh ID"


def test_same_person_returning_later_recovers_original_id():
    sc = Scenario()
    emb_a = unit(0)
    gid_a = sc.walk_out_right(track_id=1, emb=emb_a)
    sc.idle(5.0)

    got = None
    for k in range(10):
        out = sc.step(7, W - 70.0 - 5.0 * k, emb_a)  # brand-new track id, same appearance
        got = out.get(7, got)

    assert got == gid_a


def test_ambiguous_return_between_two_lookalikes_gets_new_id():
    """If the returning person matches two exited workers almost equally, do not guess."""
    sc = Scenario()
    emb_a = unit(0)
    gid_a = sc.walk_out_right(track_id=1, emb=emb_a)
    sc.idle(1.0)
    emb_c = with_similarity(emb_a, other_axis=2, sim=0.75)  # worker C resembles A (below re-entry threshold)
    gid_c = sc.walk_out_right(track_id=2, emb=emb_c)
    assert gid_c is not None and gid_c != gid_a
    sc.idle(4.0)

    # Returning person sits exactly between A and C.
    emb_x = (emb_a + emb_c) / np.linalg.norm(emb_a + emb_c)
    got = None
    for k in range(10):
        out = sc.step(9, W - 70.0 - 5.0 * k, emb_x.astype(np.float32))
        got = out.get(9, got)

    assert got not in (gid_a, gid_c)


def test_in_yard_occlusion_still_recovers_identity():
    """A worker hidden in the middle of the yard (not at a border) must keep the same ID."""
    sc = Scenario()
    emb_a = unit(0)
    gid = None
    for k in range(12):
        gid = sc.step(1, 200.0 + 4.0 * k, emb_a).get(1, gid)
    sc.idle(1.0)  # hidden behind a forklift

    got = None
    for k in range(10):
        got = sc.step(1, 250.0 + 4.0 * k, emb_a).get(1, got)
    assert got == gid


def test_exit_is_reported_once_to_caller():
    sc = Scenario()
    gid_a = sc.walk_out_right(track_id=1, emb=unit(0))
    sc.idle(1.0)
    assert sc.mgr.pop_newly_exited() == [gid_a]
    assert sc.mgr.pop_newly_exited() == []
