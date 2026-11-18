"""
Identity safeguards for uniform-wearing workforces: learned open-set threshold, persistence across restarts,
helmet markers (hard identity) and exit doors inside the image.
"""
import numpy as np

from app.core.clock import StreamClock
from app.detection.detector import PoseDetection
from app.identity.manager import STATUS_EXITED, EnterpriseIdentityManager
from app.identity.open_set import OpenSetCalibrator
from app.identity.reid_model import CropQualityReport
from app.identity.store import IdentityStore
from app.logic.geometry import MetricBEVProjector
from test_identity_reentry import FPS, H, W, unit, with_similarity


class MapReid:
    """Fake extractor: embedding chosen by the person's box x1 (one entry per person on screen)."""

    def __init__(self):
        self.by_x = {}

    def evaluate_and_extract(self, frame, bbox, kxy, kconf, inter):
        return self.by_x[int(round(float(bbox[0])))].copy(), CropQualityReport(True, True, 50.0, 1.0, "FRONT", "OK")


class FakeMarkers:
    def __init__(self):
        self.by_x = {}

    def read(self, frame, bbox, kxy, kconf):
        return self.by_x.get(int(round(float(bbox[0]))))


class Multi:
    def __init__(self, **kw):
        self.reid = MapReid()
        kw.setdefault("bev_projector", MetricBEVProjector())
        self.mgr = EnterpriseIdentityManager(reid_extractor=self.reid, **kw)
        self.frame = np.zeros((H, W, 3), np.uint8)
        self.ts = 100.0

    def step(self, people):
        """people: list of (track_id, x1, embedding)."""
        self.ts += 1.0 / FPS
        dets = []
        for tid, x, emb in people:
            self.reid.by_x[int(round(x))] = emb
            dets.append(PoseDetection(track_id=tid, bbox_xyxy=np.array([x, 200.0, x + 60.0, 360.0], np.float32),
                                      conf=0.9, keypoints_xy=np.zeros((17, 2), np.float32),
                                      keypoints_conf=np.zeros(17, np.float32)))
        return self.mgr.process_frame(self.frame, dets, self.ts)

    def idle(self, sec):
        for _ in range(int(sec * FPS)):
            self.step([])

    def walk_out_right(self, tid, emb, x=300.0, others=()):
        gid = None
        while x + 60.0 < W:
            gid = self.step([(tid, x, emb)] + list(others)).get(tid, gid)
            x += 20.0
        return self.step([(tid, W - 60.0, emb)] + list(others)).get(tid, gid)

    def enter_right(self, tid, emb, frames=12, others=()):
        out = None
        for k in range(frames):
            out = self.step([(tid, W - 70.0 - 5.0 * k, emb)] + list(others)).get(tid, out)
        return out


def _uniform_site(open_set_enabled):
    sc = Multi(open_set=OpenSetCalibrator(0.82, quantile=0.99, margin=0.03, min_samples=8,
                                          enabled=open_set_enabled))
    a, c = unit(0), with_similarity(unit(0), 1, 0.88)    # two different workers in the same uniform
    for _ in range(80):                                    # visible together for 8 s, 2.8 m apart
        sc.step([(1, 300.0, a), (2, 20.0, c)])
    gid_a = sc.walk_out_right(1, a, x=300.0, others=[(2, 20.0, c)])
    assert gid_a == 1
    sc.idle(3.0)
    return sc, a, gid_a


def test_lookalike_newcomer_is_refused_once_the_site_impostor_level_is_learned():
    sc, a, gid_a = _uniform_site(open_set_enabled=True)
    assert sc.mgr.open_set.learned_threshold() is not None and sc.mgr.reentry_threshold() > 0.88
    newcomer = with_similarity(a, 2, 0.87)                 # above the fixed 0.82, but typical for this uniform
    assert sc.enter_right(5, newcomer, frames=15) not in (gid_a, None)


def test_without_open_set_the_same_lookalike_would_inherit_the_id():
    """Documents the failure mode the learned threshold prevents (fixed 0.82 threshold)."""
    sc, a, gid_a = _uniform_site(open_set_enabled=False)
    assert sc.enter_right(5, with_similarity(a, 2, 0.87), frames=15) == gid_a


def test_the_real_worker_still_comes_back_with_his_id():
    sc, a, gid_a = _uniform_site(open_set_enabled=True)
    assert sc.enter_right(6, a, frames=15) == gid_a


def test_identities_survive_a_restart(tmp_path):
    clock = StreamClock(is_live=True)
    sc = Multi()
    gid_a = sc.walk_out_right(1, unit(0))
    sc.idle(1.0)
    IdentityStore(str(tmp_path), clock, model_tag="m").save(sc.mgr)

    sc2 = Multi()
    sc2.ts = sc.ts + 5.0
    n = IdentityStore(str(tmp_path), clock, model_tag="m").load_into(sc2.mgr, sc2.ts)
    assert n == 1 and sc2.mgr.identities[gid_a].status == STATUS_EXITED
    assert sc2.enter_right(9, unit(0), frames=12) == gid_a           # recognised after the restart
    assert sc2.enter_right(10, unit(3), frames=12) > gid_a           # a newcomer never re-uses an old number


def test_model_change_discards_galleries_but_keeps_numbering(tmp_path):
    clock = StreamClock(is_live=True)
    sc = Multi()
    sc.walk_out_right(1, unit(0))
    IdentityStore(str(tmp_path), clock, model_tag="old").save(sc.mgr)
    sc2 = Multi()
    assert IdentityStore(str(tmp_path), clock, model_tag="new").load_into(sc2.mgr, sc.ts) == 0
    assert sc2.mgr._next_global_id == sc.mgr._next_global_id


def test_helmet_marker_recovers_a_worker_whose_appearance_changed():
    markers = FakeMarkers()
    sc = Multi(marker_reader=markers)
    markers.by_x = {300.0 + 20 * k: "E-7" for k in range(20)}
    gid_a = sc.walk_out_right(1, unit(0))
    assert sc.mgr.identity_label(gid_a) == "E-7"
    sc.idle(60.0)
    # back without his jacket (very different appearance), marker readable
    markers.by_x = {int(W - 70.0 - 5.0 * k): "E-7" for k in range(20)}
    assert sc.enter_right(9, unit(5), frames=15) == gid_a


def test_a_different_marker_is_never_given_the_id_even_with_identical_appearance():
    markers = FakeMarkers()
    sc = Multi(marker_reader=markers)
    markers.by_x = {300.0 + 20 * k: "E-7" for k in range(20)}
    gid_a = sc.walk_out_right(1, unit(0))
    sc.idle(3.0)
    markers.by_x = {int(W - 70.0 - 5.0 * k): "E-8" for k in range(20)}
    got = sc.enter_right(9, unit(0), frames=15)
    assert got != gid_a and sc.mgr.identity_label(got) == "E-8"


def test_worker_first_seen_next_to_others_keeps_his_id_after_they_leave():
    """Regression: an identity created while NOT isolated had no gallery, its similarity was stuck at 0.07 and the
    mismatch breaker gave the visible, unhidden worker a new ID as soon as he walked away from the others."""
    sc = Multi()
    me, other = unit(0), unit(3)
    gid = None
    for k in range(15):                                   # 0.8 m from a colleague: not isolated
        out = sc.step([(1, 300.0 + 2 * k, me), (2, 400.0, other)])   # boxes apart, but < 1.35 m: not isolated
        gid = out.get(1, gid)
    assert gid is not None and sc.mgr.identities[gid].has_gallery()
    assert sc.mgr.identities[gid].compute_similarity(me, "FRONT") > 0.9
    ids = {sc.step([(1, 330.0 + 8 * k, me)]).get(1) for k in range(40)}   # walks away, alone, for 4 s
    assert ids == {gid}


def test_person_mostly_covered_by_a_forklift_is_not_used_for_appearance():
    sc = Multi()
    me = unit(0)
    gid = None
    for k in range(15):
        gid = sc.step([(1, 100.0 + 2 * k, me)]).get(1, gid)
    n_before = len(sc.mgr.identities[gid].recent_embeddings)
    for k in range(20):    # half hidden behind a forklift: garbage crops must not break or pollute the binding
        sc.ts += 1.0 / FPS
        det = PoseDetection(track_id=1, bbox_xyxy=np.array([130.0, 200.0, 190.0, 360.0], np.float32), conf=0.9,
                            keypoints_xy=np.zeros((17, 2), np.float32), keypoints_conf=np.zeros(17, np.float32))
        sc.reid.by_x[130] = unit(7)
        out = sc.mgr.process_frame(sc.frame, [det], sc.ts, obstacles=[[120.0, 150.0, 200.0, 365.0]])
    assert out.get(1) == gid and len(sc.mgr.identities[gid].recent_embeddings) == n_before


def test_real_aruco_marker_on_a_helmet_is_read_and_mapped_to_the_worker():
    import cv2
    import pytest

    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "ArucoDetector"):
        pytest.skip("OpenCV without ArUco")
    from app.identity.markers import HelmetMarkerReader, WorkerRegistry

    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 2, 48)
    marker = cv2.copyMakeBorder(marker, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=255)
    frame = np.full((600, 400, 3), 120, np.uint8)
    frame[40:104, 168:232] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)       # sticker on the helmet
    reg = WorkerRegistry(None)
    reg.by_marker = {2: {"marker_id": "2", "name": "Ali", "employee_id": "E-2"}}
    reader = HelmetMarkerReader(reg)
    person_box = np.array([150, 30, 250, 430], np.float32)
    no_kpts = (np.zeros((17, 2), np.float32), np.zeros(17, np.float32))
    assert reader.read(frame, person_box, *no_kpts) == "E-2" and reg.display_name("E-2") == "Ali"
    assert reader.read(np.full((600, 400, 3), 120, np.uint8), person_box, *no_kpts) is None
    reg.by_marker = {5: {"marker_id": "5", "name": "Reza"}}                  # marker 2 is not registered
    assert reader.read(frame, person_box, *no_kpts) is None


def test_worker_leaving_through_a_door_inside_the_image():
    door = [[250, 300], [350, 300], [350, 400], [250, 400]]
    sc = Multi(exit_regions={"side_door": door})
    gid = None
    for k in range(15):                                  # walks to the door (feet at y=360, x ~ 280-300)
        gid = sc.step([(1, 200.0 + 6.0 * k, unit(0))]).get(1, gid)
    sc.idle(2.0)
    ident = sc.mgr.identities[gid]
    assert ident.status == STATUS_EXITED and ident.exit_reason == "DOOR" and ident.exit_edge == "side_door"
    assert sc.mgr.pop_newly_exited() == [gid]
    got = None
    for _ in range(12):                                  # comes back through the same door
        got = sc.step([(4, 260.0, unit(0))]).get(4, got)
    assert got == gid
