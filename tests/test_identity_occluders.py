"""
A worker who walks behind a large occluder (truck, machine) and comes out again must keep his ID.

Uniform-site numbers: two different workers score ~0.85 with each other, the same worker after an occlusion
~0.87-0.90 (lighting / pose change), crops of a person still half behind the truck are garbage (~0.3).
Settings are the production defaults of app/core/config.py.
"""
import numpy as np

from app.core.config import DEFAULT_SETTINGS
from app.detection.detector import PoseDetection
from app.identity.manager import EnterpriseIdentityManager
from app.identity.open_set import OpenSetCalibrator
from app.identity.reid_model import CropQualityReport
from app.logic.geometry import MetricBEVProjector

W, H = 1920, 1080
FPS = 10.0
DIM = 32
BOX_W, BOX_H = 60.0, 160.0          # ~93 px per metre at this person height
TRUCK_X1, TRUCK_X2 = 700.0, 1500.0  # ~8.6 m long truck in the middle of the image


def unit(i):
    v = np.zeros(DIM, np.float32)
    v[i] = 1.0
    return v


def sim_to(base, axis, s):
    v = s * base + np.sqrt(1.0 - s * s) * unit(axis)
    return (v / np.linalg.norm(v)).astype(np.float32)


class MapReid:
    def __init__(self):
        self.by_x = {}

    def evaluate_and_extract(self, frame, bbox, kxy, kconf, inter):
        return self.by_x[int(round(float(bbox[0])))].copy(), CropQualityReport(True, True, 50.0, 1.0, "FRONT", "OK")


def make_manager(reid, **over):
    r, b = DEFAULT_SETTINGS["reid"], DEFAULT_SETTINGS["physics_bev"]
    kw = dict(
        cos_sim_thresh=r["cos_sim_thresh"], spatial_recovery_sim_thresh=r["spatial_recovery_sim_thresh"],
        reentry_sim_thresh=r["reentry_sim_thresh"], max_worker_speed_mps=b["max_worker_speed_mps"],
        max_internal_occlusion_m=b["max_internal_occlusion_m"], lag_diffusion_coeff=b["lag_diffusion_coeff"],
        base_pose_jitter_m=b["base_pose_jitter_m"], reid_interval_sec=r["reid_interval_sec"],
        stitch_sim_thresh=r["stitch_sim_thresh"], stitch_margin=r["stitch_margin"],
        reentry_min_gap_sec=r["reentry_min_gap_sec"], reentry_margin=r["reentry_margin"],
        reentry_min_samples=r["reentry_min_samples"], occlusion_max_sec=r["occlusion_max_sec"],
        border_margin_ratio=r["border_margin_ratio"], track_gap_reset_sec=r["track_gap_reset_sec"],
        exit_confirm_sec=r["exit_confirm_sec"], mismatch_min_span_sec=r["mismatch_min_span_sec"],
        anchor_min_height_px=r["anchor_min_height_px"], pool_max_samples=r["pool_max_samples"],
        pool_window_sec=r["pool_window_sec"], recover_strict_after_sec=r["recover_strict_after_sec"],
        open_set=OpenSetCalibrator(r["reentry_sim_thresh"], quantile=0.99, margin=0.03, min_samples=8,
                                   enabled=True),
    )
    kw.update(over)
    return EnterpriseIdentityManager(reid_extractor=reid, bev_projector=MetricBEVProjector(), **kw)


class Site:
    def __init__(self, **over):
        self.reid = MapReid()
        self.mgr = make_manager(self.reid, **over)
        self.frame = np.zeros((H, W, 3), np.uint8)
        self.ts = 100.0
        self.next_tid = 1

    def tid(self):
        self.next_tid += 1
        return self.next_tid

    def step(self, people, obstacles=None):
        """people: list of (track_id, x1, embedding)."""
        self.ts += 1.0 / FPS
        dets = []
        for tid, x, emb in people:
            self.reid.by_x[int(round(x))] = emb
            dets.append(PoseDetection(track_id=tid, bbox_xyxy=np.array([x, 600.0, x + BOX_W, 600.0 + BOX_H],
                                                                       np.float32),
                                      conf=0.9, keypoints_xy=np.zeros((17, 2), np.float32),
                                      keypoints_conf=np.zeros(17, np.float32)))
        return self.mgr.process_frame(self.frame, dets, self.ts, obstacles=obstacles)

    def run(self, sec, people_fn):
        out = {}
        for k in range(int(round(sec * FPS))):
            out = self.step(people_fn(k))
        return out


def uniform_site():
    """Worker A and colleague B (same uniform, 0.85) work 10 s in view; 4 lookalikes left earlier."""
    s = Site()
    a = unit(0)
    b = sim_to(a, 1, 0.85)
    for i in range(4):                                   # earlier shift: lookalikes who left through the left edge
        e = sim_to(a, 2 + i, 0.86)
        t = s.tid()
        for k in range(20):
            s.step([(t, 400.0 - 20.0 * k, e)])
        s.step([(t, 0.0, e)])
        s.run(1.5, lambda k: [])
    tb = s.tid()
    ta = s.tid()
    s.run(10.0, lambda k: [(ta, 300.0 + 3.0 * (k % 20), a), (tb, 100.0 + (k % 3), b)])   # learns impostor level
    gid_a = s.step([(ta, 330.0, a), (tb, 100.0, b)])[ta]
    gid_b = s.mgr.track_to_global[tb]
    return s, a, b, gid_a, gid_b, tb


def walk_behind_truck(s, a, tb, b):
    """A walks to the truck and disappears behind it; B stays visible."""
    ta = s.tid()
    for k in range(16):
        s.step([(ta, 330.0 + 20.0 * k, a), (tb, 100.0, b)])   # reaches x1=630 (x2=690, truck starts at 700)


def emerge(s, x0, emb_seq, tb, b, dx=8.0):
    """New tracker id comes out of the truck at x0: emb_seq[k] is the crop embedding of frame k."""
    t = s.tid()
    gids = []
    for k, e in enumerate(emb_seq):
        out = s.step([(t, x0 + dx * k, e), (tb, 100.0, b)])
        gids.append(out.get(t))
    return gids


def test_comes_out_at_the_far_end_of_a_long_truck():
    s, a, b, gid_a, gid_b, tb = uniform_site()
    walk_behind_truck(s, a, tb, b)
    s.run(7.0, lambda k: [(tb, 100.0, b)])                  # 7 s walking along behind the 8.6 m truck
    back = sim_to(a, 20, 0.88)
    gids = emerge(s, TRUCK_X2 + 5.0, [back] * 30, tb, b)
    assert gids[-1] == gid_a, (gids, s.mgr.stats())


def test_stays_behind_the_truck_for_a_minute_while_loading():
    s, a, b, gid_a, gid_b, tb = uniform_site()
    walk_behind_truck(s, a, tb, b)
    s.run(60.0, lambda k: [(tb, 100.0, b)])
    back = sim_to(a, 20, 0.87)
    gids = emerge(s, 640.0, [back] * 30, tb, b, dx=-8.0)
    assert gids[-1] == gid_a, gids


def test_first_crops_while_still_half_behind_the_truck_do_not_create_a_new_id():
    s, a, b, gid_a, gid_b, tb = uniform_site()
    walk_behind_truck(s, a, tb, b)
    s.run(3.0, lambda k: [(tb, 100.0, b)])
    half = sim_to(a, 21, 0.30)                              # half body + truck pixels
    back = sim_to(a, 20, 0.88)
    gids = emerge(s, TRUCK_X2 + 5.0, [half] * 6 + [back] * 24, tb, b)
    assert set(g for g in gids if g is not None) == {gid_a}, gids


def test_a_different_person_coming_out_of_the_truck_gets_a_new_id():
    s, a, b, gid_a, gid_b, tb = uniform_site()
    walk_behind_truck(s, a, tb, b)
    s.run(4.0, lambda k: [(tb, 100.0, b)])
    driver = sim_to(a, 22, 0.20)                            # truck driver, not in uniform
    gids = emerge(s, TRUCK_X2 + 5.0, [driver] * 30, tb, b)
    assert gids[-1] not in (None, gid_a, gid_b)


def test_two_workers_behind_the_same_truck_get_their_own_ids_back():
    s, a, b, gid_a, gid_b, tb = uniform_site()
    d = sim_to(a, 23, 0.80)
    td = s.tid()
    s.run(5.0, lambda k: [(td, 1700.0 - 2.0 * (k % 10), d), (tb, 100.0, b)])
    gid_d = s.mgr.track_to_global[td]
    ta = s.tid()
    for k in range(16):                                     # A disappears on the left, D on the right of the truck
        s.step([(ta, 330.0 + 20.0 * k, a), (td, 1690.0 - 12.0 * k, d), (tb, 100.0, b)])
    s.run(5.0, lambda k: [(tb, 100.0, b)])
    a_back, d_back = sim_to(a, 24, 0.90), sim_to(d, 25, 0.90)
    t1, t2 = s.tid(), s.tid()
    out = {}
    for k in range(30):                                     # they come out swapped: A on the right, D on the left
        out = s.step([(t1, 1510.0 + 6.0 * k, a_back), (t2, 630.0 - 6.0 * k, d_back), (tb, 100.0, b)])
    assert out.get(t1) == gid_a and out.get(t2) == gid_d, (out, gid_a, gid_d)
