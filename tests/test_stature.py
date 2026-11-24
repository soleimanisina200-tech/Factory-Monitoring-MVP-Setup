"""
Relative body height: a uniform-independent soft biometric used as a VETO (clearly different height = different
person). Unit tests of the measurement / camera model, then the truck scenario with a lookalike colleague.
"""
import numpy as np

from app.identity.stature import (
    StatureEstimate,
    StatureModel,
    StatureSamples,
    measure_upright_height,
    statures_conflict,
)
from test_identity_occluders import (
    TRUCK_X2,
    W,
    H,
    Site,
    sim_to,
    unit,
    upright_pose,
)


# ---------------------------------------------------------------- measurement
def test_upright_person_is_measured_and_bad_poses_are_skipped():
    box, k, c = upright_pose(500.0, 150.0)
    m = measure_upright_height(box, k, c, W, H)
    assert m is not None and abs(m[2] - 150.0) < 1.0

    bent = k.copy()                                   # stance knee bent (crouching / sitting)
    bent[13, 0] += 25.0
    bent[14, 0] += 25.0
    assert measure_upright_height(box, bent, c, W, H) is None

    arm_up = k.copy()                                 # raised arm lifts the box top
    arm_up[9, 1] = box[1] - 5.0
    assert measure_upright_height(box, arm_up, c, W, H) is None

    hidden_feet = c.copy()                            # ankles behind a pallet / truck
    hidden_feet[[15, 16]] = 0.1
    assert measure_upright_height(box, k, hidden_feet, W, H) is None

    cut = box.copy()                                  # cut by the image border
    cut[0] = 0.0
    assert measure_upright_height(cut, k, c, W, H) is None


def test_walking_person_with_the_swing_knee_bent_is_still_measured():
    box, k, c = upright_pose(500.0, 150.0)
    k[14] = (k[14][0] + 20.0, k[14][1] - 10.0)       # right (swing) leg bent and lifted
    k[16] = (k[16][0] + 25.0, k[16][1] - 25.0)
    m = measure_upright_height(box, k, c, W, H)
    assert m is not None and abs(m[2] - 150.0) < 1.0


# ---------------------------------------------------------------- camera height model
def test_camera_learns_perspective_and_gives_relative_height():
    rng = np.random.default_rng(0)
    model = StatureModel(min_samples=150, min_tracks=4)
    people = {1: 0.95, 2: 1.0, 3: 1.0, 4: 1.04, 5: 1.08, 6: 0.98}
    samples = {pid: StatureSamples() for pid in people}
    for _ in range(80):
        for pid, r in people.items():
            u, v = rng.uniform(200, 1700), rng.uniform(350, 1000)
            h = (0.30 * v - 60.0) * r * (1.0 + rng.normal(0, 0.01))          # far = small, near = large
            model.add(u, v, h, pid)
            samples[pid].add(u, v, h)
    assert model.ready
    est = {pid: samples[pid].estimate(model) for pid in people}
    mean = np.mean(list(people.values()))
    for pid, r in people.items():
        assert abs(est[pid].ratio - r / mean) < 0.012, (pid, est[pid])
    assert statures_conflict(est[1], est[5])          # 13 % apart: different people
    assert not statures_conflict(est[2], est[3])      # same height: proves nothing
    assert not statures_conflict(est[2], est[6])      # 2 % apart: within what one person varies


def test_no_prediction_outside_the_floor_area_seen():
    model = StatureModel(min_samples=20, min_tracks=2)
    for i in range(40):
        model.add(500.0 + i, 600.0 + i, 150.0, i % 3)
    assert model.predict(520.0, 620.0) is not None
    assert model.predict(520.0, 1000.0) is None


def test_unknown_height_never_conflicts():
    e = StatureEstimate(1.10, 0.005, 20)
    assert not statures_conflict(None, e) and not statures_conflict(e, None)


# ---------------------------------------------------------------- the truck scenario
def _pose_site():
    s = Site(pose=True, stature_model_min_samples=40)
    a = unit(0)
    b = sim_to(a, 1, 0.85)
    for i, sc in enumerate((0.97, 1.02, 0.99, 1.03)):                  # earlier lookalikes
        e = sim_to(a, 2 + i, 0.86)
        t = s.tid()
        for k in range(20):
            s.step([(t, 400.0 - 15.0 * k, e, sc)])
        s.step([(t, 0.0, e, sc)])
        s.run(1.5, lambda k: [])
    tb, ta = s.tid(), s.tid()
    s.run(10.0, lambda k: [(ta, 300.0 + 3.0 * (k % 20), a, 1.0), (tb, 100.0 + (k % 3), b, 1.0)])
    gid_a = s.mgr.track_to_global[ta]
    assert s.mgr.stature_model.ready and s.mgr.stature_estimate(gid_a) is not None
    t = s.tid()
    for k in range(16):                                                 # A goes behind the truck
        s.step([(t, 330.0 + 20.0 * k, a, 1.0), (tb, 100.0, b, 1.0)])
    s.run(4.0, lambda k: [(tb, 100.0, b, 1.0)])
    return s, a, b, tb, gid_a


def _emerge(s, emb, scale, tb, b, frames=40):
    t = s.tid()
    gids = []
    for k in range(frames):
        out = s.step([(t, TRUCK_X2 + 5.0 + 6.0 * k, emb, scale), (tb, 100.0, b, 1.0)])
        gids.append(out.get(t))
    return t, gids


def test_taller_lookalike_coming_out_of_the_truck_does_not_keep_the_hidden_workers_id():
    s, a, b, tb, gid_a = _pose_site()
    other = sim_to(a, 20, 0.88)                     # never seen before, same uniform, 8 % taller (~14 cm)
    _, gids = _emerge(s, other, 1.08, tb, b)
    assert gids[-1] not in (None, gid_a), gids
    ident_a = s.mgr.identities[gid_a]
    assert abs(ident_a.last_bbox_xyxy[0] - 630.0) < 1.0            # A's last position was restored
    a_back = sim_to(a, 21, 0.88)
    _, gids_a = _emerge(s, a_back, 1.0, tb, b)                     # the real A comes out later
    assert gids_a[-1] == gid_a, gids_a


def test_the_hidden_worker_himself_is_not_vetoed_by_measurement_noise():
    s, a, b, tb, gid_a = _pose_site()
    _, gids = _emerge(s, sim_to(a, 21, 0.88), 1.0, tb, b)
    assert set(g for g in gids if g is not None) == {gid_a}, gids


def test_heights_survive_a_restart(tmp_path):
    from app.core.clock import StreamClock
    from app.identity.store import IdentityStore

    s, a, b, tb, gid_a = _pose_site()
    clock = StreamClock(is_live=True)
    IdentityStore(str(tmp_path), clock, model_tag="m").save(s.mgr)
    s2 = Site(pose=True)
    IdentityStore(str(tmp_path), clock, model_tag="m").load_into(s2.mgr, s.ts)
    assert s2.mgr.stature_model.ready
    assert abs(s2.mgr.stature_estimate(gid_a).ratio - s.mgr.stature_estimate(gid_a).ratio) < 1e-6
