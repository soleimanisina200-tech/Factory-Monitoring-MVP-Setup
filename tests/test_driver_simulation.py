"""
Driver detection against ground truth from the 3-D simulator (app/vehicle_monitoring/simulator.py): a camera,
forklifts that hide the body parts behind / inside them, noisy pose detections, bystanders, id changes.

These are REGRESSION floors (well below the measured values, see scripts/benchmark_driver.py for the full table),
plus the individual mechanisms that the measurements showed to matter.
"""
import numpy as np
import pytest

from app.vehicle_monitoring import simulator as S
from app.vehicle_monitoring.distance_estimator import point_polygon_distance, signed_gap
from app.vehicle_monitoring.driver_vehicle_fsm import PersonTrackHistory

SHAPE = (S.IMG_H, S.IMG_W, 3)
CORNERS = np.array([(-1.2, -0.6), (2.2, -0.6), (2.2, 0.6), (-1.2, 0.6)])


def _iou(a, b):
    import cv2

    a, b = cv2.convexHull(a.astype(np.float32)), cv2.convexHull(b.astype(np.float32))
    inter, _ = cv2.intersectConvexConvex(a, b)
    union = cv2.contourArea(a) + cv2.contourArea(b) - inter
    return inter / union if union > 0 else 0.0


def _observe(fn, tries=20):
    for _ in range(tries):
        out = fn()
        if out is not None:
            return out
    return None


# ------------------------------------------------------------------------------------------------ geometry
def test_signed_gap_and_distance_to_a_polygon():
    sq = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], float)
    assert signed_gap((1, 1), sq) == pytest.approx(-1.0)
    assert signed_gap((3, 1), sq) == pytest.approx(1.0)
    assert signed_gap((1, 0.25), sq) == pytest.approx(-0.25)
    assert point_polygon_distance((1, 1), sq)[0] == 0.0
    d, nearest = point_polygon_distance((3, 3), sq)
    assert d == pytest.approx(2 ** 0.5) and nearest == pytest.approx((2.0, 2.0))


@pytest.mark.parametrize("camera", list(S.CAMERAS))
def test_footprint_from_the_mask_outline_has_the_real_size_and_place(camera):
    ious = []
    for heading in range(0, 360, 30):
        sim = S.DriverSim(S.CAMERAS[camera](), seed=heading)
        v = sim.vehicle(1.0, 4.5, heading, "open")
        o = _observe(lambda: sim._observe_vehicle(v))
        fp, _ = sim.estimator.vehicle_footprint(o["bbox"], SHAPE, o["contour"])
        ious.append(_iou(fp, v.footprint()))
    assert np.mean(ious) > 0.80 and min(ious) > 0.55, ious          # measured: mean ~0.90, min ~0.7


def test_footprint_of_a_parked_forklift_does_not_jitter_into_motion():
    sim = S.DriverSim(S.CAMERAS["steep"](), seed=3)
    sim.vehicle(0.5, 4.5, 90.0, "roof")
    sim.run(25.0)
    rec = sim.fsm.vehicles["1"]
    assert not rec.has_ever_moved and not sim.events                 # no "moving", no "unknown driver" events


# ------------------------------------------------------------------------------------------------ cab evidence
def _evidence(camera, kind, n_headings=8, reps=2):
    out = {"driver": [], "behind": [], "front": []}
    for h in np.linspace(0, 360, n_headings, endpoint=False):
        for rep in range(reps):
            sim = S.DriverSim(S.CAMERAS[camera](), seed=int(100 * rep + h))
            v = sim.vehicle(float(sim.rng.uniform(-3, 3)), float(sim.rng.uniform(3, 7)), float(h), kind)
            vo = _observe(lambda: sim._observe_vehicle(v))
            fp, _ = sim.estimator.vehicle_footprint(vo["bbox"], SHAPE, vo["contour"])
            away = np.array([v.x, v.y]) - sim.cam.pos[:2]
            away /= np.hypot(*away)
            a_l = v.to_local(np.array([[v.x + away[0], v.y + away[1], 0.0]]))[0][:2]
            far, near = float((CORNERS * a_l).sum(axis=1).max()), float((CORNERS * -a_l).sum(axis=1).max())
            d = sim.person(0, 0)
            d.vehicle, d.seated = v, True
            people = {"driver": d,
                      "behind": sim.person(*(np.array([v.x, v.y]) + away * (far + 0.6))),
                      "front": sim.person(*(np.array([v.x, v.y]) - away * (near + 0.5)))}
            for name, p in people.items():
                for _ in range(3):
                    o = sim._observe_person(p)
                    if o is not None:
                        out[name].append(sim.fsm.cab_score(o, vo["bbox"], fp))
    return out


@pytest.mark.parametrize("camera, kind, floor", [("oblique", "open", 0.90), ("side_low", "roof", 0.90),
                                                 ("steep", "roof", 0.60)])
def test_seated_driver_is_proven_inside_and_bystanders_are_not(camera, kind, floor):
    ev = _evidence(camera, kind)
    thr = 0.5
    proven = lambda lst: float(np.mean([e.score >= thr and e.strong for e in lst]))        # noqa: E731
    assert proven(ev["driver"]) >= floor                 # seated pose / body within the footprint / under the roof
    assert proven(ev["behind"]) <= 0.05                  # legs hidden behind the chassis: never PROOF of being inside
    assert np.mean([e.on_floor for e in ev["front"]]) >= 0.9         # feet on the floor in front of the forklift
    assert np.mean([e.score >= thr for e in ev["front"]]) <= 0.02


def test_floor_speed_ignores_the_jitter_of_a_seated_person():
    rng = np.random.default_rng(0)
    still = PersonTrackHistory("1", 0.0, (0, 0), (0.0, 0.0), (0.0, 0.0))
    walk = PersonTrackHistory("2", 0.0, (0, 0), (0.0, 0.0), (0.0, 0.0))
    for k in range(30):
        t = 0.1 * k
        still.update_position((5.0 + rng.normal(0, 0.2), 5.0 + rng.normal(0, 0.2)), t)     # hidden legs: +-20 cm
        walk.update_position((1.2 * t + rng.normal(0, 0.05), 5.0), t)
    assert still.get_speed_mps() < 0.45 and walk.get_speed_mps() > 0.9


def test_only_the_far_side_of_a_forklift_can_hide_a_person():
    sim = S.DriverSim(S.CAMERAS["oblique"](), seed=0)
    v = sim.vehicle(0.0, 5.0, 0.0, "cabin")                 # seen from the side; the camera is at y = -9
    sim.run(0.5)
    rec = sim.fsm.vehicles["1"]
    assert sim.fsm._behind_vehicle(rec, (0.5, 6.5))          # beyond the far side
    assert not sim.fsm._behind_vehicle(rec, (0.5, 3.6))      # in front (camera side)
    assert not sim.fsm._behind_vehicle(rec, (-3.0, 5.0))     # beside it


# ------------------------------------------------------------------------------------------------ scenarios
def test_turning_on_the_spot_does_not_unbind_the_driver():
    sim = S.run_case("board_drive_alarm", "open", 180.0, "oblique", 0)      # must turn round before it drives
    reasons = [e["details"].get("reason") for e in sim.events if e["type"] == "DRIVER_SESSION_END"]
    assert "NOT_MOVING_WITH_VEHICLE" not in reasons and "VEHICLE_MOVED_WITHOUT_DRIVER" not in reasons
    m = sim.scores().summary()
    assert m["driver_correct"] > 0.93 and m["driver_wrong"] == 0.0 and m["alarm_recall"] == 1.0


@pytest.mark.parametrize("camera, heading", [("oblique", 0.0), ("oblique", 270.0), ("steep", 0.0), ("side_low", 90.0)])
def test_bystander_behind_a_parked_forklift_never_becomes_its_driver(camera, heading):
    sim = S.run_case("bystander_behind", "open", heading, camera, 0)
    m = sim.scores().summary()
    assert m["false_driver"] == 0.0                         # 24 s standing right behind the empty forklift
    assert m["driver_correct"] > 0.85                       # ... and the real driver is found when he boards


def test_closed_cabin_driver_is_inferred_from_disappearing_at_the_cab():
    sim = S.run_case("work_near_vehicle", "cabin", 0.0, "oblique", 0)      # never visible inside, never drives
    how = [e["details"]["how"] for e in sim.events if e["type"] == "DRIVER_BOARDED"]
    assert "VANISHED_INTO_CAB" in how
    m = sim.scores().summary()
    assert m["driver_correct"] > 0.9 and m["alarm_recall"] == 1.0 and m["false_alarms"] == 0


def test_benchmark_subset_meets_the_accuracy_floor():
    total = S.Scores()
    res = S.run_benchmark(scenarios=["board_drive_alarm", "driver_change", "two_forklifts", "helper_at_forks"],
                          kinds=("open", "roof"), headings=(0.0, 270.0), cameras=["oblique"], seeds=(0,))
    for s in res.values():
        total.add(s)
    m = total.summary()
    assert m["driver_correct"] >= 0.94, m                   # measured ~0.97
    assert m["driver_wrong"] <= 0.02, m                     # measured ~0.005
    assert m["false_driver"] <= 0.01, m
    assert m["onboard_state"] >= 0.97, m
    assert m["alarm_recall"] >= 0.95 and m["false_alarms"] == 0, m
