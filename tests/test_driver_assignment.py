"""
Who is the driver: boarding, co-motion, one-to-one, tracker id switches, cab emergence, lost-driver policy.

Test calibration: 100 px == 1.0 m on the ground plane (see test_vehicle_monitoring.get_test_config).
"""
import numpy as np

from app.vehicle_monitoring.distance_estimator import RealWorldDistanceEstimator
from app.vehicle_monitoring.driver_vehicle_fsm import DriverVehicleMonitorFSM, VehicleSafetyState
from app.vehicle_monitoring.pipeline import VehicleMonitoringPipeline
from test_vehicle_monitoring import get_test_config

SHAPE = (800, 800, 3)
DT = 0.1


def make_fsm(**pairing):
    cfg = get_test_config()
    cfg["pairing"] = {"board_confirm_sec": 1.5, "seated_board_confirm_sec": 0.6, "dismount_confirm_sec": 0.6}
    cfg["pairing"].update(pairing)
    return DriverVehicleMonitorFSM(cfg, RealWorldDistanceEstimator(cfg))


def vehicle(vid, x1=100.0, y1=150.0, x2=200.0, y2=300.0):
    return {"id": str(vid), "bbox": [x1, y1, x2, y2]}


def person(pid, x1, y1=170.0, x2=None, y2=290.0, **extra):
    d = {"id": str(pid), "bbox": [x1, y1, x2 if x2 is not None else x1 + 60.0, y2]}
    d.update(extra)
    return d


def hold(fsm, vehicles, persons, t, seconds):
    recs, events = None, []
    for _ in range(int(round(seconds / DT))):
        t += DT
        recs, ev = fsm.update(vehicles, persons, SHAPE, now=t)
        events += ev
    return recs, events, t


# ----------------------------------------------------------------------------------------------
# Input normalisation (pipeline.py)
# ----------------------------------------------------------------------------------------------
def _bare_pipeline():
    return VehicleMonitoringPipeline.__new__(VehicleMonitoringPipeline)


def test_unverified_persons_keep_distinct_ids_and_bev_is_passed_through():
    pl = _bare_pipeline()
    raw = [
        {"id": "T3", "global_id": -1, "bbox": [10, 10, 60, 160], "keypoints": None,
         "foot_px": (35, 160), "bev_xy": np.array([1.0, 2.0], dtype=np.float32)},
        {"id": "T4", "global_id": -1, "bbox": [300, 10, 350, 160], "keypoints": None,
         "foot_px": (325, 160), "bev_xy": np.array([3.0, 2.0], dtype=np.float32)},
        {"id": "7", "global_id": 7, "bbox": [400, 10, 450, 160], "keypoints": None,
         "foot_px": (425, 160), "bev_xy": None, "ground_uncertain": True},
    ]
    out = pl.normalize_pose_persons(np.zeros((480, 640, 3), np.uint8), raw)
    assert [p["id"] for p in out] == ["T3", "T4", "7"]  # no collapse into a single "-1"
    assert out[0]["bev_xy"] == (1.0, 2.0)
    assert out[2]["bev_xy"] is None and out[2]["ground_uncertain"] is True


def test_fsm_accepts_numpy_bev_without_crashing():
    fsm = make_fsm()
    fsm.update([vehicle(1)], [person(5, 120, bev_xy=np.array([1.0, 2.0], dtype=np.float32))], SHAPE, now=0.0)


def test_vehicle_classes_are_resolved_by_name():
    from app.vehicle_monitoring.pipeline import resolve_class_ids

    assert resolve_class_ids({0: "person", 3: "Forklift"}, ["forklift"], []) == [3]
    assert resolve_class_ids({0: "person", 7: "truck"}, None, [7]) == [7]


# ----------------------------------------------------------------------------------------------
# Who counts as the operator of a MOVING vehicle
# ----------------------------------------------------------------------------------------------
def _drive(fsm, persons_at, frames=30, step_px=10.0, t0=0.0):
    rec = None
    events = []
    for k in range(frames):
        vx = 100.0 + step_px * k
        recs, ev = fsm.update([vehicle(1, vx, 150, vx + 100, 300)], persons_at(k, vx), SHAPE, now=t0 + k * DT)
        rec = recs["1"]
        events += ev
    return rec, events


def test_stationary_pedestrian_overlapped_by_a_passing_forklift_is_not_the_driver():
    fsm = make_fsm()
    ped = person(9, 300, 170, 360, 290)
    for k in range(15):
        fsm.update([], [ped], SHAPE, now=k * DT)
    rec, _ = _drive(fsm, lambda k, vx: [ped], t0=15 * DT)
    assert rec.associated_driver_id is None
    # moving with nobody visible aboard: someone hidden must be driving it (never "parked, no driver")
    assert rec.state == VehicleSafetyState.OCCUPIED_UNKNOWN_DRIVER


def test_person_travelling_with_the_forklift_becomes_its_driver():
    fsm = make_fsm()
    rec, events = _drive(fsm, lambda k, vx: [person(5, vx + 20, 170, vx + 80, 290)])
    assert rec.associated_driver_id == "5"
    assert rec.state == VehicleSafetyState.DRIVER_ONBOARD
    assert not any(e["type"] == "DRIVER_EXITED" for e in events)


def test_one_person_cannot_be_claimed_by_two_vehicles():
    fsm = make_fsm()
    vehicles = [vehicle(1, 100, 150, 200, 300), vehicle(2, 150, 150, 250, 300)]
    recs, _, _ = hold(fsm, vehicles, [person(5, 150, 170, 200, 290)], 0.0, 2.5)
    assert [r.associated_driver_id for r in recs.values()].count("5") == 1


def test_forklift_driving_toward_the_camera_still_gets_its_operator():
    fsm = make_fsm()
    rec = None
    for k in range(30):
        y2, y1 = 300.0 + 6.0 * k, 150.0 + 2.0 * k
        veh = [vehicle(1, 100.0 - k, y1, 200.0 + k, y2)]
        drv = person(5, 125.0 - 0.5 * k, y1 + 20.0, 175.0 + 0.5 * k, y2 - 10.0)
        recs, _ = fsm.update(veh, [drv], SHAPE, now=k * DT)
        rec = recs["1"]
    assert rec.associated_driver_id == "5"


# ----------------------------------------------------------------------------------------------
# Driver id changes (tracker/identity switch) must not lose the driver
# ----------------------------------------------------------------------------------------------
def test_driver_binding_survives_an_id_change_at_the_same_place():
    fsm = make_fsm()
    veh = [vehicle(101)]
    recs, _, t = hold(fsm, veh, [person(5, 120, 170, 180, 290)], 0.0, 2.0)
    assert recs["101"].associated_driver_id == "5"
    recs, _, t = hold(fsm, veh, [person(77, 120, 170, 180, 290), person(88, 500, 170, 560, 290)], t, 0.2)
    assert recs["101"].associated_driver_id == "77"


def test_handover_is_refused_when_embeddings_clearly_differ():
    fsm = make_fsm()
    veh = [vehicle(101)]
    emb_a, emb_b = np.array([1.0, 0.0, 0.0], np.float32), np.array([0.0, 1.0, 0.0], np.float32)
    recs, _, t = hold(fsm, veh, [person(5, 120, 170, 180, 290, embedding=emb_a)], 0.0, 2.0)
    recs, _, t = hold(fsm, veh, [person(77, 120, 170, 180, 290, embedding=emb_b)], t, 0.1)
    assert recs["101"].associated_driver_id == "5"


def test_identity_merge_renames_the_driver():
    fsm = make_fsm()
    recs, _, t = hold(fsm, [vehicle(101)], [person("T12", 120, 170, 180, 290)], 0.0, 2.0)
    assert recs["101"].associated_driver_id == "T12"
    fsm.rename_person("T12", "7")
    assert fsm.vehicles["101"].associated_driver_id == "7" and fsm.driver_of == {"7": "101"}


# ----------------------------------------------------------------------------------------------
# "Driver steps out of the cab" binding must not fire for any newly seen pedestrian
# ----------------------------------------------------------------------------------------------
def _occupied_parked_forklift(fsm, x_start=100.0, x_park=240.0):
    """The forklift drives in (driver hidden by the roof) and parks with its box at x_park..x_park+100."""
    t = 0.0
    for k in range(15):
        t += DT
        x = x_start + (x_park - x_start) * k / 14.0
        fsm.update([vehicle(1, x, 150, x + 100, 300)], [], SHAPE, now=t)
    for _ in range(35):
        t += DT
        fsm.update([vehicle(1, x_park, 150, x_park + 100, 300)], [], SHAPE, now=t)
    return t


def test_person_emerging_next_to_an_occupied_parked_forklift_is_its_driver():
    fsm = make_fsm()
    t = _occupied_parked_forklift(fsm)
    assert fsm.vehicles["1"].state == VehicleSafetyState.OCCUPIED_UNKNOWN_DRIVER
    recs, events, t = hold(fsm, [vehicle(1, 240, 150, 340, 300)], [person(9, 350, 170, 410, 300)], t, 0.5)
    assert recs["1"].associated_driver_id == "9"
    assert any(e["type"] == "DRIVER_EXITED" and e["details"]["how"] == "EMERGED_FROM_CAB" for e in events)


def test_person_appearing_next_to_a_never_used_parked_forklift_is_not_a_driver():
    fsm = make_fsm()
    recs, _, _ = hold(fsm, [vehicle(1)], [person(9, 210, 170, 270, 300)], 0.0, 1.0)
    assert recs["1"].associated_driver_id is None


def test_provisional_id_cannot_become_a_driver_by_emergence():
    fsm = make_fsm()
    t = _occupied_parked_forklift(fsm)
    recs, _, _ = hold(fsm, [vehicle(1, 240, 150, 340, 300)], [person("T9", 350, 170, 410, 300)], t, 0.5)
    assert recs["1"].associated_driver_id is None


def test_person_who_walked_in_from_the_image_border_cannot_become_a_driver():
    fsm = make_fsm()
    t = _occupied_parked_forklift(fsm, x_start=460.0, x_park=600.0)
    assert fsm.vehicles["1"].state == VehicleSafetyState.OCCUPIED_UNKNOWN_DRIVER
    # 0.7 m from the forklift, but first seen inside the 40 px border band -> walked in from outside the view
    recs, _, _ = hold(fsm, [vehicle(1, 600, 150, 700, 300)], [person(9, 745, 170, 799, 300)], t, 0.5)
    assert recs["1"].associated_driver_id is None


# ----------------------------------------------------------------------------------------------
# Fail-safe policy: driver vanished next to a stationary vehicle
# ----------------------------------------------------------------------------------------------
def _board_then_vanish(fsm, seconds, step_out_first=False):
    veh = [vehicle(101)]
    _, events, t = hold(fsm, veh, [person(5, 120, 170, 180, 290)], 0.0, 2.0)
    if step_out_first:
        _, ev, t = hold(fsm, veh, [person(5, 270, 180, 330, 300)], t, 1.0)
        events += ev
    recs, ev, t = hold(fsm, veh, [], t, seconds)
    return recs["101"], events + ev


def test_driver_lost_alarm_is_disabled_by_default():
    rec, events = _board_then_vanish(make_fsm(), seconds=30, step_out_first=True)
    assert rec.state == VehicleSafetyState.DRIVER_NOT_VISIBLE
    assert not any(e["severity"] == "alarm" for e in events)


def test_driver_hidden_inside_the_cab_stays_onboard_without_alarm():
    rec, events = _board_then_vanish(make_fsm(driver_lost_alarm_sec=2.0), seconds=10)
    assert rec.state == VehicleSafetyState.DRIVER_ONBOARD and rec.driver_hidden_onboard
    assert not any(e["severity"] == "alarm" for e in events)


def test_driver_lost_alarm_fires_once_after_the_configured_time():
    rec, events = _board_then_vanish(make_fsm(driver_lost_alarm_sec=2.0), seconds=6, step_out_first=True)
    assert rec.state == VehicleSafetyState.DRIVER_NOT_VISIBLE_ALARM and rec.alarm_active
    assert sum(e["type"] == "DRIVER_NOT_VISIBLE_ALARM" for e in events) == 1


# ----------------------------------------------------------------------------------------------
# Clocks and memory
# ----------------------------------------------------------------------------------------------
def test_events_carry_the_stream_time_of_the_moment():
    fsm = make_fsm(board_confirm_sec=0.15)
    _, events, _ = hold(fsm, [vehicle(101)], [person(5, 120, 170, 180, 290)], 5.0, 0.5)
    boarded = next(e for e in events if e["type"] == "DRIVER_BOARDED")
    assert 5.0 <= boarded["ts"] <= 5.5 and boarded["vehicle_id"] == "101" and boarded["worker_id"] == 5


def test_idle_person_histories_are_forgotten():
    fsm = make_fsm()
    fsm.update([], [person(42, 300)], SHAPE, now=0.0)
    assert "42" in fsm.person_histories
    fsm.update([], [person(43, 500)], SHAPE, now=100.0)
    assert "42" not in fsm.person_histories and "43" in fsm.person_histories
