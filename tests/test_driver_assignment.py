"""
Regression tests for driver <-> vehicle assignment and the person-input normalisation.

Test calibration: 100 px == 1.0 m on the ground plane (see get_test_config).
"""
import time

import numpy as np

from app.vehicle_monitoring.distance_estimator import RealWorldDistanceEstimator
from app.vehicle_monitoring.driver_vehicle_fsm import DriverVehicleMonitorFSM, VehicleSafetyState
from app.vehicle_monitoring.pipeline import VehicleMonitoringPipeline
from test_vehicle_monitoring import get_test_config

SHAPE = (800, 800, 3)
DT = 0.1  # frame period of the synthetic clock (seconds)


def make_fsm(**pairing_overrides):
    cfg = get_test_config()
    cfg["pairing"].update(pairing_overrides)
    est = RealWorldDistanceEstimator(cfg)
    return DriverVehicleMonitorFSM(cfg, est)


def vehicle(vid, x1=100.0, y1=150.0, x2=200.0, y2=300.0):
    return {"id": str(vid), "bbox": [x1, y1, x2, y2]}


def person(pid, x1, y1=170.0, x2=None, y2=290.0, **extra):
    d = {"id": str(pid), "bbox": [x1, y1, x2 if x2 is not None else x1 + 60.0, y2]}
    d.update(extra)
    return d


# ----------------------------------------------------------------------------------------------
# Input normalisation (pipeline.py)
# ----------------------------------------------------------------------------------------------
def _bare_pipeline():
    pl = VehicleMonitoringPipeline.__new__(VehicleMonitoringPipeline)
    pl.reid_extractor = None
    return pl


def test_unverified_persons_keep_distinct_ids_and_bev_is_passed_through():
    pl = _bare_pipeline()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    raw = [
        {"id": "T3", "global_id": -1, "bbox": [10, 10, 60, 160], "keypoints": None,
         "foot_px": (35, 160), "bev_xy": np.array([1.0, 2.0], dtype=np.float32)},
        {"id": "T4", "global_id": -1, "bbox": [300, 10, 350, 160], "keypoints": None,
         "foot_px": (325, 160), "bev_xy": np.array([3.0, 2.0], dtype=np.float32)},
        {"id": "7", "global_id": 7, "bbox": [400, 10, 450, 160], "keypoints": None,
         "foot_px": (425, 160), "bev_xy": None},
    ]
    out = pl.normalize_pose_persons(frame, raw)

    assert [p["id"] for p in out] == ["T3", "T4", "7"]  # no collapse into a single "-1"
    assert out[0]["bev_xy"] == (1.0, 2.0)
    assert out[2]["bev_xy"] is None


def test_fsm_accepts_numpy_bev_without_crashing():
    fsm = make_fsm()
    p = person(5, 120, bev_xy=np.array([1.0, 2.0], dtype=np.float32))
    fsm.update([vehicle(1)], [p], SHAPE, now=0.0)  # used to raise "truth value of an array is ambiguous"


# ----------------------------------------------------------------------------------------------
# Who counts as the operator of a MOVING vehicle
# ----------------------------------------------------------------------------------------------
def _drive(fsm, persons_at, frames=30, step_px=10.0, t0=0.0):
    """Vehicle drives right at `step_px` per frame; persons_at(k, vx1) -> list of person dicts."""
    rec = None
    for k in range(frames):
        vx = 100.0 + step_px * k
        recs, _ = fsm.update(
            [vehicle(1, vx, 150, vx + 100, 300)], persons_at(k, vx), SHAPE, now=t0 + k * DT
        )
        rec = recs["1"]
    return rec


def test_stationary_pedestrian_overlapped_by_a_passing_forklift_is_not_the_driver():
    fsm = make_fsm()
    ped = person(9, 300, 170, 360, 290)  # stands still, the forklift drives past/over him in the image
    for k in range(15):  # he has been standing there for a while (so he is not a "fresh" dismount candidate)
        fsm.update([], [ped], SHAPE, now=k * DT)
    rec = _drive(fsm, lambda k, vx: [ped], t0=15 * DT)
    assert rec.associated_driver_id is None
    assert rec.state == VehicleSafetyState.UNASSIGNED


def test_person_travelling_with_the_forklift_becomes_its_driver():
    fsm = make_fsm()
    rec = _drive(fsm, lambda k, vx: [person(5, vx + 20, 170, vx + 80, 290)])
    assert rec.associated_driver_id == "5"
    assert rec.state == VehicleSafetyState.DRIVER_ONBOARD


def test_one_person_cannot_be_claimed_by_two_vehicles():
    fsm = make_fsm()
    vehicles = [vehicle(1, 100, 150, 200, 300), vehicle(2, 150, 150, 250, 300)]
    p = person(5, 150, 170, 200, 290)  # overlaps both cab regions
    for k in range(6):
        recs, _ = fsm.update(vehicles, [p], SHAPE, now=k * DT)
    assert [r.associated_driver_id for r in recs.values()].count("5") == 1


# ----------------------------------------------------------------------------------------------
# Driver id changes (tracker/identity switch) must not lose the driver
# ----------------------------------------------------------------------------------------------
def test_driver_binding_survives_an_id_change_at_the_same_place():
    fsm = make_fsm()
    veh = [vehicle(101)]
    inside = person(5, 120, 170, 180, 290)
    fsm.update(veh, [inside], SHAPE, now=0.0)
    recs, _ = fsm.update(veh, [inside], SHAPE, now=0.1)
    assert recs["101"].associated_driver_id == "5"

    switched = person(77, 120, 170, 180, 290)  # same body, new id
    bystander = person(88, 500, 170, 560, 290)
    recs, _ = fsm.update(veh, [switched, bystander], SHAPE, now=0.2)
    assert recs["101"].associated_driver_id == "77"


def test_handover_is_refused_when_embeddings_clearly_differ():
    fsm = make_fsm()
    veh = [vehicle(101)]
    emb_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    emb_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    inside = person(5, 120, 170, 180, 290, embedding=emb_a)
    fsm.update(veh, [inside], SHAPE, now=0.0)
    fsm.update(veh, [inside], SHAPE, now=0.1)

    stranger = person(77, 120, 170, 180, 290, embedding=emb_b)  # different person standing there
    recs, _ = fsm.update(veh, [stranger], SHAPE, now=0.2)
    assert recs["101"].associated_driver_id != "77"


# ----------------------------------------------------------------------------------------------
# "Driver steps out of the cab" auto-binding must not fire for any newly seen pedestrian
# ----------------------------------------------------------------------------------------------
def _spawn_near_parked_vehicle(pid, x1, x2):
    fsm = make_fsm()
    veh = [vehicle(1, 650, 150, 750, 300)] if x1 > 700 else [vehicle(1, 100, 150, 200, 300)]
    p = person(pid, x1, 170, x2, 290)
    rec = None
    for k in range(5):
        recs, _ = fsm.update(veh, [p], SHAPE, now=k * DT)
        rec = recs["1"]
    return rec


def test_verified_person_appearing_beside_a_vehicle_is_bound_as_exited_driver():
    rec = _spawn_near_parked_vehicle(9, 210, 270)
    assert rec.associated_driver_id == "9"


def test_provisional_id_cannot_become_a_driver():
    rec = _spawn_near_parked_vehicle("T9", 210, 270)
    assert rec.associated_driver_id is None


def test_person_who_walked_in_from_the_image_border_cannot_become_a_driver():
    rec = _spawn_near_parked_vehicle(9, 755, 799)  # centre x = 777 -> inside the 40 px border band
    assert rec.associated_driver_id is None


# ----------------------------------------------------------------------------------------------
# Fail-safe: driver vanished next to a stationary vehicle
# ----------------------------------------------------------------------------------------------
def _board_then_vanish(fsm, seconds, step=0.5):
    veh = [vehicle(101)]
    inside = person(5, 120, 170, 180, 290)
    fsm.update(veh, [inside], SHAPE, now=0.0)
    fsm.update(veh, [inside], SHAPE, now=0.1)
    events, t, rec = [], 0.1, None
    while t < seconds:
        t += step
        recs, ev = fsm.update(veh, [], SHAPE, now=t)
        rec, events = recs["101"], events + ev
    return rec, events


def test_driver_lost_alarm_is_disabled_by_default():
    rec, events = _board_then_vanish(make_fsm(), seconds=30)
    assert rec.state != VehicleSafetyState.DRIVER_NOT_VISIBLE_ALARM
    assert not any(e["event_type"] == "DRIVER_NOT_VISIBLE_ALARM" for e in events)


def test_driver_lost_alarm_fires_once_after_the_configured_time():
    rec, events = _board_then_vanish(make_fsm(driver_lost_alarm_sec=2.0), seconds=6)
    assert rec.state == VehicleSafetyState.DRIVER_NOT_VISIBLE_ALARM
    assert rec.alarm_active is True
    assert sum(e["event_type"] == "DRIVER_NOT_VISIBLE_ALARM" for e in events) == 1


# ----------------------------------------------------------------------------------------------
# Clocks and memory
# ----------------------------------------------------------------------------------------------
def test_logic_uses_the_frame_clock_but_events_carry_wall_clock_time():
    fsm = make_fsm()
    veh = [vehicle(101)]
    inside = person(5, 120, 170, 180, 290)
    fsm.update(veh, [inside], SHAPE, now=5.0)
    recs, events = fsm.update(veh, [inside], SHAPE, now=5.1)
    assert recs["101"].last_seen_ts == 5.1  # frame clock
    boarded = next(e for e in events if e["event_type"] == "DRIVER_BOARDED")
    assert abs(boarded["timestamp"] - time.time()) < 60  # wall clock, usable in the audit log


def test_idle_person_histories_are_forgotten():
    fsm = make_fsm()
    fsm.update([], [person(42, 300)], SHAPE, now=0.0)
    assert "42" in fsm.person_histories
    fsm.update([], [person(43, 500)], SHAPE, now=100.0)  # 100 s later (> 60 s TTL)
    assert "42" not in fsm.person_histories
    assert "43" in fsm.person_histories
