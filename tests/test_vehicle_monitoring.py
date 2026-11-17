"""
Metric distance (ground footprint) and the dismount -> distance -> alarm cycle.

Test calibration: 100 px == 1.0 m on the ground plane; image y grows with world y.
"""
import numpy as np

from app.core.alarms import AlarmDispatcher
from app.core.clock import StreamClock
from app.core.events import EventStore
from app.vehicle_monitoring.distance_estimator import RealWorldDistanceEstimator
from app.vehicle_monitoring.driver_vehicle_fsm import DriverVehicleMonitorFSM, VehicleSafetyState

SHAPE = (800, 800, 3)
DT = 0.1


def get_test_config():
    return {
        "pairing": {
            "board_confirm_sec": 0.15,
            "seated_board_confirm_sec": 0.15,
            "dismount_confirm_sec": 0.15,
        },
        "distance": {
            "safe_distance_threshold_m": 3.0,
            "warning_distance_threshold_m": 2.0,
            "alarm_confirm_sec": 0.25,
            "alarm_clear_sec": 0.25,
            "ema_smoothing_alpha": 1.0,  # raw coordinates for exact assertions
            "max_step_jump_m": 10.0,
            "min_bbox_aspect_ratio": 1.15,
            "border_margin_px": 10,
        },
        "calibration": {
            "image_points_px": [[100, 100], [700, 100], [700, 700], [100, 700]],
            "world_points_m": [[0.0, 0.0], [6.0, 0.0], [6.0, 6.0], [0.0, 6.0]],
        },
    }


def make_fsm(**pairing):
    cfg = get_test_config()
    cfg["pairing"].update(pairing)
    return DriverVehicleMonitorFSM(cfg, RealWorldDistanceEstimator(cfg))


VEH = [{"id": "101", "bbox": [100, 150, 200, 300]}]


def p(pid, bbox, **kw):
    d = {"id": str(pid), "bbox": list(bbox)}
    d.update(kw)
    return d


INSIDE = [120, 170, 180, 290]
NEAR = [270, 180, 330, 300]     # feet at x=300 -> 1.0 m from the forklift side
FAR = [550, 180, 610, 300]      # feet at x=580 -> 3.8 m


def run(fsm, persons, t0, seconds, vehicles=VEH):
    events, t, recs = [], t0, None
    for _ in range(int(round(seconds / DT))):
        t += DT
        recs, ev = fsm.update(vehicles, persons, SHAPE, now=t)
        events += ev
    return recs, events, t


# ------------------------------------------------------------------------------------------------------------------
def test_real_world_homography_distance_in_meters():
    est = RealWorldDistanceEstimator(get_test_config())
    meas = est.estimate_driver_vehicle_distance("1", [520, 180, 580, 300], "10", [100, 150, 200, 300], SHAPE)
    assert not meas.is_uncertain
    assert abs(meas.distance_m - 3.50) < 0.05


def test_occlusion_flags_uncertainty_instead_of_false_distance():
    est = RealWorldDistanceEstimator(get_test_config())
    meas = est.estimate_driver_vehicle_distance("1", [500, 250, 600, 300], "10", [100, 150, 200, 300], SHAPE)
    assert meas.is_uncertain and "OCCLUDED" in meas.uncertainty_reason


def test_footprint_uses_only_floor_points():
    """A person standing just behind the visible bottom edge is ON the footprint (0 m), not metres away."""
    est = RealWorldDistanceEstimator(get_test_config())
    behind = est.estimate_driver_vehicle_distance("1", [130, 130, 170, 250], "10", [100, 150, 200, 300], SHAPE)
    assert behind.distance_m == 0.0
    side = est.estimate_driver_vehicle_distance("2", [230, 180, 270, 300], "10", [100, 150, 200, 300], SHAPE)
    assert abs(side.distance_m - 0.5) < 0.05


def test_feet_hidden_ground_point_is_uncertain():
    est = RealWorldDistanceEstimator(get_test_config())
    kpts = np.zeros((17, 3), np.float32)  # no ankle visible
    meas = est.estimate_driver_vehicle_distance("1", [550, 180, 610, 300], "10", [100, 150, 200, 300], SHAPE,
                                                keypoints=kpts, foot_px=(580, 330), ground_uncertain=True)
    assert meas.is_uncertain


def test_board_exit_alarm_and_clear_cycle():
    fsm = make_fsm()
    recs, ev, t = run(fsm, [p(5, INSIDE)], 0.0, 0.5)
    assert recs["101"].state == VehicleSafetyState.DRIVER_ONBOARD and recs["101"].associated_driver_id == "5"
    assert [e["type"] for e in ev].count("DRIVER_BOARDED") == 1

    recs, ev, t = run(fsm, [p(5, NEAR)], t, 0.5)
    assert recs["101"].state == VehicleSafetyState.DRIVER_EXITED_SAFE
    assert any(e["type"] == "DRIVER_EXITED" for e in ev)

    recs, ev, t = run(fsm, [p(5, FAR)], t, 0.6)
    assert recs["101"].state == VehicleSafetyState.DISTANCE_VIOLATION_ALARM and recs["101"].alarm_active
    alarms = [e for e in ev if e["type"] == "SAFE_DISTANCE_VIOLATION_ALARM"]
    assert len(alarms) == 1 and alarms[0]["severity"] == "alarm" and alarms[0]["worker_id"] == 5

    recs, ev, t = run(fsm, [p(5, NEAR)], t, 0.6)
    assert not recs["101"].alarm_active
    assert any(e["type"] == "DRIVER_ALARM_CLEARED" for e in ev)


def test_one_far_frame_does_not_raise_an_alarm():
    fsm = make_fsm()
    _, _, t = run(fsm, [p(5, INSIDE)], 0.0, 0.5)
    _, _, t = run(fsm, [p(5, NEAR)], t, 0.5)
    _, ev1, t = run(fsm, [p(5, FAR)], t, 0.1)          # one noisy frame
    recs, ev2, t = run(fsm, [p(5, NEAR)], t, 0.5)
    assert not any(e["severity"] == "alarm" for e in ev1 + ev2)
    assert recs["101"].state == VehicleSafetyState.DRIVER_EXITED_SAFE


def test_uncertain_distance_alarms_only_beyond_the_margin():
    fsm = make_fsm()
    _, _, t = run(fsm, [p(5, INSIDE)], 0.0, 0.5)
    _, _, t = run(fsm, [p(5, NEAR)], t, 0.5)
    hidden_feet = dict(keypoints=np.zeros((17, 3), np.float32), ground_uncertain=True)
    # 3.8 m with hidden feet: above 3.0 but below 3.0 + 1.5 margin -> uncertain, no alarm
    recs, ev, t = run(fsm, [p(5, FAR, foot_px=(580, 300), **hidden_feet)], t, 0.6)
    assert recs["101"].state == VehicleSafetyState.DISTANCE_UNCERTAIN and not recs["101"].alarm_active
    # 5.3 m with hidden feet: certainly too far -> alarm
    recs, ev, t = run(fsm, [p(5, [700, 180, 760, 300], foot_px=(730, 300), **hidden_feet)], t, 0.6)
    assert recs["101"].alarm_active


def test_every_alarm_is_recorded_even_during_the_output_cooldown(tmp_path):
    store = EventStore(db_path=str(tmp_path / "ev.db"), csv_dir=str(tmp_path / "csv"),
                       snapshot_dir=str(tmp_path / "snap"), clock=StreamClock(False, 1_700_000_000.0))
    disp = AlarmDispatcher({"enable_audio_alarm": False, "alarm_cooldown_sec": 60.0})
    sent = []

    class Sink:
        def send(self, ev):
            sent.append(ev["type"])

        def close(self):
            pass

    disp.sinks.append(Sink())
    store.subscribe(disp.handle)
    frame = np.zeros((60, 80, 3), np.uint8)
    from app.core.events import make_event

    for k in range(3):
        store.emit(make_event("SAFE_DISTANCE_VIOLATION_ALARM", 1.0 + k, severity="alarm", vehicle_id="1",
                              worker_id=5, value=4.0), frame)
    store.close()
    import sqlite3

    n = sqlite3.connect(str(tmp_path / "ev.db")).execute("SELECT COUNT(*) FROM events").fetchone()[0]
    assert n == 3            # every alarm is in the audit trail
    assert sent == ["SAFE_DISTANCE_VIOLATION_ALARM"]   # the output was rate-limited
    assert len(list((tmp_path / "snap").rglob("*.jpg"))) == 3
