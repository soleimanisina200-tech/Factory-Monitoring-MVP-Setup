"""
Roofed forklifts (driver partly or completely hidden), driver changes, seat-time sessions, left-view alarm.

Test calibration: 100 px == 1.0 m.
"""
from app.vehicle_monitoring.driver_vehicle_fsm import VehicleSafetyState
from test_driver_assignment import DT, SHAPE, hold, make_fsm, person, vehicle


def drive(fsm, t, x_from, x_to, frames, persons_fn=lambda x: [], vid=1):
    recs, events = None, []
    for k in range(frames):
        t += DT
        x = x_from + (x_to - x_from) * k / max(1, frames - 1)
        recs, ev = fsm.update([vehicle(vid, x, 150, x + 100, 300)], persons_fn(x), SHAPE, now=t)
        events += ev
    return recs, events, t


def test_person_vanishing_at_the_cab_then_forklift_drives_off_is_the_driver():
    fsm = make_fsm()
    veh = [vehicle(1, 300, 150, 400, 300)]
    # worker 5 walks up and stands next to the parked forklift ...
    _, _, t = hold(fsm, veh, [person(5, 410, 180, 470, 300)], 0.0, 1.0)
    # ... disappears under the roof ...
    _, _, t = hold(fsm, veh, [], t, 1.0)
    # ... and the forklift drives off
    recs, events, t = drive(fsm, t, 300, 450, 20)
    rec = recs["1"]
    assert rec.associated_driver_id == "5"
    assert rec.state == VehicleSafetyState.DRIVER_ONBOARD and rec.driver_hidden_onboard
    boarded = [e for e in events if e["type"] == "DRIVER_BOARDED"]
    assert boarded and boarded[0]["details"]["how"] == "VANISHED_INTO_CAB"


def test_two_people_vanished_at_the_forklift_gives_unknown_driver_not_a_guess():
    fsm = make_fsm()
    veh = [vehicle(1, 300, 150, 400, 300)]
    _, _, t = hold(fsm, veh, [person(5, 410, 180, 470, 300), person(6, 230, 180, 290, 300)], 0.0, 1.0)
    _, _, t = hold(fsm, veh, [], t, 1.0)
    recs, _, _ = drive(fsm, t, 300, 450, 20)
    assert recs["1"].associated_driver_id is None
    assert recs["1"].state == VehicleSafetyState.OCCUPIED_UNKNOWN_DRIVER


def test_hidden_driver_on_a_parked_forklift_keeps_counting_seat_time():
    fsm = make_fsm()
    veh = [vehicle(101)]
    _, _, t = hold(fsm, veh, [person(5, 120, 170, 180, 290)], 0.0, 2.0)
    recs, events, t = hold(fsm, veh, [], t, 60.0)   # roof hides him for a minute
    rec = recs["101"]
    assert rec.state == VehicleSafetyState.DRIVER_ONBOARD and rec.driver_hidden_onboard
    assert rec.onboard_seconds(t) > 60.0
    assert not any(e["severity"] == "alarm" for e in events)


def test_driver_emerging_from_a_roofed_cab_is_rebound_and_monitored():
    fsm = make_fsm()
    veh = [vehicle(1, 300, 150, 400, 300)]
    _, _, t = hold(fsm, veh, [person(5, 320, 170, 380, 290)], 0.0, 2.0)     # boards visibly
    _, _, t = hold(fsm, veh, [], t, 5.0)                                     # roof: hidden
    # the identity layer could not re-identify him: he climbs out as new worker 9
    recs, events, t = hold(fsm, veh, [person(9, 410, 180, 470, 300)], t, 0.5)
    assert recs["1"].associated_driver_id == "9"
    assert any(e["type"] == "DRIVER_REBOUND" for e in events)
    assert any(e["type"] == "DRIVER_EXITED" for e in events)
    # ... and walks away -> alarm
    recs, events, t = hold(fsm, veh, [person(9, 760, 180, 790, 300)], t, 2.0)
    assert recs["1"].alarm_active and recs["1"].state == VehicleSafetyState.DISTANCE_VIOLATION_ALARM


def test_new_driver_boarding_closes_the_old_session_and_clears_its_alarm():
    fsm = make_fsm()
    veh = [vehicle(101)]
    _, _, t = hold(fsm, veh, [person(5, 120, 170, 180, 290)], 0.0, 2.0)
    _, _, t = hold(fsm, veh, [person(5, 270, 180, 330, 300)], t, 1.0)
    recs, _, t = hold(fsm, veh, [person(5, 600, 180, 660, 300)], t, 2.0)
    assert recs["101"].alarm_active
    recs, events, t = hold(fsm, veh, [person(5, 600, 180, 660, 300), person(7, 120, 170, 180, 290)], t, 2.0)
    rec = recs["101"]
    assert rec.associated_driver_id == "7" and not rec.alarm_active
    end = [e for e in events if e["type"] == "DRIVER_SESSION_END"]
    assert end and end[0]["worker_id"] == 5 and end[0]["details"]["reason"] == "DRIVER_CHANGED"
    assert any(e["type"] == "DRIVER_ALARM_CLEARED" for e in events)


def test_driver_switching_forklifts_closes_the_first_session():
    fsm = make_fsm()
    vehicles = [vehicle(1, 100, 150, 200, 300), vehicle(2, 500, 150, 600, 300)]
    _, _, t = hold(fsm, vehicles, [person(5, 120, 170, 180, 290)], 0.0, 2.0)
    _, _, t = hold(fsm, vehicles, [person(5, 350, 180, 410, 300)], t, 1.0)
    recs, events, t = hold(fsm, vehicles, [person(5, 520, 170, 580, 290)], t, 3.0)
    assert recs["2"].associated_driver_id == "5" and recs["1"].associated_driver_id is None
    end = [e for e in events if e["type"] == "DRIVER_SESSION_END"]
    assert end and end[0]["vehicle_id"] == "1" and end[0]["details"]["reason"] == "DRIVER_MOVED_TO_OTHER_VEHICLE"


def test_seat_time_is_reported_when_the_session_ends():
    fsm = make_fsm()
    veh = [vehicle(101)]
    _, ev1, t = hold(fsm, veh, [person(5, 120, 170, 180, 290)], 0.0, 10.0)
    _, ev2, t = hold(fsm, veh, [person(5, 270, 180, 330, 300)], t, 1.0)
    _, ev3, t = hold(fsm, [], [person(5, 270, 180, 330, 300)], t, 31.0)   # forklift gone for > 30 s
    events = ev1 + ev2 + ev3
    exited = next(e for e in events if e["type"] == "DRIVER_EXITED")
    end = next(e for e in events if e["type"] == "DRIVER_SESSION_END")
    assert 9.5 <= exited["value"] <= 10.5                   # first stint (boarding confirmation counted)
    assert abs(end["value"] - exited["value"]) < 0.01      # seat time = time onboard, not time near the forklift
    assert end["details"]["reason"] == "VEHICLE_LEFT_VIEW"


def test_person_standing_behind_a_parked_forklift_is_never_made_its_driver():
    fsm = make_fsm()
    import numpy as np

    k = np.zeros((17, 3), np.float32)
    for i, (x, y) in {5: (330, 190), 6: (370, 190), 11: (335, 240), 12: (365, 240)}.items():
        k[i] = (x, y, 0.9)                         # torso visible, legs hidden by the forklift
    behind = person(8, 320, 170, 380, 260, keypoints=k)
    _, ev1, t = hold(fsm, [vehicle(1, 300, 150, 400, 300)], [behind], 0.0, 20.0)
    # legs hidden and no proof of being INSIDE (no seated pose, body not within the footprint): only a candidate.
    # (The old logic bound him after 1.5 s and had to undo it when the forklift drove away.)
    assert fsm.vehicles["1"].associated_driver_id is None
    recs, ev2, _ = drive(fsm, t, 300, 600, 25, persons_fn=lambda x: [behind])
    assert recs["1"].associated_driver_id is None                   # he did not move with it
    assert recs["1"].state == VehicleSafetyState.OCCUPIED_UNKNOWN_DRIVER   # the real (hidden) driver drives
    events = ev1 + ev2
    assert not any(e["type"] == "DRIVER_BOARDED" for e in events)
    assert not any(e["severity"] == "alarm" for e in events)        # and no false distance alarm


def test_dismounted_driver_walking_past_his_forklift_in_the_image_is_not_reboarded():
    fsm = make_fsm()
    veh = [vehicle(1, 300, 150, 400, 300)]
    _, _, t = hold(fsm, veh, [person(5, 320, 170, 380, 290)], 0.0, 2.0)          # boards
    _, _, t = hold(fsm, veh, [person(5, 410, 180, 470, 300)], t, 1.0)            # climbs out on the right
    events = []
    for k in range(25):                                                           # walks left ACROSS the box
        t += DT
        x = 410.0 - 12.0 * k
        _, ev = fsm.update(veh, [person(5, x, 180, x + 60, 300)], SHAPE, now=t)
        events += ev
    assert not any(e["type"] == "DRIVER_BOARDED" for e in events)
    assert not fsm.vehicles["1"].onboard


def test_driver_climbing_back_in_is_reboarded_after_the_confirmation_time():
    fsm = make_fsm()
    veh = [vehicle(1, 300, 150, 400, 300)]
    _, _, t = hold(fsm, veh, [person(5, 320, 170, 380, 290)], 0.0, 2.0)
    _, _, t = hold(fsm, veh, [person(5, 410, 180, 470, 300)], t, 3.0)
    recs, events, t = hold(fsm, veh, [person(5, 320, 170, 380, 290)], t, 3.5)
    boarded = [e for e in events if e["type"] == "DRIVER_BOARDED"]
    assert len(boarded) == 1 and boarded[0]["details"]["reboard"] is True
    assert recs["1"].state == VehicleSafetyState.DRIVER_ONBOARD


def test_dismounted_driver_walking_out_of_the_camera_raises_the_left_view_alarm():
    fsm = make_fsm()
    veh = [vehicle(101)]
    _, _, t = hold(fsm, veh, [person(5, 120, 170, 180, 290)], 0.0, 2.0)
    _, _, t = hold(fsm, veh, [person(5, 270, 180, 330, 300)], t, 1.0)
    t += DT
    recs, events = fsm.update(veh, [], SHAPE, now=t, left_view_ids=["5"])
    assert recs["101"].state == VehicleSafetyState.DRIVER_LEFT_VIEW_ALARM
    assert [e["type"] for e in events] == ["DRIVER_LEFT_VIEW_ALARM"]
    # he comes back next to the forklift -> cleared
    recs, events, _ = hold(fsm, veh, [person(5, 270, 180, 330, 300)], t, 3.0)
    assert not recs["101"].alarm_active
