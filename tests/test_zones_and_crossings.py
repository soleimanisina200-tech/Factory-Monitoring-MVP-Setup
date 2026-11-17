"""Zone dwell per worker, directional lines/regions, and late attribution of provisional tracks."""
from app.analytics.crossings import CrossingAnalytics
from app.analytics.entities import AttributionBuffer, entity_key, split_entity
from app.analytics.zones import ZoneAnalytics
from app.core.events import make_event

DOCK = {"dock": {"polygon": [[0, 0], [100, 0], [100, 100], [0, 100]]}}
DT = 0.1


def feed(an, key, pts, t0):
    events, t = [], t0
    for p in pts:
        t += DT
        events += an.update({key: p}, t)
    return events, t


# --------------------------------------------------------------------------- zones
def test_zone_visits_accumulate_per_worker():
    z = ZoneAnalytics(DOCK, entry_confirm_sec=0.2, exit_confirm_sec=0.3)
    ev1, t = feed(z, 7, [(50, 50)] * 20, 0.0)          # 2 s inside
    ev2, t = feed(z, 7, [(300, 300)] * 10, t)           # leaves
    ev3, t = feed(z, 7, [(50, 50)] * 30, t)             # 3 s inside again
    ev4, t = feed(z, 7, [(300, 300)] * 10, t)
    exits = [e for e in ev1 + ev2 + ev3 + ev4 if e["type"] == "ZONE_EXIT"]
    assert len(exits) == 2 and all(e["worker_id"] == 7 for e in exits)
    assert abs(exits[0]["value"] - 2.0) < 0.15 and abs(exits[1]["value"] - 3.0) < 0.15
    totals = z.worker_totals()[7]["dock"]
    assert totals["visits"] == 2 and abs(totals["total_sec"] - 5.0) < 0.3   # the old code kept only the last visit


def test_short_jitter_outside_does_not_split_the_visit():
    z = ZoneAnalytics(DOCK, entry_confirm_sec=0.2, exit_confirm_sec=1.0)
    path = [(50, 50)] * 10 + [(105, 50)] * 3 + [(50, 50)] * 10 + [(300, 300)] * 15
    ev, _ = feed(z, 3, path, 0.0)
    assert [e["type"] for e in ev] == ["ZONE_ENTER", "ZONE_EXIT"]


def test_worker_who_leaves_the_camera_inside_a_zone_is_closed_at_last_sighting():
    z = ZoneAnalytics(DOCK, entry_confirm_sec=0.2, exit_confirm_sec=1.0, lost_timeout_sec=15)
    _, t = feed(z, 4, [(50, 50)] * 20, 0.0)
    ev = z.close_entity(4, t + 1.0)
    assert ev[0]["type"] == "ZONE_EXIT" and ev[0]["details"]["reason"] == "LEFT_CAMERA"
    assert abs(ev[0]["value"] - 2.0) < 0.15


def test_stopping_the_service_records_the_visits_in_progress():
    z = ZoneAnalytics(DOCK, entry_confirm_sec=0.2, exit_confirm_sec=1.0)
    _, t = feed(z, 4, [(50, 50)] * 30, 0.0)
    ev = z.close_all(t)
    assert [e["type"] for e in ev] == ["ZONE_EXIT"] and ev[0]["details"]["reason"] == "SHUTDOWN"
    assert abs(ev[0]["value"] - 3.0) < 0.15 and z.worker_totals()[4]["dock"]["visits"] == 1
    assert z.close_all(t) == []


def test_restricted_zone_raises_an_intrusion_alarm_and_clear():
    z = ZoneAnalytics({"aisle": {"polygon": [[0, 0], [100, 0], [100, 100], [0, 100]], "type": "restricted"}},
                      entry_confirm_sec=0.2, exit_confirm_sec=0.2)
    ev, t = feed(z, 9, [(50, 50)] * 5 + [(300, 300)] * 5, 0.0)
    types = [e["type"] for e in ev]
    assert "ZONE_INTRUSION" in types and "ZONE_INTRUSION_CLEARED" in types
    assert next(e for e in ev if e["type"] == "ZONE_INTRUSION")["severity"] == "alarm"


def test_dwell_limit_and_capacity_rules():
    spec = {"dock": {"polygon": [[0, 0], [100, 0], [100, 100], [0, 100]], "max_dwell_sec": 1.0, "max_occupancy": 1}}
    z = ZoneAnalytics(spec, entry_confirm_sec=0.1, exit_confirm_sec=0.5)
    events, t = [], 0.0
    for _ in range(20):
        t += DT
        events += z.update({1: (20, 20), 2: (80, 80)}, t)
    types = [e["type"] for e in events]
    assert types.count("ZONE_DWELL_EXCEEDED") == 2 and types.count("ZONE_OVER_CAPACITY") == 1


def test_provisional_track_is_promoted_and_merged_identities_are_summed():
    z = ZoneAnalytics(DOCK, entry_confirm_sec=0.2, exit_confirm_sec=0.3)
    _, t = feed(z, "T12", [(50, 50)] * 10, 0.0)
    z.promote(12, 7)
    assert z.zones_of(7) == ["dock"] and z.zones_of("T12") == []
    _, t = feed(z, 7, [(300, 300)] * 5, t)
    _, t = feed(z, 9, [(50, 50)] * 10 + [(300, 300)] * 5, t)
    z.rename(9, 7)
    assert z.worker_totals()[7]["dock"]["visits"] == 2


# --------------------------------------------------------------------------- lines
LINE = {"gate": {"a": [100, 0], "b": [100, 200]}}   # vertical line x=100; left->right = positive side


def test_line_crossing_direction_and_jitter_immunity():
    c = CrossingAnalytics(LINE, {}, dead_band_px=10, cooldown_sec=0.5)
    path = [(60, 100), (80, 100), (95, 100), (104, 100), (97, 100), (106, 100), (99, 100), (130, 100), (150, 100)]
    ev, t = feed(c, 5, path, 0.0)
    assert [(e["type"], e["direction"]) for e in ev] == [("LINE_CROSS", "OUT")]   # (b-a)x(p-a): x>100 is "-"
    ev, _ = feed(c, 5, [(150, 100), (120, 100), (90, 100), (60, 100)], t)
    assert [e["direction"] for e in ev] == ["IN"]
    assert c.counts["gate"] == {"IN": 1, "OUT": 1}


def test_crossing_outside_the_segment_is_ignored_and_invert_swaps_labels():
    c = CrossingAnalytics({"gate": {"a": [100, 0], "b": [100, 200], "invert": True, "labels": ["ENTRY", "EXIT"]}}, {})
    ev, t = feed(c, 1, [(60, 400), (140, 400)], 0.0)       # passes far below the end point b
    assert ev == []
    ev, _ = feed(c, 2, [(60, 100), (140, 100)], t)
    assert [e["direction"] for e in ev] == ["ENTRY"]         # inverted


def test_crossing_hidden_by_a_short_occlusion_still_counts():
    c = CrossingAnalytics(LINE, {}, max_gap_sec=4.0)
    ev, t = feed(c, 3, [(40, 100), (60, 100)], 0.0)
    ev, _ = feed(c, 3, [(160, 100)], t + 2.0)
    assert len(ev) == 1


def test_wrong_way_on_a_one_way_line():
    c = CrossingAnalytics({"gate": {"a": [100, 0], "b": [100, 200], "allowed_direction": "IN"}}, {})
    ev, _ = feed(c, 3, [(40, 100), (160, 100)], 0.0)
    assert [e["type"] for e in ev] == ["LINE_CROSS", "WRONG_WAY"]


# --------------------------------------------------------------------------- regions
GATE = {"door": {"polygon": [[100, 0], [200, 0], [200, 200], [100, 200]], "entry_vector": [1, 0],
                 "min_displacement_px": 40}}


def test_region_pass_direction():
    c = CrossingAnalytics({}, GATE, region_exit_confirm_sec=0.2)
    ev, t = feed(c, 1, [(50 + 15 * k, 100) for k in range(14)] + [(260, 100)] * 4, 0.0)
    assert [(e["type"], e["direction"]) for e in ev] == [("REGION_PASS", "IN")]
    ev, _ = feed(c, 1, [(260 - 15 * k, 100) for k in range(14)] + [(40, 100)] * 4, t)
    assert [e["direction"] for e in ev] == ["OUT"]


def test_wandering_inside_a_region_is_not_a_pass():
    c = CrossingAnalytics({}, GATE, region_exit_confirm_sec=0.2)
    ev, _ = feed(c, 1, [(90, 100), (110, 100), (130, 100), (115, 100), (90, 100), (60, 100), (60, 100), (60, 100)],
                 0.0)
    assert ev == []


def test_worker_vanishing_through_a_door_inside_the_region_is_counted():
    c = CrossingAnalytics({}, GATE, region_lost_sec=2.0)
    ev, t = feed(c, 1, [(60, 100), (90, 100), (120, 100), (150, 100), (185, 100)], 0.0)
    ev2 = c.close_entity(1, t + 1.0)          # identity manager: he left the camera here
    assert [(e["type"], e["direction"], e["details"]["reason"]) for e in ev + ev2] == [
        ("REGION_PASS", "IN", "LEFT_CAMERA")]


# --------------------------------------------------------------------------- attribution
def test_events_of_a_provisional_track_are_attributed_when_it_gets_its_id():
    out = []
    buf = AttributionBuffer(out.append, timeout_sec=5.0)
    ev = make_event("LINE_CROSS", 1.0, track_id=12, subject="gate", direction="IN")
    buf.submit(ev)
    assert out == []
    buf.resolve(12, 7)
    assert out[0]["worker_id"] == 7 and out[0]["details"]["attributed_late"]


def test_unresolved_events_are_released_unidentified_after_the_timeout():
    out = []
    buf = AttributionBuffer(out.append, timeout_sec=5.0)
    buf.submit(make_event("ZONE_ENTER", 1.0, track_id=3, subject="dock"))
    buf.tick(3.0)
    assert out == []
    buf.tick(7.0)
    assert out[0]["worker_id"] is None and out[0]["details"]["unidentified"]


def test_entity_keys():
    assert entity_key(7, 3) == 7 and entity_key(None, 3) == "T3" and entity_key(-1, 3) == "T3"
    assert split_entity("T3") == (None, 3) and split_entity(7) == (7, None)
