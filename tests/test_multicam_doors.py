"""
Door handover between NON-overlapping cameras (hall -> yard): place and timing decide, appearance only has to agree.
The light differs a lot between the hall and the yard, so the same worker scores only ~0.70 across the two cameras
(below the normal cross-camera appearance threshold of 0.80).
"""
import numpy as np
import pytest

from app.multicam.registry import CameraTopology, CrossCameraLinker

DIM = 32
T0 = 1_700_000_000.0

DOORS = {"doors": [{"name": "hall_main_door",
                    "from": {"camera": "hall", "region": "hall_door"},
                    "to": {"camera": "yard", "region": "yard_entrance"},
                    "min_sec": 2, "max_sec": 15}]}


def unit(i):
    v = np.zeros(DIM, np.float32)
    v[i] = 1.0
    return v


def near(base, axis, sim):
    v = sim * base + np.sqrt(1 - sim * sim) * unit(axis)
    return (v / np.linalg.norm(v)).astype(np.float32)


def tr(local_id, emb, first, last, active=False, entry=None, exit_gate=None, stature=None, n_obs=10):
    return {"local_id": local_id, "label": None, "embs": np.stack([emb, emb]), "n_obs": n_obs,
            "first_seen": T0 + first, "last_seen": T0 + last, "active": active,
            "entry_gate": entry, "entry_wall": T0 + first,
            "exit_gate": exit_gate, "exit_wall": (T0 + last) if exit_gate else None,
            "stature": stature}


def linkers(tmp_path, topology=DOORS):
    topo = CameraTopology(topology)
    db = str(tmp_path / "cc.db")
    return (CrossCameraLinker(db, "hall", topo, background=False, decide_after_sec=5.0),
            CrossCameraLinker(db, "yard", topo, background=False, decide_after_sec=5.0))


def hall_worker(hall, lid, emb, first, last, **kw):
    snap = [tr(lid, emb, first, last, exit_gate="door:hall_door", **kw)]
    hall.sync_now(snap, T0 + last + 1)
    hall.sync_now(snap, T0 + last + 8)           # -> person after decide_after_sec
    return snap


def test_worker_walking_out_of_the_hall_door_keeps_his_person_id_in_the_yard(tmp_path):
    hall, yard = linkers(tmp_path)
    a = unit(0)
    hall_worker(hall, 1, a, 0, 100)
    p = hall.person_of(1)
    assert p is not None
    yard.sync_now([tr(5, near(a, 3, 0.70), 106, 112, active=True, entry="door:yard_entrance")], T0 + 112)
    assert yard.person_of(5) == p


def test_without_the_door_definition_the_same_case_becomes_a_new_person(tmp_path):
    hall, yard = linkers(tmp_path, topology={})
    a = unit(0)
    hall_worker(hall, 1, a, 0, 100)
    snap = [tr(5, near(a, 3, 0.70), 106, 112, active=True, entry="door:yard_entrance")]
    yard.sync_now(snap, T0 + 112)
    yard.sync_now(snap, T0 + 125)
    assert yard.person_of(5) not in (None, hall.person_of(1))


def test_arriving_outside_the_walking_time_window_is_not_a_door_handover(tmp_path):
    hall, yard = linkers(tmp_path)
    a = unit(0)
    hall_worker(hall, 1, a, 0, 100)
    yard.sync_now([tr(5, near(a, 3, 0.70), 160, 170, active=True, entry="door:yard_entrance")], T0 + 170)
    assert yard.person_of(5) != hall.person_of(1)


def test_entering_the_yard_elsewhere_is_not_a_door_handover(tmp_path):
    hall, yard = linkers(tmp_path)
    a = unit(0)
    hall_worker(hall, 1, a, 0, 100)
    yard.sync_now([tr(5, near(a, 3, 0.70), 106, 112, active=True, entry="edge:left")], T0 + 112)
    assert yard.person_of(5) != hall.person_of(1)


def test_a_visitor_without_uniform_is_not_given_the_workers_id(tmp_path):
    hall, yard = linkers(tmp_path)
    a = unit(0)
    hall_worker(hall, 1, a, 0, 100)
    yard.sync_now([tr(5, near(a, 3, 0.30), 106, 112, active=True, entry="door:yard_entrance")], T0 + 112)
    assert yard.person_of(5) != hall.person_of(1)


def test_two_lookalikes_through_the_door_are_not_guessed(tmp_path):
    hall, yard = linkers(tmp_path)
    a, b = unit(0), near(unit(0), 1, 0.97)
    snap = [tr(1, a, 0, 100, exit_gate="door:hall_door"), tr(2, b, 0, 101, exit_gate="door:hall_door")]
    hall.sync_now(snap, T0 + 102)
    hall.sync_now(snap, T0 + 110)
    x = (a + b) / np.linalg.norm(a + b)
    yard.sync_now([tr(5, x.astype(np.float32), 106, 112, active=True, entry="door:yard_entrance")], T0 + 112)
    assert yard.person_of(5) not in (hall.person_of(1), hall.person_of(2))


def test_two_workers_through_the_door_are_told_apart_by_appearance(tmp_path):
    hall, yard = linkers(tmp_path)
    a, b = unit(0), unit(1)
    snap = [tr(1, a, 0, 100, exit_gate="door:hall_door"), tr(2, b, 0, 101, exit_gate="door:hall_door")]
    hall.sync_now(snap, T0 + 102)
    hall.sync_now(snap, T0 + 110)
    yard.sync_now([tr(5, near(b, 3, 0.70), 106, 112, active=True, entry="door:yard_entrance")], T0 + 112)
    assert yard.person_of(5) == hall.person_of(2)


def test_one_exit_is_used_by_one_person_only(tmp_path):
    hall, yard = linkers(tmp_path)
    a = unit(0)
    hall_worker(hall, 1, a, 0, 100)
    yard.sync_now([tr(5, near(a, 3, 0.70), 104, 112, entry="door:yard_entrance")], T0 + 112)
    assert yard.person_of(5) == hall.person_of(1)
    snap = [tr(5, near(a, 3, 0.70), 104, 112, entry="door:yard_entrance"),
            tr(6, near(a, 4, 0.70), 108, 116, active=True, entry="door:yard_entrance")]
    yard.sync_now(snap, T0 + 116)
    yard.sync_now(snap, T0 + 125)
    assert yard.person_of(6) not in (None, hall.person_of(1))


def test_clearly_different_height_blocks_the_door_handover(tmp_path):
    hall, yard = linkers(tmp_path)
    a = unit(0)
    hall_worker(hall, 1, a, 0, 100, stature=(0.97, 0.004, 30))
    yard.sync_now([tr(5, near(a, 3, 0.75), 106, 112, active=True, entry="door:yard_entrance",
                      stature=(1.07, 0.004, 30))], T0 + 112)
    assert yard.person_of(5) != hall.person_of(1)


def test_bad_door_definitions_are_reported():
    with pytest.raises(ValueError):
        CameraTopology({"doors": [{"name": "x", "from": {"camera": "hall"}, "to": {"camera": "yard", "edge": "left"}}]})
    with pytest.raises(ValueError):
        CameraTopology({"doors": [{"name": "x", "from": {"camera": "a", "edge": "left"},
                                   "to": {"camera": "b", "edge": "up"}}]})
