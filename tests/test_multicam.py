"""Cross-camera identity: two camera linkers sharing one registry file (synchronous, no threads)."""
import numpy as np

from app.multicam.registry import CameraTopology, CrossCameraLinker, load_person_map

DIM = 32
T0 = 1_700_000_000.0


def unit(i):
    v = np.zeros(DIM, np.float32)
    v[i] = 1.0
    return v


def near(base, axis, sim):
    v = sim * base + np.sqrt(1 - sim * sim) * unit(axis)
    return (v / np.linalg.norm(v)).astype(np.float32)


def tr(local_id, emb, first, last, active=False, label=None, n_obs=10):
    return {"local_id": local_id, "label": label, "embs": np.stack([emb, emb]), "n_obs": n_obs,
            "first_seen": T0 + first, "last_seen": T0 + last, "active": active}


def linker(path, cam, **kw):
    kw.setdefault("decide_after_sec", 5.0)
    return CrossCameraLinker(str(path), cam, background=False, **kw)


def test_same_worker_gets_the_same_person_id_in_the_second_camera(tmp_path):
    db = tmp_path / "cc.db"
    cam1, cam2 = linker(db, "cam1"), linker(db, "cam2")
    cam1.sync_now([tr(1, unit(0), 0, 30)], T0 + 31)            # nobody else known yet
    cam1.sync_now([tr(1, unit(0), 0, 30)], T0 + 40)            # -> new person after decide_after_sec
    p1 = cam1.person_of(1)
    assert p1 is not None
    cam2.sync_now([tr(7, near(unit(0), 3, 0.93), 60, 70, active=True)], T0 + 70)
    assert cam2.person_of(7) == p1
    assert load_person_map(str(db)) == {("cam1", 1): p1, ("cam2", 7): p1}


def test_someone_visible_in_another_camera_at_the_same_time_is_a_different_person(tmp_path):
    db = tmp_path / "cc.db"
    cam1, cam2 = linker(db, "cam1"), linker(db, "cam2")
    cam1.sync_now([tr(1, unit(0), 0, 100, active=True)], T0 + 50)
    cam1.sync_now([tr(1, unit(0), 0, 100, active=True)], T0 + 60)
    cam2.sync_now([tr(7, unit(0), 40, 90, active=True)], T0 + 90)     # identical look, same time, other place
    cam2.sync_now([tr(7, unit(0), 40, 90, active=True)], T0 + 100)
    assert cam2.person_of(7) is not None and cam2.person_of(7) != cam1.person_of(1)


def test_overlapping_cameras_may_see_the_same_person_at_once(tmp_path):
    db = tmp_path / "cc.db"
    topo = CameraTopology({"overlaps": [["cam1", "cam2"]]})
    cam1, cam2 = linker(db, "cam1", topology=topo), linker(db, "cam2", topology=topo)
    cam1.sync_now([tr(1, unit(0), 0, 100, active=True)], T0 + 50)
    cam1.sync_now([tr(1, unit(0), 0, 100, active=True)], T0 + 60)
    cam2.sync_now([tr(7, unit(0), 40, 90, active=True)], T0 + 90)
    assert cam2.person_of(7) == cam1.person_of(1)


def test_two_lookalikes_are_not_guessed(tmp_path):
    db = tmp_path / "cc.db"
    cam1, cam2 = linker(db, "cam1"), linker(db, "cam2")
    a, b = unit(0), near(unit(0), 1, 0.97)                        # two workers in the same uniform
    snap = [tr(1, a, 0, 20), tr(2, b, 30, 50)]
    cam1.sync_now(snap, T0 + 55)
    cam1.sync_now(snap, T0 + 65)
    assert cam1.person_of(1) != cam1.person_of(2)
    x = (a + b) / np.linalg.norm(a + b)                           # looks like both
    cam2.sync_now([tr(9, x.astype(np.float32), 100, 103, active=True)], T0 + 103)
    assert cam2.person_of(9) is None                              # undecided: margin too small, still waiting
    cam2.sync_now([tr(9, x.astype(np.float32), 100, 120, active=True)], T0 + 120)
    assert cam2.person_of(9) not in (None, cam1.person_of(1), cam1.person_of(2))   # -> its own new person


def test_transit_time_between_cameras_is_enforced(tmp_path):
    db = tmp_path / "cc.db"
    topo = CameraTopology({"transitions": [{"from": "cam1", "to": "cam2", "min_sec": 30, "max_sec": 600}]})
    cam1, cam2 = linker(db, "cam1", topology=topo), linker(db, "cam2", topology=topo)
    cam1.sync_now([tr(1, unit(0), 0, 100)], T0 + 101)
    cam1.sync_now([tr(1, unit(0), 0, 100)], T0 + 110)
    cam2.sync_now([tr(7, unit(0), 105, 115, active=True)], T0 + 115)    # 5 s later: cannot have walked there
    cam2.sync_now([tr(7, unit(0), 105, 125, active=True)], T0 + 125)
    assert cam2.person_of(7) != cam1.person_of(1)
    cam2.sync_now([tr(7, unit(0), 105, 125), tr(8, unit(0), 200, 210, active=True)], T0 + 210)   # 100 s later: ok
    assert cam2.person_of(8) == cam1.person_of(1)


def test_helmet_marker_links_across_cameras_whatever_the_appearance(tmp_path):
    db = tmp_path / "cc.db"
    cam1, cam2 = linker(db, "cam1"), linker(db, "cam2")
    cam1.sync_now([tr(1, unit(0), 0, 30, label="E-7")], T0 + 31)
    cam2.sync_now([tr(4, unit(9), 60, 70, active=True, label="E-7")], T0 + 70)     # completely different look
    assert cam2.person_of(4) == cam1.person_of(1) is not None
    cam2.sync_now([tr(4, unit(9), 60, 70), tr(5, unit(0), 80, 90, active=True, label="E-8")], T0 + 90)
    assert cam2.person_of(5) != cam1.person_of(1)                  # same look, other marker


def test_local_merge_moves_the_person_link(tmp_path):
    db = tmp_path / "cc.db"
    cam1, cam2 = linker(db, "cam1"), linker(db, "cam2")
    cam1.sync_now([tr(1, unit(0), 0, 30)], T0 + 31)
    cam1.sync_now([tr(1, unit(0), 0, 30)], T0 + 40)
    cam2.sync_now([tr(7, unit(0), 60, 70, active=True)], T0 + 70)
    p = cam2.person_of(7)
    cam2.notify_merge(7, 3)                                         # local manager: ID 7 was really ID 3
    cam2.sync_now([tr(3, unit(0), 50, 80, active=True)], T0 + 80)
    assert cam2.person_of(3) == p == cam1.person_of(1) and cam2.person_of(7) is None


def test_new_person_is_merged_later_when_strong_evidence_arrives(tmp_path):
    db = tmp_path / "cc.db"
    cam1, cam2 = linker(db, "cam1"), linker(db, "cam2")
    cam1.sync_now([tr(1, unit(0), 0, 30)], T0 + 31)
    cam1.sync_now([tr(1, unit(0), 0, 30)], T0 + 40)
    weak = near(unit(0), 5, 0.6)                                    # first crops are poor
    cam2.sync_now([tr(7, weak, 60, 62, active=True)], T0 + 62)
    cam2.sync_now([tr(7, weak, 60, 70, active=True)], T0 + 70)
    assert cam2.person_of(7) not in (None, cam1.person_of(1))
    cam2.sync_now([tr(7, unit(0), 60, 80, active=True, n_obs=30)], T0 + 80)   # now a clear view
    assert cam2.person_of(7) == cam1.person_of(1)
