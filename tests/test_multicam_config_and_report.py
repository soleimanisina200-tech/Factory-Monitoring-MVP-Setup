"""configs/cameras.yaml handling and the per-person report across cameras."""
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from app.core.clock import StreamClock
from app.core.config import ConfigError
from app.core.events import EventStore, make_event
from app.multicam.cameras import camera_ids, load_camera_setup
from app.multicam.registry import CrossCameraLinker

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import report as report_script  # noqa: E402

T0 = 1_700_000_000.0


def write_cameras(tmp_path, extra=None):
    base = tmp_path / "settings.yaml"
    base.write_text(yaml.safe_dump({"system": {"display_gui": True}, "reid": {"reentry_sim_thresh": 0.84}}),
                    encoding="utf-8")
    data = {
        "base_settings": str(base),
        "cross_camera": {"registry_path": str(tmp_path / "cc.db"), "topology": {"overlaps": [["cam1", "cam2"]]}},
        "cameras": {
            "cam1": {"source": "rtsp://u:p@10.0.0.1/s", "display_gui": False,
                     "settings": {"reid": {"exit_edges": ["left"]}}},
            "cam2": {"source": "rtsp://u:p@10.0.0.2/s", "vehicle_config": "",
                     "scene": "custom/scene2.yaml", "calibration": "custom/cal2.yaml"},
        },
    }
    data.update(extra or {})
    path = tmp_path / "cameras.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return str(path)


def test_camera_setup_merges_overrides_and_scopes_paths(tmp_path):
    path = write_cameras(tmp_path)
    assert camera_ids(path) == ["cam1", "cam2"]
    s1 = load_camera_setup(path, "cam1")
    c = s1.settings
    assert c["system"]["camera_id"] == "cam1" and c["stream"]["source"].endswith("10.0.0.1/s")
    assert c["system"]["display_gui"] is False
    assert c["reid"]["exit_edges"] == ["left"] and c["reid"]["reentry_sim_thresh"] == 0.84   # override + base
    assert c["cross_camera"]["enabled"] is True and c["cross_camera"]["topology"]["overlaps"] == [["cam1", "cam2"]]
    # outputs that must not be shared between camera processes live in a per-camera folder
    assert Path(c["system"]["log_dir"]).name == "cam1"
    assert Path(c["reid"]["decision_log_path"]).parent.name == "cam1"
    assert Path(c["system"]["heartbeat_path"]).parent.name == "cam1"
    assert c["scene"]["path"] == "configs/cameras/cam1/scene.yaml"
    assert s1.calibration == "configs/cameras/cam1/calibration.yaml"
    assert s1.vehicle_config == "configs/vehicle_monitoring.yaml"

    s2 = load_camera_setup(path, "cam2")
    assert (c["web"]["port"], s2.settings["web"]["port"]) == (8050, 8051)     # one web API per camera process
    assert s2.vehicle_config is None and s2.settings["scene"]["path"] == "custom/scene2.yaml"
    assert s2.calibration == "custom/cal2.yaml" and s2.settings["reid"]["exit_edges"] == ["left", "right", "top",
                                                                                         "bottom"]


def test_unknown_camera_and_bad_override_are_refused(tmp_path):
    path = write_cameras(tmp_path)
    with pytest.raises(ConfigError):
        load_camera_setup(path, "cam9")
    bad = write_cameras(tmp_path, {"cameras": {"cam1": {"settings": {"reid": {"reentry_sim_thresh": "high"}}}}})
    with pytest.raises(ConfigError):
        load_camera_setup(bad, "cam1")


def test_report_merges_one_person_seen_by_two_cameras(tmp_path):
    db, reg = str(tmp_path / "events.db"), str(tmp_path / "cc.db")
    clock = StreamClock(False, T0)
    cam1 = EventStore(db_path=db, csv_dir=None, snapshot_dir=None, clock=clock, camera_id="cam1")
    cam2 = EventStore(db_path=db, csv_dir=None, snapshot_dir=None, clock=clock, camera_id="cam2")
    # the same worker: local ID 3 in cam1 (first seen as temporary ID 9, merged) and local ID 3 ... no: 5 in cam2
    cam1.emit(make_event("ZONE_EXIT", 100.0, worker_id=9, subject="dock", value=60.0))
    cam1.add_alias(9, 3, 101.0)
    cam1.emit(make_event("LINE_CROSS", 120.0, worker_id=3, subject="gate", direction="OUT"))
    cam2.emit(make_event("ZONE_EXIT", 400.0, worker_id=5, subject="dock", value=30.0))
    cam2.emit(make_event("ZONE_EXIT", 500.0, worker_id=3, subject="store", value=10.0))   # another man, same number
    cam1.close()
    cam2.close()

    emb = np.zeros((2, 8), np.float32)
    emb[:, 0] = 1.0
    l1 = CrossCameraLinker(reg, "cam1", background=False, decide_after_sec=1.0)
    l2 = CrossCameraLinker(reg, "cam2", background=False, decide_after_sec=1.0)
    t = lambda lid, a, b: {"local_id": lid, "label": None, "embs": emb, "n_obs": 10,   # noqa: E731
                           "first_seen": T0 + a, "last_seen": T0 + b, "active": False}
    l1.sync_now([t(3, 0, 130)], T0 + 140)
    l1.sync_now([t(3, 0, 130)], T0 + 150)
    l2.sync_now([t(5, 300, 400)], T0 + 410)
    assert l2.person_of(5) == l1.person_of(3)

    events, resolve, labels = report_script.load(db, T0 - 10, T0 + 86400, None)
    rep = report_script.build(events, resolve, labels, report_script.load_persons(reg))
    person = f"P{l1.person_of(3)}"
    w = rep["workers"]
    assert set(w) == {person, "cam2:3"}                       # cam2's local 3 is NOT cam1's local 3
    assert w[person]["zone_sec"]["dock"] == 90.0 and w[person]["zone_visits"]["dock"] == 2
    assert w[person]["cross"]["gate:OUT"] == 1
    assert w["cam2:3"]["zone_sec"]["store"] == 10.0
