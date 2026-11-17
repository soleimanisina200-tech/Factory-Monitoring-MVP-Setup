"""Calibration scaling, scene geometry, settings validation, event store, cheap overlays."""
import sqlite3

import cv2
import numpy as np
import pytest
import yaml

from app.core.clock import StreamClock
from app.core.config import ConfigError, load_settings
from app.core.events import EventStore, make_event
from app.core.scene import Scene, load_scene, save_scene
from app.logic.calibration import GroundCalibration, load_calibration, save_calibration
from app.logic.geometry import MetricBEVProjector
from app.vehicle_monitoring.dashboard import VehicleSafetyDashboard, blend_rect

SQUARE_PX = [[100, 100], [700, 100], [700, 700], [100, 700]]
SQUARE_M = [[0.0, 0.0], [6.0, 0.0], [6.0, 6.0], [0.0, 6.0]]


def test_calibration_round_trip_and_exact_rescaling(tmp_path):
    path = str(tmp_path / "calibration.yaml")
    save_calibration(GroundCalibration(SQUARE_PX, SQUARE_M, (800, 800)), path)
    cal = load_calibration(path)
    assert cal.image_wh == (800, 800)
    p_native = cv2.perspectiveTransform(np.array([[[400.0, 400.0]]], np.float32), cal.homography((800, 800)))[0, 0]
    p_big = cv2.perspectiveTransform(np.array([[[800.0, 800.0]]], np.float32), cal.homography((1600, 1600)))[0, 0]
    assert np.allclose(p_native, [3.0, 3.0], atol=1e-3) and np.allclose(p_big, [3.0, 3.0], atol=1e-3)


def test_projector_does_not_rescale_native_resolution_points():
    pts = [[100, 900], [1200, 900], [1200, 1800], [100, 1800]]
    proj = MetricBEVProjector(pts, SQUARE_M, stream_wh=(2560, 1920), calib_ref_wh=(2560, 1920))
    assert np.allclose(proj.to_bev_meters(1200, 900), [6.0, 0.0], atol=1e-3)


def test_projector_rescales_when_reference_resolution_differs():
    proj = MetricBEVProjector(SQUARE_PX, SQUARE_M, stream_wh=(1600, 1600), calib_ref_wh=(800, 800))
    assert np.allclose(proj.to_bev_meters(1400, 1400), [6.0, 6.0], atol=1e-3)


def test_scene_round_trip_and_rescaling(tmp_path):
    path = str(tmp_path / "scene.yaml")
    scene = Scene(image_wh=(1280, 720),
                  zones={"dock": {"polygon": [[0, 0], [100, 0], [100, 100]], "type": "restricted"}},
                  lines={"gate": {"a": [10, 10], "b": [110, 10], "dead_band_px": 10}},
                  regions={"door": {"polygon": [[0, 0], [50, 0], [50, 50]], "entry_vector": [1, 1],
                                    "min_displacement_px": 20}})
    save_scene(scene, path)
    big = load_scene(path).scaled((2560, 1440))
    assert big.zones["dock"]["polygon"][1] == [200.0, 0.0] and big.zones["dock"]["type"] == "restricted"
    assert big.lines["gate"]["b"] == [220.0, 20.0] and big.lines["gate"]["dead_band_px"] == 20.0
    assert np.allclose(np.linalg.norm(big.regions["door"]["entry_vector"]), 1.0)


def test_legacy_settings_shapes_are_still_read(tmp_path):
    settings = {"zones": {"definitions": {"z": {"polygon": [[0, 0], [10, 0], [10, 10]]}}},
                "directional_tracking": {"lines": {"l": {"point_a": [0, 0], "point_b": [50, 0],
                                                         "invert_direction": True}}}}
    scene = load_scene(str(tmp_path / "missing.yaml"), settings)
    assert "z" in scene.zones and scene.lines["l"]["invert"] is True


def test_invalid_scene_is_refused(tmp_path):
    path = tmp_path / "scene.yaml"
    path.write_text(yaml.safe_dump({"zones": {"bad": {"polygon": [[0, 0], [1, 1]]}}}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_scene(str(path))


def test_settings_defaults_typos_and_type_errors(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("reid: {reentry_sim_thresh: 0.85, reentry_simthresh: 0.9}\n", encoding="utf-8")
    cfg = load_settings(str(p))
    assert cfg["reid"]["reentry_sim_thresh"] == 0.85 and cfg["zones"]["exit_confirm_sec"] == 1.0
    p.write_text("reid: {reentry_sim_thresh: high}\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(str(p))


def test_shipped_settings_file_is_valid():
    cfg = load_settings("configs/settings.yaml")
    assert cfg["reid"]["open_set"]["enabled"] is True


def test_command_line_overrides_are_validated(tmp_path):
    import main as app_main
    from app.core.config import persistence_mode

    cfg = load_settings("configs/settings.yaml")
    app_main.apply_override(cfg, "reid.reentry_sim_thresh=0.9")
    app_main.apply_override(cfg, "detection.tracker_config=configs/trackers/factory_ocsort.yaml")
    app_main.apply_override(cfg, "reid.open_set.enabled=false")
    assert cfg["reid"]["reentry_sim_thresh"] == 0.9 and cfg["reid"]["open_set"]["enabled"] is False
    assert cfg["detection"]["tracker_config"].endswith("factory_ocsort.yaml")
    for bad in ("reid.reentry_simthresh=0.9", "reid.reentry_sim_thresh=high", "no_equals_sign"):
        with pytest.raises(SystemExit):
            app_main.apply_override(cfg, bad)
    # YAML reads a bare `off` / `on` as a boolean: both spellings must work for the persistence mode
    p = tmp_path / "s.yaml"
    p.write_text("reid: {persistence: {mode: off}}\n", encoding="utf-8")
    assert persistence_mode(load_settings(str(p))["reid"]["persistence"]["mode"]) == "off"
    assert persistence_mode(True) == "on" and persistence_mode("Auto") == "auto"


def test_event_store_writes_db_csv_and_wall_time(tmp_path):
    clock = StreamClock(False, file_start_wall=1_700_000_000.0)
    store = EventStore(db_path=str(tmp_path / "e.db"), csv_dir=str(tmp_path / "csv"), snapshot_dir=None,
                       clock=clock, camera_id="camX", label_lookup=lambda wid: "E-7" if wid == 7 else None)
    store.emit(make_event("ZONE_EXIT", 12.5, worker_id=7, subject="dock", value=3.2, reason="WALKED_OUT"))
    store.add_alias(9, 7, 13.0)
    store.close()
    conn = sqlite3.connect(str(tmp_path / "e.db"))
    row = conn.execute("SELECT wall_epoch, camera_id, type, worker_id, worker_label, value FROM events").fetchone()
    assert row == (1_700_000_012.5, "camX", "ZONE_EXIT", 7, "E-7", 3.2)
    assert conn.execute("SELECT old_id, new_id FROM identity_aliases").fetchone() == (9, 7)
    assert len(list((tmp_path / "csv").glob("events_camX_*.csv"))) == 1


def test_overlays_only_touch_their_region():
    frame = np.full((400, 600, 3), 200, np.uint8)
    blend_rect(frame, 10, 10, 50, 50, (0, 0, 0), 0.5)
    assert (frame[30, 30] == 100).all() and (frame[300, 300] == 200).all()
    from app.analytics.zones import ZoneAnalytics

    ZoneAnalytics({"z": {"polygon": [[100, 100], [200, 100], [200, 200], [100, 200]]}}).draw(frame, 0.0)
    assert (frame[350, 550] == 200).all()


def test_hud_draws_in_place():
    dash = VehicleSafetyDashboard({"distance": {}})
    frame = np.zeros((300, 400, 3), np.uint8)
    assert dash.render_hud(frame, {}, [], now=0.0) is frame
