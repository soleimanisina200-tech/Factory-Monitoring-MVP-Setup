"""Calibration scaling, zone/direction state renaming after merges, memory bounds and cheap overlays."""
import numpy as np
import cv2

from app.logic.calibration import GroundCalibration, load_calibration, save_calibration
from app.logic.directional_tracker import DirectionalTracker
from app.logic.geometry import MetricBEVProjector
from app.vehicle_monitoring.dashboard import VehicleSafetyDashboard, blend_rect
from app.zones.manager import MultiZoneManager

SQUARE_PX = [[100, 100], [700, 100], [700, 700], [100, 700]]
SQUARE_M = [[0.0, 0.0], [6.0, 0.0], [6.0, 6.0], [0.0, 6.0]]


def test_calibration_round_trip_and_exact_rescaling(tmp_path):
    path = str(tmp_path / "calibration.yaml")
    save_calibration(GroundCalibration(SQUARE_PX, SQUARE_M, (800, 800)), path)
    cal = load_calibration(path)
    assert cal.image_wh == (800, 800)

    # The same physical point seen in a 2x larger stream must map to the same meters.
    H_native = cal.homography((800, 800))
    H_big = cal.homography((1600, 1600))
    p_native = cv2.perspectiveTransform(np.array([[[400.0, 400.0]]], np.float32), H_native)[0, 0]
    p_big = cv2.perspectiveTransform(np.array([[[800.0, 800.0]]], np.float32), H_big)[0, 0]
    assert np.allclose(p_native, [3.0, 3.0], atol=1e-3)
    assert np.allclose(p_big, [3.0, 3.0], atol=1e-3)


def test_projector_does_not_rescale_native_resolution_points():
    """Old heuristic: points with x <= 1280 on a 2560 stream were doubled even if picked at native res."""
    pts = [[100, 900], [1200, 900], [1200, 1800], [100, 1800]]
    proj = MetricBEVProjector(pts, SQUARE_M, stream_wh=(2560, 1920), calib_ref_wh=(2560, 1920))
    assert np.allclose(proj.to_bev_meters(1200, 900), [6.0, 0.0], atol=1e-3)


def test_projector_rescales_when_reference_resolution_differs():
    proj = MetricBEVProjector(SQUARE_PX, SQUARE_M, stream_wh=(1600, 1600), calib_ref_wh=(800, 800))
    assert np.allclose(proj.to_bev_meters(1400, 1400), [6.0, 6.0], atol=1e-3)


def test_zone_state_moves_to_the_surviving_identity_after_a_merge():
    zm = MultiZoneManager({"dock": {"polygon": [[0, 0], [100, 0], [100, 100], [0, 100]]}}, 0.1, 0.1)
    t = 0.0
    for _ in range(5):
        t += 0.1
        zm.update_all({7: (50.0, 50.0)}, t)
    zm.rename_identity(7, 3)
    states = zm.zones["dock"].states
    assert 7 not in states and states[3].is_inside and states[3].global_id == 3


def test_direction_tracker_renames_and_forgets_old_people():
    dt = DirectionalTracker({"directional_tracking": {"state_ttl_sec": 10.0}})
    dt.update_all({7: (10.0, 10.0)}, 0.0)
    dt.update_all({7: (12.0, 10.0)}, 0.1)
    dt.rename_entity(7, 3)
    assert 7 not in dt.trajectories and len(dt.trajectories[3]) == 2

    dt.update_all({"T_9": (50.0, 50.0)}, 1.0)
    dt.update_all({5: (60.0, 60.0)}, 100.0)  # 99 s later: 3 and T_9 have not been seen for > 10 s
    assert set(dt.trajectories) == {5}


def test_overlays_only_touch_their_region():
    frame = np.full((400, 600, 3), 200, np.uint8)
    blend_rect(frame, 10, 10, 50, 50, (0, 0, 0), 0.5)
    assert (frame[30, 30] == 100).all() and (frame[300, 300] == 200).all()

    zm = MultiZoneManager({"z": {"polygon": [[100, 100], [200, 100], [200, 200], [100, 200]]}})
    zm.draw_zones(frame)
    assert (frame[350, 550] == 200).all()  # far outside the zone: untouched


def test_hud_draws_in_place_and_skips_jpeg_when_web_dashboard_is_off():
    dash = VehicleSafetyDashboard({"distance": {}, "dashboard": {"enable_web_dashboard": False}})
    frame = np.zeros((300, 400, 3), np.uint8)
    out = dash.render_hud(frame, {}, [])
    assert out is frame
    assert dash._latest_frame_jpeg == b""
