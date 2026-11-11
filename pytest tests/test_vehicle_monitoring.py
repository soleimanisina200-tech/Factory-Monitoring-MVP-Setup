import numpy as np
from app.vehicle_monitoring.distance_estimator import RealWorldDistanceEstimator
from app.vehicle_monitoring.driver_vehicle_fsm import DriverVehicleMonitorFSM, VehicleSafetyState


def get_test_config():
    return {
        "pairing": {
            "entry_iou_threshold": 0.25,
            "entry_confirmation_frames": 2,
            "exit_confirmation_frames": 2,
            "reid_similarity_threshold": 0.65,
            "max_unattended_timeout_sec": 60.0,
        },
        "distance": {
            "safe_distance_threshold_m": 3.0,
            "warning_distance_threshold_m": 2.0,
            "ema_smoothing_alpha": 1.0,  # 1.0 for exact deterministic test assertions
            "max_step_jump_m": 10.0,
            "min_bbox_aspect_ratio": 1.15,
            "border_margin_px": 10,
        },
        "calibration": {
            # Linear 100px = 1.0m test mapping on ground plane
            "image_points_px": [[100, 100], [700, 100], [700, 700], [100, 700]],
            "world_points_m": [[0.0, 0.0], [6.0, 0.0], [6.0, 6.0], [0.0, 6.0]],
        },
    }


def test_real_world_homography_distance_in_meters():
    cfg = get_test_config()
    estimator = RealWorldDistanceEstimator(cfg)

    # Forklift base from x=100..200 (0.0m..1.0m), y_bottom=300 (2.0m)
    vehicle_bbox = [100, 150, 200, 300]
    # Driver feet at x_center=550 (4.5m), y_bottom=300 (2.0m) -> 3.5m from forklift right edge (1.0m)
    driver_bbox = [520, 180, 580, 300]

    meas = estimator.estimate_driver_vehicle_distance(
        driver_id="1",
        driver_bbox=driver_bbox,
        vehicle_id="10",
        vehicle_bbox=vehicle_bbox,
        frame_shape=(800, 800, 3),
    )

    assert not meas.is_uncertain
    assert abs(meas.distance_m - 3.50) < 0.05


def test_occlusion_flags_uncertainty_instead_of_false_distance():
    cfg = get_test_config()
    estimator = RealWorldDistanceEstimator(cfg)

    vehicle_bbox = [100, 150, 200, 300]
    # Truncated bbox (height/width < 1.15) simulating lower-body occlusion behind pallet
    occluded_driver_bbox = [500, 250, 600, 300]

    meas = estimator.estimate_driver_vehicle_distance(
        driver_id="1",
        driver_bbox=occluded_driver_bbox,
        vehicle_id="10",
        vehicle_bbox=vehicle_bbox,
        frame_shape=(800, 800, 3),
    )

    assert meas.is_uncertain is True
    assert "OCCLUDED" in meas.uncertainty_reason


def test_driver_pairing_exit_and_alarm_fsm():
    cfg = get_test_config()
    estimator = RealWorldDistanceEstimator(cfg)
    fsm = DriverVehicleMonitorFSM(cfg, estimator)
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    vehicle = [{"id": "101", "bbox": [100, 150, 200, 300]}]
    driver_inside = [{"id": "5", "bbox": [120, 170, 180, 290], "embedding": emb}]

    # 1. Confirm driver boarding (2 frames)
    fsm.update(vehicle, driver_inside, (800, 800, 3))
    records, events = fsm.update(vehicle, driver_inside, (800, 800, 3))
    assert records["101"].state == VehicleSafetyState.DRIVER_ONBOARD
    assert records["101"].associated_driver_id == "5"

    # 2. Driver exits nearby (1.0m away -> SAFE)
    driver_safe = [{"id": "5", "bbox": [270, 180, 330, 300], "embedding": emb}]
    fsm.update(vehicle, driver_safe, (800, 800, 3))
    records, events = fsm.update(vehicle, driver_safe, (800, 800, 3))
    assert records["101"].state == VehicleSafetyState.DRIVER_EXITED_SAFE
    assert any(e["event_type"] == "DRIVER_EXITED" for e in events)

    # 3. Driver moves > 3.0m away with a switched Track ID ("99") recovered via Re-ID -> ALARM
    driver_far_new_id = [{"id": "99", "bbox": [550, 180, 610, 300], "embedding": emb}]
    records, events = fsm.update(vehicle, driver_far_new_id, (800, 800, 3))
    assert records["101"].associated_driver_id == "99"
    assert records["101"].state == VehicleSafetyState.DISTANCE_VIOLATION_ALARM
    assert records["101"].alarm_active is True
    assert any(e["event_type"] == "SAFE_DISTANCE_VIOLATION_ALARM" for e in events)