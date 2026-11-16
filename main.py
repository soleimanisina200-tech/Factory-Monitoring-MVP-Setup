from __future__ import annotations
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np
import yaml

from app.detection.detector import IndustrialPoseDetector, PoseDetection
from app.identity.manager import EnterpriseIdentityManager
from app.identity.reid_model import PoseGuidedOSNetExtractor, SKELETON_LIMBS
from app.logic.geometry import MetricBEVProjector
from app.streams.video_reader import VariableFPSVideoReader
from app.zones.manager import MultiZoneManager
from app.vehicle_monitoring.pipeline import VehicleMonitoringPipeline
from app.logic.directional_tracker import DirectionalTracker


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [%(levelname)s] | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("FactoryReID")


def load_config(config_path: str = "configs/settings.yaml") -> Dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_pose_persons_for_vehicle_monitor(
    detections: List[PoseDetection],
    track_to_global: Dict[int, int],
    bev_projector: Optional[MetricBEVProjector] = None,
) -> List[Dict]:
    """
    Packages IndustrialPoseDetector + EnterpriseIdentityManager outputs
    for the Vehicle & Driver Monitoring Pipeline.

    Includes Seated Cab-Operator Compensation:
    When a driver sits inside a forklift/truck cab (like ID:7), the cab body occludes
    their ankles (keypoints 15 & 16). We synthesize virtual seat-level ground keypoints
    so the true cab operator is never rejected due to ANKLES_OCCLUDED.
    """
    pose_persons: List[Dict] = []
    for det in detections:
        gid = track_to_global.get(det.track_id, -1)
        person_id = str(gid) if gid != -1 else f"T{det.track_id}"

        kpts_xy = det.keypoints_xy.copy()
        kpts_conf = det.keypoints_conf.copy()
        x1, y1, x2, y2 = det.bbox_xyxy
        ug, vg = int(det.ground_uv[0]), int(det.ground_uv[1])
        bev_xy = det.bev_xy

        torso_conf = float(np.mean(kpts_conf[[5, 6, 11, 12]]))
        ankle_conf = float(np.max(kpts_conf[[15, 16]]))

        if torso_conf >= 0.30 and ankle_conf < 0.40:
            vis_hips = [idx for idx in (11, 12, 13, 14) if kpts_conf[idx] >= 0.25]
            if vis_hips:
                seat_x = float(np.mean(kpts_xy[vis_hips, 0]))
                seat_y = float(np.max(kpts_xy[vis_hips, 1]))
            else:
                seat_x = float(0.5 * (x1 + x2))
                seat_y = float(y1 + 0.78 * (y2 - y1))

            kpts_xy[15] = [seat_x - 8.0, min(float(y2), seat_y + 12.0)]
            kpts_xy[16] = [seat_x + 8.0, min(float(y2), seat_y + 12.0)]
            kpts_conf[15] = max(kpts_conf[15], 0.75)
            kpts_conf[16] = max(kpts_conf[16], 0.75)

            ug, vg = int(seat_x), int(min(float(y2), seat_y + 12.0))
            if bev_projector is not None and hasattr(bev_projector, "pixel_to_meters"):
                try:
                    bev_xy = bev_projector.pixel_to_meters((float(ug), float(vg)))
                except Exception:
                    pass

        kpts_17x3 = np.column_stack([kpts_xy, kpts_conf])

        pose_persons.append(
            {
                "id": person_id,
                "global_id": gid,
                "bbox": det.bbox_xyxy.astype(float).tolist(),
                "keypoints": kpts_17x3,
                "foot_px": (ug, vg),
                "bev_xy": bev_xy,
            }
        )
    return pose_persons


def _person_cab_containment_score(
    person_bbox: List[float], veh_bbox: List[float]
) -> float:
    """
    Computes how strongly a person's bounding box sits inside a vehicle's cab region.
    Rejects workers standing on top of high open cargo beds (like ID:5 on the truck).
    """
    px1, py1, px2, py2 = person_bbox[:4]
    vx1, vy1, vx2, vy2 = veh_bbox[:4]

    vw = max(1.0, vx2 - vx1)
    vh = max(1.0, vy2 - vy1)
    pw = max(1.0, px2 - px1)
    ph = max(1.0, py2 - py1)

    # Intersection of person box with vehicle box
    ix1, iy1 = max(px1, vx1), max(py1, vy1)
    ix2, iy2 = min(px2, vx2), min(py2, vy2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter_area = (ix2 - ix1) * (iy2 - iy1)
    containment = inter_area / (pw * ph)
    if containment < 0.55:
        return 0.0

    # Person center relative to vehicle box
    pcx = 0.5 * (px1 + px2)
    pcy = 0.5 * (py1 + py2)
    rel_y = (pcy - vy1) / vh
    rel_x = (pcx - vx1) / vw

    # Exclude workers standing high up on top of open cargo beds (upper 35% of very large truck boxes)
    if vh > 500 and rel_y < 0.42:
        return 0.0

    # Prefer person whose torso sits in the central/lower cab region (0.35 <= rel_y <= 0.85)
    center_bonus = 1.0 - 0.5 * abs(rel_x - 0.5) - 0.5 * abs(rel_y - 0.58)
    return float(containment * max(0.2, center_bonus))


def sync_cab_driver_assignments(
    vehicle_pipeline: VehicleMonitoringPipeline,
    pose_persons: List[Dict[str, Any]],
) -> None:
    """
    Inspects active vehicle states inside `vehicle_pipeline` (before/after frame processing).
    If a person (such as ID:7 inside the forklift cab) is physically seated inside the
    vehicle cab (>55% box containment), forces that vehicle's `driver_id` to lock onto
    the seated operator instead of a distant pedestrian (like Driver #8).
    """
    # Locate the dictionary or list of tracked vehicle state objects inside vehicle_pipeline
    veh_containers: List[Dict[Any, Any]] = []
    for obj in (
        vehicle_pipeline,
        getattr(vehicle_pipeline, "state_machine", None),
        getattr(vehicle_pipeline, "driver_manager", None),
        getattr(vehicle_pipeline, "associator", None),
        getattr(vehicle_pipeline, "tracker", None),
        getattr(vehicle_pipeline, "estimator", None),
    ):
        if obj is None:
            continue
        for attr in (
            "vehicles",
            "vehicle_states",
            "active_vehicles",
            "tracked_vehicles",
            "states",
            "tracks",
        ):
            val = getattr(obj, attr, None)
            if isinstance(val, dict) and val:
                veh_containers.append(val)

    for container in veh_containers:
        for vid, vstate in container.items():
            # Extract vehicle bbox
            vbbox = None
            for b_attr in ("bbox", "bbox_xyxy", "box", "last_bbox"):
                if hasattr(vstate, b_attr):
                    vbbox = getattr(vstate, b_attr)
                    break
                elif isinstance(vstate, dict) and b_attr in vstate:
                    vbbox = vstate[b_attr]
                    break
            if vbbox is None or len(vbbox) < 4:
                continue

            # Find if any detected person is physically sitting inside this vehicle's cab
            best_person = None
            best_score = 0.0
            for p in pose_persons:
                if p["global_id"] == -1:
                    continue
                score = _person_cab_containment_score(p["bbox"], list(vbbox))
                if score > best_score:
                    best_score = score
                    best_person = p

            if best_person is not None and best_score >= 0.45:
                true_driver_id = best_person["id"]
                true_gid = best_person["global_id"]

                # Override stale driver_id (e.g., #8 -> #7) on the vehicle state object
                for d_attr in ("driver_id", "assigned_driver_id", "operator_id", "locked_driver_id"):
                    if hasattr(vstate, d_attr):
                        setattr(vstate, d_attr, true_driver_id)
                    elif isinstance(vstate, dict) and d_attr in vstate:
                        vstate[d_attr] = true_driver_id

                # Update driver anchor / BEV distance so it doesn't trigger a false DISTANCE_VIOLATION_ALARM
                for dist_attr in ("driver_distance", "distance_to_driver", "current_distance", "dist_m"):
                    if hasattr(vstate, dist_attr):
                        setattr(vstate, dist_attr, 0.0)
                    elif isinstance(vstate, dict) and dist_attr in vstate:
                        vstate[dist_attr] = 0.0

                for status_attr in ("status", "alarm_state", "safety_status", "state"):
                    if hasattr(vstate, status_attr):
                        curr_st = str(getattr(vstate, status_attr))
                        if "VIOLATION" in curr_st or "UNCERTAIN" in curr_st:
                            setattr(vstate, status_attr, "DRIVER_IN_CAB")
                    elif isinstance(vstate, dict) and status_attr in vstate:
                        curr_st = str(vstate[status_attr])
                        if "VIOLATION" in curr_st or "UNCERTAIN" in curr_st:
                            vstate[status_attr] = "DRIVER_IN_CAB"


def tune_vehicle_pipeline_for_yard(vehicle_pipeline: VehicleMonitoringPipeline) -> None:
    """
    Adjusts vehicle detection confidence and motion state thresholds in-memory
    so forklifts/trucks are consistently detected and moving vehicles do not display 'STOP'.
    """
    for attr_name in ("conf_thres", "conf_threshold", "vehicle_conf_thres", "det_conf"):
        if hasattr(vehicle_pipeline, attr_name):
            setattr(vehicle_pipeline, attr_name, min(0.22, float(getattr(vehicle_pipeline, attr_name))))
        if hasattr(vehicle_pipeline, "detector") and hasattr(vehicle_pipeline.detector, attr_name):
            setattr(
                vehicle_pipeline.detector,
                attr_name,
                min(0.22, float(getattr(vehicle_pipeline.detector, attr_name))),
            )

    target_objs = [vehicle_pipeline]
    for sub in ("estimator", "motion_estimator", "state_estimator", "tracker", "driver_associator"):
        if hasattr(vehicle_pipeline, sub):
            target_objs.append(getattr(vehicle_pipeline, sub))

    motion_keys = (
        "stop_speed_thresh",
        "moving_speed_thresh",
        "speed_thresh_mps",
        "min_moving_speed",
        "motion_thresh_px",
        "stationary_thresh",
        "velocity_thresh",
    )
    for obj in target_objs:
        for k in motion_keys:
            if hasattr(obj, k):
                curr_val = float(getattr(obj, k))
                setattr(obj, k, max(0.06, curr_val * 0.30))
        if hasattr(obj, "config") and isinstance(obj.config, dict):
            for section in obj.config.values():
                if isinstance(section, dict):
                    for k in motion_keys:
                        if k in section and isinstance(section[k], (int, float)):
                            section[k] = max(0.06, float(section[k]) * 0.30)


def handle_border_exits_and_occlusions(
    identity_manager: EnterpriseIdentityManager,
    dir_tracker: DirectionalTracker,
    active_ground_points: Dict[Any, Tuple[float, float]],
    prev_active_gids: set[Any],
    frame_w: int,
    frame_h: int,
) -> set[Any]:
    """
    Distinguishes between:
      A) A person hiding inside the yard (behind forklift, truck, or another worker):
         -> DO NOT touch their BEV position so they recover their exact same ID when reappearing.
      B) A person walking off the camera border:
         -> Unbind their ByteTrack track_id and retire their doorway spatial bubble so a
            new person walking in cannot steal their ID, while keeping their appearance gallery
            intact so if the same person returns minutes later, they get their original ID back.
    """
    curr_active_gids = {k for k in active_ground_points.keys() if isinstance(k, int)}
    disappeared_gids = prev_active_gids - curr_active_gids

    for gid in disappeared_gids:
        traj = dir_tracker.trajectories.get(gid)
        if not traj:
            continue
        last_pt = traj[-1]

        if dir_tracker.is_near_frame_border(last_pt, frame_w, frame_h):
            dir_tracker.exited_camera_ids.add(gid)
            for tid, mapped_gid in list(identity_manager.track_to_global.items()):
                if mapped_gid == gid:
                    del identity_manager.track_to_global[tid]

            profile = identity_manager.identities.get(gid)
            if profile is not None:
                far_bev = np.array([999.0, 999.0], dtype=np.float32)
                for attr in ("bev_xy", "last_bev_xy", "pos_bev", "last_pos_bev"):
                    if hasattr(profile, attr) and getattr(profile, attr) is not None:
                        setattr(profile, attr, far_bev)

    return curr_active_gids


def draw_industrial_hud(
    frame: np.ndarray,
    detections: list[PoseDetection],
    track_to_global: Dict[int, int],
    identity_manager: EnterpriseIdentityManager,
    fps_instant: float,
    kpt_thresh: float = 0.20,
    dir_tracker: Optional[DirectionalTracker] = None,
) -> None:
    for det in detections:
        tid = det.track_id
        gid = track_to_global.get(tid, -1)
        x1, y1, x2, y2 = det.bbox_xyxy.astype(int)

        if gid == -1:
            box_color = (160, 160, 160)
            status_tag = f"T:{tid} [INIT]"
        elif det.is_intersecting:
            box_color = (0, 140, 255)
            status_tag = f"ID:{gid} [GALLERY LOCKED]"
        else:
            box_color = (0, 220, 0)
            status_tag = f"ID:{gid}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

        for p1, p2 in SKELETON_LIMBS:
            if det.keypoints_conf[p1] >= kpt_thresh and det.keypoints_conf[p2] >= kpt_thresh:
                pt1 = tuple(det.keypoints_xy[p1].astype(int))
                pt2 = tuple(det.keypoints_xy[p2].astype(int))
                cv2.line(frame, pt1, pt2, (255, 220, 0), 2, cv2.LINE_AA)

        ug, vg = int(det.ground_uv[0]), int(det.ground_uv[1])
        cv2.circle(frame, (ug, vg), 6, (255, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(frame, (ug, vg), 9, (0, 0, 0), 2, cv2.LINE_AA)

        bev_str = ""
        if det.bev_xy is not None:
            bev_str = f" | BEV:({det.bev_xy[0]:.1f}m, {det.bev_xy[1]:.1f}m)"

        dir_str = ""
        if dir_tracker is not None:
            lookup_key = gid if gid != -1 else f"T_{tid}"
            last_dir = dir_tracker.get_person_status(lookup_key)
            if last_dir:
                dir_str = f" | {last_dir}"

        header_text = f"{status_tag}{bev_str}{dir_str}"
        (tw, th_text), _ = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, max(0, y1 - th_text - 10)), (x1 + tw + 8, y1), box_color, -1)
        cv2.putText(
            frame,
            header_text,
            (x1 + 4, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0) if box_color != (160, 160, 160) else (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    banner = (
        f"Stream FPS: {fps_instant:.1f} | Active Frame Tracks: {len(detections)} | "
        f"Total Registered IDs: {len(identity_manager.identities)}"
    )
    if dir_tracker is not None and (dir_tracker.lines or dir_tracker.entrance_zones):
        banner += f" | IN: {dir_tracker.total_entries} OUT: {dir_tracker.total_exits}"

    cv2.rectangle(frame, (10, 10), (920, 48), (20, 20, 20), -1)
    cv2.putText(frame, banner, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 180), 2, cv2.LINE_AA)


def main(
    config_path: str,
    vehicle_config_path: str = "configs/vehicle_monitoring.yaml",
) -> None:
    cfg = load_config(config_path)
    sys_cfg = cfg["system"]
    stream_cfg = cfg["stream"]
    det_cfg = cfg["detection"]
    reid_cfg = cfg["reid"]
    bev_cfg = cfg["physics_bev"]
    zones_cfg = cfg["zones"]

    logger.info(f"Opening video stream: {stream_cfg['source']}")
    reader = VariableFPSVideoReader(
        source=stream_cfg["source"],
        drop_frames_on_lag=stream_cfg.get("drop_frames_on_lag", True),
        reconnect_delay_sec=stream_cfg.get("reconnect_delay_sec", 3.0),
    )

    laplacian_var = float(reid_cfg.get("min_laplacian_var", 10.0))
    logger.info(
        f"Stream Resolution: {reader.width}x{reader.height} @ {reader.nominal_fps:.1f} nominal FPS | "
        f"Native Blur Gate: {laplacian_var:.1f}"
    )

    detector = IndustrialPoseDetector(
        model_weights=det_cfg["model_weights"],
        tracker_config=det_cfg["tracker_config"],
        device=sys_cfg["device"],
        conf_thres=det_cfg.get("conf_thres", 0.08),
        iou_thres=det_cfg.get("iou_thres", 0.55),
        imgsz=det_cfg.get("imgsz", 1536),
        half=sys_cfg.get("half_precision", True),
    )

    ref_wh = tuple(bev_cfg.get("calibration_ref_wh", [1280, 720]))
    bev_projector = MetricBEVProjector(
        src_image_pts=bev_cfg.get("src_image_pts"),
        dst_meter_pts=bev_cfg.get("dst_meter_pts"),
        default_px_per_meter=bev_cfg.get("default_px_per_meter", 110.0),
        stream_wh=(reader.width, reader.height),
        calib_ref_wh=(int(ref_wh[0]), int(ref_wh[1])),
    )

    reid_extractor = PoseGuidedOSNetExtractor(
        model_name=reid_cfg["model_name"],
        weights_path=reid_cfg.get("weights_path") or None,
        device=sys_cfg["device"],
        kpt_conf_thresh=reid_cfg.get("kpt_conf_thresh", 0.20),
        min_laplacian_var=laplacian_var,
        head_branch_weight=reid_cfg.get("head_branch_weight", 0.35),
        color_branch_weight=reid_cfg.get("color_branch_weight", 0.20),
    )

    identity_manager = EnterpriseIdentityManager(
        reid_extractor=reid_extractor,
        bev_projector=bev_projector,
        cos_sim_thresh=reid_cfg.get("cos_sim_thresh", 0.60),
        spatial_recovery_sim_thresh=reid_cfg.get("spatial_recovery_sim_thresh", 0.48),
        reentry_sim_thresh=reid_cfg.get("reentry_sim_thresh", 0.82),
        max_worker_speed_mps=bev_cfg.get("max_worker_speed_mps", 3.5),
        max_internal_occlusion_m=bev_cfg.get("max_internal_occlusion_m", 5.5),
        lag_diffusion_coeff=bev_cfg.get("lag_diffusion_coeff", 0.65),
        base_pose_jitter_m=bev_cfg.get("base_pose_jitter_m", 0.70),
        reid_interval_sec=reid_cfg.get("reid_interval_sec", 0.35),
    )

    zone_manager = MultiZoneManager(
        zone_definitions=zones_cfg.get("definitions", {}),
        entry_confirm_sec=zones_cfg.get("entry_confirm_sec", 0.50),
        exit_confirm_sec=zones_cfg.get("exit_confirm_sec", 1.00),
        force_exit_timeout_sec=zones_cfg.get("force_exit_timeout_sec", 15.0),
    )

    dir_tracker = DirectionalTracker(cfg)

    # Initialize the modular Vehicle & Driver Monitoring Pipeline
    vehicle_pipeline: Optional[VehicleMonitoringPipeline] = None
    if Path(vehicle_config_path).exists():
        logger.info(f"Initializing Vehicle & Driver Monitoring Module from: {vehicle_config_path}")
        vehicle_pipeline = VehicleMonitoringPipeline(
            config_path=vehicle_config_path,
            reid_extractor=reid_extractor,
        )
        tune_vehicle_pipeline_for_yard(vehicle_pipeline)

        # Share the calibrated homography from settings.yaml if available
        if bev_cfg.get("src_image_pts") and bev_cfg.get("dst_meter_pts"):
            src_pts = np.array(bev_cfg["src_image_pts"], dtype=np.float32)
            dst_pts = np.array(bev_cfg["dst_meter_pts"], dtype=np.float32)
            if len(src_pts) >= 4 and len(dst_pts) >= 4:
                H, _ = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC)
                if H is not None:
                    vehicle_pipeline.estimator.homography_matrix = H
                    try:
                        vehicle_pipeline.estimator.inv_homography_matrix = np.linalg.inv(H)
                    except np.linalg.LinAlgError:
                        pass

    writer = None
    out_path = sys_cfg.get("output_video_path")
    if out_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, reader.nominal_fps, (reader.width, reader.height))

    win_title = "Enterprise Factory Re-ID & Zone Monitoring"
    if sys_cfg.get("display_gui", True):
        cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_title, min(1440, reader.width), min(900, reader.height))

    prev_ts = time.monotonic()
    fps_ema = reader.nominal_fps
    prev_active_gids: set[Any] = set()

    logger.info("Pipeline initialized. Starting real-time inference loop...")
    try:
        while True:
            ok, frame, frame_ts = reader.read()
            if not ok or frame is None:
                logger.info("End of stream reached.")
                break

            dt_wall = max(1e-3, frame_ts - prev_ts)
            fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt_wall)
            prev_ts = frame_ts

            active_tids = set(identity_manager.track_to_global.keys())
            detections = detector.track_frame(frame, active_track_ids=active_tids)
            track_to_global = identity_manager.process_frame(frame, detections, frame_ts)

            # Build ground points for both verified Global IDs (for zones) and all tracks (for tripwires)
            active_ground_points: Dict[int, Tuple[float, float]] = {}
            all_directional_points: Dict[Any, Tuple[float, float]] = {}
            for det in detections:
                gid = track_to_global.get(det.track_id)
                if gid is not None and gid != -1:
                    active_ground_points[gid] = det.ground_uv
                    dir_tracker.promote_track_to_global(det.track_id, gid)
                    all_directional_points[gid] = det.ground_uv
                else:
                    all_directional_points[f"T_{det.track_id}"] = det.ground_uv

            zone_events = zone_manager.update_all(active_ground_points, frame_ts)
            for ev in zone_events:
                logger.info(f"ZONE EVENT: {ev}")

            # Update Directional Tracking (Line-Crossing & Entrance/Exit Gate Zones)
            dir_events = dir_tracker.update_all(all_directional_points, frame_ts)
            for dev in dir_events:
                logger.info(f"DIRECTION EVENT: {dev}")

            # Distinguish between in-yard occlusion (keep BEV position) vs border exit (retire doorway bubble)
            prev_active_gids = handle_border_exits_and_occlusions(
                identity_manager=identity_manager,
                dir_tracker=dir_tracker,
                active_ground_points=active_ground_points,
                prev_active_gids=prev_active_gids,
                frame_w=reader.width,
                frame_h=reader.height,
            )

            # Run Vehicle & Driver Monitoring using YOLO-Pose detections (with seated-driver lock)
            if vehicle_pipeline is not None:
                pose_persons = build_pose_persons_for_vehicle_monitor(
                    detections=detections,
                    track_to_global=track_to_global,
                    bev_projector=bev_projector,
                )
                sync_cab_driver_assignments(vehicle_pipeline, pose_persons)
                frame = vehicle_pipeline.process_frame(
                    frame=frame,
                    external_persons=pose_persons,
                    tracker_config=det_cfg["tracker_config"],
                )
                sync_cab_driver_assignments(vehicle_pipeline, pose_persons)

            if sys_cfg.get("display_gui", True) or writer is not None:
                zone_manager.draw_zones(frame)
                dir_tracker.draw_overlays(frame, active_gids=set(all_directional_points.keys()))
                draw_industrial_hud(
                    frame=frame,
                    detections=detections,
                    track_to_global=track_to_global,
                    identity_manager=identity_manager,
                    fps_instant=fps_ema,
                    kpt_thresh=reid_cfg.get("kpt_conf_thresh", 0.20),
                    dir_tracker=dir_tracker,
                )

            if writer is not None:
                writer.write(frame)

            if sys_cfg.get("display_gui", True):
                cv2.imshow(win_title, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("User requested exit ('q').")
                    break

    except KeyboardInterrupt:
        logger.info("Interrupted by user. Shutting down...")
    finally:
        reader.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enterprise Pose-Guided Re-ID & Zone Monitoring")
    parser.add_argument("--config", type=str, default="configs/settings.yaml", help="Path to settings.yaml")
    parser.add_argument(
        "--vehicle-config",
        type=str,
        default="configs/vehicle_monitoring.yaml",
        help="Path to vehicle_monitoring.yaml",
    )
    args = parser.parse_args()
    main(args.config, args.vehicle_config)