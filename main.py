from __future__ import annotations
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np
import yaml

import torch

from app.detection.detector import IndustrialPoseDetector, PoseDetection
from app.identity.manager import EnterpriseIdentityManager
from app.identity.reid_model import PoseGuidedOSNetExtractor, SKELETON_LIMBS
from app.logic.calibration import DEFAULT_CALIBRATION_PATH, load_calibration
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


def resolve_device(requested: str, half: bool) -> Tuple[str, bool]:
    """
    Fail-soft device selection with an explicit message. A CPU-only PyTorch build (the default
    `pip install torch` on Windows) used to crash deep inside YOLO with "Torch not compiled with CUDA".
    """
    if "cuda" in str(requested) and not torch.cuda.is_available():
        build = f"torch {torch.__version__} (CUDA build: {torch.version.cuda})"
        logger.error(
            "CUDA requested (%s) but not available in %s. Falling back to CPU: expect ~1 FPS. "
            "Install the CUDA build of PyTorch from https://pytorch.org/get-started/locally/ "
            "and check `python -c \"import torch; print(torch.cuda.is_available())\"` prints True.",
            requested, build,
        )
        return "cpu", False
    if "cuda" in str(requested):
        logger.info("Using %s: %s", requested, torch.cuda.get_device_name(torch.device(requested)))
    return str(requested), half


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
    track_embeddings: Optional[Dict[int, np.ndarray]] = None,
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
            if bev_projector is not None:
                try:
                    bev_xy = bev_projector.to_bev_meters(float(ug), float(vg))
                except Exception:
                    logger.exception("BEV projection of seated operator failed")

        kpts_17x3 = np.column_stack([kpts_xy, kpts_conf])

        pose_persons.append(
            {
                "id": person_id,
                "global_id": gid,
                "bbox": det.bbox_xyxy.astype(float).tolist(),
                "keypoints": kpts_17x3,
                "foot_px": (ug, vg),
                "bev_xy": bev_xy,
                # Real OSNet embedding from the identity manager (same feature space as global IDs).
                "embedding": (track_embeddings or {}).get(det.track_id),
            }
        )
    return pose_persons


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
    source_override: Optional[str] = None,
    calibration_path: str = DEFAULT_CALIBRATION_PATH,
) -> None:
    cfg = load_config(config_path)
    if source_override:
        cfg["stream"]["source"] = source_override
    sys_cfg = cfg["system"]
    sys_cfg["device"], sys_cfg["half_precision"] = resolve_device(
        sys_cfg.get("device", "cuda:0"), bool(sys_cfg.get("half_precision", True))
    )
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

    # Ground calibration: configs/calibration.yaml (written by scripts/calibrate_ground_plane.py) wins;
    # the physics_bev points in settings.yaml are only a fallback. ONE homography serves persons and vehicles.
    calibration = load_calibration(calibration_path)
    if calibration is not None:
        logger.info(f"Ground calibration loaded from {calibration_path} (clicked at {calibration.image_wh})")
        src_pts, dst_pts, ref_wh = calibration.image_points_px, calibration.world_points_m, calibration.image_wh
    else:
        logger.warning(
            f"No {calibration_path}: using the physics_bev points of settings.yaml. Metric distances are only "
            "as good as that calibration - run scripts/calibrate_ground_plane.py on this camera."
        )
        src_pts, dst_pts = bev_cfg.get("src_image_pts"), bev_cfg.get("dst_meter_pts")
        ref_wh = tuple(bev_cfg.get("calibration_ref_wh", [reader.width, reader.height]))
    bev_projector = MetricBEVProjector(
        src_image_pts=src_pts,
        dst_meter_pts=dst_pts,
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
        min_gallery_height_px=reid_cfg.get("min_gallery_height_px", 72),
        min_visible_kpts=reid_cfg.get("min_visible_kpts", 5),
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
        stitch_sim_thresh=reid_cfg.get("stitch_sim_thresh", 0.55),
        stitch_margin=reid_cfg.get("stitch_margin", 0.04),
        reentry_min_gap_sec=reid_cfg.get("reentry_min_gap_sec", 2.0),
        reentry_margin=reid_cfg.get("reentry_margin", 0.08),
        reentry_min_samples=reid_cfg.get("reentry_min_samples", 3),
        occlusion_max_sec=reid_cfg.get("occlusion_max_sec", 20.0),
        border_margin_ratio=reid_cfg.get("border_margin_ratio", 0.06),
        track_gap_reset_sec=reid_cfg.get("track_gap_reset_sec", 1.5),
        exit_edges=reid_cfg.get("exit_edges", ["left", "right", "top", "bottom"]),
        exit_confirm_sec=reid_cfg.get("exit_confirm_sec", 0.8),
        mismatch_min_span_sec=reid_cfg.get("mismatch_min_span_sec", 1.0),
        anchor_min_height_px=reid_cfg.get("anchor_min_height_px", 110),
        track_state_ttl_sec=reid_cfg.get("track_state_ttl_sec", 10.0),
        identity_retention_sec=reid_cfg.get("identity_retention_sec", 3600.0),
        max_exited_identities=reid_cfg.get("max_exited_identities", 300),
        pool_max_samples=reid_cfg.get("pool_max_samples", 8),
        pool_window_sec=reid_cfg.get("pool_window_sec", 4.0),
        decision_log_path=reid_cfg.get("decision_log_path") or None,
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

        # ONE homography for persons and vehicles: the projector's matrix is already scaled to the stream
        # resolution. (The old code rebuilt it from the unscaled reference points -> two coordinate systems.)
        if bev_projector.H is not None:
            vehicle_pipeline.estimator.set_homography(bev_projector.H)

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

    logger.info("Pipeline initialized. Starting real-time inference loop...")
    try:
        while True:
            ok, frame, frame_ts = reader.read()
            if not ok or frame is None:
                logger.info("End of stream reached.")
                break

            now_wall = time.monotonic()
            dt_wall = max(1e-3, now_wall - prev_ts)  # real processing rate, not the video's nominal FPS
            fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt_wall)
            prev_ts = now_wall

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

            # Workers who just LEFT the camera (reported once by the identity manager, which also unbinds
            # their tracks so a newcomer can never inherit the departed worker's ID).
            # Identities merged by the self-healing consolidation: move zone/direction state to the survivor
            # so the same worker is not counted twice.
            for old_gid, new_gid in identity_manager.pop_merged():
                zone_manager.rename_identity(old_gid, new_gid)
                dir_tracker.rename_entity(old_gid, new_gid)

            for gid_left in identity_manager.pop_newly_exited():
                dir_tracker.exited_camera_ids.add(gid_left)
                logger.info(f"ID:{gid_left} left the camera view")

            # Run Vehicle & Driver Monitoring using YOLO-Pose detections (with seated-driver lock)
            if vehicle_pipeline is not None:
                track_embeddings = {
                    d.track_id: identity_manager.track_best_emb[d.track_id][0]
                    for d in detections
                    if d.track_id in identity_manager.track_best_emb
                }
                pose_persons = build_pose_persons_for_vehicle_monitor(
                    detections=detections,
                    track_to_global=track_to_global,
                    bev_projector=bev_projector,
                    track_embeddings=track_embeddings,
                )
                frame = vehicle_pipeline.process_frame(
                    frame=frame,
                    external_persons=pose_persons,
                    frame_ts=frame_ts,
                )

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
        identity_manager.close()
        if vehicle_pipeline is not None:
            vehicle_pipeline.logger.close()
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
    parser.add_argument("--source", type=str, default=None, help="Video file / RTSP URL (overrides settings.yaml)")
    parser.add_argument("--calibration", type=str, default=DEFAULT_CALIBRATION_PATH,
                        help="Ground calibration file written by scripts/calibrate_ground_plane.py")
    args = parser.parse_args()
    main(args.config, args.vehicle_config, args.source, args.calibration)