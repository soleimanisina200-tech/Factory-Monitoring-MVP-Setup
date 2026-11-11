# main.py
from __future__ import annotations
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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
) -> List[Dict]:
    """
    Packages IndustrialPoseDetector + EnterpriseIdentityManager outputs
    (global ID, bbox, 17x3 YOLO-Pose keypoints, ankle ground_uv, and metric bev_xy)
    for the Vehicle & Driver Monitoring Pipeline.
    """
    pose_persons: List[Dict] = []
    for det in detections:
        gid = track_to_global.get(det.track_id, -1)
        # Use verified global_id when available, or fallback to temporary track ID
        person_id = str(gid) if gid != -1 else f"T{det.track_id}"

        # Combine (17, 2) xy and (17,) conf into (17, 3) array [x, y, conf]
        kpts_17x3 = np.column_stack([det.keypoints_xy, det.keypoints_conf])
        ug, vg = int(det.ground_uv[0]), int(det.ground_uv[1])

        pose_persons.append(
            {
                "id": person_id,
                "global_id": gid,
                "bbox": det.bbox_xyxy.astype(float).tolist(),
                "keypoints": kpts_17x3,
                "foot_px": (ug, vg),
                "bev_xy": det.bev_xy,
            }
        )
    return pose_persons


def draw_industrial_hud(
    frame: np.ndarray,
    detections: list[PoseDetection],
    track_to_global: Dict[int, int],
    identity_manager: EnterpriseIdentityManager,
    fps_instant: float,
    kpt_thresh: float = 0.30,
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

        header_text = f"{status_tag}{bev_str}"
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
    cv2.rectangle(frame, (10, 10), (760, 48), (20, 20, 20), -1)
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

    laplacian_var = float(reid_cfg.get("min_laplacian_var", 18.0))
    logger.info(
        f"Stream Resolution: {reader.width}x{reader.height} @ {reader.nominal_fps:.1f} nominal FPS | "
        f"Native Blur Gate: {laplacian_var:.1f}"
    )

    detector = IndustrialPoseDetector(
        model_weights=det_cfg["model_weights"],
        tracker_config=det_cfg["tracker_config"],
        device=sys_cfg["device"],
        conf_thres=det_cfg.get("conf_thres", 0.15),
        iou_thres=det_cfg.get("iou_thres", 0.55),
        imgsz=det_cfg.get("imgsz", 1280),
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
        kpt_conf_thresh=reid_cfg.get("kpt_conf_thresh", 0.30),
        min_laplacian_var=laplacian_var,
        head_branch_weight=reid_cfg.get("head_branch_weight", 0.35),
        color_branch_weight=reid_cfg.get("color_branch_weight", 0.20),
    )

    identity_manager = EnterpriseIdentityManager(
        reid_extractor=reid_extractor,
        bev_projector=bev_projector,
        cos_sim_thresh=reid_cfg.get("cos_sim_thresh", 0.64),
        spatial_recovery_sim_thresh=reid_cfg.get("spatial_recovery_sim_thresh", 0.56),
        reentry_sim_thresh=reid_cfg.get("reentry_sim_thresh", 0.78),
        max_worker_speed_mps=bev_cfg.get("max_worker_speed_mps", 2.6),
        max_internal_occlusion_m=bev_cfg.get("max_internal_occlusion_m", 3.5),
        lag_diffusion_coeff=bev_cfg.get("lag_diffusion_coeff", 0.50),
        base_pose_jitter_m=bev_cfg.get("base_pose_jitter_m", 0.55),
        reid_interval_sec=reid_cfg.get("reid_interval_sec", 0.35),
    )

    zone_manager = MultiZoneManager(
        zone_definitions=zones_cfg.get("definitions", {}),
        entry_confirm_sec=zones_cfg.get("entry_confirm_sec", 0.50),
        exit_confirm_sec=zones_cfg.get("exit_confirm_sec", 1.00),
        force_exit_timeout_sec=zones_cfg.get("force_exit_timeout_sec", 15.0),
    )

    # Initialize the modular Vehicle & Driver Monitoring Pipeline
    vehicle_pipeline: Optional[VehicleMonitoringPipeline] = None
    if Path(vehicle_config_path).exists():
        logger.info(f"Initializing Vehicle & Driver Monitoring Module from: {vehicle_config_path}")
        vehicle_pipeline = VehicleMonitoringPipeline(
            config_path=vehicle_config_path,
            reid_extractor=reid_extractor,
        )
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

            active_ground_points: Dict[int, Tuple[float, float]] = {}
            for det in detections:
                gid = track_to_global.get(det.track_id)
                if gid is not None:
                    active_ground_points[gid] = det.ground_uv

            zone_events = zone_manager.update_all(active_ground_points, frame_ts)
            for ev in zone_events:
                logger.info(f"ZONE EVENT: {ev}")

            # Run Vehicle & Driver Monitoring using YOLO-Pose detections
            if vehicle_pipeline is not None:
                pose_persons = build_pose_persons_for_vehicle_monitor(
                    detections=detections,
                    track_to_global=track_to_global,
                )
                frame = vehicle_pipeline.process_frame(
                    frame=frame,
                    external_persons=pose_persons,
                    tracker_config=det_cfg["tracker_config"],
                )

            if sys_cfg.get("display_gui", True) or writer is not None:
                zone_manager.draw_zones(frame)
                draw_industrial_hud(
                    frame=frame,
                    detections=detections,
                    track_to_global=track_to_global,
                    identity_manager=identity_manager,
                    fps_instant=fps_ema,
                    kpt_thresh=reid_cfg.get("kpt_conf_thresh", 0.30),
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