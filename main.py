# main.py
from __future__ import annotations
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Tuple
import cv2
import numpy as np
import yaml

from app.detection.detector import IndustrialPoseDetector, PoseDetection
from app.identity.manager import EnterpriseIdentityManager
from app.identity.reid_model import PoseGuidedOSNetExtractor, SKELETON_LIMBS
from app.logic.geometry import MetricBEVProjector
from app.streams.video_reader import VariableFPSVideoReader
from app.zones.manager import MultiZoneManager


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


def draw_industrial_hud(
    frame: np.ndarray,
    detections: list[PoseDetection],
    track_to_global: Dict[int, int],
    identity_manager: EnterpriseIdentityManager,
    fps_instant: float,
    kpt_thresh: float = 0.45,
) -> None:
    """
    Renders skeletal pose, extrapolated ground contact point, BEV metric coordinates,
    and intersection gallery-lock indicators onto the frame.
    """
    for det in detections:
        tid = det.track_id
        gid = track_to_global.get(tid, -1)
        x1, y1, x2, y2 = det.bbox_xyxy.astype(int)

        # Color coding:
        #   Green  = Verified Global ID, Gallery Active
        #   Orange = Verified Global ID, Gallery LOCKED due to Intersection
        #   Gray   = Unverified / Pending Quality Gate
        if gid == -1:
            box_color = (160, 160, 160)
            status_tag = f"T:{tid} [PENDING GATE]"
        elif det.is_intersecting:
            box_color = (0, 140, 255)  # Orange warning: Gallery locked
            status_tag = f"ID:{gid} [GALLERY LOCKED]"
        else:
            box_color = (0, 220, 0)
            status_tag = f"ID:{gid}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

        # Draw visible skeleton limbs
        for p1, p2 in SKELETON_LIMBS:
            if det.keypoints_conf[p1] >= kpt_thresh and det.keypoints_conf[p2] >= kpt_thresh:
                pt1 = tuple(det.keypoints_xy[p1].astype(int))
                pt2 = tuple(det.keypoints_xy[p2].astype(int))
                cv2.line(frame, pt1, pt2, (255, 220, 0), 2, cv2.LINE_AA)

        # Draw True Ground Contact Point (solves machinery leg occlusion)
        ug, vg = int(det.ground_uv[0]), int(det.ground_uv[1])
        cv2.circle(frame, (ug, vg), 6, (255, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(frame, (ug, vg), 9, (0, 0, 0), 2, cv2.LINE_AA)

        # Render BEV Metric Coordinates & Gallery size
        bev_str = ""
        if det.bev_xy is not None:
            bev_str = f" | BEV:({det.bev_xy[0]:.1f}m, {det.bev_xy[1]:.1f}m)"

        header_text = f"{status_tag}{bev_str}"
        (tw, th_text), _ = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        cv2.rectangle(frame, (x1, max(0, y1 - th_text - 10)), (x1 + tw + 8, y1), box_color, -1)
        cv2.putText(
            frame,
            header_text,
            (x1 + 4, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 0, 0) if box_color != (160, 160, 160) else (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # Top-left telemetry banner
    banner = f"Stream FPS: {fps_instant:.1f} | Active Identities: {len(identity_manager.identities)}"
    cv2.rectangle(frame, (10, 10), (480, 46), (20, 20, 20), -1)
    cv2.putText(frame, banner, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 180), 2, cv2.LINE_AA)


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    sys_cfg = cfg["system"]
    stream_cfg = cfg["stream"]
    det_cfg = cfg["detection"]
    reid_cfg = cfg["reid"]
    bev_cfg = cfg["physics_bev"]
    zones_cfg = cfg["zones"]

    # 1. Initialize Variable-FPS Stream Reader
    logger.info(f"Opening video stream: {stream_cfg['source']}")
    reader = VariableFPSVideoReader(
        source=stream_cfg["source"],
        drop_frames_on_lag=stream_cfg.get("drop_frames_on_lag", True),
        reconnect_delay_sec=stream_cfg.get("reconnect_delay_sec", 3.0),
    )

    # Dynamic Resolution Auto-Scaling for Laplacian Sharpness Threshold
    res_scale = (reader.width * reader.height) / (1280.0 * 720.0)
    scaled_laplacian_var = max(20.0, reid_cfg["min_laplacian_var"] * np.sqrt(res_scale))
    logger.info(
        f"Stream Resolution: {reader.width}x{reader.height} @ {reader.nominal_fps:.1f} nominal FPS | "
        f"Scaled Blur Gate: {scaled_laplacian_var:.1f}"
    )

    # 2. Initialize Single-Pass YOLO11-Pose + ByteTrack Detector
    detector = IndustrialPoseDetector(
        model_weights=det_cfg["model_weights"],
        tracker_config=det_cfg["tracker_config"],
        device=sys_cfg["device"],
        conf_thres=det_cfg["conf_thres"],
        iou_thres=det_cfg["iou_thres"],
        imgsz=det_cfg["imgsz"],
        half=sys_cfg.get("half_precision", True),
    )

    # 3. Initialize Homography BEV Metric Projector
    bev_projector = MetricBEVProjector(
        src_image_pts=bev_cfg.get("src_image_pts"),
        dst_meter_pts=bev_cfg.get("dst_meter_pts"),
        default_px_per_meter=bev_cfg.get("default_px_per_meter", 110.0),
    )

    # 4. Initialize Pose-Guided OSNet Re-ID Engine
    reid_extractor = PoseGuidedOSNetExtractor(
        model_name=reid_cfg["model_name"],
        weights_path=reid_cfg.get("weights_path") or None,
        device=sys_cfg["device"],
        kpt_conf_thresh=reid_cfg["kpt_conf_thresh"],
        min_laplacian_var=scaled_laplacian_var,
        head_branch_weight=reid_cfg["head_branch_weight"],
    )

    # 5. Initialize Enterprise Identity Manager
    identity_manager = EnterpriseIdentityManager(
        reid_extractor=reid_extractor,
        bev_projector=bev_projector,
        cos_sim_thresh=reid_cfg["cos_sim_thresh"],
        spatial_recovery_sim_thresh=reid_cfg["spatial_recovery_sim_thresh"],
        max_worker_speed_mps=bev_cfg["max_worker_speed_mps"],
        lag_diffusion_coeff=bev_cfg["lag_diffusion_coeff"],
        base_pose_jitter_m=bev_cfg["base_pose_jitter_m"],
        reid_interval_sec=reid_cfg["reid_interval_sec"],
    )

    # 6. Initialize Time-Based Multi-Zone Manager
    zone_manager = MultiZoneManager(
        zone_definitions=zones_cfg.get("definitions", {}),
        entry_confirm_sec=zones_cfg.get("entry_confirm_sec", 0.60),
        exit_confirm_sec=zones_cfg.get("exit_confirm_sec", 1.20),
        force_exit_timeout_sec=zones_cfg.get("force_exit_timeout_sec", 15.0),
    )

    # Optional Video Writer
    writer = None
    out_path = sys_cfg.get("output_video_path")
    if out_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, reader.nominal_fps, (reader.width, reader.height))

    prev_ts = time.monotonic()
    fps_ema = reader.nominal_fps

    logger.info("Pipeline initialized. Starting real-time inference loop...")
    try:
        while True:
            ok, frame, frame_ts = reader.read()
            if not ok or frame is None:
                logger.info("End of stream reached.")
                break

            # Track instantaneous FPS for telemetry
            dt_wall = max(1e-3, frame_ts - prev_ts)
            fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt_wall)
            prev_ts = frame_ts

            # Step A: YOLO11-Pose + ByteTrack Forward Pass
            detections = detector.track_frame(frame)

            # Step B: Pose-Guided Re-ID, BEV Physics Guard & Anti-Cloning Assignment
            track_to_global = identity_manager.process_frame(frame, detections, frame_ts)

            # Step C: Map verified Global IDs to their Pose-Extrapolated Ground Contact Points for Zone Monitoring
            active_ground_points: Dict[int, Tuple[float, float]] = {}
            for det in detections:
                gid = track_to_global.get(det.track_id)
                if gid is not None:
                    active_ground_points[gid] = det.ground_uv

            # Step D: Update Time-Based Zone State Machines
            zone_events = zone_manager.update_all(active_ground_points, frame_ts)
            for ev in zone_events:
                logger.info(f"ZONE EVENT: {ev}")

            # Step E: Render Zones & Diagnostic HUD
            if sys_cfg.get("display_gui", True) or writer is not None:
                zone_manager.draw_zones(frame)
                draw_industrial_hud(
                    frame=frame,
                    detections=detections,
                    track_to_global=track_to_global,
                    identity_manager=identity_manager,
                    fps_instant=fps_ema,
                    kpt_thresh=reid_cfg["kpt_conf_thresh"],
                )

            if writer is not None:
                writer.write(frame)

            if sys_cfg.get("display_gui", True):
                cv2.imshow("Enterprise Factory Re-ID & Zone Monitoring", frame)
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
    args = parser.parse_args()
    main(args.config)