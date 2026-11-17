import os
import csv
import time
import yaml
import logging
import cv2
import numpy as np

from app.streams.video_reader import VideoReader
from app.detection.detector import PersonDetector
from app.identity.reid_model import PoseGuidedOSNetExtractor
from app.identity.manager import EnterpriseIdentityManager
from app.zones.manager import ZoneManager
from app.logic.state_machine import ZoneStateMachine
from app.logic.geometry import (
    get_foot_point,
    is_occluded_by_others,
    is_touching_border,
    HomographyTransformer,
    draw_skeleton,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("FactoryMonitor")


def load_config(config_path: str = "configs/settings.yaml") -> dict:
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def append_event_to_csv(csv_path: str, row_data: dict):
    file_exists = os.path.exists(csv_path)
    fieldnames = ["timestamp", "event", "person_id", "zone", "duration_sec"]
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": f"{row_data.get('timestamp', 0.0):.3f}",
                "event": row_data.get("event", ""),
                "person_id": row_data.get("person_id", ""),
                "zone": row_data.get("zone", ""),
                "duration_sec": f"{row_data.get('duration', 0.0):.2f}",
            }
        )


def main():
    cfg = load_config("configs/settings.yaml")

    stream_cfg = cfg.get("stream", {})
    det_cfg = cfg.get("detection", {})
    reid_cfg = cfg.get("reid", {})
    geom_cfg = cfg.get("geometry", {})
    sm_cfg = cfg.get("state_machine", {})
    zones_cfg = cfg.get("zones", {})
    out_cfg = cfg.get("output", {})

    reader = VideoReader(
        source=stream_cfg.get("source", 0),
        width=stream_cfg.get("width", 1280),
        height=stream_cfg.get("height", 720),
        reconnect_delay=stream_cfg.get("reconnect_delay", 3.0),
    )

    detector = PersonDetector(
        model_path=det_cfg.get("model_path", "yolo11m-pose.pt"),
        conf_threshold=det_cfg.get("conf_threshold", 0.35),
        iou_threshold=det_cfg.get("iou_threshold", 0.50),
        tracker_config=det_cfg.get("tracker_config", "configs/custom_bytetrack.yaml"),
        device=det_cfg.get("device", "cuda"),
    )

    reid_extractor = PoseGuidedOSNetExtractor(
        model_name=reid_cfg.get("model_name", "osnet_ain_x1_0"),
        device=reid_cfg.get("device", "cuda"),
        part_weights=reid_cfg.get("part_weights", {"head": 0.55, "shoes": 0.30, "torso": 0.15}),
        min_kp_conf=geom_cfg.get("min_keypoint_conf", 0.45),
    )

    border_margin_px = geom_cfg.get("border_margin_px", 28)
    identity_manager = EnterpriseIdentityManager(
        similarity_threshold=reid_cfg.get("similarity_threshold", 0.68),
        margin_threshold=reid_cfg.get("margin_threshold", 0.06),
        reverify_sim_floor=reid_cfg.get("reverify_sim_floor", 0.52),
        max_gallery_size=reid_cfg.get("max_gallery_size", 8),
        probation_frames=reid_cfg.get("probation_frames", 4),
        track_ttl_sec=reid_cfg.get("track_ttl_sec", 180.0),
        border_cooldown_sec=reid_cfg.get("border_cooldown_sec", 3.5),
        max_speed_px_per_sec=reid_cfg.get("max_speed_px_per_sec", 450.0),
        uniform_whitening_alpha=reid_cfg.get("uniform_whitening_alpha", 0.55),
        border_margin_px=border_margin_px,
    )

    zone_manager = ZoneManager(zones_cfg)
    homography = HomographyTransformer(
        src_points=geom_cfg.get("homography_src"),
        dst_points=geom_cfg.get("homography_dst"),
    )
    state_machine = ZoneStateMachine(
        entry_frames=sm_cfg.get("entry_frames", 8),
        exit_frames=sm_cfg.get("exit_frames", 12),
        timeout_sec=sm_cfg.get("timeout_sec", 4.0),
        entry_confirm_sec=sm_cfg.get("entry_confirm_sec", 0.32),
        exit_confirm_sec=sm_cfg.get("exit_confirm_sec", 0.48),
    )

    csv_path = out_cfg.get("events_csv_path", "zone_events_log.csv")
    save_video = out_cfg.get("save_video", False)
    show_display = out_cfg.get("show_display", True)
    reid_interval = reid_cfg.get("reid_interval_frames", 10)
    ioa_threshold = geom_cfg.get("occlusion_ioa_threshold", 0.18)

    video_writer = None
    if save_video:
        out_path = out_cfg.get("output_video_path", "output_monitored.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(out_path, fourcc, reader.fps, (reader.frame_w, reader.frame_h))

    last_processed_frame_id = -1
    fps_ema = reader.fps

    logger.info("Factory Monitoring MVP with Skeleton Pose & Forklift support started.")

    try:
        while True:
            t_start = time.perf_counter()
            ret, frame, frame_id, frame_ts = reader.read_with_meta()
            if not ret or frame is None:
                break

            if frame_id == last_processed_frame_id:
                continue
            last_processed_frame_id = frame_id

            frame_h, frame_w = frame.shape[:2]

            raw_detections = detector.track(frame)
            person_detections = [d for d in raw_detections if d.get("is_person", True)]
            vehicle_detections = [d for d in raw_detections if not d.get("is_person", True)]
            all_person_boxes = [d["box"] for d in person_detections]

            occlusion_flags = []
            indices_to_extract = []
            boxes_to_extract = []
            kpts_to_extract = []

            for i, det in enumerate(person_detections):
                tid = det["track_id"]
                box = det["box"]
                kpts = det.get("keypoints")

                is_occ = is_occluded_by_others(
                    target_box=box, all_boxes=all_person_boxes, ioa_threshold=ioa_threshold, keypoints=kpts
                )
                occlusion_flags.append(is_occ)

                on_border = is_touching_border(box, frame_w, frame_h, border_margin_px)
                if identity_manager.needs_feature_extraction(tid, is_occ, on_border, reid_interval):
                    indices_to_extract.append(i)
                    boxes_to_extract.append(box)
                    kpts_to_extract.append(kpts)

            extracted_features = [None] * len(person_detections)
            if boxes_to_extract:
                batch_feats = reid_extractor.extract_batch(
                    frame_bgr=frame, boxes=boxes_to_extract, keypoints_list=kpts_to_extract, require_complete_skeleton=False
                )
                for sub_idx, det_idx in enumerate(indices_to_extract):
                    extracted_features[det_idx] = batch_feats[sub_idx]

            id_results = identity_manager.process_frame(
                detections=person_detections,
                extracted_features=extracted_features,
                occlusion_flags=occlusion_flags,
                frame_w=frame_w,
                frame_h=frame_h,
                timestamp=frame_ts,
            )

            zone_occupants = {z: 0 for z in zone_manager.zones.keys()}
            render_items = []

            for i, det in enumerate(person_detections):
                tid = det["track_id"]
                x1, y1, x2, y2 = det["box"]
                kpts = det.get("keypoints")
                is_occ = occlusion_flags[i]

                id_info = id_results[tid]
                person_id = id_info["person_id"]
                orientation = id_info["orientation"]
                is_confirmed = id_info["is_confirmed"]
                on_border = id_info["on_border"]

                expected_ar = identity_manager.get_track_expected_aspect_ratio(tid)
                foot_px = get_foot_point(x1, y1, x2, y2, keypoints=kpts, expected_aspect_ratio=expected_ar)
                query_pt = homography.transform_point(foot_px) if homography.matrix is not None else foot_px
                momentary_zone = zone_manager.get_zone((int(query_pt[0]), int(query_pt[1])))

                status = "OUTSIDE"
                active_dwell = 0.0

                if is_confirmed:
                    event, event_data, status, active_dwell = state_machine.update(
                        person_id=person_id, momentary_zone=momentary_zone, timestamp=frame_ts
                    )
                    if event and event_data:
                        append_event_to_csv(csv_path, {
                            "timestamp": event_data["timestamp"],
                            "event": event,
                            "person_id": person_id,
                            "zone": event_data["zone"],
                            "duration": event_data.get("duration", 0.0),
                        })

                    if status == "INSIDE" and state_machine.states[person_id]["zone"] in zone_occupants:
                        z_active = state_machine.states[person_id]["zone"]
                        zone_occupants[z_active] += 1

                # محاسبه تخمین فاصله متریک فرضی (DEV) برای نمایش مشابه نسخه قبلی
                approx_dist = max(1.2, float(frame_h - foot_px[1]) * 0.012)

                render_items.append({
                    "box": (x1, y1, x2, y2),
                    "foot_px": foot_px,
                    "keypoints": kpts,
                    "person_id": person_id,
                    "orientation": orientation,
                    "is_confirmed": is_confirmed,
                    "is_occ": is_occ,
                    "on_border": on_border,
                    "status": status,
                    "dwell": active_dwell,
                    "dist": approx_dist,
                    "is_vehicle": False,
                })

            # رندر وسایل نقلیه و لیفتراک (باکس صورتی رنگ + برچسب DEV مشابه تصویر قدیمی)[cite: 3]
            for v_det in vehicle_detections:
                vx1, vy1, vx2, vy2 = v_det["box"]
                v_tid = v_det["track_id"]
                v_dist = max(2.5, float(frame_h - vy2) * 0.015)
                render_items.append({
                    "box": (vx1, vy1, vx2, vy2),
                    "foot_px": (int((vx1 + vx2) / 2), vy2),
                    "keypoints": None,
                    "person_id": f"Forklift_{v_tid}",
                    "orientation": "VEHICLE",
                    "is_confirmed": True,
                    "is_occ": False,
                    "on_border": False,
                    "status": "VEHICLE",
                    "dwell": 0.0,
                    "dist": v_dist,
                    "is_vehicle": True,
                })

            forced_exits = state_machine.process_missing_workers(frame_ts)
            for fe in forced_exits:
                append_event_to_csv(csv_path, fe)

            zone_manager.draw_zones(frame, zone_occupants=zone_occupants)

            for item in render_items:
                x1, y1, x2, y2 = item["box"]
                fx, fy = item["foot_px"]
                kpts = item["keypoints"]
                pid = item["person_id"]
                orient = item["orientation"]

                if item["is_vehicle"]:
                    box_color = (255, 0, 255)  # باکس صورتی مشخصه لیفتراک
                elif not item["is_confirmed"]:
                    box_color = (160, 160, 160)
                elif item["status"] == "INSIDE":
                    box_color = (0, 230, 118)
                elif item["is_occ"]:
                    box_color = (0, 140, 255)
                else:
                    box_color = (0, 255, 120)  # سبز برای باکس کارگران (مشابه نسخه قبلی)[cite: 3]

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                # رسم خطوط اسکلت روی بدن انسان (مشابه نسخه قبلی)[cite: 3]
                if not item["is_vehicle"] and kpts is not None:
                    draw_skeleton(frame, kpts)

                if not item["is_vehicle"]:
                    cv2.circle(frame, (fx, fy), 5, (0, 0, 255), -1)

                if item["is_vehicle"]:
                    header_text = f"{pid} | DEV: ({item['dist']:.1f}m)"
                else:
                    header_text = f"{pid} | DEV: ({item['dist']:.1f}m) [{orient}]"
                    if item["status"] == "INSIDE":
                        header_text += f" | {item['dwell']:.1f}s"
                    if item["is_occ"]:
                        header_text += " (OCC)"

                (tw, th), _ = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                ty = max(th + 6, y1 - 6)
                cv2.rectangle(frame, (x1, ty - th - 6), (x1 + tw + 8, ty + 4), box_color, -1)
                cv2.putText(
                    frame,
                    header_text,
                    (x1 + 4, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (15, 15, 15),
                    2,
                    cv2.LINE_AA,
                )

            dt_loop = max(1e-4, time.perf_counter() - t_start)
            fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt_loop)
            cv2.putText(
                frame,
                f"FPS: {fps_ema:.1f} | Workers: {len(identity_manager.workers)} | Vehicles: {len(vehicle_detections)}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (50, 255, 50),
                2,
                cv2.LINE_AA,
            )

            if video_writer is not None:
                video_writer.write(frame)

            if show_display:
                cv2.imshow("Factory Monitoring - Full Restored Version", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        reader.stop()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()