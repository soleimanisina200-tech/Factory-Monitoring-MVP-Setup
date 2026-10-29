import logging
import cv2
import os
import time
import yaml
import csv
from datetime import datetime

from app.streams.video_reader import LocalVideoReader
from app.detection.detector import PersonDetector
from app.zones.manager import ZoneManager
from app.logic.geometry import get_foot_point
from app.logic.state_machine import ZoneStateMachine
from app.identity.reid_model import AdvancedReIDExtractor
from app.identity.manager import SpatioTemporalIdentityManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_settings(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

class DynamicCameraConfig:
    """
    موتور کالیبراسیون پویا: تبدیل پارامترهای مطلق به نسبی بر اساس مشخصات هر دوربین
    """
    @staticmethod
    def calibrate(fps: float, frame_width: int, frame_height: int):
        actual_fps = fps if fps > 0 else 30.0 
        
        # زمان‌بندی عکس‌برداری (هر 1.5 ثانیه یک بار)
        reid_interval_frames = max(1, int(actual_fps * 1.5))
        
        # زمان‌بندی ماشین حالت برای ورود و خروج
        entry_frames = max(1, int(actual_fps * 1.5))
        exit_frames = max(1, int(actual_fps * 2.0))
        
        # حاشیه امن برای کیفیت عکس (2% عرض تصویر)
        margin = max(5, int(frame_width * 0.02))
        
        # حداقل سایز کراپ برای OSNet
        min_crop_h = int(frame_height * 0.07)
        min_crop_w = int(frame_width * 0.02)
        
        # نگهبان فیزیک: سرعت مجاز بر اساس عرض تصویر (35% در ثانیه)
        max_speed_px_sec = int(frame_width * 0.35)

        logger.info(f"Camera Calibrated -> FPS: {actual_fps:.1f} | Res: {frame_width}x{frame_height}")
        logger.info(f"Dyn Config -> ReID Interval: {reid_interval_frames} frames | Speed Limit: {max_speed_px_sec} px/s")

        return {
            'reid_interval_frames': reid_interval_frames,
            'entry_frames': entry_frames,
            'exit_frames': exit_frames,
            'margin': margin,
            'min_crop_h': min_crop_h,
            'min_crop_w': min_crop_w,
            'max_speed_px_sec': max_speed_px_sec
        }

def is_safe_to_extract(x1, y1, x2, y2, all_boxes, frame_w, frame_h, dynamic_margin):
    # چک کردن حاشیه کادر دوربین
    if not (dynamic_margin < x1 and dynamic_margin < y1 and x2 < frame_w - dynamic_margin and y2 < frame_h - dynamic_margin): 
        return False
        
    # چک کردن همپوشانی (Occlusion) با کارگران دیگر
    for other_box in all_boxes:
        if other_box == [x1, y1, x2, y2]: continue
        ox1, oy1, ox2, oy2 = other_box
        if max(0, min(x2, ox2) - max(x1, ox1)) * max(0, min(y2, oy2) - max(y1, oy1)) > 0: 
            return False
            
    return True

def main():
    logger.info("Starting Enterprise Architecture: Dynamic Pipeline + OSNet + Physics Guard")

    video_path = "data/input/test_video.mp4"
    config_path = "configs/settings.yaml"

    if not os.path.exists(video_path): 
        logger.error(f"Video file not found at {video_path}")
        return

    reader = LocalVideoReader(video_path)
    frame_delay = max(1, int(1000 / reader.fps)) if reader.fps > 0 else 30
    
    # کالیبراسیون پویا بر اساس متادیتا دوربین
    cam_config = DynamicCameraConfig.calibrate(reader.fps, reader.width, reader.height)

    os.makedirs("data/output", exist_ok=True)
    writer = cv2.VideoWriter("data/output/processed_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), reader.fps, (reader.width, reader.height))

    # ایجاد فایل CSV برای دمو
    csv_file_path = os.path.join("data/output", f"Factory_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    csv_file = open(csv_file_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Time', 'Worker_ID', 'Event_Type', 'Zone', 'Duration_Sec'])

    # لود کردن ماژول‌ها
    detector = PersonDetector(model_path="yolo11m.pt", conf_threshold=0.45)
    zone_manager = ZoneManager(config_path)
    reid_extractor = AdvancedReIDExtractor(device='cuda')
    
    # تزریق سرعت کالیبره شده به منیجر
    identity_manager = SpatioTemporalIdentityManager(
        similarity_threshold=0.72, 
        max_speed_px_sec=cam_config['max_speed_px_sec'], 
        max_gallery_size=8
    )

    # تزریق فریم‌های ورود و خروج داینامیک به ماشین حالت
    state_machine = ZoneStateMachine(
        entry_frames=cam_config['entry_frames'],
        exit_frames=cam_config['exit_frames'],
        timeout_sec=4.0
    )

    recent_events = []
    REID_INTERVAL = cam_config['reid_interval_frames']
    last_reid_frame = {}

    for frame, timestamp, frame_id in reader.read_frames():
        start_time = time.time()
        
        # قفل ضد کلون: ریست کردن حافظه برای فریم جدید
        identity_manager.start_new_frame(timestamp)
        
        frame_h, frame_w = frame.shape[:2]

        result = detector.track(frame)
        inference_time_ms = result.speed['inference']
        zone_manager.draw_zones(frame)

        if result.boxes is not None and len(result.boxes) > 0:
            all_boxes = [list(map(int, b.xyxy[0])) for b in result.boxes]
            all_confs = [float(b.conf[0]) for b in result.boxes]
            all_tids = [int(b.id[0]) if b.id is not None else None for b in result.boxes]

            for box, conf, tid in zip(all_boxes, all_confs, all_tids):
                if tid is None: continue
                x1, y1, x2, y2 = box
                foot_point = get_foot_point(x1, y1, x2, y2)

                feature_vector = None
                time_to_update = (frame_id - last_reid_frame.get(tid, 0)) >= REID_INTERVAL
                is_unknown = tid not in identity_manager.track_to_worker

                # شرط عکس‌برداری با حاشیه داینامیک
                if (is_unknown or time_to_update) and is_safe_to_extract(x1, y1, x2, y2, all_boxes, frame_w, frame_h, cam_config['margin']):
                    crop = frame[max(0, y1):min(frame_h, y2), max(0, x1):min(frame_w, x2)]
                    
                    # بررسی سایز حداقلی قبل از ارسال به اکسترکتور
                    if crop.shape[0] >= cam_config['min_crop_h'] and crop.shape[1] >= cam_config['min_crop_w']:
                        feature_vector = reid_extractor.extract(crop)
                        if feature_vector is not None:
                            last_reid_frame[tid] = frame_id

                person_id = identity_manager.identify(tid, feature_vector, foot_point, timestamp)
                momentary_zone = zone_manager.get_zone_for_point(foot_point)

                if person_id.startswith("Worker_"):
                    event, event_data, confirmed_state, active_dwell = state_machine.update(person_id, momentary_zone, timestamp)

                    if event == "ZONE_ENTER":
                        dt_str = datetime.fromtimestamp(event_data['timestamp']).strftime('%H:%M:%S')
                        recent_events.append(f"[{dt_str}] {person_id} ENTER {event_data['zone']}")
                        csv_writer.writerow([dt_str, person_id, "ENTER", event_data['zone'], 0.0])

                    elif event == "ZONE_EXIT":
                        dt_str = datetime.fromtimestamp(event_data['timestamp']).strftime('%H:%M:%S')
                        dur = event_data['duration']
                        recent_events.append(f"[{dt_str}] {person_id} EXIT {event_data['zone']} ({dur:.1f}s)")
                        csv_writer.writerow([dt_str, person_id, "EXIT", event_data['zone'], round(dur, 1)])
                    
                    if len(recent_events) > 5: recent_events.pop(0)

                    color = (200, 0, 200) if confirmed_state == 'INSIDE' else (255, 165, 0)
                    label = f"{person_id} [{active_dwell:.1f}s]" if confirmed_state == 'INSIDE' else f"{person_id} [OUT]"
                else:
                    state_machine.update(f"Temp_{tid}", None, timestamp)
                    color = (0, 255, 255)
                    label = person_id

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                cv2.circle(frame, foot_point, 5, (0, 0, 255), -1)

        # رفتگر خروج اجباری (FORCE_EXIT)
        timeout_events = state_machine.process_missing_workers(timestamp)
        for t_event in timeout_events:
            if t_event['person_id'].startswith("Worker_"):
                pid = t_event['person_id']
                dur = t_event['event_data']['duration']
                z = t_event['event_data']['zone']
                dt_str = datetime.fromtimestamp(t_event['event_data']['timestamp']).strftime('%H:%M:%S')
                
                recent_events.append(f"[{dt_str}] {pid} FORCE_EXIT {z} ({dur:.1f}s)")
                csv_writer.writerow([dt_str, pid, "FORCE_EXIT", z, round(dur, 1)])
                if len(recent_events) > 5: recent_events.pop(0)

        # رسم لاگ رویدادها روی ویدیو
        y_offset = 80
        cv2.putText(frame, "--- EVENT LOG ---", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        for ev_text in recent_events:
            y_offset += 20
            cv2.putText(frame, ev_text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # محاسبه FPS و رندر نهایی
        processing_time = time.time() - start_time
        fps = 1.0 / processing_time if processing_time > 0 else 0
        cv2.putText(frame, f"Frame: {frame_id} | FPS: {fps:.1f} | Infer: {inference_time_ms:.1f}ms", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        writer.write(frame)
        cv2.imshow("Factory Enterprise Pipeline", frame)
        if cv2.waitKey(frame_delay) & 0xFF == ord('q'): break

    # آزادسازی منابع
    csv_file.close()
    reader.release()
    writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()