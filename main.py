import logging
import cv2
import os
import time
import yaml
from datetime import datetime
from app.streams.video_reader import LocalVideoReader
from app.detection.detector import PersonDetector
from app.zones.manager import ZoneManager
from app.logic.geometry import get_foot_point
from app.logic.state_machine import ZoneStateMachine
from app.identity.reid_model import OSNetExtractor
from app.identity.manager import IdentityManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_settings(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def is_isolated_and_standing(x1, y1, x2, y2, all_boxes, frame_w, frame_h, margin=30):
    """بررسی کیفی: آیا کارگر کاملاً ایستاده، دور از لبه‌ها و جدا از بقیه است؟"""
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    
    # فیلتر نشستن: نسبت ارتفاع به عرض باید منطقی باشد
    if not (1.30 <= (h / w) <= 3.8):
        return False
        
    # فیلتر لبه تصویر: جلوگیری از عکس‌برداری وقتی کارگر نصفه وارد دوربین شده
    if not (margin < x1 and margin < y1 and x2 < frame_w - margin and y2 < frame_h - margin):
        return False
        
    # فیلتر تقاطع: عدم تداخل و مماس بودن با افراد دیگر
    for other_box in all_boxes:
        if other_box == [x1, y1, x2, y2]: 
            continue
        ox1, oy1, ox2, oy2 = other_box
        inter_x = max(0, min(x2, ox2) - max(x1, ox1))
        inter_y = max(0, min(y2, oy2) - max(y1, oy1))
        if (inter_x * inter_y) > 0:
            return False
            
    return True

def main():
    logger.info("Starting MVP Demo Pipeline (Head-Cropping + Active Locks + FORCE_EXIT)")

    video_path = "data/input/test_video.mp4"
    config_path = "configs/settings.yaml"

    if not os.path.exists(video_path):
        logger.error("Test video not found. Please check data/input/test_video.mp4")
        return

    reader = None
    writer = None
    try:
        settings = load_settings(config_path)
        reader = LocalVideoReader(video_path)
        frame_delay = max(1, int(1000 / reader.fps))

        os.makedirs("data/output", exist_ok=True)
        output_path = os.path.join("data/output", "processed_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, reader.fps, (reader.width, reader.height))

        # 1. ردیاب با رزولوشن 960 (تنظیم شده برای ByteTrack)
        detector = PersonDetector(model_path="yolo11m.pt", conf_threshold=0.45, imgsz=960)
        zone_manager = ZoneManager(config_path)
        
        # 2. ماژول‌های تشخیص هویت (نسخه هک‌شده برای لباس فرم)
        reid_extractor = OSNetExtractor(device='cuda')
        identity_manager = IdentityManager(similarity_threshold=0.82, frames_to_collect=3)

        # 3. ماشین حالت زمان‌سنج به همراه رفتگر FORCE_EXIT
        events_config = settings.get('events', {})
        state_machine = ZoneStateMachine(
            entry_frames=events_config.get('entry_confirmation_frames', 10),
            exit_frames=events_config.get('exit_confirmation_frames', 15),
            timeout_sec=4.0  # صدور خروج اجباری پس از 4 ثانیه غیبت
        )

        recent_events = []

        for frame, timestamp, frame_id in reader.read_frames():
            start_time = time.time()
            frame_h, frame_w = frame.shape[:2]

            result = detector.track(frame)
            inference_time_ms = result.speed['inference']
            zone_manager.draw_zones(frame)

            if result.boxes is not None and len(result.boxes) > 0:
                all_boxes = [list(map(int, b.xyxy[0])) for b in result.boxes]
                all_confs = [float(b.conf[0]) for b in result.boxes]
                all_tids = [int(b.id[0]) if b.id is not None else None for b in result.boxes]

                for box, conf, tid in zip(all_boxes, all_confs, all_tids):
                    if tid is None: 
                        continue
                    x1, y1, x2, y2 = box

                    # -------------------------------------------------------------
                    # فاز هویت‌سنجی: اگر شناسایی شده، OSNet را دور بزن (افزایش FPS)
                    # -------------------------------------------------------------
                    if identity_manager.is_track_identified(tid):
                        person_id = identity_manager.get_label(tid)
                    else:
                        # اگر ناشناس است و شرایط عکس‌برداری ایده‌آل است
                        if conf > 0.60 and is_isolated_and_standing(x1, y1, x2, y2, all_boxes, frame_w, frame_h):
                            crop = frame[max(0, y1):min(frame_h, y2), max(0, x1):min(frame_w, x2)]
                            feat_vector = reid_extractor.extract(crop)
                            
                            # ارسال لیست افراد حاضر (all_tids) به منیجر برای جلوگیری از دادن لیبل تکراری
                            person_id = identity_manager.add_feature_and_identify(tid, feat_vector, all_tids)
                        else:
                            person_id = identity_manager.get_label(tid)

                    # -------------------------------------------------------------
                    # فاز زمان‌سنجی (ماشین حالت و نواحی)
                    # -------------------------------------------------------------
                    foot_point = get_foot_point(x1, y1, x2, y2)
                    momentary_zone = zone_manager.get_zone_for_point(foot_point)

                    # فقط زمانی تایمر را برای مناطق (Zones) کنتور بینداز که فرد کاملاً شناخته شده باشد
                    if person_id.startswith("Worker_"):
                        event, event_data, confirmed_state, active_dwell = state_machine.update(
                            person_id, momentary_zone, timestamp
                        )

                        if event == "ZONE_ENTER":
                            dt_str = datetime.fromtimestamp(event_data['timestamp']).strftime('%H:%M:%S')
                            recent_events.append(f"[{dt_str}] {person_id} ENTER {event_data['zone']}")
                        elif event == "ZONE_EXIT":
                            dt_str = datetime.fromtimestamp(event_data['timestamp']).strftime('%H:%M:%S')
                            dur = event_data['duration']
                            recent_events.append(f"[{dt_str}] {person_id} EXIT {event_data['zone']} ({dur:.1f}s)")
                        
                        if len(recent_events) > 5: 
                            recent_events.pop(0)

                        if confirmed_state == 'INSIDE':
                            label = f"{person_id} [{active_dwell:.1f}s]"
                            color = (200, 0, 200)  # رنگ بنفش برای داخل ناحیه
                        else:
                            label = f"{person_id} [OUT]"
                            color = (255, 165, 0)  # رنگ نارنجی برای خارج ناحیه
                    else:
                        # در حال تحلیل (Analyzing...)، تایمر کاذب می‌سازیم تا ماشین حالت ارور ندهد
                        state_machine.update(f"Temp_{tid}", None, timestamp)
                        label = person_id
                        color = (0, 255, 255)  # رنگ زرد برای وضعیت در حال پردازش

                    # رسم گرافیک‌ها
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                    cv2.circle(frame, foot_point, 5, (0, 0, 255), -1)

            # -------------------------------------------------------------
            # رفتگر زمان‌سنج: صدور خروج اجباری برای کارگرانی که کادر را ترک کرده‌اند
            # -------------------------------------------------------------
            timeout_events = state_machine.process_missing_workers(timestamp)
            for t_event in timeout_events:
                # لاگ FORCE EXIT فقط برای کارگران واقعی (نه Аналиزها)
                if t_event['person_id'].startswith("Worker_"):
                    pid = t_event['person_id']
                    dur = t_event['event_data']['duration']
                    z = t_event['event_data']['zone']
                    dt_str = datetime.fromtimestamp(t_event['event_data']['timestamp']).strftime('%H:%M:%S')
                    recent_events.append(f"[{dt_str}] {pid} FORCE EXIT {z} ({dur:.1f}s)")
                    if len(recent_events) > 5:
                        recent_events.pop(0)

            # نمایش لاگ‌ها در گوشه تصویر
            y_offset = 80
            cv2.putText(frame, "--- EVENT LOG ---", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            for ev_text in recent_events:
                y_offset += 20
                cv2.putText(frame, ev_text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # نمایش متریک‌های سرعت
            processing_time = time.time() - start_time
            fps = 1.0 / processing_time if processing_time > 0 else 0
            cv2.putText(frame, f"Frame: {frame_id} | FPS: {fps:.1f} | Infer: {inference_time_ms:.1f}ms", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            writer.write(frame)
            cv2.imshow("Factory Monitoring MVP", frame)
            
            # کلید خروج
            if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
                logger.info("User interrupted the stream.")
                break

    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
    finally:
        if reader: reader.release()
        if writer: writer.release()
        cv2.destroyAllWindows()
        logger.info("System shut down securely.")

if __name__ == "__main__":
    main()