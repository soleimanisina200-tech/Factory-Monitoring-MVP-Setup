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

def main():
    logger.info("Starting Phase 9: Person Re-ID (Balanced Speed & Accuracy)")
    
    video_path = "data/input/test_video.mp4"
    config_path = "configs/settings.yaml"
    
    if not os.path.exists(video_path):
        logger.error("Test video not found.")
        return

    reader = None
    writer = None
    try:
        settings = load_settings(config_path)
        reader = LocalVideoReader(video_path)
        frame_delay = max(1, int(1000 / reader.fps))
        
        output_dir = "data/output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "processed_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, reader.fps, (reader.width, reader.height))
        
        detector = PersonDetector(model_path="yolo11s.pt", conf_threshold=0.4)
        zone_manager = ZoneManager(config_path)
        
        reid_extractor = OSNetExtractor(device='cuda')
        identity_manager = IdentityManager(similarity_threshold=0.60, update_rate=0.15)
        
        events_config = settings.get('events', {})
        state_machine = ZoneStateMachine(
            events_config.get('entry_confirmation_frames', 10),
            events_config.get('exit_confirmation_frames', 15)
        )
        
        recent_events = []
        MARGIN = 40 
        
        # --- سیستم تعادل بین سرعت و دقت ---
        # چون ویدئوی شما 7 FPS است، عدد 3 یعنی تقریبا هر 0.4 ثانیه آپدیت می‌شود
        REID_INTERVAL = 3 
        last_reid_update = {} 
        # -------------------------------

        for frame, timestamp, frame_id in reader.read_frames():
            start_time = time.time()
            
            result = detector.track(frame)
            inference_time_ms = result.speed['inference']
            zone_manager.draw_zones(frame)
            
            frame_height, frame_width = frame.shape[:2]
            
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    if box.id is not None:
                        track_id = int(box.id[0])
                        person_id = "Unknown"
                        
                        is_touching_edge = (x1 < MARGIN or y1 < MARGIN or x2 > frame_width - MARGIN or y2 > frame_height - MARGIN)
                        
                        needs_reid = False
                        if track_id not in identity_manager.track_to_person:
                            needs_reid = True 
                        else:
                            last_update_frame = last_reid_update.get(track_id, 0)
                            if (frame_id - last_update_frame) >= REID_INTERVAL:
                                needs_reid = True

                        # فیلتر کیفیت: فقط وقتی Re-ID کن که فرد کاملا در کادر است و YOLO با اطمینان بالای 60% او را واضح می‌بیند
                        if needs_reid and not is_touching_edge and conf > 0.60:
                            crop_y1, crop_y2 = max(0, y1), min(frame_height, y2)
                            crop_x1, crop_x2 = max(0, x1), min(frame_width, x2)
                            person_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                            
                            features = reid_extractor.extract(person_crop)
                            person_id = identity_manager.identify_and_update(track_id, features)
                            
                            last_reid_update[track_id] = frame_id
                        else:
                            # اگر شرایط عکس‌برداری خوب نبود، از حافظه استفاده کن و کارت گرافیک را درگیر نکن
                            person_id = identity_manager.track_to_person.get(track_id, f"Pending_{track_id}")
                        
                        foot_point = get_foot_point(x1, y1, x2, y2)
                        momentary_zone = zone_manager.get_zone_for_point(foot_point)
                        
                        event, event_data, confirmed_state, active_dwell = state_machine.update(
                            track_id, momentary_zone, timestamp
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
                            dwell_str = f"{active_dwell:.1f}s" if active_dwell else "0.0s"
                            label = f"{person_id} [{dwell_str}]"
                            color = (200, 0, 200)
                        else:
                            label = f"{person_id} [OUT]"
                            color = (255, 165, 0)
                    else:
                        label = "Detecting..."
                        color = (0, 255, 0)
                        foot_point = get_foot_point(x1, y1, x2, y2)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.circle(frame, foot_point, 5, (0, 0, 255), -1)
            
            y_offset = 80
            cv2.putText(frame, "--- EVENT LOG ---", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            for ev_text in recent_events:
                y_offset += 20
                cv2.putText(frame, ev_text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            processing_time = time.time() - start_time
            pipeline_fps = 1.0 / processing_time if processing_time > 0 else 0
            dt_object = datetime.fromtimestamp(timestamp)
            readable_time = dt_object.strftime('%H:%M:%S.%f')[:-3]
            metrics_text = f"Frame: {frame_id} | Time: {readable_time} | FPS: {pipeline_fps:.1f} | Inference: {inference_time_ms:.1f}ms"
            cv2.putText(frame, metrics_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            writer.write(frame)
            cv2.imshow("Factory Monitoring MVP", frame)
            
            if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
                logger.info("User interrupted the stream.")
                break
                
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
    finally:
        if reader:
            reader.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        logger.info("System shut down securely.")

if __name__ == "__main__":
    main()