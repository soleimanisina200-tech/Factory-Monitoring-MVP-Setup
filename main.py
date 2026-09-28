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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_settings(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    logger.info("Starting Phase 6: Entry/Exit State Machine")
    
    video_path = "data/input/test_video.mp4"
    config_path = "configs/settings.yaml"
    
    if not os.path.exists(video_path):
        logger.error(f"Test video not found.")
        return

    reader = None
    try:
        settings = load_settings(config_path)
        reader = LocalVideoReader(video_path)
        frame_delay = max(1, int(1000 / reader.fps))
        
        detector = PersonDetector(model_path="yolo11s.pt", conf_threshold=0.4)
        zone_manager = ZoneManager(config_path)
        
        # Initialize State Machine with parameters from YAML
        events_config = settings.get('events', {})
        entry_frames = events_config.get('entry_confirmation_frames', 10)
        exit_frames = events_config.get('exit_confirmation_frames', 15)
        
        state_machine = ZoneStateMachine(entry_frames, exit_frames)
        
        # Visual list to show recent events on screen
        recent_events = []

        for frame, timestamp, frame_id in reader.read_frames():
            start_time = time.time()
            
            result = detector.track(frame)
            inference_time_ms = result.speed['inference']
            
            zone_manager.draw_zones(frame)
            
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    if box.id is not None:
                        track_id = int(box.id[0])
                        foot_point = get_foot_point(x1, y1, x2, y2)
                        
                        # 1. Get raw momentary position
                        momentary_zone = zone_manager.get_zone_for_point(foot_point)
                        
                        # 2. Pass momentary position to State Machine to get Confirmed State
                        event, confirmed_state = state_machine.update(track_id, momentary_zone)
                        
                        # 3. Handle generated Events
                        if event:
                            dt_str = datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
                            recent_events.append(f"[{dt_str}] ID:{track_id} {event}")
                            # Keep only the last 5 events
                            if len(recent_events) > 5:
                                recent_events.pop(0)

                        # Visualization based on CONFIRMED state, not momentary
                        if confirmed_state == 'INSIDE':
                            label = f"ID:{track_id} [INSIDE]"
                            color = (200, 0, 200) # Purple
                        else:
                            label = f"ID:{track_id} [OUTSIDE]"
                            color = (255, 165, 0) # Orange
                    else:
                        label = "Detecting..."
                        color = (0, 255, 0)
                        foot_point = get_foot_point(x1, y1, x2, y2)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.circle(frame, foot_point, 5, (0, 0, 255), -1)
            
            # Draw Event Log on screen
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
            
            cv2.imshow("Factory Monitoring MVP", frame)
            
            if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
                logger.info("User interrupted the stream.")
                break
                
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
    finally:
        if reader:
            reader.release()
        cv2.destroyAllWindows()
        logger.info("System shut down securely.")

if __name__ == "__main__":
    main()