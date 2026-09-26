import logging
import cv2
import os
import time
from datetime import datetime
from app.streams.video_reader import LocalVideoReader
from app.detection.detector import PersonDetector
from app.zones.manager import ZoneManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Phase 4: Manual Zones Integration")
    
    video_path = "data/input/test_video.mp4"
    config_path = "configs/settings.yaml"
    
    if not os.path.exists(video_path):
        logger.error(f"Test video not found at {video_path}.")
        return

    reader = None
    try:
        reader = LocalVideoReader(video_path)
        frame_delay = max(1, int(1000 / reader.fps))
        
        detector = PersonDetector(model_path="yolo11s.pt", conf_threshold=0.4)
        
        # Initialize Zone Manager
        zone_manager = ZoneManager(config_path)
        
        for frame, timestamp, frame_id in reader.read_frames():
            start_time = time.time()
            
            # Run tracking
            result = detector.track(frame)
            inference_time_ms = result.speed['inference']
            
            # Draw Zones on the frame (Blue color)
            zone_manager.draw_zones(frame)
            
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    if box.id is not None:
                        track_id = int(box.id[0])
                        label = f"ID: {track_id} | Conf: {conf:.2f}"
                        color = (255, 165, 0)
                    else:
                        label = f"Detecting... | Conf: {conf:.2f}"
                        color = (0, 255, 0)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
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