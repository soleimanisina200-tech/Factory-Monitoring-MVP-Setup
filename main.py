import logging
import cv2
import os
import time
from datetime import datetime
from app.streams.video_reader import LocalVideoReader
from app.detection.detector import PersonDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Phase 2: Person Detection with YOLO11s")
    
    video_path = "data/input/test_video.mp4"
    if not os.path.exists(video_path):
        logger.error(f"Test video not found at {video_path}.")
        return

    reader = None
    try:
        reader = LocalVideoReader(video_path)
        frame_delay = max(1, int(1000 / reader.fps))
        
        # Initialize detector
        detector = PersonDetector(model_path="yolo11s.pt", conf_threshold=0.4)
        logger.info(f"Original Video FPS: {reader.fps}")
        for frame, timestamp, frame_id in reader.read_frames():
            
            # 1. Pipeline Start Time (for overall FPS calculation)
            start_time = time.time()
            
            # 2. Run Person Detection
            result = detector.detect(frame)
            
            # 3. Extract Inference Time directly from YOLO results
            # The speed dictionary contains 'preprocess', 'inference', and 'postprocess' times in ms
            inference_time_ms = result.speed['inference']
            
            # 4. Draw Bounding Boxes
            for box in result.boxes:
                # Get coordinates, confidence, and class
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                # Draw Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw Label (Person + Confidence)
                label = f"Person: {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 5. Pipeline End Time & Calculate FPS
            processing_time = time.time() - start_time
            pipeline_fps = 1.0 / processing_time if processing_time > 0 else 0
            
            # 6. UI Visualization for Metrics
            dt_object = datetime.fromtimestamp(timestamp)
            readable_time = dt_object.strftime('%H:%M:%S.%f')[:-3]
            
            # Display System Metrics
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