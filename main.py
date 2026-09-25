import logging
import cv2
import os
from datetime import datetime
from app.streams.video_reader import LocalVideoReader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Phase 1: Local Video Pipeline (Speed and Timestamp Fixed)")
    
    video_path = "data/input/test_video.mp4"
    
    if not os.path.exists(video_path):
        logger.error(f"Test video not found at {video_path}.")
        return

    reader = None
    try:
        reader = LocalVideoReader(video_path)
        
        # Calculate dynamic delay based on video FPS (1000 milliseconds / FPS)
        # We use max(1, ...) to ensure the delay is at least 1 millisecond
        frame_delay = max(1, int(1000 / reader.fps))
        logger.info(f"Calculated frame delay for natural playback: {frame_delay} ms")
        
        for frame, timestamp, frame_id in reader.read_frames():
            
            # Convert Unix timestamp to a human-readable format (HH:MM:SS.mmm)
            dt_object = datetime.fromtimestamp(timestamp)
            readable_time = dt_object.strftime('%H:%M:%S.%f')[:-3]
            
            # Visualization
            text = f"Frame: {frame_id} | Time: {readable_time}"
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Factory Monitoring MVP", frame)
            
            # Use the calculated delay instead of fixed 1ms
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