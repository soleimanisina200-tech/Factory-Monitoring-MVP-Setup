import cv2
import time
import logging

logger = logging.getLogger(__name__)

class LocalVideoReader:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.frame_count = 0
        
        if not self.cap.isOpened():
            logger.error(f"Failed to open video file: {video_path}")
            raise ValueError(f"Cannot open video: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Video opened: {video_path} | {self.width}x{self.height} @ {self.fps:.2f} FPS")

    def read_frames(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                logger.info("End of video stream reached.")
                break
            
            self.frame_count += 1
            
            # CRITICAL: Wall-clock timestamp attached exactly when frame is received
            capture_timestamp = time.time()
            
            yield frame, capture_timestamp, self.frame_count

    def release(self):
        if self.cap:
            self.cap.release()
        logger.info("Video stream released.")