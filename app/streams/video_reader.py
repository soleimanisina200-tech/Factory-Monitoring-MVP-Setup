import time
import logging
import threading
import cv2

logger = logging.getLogger(__name__)


class VideoReader:
    """
    خواننده ویدیو با پشتیبانی همزمان از استریم‌های زنده (RTSP/Webcam) در ترد مجزا
    و فایل‌های ویدیویی آفلاین، همراه با جلوگیری از پردازش فریم‌های تکراری.
    """

    def __init__(self, source, width=None, height=None, reconnect_delay=3.0):
        self.source = source
        self.width = width
        self.height = height
        self.reconnect_delay = reconnect_delay

        self.is_live_stream = isinstance(source, int) or (
            isinstance(source, str)
            and source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
        )

        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.source}")

        if self.is_live_stream:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not self.fps or self.fps <= 0 or self.fps > 120:
            self.fps = 25.0

        self.orig_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.orig_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_w = self.width if self.width else self.orig_w
        self.frame_h = self.height if self.height else self.orig_h

        self.frame = None
        self.ret = False
        self.frame_id = 0
        self.last_delivered_id = -1
        self.frame_timestamp = 0.0
        self.stopped = False

        self.condition = threading.Condition()
        self.thread = None

        if self.is_live_stream:
            self.thread = threading.Thread(target=self._update_live_stream, daemon=True)
            self.thread.start()
            logger.info(f"Started threaded live stream reader for {self.source} ({self.frame_w}x{self.frame_h} @ {self.fps:.1f} FPS)")
        else:
            logger.info(f"Opened video file reader for {self.source} ({self.frame_w}x{self.frame_h} @ {self.fps:.1f} FPS)")

    def _resize_if_needed(self, frame):
        if frame is None:
            return None
        if self.width and self.height and (frame.shape[1] != self.width or frame.shape[0] != self.height):
            return cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        return frame

    def _update_live_stream(self):
        while not self.stopped:
            if not self.cap.isOpened():
                logger.warning("Stream disconnected. Attempting to reconnect...")
                time.sleep(self.reconnect_delay)
                self.cap = cv2.VideoCapture(self.source)
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                continue

            ret, frame = self.cap.read()
            now = time.time()

            if not ret or frame is None:
                logger.warning("Empty frame from live stream; reconnecting...")
                self.cap.release()
                time.sleep(self.reconnect_delay)
                continue

            frame = self._resize_if_needed(frame)
            with self.condition:
                self.ret = True
                self.frame = frame
                self.frame_id += 1
                self.frame_timestamp = now
                self.condition.notify_all()

    def read_with_meta(self, wait_timeout=1.0):
        """
        برگرداندن (ret, frame, frame_id, timestamp) با تضمین عدم ارسال فریم تکراری.
        """
        if not self.is_live_stream:
            if self.stopped or not self.cap.isOpened():
                return False, None, self.frame_id, self.frame_timestamp

            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.stopped = True
                return False, None, self.frame_id, self.frame_timestamp

            frame = self._resize_if_needed(frame)
            self.frame_id += 1
            pos_msec = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            if pos_msec and pos_msec > 0:
                self.frame_timestamp = pos_msec / 1000.0
            else:
                self.frame_timestamp = self.frame_id / float(self.fps)

            self.last_delivered_id = self.frame_id
            return True, frame, self.frame_id, self.frame_timestamp

        with self.condition:
            if self.frame_id == self.last_delivered_id and not self.stopped:
                self.condition.wait(timeout=wait_timeout)

            if self.stopped or not self.ret or self.frame is None:
                return False, None, self.frame_id, self.frame_timestamp

            if self.frame_id == self.last_delivered_id:
                return False, None, self.frame_id, self.frame_timestamp

            self.last_delivered_id = self.frame_id
            return True, self.frame.copy(), self.frame_id, self.frame_timestamp

    def read(self):
        """متد سازگار با نسخه‌های قبلی که فقط (ret, frame) برمی‌گرداند."""
        ret, frame, _, _ = self.read_with_meta()
        return ret, frame

    def stop(self):
        self.stopped = True
        with self.condition:
            self.condition.notify_all()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

    def release(self):
        self.stop()


ThreadedVideoReader = VideoReader