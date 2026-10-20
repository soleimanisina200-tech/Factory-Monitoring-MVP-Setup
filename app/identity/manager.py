import logging
import numpy as np

logger = logging.getLogger(__name__)

class IdentityManager:
    # چون تمرکز را بردیم روی سر، آستانه را روی 0.82 تنظیم می‌کنیم
    def __init__(self, similarity_threshold=0.82, frames_to_collect=3):
        self.similarity_threshold = similarity_threshold
        self.frames_to_collect = frames_to_collect
        
        self.worker_database = {}
        self.track_to_worker = {}
        self.tracklet_buffer = {}
        self.identified_tracks = set()

    def is_track_identified(self, track_id: int) -> bool:
        return track_id in self.identified_tracks

    def get_label(self, track_id: int) -> str:
        return self.track_to_worker.get(track_id, f"Analyzing_{track_id}...")

    def get_active_workers(self, current_active_tids: list) -> set:
        """بررسی اینکه چه کارگرانی در حال حاضر در تصویر زنده هستند"""
        return {self.track_to_worker[tid] for tid in current_active_tids if tid in self.track_to_worker}

    def add_feature_and_identify(self, track_id: int, feature_vector: np.ndarray, active_tids: list) -> str:
        if feature_vector is None or track_id in self.identified_tracks:
            return self.get_label(track_id)

        if track_id not in self.tracklet_buffer:
            self.tracklet_buffer[track_id] = []
        self.tracklet_buffer[track_id].append(feature_vector)

        if len(self.tracklet_buffer[track_id]) < self.frames_to_collect:
            return f"Analyzing_{track_id}... ({len(self.tracklet_buffer[track_id])}/{self.frames_to_collect})"

        # تولید امضای طلایی
        mean_vector = np.mean(self.tracklet_buffer[track_id], axis=0)
        golden_signature = mean_vector / max(1e-6, np.linalg.norm(mean_vector))

        best_match = None
        best_sim = -1.0
        
        # لیست افرادی که همین الان در قاب دوربین هستند (نمی‌توانند فرد جدید باشند)
        currently_active_workers = self.get_active_workers(active_tids)

        for worker_id, saved_sig in self.worker_database.items():
            # فقط با کارگرانی مقایسه کن که غایب هستند!
            if worker_id in currently_active_workers:
                continue
                
            sim = float(np.dot(golden_signature, saved_sig))
            if sim > best_sim:
                best_sim = sim
                best_match = worker_id

        # اگر فرد قبلی برگشته است
        if best_match is not None and best_sim >= self.similarity_threshold:
            self.track_to_worker[track_id] = best_match
            self.identified_tracks.add(track_id)
            # آپدیت آرام دیتابیس برای یادگیری تغییر نور
            self.worker_database[best_match] = 0.85 * self.worker_database[best_match] + 0.15 * golden_signature
            self.worker_database[best_match] /= np.linalg.norm(self.worker_database[best_match])
            logger.info(f"Identity RECOVERED: Track {track_id} is {best_match} (Sim: {best_sim:.2f})")
            
        else:
            # ثبت کارگر کاملاً جدید
            new_worker_id = f"Worker_{len(self.worker_database) + 1}"
            self.worker_database[new_worker_id] = golden_signature
            self.track_to_worker[track_id] = new_worker_id
            self.identified_tracks.add(track_id)
            logger.info(f"New Worker REGISTERED: {new_worker_id} on Track {track_id}")

        del self.tracklet_buffer[track_id]
        return self.track_to_worker[track_id]