import logging
import math
import numpy as np

logger = logging.getLogger(__name__)

class SpatioTemporalIdentityManager:
    def __init__(self, similarity_threshold=0.65, max_speed_px_sec=400, max_gallery_size=8):
        self.similarity_threshold = similarity_threshold
        self.max_speed_px_sec = max_speed_px_sec 
        self.max_gallery_size = max_gallery_size

        self.worker_database = {}  
        self.track_to_worker = {}
        self.spatial_memory_pos = {}   
        self.spatial_memory_time = {}  
        
        # قفل ضد کلون: جلوگیری از تخصیص یک کارگر به دو کادر مختلف در یک فریم
        self.active_workers_in_frame = set()
        self.last_frame_time = 0

    def start_new_frame(self, current_time):
        """باید در ابتدای پردازش هر فریم صدا زده شود تا لیست کارگران حاضر صفر شود"""
        if current_time != self.last_frame_time:
            self.active_workers_in_frame = set()
            self.last_frame_time = current_time

    def _physics_allows(self, worker_id: str, current_pos: tuple, current_time: float) -> bool:
        if worker_id not in self.spatial_memory_pos: return True
        last_pos = self.spatial_memory_pos[worker_id]
        dt = current_time - self.spatial_memory_time[worker_id]
        if dt <= 0: return True
        dist = math.hypot(current_pos[0] - last_pos[0], current_pos[1] - last_pos[1])
        if (dist / dt) > self.max_speed_px_sec: return False
        return True

    def identify(self, track_id: int, feature_vector: np.ndarray, current_pos: tuple, current_time: float) -> str:
        # ۱. ردیابی مستمر و سالم
        if track_id in self.track_to_worker:
            worker_id = self.track_to_worker[track_id]
            self.spatial_memory_pos[worker_id] = current_pos
            self.spatial_memory_time[worker_id] = current_time
            self.active_workers_in_frame.add(worker_id) # قفل کردن این کارگر برای این فریم
            
            # فقط زوایای تمیز و منطقی را یاد بگیر (جلوگیری از مسمومیت)
            if feature_vector is not None:
                gallery = self.worker_database[worker_id]
                max_sim = max([float(np.dot(feature_vector, g)) for g in gallery])
                if 0.50 < max_sim < 0.85: 
                    if len(gallery) >= self.max_gallery_size: gallery.pop(1)
                    gallery.append(feature_vector.copy())
            return worker_id

        # ۲. بازیابی مکانی (Spatial Recovery) - سریعترین روش برای ویدیوی 7 فریم
        # آیا کسی اخیراً در همین شعاع گم شده بود که در این فریم لیبل نگرفته باشد؟
        best_spatial_match = None
        min_dist = float('inf')
        
        for w_id, last_pos in self.spatial_memory_pos.items():
            if w_id in self.active_workers_in_frame: continue # قبلاً در این فریم پیدا شده!
                
            dt = current_time - self.spatial_memory_time.get(w_id, 0)
            if 0 < dt < 2.0: # اگر کمتر از 2 ثانیه پیش گمش کردیم
                dist = math.hypot(current_pos[0] - last_pos[0], current_pos[1] - last_pos[1])
                if dist < 80 and dist < min_dist: # شعاع 80 پیکسلی
                    min_dist = dist
                    best_spatial_match = w_id
                    
        if best_spatial_match:
            self.track_to_worker[track_id] = best_spatial_match
            self.spatial_memory_pos[best_spatial_match] = current_pos
            self.spatial_memory_time[best_spatial_match] = current_time
            self.active_workers_in_frame.add(best_spatial_match)
            return best_spatial_match

        # ۳. کارگر کاملاً از کادر خارج شده بود و دوباره برگشته است (تشخیص با ترانسفورمر)
        if feature_vector is None: return f"Detecting_{track_id}..."

        best_match = None
        best_sim = -1.0

        for worker_id, gallery in self.worker_database.items():
            if worker_id in self.active_workers_in_frame: continue # قفل ضد کلون
                
            sim = max([float(np.dot(feature_vector, g)) for g in gallery])
            if sim > best_sim and sim >= self.similarity_threshold:
                if self._physics_allows(worker_id, current_pos, current_time):
                    best_sim = sim
                    best_match = worker_id

        if best_match is not None:
            gallery = self.worker_database[best_match]
            if len(gallery) >= self.max_gallery_size: gallery.pop(1)
            gallery.append(feature_vector.copy())
            
            self.track_to_worker[track_id] = best_match
            self.spatial_memory_pos[best_match] = current_pos
            self.spatial_memory_time[best_match] = current_time
            self.active_workers_in_frame.add(best_match)
            return best_match
            
        else:
            new_worker_id = f"Worker_{len(self.worker_database) + 1}"
            self.worker_database[new_worker_id] = [feature_vector.copy()]
            self.track_to_worker[track_id] = new_worker_id
            self.spatial_memory_pos[new_worker_id] = current_pos
            self.spatial_memory_time[new_worker_id] = current_time
            self.active_workers_in_frame.add(new_worker_id)
            return new_worker_id