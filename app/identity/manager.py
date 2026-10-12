import logging
import numpy as np
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)

class IdentityManager:
    # کاهش سخت‌گیری به 60 درصد برای پذیرش تغییرات زاویه بدن
    def __init__(self, similarity_threshold=0.60, update_rate=0.15):
        self.similarity_threshold = similarity_threshold
        # نرخ آپدیت: 15% از ظاهر فریم جدید را با 85% از ظاهر قبلی ترکیب کن
        self.update_rate = update_rate 
        
        self.known_persons = {}
        self.track_to_person = {}

    def identify_and_update(self, track_id: int, feature_vector) -> str:
        if feature_vector is None:
            return self.track_to_person.get(track_id, f"Track_{track_id}")
            
        # حالت اول: ردیاب هنوز شخص را گم نکرده است -> فقط حافظه را با ظاهر جدید آپدیت کن
        if track_id in self.track_to_person:
            person_id = self.track_to_person[track_id]
            
            # ترکیب بردار قدیمی با جدید (Continuous Learning)
            updated_vector = (1.0 - self.update_rate) * self.known_persons[person_id] + self.update_rate * feature_vector
            # نرمال‌سازی مجدد ریاضی
            norm = np.linalg.norm(updated_vector)
            if norm > 0:
                updated_vector = updated_vector / norm
                
            self.known_persons[person_id] = updated_vector
            return person_id
            
        # حالت دوم: شخص جدید است (یا پس از گم شدن برگشته است) -> جستجو در دیتابیس
        best_match_id = None
        best_similarity = -1.0

        for person_id, known_vector in self.known_persons.items():
            sim = 1 - cosine(feature_vector, known_vector)
            if sim > best_similarity:
                best_similarity = sim
                best_match_id = person_id

        # اگر شباهت کافی بود -> بازیابی هویت قدیمی
        if best_match_id is not None and best_similarity >= self.similarity_threshold:
            logger.info(f"Re-ID Match: Track {track_id} is {best_match_id} (Sim: {best_similarity:.2f})")
            self.track_to_person[track_id] = best_match_id
            return best_match_id
        else:
            # شخص کاملا جدید است -> ایجاد پروفایل جدید
            new_person_id = f"Worker_{len(self.known_persons) + 1}"
            self.known_persons[new_person_id] = feature_vector
            self.track_to_person[track_id] = new_person_id
            
            sim_log = f" (Closest sim was {best_similarity:.2f})" if best_match_id else ""
            logger.info(f"Re-ID New: Created {new_person_id} for Track {track_id}{sim_log}")
            return new_person_id