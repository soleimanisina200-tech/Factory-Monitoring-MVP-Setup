import logging
from collections import deque, Counter
import numpy as np
from scipy.optimize import linear_sum_assignment

from app.logic.geometry import (
    is_touching_border,
    is_skeleton_complete,
    estimate_body_orientation,
    estimate_normalized_height,
)

logger = logging.getLogger(__name__)

# ماتریس وزن تطبیق بین باکت‌های زاویه دید (Cross-Orientation Soft Weights)
ORIENTATION_COMPATIBILITY = {
    ("FRONT", "FRONT"): 1.00,
    ("BACK", "BACK"): 1.00,
    ("LEFT", "LEFT"): 1.00,
    ("RIGHT", "RIGHT"): 1.00,
    ("FRONT", "LEFT"): 0.90,
    ("LEFT", "FRONT"): 0.90,
    ("FRONT", "RIGHT"): 0.90,
    ("RIGHT", "FRONT"): 0.90,
    ("BACK", "LEFT"): 0.90,
    ("LEFT", "BACK"): 0.90,
    ("BACK", "RIGHT"): 0.90,
    ("RIGHT", "BACK"): 0.90,
    ("FRONT", "BACK"): 0.78,
    ("BACK", "FRONT"): 0.78,
    ("LEFT", "RIGHT"): 0.80,
    ("RIGHT", "LEFT"): 0.80,
}


class EnterpriseIdentityManager:
    """
    مدیریت هویت صنعتی مقاوم به لباس فرم یکسان، چرخش زاویه بدن و سرقت آیدی در لبه تصویر.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.68,
        margin_threshold: float = 0.06,
        reverify_sim_floor: float = 0.52,
        max_gallery_size: int = 8,
        probation_frames: int = 4,
        track_ttl_sec: float = 180.0,
        border_cooldown_sec: float = 3.5,
        max_speed_px_per_sec: float = 450.0,
        uniform_whitening_alpha: float = 0.55,
        border_margin_px: int = 28,
    ):
        self.similarity_threshold = similarity_threshold
        self.margin_threshold = margin_threshold
        self.reverify_sim_floor = reverify_sim_floor
        self.max_gallery_size = max_gallery_size
        self.probation_frames = probation_frames
        self.track_ttl_sec = track_ttl_sec
        self.border_cooldown_sec = border_cooldown_sec
        self.max_speed_px_per_sec = max_speed_px_per_sec
        self.uniform_whitening_alpha = uniform_whitening_alpha
        self.border_margin_px = border_margin_px

        self.next_worker_num = 1
        self.workers = {}          # person_id -> {buckets, norm_height, last_pos, last_seen_ts, exit_state, exit_ts, exit_edge}
        self.track_to_person = {}  # track_id -> person_id
        self.track_meta = {}       # track_id -> {pos_history, vel, orient_history, probation_vecs, aspect_ratio, last_seen_ts, was_on_border}

        self.global_uniform_mean = None
        self.frame_counter = 0

    def _whiten_feature(self, feat: np.ndarray, update_mean: bool = False) -> np.ndarray:
        """
        حذف بردار میانگین لباس فرم کارخانه (Uniform Feature Centering) و نرمال‌سازی مجدد L2.
        """
        if feat is None:
            return None

        if self.global_uniform_mean is None:
            self.global_uniform_mean = feat.copy()
            return feat

        if update_mean:
            self.global_uniform_mean = 0.992 * self.global_uniform_mean + 0.008 * feat
            norm_m = np.linalg.norm(self.global_uniform_mean)
            if norm_m > 1e-6:
                self.global_uniform_mean /= norm_m

        centered = feat - (self.uniform_whitening_alpha * self.global_uniform_mean)
        norm_c = np.linalg.norm(centered)
        return (centered / norm_c) if norm_c > 1e-6 else feat

    def _get_border_edge(self, box: list[int], frame_w: int, frame_h: int) -> str | None:
        x1, y1, x2, y2 = box
        m = self.border_margin_px
        if x1 <= m:
            return "LEFT_EDGE"
        if x2 >= frame_w - m:
            return "RIGHT_EDGE"
        if y1 <= m:
            return "TOP_EDGE"
        if y2 >= frame_h - m:
            return "BOTTOM_EDGE"
        return None

    def _update_track_kinematics(
        self,
        track_id: int,
        box: list[int],
        keypoints: np.ndarray | None,
        timestamp: float,
        frame_w: int,
        frame_h: int,
    ) -> dict:
        cx = 0.5 * (box[0] + box[2])
        cy = 0.5 * (box[1] + box[3])
        bw = max(1.0, float(box[2] - box[0]))
        bh = max(1.0, float(box[3] - box[1]))
        current_ar = bh / bw

        raw_orient, orient_conf = estimate_body_orientation(keypoints, box)
        on_border = is_touching_border(box, frame_w, frame_h, self.border_margin_px)
        border_edge = self._get_border_edge(box, frame_w, frame_h) if on_border else None

        if track_id not in self.track_meta:
            self.track_meta[track_id] = {
                "pos_history": deque(maxlen=8),
                "vel": (0.0, 0.0),
                "orient_history": deque(maxlen=5),
                "smoothed_orient": raw_orient,
                "probation_vecs": [],
                "aspect_ratio": current_ar if (not on_border and current_ar > 1.6) else 2.4,
                "last_seen_ts": timestamp,
                "was_on_border": on_border,
                "border_edge": border_edge,
                "outward_motion_streak": 0,
                "norm_height": None,
            }

        meta = self.track_meta[track_id]
        meta["last_seen_ts"] = timestamp

        # هموارسازی جهت بدن با رای‌گیری اکثریت (جلوگیری از پرش مرزی بین باکت‌ها)
        if orient_conf >= 0.45 or not meta["orient_history"]:
            meta["orient_history"].append(raw_orient)
        counts = Counter(meta["orient_history"])
        meta["smoothed_orient"] = counts.most_common(1)[0][0]

        # محاسبه بردار حرکت و تشخیص تغییر جهت ناگهانی در لبه تصویر (ضد سرقت track_id)
        hijack_detected = False
        if len(meta["pos_history"]) >= 2:
            prev_x, prev_y, prev_t = meta["pos_history"][-2]
            dt = max(0.02, timestamp - prev_t)
            vx = (cx - prev_x) / dt
            vy = (cy - prev_y) / dt
            old_vx, old_vy = meta["vel"]

            # بررسی اینکه آیا ترک قبلاً در لبه تصویر در حال حرکت به سمت بیرون بوده و ناگهان به سمت داخل برگشته است؟
            if on_border or meta["was_on_border"]:
                if border_edge == "LEFT_EDGE":
                    if old_vx < -25.0 and vx > 25.0:
                        hijack_detected = True
                elif border_edge == "RIGHT_EDGE":
                    if old_vx > 25.0 and vx < -25.0:
                        hijack_detected = True
                elif border_edge == "TOP_EDGE":
                    if old_vy < -20.0 and vy > 20.0:
                        hijack_detected = True
                elif border_edge == "BOTTOM_EDGE":
                    if old_vy > 20.0 and vy < -20.0:
                        hijack_detected = True

            # بررسی پرش غیرفیزیکی مکان روی یک track_id واحد
            step_dist = np.hypot(cx - meta["pos_history"][-1][0], cy - meta["pos_history"][-1][1])
            step_dt = max(0.03, timestamp - meta["pos_history"][-1][2])
            if (step_dist / step_dt) > (self.max_speed_px_per_sec * 1.8) and step_dist > 95.0:
                hijack_detected = True

            meta["vel"] = (0.6 * old_vx + 0.4 * vx, 0.6 * old_vy + 0.4 * vy)

        meta["pos_history"].append((cx, cy, timestamp))
        meta["was_on_border"] = on_border
        if border_edge:
            meta["border_edge"] = border_edge

        if not on_border and 1.7 <= current_ar <= 3.6:
            meta["aspect_ratio"] = 0.85 * meta["aspect_ratio"] + 0.15 * current_ar

        nh = estimate_normalized_height(box, keypoints, frame_h)
        if nh is not None and not on_border:
            meta["norm_height"] = (
                nh if meta["norm_height"] is None else (0.8 * meta["norm_height"] + 0.2 * nh)
            )

        meta["hijack_detected"] = hijack_detected
        return meta

    def _compute_worker_similarity(
        self,
        whitened_vec: np.ndarray,
        orientation: str,
        person_id: str,
        track_norm_height: float | None = None,
    ) -> float:
        """
        محاسبه شباهت با گالری چندزاویه‌ای کارگر با اعمال Cross-Orientation Soft Weighting
        و سازگاری قد بیومتریک.
        """
        worker = self.workers.get(person_id)
        if not worker or whitened_vec is None:
            return 0.0

        best_weighted_sim = -1.0
        buckets = worker["buckets"]

        for bucket_orient, vec_list in buckets.items():
            if not vec_list:
                continue
            compat_weight = ORIENTATION_COMPATIBILITY.get((orientation, bucket_orient), 0.80)
            # ضرب داخلی بردارهای نرمال‌شده L2 معادل شباهت کسینوسی است
            raw_sims = [float(np.dot(whitened_vec, saved_vec)) for saved_vec in vec_list]
            # ترکیب بیشترین شباهت و میانگین ۲ نمونه برتر باکت
            raw_sims.sort(reverse=True)
            top_sim = raw_sims[0] if len(raw_sims) == 1 else (0.7 * raw_sims[0] + 0.3 * raw_sims[1])
            weighted_sim = compat_weight * top_sim
            if weighted_sim > best_weighted_sim:
                best_weighted_sim = weighted_sim

        if best_weighted_sim < 0:
            return 0.0

        # اعمال جریمه در صورت اختلاف فاحش قد نرمال‌شده (کمک به تفکیک افراد هم‌لباس)
        worker_nh = worker.get("norm_height")
        if track_norm_height is not None and worker_nh is not None and worker_nh > 0:
            rel_height_diff = abs(track_norm_height - worker_nh) / worker_nh
            if rel_height_diff > 0.16:
                best_weighted_sim -= min(0.18, (rel_height_diff - 0.16) * 0.8)

        return float(best_weighted_sim)

    def _add_to_worker_gallery(
        self,
        person_id: str,
        whitened_vec: np.ndarray,
        orientation: str,
        norm_height: float | None,
        pos: tuple[float, float],
        timestamp: float,
    ):
        if person_id not in self.workers:
            self.workers[person_id] = {
                "buckets": {"FRONT": [], "BACK": [], "LEFT": [], "RIGHT": []},
                "norm_height": norm_height,
                "last_pos": pos,
                "last_seen_ts": timestamp,
                "exit_state": "ACTIVE",
                "exit_ts": 0.0,
                "exit_edge": None,
            }

        worker = self.workers[person_id]
        worker["last_pos"] = pos
        worker["last_seen_ts"] = timestamp
        worker["exit_state"] = "ACTIVE"

        if norm_height is not None:
            if worker["norm_height"] is None:
                worker["norm_height"] = norm_height
            else:
                worker["norm_height"] = 0.90 * worker["norm_height"] + 0.10 * norm_height

        if whitened_vec is None:
            return

        bucket = worker["buckets"].setdefault(orientation, [])
        if not bucket:
            bucket.append(whitened_vec.copy())
            return

        # اگر بردار جدید بسیار شبیه یکی از بردارهای فعلی باکت است، آن را با EMA نرم بهروزرسانی کن
        sims = [float(np.dot(whitened_vec, v)) for v in bucket]
        max_idx = int(np.argmax(sims))
        if sims[max_idx] > 0.86:
            updated = 0.85 * bucket[max_idx] + 0.15 * whitened_vec
            norm_u = np.linalg.norm(updated)
            if norm_u > 1e-6:
                bucket[max_idx] = updated / norm_u
        else:
            if len(bucket) < self.max_gallery_size:
                bucket.append(whitened_vec.copy())
            else:
                # جایگزینی قدیمی‌ترین نمونه با حفظ نمونه اول (Anchor)
                bucket.pop(1)
                bucket.append(whitened_vec.copy())

    def _mark_disappeared_border_workers(self, active_track_ids: set[int], timestamp: float):
        """
        علامت‌گذاری کارگرانی که ترک آن‌ها در حالی که به لبه کادر چسبیده بوده ناپدید شده است (EXITED_BORDER).
        """
        for tid in list(self.track_to_person.keys()):
            if tid not in active_track_ids:
                pid = self.track_to_person[tid]
                meta = self.track_meta.get(tid)
                worker = self.workers.get(pid)
                if worker and meta and meta.get("was_on_border"):
                    if worker.get("exit_state") != "EXITED_BORDER":
                        worker["exit_state"] = "EXITED_BORDER"
                        worker["exit_ts"] = timestamp
                        worker["exit_edge"] = meta.get("border_edge")
                # حذف قفل track_id ناپدیدشده تا اگر ByteTrack همان track_id را به فرد جدیدی در لبه داد، قفل نماند
                if meta and (timestamp - meta["last_seen_ts"]) > 0.45 and meta.get("was_on_border"):
                    del self.track_to_person[tid]

    def _prune_stale_memory(self, timestamp: float):
        """پاکسازی دوره‌ای ترک‌های قدیمی از حافظه برای جلوگیری از Memory Leak در اجرای ۲۴/۷."""
        stale_tids = [
            tid
            for tid, meta in self.track_meta.items()
            if (timestamp - meta["last_seen_ts"]) > self.track_ttl_sec
        ]
        for tid in stale_tids:
            self.track_meta.pop(tid, None)
            self.track_to_person.pop(tid, None)

    def process_frame(
        self,
        detections: list[dict],
        extracted_features: list[np.ndarray | None],
        occlusion_flags: list[bool],
        frame_w: int,
        frame_h: int,
        timestamp: float,
    ) -> dict[int, dict]:
        """
        پردازش یکپارچه تمام ترک‌های فریم جاری با الگوریتم مجارستانی، محافظ لبه کادر و قانون Margin.
        خروجی: دیکشنری track_id -> {'person_id': str, 'orientation': str, 'is_confirmed': bool, 'on_border': bool}
        """
        self.frame_counter += 1
        if self.frame_counter % 100 == 0:
            self._prune_stale_memory(timestamp)

        active_tids = {d["track_id"] for d in detections}
        self._mark_disappeared_border_workers(active_tids, timestamp)

        frame_results = {}
        unassigned_indices = []
        whitened_features = [None] * len(detections)

        # مرحله ۱: به‌روزرسانی سینماتیک، بررسی سلامت قفل ترک‌های فعلی و آماده‌سازی بردارهای Whitened
        for i, det in enumerate(detections):
            tid = det["track_id"]
            box = det["box"]
            kpts = det.get("keypoints")
            is_occ = occlusion_flags[i]
            raw_feat = extracted_features[i]

            meta = self._update_track_kinematics(tid, box, kpts, timestamp, frame_w, frame_h)
            on_border = meta["was_on_border"]
            skel_ok = is_skeleton_complete(kpts, frame_w, frame_h)
            clean_observation = (not is_occ) and (not on_border) and skel_ok

            if raw_feat is not None:
                whitened_features[i] = self._whiten_feature(raw_feat, update_mean=clean_observation)

            # اگر ترک قبلاً به یک Worker_ID متصل بوده است، بررسی کن که آیا سرقت ترک در لبه رخ داده یا خیر
            if tid in self.track_to_person:
                assigned_pid = self.track_to_person[tid]

                # شرط ۱: اگر تغییر جهت ناگهانی در لبه تصویر یا پرش غیرفیزیکی رخ داده، قفل را بشکن!
                if meta.get("hijack_detected"):
                    logger.info(f"[Border Protection] Broke hijacked track lock on Track_{tid} (was {assigned_pid})")
                    del self.track_to_person[tid]
                    meta["probation_vecs"].clear()
                # شرط ۲: بازبینی ظاهری در حالت غیرمسدود
                elif clean_observation and whitened_features[i] is not None:
                    current_sim = self._compute_worker_similarity(
                        whitened_features[i],
                        meta["smoothed_orient"],
                        assigned_pid,
                        meta["norm_height"],
                    )
                    if current_sim < self.reverify_sim_floor:
                        logger.info(
                            f"[Appearance Drift] Track_{tid} similarity to {assigned_pid} dropped to {current_sim:.2f} -> unlocking"
                        )
                        del self.track_to_person[tid]
                        meta["probation_vecs"].clear()

            if tid in self.track_to_person:
                pid = self.track_to_person[tid]
                cx = 0.5 * (box[0] + box[2])
                cy = 0.5 * (box[1] + box[3])
                if clean_observation and whitened_features[i] is not None:
                    self._add_to_worker_gallery(
                        pid,
                        whitened_features[i],
                        meta["smoothed_orient"],
                        meta["norm_height"],
                        (cx, cy),
                        timestamp,
                    )
                elif pid in self.workers:
                    self.workers[pid]["last_pos"] = (cx, cy)
                    self.workers[pid]["last_seen_ts"] = timestamp
                    self.workers[pid]["exit_state"] = "ACTIVE"

                frame_results[tid] = {
                    "person_id": pid,
                    "orientation": meta["smoothed_orient"],
                    "is_confirmed": True,
                    "on_border": on_border,
                }
            else:
                unassigned_indices.append(i)

        # لیست کارگرانی که هم‌اکنون در همین فریم به ترک دیگری قفل شده‌اند (غیرقابل اختصاص مجدد در همین فریم)
        locked_person_ids = {
            res["person_id"] for res in frame_results.values() if res["is_confirmed"]
        }
        available_workers = [
            pid for pid in self.workers.keys() if pid not in locked_person_ids
        ]

        # مرحله ۲: تخصیص بهینه مجارستانی (Hungarian Matching) برای ترک‌های بدون آیدی که ویژگی معتبر دارند
        candidate_det_indices = []
        for idx in unassigned_indices:
            tid = detections[idx]["track_id"]
            meta = self.track_meta[tid]
            on_border = meta["was_on_border"]
            w_feat = whitened_features[idx]

            # در لبه تصویر و در حالت انسداد، ویژگی را وارد صف تایید نکن
            if w_feat is not None and not occlusion_flags[idx] and not on_border:
                meta["probation_vecs"].append(w_feat)
                if len(meta["probation_vecs"]) > self.probation_frames:
                    meta["probation_vecs"].pop(0)

            if w_feat is not None and not on_border:
                candidate_det_indices.append(idx)

        if candidate_det_indices and available_workers:
            sim_matrix = np.zeros((len(candidate_det_indices), len(available_workers)), dtype=np.float32)

            for r, det_idx in enumerate(candidate_det_indices):
                det = detections[det_idx]
                tid = det["track_id"]
                box = det["box"]
                meta = self.track_meta[tid]
                cx, cy = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])

                # استفاده از میانگین بردارهای دوره آزمایشی ترک برای افزایش پایداری
                if meta["probation_vecs"]:
                    avg_vec = np.mean(meta["probation_vecs"], axis=0)
                    avg_vec = avg_vec / max(1e-6, np.linalg.norm(avg_vec))
                else:
                    avg_vec = whitened_features[det_idx]

                for c, pid in enumerate(available_workers):
                    worker = self.workers[pid]
                    dt = max(0.03, timestamp - worker["last_seen_ts"])
                    dist = np.hypot(cx - worker["last_pos"][0], cy - worker["last_pos"][1])

                    # ۱. فیلتر فیزیک سرعت حرکت (Kinematic Gate)
                    if dt < 2.5 and (dist / dt) > self.max_speed_px_per_sec and dist > 120.0:
                        sim_matrix[r, c] = -1.0
                        continue

                    sim = self._compute_worker_similarity(
                        avg_vec,
                        meta["smoothed_orient"],
                        pid,
                        meta["norm_height"],
                    )

                    # ۲. محافظت از کارگر خارج‌شده از لبه تصویر (Border Exit Cooldown)
                    if (
                        worker.get("exit_state") == "EXITED_BORDER"
                        and (timestamp - worker.get("exit_ts", 0.0)) < self.border_cooldown_sec
                    ):
                        # اگر کارگر قبلی به تازگی از همین لبه خارج شده، جریمه سخت‌گیرانه اعمال کن تا به فرد واردشده چسبانده نشود
                        sim -= 0.12

                    sim_matrix[r, c] = sim

            cost_matrix = 1.0 - sim_matrix
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for r, c in zip(row_ind, col_ind):
                det_idx = candidate_det_indices[r]
                det = detections[det_idx]
                tid = det["track_id"]
                pid = available_workers[c]
                best_sim = float(sim_matrix[r, c])

                # بررسی قانون حاشیه رقابتی (Relative Margin Rule) بین بهترین کاندیدا و دومین کاندیدای گالری
                row_sims = sorted(sim_matrix[r, :], reverse=True)
                second_best_sim = float(row_sims[1]) if len(row_sims) > 1 else 0.0
                margin_ok = (len(row_sims) == 1) or ((best_sim - second_best_sim) >= self.margin_threshold)

                if best_sim >= self.similarity_threshold and (margin_ok or best_sim >= 0.82):
                    self.track_to_person[tid] = pid
                    locked_person_ids.add(pid)
                    meta = self.track_meta[tid]
                    cx = 0.5 * (det["box"][0] + det["box"][2])
                    cy = 0.5 * (det["box"][1] + det["box"][3])

                    if not occlusion_flags[det_idx] and not meta["was_on_border"]:
                        self._add_to_worker_gallery(
                            pid,
                            whitened_features[det_idx],
                            meta["smoothed_orient"],
                            meta["norm_height"],
                            (cx, cy),
                            timestamp,
                        )
                    frame_results[tid] = {
                        "person_id": pid,
                        "orientation": meta["smoothed_orient"],
                        "is_confirmed": True,
                        "on_border": meta["was_on_border"],
                    }

        # مرحله ۳: صدور Worker_ID جدید برای ترک‌هایی که دوره آزمایشی (Probation) را خارج از لبه کادر تکمیل کرده‌اند
        for idx in unassigned_indices:
            det = detections[idx]
            tid = det["track_id"]
            if tid in frame_results:
                continue

            meta = self.track_meta[tid]
            on_border = meta["was_on_border"]

            if len(meta["probation_vecs"]) >= self.probation_frames and not on_border:
                new_pid = f"Worker_{self.next_worker_num}"
                self.next_worker_num += 1
                self.track_to_person[tid] = new_pid

                cx = 0.5 * (det["box"][0] + det["box"][2])
                cy = 0.5 * (det["box"][1] + det["box"][3])
                avg_vec = np.mean(meta["probation_vecs"], axis=0)
                avg_vec = avg_vec / max(1e-6, np.linalg.norm(avg_vec))

                self._add_to_worker_gallery(
                    new_pid,
                    avg_vec,
                    meta["smoothed_orient"],
                    meta["norm_height"],
                    (cx, cy),
                    timestamp,
                )
                logger.info(f"[New Worker Registered] Assigned {new_pid} to Track_{tid}")
                frame_results[tid] = {
                    "person_id": new_pid,
                    "orientation": meta["smoothed_orient"],
                    "is_confirmed": True,
                    "on_border": False,
                }
            else:
                # هنوز در لبه تصویر یا در دوره آزمایشی است -> شناسه موقت بدون آلوده کردن بانک گالری
                frame_results[tid] = {
                    "person_id": f"Pending_{tid}",
                    "orientation": meta["smoothed_orient"],
                    "is_confirmed": False,
                    "on_border": on_border,
                }

        return frame_results

    def get_track_expected_aspect_ratio(self, track_id: int) -> float | None:
        meta = self.track_meta.get(track_id)
        return meta["aspect_ratio"] if meta else None

    def needs_feature_extraction(self, track_id: int, is_occluded: bool, on_border: bool, reid_interval: int = 10) -> bool:
        """
        تعیین هوشمند اینکه آیا در فریم جاری نیاز به اجرای شبکه OSNet برای این ترک هست یا خیر.
        """
        if is_occluded or on_border:
            return False
        if track_id not in self.track_to_person:
            return True
        return (self.frame_counter % max(1, reid_interval)) == (track_id % max(1, reid_interval))

    def identify_and_update(
        self,
        track_id: int,
        feature_vector: np.ndarray | None,
        is_occluded: bool,
        active_track_ids: list[int],
        box: list[int] | None = None,
        keypoints: np.ndarray | None = None,
        frame_w: int = 1280,
        frame_h: int = 720,
        timestamp: float = 0.0,
    ) -> str:
        """متد سازگار با فراخوانی‌های تک‌ترکی در نسخه‌های قبلی."""
        dummy_box = box if box is not None else [100, 100, 180, 300]
        res = self.process_frame(
            detections=[{"track_id": track_id, "box": dummy_box, "conf": 0.9, "keypoints": keypoints}],
            extracted_features=[feature_vector],
            occlusion_flags=[is_occluded],
            frame_w=frame_w,
            frame_h=frame_h,
            timestamp=timestamp,
        )
        return res[track_id]["person_id"]


IdentityManager = EnterpriseIdentityManager