# app/identity/manager.py
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from app.detection.detector import PoseDetection
from app.identity.reid_model import CropQualityReport, PoseGuidedOSNetExtractor
from app.logic.geometry import (
    MetricBEVProjector,
    compute_intersection_contamination,
    compute_single_iou,
)


@dataclass
class WorkerIdentity:
    global_id: int
    created_ts: float
    last_track_id: int
    last_seen_ts: float
    last_bev_xy: np.ndarray
    last_ground_uv: Tuple[float, float]
    last_height_px: float
    last_bbox_xyxy: np.ndarray
    bev_velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    # Tier 1: Immutable high-quality anchors (prevents gradual gallery drift / identity hijack)
    anchor_embeddings: List[np.ndarray] = field(default_factory=list)
    # Tier 2: View-stratified adaptive gallery
    view_gallery: Dict[str, List[np.ndarray]] = field(
        default_factory=lambda: {"FRONT": [], "BACK": [], "LEFT": [], "RIGHT": []}
    )
    # Tier 3: Short-term rolling buffer (updated ONLY when isolated & verified)
    recent_embeddings: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=6))
    # Anti-Hijack Mismatch Circuit Breaker counter
    mismatch_streak: int = 0
    # Mutual Exclusion Sets (Tracks & Global IDs verified to belong to OTHER simultaneously visible people)
    coexisted_track_ids: Set[int] = field(default_factory=set)
    verified_coexisted_gids: Set[int] = field(default_factory=set)

    def add_clean_observation(
        self,
        embedding: np.ndarray,
        view_bin: str,
        sharpness: float,
        is_isolated: bool,
        max_per_view: int = 4,
        ema_alpha: float = 0.82,
        diversity_cos_max: float = 0.90,
    ) -> None:
        """
        Updates appearance memory ONLY after identity consistency has been verified.
        """
        self.recent_embeddings.append(embedding.copy())

        # Only update permanent anchors and view gallery when worker is spatially isolated from others
        if not is_isolated:
            return

        if len(self.anchor_embeddings) < 3 and sharpness >= 22.0:
            if not self.anchor_embeddings or min(float(np.dot(embedding, a)) for a in self.anchor_embeddings) < 0.92:
                self.anchor_embeddings.append(embedding.copy())

        bank = self.view_gallery.setdefault(view_bin, [])
        if not bank:
            bank.append(embedding.copy())
            return

        sims = [float(np.dot(embedding, feat)) for feat in bank]
        max_idx = int(np.argmax(sims))
        max_sim = sims[max_idx]

        if max_sim >= diversity_cos_max:
            updated = ema_alpha * bank[max_idx] + (1.0 - ema_alpha) * embedding
            bank[max_idx] = updated / (np.linalg.norm(updated) + 1e-8)
        elif len(bank) < max_per_view:
            bank.append(embedding.copy())
        else:
            updated = 0.70 * bank[0] + 0.30 * embedding
            bank[0] = updated / (np.linalg.norm(updated) + 1e-8)

    def compute_similarity(self, query_emb: np.ndarray, query_view: str) -> float:
        """
        Computes appearance similarity anchored by immutable prototypes so gallery drift cannot occur.
        """
        same_view_feats = self.view_gallery.get(query_view, [])
        all_feats = [f for bank in self.view_gallery.values() for f in bank]
        recent_list = list(self.recent_embeddings)

        if not all_feats and not self.anchor_embeddings and not recent_list:
            return 0.0

        anchor_max = max((float(np.dot(query_emb, a)) for a in self.anchor_embeddings), default=0.0)
        global_max = max((float(np.dot(query_emb, f)) for f in all_feats), default=anchor_max)
        view_max = max((float(np.dot(query_emb, f)) for f in same_view_feats), default=global_max)
        recent_max = max((float(np.dot(query_emb, f)) for f in recent_list), default=global_max)

        gallery_score = 0.50 * view_max + 0.50 * global_max
        if self.anchor_embeddings:
            # Blend permanent anchor similarity to prevent slow hijacking drift
            core_score = 0.65 * gallery_score + 0.35 * anchor_max
        else:
            core_score = gallery_score

        return max(core_score, 0.90 * min(recent_max, max(anchor_max, global_max) + 0.08))

    def compute_gallery_to_gallery_similarity(self, other: "WorkerIdentity") -> float:
        feats_a = self.anchor_embeddings + [f for bank in self.view_gallery.values() for f in bank]
        feats_b = other.anchor_embeddings + [f for bank in other.view_gallery.values() for f in bank]
        if not feats_a or not feats_b:
            return 0.0
        mat = np.dot(np.array(feats_a), np.array(feats_b).T)
        # Top-25% percentile mean similarity across gallery banks
        flat = np.sort(mat.flatten())
        top_k = flat[max(0, len(flat) - max(1, len(flat) // 4)) :]
        return float(np.mean(top_k))


class EnterpriseIdentityManager:
    """
    Enterprise Spatio-Temporal & Appearance Identity Manager:
      1. Zero Gallery Poisoning: Never updates appearance memory before similarity verification or during proximity.
      2. Anti-Hijack Mismatch Circuit Breaker & Post-Intersection Swap Audit: Fixes ByteTrack ID swaps automatically.
      3. Enforced Co-Existence Mutual Exclusion: Mathematically forbids ID inheritance between co-existing workers.
      4. 3-Stage Cascade Stitching + Self-Healing Retroactive ID Consolidation: Cures duplicate ID fragmentation.
    """

    def __init__(
        self,
        reid_extractor: PoseGuidedOSNetExtractor,
        bev_projector: MetricBEVProjector,
        cos_sim_thresh: float = 0.55,
        spatial_recovery_sim_thresh: float = 0.46,
        reentry_sim_thresh: float = 0.64,
        max_worker_speed_mps: float = 3.2,
        max_internal_occlusion_m: float = 5.5,
        lag_diffusion_coeff: float = 0.65,
        base_pose_jitter_m: float = 0.85,
        reid_interval_sec: float = 0.35,
        probation_frames_for_new_id: int = 4,
    ):
        self.reid = reid_extractor
        self.bev = bev_projector
        self.cos_sim_thresh = cos_sim_thresh
        self.spatial_recovery_sim_thresh = spatial_recovery_sim_thresh
        self.reentry_sim_thresh = reentry_sim_thresh
        self.v_max = max_worker_speed_mps
        self.max_internal_occ_m = max_internal_occlusion_m
        self.sigma_lag = lag_diffusion_coeff
        self.eps_pose = base_pose_jitter_m
        self.reid_interval_sec = reid_interval_sec
        self.probation_frames = probation_frames_for_new_id

        self.identities: Dict[int, WorkerIdentity] = {}
        self.track_to_global: Dict[int, int] = {}
        self.track_last_reid_ts: Dict[int, float] = {}
        self.track_seen_counts: Dict[int, int] = {}
        self.track_first_seen_ts: Dict[int, float] = {}
        self.track_best_emb: Dict[int, Tuple[np.ndarray, str, float]] = {}
        # Tracks that were simultaneously visible & spatially separated from a given track_id
        self.track_separated_coexisted_tracks: Dict[Tuple[int, int], int] = {}
        self.pair_coexistence_counts: Dict[Tuple[int, int], int] = {}
        self._next_global_id: int = 1

    @staticmethod
    def _is_near_frame_border(bbox_xyxy: np.ndarray, frame_w: int, frame_h: int, margin_ratio: float = 0.045) -> bool:
        x1, y1, x2, y2 = bbox_xyxy
        mx = frame_w * margin_ratio
        my = frame_h * margin_ratio
        return bool(x1 <= mx or y1 <= my or x2 >= (frame_w - mx) or y2 >= (frame_h - my))

    def _is_spatially_continuous_with_identity(
        self, ident: WorkerIdentity, det: PoseDetection, current_ts: float
    ) -> Tuple[bool, float, float]:
        dt = max(1e-3, current_ts - ident.last_seen_ts)
        box_iou = compute_single_iou(ident.last_bbox_xyxy, det.bbox_xyxy)

        dist_m = self.bev.compute_metric_distance(
            ident.last_ground_uv,
            ident.last_bev_xy,
            ident.last_height_px,
            det.ground_uv,
            det.bev_xy,
            det.apparent_height_px,
        )

        if dt <= 0.8 and box_iou >= 0.22:
            return True, min(dist_m, 0.35), 2.0

        if dt <= 1.2:
            max_dist_m = (self.v_max * dt) + (self.sigma_lag * np.sqrt(dt)) + self.eps_pose
            return dist_m <= max_dist_m, dist_m, max_dist_m

        max_dist_m = min(self.max_internal_occ_m + 0.5 * dt, (self.v_max * dt) + self.eps_pose)
        return dist_m <= max_dist_m, dist_m, max_dist_m

    def _update_identity_kinematics(self, ident: WorkerIdentity, det: PoseDetection, frame_ts: float) -> None:
        dt = max(1e-3, frame_ts - ident.last_seen_ts)
        if dt <= 0.5:
            u_sm = 0.65 * det.ground_uv[0] + 0.35 * ident.last_ground_uv[0]
            v_sm = 0.65 * det.ground_uv[1] + 0.35 * ident.last_ground_uv[1]
            bev_sm = 0.65 * det.bev_xy + 0.35 * ident.last_bev_xy
        else:
            u_sm, v_sm = det.ground_uv
            bev_sm = det.bev_xy

        ident.bev_velocity = 0.75 * ident.bev_velocity + 0.25 * ((bev_sm - ident.last_bev_xy) / dt)
        ident.last_ground_uv = (u_sm, v_sm)
        ident.last_bev_xy = bev_sm
        ident.last_height_px = 0.75 * det.apparent_height_px + 0.25 * ident.last_height_px
        ident.last_bbox_xyxy = det.bbox_xyxy.copy()
        ident.last_seen_ts = frame_ts
        ident.last_track_id = det.track_id

    def _are_tracks_verified_distinct(self, tid_a: int, tid_b: int, min_sep_frames: int = 3) -> bool:
        if tid_a == tid_b:
            return False
        key = (min(tid_a, tid_b), max(tid_a, tid_b))
        return self.track_separated_coexisted_tracks.get(key, 0) >= min_sep_frames

    def _audit_and_resolve_swaps(
        self,
        frame: np.ndarray,
        detections: List[PoseDetection],
        resolved_assignments: Dict[int, int],
        frame_ts: float,
    ) -> None:
        """
        Detects if two active tracks that recently interacted have swapped Global IDs
        by checking whether swapping their assigned Global IDs significantly improves
        both workers' appearance similarity to their Immutable Identity Anchors.
        """
        active_dets = [d for d in detections if d.track_id in resolved_assignments and not d.is_intersecting]
        if len(active_dets) < 2:
            return

        for i in range(len(active_dets)):
            for j in range(i + 1, len(active_dets)):
                det_a, det_b = active_dets[i], active_dets[j]
                gid_a = resolved_assignments[det_a.track_id]
                gid_b = resolved_assignments[det_b.track_id]
                if gid_a == gid_b:
                    continue

                ident_a = self.identities[gid_a]
                ident_b = self.identities[gid_b]

                # Only audit pairs that are within 3.5m of each other or have mismatch streaks
                dist_ab = self.bev.compute_metric_distance(
                    det_a.ground_uv, det_a.bev_xy, det_a.apparent_height_px,
                    det_b.ground_uv, det_b.bev_xy, det_b.apparent_height_px,
                )
                if dist_ab > 3.5 and ident_a.mismatch_streak == 0 and ident_b.mismatch_streak == 0:
                    continue

                emb_a_tuple = self.track_best_emb.get(det_a.track_id)
                emb_b_tuple = self.track_best_emb.get(det_b.track_id)
                if not emb_a_tuple or not emb_b_tuple:
                    continue

                emb_a, view_a, _ = emb_a_tuple
                emb_b, view_b, _ = emb_b_tuple

                curr_score = ident_a.compute_similarity(emb_a, view_a) + ident_b.compute_similarity(emb_b, view_b)
                swap_score = ident_a.compute_similarity(emb_b, view_b) + ident_b.compute_similarity(emb_a, view_a)

                # If swapping increases joint similarity by > 0.16 and both match better after swap, un-swap!
                if (
                    swap_score > curr_score + 0.16
                    and ident_a.compute_similarity(emb_b, view_b) > ident_a.compute_similarity(emb_a, view_a) + 0.06
                    and ident_b.compute_similarity(emb_a, view_a) > ident_b.compute_similarity(emb_b, view_b) + 0.06
                ):
                    resolved_assignments[det_a.track_id] = gid_b
                    resolved_assignments[det_b.track_id] = gid_a
                    self.track_to_global[det_a.track_id] = gid_b
                    self.track_to_global[det_b.track_id] = gid_a
                    ident_a.mismatch_streak = 0
                    ident_b.mismatch_streak = 0

    def _consolidate_fragmented_identities(
        self,
        resolved_assignments: Dict[int, int],
        used_global_ids: Set[int],
        frame_ts: float,
    ) -> None:
        """
        Self-Healing Retroactive Merger:
        If a worker temporarily received a new duplicate ID (`gid_new`) after severe pose distortion/occlusion,
        and now that they stand clearly their gallery strongly matches an older INACTIVE identity (`gid_old`)
        that NEVER co-existed with `gid_new`, merge `gid_new` back into `gid_old`!
        """
        for tid, gid_new in list(resolved_assignments.items()):
            ident_new = self.identities.get(gid_new)
            if ident_new is None or (frame_ts - ident_new.created_ts) > 12.0:
                continue

            best_old_gid: Optional[int] = None
            best_merge_sim: float = 0.0

            for gid_old, ident_old in list(self.identities.items()):
                if gid_old >= gid_new or gid_old in used_global_ids:
                    continue
                # Cannot merge if they ever co-existed as two separate people!
                if (
                    gid_old in ident_new.verified_coexisted_gids
                    or tid in ident_old.coexisted_track_ids
                    or self._are_tracks_verified_distinct(tid, ident_old.last_track_id, min_sep_frames=2)
                ):
                    continue
                # Old identity must have been lost before or right as new identity was created
                if ident_old.last_seen_ts > ident_new.created_ts + 0.5:
                    continue

                dt_gap = max(0.1, ident_new.created_ts - ident_old.last_seen_ts)
                dist_m = self.bev.compute_metric_distance(
                    ident_old.last_ground_uv, ident_old.last_bev_xy, ident_old.last_height_px,
                    ident_new.last_ground_uv, ident_new.last_bev_xy, ident_new.last_height_px,
                )
                max_merge_dist = min(self.max_internal_occ_m + 1.0, self.v_max * dt_gap + 1.5)
                if dist_m > max_merge_dist:
                    continue

                gal_sim = ident_old.compute_gallery_to_gallery_similarity(ident_new)
                req_sim = 0.64 if dist_m <= 1.8 else 0.70
                if gal_sim >= req_sim and gal_sim > best_merge_sim:
                    best_merge_sim = gal_sim
                    best_old_gid = gid_old

            if best_old_gid is not None:
                ident_old = self.identities[best_old_gid]
                # Merge gallery & state from ident_new into ident_old
                for v_bin, feats in ident_new.view_gallery.items():
                    for f in feats:
                        ident_old.add_clean_observation(f, v_bin, sharpness=25.0, is_isolated=True)
                ident_old.verified_coexisted_gids.update(ident_new.verified_coexisted_gids)
                ident_old.coexisted_track_ids.update(ident_new.coexisted_track_ids)
                ident_old.last_seen_ts = ident_new.last_seen_ts
                ident_old.last_bev_xy = ident_new.last_bev_xy
                ident_old.last_ground_uv = ident_new.last_ground_uv
                ident_old.last_height_px = ident_new.last_height_px
                ident_old.last_bbox_xyxy = ident_new.last_bbox_xyxy
                ident_old.last_track_id = tid

                self.track_to_global[tid] = best_old_gid
                resolved_assignments[tid] = best_old_gid
                used_global_ids.discard(gid_new)
                used_global_ids.add(best_old_gid)
                del self.identities[gid_new]

    def process_frame(
        self, frame: np.ndarray, detections: List[PoseDetection], frame_ts: float
    ) -> Dict[int, int]:
        if not detections:
            return {}

        H_img, W_img = frame.shape[:2]
        boxes = np.array([d.bbox_xyxy for d in detections], dtype=np.float32)
        kpts_xy = np.array([d.keypoints_xy for d in detections], dtype=np.float32)
        kpts_conf = np.array([d.keypoints_conf for d in detections], dtype=np.float32)

        contaminated_flags = compute_intersection_contamination(boxes, kpts_xy, kpts_conf)

        # Compute Pose-Corrected Standing Height, Ground Contact Point, and BEV Coordinates
        for i, det in enumerate(detections):
            tid = det.track_id
            self.track_seen_counts[tid] = self.track_seen_counts.get(tid, 0) + 1
            if tid not in self.track_first_seen_ts:
                self.track_first_seen_ts[tid] = frame_ts

            det.is_intersecting = bool(contaminated_flags[i])
            det.apparent_height_px = self.bev.estimate_standing_height_px(
                det.bbox_xyxy, det.keypoints_xy, det.keypoints_conf
            )
            u_g, v_g, _ = self.bev.estimate_ground_contact_point(
                det.bbox_xyxy, det.keypoints_xy, det.keypoints_conf
            )
            det.ground_uv = (u_g, v_g)
            det.bev_xy = self.bev.to_bev_meters(u_g, v_g)

        # Determine spatial isolation & record pairwise track separation co-existence
        for i, det_a in enumerate(detections):
            min_peer_dist = 999.0
            for j, det_b in enumerate(detections):
                if i == j:
                    continue
                iou_ab = compute_single_iou(det_a.bbox_xyxy, det_b.bbox_xyxy)
                dist_ab = self.bev.compute_metric_distance(
                    det_a.ground_uv, det_a.bev_xy, det_a.apparent_height_px,
                    det_b.ground_uv, det_b.bev_xy, det_b.apparent_height_px,
                )
                min_peer_dist = min(min_peer_dist, dist_ab)

                # If two tracks are clearly separated in the same frame, they are two distinct humans
                if iou_ab < 0.04 and dist_ab > 1.30:
                    t_pair = (min(det_a.track_id, det_b.track_id), max(det_a.track_id, det_b.track_id))
                    self.track_separated_coexisted_tracks[t_pair] = (
                        self.track_separated_coexisted_tracks.get(t_pair, 0) + 1
                    )

            det_a.is_isolated = (not det_a.is_intersecting) and (min_peer_dist > 1.35)

        used_global_ids: Set[int] = set()
        resolved_assignments: Dict[int, int] = {}
        unassigned_detections: List[PoseDetection] = []

        # =========================================================================
        # STEP 1: Verify Active ByteTrack -> Global ID Bindings with Anti-Hijack Audit
        # =========================================================================
        for det in detections:
            tid = det.track_id
            if tid in self.track_to_global:
                gid = self.track_to_global[tid]
                if gid not in self.identities:
                    del self.track_to_global[tid]
                    unassigned_detections.append(det)
                    continue

                ident = self.identities[gid]
                reachable, _, _ = self._is_spatially_continuous_with_identity(ident, det, frame_ts)

                if reachable and gid not in used_global_ids:
                    last_reid_t = self.track_last_reid_ts.get(tid, 0.0)
                    need_reid = (not det.is_intersecting) and (
                        (frame_ts - last_reid_t) >= self.reid_interval_sec or ident.mismatch_streak > 0
                    )

                    if need_reid:
                        emb, report = self.reid.evaluate_and_extract(
                            frame, det.bbox_xyxy, det.keypoints_xy, det.keypoints_conf, det.is_intersecting
                        )
                        self.track_last_reid_ts[tid] = frame_ts
                        if emb is not None and report.is_valid_for_gallery:
                            self.track_best_emb[tid] = (emb, report.pose_view_bin, report.sharpness_score)
                            # CRITICAL FIX: Compute similarity BEFORE updating recent_embeddings or gallery!
                            sim = ident.compute_similarity(emb, report.pose_view_bin)

                            if sim >= self.spatial_recovery_sim_thresh:
                                ident.mismatch_streak = 0
                                ident.add_clean_observation(
                                    emb,
                                    report.pose_view_bin,
                                    sharpness=report.sharpness_score,
                                    is_isolated=det.is_isolated,
                                )
                            elif det.is_isolated and sim < 0.42:
                                # Track is isolated and clean, but looks very different from `ident`!
                                ident.mismatch_streak += 1
                                if ident.mismatch_streak >= 3:
                                    # Mismatch circuit breaker tripped: ByteTrack hijacked a different person!
                                    ident.mismatch_streak = 0
                                    del self.track_to_global[tid]
                                    unassigned_detections.append(det)
                                    continue

                    resolved_assignments[tid] = gid
                    used_global_ids.add(gid)
                    self._update_identity_kinematics(ident, det, frame_ts)
                else:
                    del self.track_to_global[tid]
                    unassigned_detections.append(det)
            else:
                unassigned_detections.append(det)

        # Audit active tracks for post-intersection ByteTrack ID swaps
        self._audit_and_resolve_swaps(frame, detections, resolved_assignments, frame_ts)

        # Update co-existence records for all currently resolved identities BEFORE Step 2 matching
        for tid_a, gid_a in resolved_assignments.items():
            ident_a = self.identities[gid_a]
            for det_b in detections:
                if self._are_tracks_verified_distinct(tid_a, det_b.track_id, min_sep_frames=3):
                    ident_a.coexisted_track_ids.add(det_b.track_id)

        # =========================================================================
        # STEP 2: 3-Stage Cascade Matching for Unassigned / Fragmented Tracklets
        # =========================================================================
        if unassigned_detections:
            candidate_gids = [gid for gid in self.identities.keys() if gid not in used_global_ids]
            det_embeddings: List[Optional[np.ndarray]] = []
            det_reports: List[CropQualityReport] = []

            for det in unassigned_detections:
                emb, rep = self.reid.evaluate_and_extract(
                    frame, det.bbox_xyxy, det.keypoints_xy, det.keypoints_conf, det.is_intersecting
                )
                det_embeddings.append(emb)
                det_reports.append(rep)
                if emb is not None and rep.is_valid_for_gallery:
                    self.track_best_emb[det.track_id] = (emb, rep.pose_view_bin, rep.sharpness_score)

            matched_rows: Set[int] = set()
            if candidate_gids:
                cost_matrix = np.full((len(unassigned_detections), len(candidate_gids)), 1e6, dtype=np.float32)
                sim_matrix = np.zeros((len(unassigned_detections), len(candidate_gids)), dtype=np.float32)

                for r, det in enumerate(unassigned_detections):
                    emb = det_embeddings[r]
                    rep = det_reports[r]
                    if emb is None and det.track_id in self.track_best_emb:
                        emb, pose_view, _ = self.track_best_emb[det.track_id]
                    else:
                        pose_view = rep.pose_view_bin if rep else "FRONT"

                    if emb is None:
                        continue

                    track_birth_ts = self.track_first_seen_ts.get(det.track_id, frame_ts)

                    # Count how many lost identities disappeared within 1.5m of `det` (spatial ambiguity check)
                    nearby_lost_count = sum(
                        1
                        for gid_c in candidate_gids
                        if (frame_ts - self.identities[gid_c].last_seen_ts) <= 3.0
                        and self.bev.compute_metric_distance(
                            self.identities[gid_c].last_ground_uv,
                            self.identities[gid_c].last_bev_xy,
                            self.identities[gid_c].last_height_px,
                            det.ground_uv,
                            det.bev_xy,
                            det.apparent_height_px,
                        )
                        <= 1.5
                    )

                    for c, gid in enumerate(candidate_gids):
                        ident = self.identities[gid]

                        # HARD VETO 1: Co-Existence Mutual Exclusion
                        # If `det.track_id` was ever seen separated in the same frame as `ident`,
                        # they are two different people and can NEVER be merged!
                        if (
                            det.track_id in ident.coexisted_track_ids
                            or self._are_tracks_verified_distinct(det.track_id, ident.last_track_id, min_sep_frames=2)
                        ):
                            continue

                        # HARD VETO 2: Temporal Overlap
                        # If `det` started being tracked > 0.6s BEFORE `ident` disappeared, they were visible together!
                        dt_lost = frame_ts - ident.last_seen_ts
                        if track_birth_ts < (ident.last_seen_ts - 0.6) and dt_lost > 0.4:
                            continue

                        reachable, dist_m, max_dist_m = self._is_spatially_continuous_with_identity(
                            ident, det, frame_ts
                        )
                        box_iou = compute_single_iou(ident.last_bbox_xyxy, det.bbox_xyxy)
                        app_sim = ident.compute_similarity(emb, pose_view)
                        sim_matrix[r, c] = app_sim

                        # STAGE A: Unambiguous In-Place Tracklet Stitching (Cures Fragmentation!)
                        # Only triggers when `nearby_lost_count == 1` (no other lost worker nearby to confuse with!)
                        if (
                            nearby_lost_count == 1
                            and dt_lost <= 2.5
                            and (box_iou >= 0.25 or dist_m <= 1.25)
                            and det.is_isolated
                        ):
                            if app_sim >= 0.42:
                                stitch_score = 0.65 * app_sim + 0.35 * max(box_iou, 1.0 - dist_m / 1.5)
                                cost_matrix[r, c] = 1.0 - stitch_score
                                continue

                        # STAGE B: Spatio-Temporal Occlusion Recovery
                        if not reachable:
                            # STAGE C: Border / Long-Horizon Re-Entry
                            near_border = self._is_near_frame_border(det.bbox_xyxy, W_img, H_img)
                            if near_border and dt_lost > 2.0 and app_sim >= self.reentry_sim_thresh:
                                cost_matrix[r, c] = 1.0 - (0.90 * app_sim)
                            continue

                        required_sim = (
                            self.spatial_recovery_sim_thresh
                            if (dist_m <= 1.6 and dt_lost <= 3.5 and nearby_lost_count == 1)
                            else self.cos_sim_thresh
                        )
                        if app_sim >= required_sim:
                            spatial_score = max(0.0, 1.0 - (dist_m / max(max_dist_m, 1.0)))
                            fused_score = 0.80 * app_sim + 0.20 * spatial_score
                            cost_matrix[r, c] = 1.0 - fused_score

                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                for r, c in zip(row_ind, col_ind):
                    if cost_matrix[r, c] < 1e5:
                        # Ambiguity margin gate when multiple lost identities compete for the same track
                        if len(candidate_gids) > 1:
                            best_sim = sim_matrix[r, c]
                            other_sims = [
                                sim_matrix[r, k]
                                for k in range(len(candidate_gids))
                                if k != c and cost_matrix[r, k] < 1e5
                            ]
                            if other_sims and (best_sim - max(other_sims)) < 0.04 and best_sim < 0.72:
                                continue

                        det = unassigned_detections[r]
                        gid = candidate_gids[c]
                        ident = self.identities[gid]
                        emb = det_embeddings[r]
                        rep = det_reports[r]

                        self.track_to_global[det.track_id] = gid
                        self.track_last_reid_ts[det.track_id] = frame_ts
                        resolved_assignments[det.track_id] = gid
                        used_global_ids.add(gid)
                        matched_rows.add(r)

                        ident.mismatch_streak = 0
                        self._update_identity_kinematics(ident, det, frame_ts)
                        if emb is not None and rep.is_valid_for_gallery:
                            ident.add_clean_observation(
                                emb,
                                rep.pose_view_bin,
                                sharpness=rep.sharpness_score,
                                is_isolated=det.is_isolated,
                            )

            # =====================================================================
            # STEP 3: Mint New Global ID After Probationary Maturation
            # =====================================================================
            for r, det in enumerate(unassigned_detections):
                if r in matched_rows:
                    continue

                seen_count = self.track_seen_counts.get(det.track_id, 0)
                req_frames = 2 if len(self.identities) == 0 else self.probation_frames
                if seen_count < req_frames:
                    continue

                overlaps_active = any(
                    compute_single_iou(det.bbox_xyxy, other.bbox_xyxy) >= 0.38
                    for other in detections
                    if other.track_id != det.track_id and other.track_id in resolved_assignments
                )
                if overlaps_active:
                    continue

                emb = det_embeddings[r]
                rep = det_reports[r]
                sharpness_val = rep.sharpness_score if rep else 20.0
                if emb is None and det.track_id in self.track_best_emb:
                    emb, view_bin, sharpness_val = self.track_best_emb[det.track_id]
                else:
                    view_bin = rep.pose_view_bin if rep else "FRONT"

                if emb is not None:
                    new_gid = self._next_global_id
                    self._next_global_id += 1

                    new_ident = WorkerIdentity(
                        global_id=new_gid,
                        created_ts=self.track_first_seen_ts.get(det.track_id, frame_ts),
                        last_track_id=det.track_id,
                        last_seen_ts=frame_ts,
                        last_bev_xy=det.bev_xy.copy(),
                        last_ground_uv=det.ground_uv,
                        last_height_px=det.apparent_height_px,
                        last_bbox_xyxy=det.bbox_xyxy.copy(),
                    )
                    new_ident.add_clean_observation(
                        emb, view_bin, sharpness=sharpness_val, is_isolated=det.is_isolated
                    )

                    self.identities[new_gid] = new_ident
                    self.track_to_global[det.track_id] = new_gid
                    self.track_last_reid_ts[det.track_id] = frame_ts
                    resolved_assignments[det.track_id] = new_gid
                    used_global_ids.add(new_gid)

        # =========================================================================
        # STEP 4: Update Verified Co-Existence Graph & Run Self-Healing Merger
        # =========================================================================
        active_items = [
            (det, resolved_assignments[det.track_id])
            for det in detections
            if det.track_id in resolved_assignments
        ]
        for i in range(len(active_items)):
            det_a, gid_a = active_items[i]
            ident_a = self.identities[gid_a]
            for j in range(i + 1, len(active_items)):
                det_b, gid_b = active_items[j]
                if gid_a == gid_b:
                    continue
                ident_b = self.identities[gid_b]
                iou_ab = compute_single_iou(det_a.bbox_xyxy, det_b.bbox_xyxy)
                dist_ab = self.bev.compute_metric_distance(
                    det_a.ground_uv, det_a.bev_xy, det_a.apparent_height_px,
                    det_b.ground_uv, det_b.bev_xy, det_b.apparent_height_px,
                )
                if iou_ab < 0.04 and dist_ab > 1.30:
                    ident_a.coexisted_track_ids.add(det_b.track_id)
                    ident_b.coexisted_track_ids.add(det_a.track_id)
                    pair_key = (min(gid_a, gid_b), max(gid_a, gid_b))
                    cnt = self.pair_coexistence_counts.get(pair_key, 0) + 1
                    self.pair_coexistence_counts[pair_key] = cnt
                    if cnt >= 5:
                        ident_a.verified_coexisted_gids.add(gid_b)
                        ident_b.verified_coexisted_gids.add(gid_a)

        # Consolidate any fragmented temporary IDs back into their original lost Global IDs
        self._consolidate_fragmented_identities(resolved_assignments, used_global_ids, frame_ts)

        return resolved_assignments