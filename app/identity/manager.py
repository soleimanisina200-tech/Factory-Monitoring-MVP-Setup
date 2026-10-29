# app/identity/manager.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from app.detection.detector import PoseDetection
from app.identity.reid_model import CropQualityReport, PoseGuidedOSNetExtractor
from app.logic.geometry import (
    MetricBEVProjector,
    compute_intersection_contamination,
)


@dataclass
class WorkerIdentity:
    global_id: int
    last_track_id: int
    last_seen_ts: float
    last_bev_xy: np.ndarray  # Real-world (X, Y) in meters
    bev_velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    # Multi-view gallery: stores up to K diverse embeddings per canonical pose view
    view_gallery: Dict[str, List[np.ndarray]] = field(
        default_factory=lambda: {"FRONT": [], "BACK": [], "LEFT": [], "RIGHT": []}
    )
    anthropometric_ratio: Optional[float] = None

    def update_gallery(
        self,
        embedding: np.ndarray,
        view_bin: str,
        max_per_view: int = 4,
        ema_alpha: float = 0.85,
        diversity_cos_max: float = 0.94,
    ) -> None:
        bank = self.view_gallery.setdefault(view_bin, [])
        if not bank:
            bank.append(embedding.copy())
            return

        sims = [float(np.dot(embedding, feat)) for feat in bank]
        max_idx = int(np.argmax(sims))
        max_sim = sims[max_idx]

        # If very similar to an existing prototype in this view, smooth via EMA
        if max_sim >= diversity_cos_max:
            updated = ema_alpha * bank[max_idx] + (1.0 - ema_alpha) * embedding
            bank[max_idx] = updated / (np.linalg.norm(updated) + 1e-8)
        elif len(bank) < max_per_view:
            bank.append(embedding.copy())
        else:
            # Replace the oldest slot with EMA blend
            updated = 0.5 * bank[0] + 0.5 * embedding
            bank[0] = updated / (np.linalg.norm(updated) + 1e-8)

    def compute_similarity(self, query_emb: np.ndarray, query_view: str) -> float:
        """
        Computes view-aware cosine similarity: prioritizes matching against the same pose view
        while taking a cross-view max fallback.
        """
        same_view_feats = self.view_gallery.get(query_view, [])
        all_feats = [f for bank in self.view_gallery.values() for f in bank]
        if not all_feats:
            return 0.0

        global_max = max(float(np.dot(query_emb, f)) for f in all_feats)
        if same_view_feats:
            view_max = max(float(np.dot(query_emb, f)) for f in same_view_feats)
            return 0.65 * view_max + 0.35 * global_max
        return global_max


class EnterpriseIdentityManager:
    """
    Zero-Data Identity Manager featuring:
      1. Homography BEV Physics Guard (meters/second + variable FPS Brownian cone).
      2. Intersection Lockout (zero gallery updates during overlaps).
      3. Global Hungarian Anti-Cloning Assignment (strictly 1 Global ID per frame).
    """

    def __init__(
        self,
        reid_extractor: PoseGuidedOSNetExtractor,
        bev_projector: MetricBEVProjector,
        cos_sim_thresh: float = 0.68,
        spatial_recovery_sim_thresh: float = 0.58,
        max_worker_speed_mps: float = 2.8,    # meters / second
        lag_diffusion_coeff: float = 0.75,    # meters / sqrt(second) for variable FPS
        base_pose_jitter_m: float = 0.65,     # baseline footprint tolerance in meters
        reid_interval_sec: float = 0.40,      # Time-based interval (FPS-agnostic!)
    ):
        self.reid = reid_extractor
        self.bev = bev_projector
        self.cos_sim_thresh = cos_sim_thresh
        self.spatial_recovery_sim_thresh = spatial_recovery_sim_thresh
        self.v_max = max_worker_speed_mps
        self.sigma_lag = lag_diffusion_coeff
        self.eps_pose = base_pose_jitter_m
        self.reid_interval_sec = reid_interval_sec

        self.identities: Dict[int, WorkerIdentity] = {}
        self.track_to_global: Dict[int, int] = {}
        self.track_last_reid_ts: Dict[int, float] = {}
        self._next_global_id: int = 1

    def _is_physically_reachable(
        self, identity: WorkerIdentity, target_bev_xy: np.ndarray, current_ts: float
    ) -> Tuple[bool, float, float]:
        """
        BEV Metric Physics Guard that dynamically adapts to variable/low FPS via wall-clock delta.
        Returns: (is_reachable, distance_meters, max_allowed_meters)
        """
        dt = max(1e-3, current_ts - identity.last_seen_ts)
        dist_m = float(np.linalg.norm(target_bev_xy - identity.last_bev_xy))

        # Ballistic speed limit + Brownian network-lag diffusion cone
        max_dist_m = (self.v_max * dt) + (self.sigma_lag * np.sqrt(dt)) + self.eps_pose
        return dist_m <= max_dist_m, dist_m, max_dist_m

    def process_frame(
        self, frame: np.ndarray, detections: List[PoseDetection], frame_ts: float
    ) -> Dict[int, int]:
        """
        Processes all detections in the current frame at monotonic timestamp `frame_ts` (seconds).
        Returns mapping: {track_id: global_id}
        """
        if not detections:
            return {}

        # Step 1: Compute BEV ground coordinates & Pairwise Intersection Contamination
        boxes = np.array([d.bbox_xyxy for d in detections], dtype=np.float32)
        kpts_xy = np.array([d.keypoints_xy for d in detections], dtype=np.float32)
        kpts_conf = np.array([d.keypoints_conf for d in detections], dtype=np.float32)

        contaminated_flags = compute_intersection_contamination(boxes, kpts_xy, kpts_conf)

        for i, det in enumerate(detections):
            det.is_intersecting = bool(contaminated_flags[i])
            u_g, v_g, _ = self.bev.estimate_ground_contact_point(
                det.bbox_xyxy, det.keypoints_xy, det.keypoints_conf
            )
            det.ground_uv = (u_g, v_g)
            det.bev_xy = self.bev.to_bev_meters(u_g, v_g)

        used_global_ids: Set[int] = set()
        resolved_assignments: Dict[int, int] = {}
        unassigned_detections: List[PoseDetection] = []

        # Step 2: Validate existing ByteTrack -> Global ID bindings against BEV Physics Guard
        for det in detections:
            tid = det.track_id
            if tid in self.track_to_global:
                gid = self.track_to_global[tid]
                ident = self.identities[gid]
                reachable, dist_m, _ = self._is_physically_reachable(ident, det.bev_xy, frame_ts)

                # If ByteTrack jumped an impossible metric distance or switched during intersection
                if reachable and gid not in used_global_ids:
                    resolved_assignments[tid] = gid
                    used_global_ids.add(gid)

                    # Perform periodic gallery refinement ONLY when non-intersecting & time interval elapsed
                    last_reid_t = self.track_last_reid_ts.get(tid, 0.0)
                    if (not det.is_intersecting) and ((frame_ts - last_reid_t) >= self.reid_interval_sec):
                        emb, report = self.reid.evaluate_and_extract(
                            frame, det.bbox_xyxy, det.keypoints_xy, det.keypoints_conf, det.is_intersecting
                        )
                        self.track_last_reid_ts[tid] = frame_ts
                        if emb is not None and report.is_valid_for_gallery:
                            sim = ident.compute_similarity(emb, report.pose_view_bin)
                            # Guard against subtle post-intersection ID swap
                            if sim >= self.spatial_recovery_sim_thresh:
                                ident.update_gallery(emb, report.pose_view_bin)
                            else:
                                # Appearance mismatch detected! Revoke track binding and re-match
                                used_global_ids.remove(gid)
                                del resolved_assignments[tid]
                                del self.track_to_global[tid]
                                unassigned_detections.append(det)
                                continue

                    # Update BEV kinematics
                    dt = max(1e-3, frame_ts - ident.last_seen_ts)
                    ident.bev_velocity = 0.7 * ident.bev_velocity + 0.3 * ((det.bev_xy - ident.last_bev_xy) / dt)
                    ident.last_bev_xy = det.bev_xy
                    ident.last_seen_ts = frame_ts
                    ident.last_track_id = tid
                else:
                    # Physics guard violated: ByteTrack hijacked another track
                    del self.track_to_global[tid]
                    unassigned_detections.append(det)
            else:
                unassigned_detections.append(det)

        if not unassigned_detections:
            return resolved_assignments

        # Step 3: Global Hungarian Assignment for Unassigned / Lost Tracks (Anti-Cloning)
        candidate_gids = [gid for gid in self.identities.keys() if gid not in used_global_ids]
        det_embeddings: List[Optional[np.ndarray]] = []
        det_reports: List[CropQualityReport] = []

        for det in unassigned_detections:
            emb, rep = self.reid.evaluate_and_extract(
                frame, det.bbox_xyxy, det.keypoints_xy, det.keypoints_conf, det.is_intersecting
            )
            det_embeddings.append(emb)
            det_reports.append(rep)

        if candidate_gids:
            cost_matrix = np.full((len(unassigned_detections), len(candidate_gids)), 1e6, dtype=np.float32)

            for row_idx, det in enumerate(unassigned_detections):
                emb = det_embeddings[row_idx]
                rep = det_reports[row_idx]
                if emb is None:
                    continue

                for col_idx, gid in enumerate(candidate_gids):
                    ident = self.identities[gid]
                    reachable, dist_m, max_dist_m = self._is_physically_reachable(ident, det.bev_xy, frame_ts)
                    if not reachable:
                        continue

                    app_sim = ident.compute_similarity(emb, rep.pose_view_bin)
                    # Spatial proximity bonus in BEV meters for fast recovery after machinery occlusion
                    spatial_proxim = max(0.0, 1.0 - (dist_m / max_dist_m))
                    fused_score = 0.80 * app_sim + 0.20 * spatial_proxim

                    required_sim = (
                        self.spatial_recovery_sim_thresh if dist_m < 1.25 else self.cos_sim_thresh
                    )
                    if app_sim >= required_sim:
                        cost_matrix[row_idx, col_idx] = 1.0 - fused_score

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matched_rows: Set[int] = set()

            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 1e5:
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

                    ident.last_bev_xy = det.bev_xy
                    ident.last_seen_ts = frame_ts
                    ident.last_track_id = det.track_id
                    if emb is not None and rep.is_valid_for_gallery:
                        ident.update_gallery(emb, rep.pose_view_bin)
        else:
            matched_rows = set()

        # Step 4: Register genuinely new identities ONLY if crop passes strict Gallery Quality Gate
        for r, det in enumerate(unassigned_detections):
            if r in matched_rows:
                continue
            emb = det_embeddings[r]
            rep = det_reports[r]
            # Never spawn a brand-new Global ID from an intersecting or blurry crop!
            if emb is not None and rep.is_valid_for_gallery:
                new_gid = self._next_global_id
                self._next_global_id += 1
                new_ident = WorkerIdentity(
                    global_id=new_gid,
                    last_track_id=det.track_id,
                    last_seen_ts=frame_ts,
                    last_bev_xy=det.bev_xy,
                )
                new_ident.update_gallery(emb, rep.pose_view_bin)
                self.identities[new_gid] = new_ident
                self.track_to_global[det.track_id] = new_gid
                self.track_last_reid_ts[det.track_id] = frame_ts
                resolved_assignments[det.track_id] = new_gid
                used_global_ids.add(new_gid)

        return resolved_assignments