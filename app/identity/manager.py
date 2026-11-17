# app/identity/manager.py
from __future__ import annotations
import json
import logging
import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from app.detection.detector import PoseDetection
from app.identity.reid_model import CropQualityReport, PoseGuidedOSNetExtractor
from app.logic.geometry import (
    MetricBEVProjector,
    compute_intersection_contamination,
    compute_single_iou,
)

logger = logging.getLogger("FactoryReID")

STATUS_ACTIVE = "ACTIVE"   # visible, or hidden inside the scene (occluded) for a bounded time
STATUS_EXITED = "EXITED"   # left through the frame border (or hidden far too long)

ALL_EDGES = ("left", "right", "top", "bottom")


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
    # Image-space velocity of the ground point (px/s): tells whether a worker was walking OUT of the view.
    px_velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    # Tier 1: Immutable high-quality anchors (prevents gradual gallery drift / identity hijack)
    anchor_embeddings: List[np.ndarray] = field(default_factory=list)
    # Tier 2: View-stratified adaptive gallery
    view_gallery: Dict[str, List[np.ndarray]] = field(
        default_factory=lambda: {"FRONT": [], "BACK": [], "LEFT": [], "RIGHT": []}
    )
    # Tier 3: Short-term rolling buffer (updated ONLY when isolated & verified)
    recent_embeddings: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=6))
    # Number of verified observations (used to require evidence before re-merging an EXITED identity)
    n_obs: int = 0
    # Anti-Hijack Mismatch Circuit Breaker counter (+ when the current streak started)
    mismatch_streak: int = 0
    mismatch_first_ts: float = 0.0
    # Mutual Exclusion Sets (Tracks & Global IDs verified to belong to OTHER simultaneously visible people)
    coexisted_track_ids: Set[int] = field(default_factory=set)
    verified_coexisted_gids: Set[int] = field(default_factory=set)
    # Lifecycle: an EXITED identity may only be recovered through the strict re-entry gate.
    status: str = STATUS_ACTIVE
    exit_ts: float = 0.0
    exit_reason: str = ""  # "BORDER" | "TIMEOUT"
    exit_edge: str = ""

    def add_clean_observation(
        self,
        embedding: np.ndarray,
        view_bin: str,
        sharpness: float,
        is_isolated: bool,
        max_per_view: int = 4,
        ema_alpha: float = 0.82,
        diversity_cos_max: float = 0.90,
        crop_height_px: float = 0.0,
        anchor_min_height_px: float = 0.0,
    ) -> None:
        """
        Updates appearance memory ONLY after identity consistency has been verified.
        Anchors are permanent, so they additionally require a large enough crop.
        """
        self.n_obs += 1
        self.recent_embeddings.append(embedding.copy())

        # Only update permanent anchors and view gallery when worker is spatially isolated from others
        if not is_isolated:
            return

        if (
            len(self.anchor_embeddings) < 3
            and sharpness >= 22.0
            and crop_height_px >= anchor_min_height_px
        ):
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
      5. Identity lifecycle (ACTIVE -> EXITED) with a strict re-entry gate, so a newcomer entering where someone
         just left can never silently inherit that person's ID.
    """

    def __init__(
        self,
        reid_extractor: PoseGuidedOSNetExtractor,
        bev_projector: MetricBEVProjector,
        cos_sim_thresh: float = 0.55,
        spatial_recovery_sim_thresh: float = 0.46,
        reentry_sim_thresh: float = 0.82,
        max_worker_speed_mps: float = 3.2,
        max_internal_occlusion_m: float = 5.5,
        lag_diffusion_coeff: float = 0.65,
        base_pose_jitter_m: float = 0.85,
        reid_interval_sec: float = 0.35,
        probation_frames_for_new_id: int = 4,
        stitch_sim_thresh: float = 0.55,
        stitch_margin: float = 0.04,
        reentry_min_gap_sec: float = 2.0,
        reentry_margin: float = 0.08,
        reentry_min_samples: int = 3,
        exit_confirm_sec: float = 0.8,
        exit_edges: Sequence[str] = ALL_EDGES,
        exit_min_outward_speed_ratio: float = 0.01,
        border_clip_ratio: float = 0.01,
        occlusion_max_sec: float = 20.0,
        border_margin_ratio: float = 0.06,
        track_gap_reset_sec: float = 1.5,
        young_track_sec: float = 2.0,
        young_track_reid_interval_sec: float = 0.10,
        mismatch_min_span_sec: float = 1.0,
        unassigned_reid_interval_sec: float = 0.10,
        anchor_min_height_px: float = 110.0,
        track_state_ttl_sec: float = 10.0,
        identity_retention_sec: float = 3600.0,
        max_exited_identities: int = 300,
        decision_log_path: Optional[str] = None,
        decision_log_max_mb: float = 50.0,
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

        # --- Identity safety knobs (asymmetric cost: a wrong merge is worse than a fresh ID) ---
        self.stitch_sim_thresh = stitch_sim_thresh          # Stage A in-place stitching
        self.stitch_margin = stitch_margin                  # best-vs-runner-up gap for stitching
        self.reentry_min_gap_sec = reentry_min_gap_sec      # below this gap the re-entry bar is raised
        self.reentry_margin = reentry_margin                # best-vs-runner-up gap for EXITED identities
        self.reentry_min_samples = reentry_min_samples      # clean multi-frame samples required
        # --- Exit detection: only someone walking OUT (or cut by the frame edge) has left the camera ---
        self.exit_confirm_sec = exit_confirm_sec
        self.exit_edges = {e.lower() for e in exit_edges}
        self.exit_min_outward_speed_ratio = exit_min_outward_speed_ratio  # x max(W,H) px/s
        self.border_clip_ratio = border_clip_ratio
        self.occlusion_max_sec = occlusion_max_sec          # max in-scene occlusion before EXITED(TIMEOUT)
        self.border_margin_ratio = border_margin_ratio
        self.track_gap_reset_sec = track_gap_reset_sec      # a track id absent this long is a "resurrection"
        self.young_track_sec = young_track_sec
        self.young_track_reid_interval_sec = young_track_reid_interval_sec
        self.mismatch_min_span_sec = mismatch_min_span_sec  # mismatches must persist this long to break a binding
        self.unassigned_reid_interval_sec = unassigned_reid_interval_sec
        self.anchor_min_height_px = anchor_min_height_px
        # --- Memory bounds (long-running service) ---
        self.track_state_ttl_sec = track_state_ttl_sec
        self.identity_retention_sec = identity_retention_sec
        self.max_exited_identities = max_exited_identities
        self._last_prune_ts = -1e18

        self.track_last_seen_ts: Dict[int, float] = {}
        self.track_emb_buffer: Dict[int, Deque[np.ndarray]] = {}
        self._newly_exited: List[int] = []
        self._merged: List[Tuple[int, int]] = []  # (removed_gid, surviving_gid)

        self._decision_path: Optional[Path] = Path(decision_log_path) if decision_log_path else None
        self._decision_max_bytes = int(decision_log_max_mb * 1024 * 1024)
        self._decision_writes = 0
        self._decision_fh = None
        if self._decision_path is not None:
            self._decision_path.parent.mkdir(parents=True, exist_ok=True)
            self._decision_fh = open(self._decision_path, "a", encoding="utf-8", buffering=1)

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

    # ------------------------------------------------------------------
    # Diagnostics: one JSON line per identity decision (see `decision_log_path`)
    # ------------------------------------------------------------------
    def _log_decision(self, event: str, ts: float, **fields: Any) -> None:
        if self._decision_fh is None:
            return
        rec: Dict[str, Any] = {"event": event, "ts": round(float(ts), 3)}
        for k, v in fields.items():
            if isinstance(v, (np.floating, np.integer)):
                v = v.item()
            elif isinstance(v, float):
                v = round(v, 4)
            rec[k] = v
        try:
            self._decision_fh.write(json.dumps(rec, default=str) + "\n")
            self._decision_writes += 1
            if self._decision_writes % 500 == 0 and self._decision_fh.tell() > self._decision_max_bytes:
                self._rotate_decision_log()
        except Exception:
            logger.exception("Could not write identity decision log")

    def _rotate_decision_log(self) -> None:
        """Keep the decision log bounded: current file -> .1 (overwriting the previous .1)."""
        assert self._decision_path is not None and self._decision_fh is not None
        self._decision_fh.close()
        os.replace(self._decision_path, str(self._decision_path) + ".1")
        self._decision_fh = open(self._decision_path, "a", encoding="utf-8", buffering=1)

    def close(self) -> None:
        if self._decision_fh is not None:
            self._decision_fh.close()
            self._decision_fh = None

    # ------------------------------------------------------------------
    # Events for the caller (zones / direction tracker / vehicle FSM)
    # ------------------------------------------------------------------
    def pop_newly_exited(self) -> List[int]:
        """Global IDs that left the scene since the last call (each reported once)."""
        out, self._newly_exited = self._newly_exited, []
        return out

    def pop_merged(self) -> List[Tuple[int, int]]:
        """(removed_gid, surviving_gid) pairs merged since the last call: callers must rename their state."""
        out, self._merged = self._merged, []
        return out

    # ------------------------------------------------------------------
    # Track bookkeeping
    # ------------------------------------------------------------------
    def _forget_tracks(self, tids: Iterable[int], unbind: bool = True, forget_last_seen: bool = False) -> None:
        """Forget everything known about these track ids (dead, or re-used by someone new)."""
        tids = set(tids)
        if not tids:
            return
        for tid in tids:
            if unbind:
                self.track_to_global.pop(tid, None)
            self.track_last_reid_ts.pop(tid, None)
            self.track_seen_counts.pop(tid, None)
            self.track_first_seen_ts.pop(tid, None)
            self.track_best_emb.pop(tid, None)
            self.track_emb_buffer.pop(tid, None)
            if forget_last_seen:
                self.track_last_seen_ts.pop(tid, None)
        self.track_separated_coexisted_tracks = {
            k: v for k, v in self.track_separated_coexisted_tracks.items() if k[0] not in tids and k[1] not in tids
        }

    def _reset_track_state(self, tid: int, unbind: bool = True) -> None:
        self._forget_tracks([tid], unbind=unbind)

    def _prune(self, frame_ts: float) -> None:
        """Bound memory and per-frame cost for long runs (called about once per second)."""
        if frame_ts - self._last_prune_ts < 1.0:
            return
        self._last_prune_ts = frame_ts

        # 1. Dead tracks. ByteTrack ids only ever increase, so a dead id never comes back.
        dead = {tid for tid, ts in self.track_last_seen_ts.items() if frame_ts - ts > self.track_state_ttl_sec}
        if dead:
            self._forget_tracks(dead, unbind=True, forget_last_seen=True)
            for ident in self.identities.values():
                if ident.coexisted_track_ids & dead:
                    ident.coexisted_track_ids -= dead

        # 2. Old EXITED identities (retention window + hard cap).
        exited = sorted(
            (ident.exit_ts, gid) for gid, ident in self.identities.items() if ident.status == STATUS_EXITED
        )
        purge = {gid for ts, gid in exited if frame_ts - ts > self.identity_retention_sec}
        overflow = len(exited) - len(purge) - self.max_exited_identities
        if overflow > 0:
            purge.update(gid for _, gid in [e for e in exited if e[1] not in purge][:overflow])
        if purge:
            for gid in purge:
                del self.identities[gid]
                self._log_decision("PURGED", frame_ts, gid=gid)
            for ident in self.identities.values():
                if ident.verified_coexisted_gids & purge:
                    ident.verified_coexisted_gids -= purge
            self.pair_coexistence_counts = {
                k: v for k, v in self.pair_coexistence_counts.items() if k[0] not in purge and k[1] not in purge
            }
            for tid in [t for t, g in self.track_to_global.items() if g in purge]:
                del self.track_to_global[tid]

    def _extract_many(
        self, frame: np.ndarray, dets: List[PoseDetection]
    ) -> List[Tuple[Optional[np.ndarray], Optional[CropQualityReport]]]:
        """One batched Re-ID pass for all requested detections (falls back to per-crop calls)."""
        if not dets:
            return []
        items = [(d.bbox_xyxy, d.keypoints_xy, d.keypoints_conf, d.is_intersecting) for d in dets]
        batch_fn = getattr(self.reid, "evaluate_and_extract_batch", None)
        if batch_fn is not None:
            return list(batch_fn(frame, items))
        return [self.reid.evaluate_and_extract(frame, *it) for it in items]

    # ------------------------------------------------------------------
    # Identity lifecycle: ACTIVE -> EXITED
    # ------------------------------------------------------------------
    def _mark_exited(self, ident: WorkerIdentity, frame_ts: float, reason: str, edge: str = "") -> None:
        ident.status = STATUS_EXITED
        ident.exit_ts = frame_ts
        ident.exit_reason = reason
        ident.exit_edge = edge
        ident.mismatch_streak = 0
        # Unbind every track still pointing at this identity: ByteTrack may keep a "ghost" of the
        # leaving worker alive and hand that very track id to the next person entering there.
        bound = [tid for tid, g in self.track_to_global.items() if g == ident.global_id]
        self._forget_tracks(bound, unbind=True)
        self._newly_exited.append(ident.global_id)
        self._log_decision(
            "EXITED", frame_ts, gid=ident.global_id, reason=reason, edge=edge,
            absent_sec=frame_ts - ident.last_seen_ts,
        )

    def _exit_edge(
        self, ident: WorkerIdentity, frame_w: int, frame_h: int, allow_static_clipped: bool
    ) -> Optional[str]:
        """
        Edge the worker plausibly LEFT through, or None.
        Being near a border is not enough (a worker standing at the image edge, hidden for a second by a
        forklift, has not left). The worker must have been walking outward, or - after a longer absence -
        be cut by the frame edge without walking inward.
        """
        x1, y1, x2, y2 = (float(v) for v in ident.last_bbox_xyxy)
        mx, my = frame_w * self.border_margin_ratio, frame_h * self.border_margin_ratio
        cx, cy = max(3.0, frame_w * self.border_clip_ratio), max(3.0, frame_h * self.border_clip_ratio)
        vx, vy = float(ident.px_velocity[0]), float(ident.px_velocity[1])
        v_min = self.exit_min_outward_speed_ratio * max(frame_w, frame_h)
        edges = (
            ("left", x1 <= mx, x1 <= cx, -vx),
            ("right", x2 >= frame_w - mx, x2 >= frame_w - cx, vx),
            ("top", y1 <= my, y1 <= cy, -vy),
            ("bottom", y2 >= frame_h - my, y2 >= frame_h - cy, vy),
        )
        for name, near, clipped, outward_speed in edges:
            if name not in self.exit_edges or not near:
                continue
            if outward_speed >= v_min or (allow_static_clipped and clipped and outward_speed > -v_min):
                return name
        return None

    def _update_lifecycle(self, active_gids: Set[int], frame_ts: float, frame_w: int, frame_h: int) -> None:
        for gid, ident in self.identities.items():
            if ident.status == STATUS_EXITED or gid in active_gids:
                continue
            absent = frame_ts - ident.last_seen_ts
            if absent >= self.exit_confirm_sec:
                edge = self._exit_edge(
                    ident, frame_w, frame_h, allow_static_clipped=absent >= 2.5 * self.exit_confirm_sec
                )
                if edge is not None:
                    self._mark_exited(ident, frame_ts, "BORDER", edge)
                    continue
            if absent >= self.occlusion_max_sec:
                self._mark_exited(ident, frame_ts, "TIMEOUT")

    def _runner_up_similarity(self, emb: np.ndarray, view: str, exclude_gid: int) -> float:
        """Best similarity of `emb` against every OTHER identity, active ones included."""
        best = 0.0
        for gid, ident in self.identities.items():
            if gid == exclude_gid:
                continue
            best = max(best, ident.compute_similarity(emb, view))
        return best

    def _track_mean_embedding(self, tid: int) -> Tuple[Optional[np.ndarray], int]:
        buf = self.track_emb_buffer.get(tid)
        if not buf:
            return None, 0
        mean = np.mean(np.stack(list(buf)), axis=0)
        return (mean / (np.linalg.norm(mean) + 1e-8)).astype(np.float32), len(buf)

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
        px_step = np.array(
            [u_sm - ident.last_ground_uv[0], v_sm - ident.last_ground_uv[1]], dtype=np.float32
        ) / dt
        ident.px_velocity = (0.70 * ident.px_velocity + 0.30 * px_step).astype(np.float32)
        ident.last_ground_uv = (u_sm, v_sm)
        ident.last_bev_xy = bev_sm
        ident.last_height_px = 0.75 * det.apparent_height_px + 0.25 * ident.last_height_px
        ident.last_bbox_xyxy = det.bbox_xyxy.copy()
        ident.last_seen_ts = frame_ts
        ident.last_track_id = det.track_id

    @staticmethod
    def _reset_identity_kinematics(ident: WorkerIdentity, det: PoseDetection, frame_ts: float) -> None:
        """Hard-set the motion state from `det` (used after a swap was undone: the smoothed state was wrong)."""
        ident.last_ground_uv = (float(det.ground_uv[0]), float(det.ground_uv[1]))
        ident.last_bev_xy = np.asarray(det.bev_xy, dtype=np.float32).copy()
        ident.last_height_px = float(det.apparent_height_px)
        ident.last_bbox_xyxy = det.bbox_xyxy.copy()
        ident.bev_velocity = np.zeros(2, dtype=np.float32)
        ident.px_velocity = np.zeros(2, dtype=np.float32)
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
                    # Step 1 already smoothed each identity's motion state with the WRONG person's box:
                    # re-seed it from the correct detection, otherwise the next frame's continuity check
                    # can reject (and undo) the corrected binding.
                    self._reset_identity_kinematics(ident_a, det_b, frame_ts)
                    self._reset_identity_kinematics(ident_b, det_a, frame_ts)
                    self._log_decision(
                        "SWAP_FIXED", frame_ts, tid_a=det_a.track_id, tid_b=det_b.track_id, gid_a=gid_a, gid_b=gid_b,
                        gain=swap_score - curr_score,
                    )

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

                req_sim = 0.64 if dist_m <= 1.8 else 0.70
                if ident_old.status == STATUS_EXITED:
                    # Someone who LEFT may only be re-merged under the SAME rules as the strict re-entry gate
                    # (otherwise "reject at the gate -> mint new ID -> merge back" bypasses the gate):
                    # enough verified evidence, the strict threshold, and the extra bar for quick returns.
                    if ident_new.n_obs < self.reentry_min_samples:
                        continue
                    quick = (ident_new.created_ts - ident_old.last_seen_ts) < self.reentry_min_gap_sec
                    req_sim = max(req_sim, self.reentry_sim_thresh + (0.08 if quick else 0.0))
                gal_sim = ident_old.compute_gallery_to_gallery_similarity(ident_new)
                if gal_sim >= req_sim and gal_sim > best_merge_sim:
                    best_merge_sim = gal_sim
                    best_old_gid = gid_old

            if best_old_gid is not None and self.identities[best_old_gid].status == STATUS_EXITED:
                # Negative evidence for people who LEFT: the new identity must be clearly closer to this one
                # than to ANY other identity (exited lookalikes and currently visible workers included).
                runner_up = max(
                    (
                        ident_new.compute_gallery_to_gallery_similarity(other)
                        for g_other, other in self.identities.items()
                        if g_other not in (gid_new, best_old_gid)
                    ),
                    default=0.0,
                )
                if (best_merge_sim - runner_up) < self.reentry_margin:
                    self._log_decision(
                        "MERGE_REJECTED_AMBIGUOUS", frame_ts, tid=tid, gid_new=gid_new,
                        gid_old=best_old_gid, gallery_sim=best_merge_sim, runner_up=runner_up,
                    )
                    best_old_gid = None

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
                ident_old.px_velocity = ident_new.px_velocity
                ident_old.last_track_id = tid
                ident_old.status = STATUS_ACTIVE
                ident_old.exit_reason = ""
                self._log_decision(
                    "MERGED", frame_ts, tid=tid, gid_new=gid_new, gid_old=best_old_gid, gallery_sim=best_merge_sim,
                )

                self.track_to_global[tid] = best_old_gid
                resolved_assignments[tid] = best_old_gid
                used_global_ids.discard(gid_new)
                used_global_ids.add(best_old_gid)
                del self.identities[gid_new]
                self._merged.append((gid_new, best_old_gid))

    def process_frame(
        self, frame: np.ndarray, detections: List[PoseDetection], frame_ts: float
    ) -> Dict[int, int]:
        H_img, W_img = frame.shape[:2]
        if not detections:
            # Nobody visible: lifecycle must still advance (someone may have just left the camera).
            self._update_lifecycle(set(), frame_ts, W_img, H_img)
            self._prune(frame_ts)
            return {}

        boxes = np.array([d.bbox_xyxy for d in detections], dtype=np.float32)
        kpts_xy = np.array([d.keypoints_xy for d in detections], dtype=np.float32)
        kpts_conf = np.array([d.keypoints_conf for d in detections], dtype=np.float32)

        contaminated_flags = compute_intersection_contamination(boxes, kpts_xy, kpts_conf)

        # Compute Pose-Corrected Standing Height, Ground Contact Point, and BEV Coordinates
        for i, det in enumerate(detections):
            tid = det.track_id
            # A track id that vanished for a while and re-appears is a "resurrection": the tracker may
            # have re-used the id for someone else. Treat it as a brand-new track.
            last_seen = self.track_last_seen_ts.get(tid)
            if last_seen is not None and (frame_ts - last_seen) > self.track_gap_reset_sec:
                self._log_decision(
                    "TRACK_RESURRECTED", frame_ts, tid=tid, gid=self.track_to_global.get(tid),
                    gap_sec=frame_ts - last_seen,
                )
                self._reset_track_state(tid, unbind=True)
            self.track_last_seen_ts[tid] = frame_ts

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
        # 1a. Decide which bindings survive the motion check and which need a Re-ID verification.
        pending: List[Tuple[PoseDetection, int, WorkerIdentity, bool]] = []
        claimed: Set[int] = set()
        for det in detections:
            tid = det.track_id
            if tid not in self.track_to_global:
                unassigned_detections.append(det)
                continue
            gid = self.track_to_global[tid]
            ident = self.identities.get(gid)
            if ident is None:
                del self.track_to_global[tid]
                unassigned_detections.append(det)
                continue

            reachable, _, _ = self._is_spatially_continuous_with_identity(ident, det, frame_ts)
            if not reachable or gid in claimed:
                del self.track_to_global[tid]
                unassigned_detections.append(det)
                continue
            claimed.add(gid)

            last_reid_t = self.track_last_reid_ts.get(tid, 0.0)
            # Young tracks are the ones most likely to be a hijacked/re-used id: verify them often.
            is_young = (frame_ts - self.track_first_seen_ts.get(tid, frame_ts)) < self.young_track_sec
            reid_interval = self.young_track_reid_interval_sec if is_young else self.reid_interval_sec
            need_reid = (not det.is_intersecting) and (
                (frame_ts - last_reid_t) >= reid_interval - 1e-3 or ident.mismatch_streak > 0
            )
            pending.append((det, gid, ident, need_reid))

        # 1b. One batched OSNet pass for every binding that needs verification.
        to_verify = [p[0] for p in pending if p[3]]
        verified = dict(zip((d.track_id for d in to_verify), self._extract_many(frame, to_verify)))

        # 1c. Apply the verification results.
        for det, gid, ident, need_reid in pending:
            tid = det.track_id
            if need_reid:
                emb, report = verified[tid]
                self.track_last_reid_ts[tid] = frame_ts
                if emb is not None and report is not None and report.is_valid_for_gallery:
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
                            crop_height_px=float(det.bbox_xyxy[3] - det.bbox_xyxy[1]),
                            anchor_min_height_px=self.anchor_min_height_px,
                        )
                    elif det.is_isolated:
                        # Isolated, clean crop that does NOT reach the acceptance threshold.
                        if ident.mismatch_streak == 0:
                            ident.mismatch_first_ts = frame_ts
                        ident.mismatch_streak += 1
                        # Break only on PERSISTENT mismatch: a few bad crops in a fraction of a second
                        # (bending, motion blur) must not flip the ID back and forth.
                        if (
                            ident.mismatch_streak >= 3
                            and (frame_ts - ident.mismatch_first_ts) >= self.mismatch_min_span_sec
                        ):
                            # Mismatch circuit breaker tripped: ByteTrack hijacked a different person!
                            self._log_decision(
                                "BINDING_BROKEN", frame_ts, tid=tid, gid=gid, sim=sim,
                                view=report.pose_view_bin, streak_sec=frame_ts - ident.mismatch_first_ts,
                            )
                            ident.mismatch_streak = 0
                            del self.track_to_global[tid]
                            # The crops of this track are a mix of two people: restart probation and
                            # evidence so the next ID is not built from contaminated samples.
                            self.track_seen_counts[tid] = 0
                            self.track_emb_buffer.pop(tid, None)
                            self.track_best_emb.pop(tid, None)
                            unassigned_detections.append(det)
                            continue

            resolved_assignments[tid] = gid
            used_global_ids.add(gid)
            self._update_identity_kinematics(ident, det, frame_ts)

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

            # One batched OSNet pass; tracks extracted very recently reuse their last embedding.
            to_extract = [
                det for det in unassigned_detections
                if (frame_ts - self.track_last_reid_ts.get(det.track_id, -1e18)) >= self.unassigned_reid_interval_sec - 1e-3
            ]
            fresh = dict(zip((d.track_id for d in to_extract), self._extract_many(frame, to_extract)))

            det_embeddings: List[Optional[np.ndarray]] = []
            det_reports: List[Optional[CropQualityReport]] = []
            for det in unassigned_detections:
                emb, rep = fresh.get(det.track_id, (None, None))
                if det.track_id in fresh:
                    self.track_last_reid_ts[det.track_id] = frame_ts
                det_embeddings.append(emb)
                det_reports.append(rep)
                if emb is not None and rep is not None and rep.is_valid_for_gallery:
                    self.track_best_emb[det.track_id] = (emb, rep.pose_view_bin, rep.sharpness_score)
                    # Multi-frame evidence for the strict re-entry gate (never a single lucky frame).
                    self.track_emb_buffer.setdefault(det.track_id, deque(maxlen=6)).append(emb.copy())

            # The embedding/view actually used for matching each row (fresh, or the track's last good one)
            row_emb: List[Optional[np.ndarray]] = []
            row_view: List[str] = []
            for r, det in enumerate(unassigned_detections):
                emb, rep = det_embeddings[r], det_reports[r]
                if emb is None and det.track_id in self.track_best_emb:
                    emb, view, _ = self.track_best_emb[det.track_id]
                else:
                    view = rep.pose_view_bin if rep is not None else "FRONT"
                row_emb.append(emb)
                row_view.append(view)

            matched_rows: Set[int] = set()
            stage_of: Dict[Tuple[int, int], str] = {}
            if candidate_gids:
                cost_matrix = np.full((len(unassigned_detections), len(candidate_gids)), 1e6, dtype=np.float32)
                sim_matrix = np.zeros((len(unassigned_detections), len(candidate_gids)), dtype=np.float32)

                for r, det in enumerate(unassigned_detections):
                    emb, pose_view = row_emb[r], row_view[r]
                    if emb is None:
                        continue

                    track_birth_ts = self.track_first_seen_ts.get(det.track_id, frame_ts)
                    strict_emb, strict_n = self._track_mean_embedding(det.track_id)
                    det_near_border = self._is_near_frame_border(
                        det.bbox_xyxy, W_img, H_img, self.border_margin_ratio
                    )

                    # Count how many lost identities disappeared within 1.5m of `det` (spatial ambiguity check).
                    # EXITED identities are not "lost nearby": they can only return via the strict gate.
                    nearby_lost_count = sum(
                        1
                        for gid_c in candidate_gids
                        if self.identities[gid_c].status != STATUS_EXITED
                        and (frame_ts - self.identities[gid_c].last_seen_ts) <= 3.0
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

                        # STRICT RE-ENTRY GATE for identities that LEFT the scene. They are never eligible for
                        # in-place stitching or spatial recovery (that is exactly how a newcomer entering from
                        # the same side used to inherit the ID of the person who just left).
                        if ident.status == STATUS_EXITED:
                            if strict_emb is None or strict_n < self.reentry_min_samples:
                                continue  # not enough clean evidence yet -> keep waiting / mint a new ID
                            if ident.exit_reason == "BORDER" and not det_near_border:
                                continue  # a border-exit must come back in through a border
                            s_strict = ident.compute_similarity(strict_emb, pose_view)
                            sim_matrix[r, c] = s_strict
                            req = self.reentry_sim_thresh + (0.08 if dt_lost < self.reentry_min_gap_sec else 0.0)
                            if s_strict >= req:
                                cost_matrix[r, c] = 1.0 - (0.90 * s_strict)
                                stage_of[(r, c)] = "REENTRY_STRICT"
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
                            if app_sim >= self.stitch_sim_thresh:
                                stitch_score = 0.65 * app_sim + 0.35 * max(box_iou, 1.0 - dist_m / 1.5)
                                cost_matrix[r, c] = 1.0 - stitch_score
                                stage_of[(r, c)] = "STITCH"
                                continue

                        # STAGE B: Spatio-Temporal Occlusion Recovery
                        if not reachable:
                            # STAGE C: Border / Long-Horizon Re-Entry
                            near_border = self._is_near_frame_border(det.bbox_xyxy, W_img, H_img)
                            if near_border and dt_lost > 2.0 and app_sim >= self.reentry_sim_thresh:
                                cost_matrix[r, c] = 1.0 - (0.90 * app_sim)
                                stage_of[(r, c)] = "BORDER_REENTRY"
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
                            stage_of[(r, c)] = "RECOVER"

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
                        stage = stage_of.get((r, c), "UNKNOWN")

                        # Negative evidence: the newcomer must be clearly closer to this identity than to
                        # ANY other identity, including workers who are currently visible on screen.
                        if stage in ("REENTRY_STRICT", "STITCH", "BORDER_REENTRY"):
                            if stage == "REENTRY_STRICT":
                                m_emb, _ = self._track_mean_embedding(det.track_id)
                                margin_needed = self.reentry_margin
                            else:
                                m_emb = row_emb[r]
                                margin_needed = self.stitch_margin if stage == "STITCH" else self.reentry_margin
                            m_view = row_view[r]
                            if m_emb is not None:
                                own_sim = ident.compute_similarity(m_emb, m_view)
                                runner_up = self._runner_up_similarity(m_emb, m_view, gid)
                                if (own_sim - runner_up) < margin_needed:
                                    self._log_decision(
                                        "MATCH_REJECTED_AMBIGUOUS", frame_ts, tid=det.track_id, gid=gid,
                                        stage=stage, sim=own_sim, runner_up=runner_up,
                                        margin_needed=margin_needed,
                                    )
                                    continue

                        self._log_decision(
                            "MATCHED", frame_ts, tid=det.track_id, gid=gid, stage=stage,
                            sim=float(sim_matrix[r, c]), dt_lost=frame_ts - ident.last_seen_ts,
                            status_before=ident.status, near_border=bool(
                                self._is_near_frame_border(det.bbox_xyxy, W_img, H_img, self.border_margin_ratio)
                            ),
                        )
                        ident.status = STATUS_ACTIVE
                        ident.exit_reason = ""

                        self.track_to_global[det.track_id] = gid
                        self.track_last_reid_ts[det.track_id] = frame_ts
                        resolved_assignments[det.track_id] = gid
                        used_global_ids.add(gid)
                        matched_rows.add(r)

                        ident.mismatch_streak = 0
                        self._update_identity_kinematics(ident, det, frame_ts)
                        if emb is not None and rep is not None and rep.is_valid_for_gallery:
                            ident.add_clean_observation(
                                emb,
                                rep.pose_view_bin,
                                sharpness=rep.sharpness_score,
                                is_isolated=det.is_isolated,
                                crop_height_px=float(det.bbox_xyxy[3] - det.bbox_xyxy[1]),
                                anchor_min_height_px=self.anchor_min_height_px,
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

                emb = row_emb[r]
                view_bin = row_view[r]
                rep = det_reports[r]
                if det_embeddings[r] is not None and rep is not None:
                    sharpness_val = rep.sharpness_score
                    gallery_ok = rep.is_valid_for_gallery
                elif det.track_id in self.track_best_emb:
                    sharpness_val = self.track_best_emb[det.track_id][2]
                    gallery_ok = True  # track_best_emb only ever holds gallery-quality crops
                else:
                    sharpness_val, gallery_ok = 20.0, False

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
                    # A poor crop may seed the short-term memory, but never the gallery or the anchors.
                    new_ident.add_clean_observation(
                        emb, view_bin, sharpness=sharpness_val, is_isolated=det.is_isolated and gallery_ok,
                        crop_height_px=float(det.bbox_xyxy[3] - det.bbox_xyxy[1]),
                        anchor_min_height_px=self.anchor_min_height_px,
                    )

                    self.identities[new_gid] = new_ident
                    if self._decision_fh is not None:
                        best_other = self._runner_up_similarity(emb, view_bin, new_gid)
                        self._log_decision(
                            "NEW_ID", frame_ts, tid=det.track_id, gid=new_gid,
                            best_existing_sim=best_other, seen_frames=self.track_seen_counts.get(det.track_id, 0),
                        )
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

        # Who is gone? Mark border exits / long occlusions so they can only return via the strict gate.
        self._update_lifecycle(set(resolved_assignments.values()), frame_ts, W_img, H_img)
        self._prune(frame_ts)

        return resolved_assignments
