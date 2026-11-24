# app/identity/store.py
"""
Persistence of the identity memory across restarts (live service).

Without it every restart began again at ID 1: reports mixed different workers under the same number and nobody
present before the restart could be recognised. Saved periodically and on shutdown (atomic writes):

    <dir>/identities.json   next_global_id, per identity: label, times (wall clock), last position, status
    <dir>/galleries.npz     anchors / view galleries / recent embeddings
    (+ the open-set impostor statistics)

On start-up every saved identity is loaded as EXITED (reason RESTART): it can only come back through the strict
re-entry gate (or its helmet marker). Galleries are discarded when the Re-ID model changed (`model_tag`).
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict

import numpy as np

from app.core.clock import StreamClock
from app.identity.manager import STATUS_EXITED, EnterpriseIdentityManager, WorkerIdentity
from app.identity.stature import StatureSamples

logger = logging.getLogger("FactoryReID")

VERSION = 1


class IdentityStore:
    def __init__(self, directory: str, clock: StreamClock, model_tag: str = "", retention_sec: float = 12 * 3600):
        self.dir = Path(directory)
        self.clock = clock
        self.model_tag = str(model_tag)
        self.retention_sec = float(retention_sec)
        self.meta_path = self.dir / "identities.json"
        self.npz_path = self.dir / "galleries.npz"

    # ------------------------------------------------------------------
    def save(self, mgr: EnterpriseIdentityManager) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        arrays: Dict[str, np.ndarray] = {}
        idents = []
        for gid, ident in mgr.identities.items():
            idents.append({
                "gid": int(gid),
                "label": ident.label,
                "created_wall": self.clock.wall(ident.created_ts),
                "last_seen_wall": self.clock.wall(ident.last_seen_ts),
                "n_obs": int(ident.n_obs),
                "status": ident.status,
                "exit_reason": ident.exit_reason,
                "exit_edge": ident.exit_edge,
                "last_ground_uv": [float(ident.last_ground_uv[0]), float(ident.last_ground_uv[1])],
                "last_bev_xy": [float(v) for v in np.asarray(ident.last_bev_xy).ravel()[:2]],
                "last_height_px": float(ident.last_height_px),
                "last_bbox_xyxy": [float(v) for v in np.asarray(ident.last_bbox_xyxy).ravel()[:4]],
                "stature": ident.stature.to_list(),
            })
            if ident.anchor_embeddings:
                arrays[f"g{gid}_anchor"] = np.stack(ident.anchor_embeddings).astype(np.float32)
            for vb, feats in ident.view_gallery.items():
                if feats:
                    arrays[f"g{gid}_view_{vb}"] = np.stack(feats).astype(np.float32)
            if ident.recent_embeddings:
                arrays[f"g{gid}_recent"] = np.stack(list(ident.recent_embeddings)).astype(np.float32)
        meta = {
            "version": VERSION,
            "saved_wall": time.time(),
            "model_tag": self.model_tag,
            "next_global_id": int(mgr._next_global_id),
            "identities": idents,
            "open_set": mgr.open_set.state(),
            "stature_model": mgr.stature_model.state(),
        }
        tmp_npz = self.dir / "galleries.tmp.npz"
        np.savez_compressed(tmp_npz, **arrays)
        os.replace(tmp_npz, self.npz_path)
        tmp_meta = self.dir / "identities.tmp.json"
        tmp_meta.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_meta, self.meta_path)

    # ------------------------------------------------------------------
    def load_into(self, mgr: EnterpriseIdentityManager, now_ts: float) -> int:
        if not self.meta_path.exists():
            return 0
        try:
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Identity store %s is unreadable: starting empty", self.meta_path)
            return 0
        if int(meta.get("version", 0)) != VERSION:
            logger.warning("Identity store version mismatch: starting empty")
            return 0
        same_model = meta.get("model_tag", "") == self.model_tag
        arrays: Dict[str, np.ndarray] = {}
        if same_model and self.npz_path.exists():
            with np.load(self.npz_path) as data:
                arrays = {k: data[k] for k in data.files}
        elif not same_model:
            logger.warning("Re-ID model changed (%s -> %s): saved appearance galleries discarded, labels kept",
                           meta.get("model_tag"), self.model_tag)

        now_wall = self.clock.wall(now_ts)
        loaded = 0
        for rec in meta.get("identities", []):
            if now_wall - float(rec["last_seen_wall"]) > self.retention_sec:
                continue
            gid = int(rec["gid"])
            ident = WorkerIdentity(
                global_id=gid,
                created_ts=self.clock.from_wall(float(rec["created_wall"])),
                last_track_id=-1,
                last_seen_ts=self.clock.from_wall(float(rec["last_seen_wall"])),
                last_bev_xy=np.asarray(rec["last_bev_xy"], dtype=np.float32),
                last_ground_uv=tuple(rec["last_ground_uv"]),
                last_height_px=float(rec["last_height_px"]),
                last_bbox_xyxy=np.asarray(rec["last_bbox_xyxy"], dtype=np.float32),
                label=rec.get("label"),
            )
            ident.n_obs = int(rec.get("n_obs", 0))
            ident.stature = StatureSamples.from_list(rec.get("stature"))
            if f"g{gid}_anchor" in arrays:
                ident.anchor_embeddings = [a.copy() for a in arrays[f"g{gid}_anchor"]]
            for vb in list(ident.view_gallery):
                key = f"g{gid}_view_{vb}"
                if key in arrays:
                    ident.view_gallery[vb] = [a.copy() for a in arrays[key]]
            if f"g{gid}_recent" in arrays:
                ident.recent_embeddings = deque((a.copy() for a in arrays[f"g{gid}_recent"]), maxlen=6)
            has_appearance = bool(ident.anchor_embeddings or any(ident.view_gallery.values()))
            if not has_appearance and ident.label is None:
                continue
            ident.status = STATUS_EXITED
            ident.exit_ts = now_ts
            ident.exit_reason = "RESTART"
            mgr.identities[gid] = ident
            loaded += 1
        mgr._next_global_id = max(
            int(meta.get("next_global_id", 1)), max(mgr.identities, default=0) + 1, mgr._next_global_id
        )
        if same_model:
            mgr.open_set.load_state(meta.get("open_set"))
        mgr.stature_model.load_state(meta.get("stature_model"))   # geometry of this camera: model-independent
        logger.info("Identity store: %d identities restored, next ID %d", loaded, mgr._next_global_id)
        return loaded
