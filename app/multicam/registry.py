# app/multicam/registry.py
"""
Cross-camera identity: the same worker gets the same PERSON ID (P#) in every camera.

Architecture (one process per camera, see run_multi_camera.py):
  * every camera keeps its own identity manager and its own LOCAL worker IDs (unchanged single-camera logic);
  * each local identity is a "tracklet" in a shared SQLite registry (data/cross_camera.db, WAL, multi-process safe):
        tracklets(camera_id, local_id, person_id, label, first/last seen, active, appearance embeddings)
        persons(person_id, label, merged_into)
  * a background thread per camera (CrossCameraLinker) publishes its tracklets every few seconds and links every
    unlinked local identity to a person of the OTHER cameras - or creates a new person.

Linking rules (a wrong person is worse than a new person):
  1. helmet marker label        -> the person with that label (hard identity)
  2. physics                    -> a person seen at the same time in a NON-overlapping camera is someone else;
                                   optional transit times between cameras (configs/cameras.yaml: topology)
  3. appearance                 -> gallery similarity >= max(link_threshold, learned impostor level + margin)
                                   AND better than the runner-up person by `margin`
  4. no confident match after `decide_after_sec` -> new person. A new person may still be merged into an older one
     within `late_merge_sec` if a clearly stronger match appears.
Impostor level: tracklets visible at the same time in non-overlapping cameras (or in the same camera) are different
people; their similarities calibrate the threshold for this site's uniforms (same idea as app/identity/open_set.py).
"""
from __future__ import annotations

import logging
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from app.identity.open_set import OpenSetCalibrator
from app.identity.stature import StatureEstimate, statures_conflict

logger = logging.getLogger("FactoryReID")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS persons (
    person_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    label        TEXT,
    created_wall REAL,
    merged_into  INTEGER
);
CREATE TABLE IF NOT EXISTS tracklets (
    camera_id        TEXT NOT NULL,
    local_id         INTEGER NOT NULL,
    person_id        INTEGER,
    label            TEXT,
    first_seen_wall  REAL,
    last_seen_wall   REAL,
    active           INTEGER,
    n_obs            INTEGER,
    dim              INTEGER,
    emb              BLOB,
    emb_updated_wall REAL,
    updated_wall     REAL,
    link_method      TEXT,
    link_score       REAL,
    PRIMARY KEY (camera_id, local_id)
);
CREATE INDEX IF NOT EXISTS idx_tracklets_updated ON tracklets(updated_wall);
"""
# columns added later (older databases are migrated in place)
_MIGRATIONS = {"stature": "REAL", "stature_se": "REAL", "entry_gate": "TEXT", "entry_wall": "REAL",
               "exit_gate": "TEXT", "exit_wall": "REAL"}
# heights of the same person in two cameras: each camera's model has its own error
CROSS_CAMERA_STATURE_ERROR = 0.02


def gallery_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Mean of the top 25 % pairwise cosine similarities between two embedding sets (same as the local manager)."""
    if a.size == 0 or b.size == 0:
        return 0.0
    flat = np.sort((a @ b.T).ravel())
    k = max(1, len(flat) // 4)
    return float(np.mean(flat[-k:]))


class CameraTopology:
    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = cfg or {}
        self.overlaps: Set[Tuple[str, str]] = set()
        for pair in cfg.get("overlaps") or []:
            a, b = str(pair[0]), str(pair[1])
            self.overlaps |= {(a, b), (b, a)}
        self.transitions: Dict[Tuple[str, str], Tuple[float, float]] = {}
        for t in cfg.get("transitions") or []:
            a, b = str(t["from"]), str(t["to"])
            rng = (float(t.get("min_sec", 0.0)), float(t.get("max_sec", float("inf"))))
            self.transitions[(a, b)] = rng
            if t.get("bidirectional", True):
                self.transitions.setdefault((b, a), rng)
        self.only_listed = bool(cfg.get("only_listed_transitions", False))
        # Doors between cameras: (exit camera, exit gate, entry camera, entry gate, min_sec, max_sec)
        self.doors: List[Tuple[str, str, str, str, float, float]] = []
        for d in cfg.get("doors") or []:
            a_cam, a_gate = self.parse_gate(d.get("from"), d.get("name", "?"))
            b_cam, b_gate = self.parse_gate(d.get("to"), d.get("name", "?"))
            lo, hi = float(d.get("min_sec", 0.0)), float(d.get("max_sec", 60.0))
            if hi < lo:
                raise ValueError(f"cross_camera door '{d.get('name')}': max_sec < min_sec")
            self.doors.append((a_cam, a_gate, b_cam, b_gate, lo, hi))
            if d.get("bidirectional", True):
                self.doors.append((b_cam, b_gate, a_cam, a_gate, lo, hi))
            # a door is also a camera-level connection (with only_listed_transitions)
            for src, dst in ((a_cam, b_cam), (b_cam, a_cam)) if d.get("bidirectional", True) else ((a_cam, b_cam),):
                old = self.transitions.get((src, dst))
                self.transitions[(src, dst)] = (min(lo, old[0]), max(hi, old[1])) if old else (lo, hi)

    @staticmethod
    def parse_gate(spec: Any, door_name: str) -> Tuple[str, str]:
        """{camera: cam1, region: hall_door} -> ("cam1", "door:hall_door"); {camera: cam1, edge: left} -> edge."""
        if not isinstance(spec, dict) or not spec.get("camera"):
            raise ValueError(f"cross_camera door '{door_name}': 'from'/'to' need a camera and a region or an edge")
        if spec.get("region"):
            return str(spec["camera"]), f"door:{spec['region']}"
        if spec.get("edge") in ("left", "right", "top", "bottom"):
            return str(spec["camera"]), f"edge:{spec['edge']}"
        raise ValueError(f"cross_camera door '{door_name}': give `region: <exit door drawn in that camera>` or "
                         "`edge: left|right|top|bottom`")

    def doors_into(self, camera: str, gate: Optional[str]) -> List[Tuple[str, str, float, float]]:
        """Where somebody appearing at (camera, gate) may come from: (exit camera, exit gate, min_sec, max_sec)."""
        if not gate:
            return []
        return [(a, ag, lo, hi) for a, ag, b, bg, lo, hi in self.doors if b == camera and bg == gate]

    def overlap(self, a: str, b: str) -> bool:
        """True when the two cameras see a common area (a person may be visible in both at once)."""
        return (a, b) in self.overlaps

    def transit_ok(self, src: str, dst: str, gap_sec: float) -> bool:
        if (src, dst) in self.transitions:
            lo, hi = self.transitions[(src, dst)]
            return lo <= gap_sec <= hi
        return not self.only_listed or src == dst


@dataclass
class Tracklet:
    camera_id: str
    local_id: int
    person_id: Optional[int] = None
    label: Optional[str] = None
    first_seen: float = 0.0
    last_seen: float = 0.0
    active: bool = False
    n_obs: int = 0
    embs: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.float32))
    emb_version: float = 0.0
    link_method: Optional[str] = None
    stature: Optional[StatureEstimate] = None
    entry_gate: Optional[str] = None
    entry_wall: Optional[float] = None
    exit_gate: Optional[str] = None
    exit_wall: Optional[float] = None


def snapshot_from_manager(manager: Any, clock: Any, now_ts: float, max_embs: int = 16) -> List[Dict[str, Any]]:
    """Copy of the local identities that the linker thread can use safely."""
    out = []
    for gid, ident in list(manager.identities.items()):
        feats = list(ident.anchor_embeddings) + [f for bank in ident.view_gallery.values() for f in bank]
        if not feats:
            feats = list(ident.recent_embeddings)
        if not feats:
            continue
        est_fn = getattr(manager, "stature_estimate", None)
        est = est_fn(gid) if est_fn is not None else None
        exit_gate = ident.exit_gate() if hasattr(ident, "exit_gate") else None
        out.append({
            "local_id": int(gid),
            "label": ident.label,
            "entry_gate": getattr(ident, "entry_gate", None),
            "entry_wall": float(clock.wall(getattr(ident, "entry_ts", ident.created_ts) or ident.created_ts)),
            "exit_gate": exit_gate,
            "exit_wall": float(clock.wall(ident.last_seen_ts)) if exit_gate else None,
            "stature": None if est is None else (float(est.ratio), float(est.se), int(est.n)),
            "embs": np.stack(feats[:max_embs]).astype(np.float32),
            "n_obs": int(ident.n_obs),
            "first_seen": float(clock.wall(ident.created_ts)),
            "last_seen": float(clock.wall(ident.last_seen_ts)),
            "active": ident.status == "ACTIVE" and (now_ts - ident.last_seen_ts) < 2.0,
        })
    return out


class CrossCameraLinker:
    def __init__(
        self,
        registry_path: str,
        camera_id: str,
        topology: Optional[CameraTopology] = None,
        link_threshold: float = 0.80,
        margin: float = 0.06,
        min_obs: int = 5,
        decide_after_sec: float = 20.0,
        late_merge_sec: float = 120.0,
        retention_sec: float = 12 * 3600.0,
        sync_interval_sec: float = 2.0,
        simultaneity_tolerance_sec: float = 1.0,
        open_set: Optional[OpenSetCalibrator] = None,
        background: bool = True,
        stature_min_diff: float = 0.05,
        door_min_sim: float = 0.50,
    ):
        self.stature_min_diff = float(stature_min_diff)
        # door handover: the appearance only has to "not contradict" (light differs a lot between hall and yard)
        self.door_min_sim = float(door_min_sim)
        self.path = str(registry_path)
        self.camera_id = str(camera_id)
        self.topology = topology or CameraTopology()
        self.link_threshold = float(link_threshold)
        self.margin = float(margin)
        self.min_obs = int(min_obs)
        self.decide_after_sec = float(decide_after_sec)
        self.late_merge_sec = float(late_merge_sec)
        self.retention_sec = float(retention_sec)
        self.sync_interval_sec = float(sync_interval_sec)
        self.tol = float(simultaneity_tolerance_sec)
        self.open_set = open_set or OpenSetCalibrator(self.link_threshold, quantile=0.99, margin=0.03,
                                                      min_samples=40, max_threshold=0.95)
        self._links: Dict[int, int] = {}
        self._links_lock = threading.Lock()
        self._pending_merges: List[Tuple[int, int]] = []
        self._cache: Dict[Tuple[str, int], Tracklet] = {}
        self._last_fetch = 0.0
        self._sent: Dict[int, Tuple[int, Optional[str]]] = {}
        self._last_sync_mono = -1e18
        self._last_impostor = 0.0
        self._conn: Optional[sqlite3.Connection] = None
        self.n_links = 0
        self.background = bool(background)
        self._q: "queue.Queue[Optional[Tuple[List[Dict[str, Any]], float]]]" = queue.Queue(maxsize=1)
        if self.background:
            self._thread = threading.Thread(target=self._loop, name=f"crosscam-{camera_id}", daemon=True)
            self._thread.start()

    # ------------------------------------------------------------------ main-thread API
    def person_of(self, local_id: int) -> Optional[int]:
        with self._links_lock:
            return self._links.get(int(local_id))

    def links(self) -> Dict[int, int]:
        with self._links_lock:
            return dict(self._links)

    def notify_merge(self, old_local: int, new_local: int) -> None:
        with self._links_lock:
            self._pending_merges.append((int(old_local), int(new_local)))

    def observe(self, manager: Any, clock: Any, now_ts: float) -> None:
        """Call every frame; every sync_interval a snapshot is handed to the background thread."""
        if time.monotonic() - self._last_sync_mono < self.sync_interval_sec:
            return
        self._last_sync_mono = time.monotonic()
        snap = snapshot_from_manager(manager, clock, now_ts)
        item = (snap, float(clock.wall(now_ts)))
        if not self.background:
            self.sync_now(*item)
            return
        try:
            self._q.put_nowait(item)
        except queue.Full:
            pass  # previous sync still running: skip this round

    def close(self) -> None:
        if self.background:
            try:
                self._q.put(None, timeout=2.0)
            except queue.Full:
                pass
            self._thread.join(timeout=5.0)
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _loop(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            try:
                self.sync_now(*item)
            except Exception:
                logger.exception("Cross-camera sync failed")

    # ------------------------------------------------------------------ database
    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.path, timeout=30.0, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            cols = {r[1] for r in self._conn.execute("PRAGMA table_info(tracklets)").fetchall()}
            for name, typ in _MIGRATIONS.items():
                if name not in cols:
                    try:
                        self._conn.execute(f"ALTER TABLE tracklets ADD COLUMN {name} {typ}")
                    except sqlite3.OperationalError:
                        pass        # another camera process added it at the same moment
            self._conn.commit()
        return self._conn

    def _resolve_person(self, conn: sqlite3.Connection, pid: Optional[int]) -> Optional[int]:
        seen = set()
        while pid is not None and pid not in seen:
            seen.add(pid)
            row = conn.execute("SELECT merged_into FROM persons WHERE person_id = ?", (pid,)).fetchone()
            if row is None or row[0] is None:
                return pid
            pid = int(row[0])
        return pid

    def _new_person(self, conn: sqlite3.Connection, label: Optional[str], wall: float) -> int:
        cur = conn.execute("INSERT INTO persons (label, created_wall) VALUES (?, ?)", (label, wall))
        return int(cur.lastrowid)

    def _merge_persons(self, conn: sqlite3.Connection, a: int, b: int) -> int:
        """Persons a and b are the same: keep the older id."""
        a, b = self._resolve_person(conn, a), self._resolve_person(conn, b)
        if a is None or b is None or a == b:
            return a if a is not None else b
        keep, drop = min(a, b), max(a, b)
        conn.execute("UPDATE persons SET merged_into = ? WHERE person_id = ?", (keep, drop))
        conn.execute("UPDATE tracklets SET person_id = ? WHERE person_id = ?", (keep, drop))
        lab = conn.execute("SELECT label FROM persons WHERE person_id = ?", (drop,)).fetchone()
        if lab and lab[0]:
            conn.execute("UPDATE persons SET label = COALESCE(label, ?) WHERE person_id = ?", (lab[0], keep))
        return keep

    # ------------------------------------------------------------------ one synchronisation round
    def sync_now(self, snapshot: List[Dict[str, Any]], now_wall: float) -> None:
        conn = self._db()
        mine = {s["local_id"]: s for s in snapshot}

        # 1. publish my tracklets
        for lid, s in mine.items():
            sent = self._sent.get(lid)
            full = sent is None or sent[1] != s["label"] or s["n_obs"] - sent[0] >= 3
            if full:
                conn.execute(
                    "INSERT INTO tracklets (camera_id, local_id, label, first_seen_wall, last_seen_wall, active, n_obs, "
                    "dim, emb, emb_updated_wall, updated_wall) VALUES (?,?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(camera_id, local_id) DO UPDATE SET label=excluded.label, "
                    "last_seen_wall=excluded.last_seen_wall, active=excluded.active, n_obs=excluded.n_obs, "
                    "dim=excluded.dim, emb=excluded.emb, emb_updated_wall=excluded.emb_updated_wall, "
                    "updated_wall=excluded.updated_wall",
                    (self.camera_id, lid, s["label"], s["first_seen"], s["last_seen"], int(s["active"]), s["n_obs"],
                     int(s["embs"].shape[1]), s["embs"].astype(np.float32).tobytes(), now_wall, now_wall),
                )
                self._sent[lid] = (s["n_obs"], s["label"])
            else:
                conn.execute("UPDATE tracklets SET last_seen_wall=?, active=?, updated_wall=? "
                             "WHERE camera_id=? AND local_id=?",
                             (s["last_seen"], int(s["active"]), now_wall, self.camera_id, lid))
            st = s.get("stature")
            conn.execute("UPDATE tracklets SET stature=?, stature_se=?, entry_gate=?, entry_wall=?, exit_gate=?, "
                         "exit_wall=? WHERE camera_id=? AND local_id=?",
                         (None if st is None else st[0], None if st is None else st[1], s.get("entry_gate"),
                          s.get("entry_wall"), s.get("exit_gate"), s.get("exit_wall"), self.camera_id, lid))
        for lid in [lid for lid in self._sent if lid not in mine]:   # purged locally
            conn.execute("UPDATE tracklets SET active=0 WHERE camera_id=? AND local_id=?", (self.camera_id, lid))
            del self._sent[lid]

        # 2. local merges reported by the identity manager
        with self._links_lock:
            merges, self._pending_merges = self._pending_merges, []
        for old, new in merges:
            p_old = conn.execute("SELECT person_id FROM tracklets WHERE camera_id=? AND local_id=?",
                                 (self.camera_id, old)).fetchone()
            p_new = conn.execute("SELECT person_id FROM tracklets WHERE camera_id=? AND local_id=?",
                                 (self.camera_id, new)).fetchone()
            p_old = p_old[0] if p_old else None
            p_new = p_new[0] if p_new else None
            if p_old is not None and p_new is None:
                conn.execute("UPDATE tracklets SET person_id=?, link_method='LOCAL_MERGE' WHERE camera_id=? AND "
                             "local_id=?", (p_old, self.camera_id, new))
            elif p_old is not None and p_new is not None and p_old != p_new:
                self._merge_persons(conn, p_old, p_new)
            conn.execute("DELETE FROM tracklets WHERE camera_id=? AND local_id=?", (self.camera_id, old))
            self._sent.pop(old, None)
        conn.commit()

        # 3. refresh the cache of all tracklets (embeddings only when they changed)
        self._refresh_cache(conn, now_wall)
        my_rows = {t.local_id: t for (cam, _), t in self._cache.items() if cam == self.camera_id}

        # 4. impostor statistics: tracklets visible at the same time are different people
        if now_wall - self._last_impostor >= 5.0:
            self._last_impostor = now_wall
            self._sample_impostors(my_rows, now_wall)

        # 5. link my unlinked tracklets (and re-check young NEW persons)
        for lid, t in my_rows.items():
            if lid not in mine:
                continue
            self._link_one(conn, t, mine[lid], now_wall)
        conn.commit()

        # 6. retention
        conn.execute("DELETE FROM tracklets WHERE last_seen_wall < ?", (now_wall - self.retention_sec,))
        conn.commit()

        with self._links_lock:
            self._links = {
                t.local_id: self._resolve_person(conn, t.person_id)
                for (cam, _), t in self._cache.items() if cam == self.camera_id and t.person_id is not None
            }

    def _refresh_cache(self, conn: sqlite3.Connection, now_wall: float) -> None:
        rows = conn.execute(
            "SELECT camera_id, local_id, person_id, label, first_seen_wall, last_seen_wall, active, n_obs, dim, "
            "emb_updated_wall, link_method, stature, stature_se, entry_gate, entry_wall, exit_gate, exit_wall "
            "FROM tracklets WHERE last_seen_wall >= ?",
            (now_wall - self.retention_sec,)).fetchall()
        alive = set()
        for cam, lid, pid, label, fs, ls, active, n, dim, ev, method, st, st_se, eg, ew, xg, xw in rows:
            key = (cam, int(lid))
            alive.add(key)
            t = self._cache.get(key)
            if t is None:
                t = self._cache[key] = Tracklet(cam, int(lid))
            t.person_id, t.label, t.first_seen, t.last_seen = pid, label, fs, ls
            t.active, t.n_obs, t.link_method = bool(active), int(n or 0), method
            t.stature = None if st is None else StatureEstimate(float(st), float(st_se or 0.01), 0)
            t.entry_gate, t.entry_wall, t.exit_gate, t.exit_wall = eg, ew, xg, xw
            if ev is not None and ev > t.emb_version:
                blob = conn.execute("SELECT emb FROM tracklets WHERE camera_id=? AND local_id=?", key).fetchone()
                if blob and blob[0]:
                    t.embs = np.frombuffer(blob[0], dtype=np.float32).reshape(-1, int(dim)).copy()
                    t.emb_version = ev
        for key in [k for k in self._cache if k not in alive]:
            del self._cache[key]

    def _simultaneous(self, a: Tracklet, b: Tracklet) -> bool:
        """Visible at the same time (beyond the tolerance) in cameras that do not overlap."""
        if a.camera_id != b.camera_id and self.topology.overlap(a.camera_id, b.camera_id):
            return False
        return a.first_seen < b.last_seen - self.tol and b.first_seen < a.last_seen - self.tol

    def _sample_impostors(self, my_rows: Dict[int, Tracklet], now_wall: float) -> None:
        mine_active = [t for t in my_rows.values() if t.active and t.embs.size]
        others = [t for t in self._cache.values() if t.active and t.embs.size]
        n = 0
        for a in mine_active:
            for b in others:
                if b.camera_id == a.camera_id and b.local_id <= a.local_id:
                    continue  # itself, or a same-camera pair already counted
                if a.embs.shape[1] != b.embs.shape[1] or not self._simultaneous(a, b):
                    continue
                if a.person_id is not None and a.person_id == b.person_id:
                    continue
                self.open_set.add_impostor(gallery_similarity(a.embs, b.embs))
                n += 1
                if n >= 50:
                    return

    def threshold(self) -> float:
        return max(self.link_threshold, self.open_set.threshold())

    def _candidates(self, t: Tracklet) -> Dict[int, List[Tracklet]]:
        by_person: Dict[int, List[Tracklet]] = {}
        for o in self._cache.values():
            if o.person_id is None or (o.camera_id == t.camera_id and o.local_id == t.local_id):
                continue
            by_person.setdefault(o.person_id, []).append(o)
        out = {}
        for pid, trs in by_person.items():
            if any(self._simultaneous(t, o) for o in trs):
                continue                      # this person was somewhere else at the same time
            labels = {o.label for o in trs if o.label}
            if t.label and labels and t.label not in labels:
                continue                      # different helmet marker
            if t.stature is not None and any(
                    statures_conflict(t.stature, o.stature, self.stature_min_diff, model_error=CROSS_CAMERA_STATURE_ERROR)
                    for o in trs if o.stature is not None):
                continue                      # clearly different height: a different person, whatever the uniform
            remote = [o for o in trs if o.camera_id != t.camera_id]
            if not remote:
                continue                      # only seen in MY camera: re-entries are the local manager's job
            # walking time between the cameras (nearest sighting before, or after, my tracklet)
            before = [o for o in remote if o.last_seen <= t.first_seen + self.tol]
            after = [o for o in remote if o.first_seen >= t.last_seen - self.tol]
            if before:
                last = max(before, key=lambda o: o.last_seen)
                if not self.topology.transit_ok(last.camera_id, t.camera_id, t.first_seen - last.last_seen):
                    continue
            elif after:
                nxt = min(after, key=lambda o: o.first_seen)
                if not self.topology.transit_ok(t.camera_id, nxt.camera_id, nxt.first_seen - t.last_seen):
                    continue
            if not any(o.embs.size for o in trs) and not t.label:
                continue
            out[pid] = trs
        return out

    def _exit_taken(self, exit_tr: Tracklet, t: Tracklet) -> bool:
        """The person who used that exit has already shown up again somewhere else (any camera)."""
        for o in self._cache.values():
            if o is t or o is exit_tr or o.person_id != exit_tr.person_id:
                continue
            if o.first_seen >= exit_tr.exit_wall - self.tol or (o.camera_id == exit_tr.camera_id and o.active):
                return True
        return False

    def _door_handover(self, t: Tracklet) -> Optional[Tuple[int, float, int]]:
        """
        (person, appearance similarity, number of candidates) when `t` appeared at a door that is connected to a door
        of another camera and the person who walked out of it can be named; None otherwise.
        Timing and place carry the decision, appearance only has to agree (light differs between cameras):
          * one person used the connected door in the time window -> that person, unless the look contradicts
          * several -> appearance must pick one with a clear margin, otherwise no decision here
        """
        sources = self.topology.doors_into(t.camera_id, t.entry_gate)
        if not sources or t.entry_wall is None:
            return None
        allowed = self._candidates(t)            # physics, markers, height, camera-level transit
        exits: Dict[int, Tracklet] = {}
        for o in self._cache.values():
            if o.person_id is None or o.exit_gate is None or o.exit_wall is None or o.person_id not in allowed:
                continue
            for cam, gate, lo, hi in sources:
                gap = t.entry_wall - o.exit_wall
                if o.camera_id == cam and o.exit_gate == gate and lo - self.tol <= gap <= hi + self.tol:
                    if not self._exit_taken(o, t):
                        prev = exits.get(o.person_id)
                        if prev is None or o.exit_wall > prev.exit_wall:
                            exits[o.person_id] = o
        if not exits:
            return None
        sims = sorted(
            ((gallery_similarity(t.embs, o.embs) if o.embs.size and o.embs.shape[1] == t.embs.shape[1] else 0.0, pid)
             for pid, o in exits.items()), reverse=True)
        best, pid = sims[0]
        if best < self.door_min_sim:
            return None                          # the look contradicts (e.g. a visitor without uniform)
        if len(sims) > 1 and best - sims[1][0] < self.margin:
            return None                          # several came through and they look alike: no guess
        return pid, best, len(sims)

    def _link_one(self, conn: sqlite3.Connection, t: Tracklet, snap: Dict[str, Any], now_wall: float) -> None:
        young_new = t.link_method == "NEW" and now_wall - t.first_seen <= self.late_merge_sec
        if t.person_id is not None and not young_new and not (t.label and t.link_method != "LABEL"):
            return
        # 1. helmet marker: hard identity
        if t.label:
            row = conn.execute("SELECT person_id FROM persons WHERE label = ? AND merged_into IS NULL "
                               "ORDER BY person_id LIMIT 1", (t.label,)).fetchone()
            if row is not None:
                if t.person_id is not None and row[0] != t.person_id:
                    self._merge_persons(conn, row[0], t.person_id)
                self._set_link(conn, t, self._resolve_person(conn, row[0]), "LABEL", 1.0)
            else:
                if t.person_id is None:
                    self._set_link(conn, t, self._new_person(conn, t.label, now_wall), "LABEL", 1.0)
                else:
                    conn.execute("UPDATE persons SET label = ? WHERE person_id = ?", (t.label, t.person_id))
                    self._set_link(conn, t, t.person_id, "LABEL", 1.0)
            return
        if t.n_obs < self.min_obs or not t.embs.size:
            return
        # 2. door handover: who walked out of the connected door of the other camera a moment ago
        door = self._door_handover(t)
        if door is not None and not (young_new and door[0] == t.person_id):
            pid, sim, n_cand = door
            if young_new and t.person_id is not None:
                pid = self._merge_persons(conn, pid, t.person_id)
            self._set_link(conn, t, pid, "DOOR", sim)
            logger.info("Cross-camera: %s local ID %d -> person P%d through door %s (sim %.3f, %d candidate(s))",
                        self.camera_id, t.local_id, pid, t.entry_gate, sim, n_cand)
            return
        # 3./4. physics + appearance
        scores = []
        for pid, trs in self._candidates(t).items():
            if young_new and pid == t.person_id:
                continue
            embs = [o.embs for o in trs if o.embs.size and o.embs.shape[1] == t.embs.shape[1]]
            if embs:
                scores.append((gallery_similarity(t.embs, np.vstack(embs)), pid))
        scores.sort(reverse=True)
        thr = self.threshold() + (0.03 if young_new else 0.0)
        if scores:
            best, pid = scores[0]
            runner = scores[1][0] if len(scores) > 1 else 0.0
            if best >= thr and best - runner >= self.margin:
                if young_new and t.person_id is not None:
                    # a person created by this camera turned out to be a known one: fold it in
                    pid = self._merge_persons(conn, pid, t.person_id)
                self._set_link(conn, t, pid, "APPEARANCE", best)
                logger.info("Cross-camera: %s local ID %d -> person P%d (sim %.3f, runner-up %.3f)",
                            self.camera_id, t.local_id, pid, best, runner)
                return
        if t.person_id is None and now_wall - t.first_seen >= self.decide_after_sec:
            self._set_link(conn, t, self._new_person(conn, None, now_wall), "NEW",
                           scores[0][0] if scores else 0.0)

    def _set_link(self, conn: sqlite3.Connection, t: Tracklet, pid: int, method: str, score: float) -> None:
        conn.execute("UPDATE tracklets SET person_id=?, link_method=?, link_score=?, updated_wall=? "
                     "WHERE camera_id=? AND local_id=?",
                     (pid, method, float(score), time.time(), t.camera_id, t.local_id))
        t.person_id, t.link_method = pid, method
        self.n_links += 1


def load_person_map(registry_path: str) -> Dict[Tuple[str, int], int]:
    """(camera_id, local_id) -> person_id with merges resolved (used by scripts/report.py)."""
    conn = sqlite3.connect(f"file:{registry_path}?mode=ro", uri=True)
    merged = {int(p): int(m) for p, m in conn.execute(
        "SELECT person_id, merged_into FROM persons WHERE merged_into IS NOT NULL")}
    out = {}
    for cam, lid, pid in conn.execute("SELECT camera_id, local_id, person_id FROM tracklets WHERE person_id IS NOT NULL"):
        seen, p = set(), int(pid)
        while p in merged and p not in seen:
            seen.add(p)
            p = merged[p]
        out[(str(cam), int(lid))] = p
    conn.close()
    return out


def iter_cameras(cameras_cfg: Dict[str, Any]) -> Iterable[str]:
    return [str(c) for c in (cameras_cfg.get("cameras") or {})]
