# app/identity/finetune.py
"""
Fine-tuning the Re-ID backbone on the site's own footage - without labelling anybody by hand.

Why: the public OSNet weights were trained on street pedestrians in all kinds of clothes. In a factory everybody wears
the same uniform, and two different workers score ~0.85 with each other, close to the same worker seen twice. A model
trained on THIS site's people learns the details that do differ (face/hair, shoes, build, how the uniform is worn).

Self-supervision from the video itself (scripts/mine_reid_data.py):
  * positives: crops of one clean SEGMENT of a track (the tracker followed one person; a segment ends whenever the
    person overlaps somebody else or disappears, because the tracker may swap people right there);
  * negatives: segments visible AT THE SAME TIME are certainly different people - in a factory these are exactly the
    hard pairs (same uniform, same light).
Training uses batch-hard triplet loss inside such co-occurring groups only (two segments from different moments may
be the same worker, so they are never pushed apart), plus an L2-SP penalty that keeps the weights close to the
pre-trained ones (a few hours of video must not make the model forget what it knew).

Evaluation (the number that matters for this project): the threshold that rejects 99.5 % of different-worker pairs,
and the share of same-worker pairs still accepted at that threshold ("genuine accept rate").
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------------------------- mining: segments
@dataclass
class _OpenSegment:
    seg_id: str
    last_ts: float
    n: int = 0


class SegmentBuilder:
    """
    Splits tracker tracks into clean segments of ONE person. A segment ends when the track is not clean (overlapping
    another person / occluded) or unseen for more than `max_gap_sec` - the tracker may have swapped identities there.
    """

    def __init__(self, prefix: str, max_gap_sec: float = 1.0):
        self.prefix = prefix
        self.max_gap_sec = float(max_gap_sec)
        self._open: Dict[int, _OpenSegment] = {}
        self._counter = 0

    def update(self, ts: float, tracks: Iterable[Tuple[int, bool]]) -> Dict[int, str]:
        """tracks: (track id, clean?) of one frame. Returns track id -> segment id for the CLEAN tracks."""
        out: Dict[int, str] = {}
        for tid, clean in tracks:
            seg = self._open.get(tid)
            if not clean:
                self._open.pop(tid, None)          # whatever comes after the overlap is a new segment
                continue
            if seg is None or ts - seg.last_ts > self.max_gap_sec:
                self._counter += 1
                seg = self._open[tid] = _OpenSegment(f"{self.prefix}_{self._counter:06d}", ts)
            seg.last_ts = ts
            seg.n += 1
            out[tid] = seg.seg_id
        for tid in [t for t, s in self._open.items() if ts - s.last_ts > self.max_gap_sec]:
            del self._open[tid]
        return out


# ---------------------------------------------------------------------------------------------- dataset
@dataclass
class MinedData:
    root: Path
    segments: Dict[str, List[Tuple[str, str, float]]] = field(default_factory=dict)   # seg -> (body, upper, ts)
    video_of: Dict[str, str] = field(default_factory=dict)
    groups: List[List[str]] = field(default_factory=list)          # segments visible at the same time

    @classmethod
    def load(cls, root: str, min_samples: int = 4) -> "MinedData":
        d = cls(Path(root))
        with open(d.root / "samples.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                d.segments.setdefault(r["seg"], []).append((r["body"], r["upper"], float(r["ts"])))
                d.video_of[r["seg"]] = r["video"]
        d.segments = {s: v for s, v in d.segments.items() if len(v) >= min_samples}
        seen: Set[Tuple[str, ...]] = set()
        with open(d.root / "cooccurrence.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                segs = tuple(sorted(s for s in json.loads(line)["segs"] if s in d.segments))
                if len(segs) >= 2 and segs not in seen:
                    seen.add(segs)
                    d.groups.append(list(segs))
        return d

    def split_by_video(self, val_fraction: float = 0.2, seed: int = 0) -> Tuple["MinedData", "MinedData"]:
        """Hold out whole videos (never frames of a video that is also trained on)."""
        videos = sorted(set(self.video_of.values()))
        rng = random.Random(seed)
        rng.shuffle(videos)
        n_val = max(1, int(round(val_fraction * len(videos)))) if len(videos) > 1 else 0
        val_videos = set(videos[:n_val])

        def subset(keep) -> "MinedData":
            m = MinedData(self.root)
            m.segments = {s: v for s, v in self.segments.items() if keep(self.video_of[s])}
            m.video_of = {s: self.video_of[s] for s in m.segments}
            m.groups = [g for g in self.groups if all(s in m.segments for s in g)]
            return m

        if not val_videos:      # a single video: split it in time instead (first 80 % / last 20 %)
            t_all = sorted(ts for v in self.segments.values() for _, _, ts in v)
            cut = t_all[int(len(t_all) * (1.0 - val_fraction))] if t_all else 0.0
            first = {s for s, v in self.segments.items() if max(x[2] for x in v) < cut}
            train, val = MinedData(self.root), MinedData(self.root)
            for m, segs in ((train, first), (val, set(self.segments) - first)):
                m.segments = {s: self.segments[s] for s in segs}
                m.video_of = {s: self.video_of[s] for s in segs}
                m.groups = [g for g in self.groups if all(s in segs for s in g)]
            return train, val
        return subset(lambda v: v not in val_videos), subset(lambda v: v in val_videos)


def sample_group_batch(data: MinedData, rng: random.Random, n_groups: int = 8, p: int = 4, k: int = 4
                       ) -> List[List[Tuple[str, List[int]]]]:
    """n_groups co-occurrence groups, each with up to p segments and k sample indices per segment (spread in time)."""
    out = []
    for _ in range(n_groups):
        g = rng.choice(data.groups)
        segs = rng.sample(g, min(p, len(g)))
        item = []
        for s in segs:
            n = len(data.segments[s])
            idx = sorted(rng.sample(range(n), k)) if n >= k else [rng.randrange(n) for _ in range(k)]
            item.append((s, idx))
        out.append(item)
    return out


# ---------------------------------------------------------------------------------------------- losses / metrics
def batch_hard_triplet(feats, labels, margin: float = 0.3):
    """Batch-hard triplet loss on L2-normalised features (torch tensors). labels: (N,) segment indices."""
    import torch
    import torch.nn.functional as F

    f = F.normalize(feats, dim=1)
    dist = 2.0 - 2.0 * (f @ f.t())                                 # squared euclidean on the unit sphere
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool, device=feats.device)
    pos = dist.masked_fill(~same | eye, -1.0).max(dim=1).values      # hardest positive
    neg = dist.masked_fill(same, 1e9).min(dim=1).values              # hardest negative
    valid = (same & ~eye).any(dim=1) & (~same).any(dim=1)
    if not bool(valid.any()):
        return feats.sum() * 0.0
    return F.relu(pos - neg + margin)[valid].mean()


def l2sp_penalty(model, reference: Dict[str, "object"]):
    """Sum of squared distances to the pre-trained weights (keeps the backbone close to what it knew)."""
    total = None
    for name, p in model.named_parameters():
        if p.requires_grad and name in reference:
            d = ((p - reference[name]) ** 2).sum()
            total = d if total is None else total + d
    return total if total is not None else 0.0


def separation_report(genuine: Sequence[float], impostor: Sequence[float], quantile: float = 0.995,
                      margin: float = 0.03) -> Dict[str, float]:
    """
    threshold = impostor quantile + margin (the open-set rule of the identity manager), and how many same-person
    pairs still pass it. Higher genuine_accept = the model separates this site's workers better.
    """
    g = np.asarray(genuine, dtype=np.float64)
    i = np.asarray(impostor, dtype=np.float64)
    if len(g) == 0 or len(i) == 0:
        return {"n_genuine": float(len(g)), "n_impostor": float(len(i))}
    thr = float(min(0.99, np.quantile(i, quantile) + margin))
    ths = np.linspace(-1.0, 1.0, 401)
    far = np.array([(i >= t).mean() for t in ths])
    frr = np.array([(g < t).mean() for t in ths])
    k = int(np.argmin(np.abs(far - frr)))
    return {
        "n_genuine": float(len(g)), "n_impostor": float(len(i)),
        "genuine_median": float(np.median(g)), "impostor_median": float(np.median(i)),
        "impostor_q": float(np.quantile(i, quantile)), "threshold": thr,
        "genuine_accept": float((g >= thr).mean()), "eer": float(0.5 * (far[k] + frr[k])),
    }


def pair_similarities(data: MinedData, embed, max_groups: int = 400, seed: int = 0
                      ) -> Tuple[List[float], List[float]]:
    """
    genuine: first half vs second half of the same segment; impostor: segments visible at the same time.
    `embed(list of (body_path, upper_path)) -> (N, D) L2-normalised numpy array`.
    """
    rng = random.Random(seed)
    halves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for s, smp in data.segments.items():
        smp = sorted(smp, key=lambda x: x[2])
        if len(smp) < 4:
            continue
        parts = []
        for half in (smp[: len(smp) // 2], smp[len(smp) // 2:]):
            e = embed([(str(data.root / b), str(data.root / u)) for b, u, _ in half]).mean(axis=0)
            parts.append(e / (np.linalg.norm(e) + 1e-8))
        halves[s] = (parts[0], parts[1])
    genuine = [float(ea @ eb) for ea, eb in halves.values()]
    impostor = []
    groups = list(data.groups)
    rng.shuffle(groups)
    seen = set()
    for g in groups[:max_groups]:
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                key = (min(g[i], g[j]), max(g[i], g[j]))
                if key in seen or g[i] not in halves or g[j] not in halves:
                    continue
                seen.add(key)
                impostor.append(float(halves[g[i]][0] @ halves[g[j]][1]))
    return genuine, impostor
