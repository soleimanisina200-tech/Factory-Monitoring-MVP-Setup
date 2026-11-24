# app/identity/stature.py
"""
Relative body height ("stature") as a soft biometric that does not depend on clothing or light.

Workers in the same uniform look alike to an appearance model, but they are not all the same height: the standard
deviation of adult height is ~7 cm, so two random workers differ by ~10 cm. Height is therefore used as a VETO only:
two observations whose heights clearly differ are two different people; similar heights prove nothing.

No camera calibration is needed. On a flat floor the standing height of a person in pixels is (to a good
approximation) a linear function of where his feet are in the image:  h_px ~ a*u + b*v + c.  The camera learns
a, b, c from everybody who walks through it (StatureModel), and a person's stature is the median of
    ratio = measured h_px / predicted h_px at his feet
over many upright, fully visible frames (1.00 = average worker of this site, 1.05 = 5 % taller). The ratio is
relative to the workforce, so it is comparable between cameras too.

Only frames where the measurement is trustworthy count: ankles, knees, hips, shoulders and head visible, straight
legs, upright torso, arms not raised above the head, box not cut by the image border, nobody overlapping.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

# COCO-17 keypoints
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SH, R_SH, L_WR, R_WR = 5, 6, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 11, 12, 13, 14, 15, 16


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at b (degrees) of the polyline a-b-c."""
    v1, v2 = a - b, c - b
    n = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if n < 1e-6:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(float(np.dot(v1, v2)) / n, -1.0, 1.0))))


def measure_upright_height(
    bbox: Sequence[float], kxy: Optional[np.ndarray], kconf: Optional[np.ndarray], frame_w: int, frame_h: int,
    kpt_thresh: float = 0.5, min_height_px: float = 60.0, min_leg_angle: float = 160.0, max_torso_tilt: float = 15.0,
) -> Optional[Tuple[float, float, float]]:
    """
    (u_foot, v_foot, height_px) of a person standing upright and fully visible, or None when this frame cannot
    give a trustworthy height (bending, sitting, walking with a bent knee, cut by the border, arms up, ...).
    Height = lowest ankle -> top of the box (helmet included: the same for everybody).
    """
    if kxy is None or kconf is None or len(kconf) < 17:
        return None
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    if x1 <= 2 or y1 <= 2 or x2 >= frame_w - 2 or y2 >= frame_h - 2:
        return None
    need = (L_SH, R_SH, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK)
    if any(kconf[i] < kpt_thresh for i in need):
        return None
    if not (kconf[NOSE] >= kpt_thresh or (kconf[L_EYE] >= kpt_thresh and kconf[R_EYE] >= kpt_thresh)
            or (kconf[L_EAR] >= kpt_thresh and kconf[R_EAR] >= kpt_thresh)):
        return None
    p = np.asarray(kxy, dtype=np.float64)
    # the STANCE leg (lowest ankle, carries the body) must be straight; the swing leg of a walking person may bend
    stance = (L_HIP, L_KNEE, L_ANK) if p[L_ANK][1] >= p[R_ANK][1] else (R_HIP, R_KNEE, R_ANK)
    if _angle_deg(p[stance[0]], p[stance[1]], p[stance[2]]) < min_leg_angle:
        return None
    sh_mid, hip_mid = 0.5 * (p[L_SH] + p[R_SH]), 0.5 * (p[L_HIP] + p[R_HIP])
    torso = sh_mid - hip_mid
    if np.linalg.norm(torso) < 1e-6 or torso[1] >= 0:
        return None
    tilt = float(np.degrees(np.arctan2(abs(torso[0]), -torso[1])))
    if tilt > max_torso_tilt:
        return None
    head_y = min(float(p[i][1]) for i in (NOSE, L_EYE, R_EYE, L_EAR, R_EAR) if kconf[i] >= kpt_thresh)
    for w in (L_WR, R_WR):                     # a raised arm lifts the top of the box above the head
        if kconf[w] >= kpt_thresh and p[w][1] < head_y:
            return None
    v_foot = max(float(p[L_ANK][1]), float(p[R_ANK][1]))
    h = v_foot - y1
    if h < min_height_px or head_y - y1 > 0.25 * h:   # box top far above the head: something else in the box
        return None
    u_foot = 0.5 * float(p[L_ANK][0] + p[R_ANK][0])
    return u_foot, v_foot, h


class StatureModel:
    """Expected upright height (px) of an average person at a foot position, learned from the camera's own traffic."""

    def __init__(self, min_samples: int = 150, min_tracks: int = 4, max_samples: int = 5000, refit_every: int = 300,
                 per_track_cap: int = 20):
        self.min_samples = int(min_samples)
        self.min_tracks = int(min_tracks)
        self.samples: Deque[Tuple[float, float, float, int]] = deque(maxlen=int(max_samples))
        self.refit_every = int(refit_every)
        # every person counts the same in the fit: somebody who stands long in one spot must not "teach" the camera
        # that his own height is the average there (that would hide exactly the difference we want to see)
        self.per_track_cap = int(per_track_cap)
        self.coef: Optional[np.ndarray] = None
        self.v_range: Tuple[float, float] = (0.0, 0.0)
        self.u_range: Tuple[float, float] = (0.0, 0.0)
        self.spread: float = 0.0          # robust std of the ratio across people (population + noise)
        self._since_fit = 0

    @property
    def ready(self) -> bool:
        return self.coef is not None

    def add(self, u: float, v: float, h: float, track_id: int) -> None:
        self.samples.append((float(u), float(v), float(h), int(track_id)))
        self._since_fit += 1
        if self._since_fit >= self.refit_every or (self.coef is None and len(self.samples) >= self.min_samples):
            self.fit()

    def fit(self) -> bool:
        self._since_fit = 0
        if len(self.samples) < self.min_samples or len({s[3] for s in self.samples}) < self.min_tracks:
            return False
        per_track: Dict[int, List[Tuple[float, float, float, int]]] = {}
        for s in reversed(self.samples):                 # newest first
            lst = per_track.setdefault(s[3], [])
            if len(lst) < self.per_track_cap:
                lst.append(s)
        arr = np.asarray([s for lst in per_track.values() for s in lst], dtype=np.float64)
        X = np.column_stack([arr[:, 0], arr[:, 1], np.ones(len(arr))])
        y = arr[:, 2]
        keep = np.ones(len(arr), dtype=bool)
        coef = None
        for _ in range(3):                     # trimmed least squares: outliers (bad poses) do not bend the fit
            coef, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
            rel = y / np.maximum(X @ coef, 1.0) - 1.0
            mad = float(np.median(np.abs(rel[keep] - np.median(rel[keep])))) * 1.4826 + 1e-4
            keep = np.abs(rel) <= 3.0 * mad
            if keep.sum() < len(arr) // 2:
                break
        if coef is None:
            return False
        self.coef = coef
        self.u_range = (float(arr[keep, 0].min()), float(arr[keep, 0].max()))
        self.v_range = (float(arr[keep, 1].min()), float(arr[keep, 1].max()))
        rel = y[keep] / np.maximum(X[keep] @ coef, 1.0)
        self.spread = float(np.median(np.abs(rel - np.median(rel))) * 1.4826)
        return True

    def predict(self, u: float, v: float) -> Optional[float]:
        """Expected height at (u, v); None at depths (image rows) not seen so far: no extrapolation in depth, where
        perspective changes the size quickly. Sideways (u) the size hardly changes, so that is allowed."""
        if self.coef is None:
            return None
        v_lo, v_hi = self.v_range
        pad_v = 0.15 * max(v_hi - v_lo, 40.0)
        if not (v_lo - pad_v <= v <= v_hi + pad_v):
            return None
        h = float(self.coef[0] * u + self.coef[1] * v + self.coef[2])
        return h if h > 10.0 else None

    def ratio(self, u: float, v: float, h: float) -> Optional[float]:
        exp = self.predict(u, v)
        return None if exp is None else float(h) / exp

    # persistence
    def state(self) -> dict:
        return {"coef": None if self.coef is None else [float(c) for c in self.coef],
                "u_range": list(self.u_range), "v_range": list(self.v_range), "spread": self.spread}

    def load_state(self, st: Optional[dict]) -> None:
        if not st or not st.get("coef"):
            return
        self.coef = np.asarray(st["coef"], dtype=np.float64)
        self.u_range = tuple(st.get("u_range", (0.0, 0.0)))
        self.v_range = tuple(st.get("v_range", (0.0, 0.0)))
        self.spread = float(st.get("spread", 0.0))


@dataclass
class StatureEstimate:
    ratio: float      # median stature ratio (1.0 = average worker of this site)
    se: float         # standard error of that median
    n: int


class StatureSamples:
    """Raw height measurements of one person/track. Ratios are computed with the CURRENT model when needed."""

    def __init__(self, maxlen: int = 60):
        self.raw: Deque[Tuple[float, float, float]] = deque(maxlen=int(maxlen))

    def add(self, u: float, v: float, h: float) -> None:
        self.raw.append((float(u), float(v), float(h)))

    def extend(self, other: "StatureSamples") -> None:
        for s in other.raw:
            self.raw.append(s)

    def estimate(self, model: StatureModel, min_n: int = 8) -> Optional[StatureEstimate]:
        if not model.ready or len(self.raw) < min_n:
            return None
        r = [model.ratio(u, v, h) for u, v, h in self.raw]
        r = np.asarray([x for x in r if x is not None], dtype=np.float64)
        if len(r) < min_n:
            return None
        med = float(np.median(r))
        # floor: a few lucky-similar samples must not make a person look more precise than a pose model can be
        sigma = max(float(np.median(np.abs(r - med)) * 1.4826), 0.02)
        return StatureEstimate(med, 1.253 * sigma / np.sqrt(len(r)), len(r))

    def to_list(self) -> List[List[float]]:
        return [list(s) for s in self.raw]

    @classmethod
    def from_list(cls, rows, maxlen: int = 60) -> "StatureSamples":
        out = cls(maxlen)
        for row in rows or []:
            if len(row) == 3:
                out.add(*row)
        return out


def statures_conflict(a: Optional[StatureEstimate], b: Optional[StatureEstimate], min_diff: float = 0.05,
                      z: float = 3.0, model_error: float = 0.01) -> bool:
    """True when the two heights CLEARLY differ (=> two different people). Unknown height never conflicts."""
    if a is None or b is None:
        return False
    diff = abs(a.ratio - b.ratio)
    se = float(np.sqrt(a.se ** 2 + b.se ** 2 + model_error ** 2))
    return diff >= min_diff and diff >= z * se
