# app/identity/open_set.py
"""
Self-calibrating re-entry threshold ("how similar do two DIFFERENT workers look at this site?").

With uniforms, the appearance similarity between two different workers is much higher than in public Re-ID
datasets, so a fixed threshold (e.g. 0.82) lets a newcomer inherit the ID of a worker who left - the margin test
against the runner-up does not help when the lookalike is not in memory.

Every time two workers are visible at the same time, clearly apart, their cross-similarity is a sample of the
IMPOSTOR distribution of this camera, collected for free. The re-entry threshold becomes

    max(configured threshold, quantile_q(impostor similarities) + margin)        (capped at max_threshold)

so a uniform site automatically demands more evidence before giving an ID back, while a site with varied clothing
keeps the configured value. The state is persisted with the identity store.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Optional

import numpy as np


class OpenSetCalibrator:
    def __init__(
        self,
        base_threshold: float = 0.82,
        quantile: float = 0.995,
        margin: float = 0.03,
        min_samples: int = 60,
        max_threshold: float = 0.95,
        max_samples: int = 4000,
        enabled: bool = True,
    ):
        self.base_threshold = float(base_threshold)
        self.quantile = float(quantile)
        self.margin = float(margin)
        self.min_samples = int(min_samples)
        self.max_threshold = float(max_threshold)
        self.enabled = bool(enabled)
        self.impostor: Deque[float] = deque(maxlen=int(max_samples))
        self._cached: Optional[float] = None
        self._dirty = True

    def add_impostor(self, sim: float) -> None:
        if np.isfinite(sim):
            self.impostor.append(float(sim))
            self._dirty = True

    def learned_threshold(self) -> Optional[float]:
        if len(self.impostor) < self.min_samples:
            return None
        if self._dirty:
            self._cached = float(np.quantile(np.asarray(self.impostor, dtype=np.float64), self.quantile)) + self.margin
            self._dirty = False
        return self._cached

    def threshold(self) -> float:
        learned = self.learned_threshold() if self.enabled else None
        if learned is None:
            return self.base_threshold
        return float(min(self.max_threshold, max(self.base_threshold, learned)))

    def stats(self) -> Dict[str, Any]:
        arr = np.asarray(self.impostor, dtype=np.float64)
        return {
            "enabled": self.enabled,
            "impostor_samples": int(arr.size),
            "impostor_p50": None if arr.size == 0 else round(float(np.median(arr)), 4),
            "impostor_p99": None if arr.size == 0 else round(float(np.quantile(arr, 0.99)), 4),
            "learned_threshold": None if self.learned_threshold() is None else round(self.learned_threshold(), 4),
            "effective_threshold": round(self.threshold(), 4),
            "base_threshold": self.base_threshold,
        }

    def state(self) -> Dict[str, Any]:
        return {"impostor": [round(x, 5) for x in self.impostor]}

    def load_state(self, state: Optional[Dict[str, Any]]) -> None:
        for x in (state or {}).get("impostor", []):
            self.add_impostor(float(x))
