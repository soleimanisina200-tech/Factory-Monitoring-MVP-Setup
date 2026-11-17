# app/identity/markers.py
"""
Optional hard identity: a printed ArUco marker on each worker's helmet (or vest back).

Uniforms make appearance-only re-identification fundamentally ambiguous, especially after long absences. A small
fiducial marker is the cheapest reliable identity signal for a fixed CCTV camera: when a worker's marker is read
(twice within a few seconds), the identity manager binds the track to that worker - across exits, re-entries,
restarts - and two identities carrying different markers can never be merged.

    configs/workers.yaml
        dictionary: DICT_4X4_50
        workers:
          - {marker_id: 7, name: "Ali Rezaei", employee_id: "E-1043"}

Print markers with scripts/make_markers.py. Rule of thumb: the marker must appear >= ~20 px wide in the image
(e.g. 10 cm marker, 4 MP camera, up to ~8-10 m distance); use one per side of the helmet for all-round reading.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import yaml

logger = logging.getLogger("FactoryReID")


class WorkerRegistry:
    def __init__(self, path: Optional[str] = None, accept_unregistered: bool = False):
        self.by_marker: Dict[int, Dict[str, str]] = {}
        self.dictionary = "DICT_4X4_50"
        self.accept_unregistered = bool(accept_unregistered)
        if path and Path(path).exists():
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            self.dictionary = str(data.get("dictionary", self.dictionary))
            for w in data.get("workers", []) or []:
                mid = int(w["marker_id"])
                self.by_marker[mid] = {k: str(v) for k, v in w.items() if v is not None}
        elif path:
            logger.warning("Worker registry %s not found: markers will be reported as 'marker:<id>'", path)

    def label_for(self, marker_id: int) -> Optional[str]:
        w = self.by_marker.get(int(marker_id))
        if w is not None:
            return w.get("employee_id") or w.get("name") or f"marker:{marker_id}"
        return f"marker:{marker_id}" if self.accept_unregistered or not self.by_marker else None

    def display_name(self, label: Optional[str]) -> Optional[str]:
        if label is None:
            return None
        for w in self.by_marker.values():
            if label in (w.get("employee_id"), w.get("name")):
                return w.get("name") or label
        return label


class HelmetMarkerReader:
    def __init__(self, registry: WorkerRegistry, dictionary: Optional[str] = None, min_person_height_px: int = 120,
                 min_marker_px: int = 64):
        name = dictionary or registry.dictionary
        if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, name):
            raise RuntimeError(f"OpenCV ArUco dictionary '{name}' is not available (opencv-python >= 4.7 needed)")
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name)), params)
        self.registry = registry
        self.min_person_height_px = int(min_person_height_px)
        self.min_marker_px = int(min_marker_px)

    def _head_region(self, bbox: np.ndarray, kxy: np.ndarray, kconf: np.ndarray, w: int, h: int):
        x1, y1, x2, y2 = (float(v) for v in bbox[:4])
        bh = y2 - y1
        vis = [i for i in range(7) if kconf[i] >= 0.3]  # nose, eyes, ears, shoulders
        if len(vis) >= 2:
            pts = kxy[vis]
            cx = float(np.mean(pts[:, 0]))
            top = float(np.min(pts[:, 1]))
            half = max(0.25 * (x2 - x1), 0.12 * bh)
            rx1, rx2 = cx - 1.4 * half, cx + 1.4 * half
            ry1, ry2 = top - 1.6 * half, top + 1.2 * half
        else:
            rx1, rx2, ry1, ry2 = x1, x2, y1 - 0.05 * bh, y1 + 0.40 * bh
        rx1, ry1 = int(max(0, rx1)), int(max(0, ry1))
        rx2, ry2 = int(min(w, rx2)), int(min(h, ry2))
        return rx1, ry1, rx2, ry2

    def read(self, frame: np.ndarray, bbox: np.ndarray, kxy: np.ndarray, kconf: np.ndarray) -> Optional[str]:
        """Worker label of the marker visible on this person's helmet/upper body, or None."""
        h, w = frame.shape[:2]
        if float(bbox[3] - bbox[1]) < self.min_person_height_px:
            return None
        x1, y1, x2, y2 = self._head_region(bbox, kxy, kconf, w, h)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        scale = max(1.0, self.min_marker_px * 3.0 / max(1, min(crop.shape[:2])))
        if scale > 1.0:
            crop = cv2.resize(crop, None, fx=min(scale, 4.0), fy=min(scale, 4.0), interpolation=cv2.INTER_CUBIC)
        corners, ids, _ = self.detector.detectMarkers(crop)
        if ids is None or len(ids) == 0:
            return None
        areas = [float(cv2.contourArea(c.reshape(-1, 2).astype(np.float32))) for c in corners]
        mid = int(np.asarray(ids).reshape(-1)[int(np.argmax(areas))])   # (N,1) in OpenCV 4, (N,) in OpenCV 5
        return self.registry.label_for(mid)
