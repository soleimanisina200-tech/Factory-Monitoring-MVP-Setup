# app/logic/calibration.py
"""
Single source of truth for the ground-plane calibration (pixel -> meters).

`scripts/calibrate_ground_plane.py` writes `configs/calibration.yaml`; `main.py` (persons + vehicles) and the
standalone vehicle runner read it. The file stores the resolution the points were clicked at, so points can
be rescaled exactly to any stream resolution (no guessing).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml

DEFAULT_CALIBRATION_PATH = "configs/calibration.yaml"


@dataclass
class GroundCalibration:
    image_points_px: List[List[float]]
    world_points_m: List[List[float]]
    image_wh: Tuple[int, int]

    def scaled_image_points(self, stream_wh: Tuple[int, int]) -> np.ndarray:
        """Image points rescaled from the calibration resolution to `stream_wh`."""
        src = np.array(self.image_points_px, dtype=np.float32)
        rw, rh = self.image_wh
        sw, sh = stream_wh
        if (rw, rh) != (sw, sh) and rw > 0 and rh > 0:
            src[:, 0] *= float(sw) / float(rw)
            src[:, 1] *= float(sh) / float(rh)
        return src

    def homography(self, stream_wh: Tuple[int, int]) -> Optional[np.ndarray]:
        src = self.scaled_image_points(stream_wh)
        dst = np.array(self.world_points_m, dtype=np.float32)
        if len(src) < 4 or len(src) != len(dst):
            return None
        H, _ = cv2.findHomography(src, dst, method=0)
        return H


def load_calibration(path: str = DEFAULT_CALIBRATION_PATH) -> Optional[GroundCalibration]:
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    try:
        img = data["image_points_px"]
        world = data["world_points_m"]
        wh = data["image_wh"]
    except KeyError as e:
        raise ValueError(f"{path}: missing key {e}") from e
    if len(img) < 4 or len(img) != len(world):
        raise ValueError(f"{path}: need >= 4 matching image/world points")
    return GroundCalibration(img, world, (int(wh[0]), int(wh[1])))


def save_calibration(cal: GroundCalibration, path: str = DEFAULT_CALIBRATION_PATH) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(
            "# Ground-plane calibration written by scripts/calibrate_ground_plane.py\n"
            "# image_wh = resolution the points were clicked at; points are rescaled to the live stream.\n"
        )
        yaml.safe_dump(
            {
                "image_wh": [int(cal.image_wh[0]), int(cal.image_wh[1])],
                "image_points_px": [[float(x), float(y)] for x, y in cal.image_points_px],
                "world_points_m": [[float(x), float(y)] for x, y in cal.world_points_m],
            },
            f,
            sort_keys=False,
        )
