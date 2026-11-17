# app/core/scene.py
"""
Scene geometry of one camera: zones, tripwire lines, directional regions and exit doors.

Stored in its own file (configs/scene.yaml) instead of settings.yaml: the drawing tools rewrite it freely while
settings.yaml stays hand-edited (with its comments). The file records the resolution the shapes were drawn at
(`image_wh`) so every coordinate - and every pixel-sized parameter - is rescaled exactly to the live stream.

    image_wh: [2560, 1920]
    zones:
      loading_dock: {polygon: [[x, y], ...], type: presence, max_dwell_sec: 0, max_occupancy: 0}
      forklift_aisle: {polygon: [...], type: restricted}
    lines:
      main_gate: {a: [x, y], b: [x, y], invert: false, labels: [IN, OUT], dead_band_px: 18, cooldown_sec: 1.0}
    regions:
      entrance: {polygon: [...], entry_vector: [0.84, 0.54], min_displacement_px: 40, labels: [IN, OUT]}
    exit_regions:
      side_door: {polygon: [...]}

Legacy definitions in settings.yaml (zones.definitions, directional_tracking.lines / entrance_zones) are still
read; they carry no reference resolution and are used unscaled.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

DEFAULT_SCENE_PATH = "configs/scene.yaml"
SECTIONS = ("zones", "lines", "regions", "exit_regions")


@dataclass
class Scene:
    image_wh: Optional[Tuple[int, int]] = None
    zones: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    lines: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    regions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    exit_regions: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not any(getattr(self, s) for s in SECTIONS)

    def scaled(self, stream_wh: Tuple[int, int]) -> "Scene":
        """Copy of the scene in stream pixel coordinates."""
        out = copy.deepcopy(self)
        if self.image_wh is None:
            return out
        rw, rh = self.image_wh
        sw, sh = stream_wh
        if (rw, rh) == (sw, sh) or rw <= 0 or rh <= 0:
            return out
        fx, fy = float(sw) / float(rw), float(sh) / float(rh)
        f_len = 0.5 * (fx + fy)

        def sp(pts: Any) -> List[List[float]]:
            arr = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
            arr[:, 0] *= fx
            arr[:, 1] *= fy
            return arr.tolist()

        for section in SECTIONS:
            for spec in getattr(out, section).values():
                if spec.get("_unscaled"):
                    continue
                if "polygon" in spec:
                    spec["polygon"] = sp(spec["polygon"])
                for key in ("a", "b"):
                    if key in spec:
                        spec[key] = sp([spec[key]])[0]
                if "entry_vector" in spec:
                    vx, vy = (float(v) for v in spec["entry_vector"])
                    vx, vy = vx * fx, vy * fy
                    n = float(np.hypot(vx, vy)) or 1.0
                    spec["entry_vector"] = [vx / n, vy / n]
                for key in ("dead_band_px", "min_displacement_px"):
                    if key in spec:
                        spec[key] = float(spec[key]) * f_len
        return out


def _legacy_from_settings(settings: Dict[str, Any]) -> Scene:
    scene = Scene()
    for name, z in ((settings.get("zones") or {}).get("definitions") or {}).items():
        if z and len(z.get("polygon", [])) >= 3:
            scene.zones[name] = {"polygon": z["polygon"], "type": z.get("type", "presence"), "_unscaled": True}
    dir_cfg = settings.get("directional_tracking") or {}
    for name, ln in (dir_cfg.get("lines") or {}).items():
        if not ln:
            continue
        scene.lines[name] = {
            "a": ln.get("a", ln.get("point_a")),
            "b": ln.get("b", ln.get("point_b")),
            "invert": bool(ln.get("invert", ln.get("invert_direction", False))),
            "cooldown_sec": float(ln.get("cooldown_sec", ln.get("cooldown_seconds", 1.0))),
            "_unscaled": True,
        }
    for name, rg in (dir_cfg.get("entrance_zones") or {}).items():
        if rg and len(rg.get("polygon", [])) >= 3:
            scene.regions[name] = {
                "polygon": rg["polygon"],
                "entry_vector": rg.get("entry_vector", rg.get("entry_direction_vector", [0.0, -1.0])),
                "min_displacement_px": float(rg.get("min_displacement_px", rg.get("min_displacement", 20.0))),
                "_unscaled": True,
            }
    return scene


def validate_scene(scene: Scene) -> List[str]:
    """Human-readable problems (empty list = OK)."""
    problems = []
    for section in ("zones", "regions", "exit_regions"):
        for name, spec in getattr(scene, section).items():
            if len(spec.get("polygon") or []) < 3:
                problems.append(f"{section}.{name}: polygon needs >= 3 points")
    for name, spec in scene.lines.items():
        a, b = spec.get("a"), spec.get("b")
        if a is None or b is None or len(a) != 2 or len(b) != 2:
            problems.append(f"lines.{name}: needs points a and b")
        elif float(np.hypot(b[0] - a[0], b[1] - a[1])) < 5.0:
            problems.append(f"lines.{name}: points a and b are too close")
    for name, spec in scene.zones.items():
        if spec.get("type", "presence") not in ("presence", "restricted"):
            problems.append(f"zones.{name}: type must be 'presence' or 'restricted'")
    return problems


def load_scene(path: Optional[str] = DEFAULT_SCENE_PATH, settings: Optional[Dict[str, Any]] = None) -> Scene:
    scene = Scene()
    if path and Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        wh = data.get("image_wh")
        scene.image_wh = (int(wh[0]), int(wh[1])) if wh else None
        for section in SECTIONS:
            setattr(scene, section, {str(k): dict(v or {}) for k, v in (data.get(section) or {}).items()})
    if settings:
        legacy = _legacy_from_settings(settings)
        for section in SECTIONS:
            target = getattr(scene, section)
            for name, spec in getattr(legacy, section).items():
                target.setdefault(name, spec)
    problems = validate_scene(scene)
    if problems:
        raise ValueError(f"Invalid scene definition ({path}): " + "; ".join(problems))
    return scene


def save_scene(scene: Scene, path: str = DEFAULT_SCENE_PATH) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data: Dict[str, Any] = {"image_wh": list(scene.image_wh) if scene.image_wh else None}
    for section in SECTIONS:
        data[section] = {
            name: {k: v for k, v in spec.items() if not k.startswith("_")}
            for name, spec in getattr(scene, section).items()
            if not spec.get("_unscaled")  # legacy shapes stay in settings.yaml
        }
    with open(p, "w", encoding="utf-8") as f:
        f.write(
            "# Scene geometry written by scripts/draw_zone.py and scripts/draw_line.py.\n"
            "# image_wh = resolution the shapes were drawn at; they are rescaled to the live stream.\n"
        )
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True, default_flow_style=None)
