# scripts/draw_zone.py
"""
Interactive editor for zones and exit doors (configs/scene.yaml) + optional ground calibration
(configs/calibration.yaml). settings.yaml is no longer rewritten (its comments survive).

    python scripts/draw_zone.py [--source <video|rtsp|image>] [--scene configs/scene.yaml]

Keys:  SPACE freeze frame (live preview) | TAB/m zone <-> calibration mode | t zone type (presence / restricted /
       exit door) | L-click add point | R-click undo | ENTER close polygon & name it | d delete last shape |
       r reset calibration points | v calibration preview | s save | q quit
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core.scene import DEFAULT_SCENE_PATH, Scene, load_scene, save_scene  # noqa: E402
from app.logic.calibration import (  # noqa: E402
    DEFAULT_CALIBRATION_PATH,
    GroundCalibration,
    load_calibration,
    save_calibration,
)

ZONE_TYPES = ("presence", "restricted", "exit_door")
TYPE_COLORS = {"presence": (0, 200, 60), "restricted": (0, 0, 255), "exit_door": (255, 120, 0)}


class SceneZoneEditor:
    MODE_ZONE = "ZONE"
    MODE_BEV = "CALIBRATION"

    def __init__(self, frame: np.ndarray, scene_path: str, calibration_path: str):
        self.base_frame = frame.copy()
        self.h, self.w = frame.shape[:2]
        self.scene_path = scene_path
        self.calibration_path = calibration_path
        scene = load_scene(scene_path) if Path(scene_path).exists() else Scene()
        self.scene = scene.scaled((self.w, self.h))   # edit in THIS frame's pixels
        self.scene.image_wh = (self.w, self.h)
        self.mode = self.MODE_ZONE
        self.zone_type = "presence"
        self.mouse_xy: Tuple[int, int] = (0, 0)
        self.current: List[List[int]] = []
        cal = load_calibration(calibration_path)
        self.bev_src: List[List[float]] = cal.scaled_image_points((self.w, self.h)).tolist() if cal else []
        self.bev_w_m, self.bev_l_m = 6.0, 8.0
        if cal is not None:
            dst = np.asarray(cal.world_points_m)
            self.bev_w_m, self.bev_l_m = float(dst[:, 0].max()) or 6.0, float(dst[:, 1].max()) or 8.0
        self.input_stage: Optional[str] = None
        self.input_prompt = ""
        self.input_buffer = ""
        self.status = "TAB: mode | t: zone type | L-click: point | ENTER: close zone | s: save"

    # ------------------------------------------------------------------
    def shapes(self) -> Dict[str, Tuple[str, List]]:
        out = {n: (str(z.get("type", "presence")), z["polygon"]) for n, z in self.scene.zones.items()}
        out.update({n: ("exit_door", z["polygon"]) for n, z in self.scene.exit_regions.items()})
        return out

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        self.mouse_xy = (x, y)
        if self.input_stage is not None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.mode == self.MODE_ZONE:
                self.current.append([int(x), int(y)])
            elif len(self.bev_src) < 4:
                self.bev_src.append([float(x), float(y)])
                if len(self.bev_src) == 4:
                    self._prompt("BEV_W", "Width W in meters (point 1 -> 2): ", str(self.bev_w_m))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.mode == self.MODE_ZONE and self.current:
                self.current.pop()
            elif self.mode == self.MODE_BEV and self.bev_src:
                self.bev_src.pop()

    def _prompt(self, stage: str, prompt: str, default: str) -> None:
        self.input_stage, self.input_prompt, self.input_buffer = stage, prompt, default

    def save(self) -> None:
        save_scene(self.scene, self.scene_path)
        msg = f"Saved {len(self.scene.zones)} zones, {len(self.scene.exit_regions)} doors to {self.scene_path}"
        if len(self.bev_src) == 4:
            world = [[0.0, 0.0], [self.bev_w_m, 0.0], [self.bev_w_m, self.bev_l_m], [0.0, self.bev_l_m]]
            save_calibration(GroundCalibration(self.bev_src, world, (self.w, self.h)), self.calibration_path)
            msg += f" + calibration to {self.calibration_path}"
        self.status = msg
        print("[SAVED]", msg)

    def preview_bev(self) -> None:
        if len(self.bev_src) != 4:
            self.status = "Need 4 calibration points first"
            return
        ppm, margin = 60.0, 2.0
        out_w = int(np.clip((self.bev_w_m + 2 * margin) * ppm, 400, 1400))
        out_h = int(np.clip((self.bev_l_m + 2 * margin) * ppm, 400, 1000))
        dst = (np.array([[0, 0], [self.bev_w_m, 0], [self.bev_w_m, self.bev_l_m], [0, self.bev_l_m]],
                        np.float32) + margin) * ppm
        warped = cv2.warpPerspective(self.base_frame, cv2.getPerspectiveTransform(
            np.array(self.bev_src, np.float32), dst), (out_w, out_h))
        for xm in range(int(out_w / ppm) + 1):
            cv2.line(warped, (int(xm * ppm), 0), (int(xm * ppm), out_h), (80, 80, 80), 1)
        for ym in range(int(out_h / ppm) + 1):
            cv2.line(warped, (0, int(ym * ppm)), (out_w, int(ym * ppm)), (80, 80, 80), 1)
        cv2.polylines(warped, [dst.astype(np.int32)], True, (255, 255, 0), 2)
        cv2.imshow("Calibration check (1 cell = 1 m)", warped)

    def render(self) -> np.ndarray:
        canvas = self.base_frame.copy()
        overlay = canvas.copy()
        for name, (ztype, pts) in self.shapes().items():
            poly = np.array(pts, dtype=np.int32)
            cv2.fillPoly(overlay, [poly], TYPE_COLORS.get(ztype, (0, 200, 60)))
            cv2.polylines(canvas, [poly], True, TYPE_COLORS.get(ztype, (0, 200, 60)), 2, cv2.LINE_AA)
            c = poly.mean(axis=0).astype(int)
            cv2.putText(canvas, f"{name} [{ztype}]", (c[0] - 50, c[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2, cv2.LINE_AA)
        if self.bev_src:
            bev = np.array(self.bev_src, dtype=np.int32)
            cv2.polylines(canvas, [bev], len(bev) == 4, (255, 220, 0), 2, cv2.LINE_AA)
            for i, p in enumerate(bev):
                cv2.circle(canvas, tuple(p), 6, (0, 255, 255), -1)
                cv2.putText(canvas, str(i + 1), (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0, canvas)
        if self.mode == self.MODE_ZONE and self.current:
            cur = np.array(self.current, dtype=np.int32)
            color = TYPE_COLORS[self.zone_type]
            cv2.polylines(canvas, [cur], False, color, 2, cv2.LINE_AA)
            cv2.line(canvas, tuple(cur[-1]), self.mouse_xy, color, 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (0, 0), (self.w, 82), (20, 20, 20), -1)
        head = f"MODE: {self.mode}" + (f" | next zone type: {self.zone_type} (t)" if self.mode == self.MODE_ZONE else
                                       " | click 4 floor corners clockwise (TL, TR, BR, BL)")
        cv2.putText(canvas, head, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 120), 2, cv2.LINE_AA)
        cv2.putText(canvas, "TAB mode | t type | ENTER close | d delete last | r reset calib | v preview | s save | q quit",
                    (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1, cv2.LINE_AA)
        cv2.putText(canvas, self.status, (15, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)
        if self.input_stage is not None:
            y1, y2 = self.h // 2 - 45, self.h // 2 + 45
            cv2.rectangle(canvas, (self.w // 4, y1), (3 * self.w // 4, y2), (15, 15, 15), -1)
            cv2.putText(canvas, f"{self.input_prompt}{self.input_buffer}_", (self.w // 4 + 20, self.h // 2 + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        return canvas

    def handle_input_key(self, key: int) -> None:
        if key in (13, 10):
            val = self.input_buffer.strip()
            if self.input_stage == "NAME":
                name = val or f"{self.zone_type}_{len(self.shapes()) + 1}"
                if self.zone_type == "exit_door":
                    self.scene.exit_regions[name] = {"polygon": [list(p) for p in self.current]}
                else:
                    self.scene.zones[name] = {"polygon": [list(p) for p in self.current], "type": self.zone_type,
                                              "max_dwell_sec": 0, "max_occupancy": 0}
                self.current = []
                self.status = f"Added {self.zone_type} '{name}' (limits: edit max_dwell_sec/max_occupancy in scene)"
                self.input_stage = None
            elif self.input_stage == "BEV_W":
                try:
                    self.bev_w_m = max(0.5, float(val))
                except ValueError:
                    pass
                self._prompt("BEV_L", "Length L in meters (point 2 -> 3): ", str(self.bev_l_m))
            elif self.input_stage == "BEV_L":
                try:
                    self.bev_l_m = max(0.5, float(val))
                except ValueError:
                    pass
                self.input_stage = None
                self.status = f"Calibration {self.bev_w_m} x {self.bev_l_m} m. v = preview, s = save"
        elif key in (8, 127):
            self.input_buffer = self.input_buffer[:-1]
        elif key == 27:
            self.input_stage = None
        elif 32 <= key <= 126 and (chr(key).isalnum() or chr(key) in "_-."):
            self.input_buffer += chr(key)

    def run(self) -> None:
        win = "Zones & doors editor"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, min(1440, self.w), min(900, self.h))
        cv2.setMouseCallback(win, self.on_mouse)
        while True:
            cv2.imshow(win, self.render())
            key = cv2.waitKey(15) & 0xFF
            if key == 255:
                continue
            if self.input_stage is not None:
                self.handle_input_key(key)
                continue
            if key in (ord("q"), 27):
                break
            if key in (9, ord("m")):
                self.mode = self.MODE_BEV if self.mode == self.MODE_ZONE else self.MODE_ZONE
            elif key == ord("t"):
                self.zone_type = ZONE_TYPES[(ZONE_TYPES.index(self.zone_type) + 1) % len(ZONE_TYPES)]
            elif key in (13, 10) and self.mode == self.MODE_ZONE:
                if len(self.current) >= 3:
                    self._prompt("NAME", "Name: ", f"{self.zone_type}_{len(self.shapes()) + 1}")
                else:
                    self.status = "A polygon needs at least 3 points"
            elif key == ord("d"):
                names = list(self.shapes())
                if names:
                    last = names[-1]
                    self.scene.zones.pop(last, None)
                    self.scene.exit_regions.pop(last, None)
                    self.status = f"Deleted '{last}'"
            elif key == ord("r"):
                self.bev_src.clear()
            elif key == ord("v"):
                self.preview_bev()
            elif key == ord("s"):
                self.save()
        cv2.destroyAllWindows()


def grab_frame(source) -> np.ndarray:
    if isinstance(source, str) and Path(source).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        frame = cv2.imread(source)
        if frame is None:
            raise FileNotFoundError(source)
        return frame
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video stream: {source}")
    print("Press SPACE to freeze the frame to draw on, q to abort.")
    frame = None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frame = f
        cv2.imshow("Select frame (SPACE)", f)
        k = cv2.waitKey(25) & 0xFF
        if k == ord(" "):
            break
        if k == ord("q"):
            cap.release()
            sys.exit(0)
    cap.release()
    cv2.destroyAllWindows()
    if frame is None:
        raise RuntimeError("No frame read from the source")
    return frame


def main() -> None:
    ap = argparse.ArgumentParser(description="Zones / exit doors / ground calibration editor")
    ap.add_argument("--settings", default="configs/settings.yaml")
    ap.add_argument("--scene", default=None, help="scene file (default: scene.path of settings, configs/scene.yaml)")
    ap.add_argument("--calibration", default=DEFAULT_CALIBRATION_PATH)
    ap.add_argument("--source", default=None)
    args = ap.parse_args()
    settings = {}
    if Path(args.settings).exists():
        with open(args.settings, "r", encoding="utf-8") as f:
            settings = yaml.safe_load(f) or {}
    scene_path = args.scene or (settings.get("scene") or {}).get("path") or DEFAULT_SCENE_PATH
    source = args.source or (settings.get("stream") or {}).get("source", "0")
    SceneZoneEditor(grab_frame(source), scene_path, args.calibration).run()


if __name__ == "__main__":
    main()
