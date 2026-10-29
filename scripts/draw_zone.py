# scripts/draw_zone.py
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import yaml


class InteractiveCalibrator:
    """
    Unified GUI tool for:
      1. Drawing multi-vertex industrial zone polygons.
      2. Calibrating 4-point Bird's-Eye View (BEV) floor homography in real-world meters.
      3. Previewing top-down metric rectification with 1m x 1m grid lines.
      4. Auto-saving calibration directly into configs/settings.yaml.
    """

    MODE_ZONE = "ZONE_POLYGON"
    MODE_BEV = "BEV_HOMOGRAPHY"

    def __init__(self, frame: np.ndarray, config_path: Path, existing_cfg: Dict):
        self.base_frame = frame.copy()
        self.h, self.w = frame.shape[:2]
        self.config_path = config_path
        self.cfg = existing_cfg

        self.mode = self.MODE_ZONE
        self.mouse_xy: Tuple[int, int] = (0, 0)

        # Load existing zones from settings.yaml
        self.zones: Dict[str, List[List[int]]] = {}
        existing_zones = self.cfg.get("zones", {}).get("definitions", {})
        for z_name, z_data in existing_zones.items():
            if "polygon" in z_data and len(z_data["polygon"]) >= 3:
                self.zones[z_name] = [[int(p[0]), int(p[1])] for p in z_data["polygon"]]

        self.current_zone_pts: List[List[int]] = []

        # Load existing BEV calibration points from settings.yaml
        bev_cfg = self.cfg.get("physics_bev", {})
        self.bev_src_pts: List[List[float]] = (
            [[float(p[0]), float(p[1])] for p in bev_cfg.get("src_image_pts", [])]
            if len(bev_cfg.get("src_image_pts", [])) == 4
            else []
        )
        self.bev_width_m: float = 6.0
        self.bev_length_m: float = 8.0
        if len(bev_cfg.get("dst_meter_pts", [])) == 4:
            dst = bev_cfg["dst_meter_pts"]
            self.bev_width_m = float(max(p[0] for p in dst)) or 6.0
            self.bev_length_m = float(max(p[1] for p in dst)) or 8.0

        # In-window text input state (avoids freezing GUI with terminal input())
        self.input_prompt: Optional[str] = None
        self.input_buffer: str = ""
        self.input_stage: Optional[str] = None  # "ZONE_NAME", "BEV_W", "BEV_L"
        self.status_msg: str = "Loaded config. [TAB]: Switch Mode | [L-Click]: Add Pt | [R-Click]: Undo"

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        self.mouse_xy = (x, y)
        if self.input_stage is not None:
            return  # Ignore mouse clicks while typing in HUD prompt

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.mode == self.MODE_ZONE:
                self.current_zone_pts.append([int(x), int(y)])
                self.status_msg = f"Added zone vertex ({x}, {y}). Total: {len(self.current_zone_pts)}"
            elif self.mode == self.MODE_BEV:
                if len(self.bev_src_pts) < 4:
                    self.bev_src_pts.append([float(x), float(y)])
                    labels = ["1:Top-Left (0,0)", "2:Top-Right (W,0)", "3:Bottom-Right (W,L)", "4:Bottom-Left (0,L)"]
                    idx = len(self.bev_src_pts) - 1
                    self.status_msg = f"Set BEV Pt {labels[idx]} at ({x}, {y})"
                    if len(self.bev_src_pts) == 4:
                        self.input_stage = "BEV_W"
                        self.input_buffer = str(self.bev_width_m)
                        self.input_prompt = "Enter Floor Width W in meters (Top-Left to Top-Right): "
                else:
                    self.status_msg = "4 BEV points already set. Press [R] to reset BEV points."

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.mode == self.MODE_ZONE and self.current_zone_pts:
                removed = self.current_zone_pts.pop()
                self.status_msg = f"Removed zone vertex {removed}"
            elif self.mode == self.MODE_BEV and self.bev_src_pts:
                removed = self.bev_src_pts.pop()
                self.status_msg = f"Removed BEV point {removed}"

    def compute_dst_meter_pts(self) -> List[List[float]]:
        return [
            [0.0, 0.0],
            [round(self.bev_width_m, 3), 0.0],
            [round(self.bev_width_m, 3), round(self.bev_length_m, 3)],
            [0.0, round(self.bev_length_m, 3)],
        ]

    def show_bev_preview(self) -> None:
        """Warps the camera frame into a top-down metric view with 1m x 1m grid lines."""
        if len(self.bev_src_pts) != 4:
            self.status_msg = "Need exactly 4 BEV Floor points before opening preview!"
            return

        px_per_m = 60.0  # 60 pixels per meter in preview window
        margin_m = 2.0   # 2-meter padding around calibrated rectangle
        out_w = int((self.bev_width_m + 2 * margin_m) * px_per_m)
        out_h = int((self.bev_length_m + 2 * margin_m) * px_per_m)
        out_w = int(np.clip(out_w, 400, 1400))
        out_h = int(np.clip(out_h, 400, 1000))

        src = np.array(self.bev_src_pts, dtype=np.float32)
        dst_m = np.array(self.compute_dst_meter_pts(), dtype=np.float32)
        dst_px = (dst_m + margin_m) * px_per_m

        H_preview = cv2.getPerspectiveTransform(src, dst_px)
        warped = cv2.warpPerspective(self.base_frame, H_preview, (out_w, out_h))

        # Draw 1-meter grid lines
        total_w_m = int(np.ceil(self.bev_width_m + 2 * margin_m))
        total_h_m = int(np.ceil(self.bev_length_m + 2 * margin_m))
        for xm in range(total_w_m + 1):
            x_px = int(xm * px_per_m)
            cv2.line(warped, (x_px, 0), (x_px, out_h), (80, 80, 80), 1)
        for ym in range(total_h_m + 1):
            y_px = int(ym * px_per_m)
            cv2.line(warped, (0, y_px), (out_w, y_px), (80, 80, 80), 1)

        # Highlight calibrated reference rectangle in cyan
        rect_pts = dst_px.astype(np.int32)
        cv2.polylines(warped, [rect_pts], True, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(
            warped,
            f"BEV Metric Grid (1 cell = 1m x 1m) | Ref: {self.bev_width_m}m x {self.bev_length_m}m",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 180),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("BEV Metric Verification (1m Grid)", warped)
        self.status_msg = "Opened BEV Metric Preview window."

    def save_to_yaml(self) -> None:
        self.cfg.setdefault("zones", {})
        self.cfg["zones"]["definitions"] = {
            z_name: {"polygon": pts} for z_name, pts in self.zones.items()
        }

        if len(self.bev_src_pts) == 4:
            self.cfg.setdefault("physics_bev", {})
            self.cfg["physics_bev"]["src_image_pts"] = [
                [round(p[0], 1), round(p[1], 1)] for p in self.bev_src_pts
            ]
            self.cfg["physics_bev"]["dst_meter_pts"] = self.compute_dst_meter_pts()

        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.cfg, f, sort_keys=False, default_flow_style=None)

        self.status_msg = f"Saved {len(self.zones)} zones & BEV calibration to {self.config_path}"
        print(f"[SUCCESS] {self.status_msg}")

    def render(self) -> np.ndarray:
        canvas = self.base_frame.copy()
        overlay = canvas.copy()

        # 1. Draw saved Zone Polygons
        for z_name, pts in self.zones.items():
            poly = np.array(pts, dtype=np.int32)
            cv2.fillPoly(overlay, [poly], (0, 180, 60))
            cv2.polylines(canvas, [poly], True, (0, 255, 80), 2, cv2.LINE_AA)
            centroid = np.mean(poly, axis=0).astype(int)
            cv2.putText(
                canvas,
                f"ZONE: {z_name}",
                (centroid[0] - 45, centroid[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # 2. Draw BEV Homography Floor Rectangle
        if self.bev_src_pts:
            bev_np = np.array(self.bev_src_pts, dtype=np.int32)
            if len(self.bev_src_pts) == 4:
                cv2.fillPoly(overlay, [bev_np], (200, 120, 0))
                cv2.polylines(canvas, [bev_np], True, (255, 220, 0), 2, cv2.LINE_AA)
            else:
                cv2.polylines(canvas, [bev_np], False, (255, 220, 0), 2, cv2.LINE_AA)

            corner_tags = ["TL(0,0)", f"TR({self.bev_width_m}m,0)", f"BR({self.bev_width_m}m,{self.bev_length_m}m)", f"BL(0,{self.bev_length_m}m)"]
            for i, pt in enumerate(bev_np):
                cv2.circle(canvas, tuple(pt), 6, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.putText(
                    canvas,
                    corner_tags[i],
                    (pt[0] + 8, pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0, canvas)

        # 3. Draw active in-progress Zone Polygon + elastic mouse line
        if self.mode == self.MODE_ZONE and self.current_zone_pts:
            curr_np = np.array(self.current_zone_pts, dtype=np.int32)
            cv2.polylines(canvas, [curr_np], False, (0, 165, 255), 2, cv2.LINE_AA)
            for pt in curr_np:
                cv2.circle(canvas, tuple(pt), 5, (0, 165, 255), -1, cv2.LINE_AA)
            cv2.line(canvas, tuple(curr_np[-1]), self.mouse_xy, (0, 200, 255), 1, cv2.LINE_AA)

        # 4. Draw Top Control Banner
        cv2.rectangle(canvas, (0, 0), (self.w, 82), (20, 20, 20), -1)
        mode_color = (0, 255, 120) if self.mode == self.MODE_ZONE else (255, 220, 0)
        cv2.putText(
            canvas,
            f"ACTIVE MODE: {self.mode}  (Press [TAB] to toggle)",
            (15, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            mode_color,
            2,
            cv2.LINE_AA,
        )
        controls_help = (
            "[L-Click]: Add Pt | [R-Click]: Undo | [ENTER]: Confirm Zone | "
            "[D]: Delete Last Zone | [R]: Reset BEV | [V]: BEV Grid Preview | [S]: Save YAML | [Q]: Quit"
        )
        cv2.putText(canvas, controls_help, (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (210, 210, 210), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"STATUS: {self.status_msg}", (15, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 255), 1, cv2.LINE_AA)

        # 5. Draw Interactive Text Entry Modal if active
        if self.input_stage is not None:
            box_y1, box_y2 = self.h // 2 - 45, self.h // 2 + 45
            cv2.rectangle(canvas, (self.w // 4, box_y1), (3 * self.w // 4, box_y2), (15, 15, 15), -1)
            cv2.rectangle(canvas, (self.w // 4, box_y1), (3 * self.w // 4, box_y2), (0, 255, 255), 2)
            text_line = f"{self.input_prompt}{self.input_buffer}_"
            cv2.putText(
                canvas,
                text_line,
                (self.w // 4 + 20, self.h // 2 + 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return canvas

    def handle_text_input_key(self, key: int) -> None:
        if key in (13, 10):  # ENTER
            val = self.input_buffer.strip()
            if self.input_stage == "ZONE_NAME":
                z_name = val if val else f"Zone_{len(self.zones) + 1}"
                self.zones[z_name] = self.current_zone_pts.copy()
                self.current_zone_pts.clear()
                self.status_msg = f"Locked polygon for '{z_name}'."
                self.input_stage = None
            elif self.input_stage == "BEV_W":
                try:
                    self.bev_width_m = max(0.5, float(val))
                except ValueError:
                    pass
                self.input_stage = "BEV_L"
                self.input_buffer = str(self.bev_length_m)
                self.input_prompt = "Enter Floor Length L in meters (Top-Right to Bottom-Right): "
            elif self.input_stage == "BEV_L":
                try:
                    self.bev_length_m = max(0.5, float(val))
                except ValueError:
                    pass
                self.input_stage = None
                self.status_msg = f"BEV Homography calibrated ({self.bev_width_m}m x {self.bev_length_m}m). Press [V] to preview!"
        elif key in (8, 127):  # BACKSPACE
            self.input_buffer = self.input_buffer[:-1]
        elif key == 27:  # ESC cancels prompt
            self.input_stage = None
            self.status_msg = "Cancelled input prompt."
        elif 32 <= key <= 126:
             ch = chr(key)
             if ch.isalnum() or ch in ("_", "-", "."):
                 self.input_buffer += ch

    def run(self) -> None:
        win_name = "Enterprise Zone & BEV Homography Calibrator"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, self.w, self.h)
        cv2.setMouseCallback(win_name, self.on_mouse)

        while True:
            canvas = self.render()
            cv2.imshow(win_name, canvas)
            key = cv2.waitKey(15) & 0xFF
            if key == 255:
                continue

            if self.input_stage is not None:
                self.handle_text_input_key(key)
                continue

            if key in (ord("q"), 27):  # Q or ESC
                break
            elif key == 9 or key == ord("m"):  # TAB or M switches mode
                self.mode = self.MODE_BEV if self.mode == self.MODE_ZONE else self.MODE_ZONE
                self.status_msg = f"Switched to {self.mode} mode."
            elif key in (13, 10):  # ENTER finishes current zone polygon
                if self.mode == self.MODE_ZONE:
                    if len(self.current_zone_pts) >= 3:
                        self.input_stage = "ZONE_NAME"
                        self.input_buffer = f"Zone_{len(self.zones) + 1}"
                        self.input_prompt = "Enter Zone Name: "
                    else:
                        self.status_msg = "Need at least 3 vertices to close a zone polygon!"
            elif key == ord("d"):  # Delete last saved zone
                if self.zones:
                    last_key = list(self.zones.keys())[-1]
                    del self.zones[last_key]
                    self.status_msg = f"Deleted zone '{last_key}'."
            elif key == ord("r"):  # Reset BEV points
                self.bev_src_pts.clear()
                self.status_msg = "Cleared 4 BEV calibration points."
            elif key == ord("v"):  # Preview BEV warp
                self.show_bev_preview()
            elif key == ord("s"):  # Save to settings.yaml
                self.save_to_yaml()

        cv2.destroyAllWindows()


def capture_reference_frame(source: str | int) -> np.ndarray:
    cap_src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(cap_src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video stream: {source}")

    print("Stream opened. Press [SPACE] to freeze reference frame for calibration, or [Q] to abort.")
    selected_frame = None
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        preview = frame.copy()
        cv2.rectangle(preview, (10, 10), (660, 48), (20, 20, 20), -1)
        cv2.putText(
            preview,
            "LIVE STREAM: Press [SPACE] to capture frame for calibration",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 180),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Select Calibration Frame", preview)
        key = cv2.waitKey(25) & 0xFF
        if key == ord(" "):
            selected_frame = frame.copy()
            break
        elif key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)
        selected_frame = frame.copy()

    cap.release()
    cv2.destroyWindow("Select Calibration Frame")
    if selected_frame is None:
        raise RuntimeError("Failed to read any frames from video source.")
    return selected_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive Factory Zone & BEV Homography Calibrator")
    parser.add_argument("--config", type=str, default="configs/settings.yaml", help="Path to settings.yaml")
    parser.add_argument("--source", type=str, default=None, help="Override video/RTSP source or image path")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    existing_cfg: Dict = {}
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            existing_cfg = yaml.safe_load(f) or {}

    source = args.source or existing_cfg.get("stream", {}).get("source", "0")

    # Support calibrating directly from a static snapshot image (.jpg/.png) or live video stream
    if isinstance(source, str) and Path(source).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        frame = cv2.imread(source)
        if frame is None:
            raise FileNotFoundError(f"Could not read image file: {source}")
    else:
        frame = capture_reference_frame(source)

    calibrator = InteractiveCalibrator(frame=frame, config_path=cfg_path, existing_cfg=existing_cfg)
    calibrator.run()


if __name__ == "__main__":
    main()