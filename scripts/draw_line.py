# scripts/draw_line.py
"""
Draw a directional LINE (tripwire) or a directional REGION (gate polygon + entry arrow) into configs/scene.yaml.

    python scripts/draw_line.py --mode line --name main_gate
    python scripts/draw_line.py --mode region --name entrance --labels IN OUT --allowed IN

line   : click A then B. The green arrow shows the first label's direction (IN); 'i' flips it.
region : click 3+ corners, press 'd', then click 2 points (outside -> inside) for the entry arrow.
Keys: c clear | right-click undo | s save | q quit.  (--mode zone is accepted as an alias of region.)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core.scene import DEFAULT_SCENE_PATH, Scene, load_scene, save_scene  # noqa: E402

pts: List[Tuple[int, int]] = []
arrow: List[Tuple[int, int]] = []
state = {"mode": "line", "step": "shape", "invert": False}


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if state["mode"] == "line":
            if len(pts) >= 2:
                pts.clear()
            pts.append((x, y))
        elif state["step"] == "shape":
            pts.append((x, y))
        else:
            if len(arrow) >= 2:
                arrow.clear()
            arrow.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        if state["mode"] == "region" and state["step"] == "direction" and arrow:
            arrow.pop()
        elif pts:
            pts.pop()


def unit(a, b) -> List[float]:
    d = np.array([b[0] - a[0], b[1] - a[1]], dtype=float)
    n = float(np.linalg.norm(d))
    return [0.0, -1.0] if n < 1e-6 else [round(d[0] / n, 4), round(d[1] / n, 4)]


def main() -> None:
    ap = argparse.ArgumentParser(description="Directional line / region editor (configs/scene.yaml)")
    ap.add_argument("--mode", choices=["line", "region", "zone"], default="line")
    ap.add_argument("--name", default="main_gate")
    ap.add_argument("--labels", nargs=2, default=["IN", "OUT"], help="names of the two directions")
    ap.add_argument("--allowed", default=None, help="only this direction is allowed (other -> WRONG_WAY event)")
    ap.add_argument("--settings", default="configs/settings.yaml")
    ap.add_argument("--scene", default=None)
    ap.add_argument("--source", default=None)
    args = ap.parse_args()
    state["mode"] = "region" if args.mode in ("region", "zone") else "line"

    settings = {}
    if Path(args.settings).exists():
        with open(args.settings, "r", encoding="utf-8") as f:
            settings = yaml.safe_load(f) or {}
    scene_path = args.scene or (settings.get("scene") or {}).get("path") or DEFAULT_SCENE_PATH
    src = args.source or (settings.get("stream") or {}).get("source", "0")
    cap = cv2.VideoCapture(int(src) if str(src).isdigit() else src)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read a frame from {src}")
    h, w = frame.shape[:2]
    scene = (load_scene(scene_path) if Path(scene_path).exists() else Scene()).scaled((w, h))
    scene.image_wh = (w, h)

    win = "Directional line / region editor"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(1440, w), min(900, h))
    cv2.setMouseCallback(win, on_mouse)
    while True:
        vis = frame.copy()
        for spec in scene.lines.values():
            cv2.line(vis, tuple(map(int, spec["a"])), tuple(map(int, spec["b"])), (120, 120, 120), 2)
        for spec in scene.regions.values():
            cv2.polylines(vis, [np.array(spec["polygon"], np.int32)], True, (120, 120, 120), 2)
        cv2.rectangle(vis, (10, 10), (1180, 80), (20, 20, 20), -1)
        if state["mode"] == "line":
            l1 = f"LINE '{args.name}' | click A then B | i: flip arrow (now {'inverted' if state['invert'] else 'normal'})"
        else:
            l1 = f"REGION '{args.name}' | step: {state['step']} | d: polygon <-> entry arrow (outside -> inside)"
        cv2.putText(vis, l1, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, f"labels {args.labels}  allowed={args.allowed} | c clear | s save | q quit", (20, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 255, 200), 2, cv2.LINE_AA)
        if state["mode"] == "line":
            for i, p in enumerate(pts):
                cv2.circle(vis, p, 8, (0, 255, 0) if i == 0 else (0, 0, 255), -1)
                cv2.putText(vis, "AB"[i], (p[0] + 10, p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if len(pts) == 2:
                a, b = pts
                cv2.line(vis, a, b, (0, 255, 255), 3, cv2.LINE_AA)
                mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
                dx, dy = b[0] - a[0], b[1] - a[1]
                n = max(1.0, float(np.hypot(dx, dy)))
                s = -1.0 if state["invert"] else 1.0
                tip = (int(mid[0] - s * dy / n * 55), int(mid[1] + s * dx / n * 55))
                cv2.arrowedLine(vis, mid, tip, (0, 255, 0), 3, cv2.LINE_AA, tipLength=0.3)
                cv2.putText(vis, args.labels[0], tip, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            if len(pts) >= 2:
                cv2.polylines(vis, [np.array(pts, np.int32)], len(pts) >= 3, (255, 165, 0), 2, cv2.LINE_AA)
            for p in pts:
                cv2.circle(vis, p, 6, (255, 165, 0), -1)
            if len(arrow) == 2:
                cv2.arrowedLine(vis, arrow[0], arrow[1], (0, 255, 0), 4, cv2.LINE_AA, tipLength=0.25)
                cv2.putText(vis, args.labels[0], arrow[1], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif len(arrow) == 1:
                cv2.circle(vis, arrow[0], 7, (0, 255, 0), -1)
        cv2.imshow(win, vis)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            pts.clear()
            arrow.clear()
            state["step"] = "shape"
        elif key == ord("i") and state["mode"] == "line":
            state["invert"] = not state["invert"]
        elif key == ord("d") and state["mode"] == "region":
            state["step"] = "direction" if state["step"] == "shape" else "shape"
        elif key == ord("s"):
            if state["mode"] == "line":
                if len(pts) != 2:
                    print("[ERROR] click A and B first")
                    continue
                scene.lines[args.name] = {"a": list(pts[0]), "b": list(pts[1]), "invert": state["invert"],
                                          "labels": list(args.labels), "dead_band_px": 18.0, "cooldown_sec": 1.0,
                                          "allowed_direction": args.allowed}
            else:
                if len(pts) < 3 or len(arrow) != 2:
                    print("[ERROR] draw >= 3 corners AND the entry arrow (press d) first")
                    continue
                scene.regions[args.name] = {"polygon": [list(p) for p in pts], "entry_vector": unit(*arrow),
                                            "min_displacement_px": 40.0, "labels": list(args.labels),
                                            "allowed_direction": args.allowed}
            save_scene(scene, scene_path)
            print(f"Saved {state['mode']} '{args.name}' to {scene_path}")
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
