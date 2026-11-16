import argparse
from typing import List, Tuple
import cv2
import numpy as np
import yaml

poly_pts: List[Tuple[int, int]] = []
dir_pts: List[Tuple[int, int]] = []
draw_mode: str = "zone"      # "zone" or "line"
step_mode: str = "shape"     # "shape" (draw points) or "direction" (draw entry arrow)
invert_line: bool = False


def mouse_cb(event, x, y, flags, param):
    global poly_pts, dir_pts, step_mode
    if event == cv2.EVENT_LBUTTONDOWN:
        if draw_mode == "line":
            if len(poly_pts) >= 2:
                poly_pts.clear()
            poly_pts.append((x, y))
        elif draw_mode == "zone":
            if step_mode == "shape":
                poly_pts.append((x, y))
            elif step_mode == "direction":
                if len(dir_pts) >= 2:
                    dir_pts.clear()
                dir_pts.append((x, y))

    elif event == cv2.EVENT_RBUTTONDOWN:
        if draw_mode == "zone" and step_mode == "direction" and dir_pts:
            dir_pts.pop()
        elif poly_pts:
            poly_pts.pop()


def compute_unit_vector(pt_start: Tuple[int, int], pt_end: Tuple[int, int]) -> List[float]:
    dx = float(pt_end[0] - pt_start[0])
    dy = float(pt_end[1] - pt_start[1])
    norm = float(np.hypot(dx, dy))
    if norm < 1e-6:
        return [0.0, -1.0]
    return [round(dx / norm, 4), round(dy / norm, 4)]


def main():
    global draw_mode, step_mode, invert_line
    parser = argparse.ArgumentParser(
        description="Interactive Zone, Tripwire Line & Orientation Configurator"
    )
    parser.add_argument("--config", default="configs/settings.yaml", help="Path to settings.yaml")
    parser.add_argument(
        "--mode",
        choices=["zone", "line"],
        default="zone",
        help="'zone' for entrance polygon + orientation arrow, 'line' for Point A -> B tripwire",
    )
    parser.add_argument(
        "--name",
        default="company_entrance_gate",
        help="Name of the zone or line in settings.yaml",
    )
    args = parser.parse_args()
    draw_mode = args.mode

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    src = cfg["stream"]["source"]
    cap = cv2.VideoCapture(int(src) if str(src).isdigit() else src)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError(f"Could not read frame from stream.source: {src}")

    h, w = frame.shape[:2]
    win = "Directional Configurator"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(1440, w), min(900, h))
    cv2.setMouseCallback(win, mouse_cb)

    print("\n=== Interactive Directional Configurator ===")
    if draw_mode == "zone":
        print("1. [SHAPE MODE] Left-Click 3+ corners to draw the Entrance Zone polygon.")
        print("2. Press 'd' to switch to [DIRECTION MODE], then Left-Click 2 points")
        print("   (Start -> End) to draw the ENTRY direction arrow.")
        print("3. Press 's' to save to configs/settings.yaml, or 'q' to quit.\n")
    else:
        print("1. Left-Click Point A and Point B to draw the tripwire line.")
        print("2. Press 'i' to flip/invert the green ENTRY direction arrow.")
        print("3. Press 's' to save to configs/settings.yaml, or 'q' to quit.\n")

    while True:
        vis = frame.copy()

        # Draw instructions banner
        cv2.rectangle(vis, (10, 10), (1150, 85), (20, 20, 20), -1)
        if draw_mode == "zone":
             banner_1 = (
                 f"MODE: ZONE ({args.name}) | Current Step: {step_mode.upper()} "
                 f"| Press 'd' to toggle Polygon/Arrow | Press 'c' to clear"
             )
             banner_2 = (
                 "Left-Click: Add Point | Right-Click: Undo | "
                 "Arrow = 2 clicks (Outside -> Inside) | Press 's' to Save"
             )
        else:
            banner_1 = (
                f"MODE: LINE ({args.name}) | Invert: {invert_line} "
                f"| Press 'i' to Flip ENTRY Arrow | Press 'c' to clear"
            )
            banner_2 = "Left-Click: Point A then Point B | Right-Click: Undo | Press 's' to Save"

        cv2.putText(vis, banner_1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, banner_2, (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 255, 200), 2, cv2.LINE_AA)

        # Render Line Mode
        if draw_mode == "line":
            for idx, pt in enumerate(poly_pts):
                lbl = "A" if idx == 0 else "B"
                col = (0, 255, 0) if idx == 0 else (0, 0, 255)
                cv2.circle(vis, pt, 8, col, -1, cv2.LINE_AA)
                cv2.putText(vis, lbl, (pt[0] + 10, pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)

            if len(poly_pts) == 2:
                pt_a, pt_b = poly_pts[0], poly_pts[1]
                cv2.line(vis, pt_a, pt_b, (0, 255, 255), 3, cv2.LINE_AA)

                mid_x = int((pt_a[0] + pt_b[0]) / 2)
                mid_y = int((pt_a[1] + pt_b[1]) / 2)
                dx, dy = pt_b[0] - pt_a[0], pt_b[1] - pt_a[1]
                length = max(1.0, float(np.hypot(dx, dy)))
                sign = -1.0 if invert_line else 1.0
                nx = int(-sign * (dy / length) * 55)
                ny = int(sign * (dx / length) * 55)
                arrow_tip = (mid_x + nx, mid_y + ny)
                cv2.arrowedLine(vis, (mid_x, mid_y), arrow_tip, (0, 255, 0), 3, cv2.LINE_AA, tipLength=0.3)
                cv2.putText(vis, "ENTRY", arrow_tip, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # Render Zone Mode
        elif draw_mode == "zone":
            for idx, pt in enumerate(poly_pts):
                cv2.circle(vis, pt, 6, (0, 165, 255), -1, cv2.LINE_AA)
                cv2.putText(
                    vis,
                    str(idx + 1),
                    (pt[0] + 8, pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )

            if len(poly_pts) >= 2:
                poly_np = np.array(poly_pts, dtype=np.int32)
                cv2.polylines(
                    vis,
                    [poly_np],
                    isClosed=(len(poly_pts) >= 3),
                    color=(255, 165, 0),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )

            # Draw Orientation Arrow if defined
            if len(dir_pts) == 1:
                cv2.circle(vis, dir_pts[0], 7, (0, 255, 0), -1, cv2.LINE_AA)
                cv2.putText(vis, "START", (dir_pts[0][0] + 10, dir_pts[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            elif len(dir_pts) == 2:
                cv2.arrowedLine(vis, dir_pts[0], dir_pts[1], (0, 255, 0), 4, cv2.LINE_AA, tipLength=0.25)
                uvec = compute_unit_vector(dir_pts[0], dir_pts[1])
                cv2.putText(
                    vis,
                    f"ENTRY {uvec}",
                    (dir_pts[1][0] + 10, dir_pts[1][1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        cv2.imshow(win, vis)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("d") and draw_mode == "zone":
            step_mode = "direction" if step_mode == "shape" else "shape"
            print(f"Switched to [{step_mode.upper()}] mode.")
        elif key == ord("i") and draw_mode == "line":
            invert_line = not invert_line
            print(f"Line invert_direction set to: {invert_line}")
        elif key == ord("c"):
            poly_pts.clear()
            dir_pts.clear()
            step_mode = "shape"
            print("Cleared all points.")
        elif key == ord("s"):
            dir_cfg = cfg.setdefault("directional_tracking", {})
            if draw_mode == "line":
                if len(poly_pts) != 2:
                    print("[ERROR] Please click 2 points (A and B) before saving.")
                    continue
                dir_cfg.setdefault("lines", {})[args.name] = {
                    "point_a": [int(poly_pts[0][0]), int(poly_pts[0][1])],
                    "point_b": [int(poly_pts[1][0]), int(poly_pts[1][1])],
                    "invert_direction": bool(invert_line),
                    "cooldown_seconds": 2.0,
                    "color": [0, 255, 255],
                }
                with open(args.config, "w", encoding="utf-8") as f:
                    yaml.safe_dump(cfg, f, sort_keys=False)
                print(f"Saved tripwire line '{args.name}' to {args.config}")
                break

            elif draw_mode == "zone":
                if len(poly_pts) < 3:
                    print("[ERROR] Please click at least 3 polygon corners before saving.")
                    continue
                entry_vec = (
                    compute_unit_vector(dir_pts[0], dir_pts[1])
                    if len(dir_pts) == 2
                    else [0.0, -1.0]
                )
                dir_cfg.setdefault("entrance_zones", {})[args.name] = {
                    "polygon": [[int(p[0]), int(p[1])] for p in poly_pts],
                    "entry_direction_vector": entry_vec,
                    "min_displacement": 20.0,
                    "color": [255, 165, 0],
                }
                with open(args.config, "w", encoding="utf-8") as f:
                    yaml.safe_dump(cfg, f, sort_keys=False)
                print(
                    f"Saved entrance zone '{args.name}' with entry_direction_vector={entry_vec} to {args.config}"
                )
                break
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()