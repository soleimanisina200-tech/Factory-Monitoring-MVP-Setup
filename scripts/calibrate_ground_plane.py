import cv2
import yaml
import argparse
import numpy as np
from pathlib import Path

clicked_points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        clicked_points.append([int(x), int(y)])
        print(f"[Calibration] Point {len(clicked_points)} recorded at pixel: ({x}, {y})")

def main():
    parser = argparse.ArgumentParser(description="Calibrate Ground Plane for Real-World Distance (Meters)")
    parser.add_argument("--source", type=str, required=True, help="Video file or RTSP URL")
    parser.add_argument("--config", type=str, default="configs/vehicle_monitoring.yaml", help="Config path")
    parser.add_argument("--width-m", type=float, required=True, help="Real-world width (meters) between Pt1-Pt2")
    parser.add_argument("--length-m", type=float, required=True, help="Real-world length (meters) between Pt2-Pt3")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.source)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Failed to read frame from source: {args.source}")

    window_name = "Ground Plane Calibration - Click 4 Floor Corners (Clockwise: Top-Left, Top-Right, Bottom-Right, Bottom-Left)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        display = frame.copy()
        for idx, pt in enumerate(clicked_points):
            cv2.circle(display, tuple(pt), 6, (0, 255, 0), -1)
            cv2.putText(display, f"P{idx+1}", (pt[0] + 8, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if len(clicked_points) > 1:
            cv2.polylines(display, [np.array(clicked_points)], len(clicked_points) == 4, (0, 255, 255), 2)

        cv2.putText(display, "Click 4 floor points clockwise. Press 's' to save, 'r' to reset, 'q' to quit.",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(window_name, display)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('r'):
            clicked_points.clear()
        elif key == ord('s') and len(clicked_points) == 4:
            break
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()

    world_points = [
        [0.0, 0.0],
        [float(args.width_m), 0.0],
        [float(args.width_m), float(args.length_m)],
        [0.0, float(args.length_m)]
    ]

    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["vehicle_monitoring"]["calibration"]["image_points_px"] = clicked_points
    cfg["vehicle_monitoring"]["calibration"]["world_points_m"] = world_points

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"[Calibration] Successfully saved 4-point metric homography to {config_path}")

if __name__ == "__main__":
    main()