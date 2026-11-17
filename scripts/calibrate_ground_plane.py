"""
Ground-plane calibration: click 4 floor points whose real-world rectangle you measured.

Writes configs/calibration.yaml, which BOTH main.py (worker + vehicle distances) and
run_vehicle_monitoring.py read. The click resolution is stored, so the points are rescaled
exactly to whatever resolution the live stream has.

Example:
    python scripts/calibrate_ground_plane.py --source data/input/cam1.mp4 --width-m 6.0 --length-m 8.0
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.logic.calibration import DEFAULT_CALIBRATION_PATH, GroundCalibration, save_calibration  # noqa: E402

clicked_points = []


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        clicked_points.append([int(x), int(y)])
        print(f"[Calibration] Point {len(clicked_points)} recorded at pixel: ({x}, {y})")


def main():
    parser = argparse.ArgumentParser(description="Calibrate Ground Plane for Real-World Distance (Meters)")
    parser.add_argument("--source", type=str, required=True, help="Video file or RTSP URL")
    parser.add_argument("--out", type=str, default=DEFAULT_CALIBRATION_PATH, help="Calibration file to write")
    parser.add_argument("--width-m", type=float, required=True, help="Real-world width (meters) between Pt1-Pt2")
    parser.add_argument("--length-m", type=float, required=True, help="Real-world length (meters) between Pt2-Pt3")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.source)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Failed to read frame from source: {args.source}")
    frame_h, frame_w = frame.shape[:2]

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
        [0.0, float(args.length_m)],
    ]
    # Mouse coordinates are in the frame's native pixel space (cv2 scales the window, not the callback).
    save_calibration(GroundCalibration(clicked_points, world_points, (frame_w, frame_h)), args.out)
    print(f"[Calibration] Saved 4-point metric homography ({frame_w}x{frame_h}) to {args.out}")
    print("[Calibration] Validate it: measure a known distance on the floor and compare with the HUD.")


if __name__ == "__main__":
    main()
