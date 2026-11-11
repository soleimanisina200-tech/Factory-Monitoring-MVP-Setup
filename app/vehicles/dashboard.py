# app/vehicles/dashboard.py
from typing import Dict, List, Optional
import cv2
import numpy as np

from app.vehicles.association import ForkliftSession, PersonObservation, VehicleSafetyState


class VehicleDashboardRenderer:
    """
    Renders real-time vehicle & driver monitoring visuals onto the OpenCV frame:
    - Calibrated ground-plane boundary
    - Forklift bounding boxes, status badges, and paired driver IDs
    - Real-world metric distance tether lines (in meters)
    - Uncertainty badges when occlusion or out-of-bounds occurs
    - Flashing visual alarm banners when safe distance is exceeded
    """

    STATE_COLORS = {
        VehicleSafetyState.UNASSIGNED: (180, 180, 180),        # Gray
        VehicleSafetyState.OPERATED: (0, 210, 0),              # Green
        VehicleSafetyState.DRIVER_EXITED_SAFE: (255, 200, 0),  # Cyan
        VehicleSafetyState.WARNING_PROXIMITY: (0, 165, 255),   # Amber/Orange
        VehicleSafetyState.ALARM_VIOLATION: (0, 0, 255),       # Red
        VehicleSafetyState.UNCERTAIN_DISTANCE: (255, 0, 255),  # Magenta
    }

    def __init__(self, config: Dict):
        dist_cfg = config.get("distance", {})
        cal_cfg = config.get("calibration", {})
        self.safe_threshold_m = float(dist_cfg.get("safe_threshold_m", 3.0))
        self.calib_pts_px: Optional[np.ndarray] = None
        if cal_cfg.get("calibrated") and cal_cfg.get("image_points_px"):
            self.calib_pts_px = np.array(cal_cfg["image_points_px"], dtype=np.int32)
        self._frame_counter = 0

    def draw(
        self,
        frame: np.ndarray,
        sessions: Dict[int, ForkliftSession],
        persons: List[PersonObservation],
        recent_events: List[Dict],
    ) -> np.ndarray:
        self._frame_counter += 1
        canvas = frame.copy()
        h, w = canvas.shape[:2]
        person_map = {p.global_id: p for p in persons}
        any_active_alarm = False

        # 1. Draw subtle calibrated ground-plane boundary
        if self.calib_pts_px is not None and len(self.calib_pts_px) >= 4:
            cv2.polylines(
                canvas, [self.calib_pts_px], isClosed=True, color=(90, 90, 90), thickness=1
            )

        # 2. Render each tracked forklift and its associated driver
        for fid, session in sessions.items():
            color = self.STATE_COLORS.get(session.state, (200, 200, 200))
            vx1, vy1, vx2, vy2 = map(int, session.bbox)

            cv2.rectangle(canvas, (vx1, vy1), (vx2, vy2), color, 3)
            drv_str = f"D#{session.driver_id}" if session.driver_id is not None else "Unassigned"
            label = f"Forklift #{fid} [{drv_str}] | {session.state.value}"
            cv2.putText(
                canvas,
                label,
                (vx1, max(24, vy1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                color,
                2,
            )

            # Draw tether and real-world metric distance when driver has exited
            if (
                session.driver_id is not None
                and session.state != VehicleSafetyState.OPERATED
                and session.last_measurement is not None
            ):
                meas = session.last_measurement
                p_drv = meas.driver_ground_px
                p_veh = meas.vehicle_ground_px

                if session.driver_id in person_map:
                    dx1, dy1, dx2, dy2 = map(int, person_map[session.driver_id].bbox)
                    cv2.rectangle(canvas, (dx1, dy1), (dx2, dy2), color, 2)

                cv2.circle(canvas, p_drv, 6, color, -1)
                cv2.circle(canvas, p_veh, 6, color, -1)
                cv2.line(canvas, p_drv, p_veh, color, 2, cv2.LINE_AA)

                mid_x = int((p_drv[0] + p_veh[0]) / 2)
                mid_y = int((p_drv[1] + p_veh[1]) / 2) - 10

                if not meas.is_certain or meas.distance_m is None:
                    tether_text = f"UNCERTAIN ({meas.uncertainty_reason})"
                else:
                    tether_text = f"{meas.distance_m:.2f}m / {self.safe_threshold_m:.1f}m"

                cv2.putText(
                    canvas,
                    tether_text,
                    (mid_x, max(25, mid_y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            if session.alarm_active:
                any_active_alarm = True

        # 3. Draw translucent Real-Time Monitoring Dashboard overlay (top-left)
        panel_rows = max(1, len(sessions)) + min(3, len(recent_events))
        panel_h = min(h - 20, 65 + panel_rows * 24)
        overlay = canvas.copy()
        cv2.rectangle(overlay, (12, 12), (560, panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

        cv2.putText(
            canvas,
            "AI FORKLIFT & DRIVER PROXIMITY MONITOR",
            (24, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        y_off = 62
        for fid, session in sessions.items():
            if session.state == VehicleSafetyState.OPERATED:
                d_label = "0.00m (IN CABIN)"
            elif session.last_measurement is not None:
                if not session.last_measurement.is_certain:
                    d_label = f"UNCERTAIN [{session.last_measurement.uncertainty_reason}]"
                else:
                    d_label = f"{session.last_measurement.distance_m:.2f}m"
            else:
                d_label = "N/A"

            row = (
                f"F#{fid} | Driver: {session.driver_id or '-'} | "
                f"{session.state.value} | Dist: {d_label}"
            )
            cv2.putText(
                canvas,
                row,
                (24, y_off),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                self.STATE_COLORS[session.state],
                1,
            )
            y_off += 22

        # 4. Flashing Visual Alarm Banner on Threshold Violation
        if any_active_alarm and (self._frame_counter % 10 < 7):
            cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), (0, 0, 255), 10)
            cv2.rectangle(
                canvas, (w // 2 - 310, 16), (w // 2 + 310, 64), (0, 0, 210), -1
            )
            cv2.putText(
                canvas,
                "ALARM: DRIVER EXCEEDED SAFE FORKLIFT DISTANCE!",
                (w // 2 - 290, 47),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (255, 255, 255),
                2,
            )

        return canvas