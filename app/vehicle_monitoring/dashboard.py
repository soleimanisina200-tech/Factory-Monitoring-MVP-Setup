# app/vehicle_monitoring/dashboard.py
"""On-frame HUD of the forklift module (the web dashboard lives in app/service/web.py)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from app.vehicle_monitoring.driver_vehicle_fsm import VehicleSafetyState, VehicleSessionRecord


def blend_rect(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, color, alpha: float) -> None:
    """Semi-transparent filled rectangle, blended on the rectangle ROI only (not the whole frame)."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    fill = np.empty_like(roi)
    fill[:] = color
    cv2.addWeighted(fill, alpha, roi, 1.0 - alpha, 0, roi)


def _mmss(sec: float) -> str:
    sec = max(0, int(sec))
    return f"{sec // 3600:d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}" if sec >= 3600 else f"{sec // 60:02d}:{sec % 60:02d}"


class VehicleSafetyDashboard:
    STATE_COLORS = {
        VehicleSafetyState.UNASSIGNED: (160, 160, 160),
        VehicleSafetyState.OCCUPIED_UNKNOWN_DRIVER: (200, 120, 0),
        VehicleSafetyState.DRIVER_ONBOARD: (0, 200, 0),
        VehicleSafetyState.DRIVER_EXITED_SAFE: (255, 200, 0),
        VehicleSafetyState.DRIVER_EXITED_WARNING: (0, 180, 255),
        VehicleSafetyState.DISTANCE_VIOLATION_ALARM: (0, 0, 255),
        VehicleSafetyState.DISTANCE_UNCERTAIN: (255, 0, 255),
        VehicleSafetyState.DRIVER_NOT_VISIBLE: (180, 0, 180),
        VehicleSafetyState.DRIVER_NOT_VISIBLE_ALARM: (0, 0, 200),
        VehicleSafetyState.DRIVER_LEFT_VIEW_ALARM: (0, 0, 200),
    }

    def __init__(self, config: dict, estimator: Any = None):
        self.cfg = config
        self.safe_thresh_m = float((config.get("distance") or {}).get("safe_distance_threshold_m", 3.0))
        self.draw_footprint = bool((config.get("dashboard") or {}).get("draw_footprint", True))
        self.estimator = estimator
        self.labels: Dict[str, str] = {}

    def _driver_name(self, pid: Optional[str]) -> str:
        if pid is None:
            return "-"
        return f"{pid} ({self.labels[pid]})" if pid in self.labels else str(pid)

    def render_hud(self, frame: np.ndarray, vehicles: Dict[str, VehicleSessionRecord],
                   recent_events: Optional[List[Dict]] = None, now: Optional[float] = None) -> np.ndarray:
        """Draws IN PLACE on `frame` and returns it."""
        h, w = frame.shape[:2]
        any_alarm = False
        visible = [r for r in vehicles.values() if r.is_visible_in_current_frame]
        for rec in visible:
            color = self.STATE_COLORS.get(rec.state, (160, 160, 160))
            vx1, vy1, vx2, vy2 = map(int, rec.bbox)
            occupied = rec.associated_driver_id is not None or rec.unknown_occupied
            cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), color, 3 if occupied else 1)
            if self.draw_footprint and self.estimator is not None and self.estimator.is_metric:
                if rec.footprint_world is None:
                    rec.footprint_world, _ = self.estimator.vehicle_footprint(rec.bbox, frame.shape, rec.contour)
                px = [self.estimator.world_to_pixel(p) for p in rec.footprint_world]
                if all(p is not None for p in px):
                    cv2.polylines(frame, [np.array(px, dtype=np.int32)], True, color, 1, cv2.LINE_AA)

            if rec.state == VehicleSafetyState.UNASSIGNED:
                header = f"Forklift #{rec.vehicle_id} | {'MOVING' if rec.is_currently_moving else 'PARKED'}"
            elif rec.state == VehicleSafetyState.OCCUPIED_UNKNOWN_DRIVER:
                header = f"Forklift #{rec.vehicle_id} | DRIVER HIDDEN (unknown)"
            else:
                hidden = " [hidden/roof]" if rec.driver_hidden_onboard else ""
                t = f" | {_mmss(rec.onboard_seconds(now))}" if now is not None else ""
                header = f"Forklift #{rec.vehicle_id} | Driver {self._driver_name(rec.associated_driver_id)}{hidden}" \
                         f" | {rec.state.value}{t}"
            cv2.putText(frame, header, (vx1, max(25, vy1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)

            if (rec.associated_driver_id is not None and rec.driver_bbox is not None and rec.driver_visible
                    and not rec.onboard):
                dx1, dy1, dx2, dy2 = map(int, rec.driver_bbox)
                cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), color, 2)
                meas = rec.last_measurement
                if meas is not None and meas.distance_m is not None:
                    p1, p2 = meas.driver_ground_px, meas.vehicle_nearest_px
                    cv2.circle(frame, p1, 6, color, -1)
                    cv2.circle(frame, p2, 6, color, -1)
                    cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)
                    mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2 - 10)
                    txt = f"{meas.distance_m:.2f}m / Max {self.safe_thresh_m:.1f}m"
                    if meas.is_uncertain:
                        txt += f" (~ {meas.uncertainty_reason})"
                    cv2.putText(frame, txt, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            any_alarm = any_alarm or rec.alarm_active

        rows = sorted(visible, key=lambda r: (r.associated_driver_id is None,
                                              int(r.vehicle_id) if r.vehicle_id.isdigit() else 1 << 30))[:10]
        if rows:   # no panel when no forklift is in view
            panel_h = 45 + 24 * len(rows)
            blend_rect(frame, 10, 60, 640, min(h - 10, 60 + panel_h), (20, 20, 20), 0.75)
            cv2.putText(frame, "FORKLIFT & DRIVER SAFETY", (20, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y = 112
        for rec in rows:
            if rec.state == VehicleSafetyState.DRIVER_ONBOARD:
                dist = "onboard" + (" (hidden)" if rec.driver_hidden_onboard else "")
            elif rec.last_measurement is not None and rec.last_measurement.distance_m is not None and not rec.onboard:
                dist = f"{rec.last_measurement.distance_m:.2f} m" + ("~" if rec.last_measurement.is_uncertain else "")
            else:
                dist = "-"
            t = f" | seat {_mmss(rec.onboard_seconds(now))}" if (now is not None and rec.associated_driver_id) else ""
            txt = f"F#{rec.vehicle_id} D#{self._driver_name(rec.associated_driver_id)} | {rec.state.value} | {dist}{t}"
            cv2.putText(frame, txt, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, self.STATE_COLORS[rec.state], 1)
            y += 24

        if any_alarm:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 10)
            cv2.rectangle(frame, (w // 2 - 290, 60), (w // 2 + 290, 110), (0, 0, 220), -1)
            cv2.putText(frame, "ALARM: FORKLIFT DRIVER SAFETY", (w // 2 - 270, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
        return frame

    @staticmethod
    def telemetry(vehicles: Dict[str, VehicleSessionRecord], now: float) -> List[Dict[str, Any]]:
        out = []
        for v in vehicles.values():
            m = v.last_measurement
            out.append({
                "vehicle_id": v.vehicle_id,
                "visible": v.is_visible_in_current_frame,
                "moving": v.is_currently_moving,
                "state": v.state.value,
                "driver_id": v.associated_driver_id,
                "driver_visible": v.driver_visible,
                "driver_hidden_onboard": v.driver_hidden_onboard,
                "driver_inferred": v.driver_inferred,
                "onboard_sec": round(v.onboard_seconds(now), 1) if v.associated_driver_id else 0.0,
                "distance_m": None if (m is None or v.onboard) else m.distance_m,
                "distance_uncertain": bool(m.is_uncertain) if m is not None else False,
                "alarm_active": v.alarm_active,
                "alarm_type": v.alarm_type,
            })
        return out
