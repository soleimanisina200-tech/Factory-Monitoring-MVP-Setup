import threading
from typing import Dict, List
import cv2
import numpy as np
from app.vehicle_monitoring.driver_vehicle_fsm import VehicleSafetyState, VehicleSessionRecord


class VehicleSafetyDashboard:
    """
    Renders the Real-Time OpenCV Heads-Up Display (HUD) for active vehicles only,
    preventing ghost bounding boxes and displaying accurate parked vs. active status.
    """

    STATE_COLORS = {
        VehicleSafetyState.UNASSIGNED: (160, 160, 160),               # Gray (Parked / No Driver)
        VehicleSafetyState.DRIVER_ONBOARD: (0, 200, 0),               # Green
        VehicleSafetyState.DRIVER_EXITED_SAFE: (255, 200, 0),         # Cyan
        VehicleSafetyState.DRIVER_EXITED_WARNING: (0, 180, 255),      # Orange/Amber
        VehicleSafetyState.DISTANCE_VIOLATION_ALARM: (0, 0, 255),     # Red
        VehicleSafetyState.DISTANCE_UNCERTAIN: (255, 0, 255),         # Magenta
    }

    def __init__(self, config: dict):
        self.cfg = config
        self.safe_thresh_m = float(config.get("distance", {}).get("safe_distance_threshold_m", 3.0))
        self._latest_frame_jpeg: bytes = b""
        self._latest_telemetry: Dict = {"vehicles": [], "events": []}
        self._lock = threading.Lock()

        dash_cfg = config.get("dashboard", {})
        if dash_cfg.get("enable_web_dashboard", False):
            self._start_web_server(
                host=dash_cfg.get("host", "0.0.0.0"),
                port=int(dash_cfg.get("port", 8050))
            )

    def render_hud(
        self,
        frame: np.ndarray,
        vehicles: Dict[str, VehicleSessionRecord],
        recent_events: List[Dict]
    ) -> np.ndarray:
        canvas = frame.copy()
        h, w = canvas.shape[:2]
        any_alarm_active = False

        # Only render vehicles currently visible in this frame
        visible_vehicles = {
            vid: rec for vid, rec in vehicles.items() if rec.is_visible_in_current_frame
        }

        for vid, rec in visible_vehicles.items():
            color = self.STATE_COLORS.get(rec.state, (160, 160, 160))
            vx1, vy1, vx2, vy2 = map(int, rec.bbox)

            thickness = 3 if rec.associated_driver_id is not None else 1
            cv2.rectangle(canvas, (vx1, vy1), (vx2, vy2), color, thickness)

            if rec.state == VehicleSafetyState.UNASSIGNED:
                header = f"Vehicle #{vid} | PARKED (No Driver)"
            else:
                header = f"Vehicle #{vid} | Driver: #{rec.associated_driver_id} | {rec.state.value}"

            cv2.putText(canvas, header, (vx1, max(25, vy1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)

            # Draw Associated Driver & Real-World Distance Tether ONLY for verified exited drivers
            if (
                rec.associated_driver_id is not None
                and rec.driver_bbox is not None
                and rec.state not in (VehicleSafetyState.UNASSIGNED, VehicleSafetyState.DRIVER_ONBOARD)
            ):
                dx1, dy1, dx2, dy2 = map(int, rec.driver_bbox)
                cv2.rectangle(canvas, (dx1, dy1), (dx2, dy2), color, 2)

                meas = rec.last_measurement
                if meas is not None and meas.driver_ground_px != (0, 0):
                    p1 = meas.driver_ground_px
                    p2 = meas.vehicle_nearest_px
                    cv2.circle(canvas, p1, 6, color, -1)
                    cv2.circle(canvas, p2, 6, color, -1)
                    cv2.line(canvas, p1, p2, color, 2, cv2.LINE_AA)

                    mid_pt = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2 - 10)
                    if meas.is_uncertain:
                        dist_label = f"DIST: UNCERTAIN ({meas.uncertainty_reason})"
                    else:
                        dist_label = f"{meas.distance_m:.2f}m / Max {self.safe_thresh_m:.1f}m"

                    cv2.putText(canvas, dist_label, mid_pt,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            if rec.alarm_active:
                any_alarm_active = True

        # Top-Left Telemetry Summary Panel (prioritize vehicles with assigned drivers)
        sorted_vehicles = sorted(
            visible_vehicles.values(),
            key=lambda r: (r.associated_driver_id is None, int(r.vehicle_id) if r.vehicle_id.isdigit() else 999)
        )[:10]

        panel_h = 45 + 26 * max(1, len(sorted_vehicles))
        overlay = canvas.copy()
        cv2.rectangle(overlay, (10, 10), (560, min(h - 10, panel_h)), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

        cv2.putText(canvas, "VEHICLE & DRIVER SAFETY MONITOR (METRIC)", (20, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2)

        y_cursor = 60
        for rec in sorted_vehicles:
            if rec.state == VehicleSafetyState.UNASSIGNED:
                dist_str = "PARKED / NO DRIVER"
            elif rec.state == VehicleSafetyState.DRIVER_ONBOARD:
                dist_str = "ONBOARD (0.00 m)"
            elif rec.last_measurement is not None:
                if rec.last_measurement.is_uncertain:
                    dist_str = f"UNCERTAIN [{rec.last_measurement.uncertainty_reason}]"
                elif rec.last_measurement.distance_m is not None:
                    dist_str = f"{rec.last_measurement.distance_m:.2f} m"
                else:
                    dist_str = "N/A"
            else:
                dist_str = "N/A"

            row_text = (
                f"V#{rec.vehicle_id} <-> D#{rec.associated_driver_id or '-'} | "
                f"{rec.state.value} | {dist_str}"
            )
            cv2.putText(canvas, row_text, (20, y_cursor),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.STATE_COLORS[rec.state], 1)
            y_cursor += 24

        if any_alarm_active:
            cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), (0, 0, 255), 10)
            cv2.rectangle(canvas, (w // 2 - 290, 15), (w // 2 + 290, 65), (0, 0, 220), -1)
            cv2.putText(canvas, "ALARM: DRIVER EXCEEDED SAFE FORKLIFT DISTANCE!",
                        (w // 2 - 275, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)

        with self._lock:
            ok, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                self._latest_frame_jpeg = buf.tobytes()
            self._latest_telemetry = {
                "vehicles": [
                    {
                        "vehicle_id": v.vehicle_id,
                        "driver_id": v.associated_driver_id,
                        "state": v.state.value,
                        "distance_m": v.last_measurement.distance_m if v.last_measurement else None,
                        "is_uncertain": v.last_measurement.is_uncertain if v.last_measurement else False,
                        "uncertainty_reason": v.last_measurement.uncertainty_reason if v.last_measurement else "",
                        "alarm_active": v.alarm_active,
                    }
                    for v in sorted_vehicles
                ],
                "events": recent_events[:15],
            }

        return canvas

    def _start_web_server(self, host: str, port: int):
        def _run():
            try:
                from fastapi import FastAPI
                from fastapi.responses import JSONResponse, StreamingResponse
                import uvicorn

                app = FastAPI(title="Forklift & Driver Safety Monitoring Dashboard")

                @app.get("/api/telemetry")
                def get_telemetry():
                    with self._lock:
                        return JSONResponse(self._latest_telemetry)

                @app.get("/video_feed")
                def video_feed():
                    def _gen():
                        import time
                        while True:
                            with self._lock:
                                frame_bytes = self._latest_frame_jpeg
                            if frame_bytes:
                                yield (b"--frame\r\n"
                                       b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
                            time.sleep(0.05)
                    return StreamingResponse(_gen(), media_type="multipart/x-mixed-replace; boundary=frame")

                uvicorn.run(app, host=host, port=port, log_level="warning")
            except Exception as e:
                print(f"[VehicleSafetyDashboard] Web server skipped or stopped: {e}")

        threading.Thread(target=_run, daemon=True).start()