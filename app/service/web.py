# app/service/web.py
"""
Telemetry hub + optional web API (FastAPI/uvicorn, started only when web.enabled is true).

    GET /api/status      service health: fps, stream state, identity statistics, active alarms
    GET /api/telemetry   persons, zones (occupants + dwell), line/region counts, forklifts
    GET /api/events      recent events (?limit=50&min_severity=warning)
    GET /video_feed      MJPEG of the annotated frames (rate-limited, downscaled)
If `web.token` is set every request needs `?token=...` or the header `X-Token: ...`. Bind to 127.0.0.1 unless the
network is protected (no TLS here: put a reverse proxy in front for remote access).
"""
# NOTE: no `from __future__ import annotations` here. FastAPI resolves the endpoint parameter types at decoration
# time; with postponed (string) annotations it cannot see the locally imported `Request` and would treat it as a
# query parameter (HTTP 422 on every call).
import json
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

from app.core.events import _json_default

logger = logging.getLogger("FactoryReID")


def _jsonable(obj: Any) -> Any:
    """Plain JSON types only (NumPy scalars/arrays from the pipeline would break the HTTP response)."""
    return json.loads(json.dumps(obj, default=_json_default))


class TelemetryHub:
    def __init__(self, stream_max_fps: float = 10.0, stream_width: int = 1280, jpeg_enabled: bool = False):
        self._lock = threading.Lock()
        self._snapshot: Dict[str, Any] = {"status": {}, "telemetry": {}}
        self._jpeg = b""
        self._last_jpeg = 0.0
        self.stream_max_fps = float(stream_max_fps)
        self.stream_width = int(stream_width)
        self.jpeg_enabled = bool(jpeg_enabled)
        self.events_provider: Optional[Callable[[int, str], List[Dict[str, Any]]]] = None

    def publish(self, status: Dict[str, Any], telemetry: Dict[str, Any], frame: Optional[np.ndarray] = None) -> None:
        jpeg = None
        now = time.monotonic()
        if self.jpeg_enabled and frame is not None and now - self._last_jpeg >= 1.0 / max(self.stream_max_fps, 0.1):
            self._last_jpeg = now
            h, w = frame.shape[:2]
            small = frame
            if self.stream_width > 0 and w > self.stream_width:
                small = cv2.resize(frame, (self.stream_width, int(h * self.stream_width / w)),
                                   interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            jpeg = buf.tobytes() if ok else None
        snapshot = _jsonable({"status": status, "telemetry": telemetry})
        with self._lock:
            self._snapshot = snapshot
            if jpeg is not None:
                self._jpeg = jpeg

    def get(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._snapshot)

    def jpeg(self) -> bytes:
        with self._lock:
            return self._jpeg


def start_web_service(hub: TelemetryHub, host: str = "127.0.0.1", port: int = 8050, token: str = "") -> None:
    def _run() -> None:
        try:
            from fastapi import FastAPI, HTTPException, Request
            from fastapi.responses import JSONResponse, StreamingResponse
            import uvicorn
        except ImportError:
            logger.error("web.enabled is true but fastapi/uvicorn are not installed (pip install fastapi uvicorn)")
            return

        app = FastAPI(title="Factory Monitoring")

        def _auth(request: Request) -> None:
            if token and request.query_params.get("token") != token and request.headers.get("X-Token") != token:
                raise HTTPException(status_code=401, detail="invalid token")

        @app.get("/api/status")
        def status(request: Request):
            _auth(request)
            return JSONResponse(hub.get().get("status", {}))

        @app.get("/api/telemetry")
        def telemetry(request: Request):
            _auth(request)
            return JSONResponse(hub.get().get("telemetry", {}))

        @app.get("/api/events")
        def events(request: Request, limit: int = 50, min_severity: str = "info"):
            _auth(request)
            items = hub.events_provider(limit, min_severity) if hub.events_provider else []
            return JSONResponse(_jsonable(items))

        @app.get("/video_feed")
        def video_feed(request: Request):
            _auth(request)

            def _gen():
                while True:
                    data = hub.jpeg()
                    if data:
                        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
                    time.sleep(1.0 / max(hub.stream_max_fps, 0.5))

            return StreamingResponse(_gen(), media_type="multipart/x-mixed-replace; boundary=frame")

        try:
            uvicorn.run(app, host=host, port=int(port), log_level="warning")
        except Exception:
            logger.exception("Web service stopped")

    threading.Thread(target=_run, name="web-service", daemon=True).start()
    logger.info("Web service on http://%s:%s (api: /api/status /api/telemetry /api/events, video: /video_feed)",
                host, port)
