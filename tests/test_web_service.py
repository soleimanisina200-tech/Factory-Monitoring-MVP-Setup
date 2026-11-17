"""Web API: endpoints answer, the token is enforced, NumPy values do not break the JSON response."""
import json
import socket
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from app.service.web import TelemetryHub, start_web_service  # noqa: E402


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get(port, path, headers=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", headers=headers or {})
    for _ in range(60):
        try:
            with urllib.request.urlopen(req, timeout=3) as r:
                return r.status, r.headers.get("content-type", ""), r.read(4096)
        except urllib.error.HTTPError as e:
            return e.code, "", e.read()
        except OSError:
            time.sleep(0.2)          # server thread still starting
    raise AssertionError("web service did not start")


def test_web_api_endpoints_and_token():
    port = _free_port()
    hub = TelemetryHub(jpeg_enabled=True)
    hub.events_provider = lambda limit, sev: [{"type": "ZONE_ENTER", "value": np.float32(1.5)}][:limit]
    start_web_service(hub, "127.0.0.1", port, token="secret")
    hub.publish({"fps": np.float64(12.5), "frames": np.int64(3)}, {"persons": [{"bbox": np.array([1.0, 2.0])}]},
                np.zeros((90, 160, 3), np.uint8))

    code, _, body = _get(port, "/api/status?token=secret")
    assert code == 200 and json.loads(body) == {"fps": 12.5, "frames": 3}
    code, _, body = _get(port, "/api/telemetry", headers={"X-Token": "secret"})
    assert code == 200 and json.loads(body)["persons"][0]["bbox"] == [1.0, 2.0]
    code, _, body = _get(port, "/api/events?token=secret&limit=5")
    assert code == 200 and json.loads(body) == [{"type": "ZONE_ENTER", "value": 1.5}]
    assert _get(port, "/api/status")[0] == 401 and _get(port, "/api/status?token=wrong")[0] == 401
    code, ctype, chunk = _get(port, "/video_feed?token=secret")
    assert code == 200 and "multipart/x-mixed-replace" in ctype and b"image/jpeg" in chunk
