# app/core/alarms.py
"""
Alarm outputs. Subscribed to the EventStore: every event passes through `AlarmDispatcher.handle`.

Rate limiting (cooldown) applies ONLY to the outputs (beep / webhook / MQTT). The audit record of every alarm is
always written by the EventStore - the old logger skipped the record itself during the cooldown.

Outputs:
  * local beep (winsound on Windows, terminal bell elsewhere)
  * HTTP webhook(s): JSON POST, background thread, retries   -> sirens / PLC gateways / SCADA / chat bots
  * MQTT (optional, needs `pip install paho-mqtt`)
"*_CLEARED" events are forwarded too, so external sirens can be switched off.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from app.core.events import ALARM, SEVERITY_RANK, _json_default

logger = logging.getLogger("FactoryReID")


def _payload(ev: Dict[str, Any]) -> Dict[str, Any]:
    keys = ("type", "severity", "wall_time", "camera_id", "worker_id", "worker_label", "track_id", "vehicle_id",
            "subject", "direction", "value", "details", "snapshot_path")
    return {k: ev.get(k) for k in keys}


class BeepSink:
    def __init__(self, freq_hz: int = 2200, duration_ms: int = 400):
        self.freq_hz, self.duration_ms = int(freq_hz), int(duration_ms)

    def send(self, ev: Dict[str, Any]) -> None:
        if ev.get("severity") != ALARM:
            return

        def _beep() -> None:
            try:
                import winsound

                winsound.Beep(self.freq_hz, self.duration_ms)
            except Exception:
                print("\a", end="", flush=True)

        threading.Thread(target=_beep, daemon=True).start()

    def close(self) -> None:
        pass


class WebhookSink:
    def __init__(self, urls: List[str], headers: Optional[Dict[str, str]] = None, timeout_sec: float = 3.0,
                 retries: int = 3):
        self.urls = [u for u in urls if u]
        self.headers = {"Content-Type": "application/json", **(headers or {})}
        self.timeout = float(timeout_sec)
        self.retries = int(retries)
        self._q: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue(maxsize=1000)
        self._t = threading.Thread(target=self._loop, name="alarm-webhook", daemon=True)
        self._t.start()

    def send(self, ev: Dict[str, Any]) -> None:
        try:
            self._q.put_nowait(_payload(ev))
        except queue.Full:
            logger.error("Webhook queue full: alarm %s not forwarded", ev.get("type"))

    def _post(self, url: str, body: bytes) -> None:
        req = urllib.request.Request(url, data=body, headers=self.headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 (configured URL)
            resp.read()

    def _loop(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            body = json.dumps(item, default=_json_default, ensure_ascii=False).encode("utf-8")
            for url in self.urls:
                for attempt in range(self.retries):
                    try:
                        self._post(url, body)
                        break
                    except Exception as e:
                        if attempt == self.retries - 1:
                            logger.error("Webhook %s failed for %s: %s", url, item.get("type"), e)
                        else:
                            time.sleep(0.5 * (2 ** attempt))

    def close(self) -> None:
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        self._t.join(timeout=2.0)


class MqttSink:
    def __init__(self, host: str, port: int, topic: str, username: str = "", password: str = ""):
        import paho.mqtt.client as mqtt  # optional dependency

        self.topic = topic
        try:
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        except AttributeError:  # paho-mqtt < 2.0
            self.client = mqtt.Client()
        if username:
            self.client.username_pw_set(username, password or None)
        self.client.connect_async(host, int(port))
        self.client.loop_start()

    def send(self, ev: Dict[str, Any]) -> None:
        self.client.publish(self.topic, json.dumps(_payload(ev), default=_json_default, ensure_ascii=False), qos=1)

    def close(self) -> None:
        self.client.loop_stop()
        self.client.disconnect()


class AlarmDispatcher:
    def __init__(self, cfg: Dict[str, Any]):
        cfg = cfg or {}
        self.cooldown_sec = float(cfg.get("alarm_cooldown_sec", 5.0))
        self.min_rank = SEVERITY_RANK.get(str(cfg.get("forward_min_severity", ALARM)), 2)
        self._last: Dict[Tuple[Any, ...], float] = {}
        self.sinks: List[Any] = []
        if cfg.get("enable_audio_alarm", True):
            self.sinks.append(BeepSink(cfg.get("audio_frequency_hz", 2200), cfg.get("audio_duration_ms", 400)))
        urls = cfg.get("webhook_urls") or []
        if urls:
            self.sinks.append(WebhookSink(urls, cfg.get("webhook_headers") or {}, cfg.get("webhook_timeout_sec", 3.0)))
        mq = cfg.get("mqtt") or {}
        if mq.get("enabled"):
            try:
                self.sinks.append(MqttSink(mq.get("host", "127.0.0.1"), mq.get("port", 1883),
                                           mq.get("topic", "factory/alarms"), mq.get("username", ""),
                                           mq.get("password", "")))
            except Exception:
                logger.exception("MQTT alarm output disabled (is paho-mqtt installed and the broker reachable?)")

    def handle(self, ev: Dict[str, Any]) -> None:
        is_clear = str(ev.get("type", "")).endswith("_CLEARED")
        if not is_clear and SEVERITY_RANK.get(ev.get("severity", "info"), 0) < self.min_rank:
            return
        if not is_clear:
            key = (ev.get("type"), ev.get("vehicle_id"), ev.get("subject"), ev.get("worker_id"))
            now = time.monotonic()
            if now - self._last.get(key, -1e9) < self.cooldown_sec:
                return  # the EventStore has already recorded it; only the outputs are rate-limited
            self._last[key] = now
        for sink in self.sinks:
            try:
                sink.send(ev)
            except Exception:
                logger.exception("Alarm output %s failed", type(sink).__name__)

    __call__ = handle

    def close(self) -> None:
        for sink in self.sinks:
            try:
                sink.close()
            except Exception:
                pass
