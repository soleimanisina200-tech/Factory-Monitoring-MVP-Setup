"""
End-to-end run of main.FactoryMonitor with scripted fake models (no GPU / weights / video needed):
real identity manager, zones, lines, forklift state machine, event store, alarm dispatcher and report.

Scene (800x800, 100 px = 1 m, 10 FPS):
  worker A walks left -> right: crosses the line x=300 and passes through the zone x 350..550
  worker B sits in the cab of a parked forklift, climbs out and walks away -> distance alarm
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

import main as app_main
from app.core.config import load_settings
from app.core.scene import Scene, save_scene
from app.detection.detector import PoseDetection
from app.identity.reid_model import CropQualityReport
from app.logic.calibration import GroundCalibration, save_calibration

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import report as report_script  # noqa: E402

N_FRAMES = 140
FORKLIFT = [600.0, 150.0, 700.0, 300.0]


def _emb(i):
    v = np.zeros(16, np.float32)
    v[i] = 1.0
    return v


class FakeReader:
    def __init__(self, **_):
        self.width, self.height, self.nominal_fps, self.is_live = 800, 800, 10.0, False
        self.i, self._ended = 0, False

    def read(self):
        if self.i >= N_FRAMES:
            self._ended = True
            return False, None, 0.0
        self.i += 1
        return True, np.zeros((800, 800, 3), np.uint8), self.i * 0.1

    @property
    def ended(self):
        return self._ended

    def status(self):
        return {"seconds_since_last_frame": 0.0, "connected": True}

    def release(self):
        pass


class FakeDetector:
    def __init__(self, **_):
        self.k = 0

    def track_frame(self, frame, active_track_ids=None):
        self.k += 1
        k = self.k
        dets = []

        def person(tid, x1, y1, x2, y2, feet_visible=True):
            kxy, kconf = np.zeros((17, 2), np.float32), np.zeros(17, np.float32)
            if feet_visible:                            # ankles seen by the pose model -> certain ground point
                kxy[15], kxy[16] = (0.5 * (x1 + x2) - 5, y2), (0.5 * (x1 + x2) + 5, y2)
                kconf[15] = kconf[16] = 0.9
            dets.append(PoseDetection(track_id=tid, bbox_xyxy=np.array([x1, y1, x2, y2], np.float32), conf=0.9,
                                      keypoints_xy=kxy, keypoints_conf=kconf))

        if k <= 110:                                    # worker A: feet at y=500, 5 px per frame
            x = 100.0 + 5.0 * k
            person(1, x - 30, 340, x + 30, 500)
        if k <= 60:                                     # worker B in the cab (legs hidden by the chassis)
            person(2, 620, 170, 680, 290, feet_visible=False)
        elif k <= 75:                                   # climbs out next to the forklift (feet 0.4 m away)
            person(2, 710, 180, 770, 300)
        else:                                           # walks away to the left, 4 px per frame (0.4 m/s)... far
            x = max(120.0, 710.0 - 12.0 * (k - 75))
            person(2, x, 180, x + 60, 300)
        return dets


class FakeReid:
    def __init__(self, **_):
        pass

    def evaluate_and_extract(self, frame, bbox, kxy, kconf, inter):
        emb = _emb(0) if float(bbox[3]) > 400 else _emb(1)      # A (feet at 500) vs B
        return emb, CropQualityReport(True, True, 50.0, 1.0, "FRONT", "OK")


class _Boxes:
    def __init__(self):
        self.xyxy = torch.tensor([FORKLIFT])
        self.id = torch.tensor([1.0])
        self.cls = torch.tensor([7.0])
        self.conf = torch.tensor([0.9])


class _Result:
    boxes = _Boxes()
    masks = None


class FakeYOLO:
    names = {0: "person", 7: "truck"}

    def __init__(self, *_a, **_k):
        pass

    def track(self, frame, **_):
        return [_Result()]


@pytest.fixture()
def monitor(tmp_path, monkeypatch):
    save_calibration(GroundCalibration([[100, 100], [700, 100], [700, 700], [100, 700]],
                                       [[0, 0], [6, 0], [6, 6], [0, 6]], (800, 800)), str(tmp_path / "cal.yaml"))
    save_scene(Scene(image_wh=(800, 800),
                     zones={"dock": {"polygon": [[350, 400], [550, 400], [550, 600], [350, 600]]}},
                     lines={"gate": {"a": [300, 300], "b": [300, 700], "dead_band_px": 10}}),
               str(tmp_path / "scene.yaml"))
    veh = tmp_path / "vehicle.yaml"
    veh.write_text(yaml.safe_dump({"vehicle_monitoring": {
        "vehicle_class_ids": [7],
        "pairing": {"board_confirm_sec": 1.0},
        "distance": {"ema_smoothing_alpha": 1.0, "max_step_jump_m": 50.0},
    }}), encoding="utf-8")
    settings = tmp_path / "settings.yaml"
    settings.write_text(yaml.safe_dump({
        "system": {"display_gui": False, "device": "cpu", "log_dir": str(tmp_path / "logs"),
                   "heartbeat_path": str(tmp_path / "logs" / "hb.json"), "camera_id": "camT"},
        "stream": {"source": "fake.mp4", "recording_start_time": "2026-11-17 08:00:00"},
        "recording": {"output_video_path": ""},
        "reid": {"decision_log_path": str(tmp_path / "logs" / "dec.jsonl"), "persistence": {"mode": "off"}},
        "scene": {"path": str(tmp_path / "scene.yaml")},
        "events": {"db_path": str(tmp_path / "events.db"), "csv_dir": str(tmp_path / "csv"),
                   "snapshot_dir": str(tmp_path / "snap")},
        "alarms": {"enable_audio_alarm": False},
    }), encoding="utf-8")
    monkeypatch.setattr(app_main, "VariableFPSVideoReader", FakeReader)
    monkeypatch.setattr(app_main, "IndustrialPoseDetector", FakeDetector)
    monkeypatch.setattr(app_main, "PoseGuidedOSNetExtractor", FakeReid)
    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)
    m = app_main.FactoryMonitor(load_settings(str(settings)), str(veh), str(tmp_path / "cal.yaml"))
    yield m, tmp_path


def test_full_pipeline_produces_attributed_events_and_a_report(monitor):
    m, tmp = monitor
    m.run()
    m.close()
    assert m.frames == N_FRAMES and m.errors == 0, m.last_error

    conn = sqlite3.connect(str(tmp / "events.db"))
    rows = conn.execute("SELECT type, severity, worker_id, subject, direction, value, vehicle_id, wall_time "
                        "FROM events ORDER BY id").fetchall()
    types = [r[0] for r in rows]
    by = lambda t: [r for r in rows if r[0] == t]   # noqa: E731

    # worker A walks left -> right over the line (OUT side) and through the zone; worker B later walks right -> left
    # over the same line (IN): each passage is attributed to the right worker with its direction
    cross = by("LINE_CROSS")
    assert [(r[3], r[4]) for r in cross] == [("gate", "OUT"), ("gate", "IN")]
    assert cross[0][2] is not None and cross[1][2] not in (None, cross[0][2])
    a_id = cross[0][2]
    zexit = by("ZONE_EXIT")
    assert len(zexit) == 1 and zexit[0][2] == a_id and zexit[0][3] == "dock"
    assert 3.4 <= zexit[0][5] <= 4.6                       # 200 px at 50 px/s = 4 s in the zone

    # worker B: driver of the forklift, dismount, then distance alarm (with snapshot)
    boarded = by("DRIVER_BOARDED")
    assert len(boarded) == 1 and boarded[0][6] == "1" and boarded[0][2] not in (None, a_id)
    b_id = boarded[0][2]
    assert cross[1][2] == b_id
    assert by("DRIVER_EXITED")[0][2] == b_id
    alarm = by("SAFE_DISTANCE_VIOLATION_ALARM")
    assert len(alarm) == 1 and alarm[0][1] == "alarm" and alarm[0][2] == b_id and alarm[0][5] > 3.0
    assert len(list((tmp / "snap").rglob("*.jpg"))) == 1
    # stopping the service closes the session in progress, so the seat time is not lost
    end = conn.execute("SELECT worker_id, value, details FROM events WHERE type = 'DRIVER_SESSION_END'").fetchall()
    assert len(end) == 1 and end[0][0] == b_id and 5.5 <= end[0][1] <= 6.5 and '"SHUTDOWN"' in end[0][2]
    assert rows[0][7].startswith("2026-11-17 08:00")       # wall time = recording start + position in the clip
    assert "DRIVER_BOARDED" in types and types.index("DRIVER_BOARDED") < types.index("DRIVER_EXITED")

    # per-worker report
    t0 = report_script.datetime(2026, 11, 17).timestamp()
    events, resolve, labels = report_script.load(str(tmp / "events.db"), t0, t0 + 86400, None)
    rep = report_script.build(events, resolve, labels)
    w = rep["workers"]
    assert w[str(a_id)]["zone_visits"]["dock"] == 1 and sum(w[str(a_id)]["cross"].values()) == 1
    assert w[str(b_id)]["alarms"] == 1
    text = report_script.write(rep, tmp / "report", "test")
    assert f"Worker {a_id}" in text and (tmp / "report" / "workers.csv").exists()

    hb = (tmp / "logs" / "hb.json")
    assert hb.exists() and '"frames"' in hb.read_text(encoding="utf-8")
