"""
Standalone forklift / driver module (no pose, no Re-ID: persons carry tracker IDs only). For quick tests of the
forklift part; production runs use main.py, which feeds stable worker IDs, keypoints and foot points.

    python run_vehicle_monitoring.py --source data/input/clip1.mp4
"""
import argparse
import logging
from pathlib import Path

import cv2
import yaml

from app.core.alarms import AlarmDispatcher
from app.core.clock import StreamClock
from app.core.events import EventStore
from app.core.logging_setup import setup_logging
from app.logic.calibration import DEFAULT_CALIBRATION_PATH
from app.streams.video_reader import VariableFPSVideoReader
from app.vehicle_monitoring.pipeline import VehicleMonitoringPipeline

logger = logging.getLogger("FactoryReID")


def main() -> None:
    parser = argparse.ArgumentParser(description="Forklift & driver safety monitoring (standalone)")
    parser.add_argument("--source", default=None, help="video file / RTSP URL (default: stream.source of settings)")
    parser.add_argument("--config", default="configs/vehicle_monitoring.yaml")
    parser.add_argument("--settings", default="configs/settings.yaml")
    parser.add_argument("--calibration", default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--no-gui", action="store_true")
    args = parser.parse_args()
    setup_logging("logs")

    settings = {}
    if Path(args.settings).exists():
        with open(args.settings, "r", encoding="utf-8") as f:
            settings = yaml.safe_load(f) or {}
    source = args.source or (settings.get("stream") or {}).get("source") or 0
    ev_cfg = settings.get("events") or {}

    reader = VariableFPSVideoReader(source)
    clock = StreamClock(reader.is_live)
    store = EventStore(db_path=ev_cfg.get("db_path", "data/factory_events.db"),
                       csv_dir=ev_cfg.get("csv_dir", "data/events"),
                       snapshot_dir=ev_cfg.get("snapshot_dir", "data/snapshots"), clock=clock,
                       camera_id=(settings.get("system") or {}).get("camera_id", "cam1"))
    alarms = AlarmDispatcher(settings.get("alarms") or {})
    store.subscribe(alarms.handle)
    pipeline = VehicleMonitoringPipeline(config_path=args.config, calibration_path=args.calibration,
                                         event_sink=lambda ev, frame: store.emit(ev, frame))

    logger.info("Forklift monitoring running. Press 'q' to stop.")
    try:
        while True:
            ok, frame, ts = reader.read()
            if not ok:
                if reader.ended:
                    break
                continue
            out = pipeline.process_frame(frame, frame_ts=ts)
            if not args.no_gui:
                cv2.imshow("Forklift & Driver Safety", out)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        reader.release()
        store.close()
        alarms.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
