import cv2
import yaml
import argparse
from pathlib import Path
from app.vehicle_monitoring.pipeline import VehicleMonitoringPipeline


def load_existing_reid_if_available():
    """Attempts to reuse the existing project's ReIDModel from app/identity/reid_model.py."""
    try:
        from app.identity.reid_model import ReIDModel
        return ReIDModel()
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="AI Vehicle & Driver Safety Monitoring Runner")
    parser.add_argument("--source", type=str, default=None, help="Video path or RTSP stream (defaults to configs/settings.yaml)")
    parser.add_argument("--config", type=str, default="configs/vehicle_monitoring.yaml", help="Vehicle monitoring config")
    args = parser.parse_args()

    video_source = args.source
    if video_source is None and Path("configs/settings.yaml").exists():
        with open("configs/settings.yaml", "r", encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f)
            # Automatically pick up video source from existing settings.yaml
            video_source = (
                base_cfg.get("video_source")
                or base_cfg.get("stream", {}).get("source")
                or 0
            )

    reid_model = load_existing_reid_if_available()
    pipeline = VehicleMonitoringPipeline(config_path=args.config, reid_extractor=reid_model)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video stream: {video_source}")

    print("[VehicleMonitoring] Running real-time monitoring. Press 'q' to stop.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame = pipeline.process_frame(frame)
        cv2.imshow("Factory Monitoring - Vehicle & Driver Safety", annotated_frame)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()