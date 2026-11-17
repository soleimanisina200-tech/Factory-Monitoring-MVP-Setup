import cv2
import yaml
import argparse
from pathlib import Path
from app.vehicle_monitoring.pipeline import VehicleMonitoringPipeline


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

    # Standalone mode has no pose/Re-ID stack: persons carry no embeddings, so the driver Re-ID
    # fallback is simply disabled (a colour-histogram stand-in used to make it match wrong people).
    pipeline = VehicleMonitoringPipeline(config_path=args.config, reid_extractor=None)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video stream: {video_source}")

    print("[VehicleMonitoring] Running real-time monitoring. Press 'q' to stop.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        frame_ts = (pts_ms / 1000.0) if pts_ms and pts_ms > 0 else None
        annotated_frame = pipeline.process_frame(frame, frame_ts=frame_ts)
        cv2.imshow("Factory Monitoring - Vehicle & Driver Safety", annotated_frame)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    pipeline.logger.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()