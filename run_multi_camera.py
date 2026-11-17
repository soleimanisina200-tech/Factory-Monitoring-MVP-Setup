"""
Multi-camera supervisor: one independent main.py process per camera of configs/cameras.yaml.

    python run_multi_camera.py                              # all cameras, headless
    python run_multi_camera.py --only cam1 cam3             # a subset
    python run_multi_camera.py --gui cam1                   # show the window of one camera

Each camera has its own detector, tracker and local worker IDs; the shared registry (data/cross_camera.db) gives the
same worker the same PERSON id (P#) in every camera. A camera process that dies is restarted (with back-off)
without touching the others. Ctrl+C stops all of them cleanly (identities and events are flushed).
"""
from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from app.multicam.cameras import DEFAULT_CAMERAS_PATH, camera_ids

IS_WINDOWS = sys.platform.startswith("win")


class CameraProcess:
    def __init__(self, camera_id: str, cameras_path: str, gui: bool, max_frames: int = 0):
        self.camera_id = camera_id
        self.cmd = [sys.executable, str(Path(__file__).with_name("main.py")), "--cameras", cameras_path,
                    "--camera-id", camera_id] + ([] if gui else ["--no-gui"])
        if max_frames > 0:
            self.cmd += ["--max-frames", str(max_frames)]
        self.proc: Optional[subprocess.Popen] = None
        self.restarts = 0
        self.next_start = 0.0
        self.delay = 5.0
        self.started_at = 0.0

    def start(self) -> None:
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if IS_WINDOWS else 0
        self.proc = subprocess.Popen(self.cmd, creationflags=flags)
        self.started_at = time.time()
        print(f"[supervisor] {self.camera_id}: started (pid {self.proc.pid})", flush=True)

    def poll(self, restart: bool) -> None:
        if self.proc is None:
            if restart and time.time() >= self.next_start:
                self.start()
            return
        code = self.proc.poll()
        if code is None:
            if time.time() - self.started_at > 300.0:
                self.delay = 5.0          # ran fine for 5 minutes: reset the back-off
            return
        self.proc = None
        if code == 0 or not restart:
            print(f"[supervisor] {self.camera_id}: finished (exit {code})", flush=True)
            self.next_start = float("inf")
            return
        self.restarts += 1
        self.next_start = time.time() + self.delay
        print(f"[supervisor] {self.camera_id}: exited with code {code}; restart #{self.restarts} in "
              f"{self.delay:.0f} s", flush=True)
        self.delay = min(300.0, self.delay * 2.0)

    def stop(self, timeout: float = 20.0) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        try:
            self.proc.send_signal(signal.CTRL_BREAK_EVENT if IS_WINDOWS else signal.SIGTERM)
            self.proc.wait(timeout=timeout)
        except Exception:
            self.proc.kill()

    @property
    def done(self) -> bool:
        return self.proc is None and self.next_start == float("inf")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run every camera of configs/cameras.yaml in its own process")
    ap.add_argument("--cameras", default=DEFAULT_CAMERAS_PATH)
    ap.add_argument("--only", nargs="*", default=None, help="run only these camera ids")
    ap.add_argument("--gui", nargs="*", default=[], help="camera ids that open a display window")
    ap.add_argument("--no-restart", action="store_true", help="do not restart a camera process that failed")
    ap.add_argument("--max-frames", type=int, default=0, help="stop every camera after N frames (testing)")
    args = ap.parse_args()

    ids: List[str] = camera_ids(args.cameras)
    if args.only:
        unknown = [c for c in args.only if c not in ids]
        if unknown:
            raise SystemExit(f"Unknown camera ids {unknown}; {args.cameras} defines {ids}")
        ids = [c for c in ids if c in args.only]
    procs: Dict[str, CameraProcess] = {
        c: CameraProcess(c, args.cameras, c in args.gui, args.max_frames) for c in ids}
    print(f"[supervisor] cameras: {ids}", flush=True)
    for p in procs.values():
        p.start()
        time.sleep(2.0)   # stagger model loading on the GPU
    try:
        while not all(p.done for p in procs.values()):
            for p in procs.values():
                p.poll(restart=not args.no_restart)
            time.sleep(2.0)
    except KeyboardInterrupt:
        print("[supervisor] stopping all cameras ...", flush=True)
    finally:
        for p in procs.values():
            p.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
