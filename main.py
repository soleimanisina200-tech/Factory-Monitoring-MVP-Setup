"""
Factory monitoring service: worker identity, zones, directional lines/regions, forklift driver safety.

    python main.py --source data/input/clip1.mp4
    python main.py --source rtsp://user:pass@cam/stream --restart-on-failure      (24/7 live camera)

Per frame:
  detector (YOLO-pose + tracker) -> identity manager (stable worker IDs, re-entry, markers)
  -> zones / lines / regions (per worker, with late attribution of provisional tracks)
  -> forklift detection + driver state machine
  -> one event store (SQLite + CSV + snapshots) -> alarm outputs (beep / webhook / MQTT)
  -> HUD, segmented recording, telemetry API, heartbeat file
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from app.analytics.crossings import CrossingAnalytics
from app.analytics.entities import AttributionBuffer, entity_key
from app.analytics.zones import ZoneAnalytics
from app.core.alarms import AlarmDispatcher
from app.core.clock import StreamClock
from app.core.config import load_settings, persistence_mode
from app.core.events import ALARM, EventStore, make_event
from app.core.logging_setup import setup_logging
from app.core.scene import load_scene
from app.detection.detector import IndustrialPoseDetector, PoseDetection
from app.identity.manager import EnterpriseIdentityManager
from app.identity.open_set import OpenSetCalibrator
from app.identity.reid_model import SKELETON_LIMBS, PoseGuidedOSNetExtractor
from app.logic.calibration import DEFAULT_CALIBRATION_PATH, load_calibration
from app.logic.geometry import MetricBEVProjector
from app.output.recorder import SegmentedVideoWriter, resolve_output_path
from app.service.web import TelemetryHub, start_web_service
from app.streams.video_reader import VariableFPSVideoReader

logger = logging.getLogger("FactoryReID")


def resolve_device(requested: str, half: bool) -> Tuple[str, bool]:
    """Fail-soft device selection with an explicit message (CPU-only PyTorch builds used to crash deep in YOLO)."""
    if "cuda" in str(requested) and not torch.cuda.is_available():
        logger.error(
            "CUDA requested (%s) but not available in torch %s (CUDA build: %s). Falling back to CPU: expect ~1 FPS. "
            "Install the CUDA build of PyTorch from https://pytorch.org/get-started/locally/.",
            requested, torch.__version__, torch.version.cuda,
        )
        return "cpu", False
    if "cuda" in str(requested):
        logger.info("Using %s: %s", requested, torch.cuda.get_device_name(torch.device(requested)))
    return str(requested), half


def build_persons_for_vehicle_monitor(
    detections: List[PoseDetection],
    track_to_global: Dict[int, int],
    track_embeddings: Optional[Dict[int, np.ndarray]] = None,
) -> List[Dict[str, Any]]:
    """
    Person records for the forklift module. The REAL keypoints and ground point are passed with an explicit
    `ground_uncertain` flag. (The old code invented ankle keypoints with 0.75 confidence for anybody whose legs were
    hidden: that turned a worker behind a pallet into a "seated operator" and disabled the distance-uncertainty
    gate. Seated drivers are recognised inside the driver state machine instead.)
    """
    out = []
    for det in detections:
        gid = track_to_global.get(det.track_id)
        out.append({
            "id": str(gid) if gid is not None else f"T{det.track_id}",
            "global_id": gid if gid is not None else -1,
            "track_id": det.track_id,
            "bbox": det.bbox_xyxy.astype(float).tolist(),
            "keypoints": np.column_stack([det.keypoints_xy, det.keypoints_conf]),
            "foot_px": (float(det.ground_uv[0]), float(det.ground_uv[1])),
            "ground_uncertain": bool(det.ground_extrapolated),
            "bev_xy": det.bev_xy,
            "embedding": (track_embeddings or {}).get(det.track_id),
        })
    return out


class FactoryMonitor:
    def __init__(self, cfg: Dict[str, Any], vehicle_config_path: str, calibration_path: str):
        self.cfg = cfg
        sys_cfg, stream_cfg, det_cfg, reid_cfg = cfg["system"], cfg["stream"], cfg["detection"], cfg["reid"]
        bev_cfg = cfg["physics_bev"]
        self.camera_id = str(sys_cfg.get("camera_id", "cam1"))
        self.display = bool(sys_cfg.get("display_gui", True))
        self.max_errors = int(sys_cfg.get("max_consecutive_errors", 100))
        self.heartbeat_path = sys_cfg.get("heartbeat_path") or None
        self.max_stall_sec = float(stream_cfg.get("max_stall_sec", 0.0))
        sys_cfg["device"], sys_cfg["half_precision"] = resolve_device(
            sys_cfg.get("device", "cuda:0"), bool(sys_cfg.get("half_precision", True)))

        # ---- stream & clock -----------------------------------------------------------------------------------
        logger.info("Opening video stream: %s", "(live camera)" if "://" in str(stream_cfg["source"])
                    else stream_cfg["source"])
        self.reader = VariableFPSVideoReader(
            source=stream_cfg["source"],
            drop_frames_on_lag=stream_cfg.get("drop_frames_on_lag", True),
            reconnect_delay_sec=stream_cfg.get("reconnect_delay_sec", 3.0),
            max_reconnect_delay_sec=stream_cfg.get("max_reconnect_delay_sec", 30.0),
            rtsp_transport=stream_cfg.get("rtsp_transport", "tcp"),
            read_timeout_sec=stream_cfg.get("read_timeout_sec", 5.0),
        )
        self.W, self.H = self.reader.width, self.reader.height
        self.clock = StreamClock(self.reader.is_live,
                                 StreamClock.parse_start_time(stream_cfg.get("recording_start_time")))
        logger.info("Stream %dx%d @ %.1f nominal FPS (%s)", self.W, self.H, self.reader.nominal_fps,
                    "live" if self.reader.is_live else "file")

        # ---- detector, calibration, Re-ID ---------------------------------------------------------------------
        self.detector = IndustrialPoseDetector(
            model_weights=det_cfg["model_weights"], tracker_config=det_cfg["tracker_config"],
            device=sys_cfg["device"], conf_thres=det_cfg.get("conf_thres", 0.08),
            iou_thres=det_cfg.get("iou_thres", 0.6), imgsz=det_cfg.get("imgsz", 1536),
            half=sys_cfg.get("half_precision", True),
        )
        calibration = load_calibration(calibration_path)
        if calibration is not None:
            logger.info("Ground calibration loaded from %s (clicked at %s)", calibration_path, calibration.image_wh)
            src, dst, ref = calibration.image_points_px, calibration.world_points_m, calibration.image_wh
        else:
            logger.warning("No %s: metric distances are NOT valid until scripts/calibrate_ground_plane.py is run "
                           "for this camera.", calibration_path)
            src, dst = bev_cfg.get("src_image_pts"), bev_cfg.get("dst_meter_pts")
            ref = bev_cfg.get("calibration_ref_wh") or [self.W, self.H]
        self.projector = MetricBEVProjector(src_image_pts=src, dst_meter_pts=dst,
                                            default_px_per_meter=bev_cfg.get("default_px_per_meter", 110.0),
                                            stream_wh=(self.W, self.H), calib_ref_wh=(int(ref[0]), int(ref[1])))
        self.reid = PoseGuidedOSNetExtractor(
            model_name=reid_cfg["model_name"], weights_path=reid_cfg.get("weights_path") or None,
            device=sys_cfg["device"], kpt_conf_thresh=reid_cfg.get("kpt_conf_thresh", 0.20),
            min_laplacian_var=reid_cfg.get("min_laplacian_var", 10.0),
            head_branch_weight=reid_cfg.get("head_branch_weight", 0.35),
            color_branch_weight=reid_cfg.get("color_branch_weight", 0.20),
            min_gallery_height_px=reid_cfg.get("min_gallery_height_px", 72),
            min_visible_kpts=reid_cfg.get("min_visible_kpts", 5),
        )

        # ---- scene (zones, lines, regions, doors) -------------------------------------------------------------
        self.scene = load_scene(cfg["scene"]["path"], settings=cfg).scaled((self.W, self.H))
        logger.info("Scene: %d zones, %d lines, %d regions, %d exit doors", len(self.scene.zones),
                    len(self.scene.lines), len(self.scene.regions), len(self.scene.exit_regions))

        # ---- optional helmet markers --------------------------------------------------------------------------
        self.registry = None
        marker_reader = None
        mk = cfg["markers"]
        if mk.get("enabled"):
            from app.identity.markers import HelmetMarkerReader, WorkerRegistry

            self.registry = WorkerRegistry(mk.get("registry_path"), mk.get("accept_unregistered", False))
            marker_reader = HelmetMarkerReader(self.registry, mk.get("dictionary"),
                                               int(mk.get("min_person_height_px", 120)),
                                               regions=mk.get("regions") or ["head", "torso"])
            logger.info("Worker markers enabled (%s): %d registered workers", "+".join(marker_reader.regions),
                        len(self.registry.by_marker))

        osc = reid_cfg["open_set"]
        st = reid_cfg["stature"]
        self.identity = EnterpriseIdentityManager(
            reid_extractor=self.reid, bev_projector=self.projector,
            cos_sim_thresh=reid_cfg["cos_sim_thresh"],
            spatial_recovery_sim_thresh=reid_cfg["spatial_recovery_sim_thresh"],
            reentry_sim_thresh=reid_cfg["reentry_sim_thresh"],
            max_worker_speed_mps=bev_cfg["max_worker_speed_mps"],
            max_internal_occlusion_m=bev_cfg["max_internal_occlusion_m"],
            lag_diffusion_coeff=bev_cfg["lag_diffusion_coeff"], base_pose_jitter_m=bev_cfg["base_pose_jitter_m"],
            reid_interval_sec=reid_cfg["reid_interval_sec"], stitch_sim_thresh=reid_cfg["stitch_sim_thresh"],
            stitch_margin=reid_cfg["stitch_margin"], reentry_min_gap_sec=reid_cfg["reentry_min_gap_sec"],
            reentry_margin=reid_cfg["reentry_margin"], reentry_min_samples=reid_cfg["reentry_min_samples"],
            occlusion_max_sec=reid_cfg["occlusion_max_sec"], border_margin_ratio=reid_cfg["border_margin_ratio"],
            track_gap_reset_sec=reid_cfg["track_gap_reset_sec"], exit_edges=reid_cfg["exit_edges"],
            exit_confirm_sec=reid_cfg["exit_confirm_sec"], mismatch_min_span_sec=reid_cfg["mismatch_min_span_sec"],
            anchor_min_height_px=reid_cfg["anchor_min_height_px"],
            track_state_ttl_sec=reid_cfg["track_state_ttl_sec"],
            identity_retention_sec=reid_cfg["identity_retention_sec"],
            max_exited_identities=reid_cfg["max_exited_identities"],
            pool_max_samples=reid_cfg["pool_max_samples"], pool_window_sec=reid_cfg["pool_window_sec"],
            decision_log_path=reid_cfg.get("decision_log_path") or None,
            recover_strict_after_sec=reid_cfg["recover_strict_after_sec"],
            emerge_wait_sec=reid_cfg["emerge_wait_sec"],
            exit_regions={n: s["polygon"] for n, s in self.scene.exit_regions.items()},
            open_set=OpenSetCalibrator(
                base_threshold=reid_cfg["reentry_sim_thresh"], quantile=osc["quantile"], margin=osc["margin"],
                min_samples=osc["min_samples"], max_threshold=osc["max_threshold"], enabled=osc["enabled"]),
            impostor_sample_interval_sec=osc["sample_interval_sec"],
            marker_reader=marker_reader,
            marker_read_interval_sec=mk.get("read_interval_sec", 0.5),
            marker_min_votes=mk.get("min_votes", 2), marker_vote_window_sec=mk.get("vote_window_sec", 4.0),
            stature_enabled=st["enabled"], stature_min_diff=st["min_diff"], stature_min_samples=st["min_samples"],
            stature_model_min_samples=st["model_min_samples"],
            stature_sample_interval_sec=st["sample_interval_sec"],
        )

        # identity persistence (auto = only for live cameras: re-running a clip must not "remember" it)
        pers = reid_cfg["persistence"]
        mode = persistence_mode(pers.get("mode", "auto"))
        self.id_store = None
        self._last_id_save = time.monotonic()
        self.id_save_interval = float(pers.get("save_interval_sec", 60.0))
        if mode == "on" or (mode == "auto" and self.reader.is_live):
            from app.identity.store import IdentityStore

            tag = f"{reid_cfg['model_name']}|{Path(str(reid_cfg.get('weights_path') or '')).name}"
            store_dir = Path(pers.get("dir", "data/identity_store")) / self.camera_id   # local IDs are per camera
            self.id_store = IdentityStore(str(store_dir), self.clock, model_tag=tag,
                                          retention_sec=float(reid_cfg["identity_retention_sec"]))
            self.id_store.load_into(self.identity, time.monotonic() if self.reader.is_live else 0.0)

        # ---- cross-camera identity (same worker = same person id P# in every camera) --------------------------
        self.linker = None
        self._person_seen: Dict[int, int] = {}
        cc = cfg["cross_camera"]
        if cc.get("enabled"):
            from app.multicam.registry import CameraTopology, CrossCameraLinker

            Path(cc["registry_path"]).parent.mkdir(parents=True, exist_ok=True)
            self.linker = CrossCameraLinker(
                cc["registry_path"], self.camera_id, CameraTopology(cc.get("topology")),
                link_threshold=cc["link_threshold"], margin=cc["margin"], min_obs=cc["min_obs"],
                decide_after_sec=cc["decide_after_sec"], late_merge_sec=cc["late_merge_sec"],
                retention_sec=cc["retention_sec"], sync_interval_sec=cc["sync_interval_sec"],
                stature_min_diff=st["min_diff"], door_min_sim=cc["door_min_sim"],
            )
            logger.info("Cross-camera identity enabled (registry %s)", cc["registry_path"])
            for a_cam, a_gate, b_cam, b_gate, _, _ in self.linker.topology.doors:
                for cam, gate in ((a_cam, a_gate), (b_cam, b_gate)):
                    if cam == self.camera_id and gate.startswith("door:") \
                            and gate[5:] not in self.scene.exit_regions:
                        logger.warning("Cross-camera door: region '%s' is not an exit door of camera %s (draw it "
                                       "with scripts/draw_zone.py as a door) - this door handover cannot work",
                                       gate[5:], cam)
            if not self.reader.is_live and not stream_cfg.get("recording_start_time"):
                logger.warning("Cross-camera identity with video FILES needs stream.recording_start_time for every "
                               "camera, otherwise the clips are not aligned in time.")

        # ---- events, alarms, analytics ------------------------------------------------------------------------
        ev = cfg["events"]
        self.events = EventStore(
            db_path=ev.get("db_path"), csv_dir=ev.get("csv_dir"), snapshot_dir=ev.get("snapshot_dir"),
            clock=self.clock, camera_id=self.camera_id, snapshot_types=ev.get("snapshot_event_types") or [],
            retention_days=float(ev.get("retention_days", 0.0)), label_lookup=self._label_of,
        )
        self.alarms = AlarmDispatcher(cfg["alarms"])
        self.events.subscribe(self.alarms.handle)
        att = cfg["attribution"]
        self.attribution = AttributionBuffer(lambda e: self.events.emit(e), att.get("timeout_sec", 20.0),
                                             att.get("record_unidentified", True))
        zc, cc = cfg["zones"], cfg["crossings"]
        self.zones = ZoneAnalytics(self.scene.zones, zc["entry_confirm_sec"], zc["exit_confirm_sec"],
                                   zc["lost_timeout_sec"])
        self.include_unverified_in_zones = bool(zc.get("include_unverified", True))
        self.crossings = CrossingAnalytics(
            self.scene.lines, self.scene.regions, dead_band_px=cc["dead_band_px"], cooldown_sec=cc["cooldown_sec"],
            max_gap_sec=cc["max_gap_sec"], segment_margin=cc["segment_margin"],
            region_exit_confirm_sec=cc["region_exit_confirm_sec"], region_lost_sec=cc["region_lost_sec"],
            draw_trails=cc.get("draw_trails", True),
        )
        self._provisional: Dict[int, float] = {}   # track ids used as "T<tid>" analytics keys -> last seen

        # ---- forklifts ----------------------------------------------------------------------------------------
        self.vehicles = None
        if vehicle_config_path and Path(vehicle_config_path).exists():
            from app.vehicle_monitoring.pipeline import VehicleMonitoringPipeline

            self.vehicles = VehicleMonitoringPipeline(config_path=vehicle_config_path, device=sys_cfg["device"])
            if not self.vehicles.cfg.get("enabled", True):
                self.vehicles = None
            elif self.projector.H is not None:
                self.vehicles.estimator.set_homography(self.projector.H)   # ONE homography for people & vehicles
            if self.vehicles is not None:
                logger.info("Forklift / driver monitoring enabled (%s)", vehicle_config_path)

        # ---- outputs ------------------------------------------------------------------------------------------
        rec = cfg["recording"]
        self.recorder = None
        out_path = resolve_output_path(rec.get("output_video_path"), cfg["stream"]["source"], self.camera_id)
        if out_path:
            self.recorder = SegmentedVideoWriter(out_path, self.reader.nominal_fps, (self.W, self.H),
                                                 rec.get("segment_minutes", 0.0), rec.get("max_segments", 96))
            logger.info("Annotated video will be saved to %s", out_path)
        web = cfg["web"]
        self.hub = TelemetryHub(web.get("stream_max_fps", 10.0), web.get("stream_width", 1280),
                                jpeg_enabled=bool(web.get("enabled")))
        self.hub.events_provider = lambda limit, sev: self.events.recent(limit, sev)
        if web.get("enabled"):
            start_web_service(self.hub, web.get("host", "127.0.0.1"), web.get("port", 8050), web.get("token", ""))
        self.win_title = f"Factory Monitoring - {self.camera_id}"
        if self.display:
            cv2.namedWindow(self.win_title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.win_title, min(1440, self.W), min(900, self.H))

        self.max_frames = 0   # >0: stop after this many frames (--max-frames, smoke tests / benchmarks)
        self._last_ts: Optional[float] = None
        self.fps = 0.0
        self.frames = 0
        self.errors = 0
        self.last_error = ""
        self.started_wall = time.time()
        self._prev_mono = time.monotonic()
        self._last_publish = 0.0
        self._last_heartbeat = 0.0
        self._stop = False

    # ------------------------------------------------------------------
    def _label_of(self, gid: int) -> Optional[str]:
        label = self.identity.identity_label(gid)
        return self.registry.display_name(label) if (label and self.registry is not None) else label

    def _route(self, ev: Dict[str, Any], frame: Optional[np.ndarray]) -> None:
        """Alarms go out immediately (with a snapshot); other events of provisional tracks wait for the worker ID."""
        if ev.get("severity") == ALARM:
            self.events.emit(ev, frame)
        else:
            self.attribution.submit(ev)

    # ------------------------------------------------------------------
    def process(self, frame: np.ndarray, ts: float) -> np.ndarray:
        self._last_ts = ts
        active_tids = set(self.identity.track_to_global.keys())
        detections = self.detector.track_frame(frame, active_track_ids=active_tids)
        fsm = self.vehicles.fsm if self.vehicles is not None else None
        obstacles = [r.bbox for r in fsm.vehicles.values() if r.is_visible_in_current_frame] if fsm else None
        t2g = self.identity.process_frame(frame, detections, ts, obstacles=obstacles)
        # identity bookkeeping -> analytics state and event attribution
        for old, new in self.identity.pop_merged():
            self.zones.rename(old, new)
            self.crossings.rename(old, new)
            if fsm is not None:
                fsm.rename_person(str(old), str(new))
            self.events.add_alias(old, new, ts)
            if self.linker is not None:
                self.linker.notify_merge(old, new)
            self._person_seen.pop(old, None)
        if self.linker is not None:
            self.linker.observe(self.identity, self.clock, ts)
            for gid, pid in self.linker.links().items():
                if self._person_seen.get(gid) != pid:
                    self._person_seen[gid] = pid
                    self.events.emit(make_event("PERSON_LINKED", ts, worker_id=gid, subject=f"P{pid}", value=pid))
        for gid, label in self.identity.pop_labeled():
            self.events.set_label(gid, label, ts)
            self.events.emit(make_event("WORKER_IDENTIFIED", ts, worker_id=gid, subject=self._label_of(gid),
                                        method="helmet_marker"))
        for det in detections:
            gid = t2g.get(det.track_id)
            if gid is not None and det.track_id in self._provisional:
                del self._provisional[det.track_id]
                self.zones.promote(det.track_id, gid)
                self.crossings.promote(det.track_id, gid)
                if fsm is not None:
                    fsm.rename_person(f"T{det.track_id}", str(gid))
                self.attribution.resolve(det.track_id, gid)
        exited = self.identity.pop_newly_exited()
        for gid in exited:
            for ev in self.zones.close_entity(gid, ts) + self.crossings.close_entity(gid, ts):
                self._route(ev, frame)
            ident = self.identity.identities.get(gid)
            self.events.emit(make_event(
                "WORKER_LEFT_CAMERA", ts, worker_id=gid, subject=None if ident is None else ident.exit_edge,
                reason=None if ident is None else ident.exit_reason))

        # zones / lines / regions
        points_all: Dict[Any, Tuple[float, float]] = {}
        for det in detections:
            gid = t2g.get(det.track_id)
            key = entity_key(gid, det.track_id)
            if gid is None:
                self._provisional[det.track_id] = ts
            points_all[key] = det.ground_uv
        zone_points = points_all if self.include_unverified_in_zones else {
            k: v for k, v in points_all.items() if isinstance(k, int)}
        for ev in self.zones.update(zone_points, ts):
            self._route(ev, frame)
        if self.crossings.enabled:
            for ev in self.crossings.update(points_all, ts):
                self._route(ev, frame)
        self.attribution.tick(ts)
        for tid in [t for t, last in self._provisional.items() if ts - last > 60.0]:
            del self._provisional[tid]

        # forklifts
        draw = self.display or self.recorder is not None or self.hub.jpeg_enabled
        if self.vehicles is not None:
            embs = {d.track_id: self.identity.track_best_emb[d.track_id][0]
                    for d in detections if d.track_id in self.identity.track_best_emb}
            persons = build_persons_for_vehicle_monitor(detections, t2g, embs)
            if self.registry is not None:
                self.vehicles.dashboard.labels = {
                    str(g): self._label_of(g) for g in t2g.values() if self.identity.identity_label(g)}
            frame = self.vehicles.process_frame(frame, external_persons=persons, frame_ts=ts,
                                                left_view_ids=[str(g) for g in exited], draw=draw)
            for ev in self.vehicles.last_events:
                self._route(ev, frame)

        if draw:
            self.zones.draw(frame, ts)
            self.crossings.draw(frame, active_keys=set(points_all))
            self._draw_people(frame, detections, t2g)
            self._draw_banner(frame)
        self._publish(ts, detections, t2g, frame)
        return frame

    # ------------------------------------------------------------------
    def _draw_people(self, frame: np.ndarray, detections: List[PoseDetection], t2g: Dict[int, int]) -> None:
        drivers = self.vehicles.fsm.driver_of if self.vehicles is not None else {}
        for det in detections:
            gid = t2g.get(det.track_id)
            x1, y1, x2, y2 = det.bbox_xyxy.astype(int)
            if gid is None:
                color, tag = (160, 160, 160), f"T:{det.track_id} [INIT]"
            else:
                label = self._label_of(gid)
                color = (0, 140, 255) if det.is_intersecting else (0, 220, 0)
                pid = self._person_seen.get(gid)
                tag = f"ID:{gid}" + (f" P{pid}" if pid is not None else "") + (f" {label}" if label else "")
            key = entity_key(gid, det.track_id)
            zones = self.zones.zones_of(key)
            if zones:
                tag += " @" + ",".join(zones)
            last = self.crossings.last_label.get(key)
            if last:
                tag += f" | {last}"
            vid = drivers.get(str(gid) if gid is not None else f"T{det.track_id}")
            if vid is not None:
                tag += f" | DRIVER F#{vid}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            for p1, p2 in SKELETON_LIMBS:
                if det.keypoints_conf[p1] >= 0.2 and det.keypoints_conf[p2] >= 0.2:
                    cv2.line(frame, tuple(det.keypoints_xy[p1].astype(int)), tuple(det.keypoints_xy[p2].astype(int)),
                             (255, 220, 0), 2, cv2.LINE_AA)
            ug, vg = int(det.ground_uv[0]), int(det.ground_uv[1])
            cv2.circle(frame, (ug, vg), 6, (255, 255, 0) if not det.ground_extrapolated else (0, 165, 255), -1)
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 10)), (x1 + tw + 8, y1), color, -1)
            cv2.putText(frame, tag, (x1 + 4, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255) if gid is None else (0, 0, 0), 2, cv2.LINE_AA)

    def _draw_banner(self, frame: np.ndarray) -> None:
        st = self.identity.stats()
        txt = (f"{self.camera_id} | FPS {self.fps:.1f} | IDs active {st['active']} / known {st['identities']} | "
               f"re-entry thr {self.identity.reentry_threshold():.2f}")
        if self.crossings.enabled:
            totals: Dict[str, int] = {}
            for counts in self.crossings.counts.values():
                for k, v in counts.items():
                    totals[k] = totals.get(k, 0) + v
            txt += " | " + " ".join(f"{k}:{v}" for k, v in totals.items())
        cv2.rectangle(frame, (10, 10), (min(frame.shape[1] - 10, 20 + 11 * len(txt)), 48), (20, 20, 20), -1)
        cv2.putText(frame, txt, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 180), 2, cv2.LINE_AA)

    def _status(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "uptime_sec": round(time.time() - self.started_wall, 1),
            "fps": round(self.fps, 2),
            "frames": self.frames,
            "consecutive_errors": self.errors,
            "last_error": self.last_error,
            "stream": self.reader.status(),
            "identity": self.identity.stats(),
            "cross_camera": None if self.linker is None else {
                "linked_local_ids": len(self._person_seen), "link_threshold": round(self.linker.threshold(), 3)},
            "events": {"written": self.events.n_written, "failed": self.events.n_failed},
            "active_alarms": [
                {"vehicle_id": v.vehicle_id, "type": v.alarm_type, "driver_id": v.associated_driver_id}
                for v in (self.vehicles.fsm.vehicles.values() if self.vehicles is not None else []) if v.alarm_active
            ],
        }

    def _publish(self, ts: float, detections: List[PoseDetection], t2g: Dict[int, int], frame: np.ndarray) -> None:
        now = time.monotonic()
        if now - self._last_publish < 0.5 and not self.hub.jpeg_enabled:
            return
        self._last_publish = now
        persons = []
        for det in detections:
            gid = t2g.get(det.track_id)
            key = entity_key(gid, det.track_id)
            persons.append({"worker_id": gid, "person_id": self._person_seen.get(gid),
                            "label": self._label_of(gid) if gid is not None else None,
                            "track_id": det.track_id, "bbox": [round(float(v), 1) for v in det.bbox_xyxy],
                            "zones": self.zones.zones_of(key)})
        telemetry = {
            "time": self.clock.iso(ts),
            "persons": persons,
            "zones": self.zones.snapshot(ts),
            "zone_totals": {str(k): v for k, v in self.zones.worker_totals(ts).items()},
            "crossings": self.crossings.snapshot(),
            "vehicles": self.vehicles.telemetry(ts) if self.vehicles is not None else [],
        }
        self.hub.publish(self._status(), telemetry, frame)

    def _heartbeat(self) -> None:
        if not self.heartbeat_path or time.monotonic() - self._last_heartbeat < 5.0:
            return
        self._last_heartbeat = time.monotonic()
        try:
            p = Path(self.heartbeat_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._status(), default=str, indent=1), encoding="utf-8")
            os.replace(tmp, p)
        except Exception:
            logger.exception("Heartbeat write failed")

    def _maybe_save_identities(self, force: bool = False) -> None:
        if self.id_store is None:
            return
        if force or time.monotonic() - self._last_id_save >= self.id_save_interval:
            self._last_id_save = time.monotonic()
            try:
                self.id_store.save(self.identity)
            except Exception:
                logger.exception("Identity store save failed")

    # ------------------------------------------------------------------
    def stop(self, *_: Any) -> None:
        self._stop = True

    def run(self) -> None:
        logger.info("Pipeline initialised. Processing...")
        last_wait_log = 0.0
        while not self._stop:
            ok, frame, ts = self.reader.read()
            if not ok or frame is None:
                if self.reader.ended:
                    logger.info("End of stream reached.")
                    break
                stalled = self.reader.status()["seconds_since_last_frame"]
                if time.monotonic() - last_wait_log > 10.0:
                    last_wait_log = time.monotonic()
                    logger.warning("No frame from the camera for %.0f s (reconnecting)", stalled)
                if self.max_stall_sec > 0 and stalled > self.max_stall_sec:
                    raise RuntimeError(f"camera stalled for {stalled:.0f} s")
                self._heartbeat()
                continue

            now = time.monotonic()
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(1e-3, now - self._prev_mono)) if self.frames else 0.0
            self._prev_mono = now
            self.frames += 1
            try:
                frame = self.process(frame, ts)
                self.errors = 0
            except Exception as e:
                self.errors += 1
                self.last_error = f"{type(e).__name__}: {e}"
                logger.exception("Frame %d failed (%d consecutive)", self.frames, self.errors)
                if self.errors >= self.max_errors:
                    raise
                continue

            if self.recorder is not None:
                self.recorder.write(frame, self.clock.wall(ts))
            if self.display:
                cv2.imshow(self.win_title, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("User requested exit ('q').")
                    break
            self._heartbeat()
            self._maybe_save_identities()
            if self.max_frames and self.frames >= self.max_frames:
                logger.info("Stopping after %d frames (--max-frames).", self.frames)
                break

    def _close_open_records(self) -> None:
        """Visits and driver sessions still in progress are recorded up to now (otherwise a stop loses them)."""
        if self._last_ts is None:
            return
        events = self.zones.close_all(self._last_ts)
        if self.vehicles is not None:
            events += self.vehicles.fsm.close_all(self._last_ts)
        for ev in events:
            self._route(ev, None)

    def close(self) -> None:
        for name, fn in (
            ("open visits / sessions", self._close_open_records),
            ("attribution", self.attribution.flush_all),
            ("identity store", lambda: self._maybe_save_identities(force=True)),
            ("identity log", self.identity.close),
            ("cross-camera linker", self.linker.close if self.linker else (lambda: None)),
            ("events", self.events.close),
            ("alarms", self.alarms.close),
            ("recorder", self.recorder.close if self.recorder else (lambda: None)),
            ("reader", self.reader.release),
        ):
            try:
                fn()
            except Exception:
                logger.exception("Shutdown step '%s' failed", name)
        if self.display:
            cv2.destroyAllWindows()


def resolve_run_config(args: argparse.Namespace) -> Tuple[Dict[str, Any], Optional[str], str]:
    """-> (settings, forklift config path or None, calibration path) for single- or multi-camera runs."""
    if args.cameras or args.camera_id:
        from app.multicam.cameras import DEFAULT_CAMERAS_PATH, camera_ids, load_camera_setup

        cameras_path = args.cameras or DEFAULT_CAMERAS_PATH
        camera_id = args.camera_id
        if not camera_id:
            ids = camera_ids(cameras_path)
            if len(ids) != 1:
                raise SystemExit(f"--camera-id is required ({cameras_path} defines {ids}); to run them all use "
                                 "run_multi_camera.py")
            camera_id = ids[0]
        setup = load_camera_setup(cameras_path, camera_id)
        cfg, vehicle_config, calibration = setup.settings, setup.vehicle_config, setup.calibration
    else:
        cfg, vehicle_config, calibration = load_settings(args.config), args.vehicle_config, args.calibration
    if args.source:
        cfg["stream"]["source"] = args.source
    if args.no_gui:
        cfg["system"]["display_gui"] = False
    if args.output:
        cfg["recording"]["output_video_path"] = args.output
    for item in args.set or []:
        apply_override(cfg, item)
    return cfg, vehicle_config, calibration


def apply_override(cfg: Dict[str, Any], item: str) -> None:
    """--set section.key=value (value parsed as YAML), checked against the known settings and their types."""
    import yaml

    from app.core.config import validate

    key, sep, raw = item.partition("=")
    parts = [p for p in key.strip().split(".") if p]
    if not sep or not parts:
        raise SystemExit(f"--set expects section.key=value, got '{item}'")
    nested: Dict[str, Any] = {}
    node = nested
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = yaml.safe_load(raw)
    errors, warnings = validate(nested)
    if errors or warnings:
        raise SystemExit(f"--set {item}: " + "; ".join(errors + warnings))
    node = cfg
    for p in parts[:-1]:
        node = node[p]
    node[parts[-1]] = yaml.safe_load(raw)


def run_once(cfg: Dict[str, Any], vehicle_config: Optional[str], calibration: str, max_frames: int = 0) -> None:
    monitor = FactoryMonitor(cfg, vehicle_config or "", calibration)
    monitor.max_frames = int(max_frames)
    for name in ("SIGTERM", "SIGBREAK"):   # SIGBREAK: graceful stop sent by run_multi_camera.py on Windows
        try:
            signal.signal(getattr(signal, name), monitor.stop)
        except (ValueError, AttributeError, OSError):
            pass
    try:
        monitor.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Shutting down...")
    finally:
        monitor.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Factory monitoring: worker identity, zones, lines, forklift safety")
    parser.add_argument("--config", default="configs/settings.yaml")
    parser.add_argument("--vehicle-config", default="configs/vehicle_monitoring.yaml",
                        help="forklift module config (missing file or enabled:false = module off)")
    parser.add_argument("--source", default=None, help="video file / RTSP URL (overrides the settings)")
    parser.add_argument("--output", default=None,
                        help="where to save the annotated video (default: outputs/<video name>_annotated.mp4)")
    parser.add_argument("--calibration", default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--cameras", default=None,
                        help="multi-camera list (configs/cameras.yaml); use with --camera-id")
    parser.add_argument("--camera-id", default=None, help="which camera of --cameras this process runs")
    parser.add_argument("--no-gui", action="store_true", help="no display window (service mode)")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after N frames (tests / benchmarks)")
    parser.add_argument("--set", action="append", metavar="SECTION.KEY=VALUE",
                        help="override one setting for this run, e.g. --set reid.reentry_sim_thresh=0.85 "
                             "(repeatable)")
    parser.add_argument("--restart-on-failure", action="store_true",
                        help="live cameras: rebuild the pipeline after a fatal error instead of exiting")
    args = parser.parse_args()

    cfg, vehicle_config, calibration = resolve_run_config(args)
    setup_logging(cfg["system"].get("log_dir", "logs"), cfg["system"].get("log_level", "INFO"))
    delay = 10.0
    while True:
        try:
            run_once(cfg, vehicle_config, calibration, args.max_frames)
            return 0
        except KeyboardInterrupt:
            return 0
        except Exception:
            logger.exception("Fatal error in the monitoring pipeline")
            if not args.restart_on_failure:
                return 1
            logger.info("Restarting in %.0f s ...", delay)
            time.sleep(delay)
            delay = min(300.0, delay * 2.0)


if __name__ == "__main__":
    raise SystemExit(main())
