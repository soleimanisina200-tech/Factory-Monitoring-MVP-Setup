# app/detection/detector.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple
import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class PoseDetection:
    track_id: int
    bbox_xyxy: np.ndarray          # shape (4,) float32 [x1, y1, x2, y2]
    conf: float
    keypoints_xy: np.ndarray       # shape (17, 2) float32
    keypoints_conf: np.ndarray     # shape (17,) float32
    is_intersecting: bool = False
    is_isolated: bool = True
    ground_uv: Tuple[float, float] = (0.0, 0.0)
    bev_xy: Optional[np.ndarray] = None
    apparent_height_px: float = 100.0


class IndustrialPoseDetector:
    """
    Single-Pass YOLO11-Pose + ByteTrack wrapper with Track-Preserving Anatomical NMS.
    Never allows a 1-frame ghost detection to suppress an established active tracklet,
    and never deletes real workers crossing paths unless they share identical skeleton joints.
    """

    def __init__(
        self,
        model_weights: str = "yolo11m-pose.pt",
        tracker_config: str = "configs/custom_bytetrack.yaml",
        device: str = "cuda:0",
        conf_thres: float = 0.18,
        iou_thres: float = 0.48,
        imgsz: int = 1280,
        half: bool = True,
    ):
        self.device = device
        self.tracker_config = tracker_config
        self.conf_thres = min(float(conf_thres), 0.22)
        self.iou_thres = float(iou_thres)
        self.imgsz = max(int(imgsz), 1280)

        self.model = YOLO(model_weights)
        self.model.to(self.device)

        if half and "cuda" in str(self.device) and torch.cuda.is_available():
            self.model.model.half()

    @staticmethod
    def _filter_duplicate_and_ghost_detections(
        detections: List[PoseDetection],
        active_track_ids: Optional[Set[int]] = None,
        kpt_thresh: float = 0.35,
    ) -> List[PoseDetection]:
        if len(detections) <= 1:
            return detections

        active_set = active_track_ids or set()

        # CRITICAL: Prioritize established active tracks FIRST, then confidence.
        # Prevents a 1-frame ghost box from suppressing a 500-frame established track!
        sorted_dets = sorted(
            detections,
            key=lambda d: (1 if d.track_id in active_set else 0, d.conf),
            reverse=True,
        )
        keep: List[PoseDetection] = []

        for cand in sorted_dets:
            x1, y1, x2, y2 = cand.bbox_xyxy
            bw, bh = float(x2 - x1), float(y2 - y1)
            area_c = max(1.0, bw * bh)

            # Allow lower confidence for established tracks or distant small workers
            if cand.track_id not in active_set:
                min_req_conf = float(np.clip(0.18 + 0.08 * ((bh - 75.0) / 180.0), 0.18, 0.26))
                if cand.conf < min_req_conf:
                    continue

            is_duplicate = False
            for kept in keep:
                kx1, ky1, kx2, ky2 = kept.bbox_xyxy
                area_k = max(1.0, float((kx2 - kx1) * (ky2 - ky1)))

                ix1, iy1 = max(x1, kx1), max(y1, ky1)
                ix2, iy2 = min(x2, kx2), min(y2, ky2)
                inter = max(0.0, float(ix2 - ix1)) * max(0.0, float(iy2 - iy1))

                iou = inter / (area_c + area_k - inter + 1e-6)
                iomin = inter / min(area_c, area_k)

                # 1. Very high full-box IoU (> 0.62) is always a duplicate box on the same person
                if iou >= 0.62:
                    is_duplicate = True
                    break

                # 2. For nested boxes (iomin >= 0.72) or moderate overlap (iou >= 0.25),
                # verify whether they share the same physical skeleton before suppressing!
                # This ensures two distinct workers crossing paths are NEVER deleted.
                if iomin >= 0.72 or iou >= 0.25:
                    both_vis = (cand.keypoints_conf >= kpt_thresh) & (kept.keypoints_conf >= kpt_thresh)
                    shared_count = int(np.sum(both_vis))
                    if shared_count >= 3:
                        dists = np.linalg.norm(
                            cand.keypoints_xy[both_vis] - kept.keypoints_xy[both_vis], axis=1
                        )
                        if float(np.median(dists)) < 0.16 * min(bh, kept.apparent_height_px):
                            is_duplicate = True
                            break
                    elif iomin >= 0.85 and cand.track_id not in active_set:
                        # Un-tracked sub-box without skeleton completely inside an active person box
                        is_duplicate = True
                        break

            if not is_duplicate:
                keep.append(cand)

        return keep

    def track_frame(
        self, frame: np.ndarray, active_track_ids: Optional[Set[int]] = None
    ) -> List[PoseDetection]:
        h_img, w_img = frame.shape[:2]
        effective_imgsz = max(self.imgsz, 1280) if w_img >= 1920 else self.imgsz

        results = self.model.track(
            source=frame,
            tracker=self.tracker_config,
            persist=True,
            classes=[0],
            conf=self.conf_thres,
            iou=self.iou_thres,
            imgsz=effective_imgsz,
            device=self.device,
            max_det=80,
            verbose=False,
        )

        if not results or results[0].boxes is None or results[0].boxes.id is None:
            return []

        res = results[0]
        boxes_xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float32)
        track_ids = res.boxes.id.int().cpu().numpy()
        confs = res.boxes.conf.cpu().numpy().astype(np.float32)

        if res.keypoints is not None and res.keypoints.xy is not None:
            kpts_xy = res.keypoints.xy.cpu().numpy().astype(np.float32)
            kpts_conf = (
                res.keypoints.conf.cpu().numpy().astype(np.float32)
                if res.keypoints.conf is not None
                else np.zeros((len(boxes_xyxy), 17), dtype=np.float32)
            )
        else:
            kpts_xy = np.zeros((len(boxes_xyxy), 17, 2), dtype=np.float32)
            kpts_conf = np.zeros((len(boxes_xyxy), 17), dtype=np.float32)

        raw_detections: List[PoseDetection] = []
        for i in range(len(boxes_xyxy)):
            x1, y1, x2, y2 = boxes_xyxy[i]
            bh = max(16.0, float(y2 - y1))
            raw_detections.append(
                PoseDetection(
                    track_id=int(track_ids[i]),
                    bbox_xyxy=boxes_xyxy[i],
                    conf=float(confs[i]),
                    keypoints_xy=kpts_xy[i],
                    keypoints_conf=kpts_conf[i],
                    apparent_height_px=bh,
                )
            )

        return self._filter_duplicate_and_ghost_detections(raw_detections, active_track_ids)