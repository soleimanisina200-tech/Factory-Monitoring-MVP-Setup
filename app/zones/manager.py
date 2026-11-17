# app/zones/manager.py
from __future__ import annotations
from typing import Dict, List, Tuple
import cv2
import numpy as np
from app.logic.state_machine import TimeBasedZoneStateMachine


class MultiZoneManager:
    """
    Orchestrates multiple TimeBasedZoneStateMachines across the camera view.
    """

    def __init__(
        self,
        zone_definitions: Dict[str, Dict],
        entry_confirm_sec: float = 0.60,
        exit_confirm_sec: float = 1.20,
        force_exit_timeout_sec: float = 15.0,
    ):
        self.zones: Dict[str, TimeBasedZoneStateMachine] = {}
        for zone_id, cfg in zone_definitions.items():
            self.zones[zone_id] = TimeBasedZoneStateMachine(
                zone_id=zone_id,
                polygon_pts=cfg["polygon"],
                entry_confirm_sec=entry_confirm_sec,
                exit_confirm_sec=exit_confirm_sec,
                force_exit_timeout_sec=force_exit_timeout_sec,
            )

    def update_all(
        self,
        active_workers_ground_uv: Dict[int, Tuple[float, float]],
        frame_ts: float,
    ) -> List[Dict]:
        all_events: List[Dict] = []
        for zm in self.zones.values():
            events = zm.update(active_workers_ground_uv, frame_ts)
            all_events.extend(events)
        return all_events

    def rename_identity(self, old_gid: int, new_gid: int) -> None:
        """Identity `old_gid` was merged into `new_gid`: keep one zone state, under the surviving ID."""
        for zm in self.zones.values():
            st = zm.states.pop(old_gid, None)
            if st is not None and new_gid not in zm.states:
                st.global_id = new_gid
                zm.states[new_gid] = st

    def draw_zones(self, frame: np.ndarray) -> None:
        if not self.zones:
            return  # nothing to draw: skip the (full-frame) copy + blend entirely
        h, w = frame.shape[:2]
        all_pts = np.vstack([zm.polygon for zm in self.zones.values()])
        x1, y1 = (int(max(0, v)) for v in all_pts.min(axis=0))
        x2, y2 = int(min(w, all_pts[:, 0].max() + 1)), int(min(h, all_pts[:, 1].max() + 1))
        if x2 > x1 and y2 > y1:
            # Blend only the bounding box of the zones instead of the whole frame.
            roi = frame[y1:y2, x1:x2]
            overlay = roi.copy()
            for zm in self.zones.values():
                occupied = any(st.is_inside for st in zm.states.values())
                color = (0, 180, 0) if occupied else (200, 160, 0)
                cv2.fillPoly(overlay, [zm.polygon - np.array([x1, y1], dtype=np.int32)], color)
            cv2.addWeighted(overlay, 0.18, roi, 0.82, 0, roi)

        for zone_id, zm in self.zones.items():
            occupants = [gid for gid, st in zm.states.items() if st.is_inside]
            color = (0, 180, 0) if occupants else (200, 160, 0)
            cv2.polylines(frame, [zm.polygon], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

            # Label zone header with active Global IDs and dwell time
            top_pt = zm.polygon[np.argmin(zm.polygon[:, 1])]
            label = f"{zone_id} | Occupants: {occupants}"
            cv2.putText(
                frame,
                label,
                (int(top_pt[0]), max(25, int(top_pt[1]) - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
