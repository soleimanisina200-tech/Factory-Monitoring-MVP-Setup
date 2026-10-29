# app/logic/state_machine.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import cv2
import numpy as np


@dataclass
class WorkerZoneState:
    global_id: int
    is_inside: bool = False
    entry_accum_sec: float = 0.0
    exit_accum_sec: float = 0.0
    entered_at_ts: float = 0.0
    last_seen_ts: float = 0.0
    total_dwell_sec: float = 0.0


class TimeBasedZoneStateMachine:
    """
    Variable-FPS resilient Zone State Machine.
    Uses monotonic seconds instead of frame counters for entry/exit hysteresis and FORCE_EXIT GC.
    """

    def __init__(
        self,
        zone_id: str,
        polygon_pts: List[List[int]],
        entry_confirm_sec: float = 0.60,
        exit_confirm_sec: float = 1.20,
        force_exit_timeout_sec: float = 15.0,
    ):
        self.zone_id = zone_id
        self.polygon = np.array(polygon_pts, dtype=np.int32)
        self.entry_confirm_sec = entry_confirm_sec
        self.exit_confirm_sec = exit_confirm_sec
        self.force_exit_timeout_sec = force_exit_timeout_sec
        self.states: Dict[int, WorkerZoneState] = {}
        self.prev_frame_ts: float = 0.0

    def update(
        self,
        active_workers_ground_uv: Dict[int, Tuple[float, float]],
        frame_ts: float,
    ) -> List[Dict]:
        events: List[Dict] = []
        dt = max(0.01, min(frame_ts - self.prev_frame_ts, 2.0)) if self.prev_frame_ts > 0 else 0.033
        self.prev_frame_ts = frame_ts

        # 1. Process visible workers using Pose-Extrapolated Ground Contact Point
        for gid, (u_g, v_g) in active_workers_ground_uv.items():
            in_poly = cv2.pointPolygonTest(self.polygon, (float(u_g), float(v_g)), False) >= 0
            st = self.states.setdefault(gid, WorkerZoneState(global_id=gid, last_seen_ts=frame_ts))
            st.last_seen_ts = frame_ts

            if in_poly:
                st.exit_accum_sec = 0.0
                if not st.is_inside:
                    st.entry_accum_sec += dt
                    if st.entry_accum_sec >= self.entry_confirm_sec:
                        st.is_inside = True
                        st.entered_at_ts = frame_ts - st.entry_accum_sec
                        events.append({"event": "ZONE_ENTER", "zone": self.zone_id, "global_id": gid, "ts": frame_ts})
                else:
                    st.total_dwell_sec = frame_ts - st.entered_at_ts
            else:
                st.entry_accum_sec = 0.0
                if st.is_inside:
                    st.exit_accum_sec += dt
                    if st.exit_accum_sec >= self.exit_confirm_sec:
                        st.is_inside = False
                        dwell = frame_ts - st.entered_at_ts
                        events.append(
                            {"event": "ZONE_EXIT", "zone": self.zone_id, "global_id": gid, "dwell_sec": round(dwell, 2), "ts": frame_ts}
                        )

        # 2. FORCE_EXIT Garbage Collector (Time-based, surviving temporary camera drops)
        stale_gids = []
        for gid, st in self.states.items():
            if gid not in active_workers_ground_uv:
                unseen_sec = frame_ts - st.last_seen_ts
                if st.is_inside and unseen_sec >= self.force_exit_timeout_sec:
                    st.is_inside = False
                    dwell = st.last_seen_ts - st.entered_at_ts
                    events.append(
                        {"event": "FORCE_EXIT", "zone": self.zone_id, "global_id": gid, "dwell_sec": round(dwell, 2), "ts": frame_ts}
                    )
                    stale_gids.append(gid)
                elif not st.is_inside and unseen_sec >= self.force_exit_timeout_sec * 2:
                    stale_gids.append(gid)

        for gid in stale_gids:
            del self.states[gid]

        return events