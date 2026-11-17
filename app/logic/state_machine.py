import logging

logger = logging.getLogger(__name__)


class ZoneStateMachine:
    """
    ماشین وضعیت زمان‌محور و فریم‌محور بدون خطای بایاس زمانی (Zero-Bias Dwell Time)
    با پشتیبانی از انتقال مستقیم بین نواحی همسایه و مکانیزم FORCE_EXIT.
    """

    def __init__(
        self,
        entry_frames: int = 8,
        exit_frames: int = 12,
        timeout_sec: float = 4.0,
        entry_confirm_sec: float = 0.30,
        exit_confirm_sec: float = 0.45,
        stale_cleanup_sec: float = 300.0,
    ):
        self.entry_frames = entry_frames
        self.exit_frames = exit_frames
        self.timeout_sec = timeout_sec
        self.entry_confirm_sec = entry_confirm_sec
        self.exit_confirm_sec = exit_confirm_sec
        self.stale_cleanup_sec = stale_cleanup_sec
        self.states = {}

    def _init_person_state(self, timestamp: float) -> dict:
        return {
            "status": "OUTSIDE",
            "zone": None,
            "entry_time": 0.0,
            "last_seen": timestamp,
            "last_seen_inside": timestamp,
            # بافرهای کاندیدای ورود و خروج بدون خطای بایاس
            "candidate_zone": None,
            "candidate_frames": 0,
            "candidate_start_ts": None,
            "exit_buffer_frames": 0,
            "exit_start_ts": None,
            "pending_events": [],
        }

    def update(self, person_id: str, momentary_zone: str | None, timestamp: float):
        """
        به‌روزرسانی وضعیت حضور کارگر در نواحی.
        خروجی: (event, event_data, status, active_dwell)
        """
        if person_id not in self.states:
            self.states[person_id] = self._init_person_state(timestamp)

        state = self.states[person_id]
        state["last_seen"] = timestamp

        event = None
        event_data = None
        active_dwell = 0.0

        # اگر از فریم قبلی رویداد انتقالی (مثلاً ورود بلافاصله پس از خروج از زون همسایه) در صف است
        if state["pending_events"]:
            event, event_data = state["pending_events"].pop(0)

        if state["status"] == "OUTSIDE":
            if momentary_zone is not None:
                if state["candidate_zone"] != momentary_zone:
                    state["candidate_zone"] = momentary_zone
                    state["candidate_frames"] = 1
                    state["candidate_start_ts"] = timestamp
                else:
                    state["candidate_frames"] += 1

                elapsed_entry = timestamp - (state["candidate_start_ts"] or timestamp)
                if (
                    state["candidate_frames"] >= self.entry_frames
                    or elapsed_entry >= self.entry_confirm_sec
                ):
                    true_entry_ts = state["candidate_start_ts"] or timestamp
                    state["status"] = "INSIDE"
                    state["zone"] = momentary_zone
                    state["entry_time"] = true_entry_ts
                    state["last_seen_inside"] = timestamp
                    state["candidate_zone"] = None
                    state["candidate_frames"] = 0
                    state["candidate_start_ts"] = None
                    state["exit_buffer_frames"] = 0
                    state["exit_start_ts"] = None

                    active_dwell = max(0.0, timestamp - true_entry_ts)
                    event = "ZONE_ENTER"
                    event_data = {
                        "person_id": person_id,
                        "zone": momentary_zone,
                        "timestamp": true_entry_ts,
                        "duration": 0.0,
                    }
            else:
                state["candidate_zone"] = None
                state["candidate_frames"] = 0
                state["candidate_start_ts"] = None

        elif state["status"] == "INSIDE":
            current_zone = state["zone"]

            if momentary_zone == current_zone:
                # کارگر همچنان داخل همان زون است
                state["last_seen_inside"] = timestamp
                state["exit_buffer_frames"] = 0
                state["exit_start_ts"] = None
                state["candidate_zone"] = None
                state["candidate_frames"] = 0
                state["candidate_start_ts"] = None
                active_dwell = max(0.0, timestamp - state["entry_time"])
            else:
                # کارگر خارج از زون فعلی دیده شده (یا بیرون رفته یا وارد زون همسایه شده است)
                if state["exit_buffer_frames"] == 0:
                    state["exit_start_ts"] = timestamp
                state["exit_buffer_frames"] += 1

                # اگر مستقیماً وارد یک زون همسایه (momentary_zone != None) شده، همزمان شمارش ورود به زون جدید را انجام بده
                if momentary_zone is not None:
                    if state["candidate_zone"] != momentary_zone:
                        state["candidate_zone"] = momentary_zone
                        state["candidate_frames"] = 1
                        state["candidate_start_ts"] = timestamp
                    else:
                        state["candidate_frames"] += 1
                else:
                    state["candidate_zone"] = None
                    state["candidate_frames"] = 0
                    state["candidate_start_ts"] = None

                elapsed_exit = timestamp - (state["exit_start_ts"] or timestamp)
                active_dwell = max(0.0, timestamp - state["entry_time"])

                if (
                    state["exit_buffer_frames"] >= self.exit_frames
                    or elapsed_exit >= self.exit_confirm_sec
                ):
                    # زمان واقعی خروج همان لحظه اول خروج از مرز زون است (رفع خطای بایاس duration)
                    true_exit_ts = state["exit_start_ts"] or timestamp
                    duration = max(0.0, true_exit_ts - state["entry_time"])

                    event = "ZONE_EXIT"
                    event_data = {
                        "person_id": person_id,
                        "zone": current_zone,
                        "timestamp": true_exit_ts,
                        "duration": duration,
                    }

                    # بررسی اینکه آیا همزمان ورود به زون همسایه نیز تایید شده است؟
                    next_zone = state["candidate_zone"]
                    next_frames = state["candidate_frames"]
                    next_start_ts = state["candidate_start_ts"]
                    elapsed_next = (
                        (timestamp - next_start_ts) if next_start_ts is not None else 0.0
                    )

                    if next_zone is not None and (
                        next_frames >= self.entry_frames
                        or elapsed_next >= self.entry_confirm_sec
                    ):
                        state["status"] = "INSIDE"
                        state["zone"] = next_zone
                        state["entry_time"] = next_start_ts or true_exit_ts
                        state["last_seen_inside"] = timestamp
                        state["exit_buffer_frames"] = 0
                        state["exit_start_ts"] = None
                        state["candidate_zone"] = None
                        state["candidate_frames"] = 0
                        state["candidate_start_ts"] = None
                        active_dwell = max(0.0, timestamp - state["entry_time"])

                        state["pending_events"].append(
                            (
                                "ZONE_ENTER",
                                {
                                    "person_id": person_id,
                                    "zone": next_zone,
                                    "timestamp": state["entry_time"],
                                    "duration": 0.0,
                                },
                            )
                        )
                    else:
                        state["status"] = "OUTSIDE"
                        state["zone"] = None
                        state["entry_time"] = 0.0
                        state["exit_buffer_frames"] = 0
                        state["exit_start_ts"] = None
                        active_dwell = 0.0

        return event, event_data, state["status"], active_dwell

    def process_missing_workers(self, current_timestamp: float) -> list[dict]:
        """
        بررسی کارگرانی که در وضعیت INSIDE بوده‌اند اما بیش از timeout_sec دیده نشده‌اند (FORCE_EXIT)
        و پاکسازی وضعیت‌های قدیمی برای جلوگیری از نشت حافظه.
        """
        forced_events = []
        to_delete = []

        for person_id, state in list(self.states.items()):
            time_since_seen = current_timestamp - state["last_seen"]

            if state["status"] == "INSIDE" and time_since_seen >= self.timeout_sec:
                # زمان واقعی خروج آخرین لحظه‌ای است که فرد واقعاً در تصویر/زون دیده شده است
                true_exit_ts = state.get("last_seen_inside", state["last_seen"])
                duration = max(0.0, true_exit_ts - state["entry_time"])
                zone_name = state["zone"]

                forced_events.append(
                    {
                        "event": "FORCE_EXIT",
                        "person_id": person_id,
                        "zone": zone_name,
                        "timestamp": true_exit_ts,
                        "duration": duration,
                    }
                )
                logger.info(
                    f"[FORCE_EXIT] {person_id} missing for {time_since_seen:.1f}s -> closed session in {zone_name} (duration={duration:.1f}s)"
                )
                state["status"] = "OUTSIDE"
                state["zone"] = None
                state["entry_time"] = 0.0
                state["exit_buffer_frames"] = 0
                state["exit_start_ts"] = None

            elif state["status"] == "OUTSIDE" and time_since_seen > self.stale_cleanup_sec:
                to_delete.append(person_id)

        for pid in to_delete:
            del self.states[pid]

        return forced_events