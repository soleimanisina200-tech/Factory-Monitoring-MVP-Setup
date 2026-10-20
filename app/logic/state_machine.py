import logging

logger = logging.getLogger(__name__)

class ZoneStateMachine:
    # timeout_sec = 4.0 یعنی اگر کارگر 4 ثانیه دیده نشد، خروج اجباری صادر شود
    def __init__(self, entry_frames=8, exit_frames=12, timeout_sec=4.0):
        self.entry_frames = entry_frames
        self.exit_frames = exit_frames
        self.timeout_sec = timeout_sec
        
        # ساختار حافظه: وضعیت حضور، نام زون، زمان ورود، و آخرین لحظه رویت در دوربین
        self.states = {}

    def update(self, person_id: str, momentary_zone: str, timestamp: float):
        if person_id not in self.states:
            self.states[person_id] = {
                'status': 'OUTSIDE',
                'zone': None,
                'buffer': 0,
                'entry_time': 0.0,
                'last_seen': timestamp
            }
            
        state = self.states[person_id]
        state['last_seen'] = timestamp  # آپدیت مداوم آخرین لحظه‌ای که ردیاب او را دیده است
        
        event = None
        event_data = None
        active_dwell = 0.0
        
        if state['status'] == 'OUTSIDE':
            if momentary_zone:
                state['buffer'] += 1
                if state['buffer'] >= self.entry_frames:
                    state['status'] = 'INSIDE'
                    state['zone'] = momentary_zone
                    state['entry_time'] = timestamp
                    state['buffer'] = 0
                    event = "ZONE_ENTER"
                    event_data = {'zone': momentary_zone, 'timestamp': timestamp}
            else:
                state['buffer'] = 0
                
        elif state['status'] == 'INSIDE':
            active_dwell = timestamp - state['entry_time']
            if momentary_zone != state['zone']:
                state['buffer'] += 1
                if state['buffer'] >= self.exit_frames:
                    event = "ZONE_EXIT"
                    event_data = {
                        'zone': state['zone'], 
                        'timestamp': timestamp,
                        'duration': timestamp - state['entry_time']
                    }
                    state['status'] = 'OUTSIDE'
                    state['zone'] = None
                    state['buffer'] = 0
                    active_dwell = 0.0
            else:
                state['buffer'] = 0
                
        return event, event_data, state['status'], active_dwell

    def process_missing_workers(self, current_timestamp: float) -> list:
        """
        رفتگر زمان‌سنج: بررسی کارگرانی که غیب شده‌اند (خروج کامل از کادر یا پنهان شدن مطلق)
        """
        timeout_events = []
        # تبدیل به لیست برای جلوگیری از خطای تغییر سایز دیکشنری در حین حلقه
        for person_id, state in list(self.states.items()):
            time_since_last_seen = current_timestamp - state['last_seen']
            
            # اگر زمان غیبت از آستانه مجاز بیشتر شد
            if time_since_last_seen > self.timeout_sec:
                if state['status'] == 'INSIDE':
                    duration = state['last_seen'] - state['entry_time']
                    timeout_events.append({
                        'person_id': person_id,
                        'event': 'ZONE_EXIT',
                        'event_data': {
                            'zone': state['zone'],
                            'timestamp': state['last_seen'],
                            'duration': duration,
                            'reason': 'FORCE_EXIT'
                        }
                    })
                    logger.info(f"FORCE_EXIT applied for {person_id} from {state['zone']} (Missing for {time_since_last_seen:.1f}s)")
                
                # پاک‌سازی کامل از حافظه رم
                del self.states[person_id]
                
        return timeout_events