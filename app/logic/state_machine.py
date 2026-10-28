import logging

logger = logging.getLogger(__name__)

class ZoneStateMachine:
    def __init__(self, entry_frames=8, exit_frames=12, timeout_sec=4.0):
        self.entry_frames = entry_frames
        self.exit_frames = exit_frames
        self.timeout_sec = timeout_sec
        self.states = {}

    def update(self, person_id: str, momentary_zone: str, timestamp: float):
        if person_id not in self.states:
            self.states[person_id] = {'status': 'OUTSIDE', 'zone': None, 'buffer': 0, 'entry_time': 0.0, 'last_seen': timestamp}
            
        state = self.states[person_id]
        state['last_seen'] = timestamp
        
        event, event_data, active_dwell = None, None, 0.0
        
        if state['status'] == 'OUTSIDE':
            if momentary_zone:
                state['buffer'] += 1
                if state['buffer'] >= self.entry_frames:
                    state.update({'status': 'INSIDE', 'zone': momentary_zone, 'entry_time': timestamp, 'buffer': 0})
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
                    event_data = {'zone': state['zone'], 'timestamp': timestamp, 'duration': timestamp - state['entry_time']}
                    state.update({'status': 'OUTSIDE', 'zone': None, 'buffer': 0})
                    active_dwell = 0.0
            else:
                state['buffer'] = 0
                
        return event, event_data, state['status'], active_dwell

    def process_missing_workers(self, current_timestamp: float) -> list:
        timeout_events = []
        for person_id, state in list(self.states.items()):
            if (current_timestamp - state['last_seen']) > self.timeout_sec:
                if state['status'] == 'INSIDE':
                    duration = state['last_seen'] - state['entry_time']
                    timeout_events.append({
                        'person_id': person_id, 'event': 'ZONE_EXIT',
                        'event_data': {'zone': state['zone'], 'timestamp': state['last_seen'], 'duration': duration, 'reason': 'FORCE_EXIT'}
                    })
                del self.states[person_id]
        return timeout_events