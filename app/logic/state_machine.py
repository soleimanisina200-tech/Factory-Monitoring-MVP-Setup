import logging

logger = logging.getLogger(__name__)

class ZoneStateMachine:
    def __init__(self, entry_frames: int = 10, exit_frames: int = 15):
        self.entry_frames_threshold = entry_frames
        self.exit_frames_threshold = exit_frames
        self.tracks_state = {}

    def update(self, track_id: int, momentary_zone: str, current_timestamp: float):
        if track_id not in self.tracks_state:
            self.tracks_state[track_id] = {
                'state': 'OUTSIDE',
                'current_zone': None,
                'pending_zone': momentary_zone,
                'counter': 0,
                'entry_time': None
            }
            
        track_data = self.tracks_state[track_id]
        current_state = track_data['state']
        event = None
        event_data = None
        active_dwell_time = None

        if current_state == 'OUTSIDE':
            if momentary_zone != "OUTSIDE":
                if track_data['pending_zone'] == momentary_zone:
                    track_data['counter'] += 1
                    if track_data['counter'] >= self.entry_frames_threshold:
                        track_data['state'] = 'INSIDE'
                        track_data['current_zone'] = momentary_zone
                        track_data['counter'] = 0
                        track_data['entry_time'] = current_timestamp
                        
                        event = "ZONE_ENTER"
                        event_data = {'zone': momentary_zone, 'timestamp': current_timestamp}
                        logger.info(f"Event: Track {track_id} ENTERED {momentary_zone}")
                else:
                    track_data['pending_zone'] = momentary_zone
                    track_data['counter'] = 1
            else:
                track_data['pending_zone'] = "OUTSIDE"
                track_data['counter'] = 0

        elif current_state == 'INSIDE':
            inside_zone_name = track_data['current_zone']
            
            if track_data['entry_time'] is not None:
                active_dwell_time = current_timestamp - track_data['entry_time']
            
            if momentary_zone != inside_zone_name:
                if track_data['pending_zone'] == momentary_zone:
                    track_data['counter'] += 1
                    if track_data['counter'] >= self.exit_frames_threshold:
                        final_duration = current_timestamp - track_data['entry_time']
                        
                        track_data['state'] = 'OUTSIDE'
                        track_data['current_zone'] = None
                        track_data['counter'] = 0
                        track_data['entry_time'] = None
                        
                        event = "ZONE_EXIT"
                        event_data = {
                            'zone': inside_zone_name, 
                            'timestamp': current_timestamp,
                            'duration': final_duration
                        }
                        logger.info(f"Event: Track {track_id} EXITED {inside_zone_name} | Dwell Time: {final_duration:.2f}s")
                else:
                    track_data['pending_zone'] = momentary_zone
                    track_data['counter'] = 1
            else:
                track_data['pending_zone'] = inside_zone_name
                track_data['counter'] = 0

        return event, event_data, track_data['state'], active_dwell_time