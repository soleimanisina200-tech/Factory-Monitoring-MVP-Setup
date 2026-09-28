import logging

logger = logging.getLogger(__name__)

class ZoneStateMachine:
    def __init__(self, entry_frames: int = 10, exit_frames: int = 15):
        self.entry_frames_threshold = entry_frames
        self.exit_frames_threshold = exit_frames
        
        # Structure: 
        # { 
        #   track_id: {
        #       'state': 'OUTSIDE' | 'INSIDE',
        #       'current_zone': None or str (if INSIDE),
        #       'pending_zone': str (the zone they are transitioning to/from),
        #       'counter': int (counts consecutive frames in pending_zone)
        #   } 
        # }
        self.tracks_state = {}

    def update(self, track_id: int, momentary_zone: str):
        """
        momentary_zone is the zone detected in the CURRENT frame (can be "OUTSIDE" or a zone name).
        """
        if track_id not in self.tracks_state:
            self.tracks_state[track_id] = {
                'state': 'OUTSIDE',
                'current_zone': None,
                'pending_zone': momentary_zone,
                'counter': 0
            }
            
        track_data = self.tracks_state[track_id]
        current_state = track_data['state']
        event = None

        # ---------------------------------------------------------
        # Scenario 1: Object is currently OUTSIDE
        # ---------------------------------------------------------
        if current_state == 'OUTSIDE':
            # If they are seen inside a specific machine zone
            if momentary_zone != "OUTSIDE":
                # Are they accumulating frames for this specific machine?
                if track_data['pending_zone'] == momentary_zone:
                    track_data['counter'] += 1
                    
                    # Did they hit the threshold to officially ENTER?
                    if track_data['counter'] >= self.entry_frames_threshold:
                        track_data['state'] = 'INSIDE'
                        track_data['current_zone'] = momentary_zone
                        track_data['counter'] = 0
                        event = "ZONE_ENTER"
                        logger.info(f"Event: Track {track_id} CONFIRMED ENTER {momentary_zone}")
                else:
                    # They just stepped into a new/different zone, start counting from 1
                    track_data['pending_zone'] = momentary_zone
                    track_data['counter'] = 1
            else:
                # Still OUTSIDE, reset everything
                track_data['pending_zone'] = "OUTSIDE"
                track_data['counter'] = 0

        # ---------------------------------------------------------
        # Scenario 2: Object is currently INSIDE a zone
        # ---------------------------------------------------------
        elif current_state == 'INSIDE':
            inside_zone_name = track_data['current_zone']
            
            # If they are no longer in the zone they officially entered
            if momentary_zone != inside_zone_name:
                # Are they accumulating frames to EXIT?
                if track_data['pending_zone'] == momentary_zone:
                    track_data['counter'] += 1
                    
                    # Did they hit the threshold to officially EXIT?
                    if track_data['counter'] >= self.exit_frames_threshold:
                        track_data['state'] = 'OUTSIDE'
                        track_data['current_zone'] = None
                        track_data['counter'] = 0
                        event = "ZONE_EXIT"
                        logger.info(f"Event: Track {track_id} CONFIRMED EXIT {inside_zone_name}")
                        
                        # Note: If momentary_zone is another machine, they will start 
                        # entering it on the next frame (Scenario 1)
                else:
                    # They just stepped out of their zone, start counting from 1
                    track_data['pending_zone'] = momentary_zone
                    track_data['counter'] = 1
            else:
                # They are safely inside their zone, reset the exit counter
                track_data['pending_zone'] = inside_zone_name
                track_data['counter'] = 0

        return event, track_data['state']