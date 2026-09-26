import yaml
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

class ZoneManager:
    def __init__(self, config_path: str):
        logger.info(f"Loading zones from {config_path}...")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        self.zones = config.get('zones', [])
        
        # Convert polygon lists to numpy arrays for OpenCV processing
        for zone in self.zones:
            zone['polygon'] = np.array(zone['polygon'], np.int32)
            
        logger.info(f"Successfully loaded {len(self.zones)} zones.")

    def draw_zones(self, frame):
        # Draw each zone on the frame
        for zone in self.zones:
            # Reshape for OpenCV polylines function requirement
            pts = zone['polygon'].reshape((-1, 1, 2))
            
            # Draw the polygon boundaries
            cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
            
            # Put the zone name near the first point of the polygon
            x, y = pts[0][0]
            cv2.putText(frame, zone['name'], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)