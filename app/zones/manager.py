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
        
        for zone in self.zones:
            zone['polygon'] = np.array(zone['polygon'], np.int32)
            
        logger.info(f"Successfully loaded {len(self.zones)} zones.")

    def draw_zones(self, frame):
        for zone in self.zones:
            pts = zone['polygon'].reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
            
            x, y = pts[0][0]
            cv2.putText(frame, zone['name'], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    def get_zone_for_point(self, point: tuple) -> str:
        # Check if the given (x, y) point is inside any defined zone
        for zone in self.zones:
            # cv2.pointPolygonTest returns:
            # +1 if point is inside
            #  0 if point is exactly on the boundary
            # -1 if point is outside
            dist = cv2.pointPolygonTest(zone['polygon'], point, measureDist=False)
            
            # If inside or exactly on the boundary, we consider it assigned to this zone
            if dist >= 0:
                return zone['name']
                
        return "OUTSIDE"