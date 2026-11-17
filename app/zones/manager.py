import cv2
import numpy as np


class ZoneManager:
    """
    مدیریت نواحی نظارتی کارخانه (Polygon Zones) و بررسی حضور نقطه تماس پای کارگر در هر ناحیه.
    """

    def __init__(self, zones_config: dict):
        self.zones = {}
        self.colors = {}

        palette = [
            (0, 200, 255),
            (255, 180, 0),
            (50, 220, 100),
            (255, 100, 180),
            (180, 120, 255),
        ]

        if zones_config:
            for idx, (zone_name, pts) in enumerate(zones_config.items()):
                if pts and len(pts) >= 3:
                    self.zones[zone_name] = np.array(pts, dtype=np.int32)
                    self.colors[zone_name] = palette[idx % len(palette)]

    def get_zone(self, foot_point: tuple[int, int]) -> str | None:
        """بررسی اینکه نقطه پای کارگر داخل کدام ناحیه قرار دارد."""
        if foot_point is None:
            return None

        pt = (float(foot_point[0]), float(foot_point[1]))
        for zone_name, polygon in self.zones.items():
            if cv2.pointPolygonTest(polygon, pt, False) >= 0:
                return zone_name
        return None

    def check_point(self, foot_point: tuple[int, int]) -> str | None:
        """نام مستعار سازگار با نسخه‌های قبلی."""
        return self.get_zone(foot_point)

    def draw_zones(self, frame: np.ndarray, zone_occupants: dict[str, int] | None = None) -> np.ndarray:
        """رسم نیمه‌شفاف نواحی و تعداد افراد حاضر در هر ناحیه روی تصویر."""
        if not self.zones:
            return frame

        overlay = frame.copy()
        for zone_name, polygon in self.zones.items():
            color = self.colors.get(zone_name, (0, 200, 255))
            cv2.fillPoly(overlay, [polygon], color)

        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

        for zone_name, polygon in self.zones.items():
            color = self.colors.get(zone_name, (0, 200, 255))
            cv2.polylines(frame, [polygon], isClosed=True, color=color, thickness=2)

            top_pt = polygon[np.argmin(polygon[:, 1])]
            count = zone_occupants.get(zone_name, 0) if zone_occupants else 0
            label = f"{zone_name} [Active: {count}]"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            bx = max(5, int(top_pt[0]))
            by = max(th + 10, int(top_pt[1]) - 8)
            cv2.rectangle(frame, (bx - 4, by - th - 6), (bx + tw + 6, by + 6), (25, 25, 25), -1)
            cv2.putText(
                frame,
                label,
                (bx, by),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

        return frame