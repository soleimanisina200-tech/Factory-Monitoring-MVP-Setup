import cv2
import numpy as np

# ایندکس‌های استاندارد ۱۷ مفصل COCO Keypoints
KP_NOSE = 0
KP_L_EYE = 1
KP_R_EYE = 2
KP_L_EAR = 3
KP_R_EAR = 4
KP_L_SHOULDER = 5
KP_R_SHOULDER = 6
KP_L_ELBOW = 7
KP_R_ELBOW = 8
KP_L_WRIST = 9
KP_R_WRIST = 10
KP_L_HIP = 11
KP_R_HIP = 12
KP_L_KNEE = 13
KP_R_KNEE = 14
KP_L_ANKLE = 15
KP_R_ANKLE = 16

# اتصالات استخوان‌های اسکلت انسان برای رسم گرافیکی (مشابه نسخه قبلی)
SKELETON_CONNECTIONS = [
    (5, 6),   # شانه به شانه
    (5, 7),   # شانه چپ به آرنج چپ
    (7, 9),   # آرنج چپ به مچ دست چپ
    (6, 8),   # شانه راست به آرنج راست
    (8, 10),  # آرنج راست به مچ دست راست
    (5, 11),  # شانه چپ به لگن چپ
    (6, 12),  # شانه راست به لگن راست
    (11, 12), # لگن به لگن
    (11, 13), # لگن چپ به زانو چپ
    (13, 15), # زانو چپ به مچ پا چپ
    (12, 14), # لگن راست به زانو راست
    (14, 16), # زانو راست به مچ پا راست
]


class HomographyTransformer:
    def __init__(self, src_points=None, dst_points=None):
        self.matrix = None
        if src_points and dst_points and len(src_points) >= 4 and len(dst_points) >= 4:
            src = np.array(src_points, dtype=np.float32)
            dst = np.array(dst_points, dtype=np.float32)
            self.matrix, _ = cv2.findHomography(src, dst)

    def transform_point(self, point: tuple[int, int]) -> tuple[float, float]:
        if self.matrix is None:
            return float(point[0]), float(point[1])
        pts = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32)
        projected = cv2.perspectiveTransform(pts, self.matrix)
        return float(projected[0, 0, 0]), float(projected[0, 0, 1])


def calculate_ioa(target_box: list[int], other_box: list[int]) -> float:
    xA = max(target_box[0], other_box[0])
    yA = max(target_box[1], other_box[1])
    xB = min(target_box[2], other_box[2])
    yB = min(target_box[3], other_box[3])
    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight
    if interArea == 0:
        return 0.0
    targetArea = max(1, (target_box[2] - target_box[0]) * (target_box[3] - target_box[1]))
    return interArea / float(targetArea)


def is_touching_border(box: list[int], frame_w: int, frame_h: int, margin: int = 28) -> bool:
    x1, y1, x2, y2 = box
    return (x1 <= margin) or (y1 <= margin) or (x2 >= frame_w - margin) or (y2 >= frame_h - margin)


def is_skeleton_complete(
    keypoints: np.ndarray | None,
    frame_w: int,
    frame_h: int,
    margin: int = 20,
    min_conf: float = 0.45,
) -> bool:
    if keypoints is None or len(keypoints) < 17:
        return False
    core_indices = [KP_L_SHOULDER, KP_R_SHOULDER, KP_L_HIP, KP_R_HIP]
    for idx in core_indices:
        x, y, conf = keypoints[idx]
        if conf < min_conf:
            return False
        if x <= margin or x >= (frame_w - margin) or y <= margin or y >= (frame_h - margin):
            return False
    return True


def is_occluded_by_others(
    target_box: list[int],
    all_boxes: list[list[int]],
    ioa_threshold: float = 0.18,
    keypoints: np.ndarray | None = None,
    min_kp_conf: float = 0.42,
) -> bool:
    tx1, ty1, tx2, ty2 = target_box
    bw = max(1, tx2 - tx1)
    bh = max(1, ty2 - ty1)
    aspect_ratio = bh / float(bw)

    for other_box in all_boxes:
        if target_box == other_box:
            continue
        ox1, oy1, ox2, oy2 = other_box
        overlap_ioa = calculate_ioa(target_box, other_box)
        if overlap_ioa > ioa_threshold:
            if oy2 >= (ty2 - 12) or overlap_ioa > 0.45:
                return True

    if aspect_ratio < 1.30:
        return True

    if keypoints is not None and len(keypoints) >= 17:
        shoulders_vis = (
            keypoints[KP_L_SHOULDER][2] >= min_kp_conf
            or keypoints[KP_R_SHOULDER][2] >= min_kp_conf
        )
        ankles_vis = (
            keypoints[KP_L_ANKLE][2] >= min_kp_conf
            or keypoints[KP_R_ANKLE][2] >= min_kp_conf
        )
        knees_vis = (
            keypoints[KP_L_KNEE][2] >= min_kp_conf
            or keypoints[KP_R_KNEE][2] >= min_kp_conf
        )
        if shoulders_vis and (not ankles_vis) and (not knees_vis):
            return True

    return False


def get_foot_point(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    keypoints: np.ndarray | None = None,
    expected_aspect_ratio: float | None = None,
    min_kp_conf: float = 0.45,
) -> tuple[int, int]:
    center_x = int((x1 + x2) / 2)
    bottom_y = int(y2)

    if keypoints is not None and len(keypoints) >= 17:
        l_ank = keypoints[KP_L_ANKLE]
        r_ank = keypoints[KP_R_ANKLE]

        vis_ankles_x = []
        vis_ankles_y = []
        if l_ank[2] >= min_kp_conf and l_ank[0] > 0 and l_ank[1] > 0:
            vis_ankles_x.append(l_ank[0])
            vis_ankles_y.append(l_ank[1])
        if r_ank[2] >= min_kp_conf and r_ank[0] > 0 and r_ank[1] > 0:
            vis_ankles_x.append(r_ank[0])
            vis_ankles_y.append(r_ank[1])

        if vis_ankles_y:
            foot_x = int(np.mean(vis_ankles_x))
            foot_y = int(max(vis_ankles_y))
            shoe_offset = int(0.03 * (y2 - y1))
            return foot_x, max(bottom_y, foot_y + shoe_offset)

    return center_x, bottom_y


def estimate_body_orientation(
    keypoints: np.ndarray | None,
    box: list[int] | None = None,
    min_kp_conf: float = 0.40,
) -> tuple[str, float]:
    if keypoints is None or len(keypoints) < 17:
        return "FRONT", 0.35

    l_sh = keypoints[KP_L_SHOULDER]
    r_sh = keypoints[KP_R_SHOULDER]
    l_hip = keypoints[KP_L_HIP]
    r_hip = keypoints[KP_R_HIP]

    sh_valid = l_sh[2] >= min_kp_conf and r_sh[2] >= min_kp_conf
    hip_valid = l_hip[2] >= min_kp_conf and r_hip[2] >= min_kp_conf

    if not sh_valid and not hip_valid:
        return "FRONT", 0.35

    dx_sh = (l_sh[0] - r_sh[0]) if sh_valid else 0.0
    dx_hip = (l_hip[0] - r_hip[0]) if hip_valid else 0.0
    combined_dx = 0.65 * dx_sh + 0.35 * dx_hip
    torso_h = max(20.0, (box[3] - box[1]) * 0.35) if box is not None else 60.0
    width_ratio = abs(combined_dx) / float(torso_h)

    if width_ratio < 0.34:
        return ("LEFT" if combined_dx >= 0 else "RIGHT"), 0.60

    total_front_score = np.clip(combined_dx / (0.5 * torso_h), -1.0, 1.0)
    conf = float(min(0.95, 0.55 + 0.35 * abs(total_front_score)))
    return ("FRONT" if total_front_score >= 0.0 else "BACK"), conf


def estimate_normalized_height(box: list[int], keypoints: np.ndarray | None, frame_h: int) -> float | None:
    x1, y1, x2, y2 = box
    bh = float(y2 - y1)
    bw = float(max(1, x2 - x1))
    if (bh / bw) < 1.65:
        return None
    _, foot_y = get_foot_point(x1, y1, x2, y2, keypoints=keypoints)
    depth_scale = max(0.35, float(foot_y) / float(max(1, frame_h)))
    return bh / depth_scale


def draw_skeleton(frame: np.ndarray, keypoints: np.ndarray | None, min_conf: float = 0.40):
    """
    رسم خطوط استخوان‌های اسکلت و مفاصل روی بدن انسان (مشابه تصویر نسخه قبلی).
    """
    if keypoints is None or len(keypoints) < 17:
        return

    # رسم استخوان‌ها (Lines)
    for pt1_idx, pt2_idx in SKELETON_CONNECTIONS:
        pt1 = keypoints[pt1_idx]
        pt2 = keypoints[pt2_idx]
        if pt1[2] >= min_conf and pt2[2] >= min_conf:
            p1 = (int(pt1[0]), int(pt1[1]))
            p2 = (int(pt2[0]), int(pt2[1]))
            cv2.line(frame, p1, p2, (0, 255, 120), 2)

    # رسم مفصل‌ها (Circles)
    for idx in range(17):
        pt = keypoints[idx]
        if pt[2] >= min_conf:
            center = (int(pt[0]), int(pt[1]))
            cv2.circle(frame, center, 3, (0, 255, 255), -1)