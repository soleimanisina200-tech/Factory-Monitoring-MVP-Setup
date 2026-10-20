import math

def get_foot_point(x1: int, y1: int, x2: int, y2: int) -> tuple[int, int]:
    """نقطه پایین-مرکز کادر برای بررسی حضور در ناحیه (Zone)"""
    return (int((x1 + x2) / 2), int(y2))

def calculate_iou(boxA: list[int], boxB: list[int]) -> float:
    """محاسبه همپوشانی دو کادر [x1, y1, x2, y2]"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return interArea / float(boxAArea + boxBArea - interArea)

def is_occluded_by_others(target_box: list[int], all_boxes: list[list[int]], iou_threshold: float = 0.12) -> bool:
    """بررسی همپوشانی کادر با سایر افراد حاضر در صحنه"""
    for other_box in all_boxes:
        if target_box == other_box:
            continue
        if calculate_iou(target_box, other_box) > iou_threshold:
            return True
    return False

def evaluate_spatial_state(
    target_idx: int,
    all_boxes: list[list[int]],
    frame_w: int,
    frame_h: int,
    margin: int = 35
) -> dict:
    """
    تحلیل وضعیت هندسی کارگر در فریم جاری:
    - is_standing_clean: ایستاده، تمام‌قد، و داخل مرز امن تصویر (مناسب برای ثبت در آلبوم مرجع)
    - is_occluded: در حال تقاطع یا مماس با کارگر دیگر
    - is_isolated: با فاصله امن از سایر کارگران
    - center: مختصات مرکز کادر برای بازیابی افراد نشسته/پشت دستگاه
    """
    x1, y1, x2, y2 = all_boxes[target_idx]
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    aspect_ratio = h / w
    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    is_inside_margin = (
        x1 > margin and y1 > margin and
        x2 < (frame_w - margin) and y2 < (frame_h - margin)
    )
    # انسان ایستاده نرمال نسبت ارتفاع به عرض بین 1.35 تا 3.8 دارد
    is_standing_clean = (1.35 <= aspect_ratio <= 3.80) and (h >= 85) and is_inside_margin

    is_occluded = False
    is_isolated = True

    for j, other_box in enumerate(all_boxes):
        if j == target_idx:
            continue
        iou = calculate_iou([x1, y1, x2, y2], other_box)
        ox1, oy1, ox2, oy2 = other_box
        other_center = ((ox1 + ox2) / 2.0, (oy1 + oy2) / 2.0)
        center_dist = math.hypot(center[0] - other_center[0], center[1] - other_center[1])

        # اگر همپوشانی دارند یا فاصله مراکزشان کمتر از عرض شانه است -> در حال تقاطع
        if iou > 0.10 or center_dist < (w * 0.95):
            is_occluded = True
            is_isolated = False
            break
        elif iou > 0.01 or center_dist < (w * 1.65):
            is_isolated = False

    return {
        'is_standing_clean': is_standing_clean,
        'is_occluded': is_occluded,
        'is_isolated': is_isolated,
        'center': center,
        'bbox': [x1, y1, x2, y2],
        'aspect_ratio': aspect_ratio
    }