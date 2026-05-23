"""
Cloth Analyzer - 衣物颜色提取与人物关联

无状态，每帧独立调用。将 YOLO 检出的衣物框通过 IOU 匹配到人，
用 HSV 提取颜色，写入 person 对象的 cloth_color / cloth_type 字段。

不依赖 ROS，不引入额外模型。
"""

import numpy as np
import cv2


# 衣物 HSV 阈值（经验值，待实地标定）
SAT_LOW = 30     # S < 此值为无彩色（白/灰/黑）
V_HIGH = 200     # V > 此值为白
V_LOW = 80       # V < 此值为黑

# H 分段（OpenCV H ∈ [0, 180]）
H_RED_LO = 0
H_RED_HI = 15
H_YELLOW_LO = 16
H_YELLOW_HI = 35
H_BLUE_LO = 90
H_BLUE_HI = 140


def _iou(box_a, box_b):
    xa, ya = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    xb, yb = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def _is_clothing(class_name):
    """不依赖固定列表：只要 class_name 不是 person 且不包含 person，就视为衣物。"""
    name = class_name.lower()
    return "person" not in name


def _hsv_to_color(hsv_mean):
    """HSV 均值 → 颜色名。"""
    h, s, v = hsv_mean
    if s < SAT_LOW:
        if v > V_HIGH:
            return "white"
        elif v < V_LOW:
            return "black"
        return "gray"
    if H_RED_LO <= h <= H_RED_HI or h >= 170:
        return "red"
    if H_YELLOW_LO <= h <= H_YELLOW_HI:
        return "yellow"
    if H_BLUE_LO <= h <= H_BLUE_HI:
        return "blue"
    return "unknown"


def _extract_color(color_image, bbox):
    """从原图裁剪衣服框中心区域，取 HSV 均值，映射颜色名。"""
    x1, y1, x2, y2 = bbox
    h, w = color_image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return "unknown"

    crop = color_image[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    inset_x = int(cw * 0.15)
    inset_y = int(ch * 0.15)
    if inset_x * 2 >= cw or inset_y * 2 >= ch:
        inset_x = inset_y = 0
    center = crop[inset_y:ch - inset_y, inset_x:cw - inset_x]
    if center.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    mean = hsv.reshape(-1, 3).mean(axis=0)
    return _hsv_to_color(mean)


def associate_clothing(detections, color_image):
    """
    主入口。将衣物框匹配到人框，提取颜色和类型。

    Args:
        detections: [obj_info, ...]  含 class_name, bbox
        color_image: BGR numpy array（原始帧）

    修改 detections 中的 person 对象，增加 cloth_color / cloth_type 字段。
    """
    persons = [d for d in detections if d["class_name"] == "person"]
    cloths = [d for d in detections if _is_clothing(d["class_name"])]

    if not persons or not cloths:
        for p in persons:
            p.setdefault("cloth_color", "unknown")
            p.setdefault("cloth_type", "unknown")
        return

    # 每个衣服找 IOU 最大的 person
    matches = []  # [(person_idx, cloth_idx, iou), ...]
    for ci, cloth in enumerate(cloths):
        best_pi, best_iou = -1, 0.0
        for pi, person in enumerate(persons):
            iou = _iou(person["bbox"], cloth["bbox"])
            if iou > best_iou:
                best_iou, best_pi = iou, pi
        if best_iou > 0.3:
            matches.append((best_pi, ci, best_iou))

    # 对每个人，上衣和下装各保留 IOU 最高的一件
    UPPER_LABELS = {"t-shirt", "shirt", "sweater", "jacket", "coat",
                    "blouse", "hoodie", "vest", "top"}
    person_upper = {}
    person_lower = {}
    for pi, ci, iou in matches:
        ctype = cloths[ci]["class_name"].lower()
        if ctype in UPPER_LABELS or "jacket" in ctype or "shirt" in ctype or "sweater" in ctype or "coat" in ctype or "hoodie" in ctype or "vest" in ctype or "top" in ctype or "blouse" in ctype:
            if pi not in person_upper or iou > person_upper[pi][1]:
                person_upper[pi] = (ci, iou)
        else:
            if pi not in person_lower or iou > person_lower[pi][1]:
                person_lower[pi] = (ci, iou)

    # 写入 person 对象
    for pi, person in enumerate(persons):
        person.setdefault("cloth_color", "unknown")
        person.setdefault("cloth_type", "unknown")

        colors = []
        types = []
        if pi in person_upper:
            ci, _ = person_upper[pi]
            cloth = cloths[ci]
            color = _extract_color(color_image, cloth["bbox"])
            colors.append(color)
            types.append(cloth["class_name"])
        if pi in person_lower:
            ci, _ = person_lower[pi]
            cloth = cloths[ci]
            color = _extract_color(color_image, cloth["bbox"])
            colors.append(color)
            types.append(cloth["class_name"])

        if colors:
            person["cloth_color"] = "/".join(colors)
            person["cloth_type"] = "/".join(types)
