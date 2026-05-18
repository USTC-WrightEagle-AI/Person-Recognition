"""
CadeTracker - 跨帧追踪器

3 层约束匹配，保障每个检测对象获得稳定的跨帧 track_id：
- 第 1 层：3D 位置连续性（欧氏距离 < MAX_3D_DIST）
- 第 2 层：2D bbox IOU（> MIN_IOU）
- 第 3 层：匈牙利算法全局最优匹配
- 生命周期：连续 MAX_AGE 帧未匹配则自动过期
"""

import math
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class CadeTracker:
    """跨帧追踪器：3D 位置 → 2D IOU → 匈牙利全局最优。"""

    MAX_3D_DIST = 0.3   # 3D 匹配阈值（米）
    MIN_IOU = 0.3       # 2D IOU 匹配阈值
    MAX_AGE = 5         # 连续未匹配帧数上限

    def __init__(self):
        self._tracks = []
        self._next_id = 0

    def assign(self, detections: list) -> list:
        """为当前帧检测分配 track_id，返回等长列表。"""
        n_tracks = len(self._tracks)
        n_dets = len(detections)

        if n_tracks == 0 or n_dets == 0:
            track_ids = [self._next_id + i for i in range(n_dets)]
            self._next_id += n_dets
            self._tracks = self._new_tracks(detections, track_ids)
            return track_ids

        cost = np.full((n_tracks, n_dets), 1e6)
        for ti, tr in enumerate(self._tracks):
            for di, det in enumerate(detections):
                cost[ti, di] = self._cost(tr, det)

        if SCIPY_AVAILABLE:
            row_ind, col_ind = linear_sum_assignment(cost)
        else:
            row_ind, col_ind = self._greedy_match(cost)

        track_ids = [-1] * n_dets
        matched_t = set()
        matched_d = set()

        for ti, di in zip(row_ind, col_ind):
            if cost[ti, di] < 1.0:
                track_ids[di] = self._tracks[ti]["id"]
                matched_t.add(ti)
                matched_d.add(di)

        for di in range(n_dets):
            if di not in matched_d:
                track_ids[di] = self._next_id
                self._next_id += 1

        new_tracks = []
        for ti, di in zip(row_ind, col_ind):
            if cost[ti, di] < 1.0:
                new_tracks.append(self._update(self._tracks[ti], detections[di]))
        for ti in range(n_tracks):
            if ti not in matched_t:
                tr = self._tracks[ti]
                tr["age"] += 1
                if tr["age"] < self.MAX_AGE:
                    new_tracks.append(tr)
        for di in range(n_dets):
            if di not in matched_d:
                new_tracks.append(self._update(
                    {"id": track_ids[di], "age": 0}, detections[di]))
        self._tracks = new_tracks
        return track_ids

    def _cost(self, track, det) -> float:
        """匹配代价（越小越好）。第 1 层 3D 距离，第 2 层 2D IOU。"""
        pos_t = track.get("pos_3d")
        pos_d = det.get("position_3d")
        if pos_t is not None and pos_d is not None:
            d = math.sqrt((pos_t[0]-pos_d[0])**2 + (pos_t[1]-pos_d[1])**2 + (pos_t[2]-pos_d[2])**2)
            if d < self.MAX_3D_DIST:
                return d / self.MAX_3D_DIST
            return 1.0
        iou = self._iou(track.get("bbox"), det.get("bbox"))
        if iou is not None and iou > self.MIN_IOU:
            return 1.0 - iou
        return 1.0

    @staticmethod
    def _iou(box_a, box_b):
        if box_a is None or box_b is None:
            return None
        xa, ya = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
        xb, yb = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        return inter / (area_a + area_b - inter + 1e-6)

    @staticmethod
    def _update(track, det):
        return {"id": track["id"], "bbox": det.get("bbox"), "center": det.get("center"),
                "pos_3d": det.get("position_3d"), "age": 0, "class_name": det.get("class_name")}

    @staticmethod
    def _new_tracks(detections, track_ids):
        return [CadeTracker._update({"id": tid, "age": 0}, d)
                for d, tid in zip(detections, track_ids)]

    @staticmethod
    def _greedy_match(cost):
        """无 scipy 时的贪心匹配回退。"""
        n, m = cost.shape
        used_c = set()
        pairs = []
        rows = list(range(n))
        rows.sort(key=lambda r: cost[r].min())
        for r in rows:
            best_c, best_c_idx = 1e6, -1
            for c in range(m):
                if c not in used_c and cost[r, c] < best_c:
                    best_c, best_c_idx = cost[r, c], c
            if best_c_idx >= 0:
                pairs.append((r, best_c_idx))
                used_c.add(best_c_idx)
        row_arr = [p[0] for p in pairs]
        col_arr = [p[1] for p in pairs]
        return np.array(row_arr), np.array(col_arr)
