"""
Posture & Gesture Analyzer - 基于 MediaPipe Pose 的姿态和手势识别

职责：
- 接收人框裁剪图，输出 posture（standing/sitting/lying/unknown）
  和 gesture（waving/raising_left_arm/raising_right_arm/pointing_left/pointing_right/none/unknown）
- 维护每人 30 帧环形缓冲区，支持跨帧挥手检测（两路 OR：前臂摆动 + 手腕旋转）
"""

import math
import numpy as np
from collections import defaultdict
import cv2

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


# MediaPipe Pose 关键点索引
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_INDEX = 19   # 左手食指 MCP（掌指关节）
RIGHT_INDEX = 20  # 右手食指 MCP


# ==================== RingBuffer ====================

class RingBuffer:
    """每人维护一个环形缓冲区，存储最近 N 帧的 forearm + wrist 角度"""

    MIN_POST_JUMP = 15  # 跳变后最少保留帧数，不够则方差返回 0

    MAX_CONSECUTIVE_NONE = 5  # 连续 None 超过此次数 → 清空缓冲区

    def __init__(self, capacity=30):
        self.capacity = capacity
        self.forearm = defaultdict(lambda: {"left": [], "right": []})
        self.wrist_rot = defaultdict(lambda: {"left": [], "right": []})
        # 跟踪连续 None 次数: _none[person_id][source_name][side] = count
        self._none = defaultdict(lambda: defaultdict(lambda: {"left": 0, "right": 0}))

    def _push_side(self, source_name, store, person_id, side, angle):
        """单侧推送：非 None 追加并重置计数；None 累计计数，连续超阈值则清空旧数据"""
        if angle is not None:
            store[person_id][side].append(angle)
            if len(store[person_id][side]) > self.capacity:
                store[person_id][side].pop(0)
            self._none[person_id][source_name][side] = 0
        else:
            self._none[person_id][source_name][side] += 1
            if self._none[person_id][source_name][side] >= self.MAX_CONSECUTIVE_NONE:
                store[person_id][side].clear()

    def push_forearm(self, person_id, angle_left=None, angle_right=None):
        """Rule A: 前臂角度（弧度）"""
        self._push_side("forearm", self.forearm, person_id, "left", angle_left)
        self._push_side("forearm", self.forearm, person_id, "right", angle_right)

    def push_wrist(self, person_id, angle_left=None, angle_right=None):
        """Rule B: 手腕旋转角度（度）"""
        self._push_side("wrist", self.wrist_rot, person_id, "left", angle_left)
        self._push_side("wrist", self.wrist_rot, person_id, "right", angle_right)

    @staticmethod
    def _jump_aware_variance(angles, jump_threshold, window=30):
        """
        从最新帧往前扫描跳变点，只取跳变之后的数据计算方差。

        ① 计算 diff_n = |θₙ − θₙ₋₁|，从最新帧往前扫
        ② 找到第一个 > jump_threshold 的跳变点，切掉跳变前数据
        ③ 跳变后数据 < MIN_POST_JUMP → 返回 0
        ④ 无跳变 → 照常算

        Returns:
            (variance, post_jump_count) 或 (0.0, 0)
        """
        if len(angles) < 2:
            return 0.0, 0

        recent = angles[-min(window, len(angles)):]

        # 从最新帧往前扫描跳变
        cut_idx = 0  # 切掉 [0:cut_idx]，保留 [cut_idx:]
        n = len(recent)
        for i in range(n - 1, 0, -1):
            diff = abs(recent[i] - recent[i - 1])
            if diff > jump_threshold:
                cut_idx = i
                break

        post_jump = recent[cut_idx:]
        if len(post_jump) < RingBuffer.MIN_POST_JUMP:
            return 0.0, len(post_jump)

        return float(np.var(post_jump)), len(post_jump)

    def get_forearm_variance(self, person_id, side="left", window=30, jump_threshold=None):
        """Rule A 方差（弧度²），带跳变过滤"""
        angles = self.forearm[person_id][side]
        if not angles:
            return 0.0
        if jump_threshold is not None:
            var, _ = self._jump_aware_variance(angles, jump_threshold, window)
            return var
        # 无跳变阈值 → 照常
        if len(angles) < max(5, window // 2):
            return 0.0
        recent = angles[-min(window, len(angles)):]
        return float(np.var(recent))

    def get_wrist_variance(self, person_id, side="left", window=30, jump_threshold=None):
        """Rule B 方差（度数²），带跳变过滤"""
        angles = self.wrist_rot[person_id][side]
        if not angles:
            return 0.0
        if jump_threshold is not None:
            var, _ = self._jump_aware_variance(angles, jump_threshold, window)
            return var
        # 无跳变阈值 → 照常
        if len(angles) < max(5, window // 2):
            return 0.0
        recent = angles[-min(window, len(angles)):]
        return float(np.var(recent))

    def clear(self):
        self.forearm.clear()
        self.wrist_rot.clear()
        self._none.clear()


# ==================== PostureGestureAnalyzer ====================

class PostureGestureAnalyzer:
    """基于 MediaPipe Pose 的姿态 + 手势分析器"""

    def __init__(self, T_FOREARM=300, T_WRIST=500, SHOULDER_OFFSET=0.15,
                 JUMP_THRESHOLD=50):
        """
        Args:
            T_FOREARM: 前臂角度方差阈值（度²），默认 300
            T_WRIST:  手掌方向角度方差阈值（度²），默认 200
            SHOULDER_OFFSET: IMCP 低于肩部的最大容忍值（归一化坐标），默认 0.15
            JUMP_THRESHOLD: 角度跳变检测阈值，默认 20 度
        """
        self.available = MEDIAPIPE_AVAILABLE
        self.pose = None
        self.hands = None
        self.ring_buffer = RingBuffer(capacity=30)

        # 挥手检测阈值（可调参数）
        self.T_FOREARM = T_FOREARM
        self.T_WRIST = T_WRIST
        self.SHOULDER_OFFSET = SHOULDER_OFFSET
        self.JUMP_THRESHOLD = JUMP_THRESHOLD

        if MEDIAPIPE_AVAILABLE:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

    def process_pose(self, person_crop: np.ndarray) -> dict:
        """
        对单个人框裁剪图跑 MediaPipe Pose，返回原始关键点（不做分类）。

        Returns:
            {"landmarks": [(x,y,score), ... 33 keypoints], "img_h": int, "img_w": int}
            或 None（如果 Pose 不可用或未检测到人）。
        """
        if not self.available or self.pose is None or person_crop.size == 0:
            return None
        h, w = person_crop.shape[:2]
        if h < 30 or w < 30:
            return None
        rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        if results.pose_landmarks is None:
            return None
        landmarks = [(lm.x, lm.y, lm.visibility)
                     for lm in results.pose_landmarks.landmark]
        return {"landmarks": landmarks, "img_h": h, "img_w": w}

    def process_hands(self, person_crop: np.ndarray) -> dict:
        """
        对单个人框裁剪图跑 MediaPipe Hands，返回手腕/食指指尖的精细坐标 + 全部 21 点。

        Returns:
            {
                "Left":  {"wrist": (x,y), "index_tip": (x,y), "landmarks": [(x,y),...]} | None,
                "Right": ...,
            }
            仅当 hand detection confidence > 0.5 时才返回该侧数据。
        """
        result = {"Left": None, "Right": None}
        if not self.available or self.hands is None or person_crop.size == 0:
            return result

        h, w = person_crop.shape[:2]
        if h < 30 or w < 30:
            return result

        rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        hands_results = self.hands.process(rgb)

        if hands_results.multi_hand_landmarks:
            for idx, hand_lms in enumerate(hands_results.multi_hand_landmarks):
                handedness = hands_results.multi_handedness[idx]
                label = handedness.classification[0].label  # "Left" or "Right"
                score = handedness.classification[0].score
                if score < 0.5:
                    continue
                wrist = (hand_lms.landmark[0].x, hand_lms.landmark[0].y)
                index_tip = (hand_lms.landmark[8].x, hand_lms.landmark[8].y)
                landmarks = [(lm.x, lm.y) for lm in hand_lms.landmark]
                result[label] = {
                    "wrist": wrist,
                    "index_tip": index_tip,
                    "landmarks": landmarks,
                }

        return result

    def analyze_from_landmarks(self, pose_data: dict, person_id=None,
                               hands_data: dict = None, with_temporal: bool = True,
                               keypoints_3d: dict = None) -> dict:
        """
        从预计算的 landmarks 做分类 + 角度计算（不再跑模型推理）。

        Args:
            pose_data: process_pose() 的输出 {"landmarks": [...], "img_h": h, "img_w": w}
            person_id:  时序挥手检测用的人物 ID（with_temporal=True 时必传）
            hands_data: process_hands() 的输出
            with_temporal: 是否跑时序挥手检测
            keypoints_3d: {23: (x,y,z), 24: (x,y,z), ...} 深度图提取的 3D 关键点

        Returns:
            同 analyze() 或 analyze_with_temporal()
        """
        if pose_data is None:
            return self._unknown_result()

        landmarks = pose_data["landmarks"]
        h, w = pose_data["img_h"], pose_data["img_w"]

        posture, posture_conf = self._classify_posture(landmarks, h, keypoints_3d)
        gesture, gesture_conf, elbow_l, elbow_r, wrist_l, wrist_r = \
            self._classify_static_gesture(landmarks, h, w, hands_data, person_id)

        result = {
            "landmarks": landmarks,
            "posture": posture,
            "gesture": gesture,
            "confidence": min(posture_conf, gesture_conf),
            "elbow_angle_left": elbow_l,
            "elbow_angle_right": elbow_r,
            "wrist_angle_left": wrist_l,
            "wrist_angle_right": wrist_r,
        }

        if with_temporal and person_id is not None:
            result = self._apply_temporal(result, person_id)

        return result

    def _apply_temporal(self, result: dict, person_id) -> dict:
        """
        时序挥手检测（缓冲区 + 方差计算），不跑模型推理。
        供 analyze_from_landmarks(with_temporal=True) 调用。

        Rule A: 前臂摆动（wrist→elbow atan2 方差 > T_FOREARM）
        Rule B: 手腕旋转（∠(INDEX_MCP, WRIST, ELBOW) 方差 > T_WRIST）
        辅助条件: INDEX_MCP 不低于同侧肩膀过多（< SHOULDER_OFFSET）

        waving = (RuleA or RuleB) and 辅助条件
        """
        # 始终更新缓冲区（不按静态手势门控）：
        # 挥手涉及手臂周期性运动，各帧静态分类可能在 unknown/raising/pointing/none
        # 之间切换，应由时序信号判定，不应被单帧分类阻断。
        # 每侧独立推送：左侧为 None 不影响右侧数据入 buffer
        self.ring_buffer.push_forearm(
            person_id,
            result["elbow_angle_left"],
            result["elbow_angle_right"],
        )
        self.ring_buffer.push_wrist(
            person_id,
            result["wrist_angle_left"],
            result["wrist_angle_right"],
        )

        # Rule A: 前臂摆动方差（带跳变过滤）
        var_fa_l = self.ring_buffer.get_forearm_variance(
            person_id, "left", jump_threshold=self.JUMP_THRESHOLD)
        var_fa_r = self.ring_buffer.get_forearm_variance(
            person_id, "right", jump_threshold=self.JUMP_THRESHOLD)
        max_fa = max(var_fa_l, var_fa_r)
        rule_a = max_fa > self.T_FOREARM

        # Rule B: 手腕旋转方差（带跳变过滤）
        var_wr_l = self.ring_buffer.get_wrist_variance(
            person_id, "left", jump_threshold=self.JUMP_THRESHOLD)
        var_wr_r = self.ring_buffer.get_wrist_variance(
            person_id, "right", jump_threshold=self.JUMP_THRESHOLD)
        max_wr = max(var_wr_l, var_wr_r)
        rule_b = max_wr > self.T_WRIST

        # 辅助条件: INDEX_MCP 不低于肩膀过多
        shoulder_ok = self._check_shoulder_proximity(
            result.get("landmarks", [])
        )

        # debug: ban rule_a
        # if rule_b and shoulder_ok:
        if (rule_a or rule_b) and shoulder_ok:
            result["gesture"] = "waving"

        # 将方差数据挂到 result 上，供外部（如测试脚本）读取，不做日志记录
        result["max_fa"] = max_fa
        result["max_wr"] = max_wr
        result["var_fa_l"] = var_fa_l
        result["var_fa_r"] = var_fa_r
        result["var_wr_l"] = var_wr_l
        result["var_wr_r"] = var_wr_r
        result["shoulder_ok"] = shoulder_ok

        return result

    def analyze(self, person_crop: np.ndarray, hands_data: dict = None) -> dict:
        """
        对单个人框裁剪图做姿态+静态手势分析（便捷方法，内部跑 Pose 模型）。

        如需并行 Pose + Hands，请使用 process_pose() + process_hands() +
        analyze_from_landmarks() 替代。
        """
        return self.analyze_from_landmarks(
            self.process_pose(person_crop), hands_data=hands_data, with_temporal=False)

    def analyze_with_temporal(self, person_id, person_crop: np.ndarray,
                              hands_data: dict = None) -> dict:
        """
        包含时序挥手检测的完整分析（便捷方法，内部跑 Pose 模型）。

        如需并行 Pose + Hands，请使用 process_pose() + process_hands() +
        analyze_from_landmarks(with_temporal=True) 替代。
        """
        return self.analyze_from_landmarks(
            self.process_pose(person_crop), person_id=person_id,
            hands_data=hands_data, with_temporal=True)

    def _check_shoulder_proximity(self, landmarks) -> bool:
        """
        辅助条件：INDEX_MCP 在同侧肩膀高度附近。

        至少有一侧 INDEX_MCP 满足:
          index_mcp_y <= shoulder_y + SHOULDER_OFFSET
        """
        if len(landmarks) <= RIGHT_INDEX:
            return False

        def visible(idx):
            return landmarks[idx][2] > 0.5

        ok_left = False
        if visible(LEFT_SHOULDER) and visible(LEFT_INDEX):
            index_y = landmarks[LEFT_INDEX][1]
            shoulder_y = landmarks[LEFT_SHOULDER][1]
            ok_left = index_y <= shoulder_y + self.SHOULDER_OFFSET
        else:
            # 该侧不可见 → 默认不通过此侧
            pass

        ok_right = False
        if visible(RIGHT_SHOULDER) and visible(RIGHT_INDEX):
            index_y = landmarks[RIGHT_INDEX][1]
            shoulder_y = landmarks[RIGHT_SHOULDER][1]
            ok_right = index_y <= shoulder_y + self.SHOULDER_OFFSET

        return ok_left or ok_right

    # ==================== 姿态分类 ====================

    def _classify_posture(self, landmarks, img_height, keypoints_3d: dict = None):
        """
        三规则姿态分类：躯干平面竖直 + 膝角 > 150° + 肩髋膝踝高差链。

        keypoints_3d: {11: (x,y,z), 12: ..., 23: ..., 24: ..., 25: ..., 26: ...}
                      None 表示无深度图，回退到旧逻辑。
        """
        def get_2d(idx):
            return np.array([landmarks[idx][0], landmarks[idx][1]])

        def is_visible(idx):
            return landmarks[idx][2] > 0.5

        def leg_angle_2d(hip_idx, knee_idx, ankle_idx):
            if not all(is_visible(i) for i in [hip_idx, knee_idx, ankle_idx]):
                return None
            hip = get_2d(hip_idx)
            knee = get_2d(knee_idx)
            ankle = get_2d(ankle_idx)
            v1 = knee - hip
            v2 = ankle - knee
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 < 1e-6 or n2 < 1e-6:
                return None
            cos_a = np.dot(v1, v2) / (n1 * n2)
            cos_a = np.clip(cos_a, -1.0, 1.0)
            return math.degrees(math.acos(cos_a))

        # 若无 3D 关键点，保持原有逻辑
        if not keypoints_3d:
            left_angle = leg_angle_2d(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
            right_angle = leg_angle_2d(RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)
            angles = [a for a in (left_angle, right_angle) if a is not None]
            if not angles:
                return "unknown", 0.0
            avg = sum(angles) / len(angles)
            if avg > 150:
                return "standing", 0.8
            elif 80 < avg < 120:
                return "sitting", 0.8
            elif avg < 30:
                return "lying", 0.7
            return "unknown", 0.3

        def kp(idx):
            return keypoints_3d.get(idx)

        def height(idx):
            p = kp(idx)
            return p[1] if p is not None else None  # Y in camera coords

        # ---- 规则 1: 躯干平面竖直 (3D) ----
        torso_ok = False
        ls, rs, lh, rh = kp(11), kp(12), kp(23), kp(24)
        if all(v is not None for v in (ls, rs, lh, rh)):
            shoulder_mid = (np.array(ls) + np.array(rs)) / 2
            hip_mid = (np.array(lh) + np.array(rh)) / 2
            v1 = np.array(rs) - np.array(ls)         # 肩线（水平）
            v2 = shoulder_mid - hip_mid              # 脊柱（竖直）
            n = np.cross(v1, v2)
            gravity = np.array([0.0, -1.0, 0.0])     # Y- 是上，重力沿 Y+
            n_norm = np.linalg.norm(n)
            if n_norm > 1e-6:
                cos_theta = abs(np.dot(n, gravity)) / n_norm
                cos_theta = np.clip(cos_theta, 0.0, 1.0)
                theta = math.degrees(math.acos(cos_theta))
                torso_ok = 65 <= theta <= 115

        # ---- 规则 2: 任一侧膝角 > 150° ----
        left_angle = leg_angle_2d(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
        right_angle = leg_angle_2d(RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)
        knee_ok = False
        knee_skip = True
        if left_angle is not None:
            knee_skip = False
            if left_angle >= 150:
                knee_ok = True
        if right_angle is not None:
            knee_skip = False
            if right_angle >= 150:
                knee_ok = True
        # 双侧都不可见 → 跳过规则 2

        # ---- 规则 3: 肩 > 髋 > 膝 > 踝 高差链 (3D) ----
        chain_ok = False
        chain_skip = True

        def height(idx):
            p = kp(idx)
            return p[1] if p is not None else None  # Y in camera coords

        sh = height(11) or height(12)  # shoulder
        hi = height(23) or height(24)  # hip
        kn = height(25) or height(26)  # knee
        an = height(27) or height(28)  # ankle

        if sh is not None and hi is not None and kn is not None and an is not None:
            chain_skip = False
            EPS = 0.02  # 2cm 容差
            # Y- = up, Y+ = down → 站立时肩(−0.4) < 髋(−0.05) < 膝(+0.3) < 踝(+0.75)
            chain_ok = (sh < hi - EPS and hi < kn - EPS and kn < an - EPS)

        # ---- Sitting 判定：躯干竖直 + 脊柱 ⊥ 大腿 (3D) ----
        sitting_ok = False
        lk, rk = kp(25), kp(26)
        if torso_ok and all(v is not None for v in (ls, rs, lh, rh, lk, rk)):
            # 复用 Rule 1 的 shoulder_mid, hip_mid, spine (v2)
            spine = v2  # shoulder_mid - hip_mid (3D)
            knee_mid = (np.array(lk) + np.array(rk)) / 2
            thigh = knee_mid - hip_mid  # knee → hip, 指向前下方
            s_norm = np.linalg.norm(spine)
            t_norm = np.linalg.norm(thigh)
            if s_norm > 1e-6 and t_norm > 1e-6:
                cos_angle = abs(np.dot(spine, thigh)) / (s_norm * t_norm)
                sitting_ok = cos_angle < 0.5  # angle ∈ [60°, 120°]

        if sitting_ok:
            return "sitting", 0.85

        # ---- 综合判定 ----
        rules_passed = []
        if torso_ok:
            rules_passed.append("torso")
        if knee_skip:
            pass
        elif knee_ok:
            rules_passed.append("knee")
        if chain_skip:
            pass
        elif chain_ok:
            rules_passed.append("chain")

        n_passed = len(rules_passed)
        if n_passed == 0:
            return "unknown", 0.3

        if n_passed >= 2 or (torso_ok and n_passed >= 1):
            # standing 候选 → lying 深度校验
            if knee_ok:
                sh_y = height(11) or height(12)
                hi_y = height(23) or height(24)
                kn_y = height(25) or height(26)
                if all(v is not None for v in [sh_y, hi_y, kn_y]):
                    spread = max(sh_y, hi_y, kn_y) - min(sh_y, hi_y, kn_y)
                    if spread < 0.25:
                        return "lying", 0.85
            return "standing", 0.8

        # Trigger B: standing 规则不够但膝角可见 → 可能躺着（腿伸直或蜷缩都适用）
        if not knee_skip:
            sh_y = height(11) or height(12)
            hi_y = height(23) or height(24)
            kn_y = height(25) or height(26)
            if all(v is not None for v in [sh_y, hi_y, kn_y]):
                spread = max(sh_y, hi_y, kn_y) - min(sh_y, hi_y, kn_y)
                if spread < 0.25:
                    return "lying", 0.85

        return "unknown", 0.3

    # ==================== 静态手势分类 ====================

    def _classify_static_gesture(self, landmarks, img_height, img_width,
                                 hands_data: dict = None, person_id=None):
        """基于上肢关键点分类静态手势，同时返回时序分析所需的角度。
        若提供 hands_data，优先用 Hands 模型的精细 wrist/index 坐标替代 Pose 的粗关键点。"""
        def get_landmark(idx):
            return np.array([landmarks[idx][0], landmarks[idx][1]])

        def is_visible(idx):
            return landmarks[idx][2] > 0.5

        def is_visible_relaxed(idx):
            """松弛可见性：visibility > 0.3 即可用于角度计算"""
            return landmarks[idx][2] > 0.3

        # 从 hands_data 提取精细 wrist + index_tip（仅用于 Rule B，两者缺一则该侧 None）
        fine_wrist = {"left": None, "right": None}
        fine_index = {"left": None, "right": None}
        if hands_data:
            for side_label, pose_side in [("Left", "left"), ("Right", "right")]:
                hd = hands_data.get(side_label)
                if hd and hd.get("wrist") and hd.get("index_tip"):
                    fine_wrist[pose_side] = np.array(hd["wrist"])
                    fine_index[pose_side] = np.array(hd["index_tip"])

        def _compute_side_angles(side, elbow_idx, wrist_idx):
            """返回 (elbow_angle, wrist_angle)。
            Rule A: 始终用 Pose wrist。
            Rule B: 用 Hands wrist + index_tip，两者缺一则该侧返回 None。"""
            if not is_visible_relaxed(elbow_idx) or not is_visible_relaxed(wrist_idx):
                return None, None

            elbow = get_landmark(elbow_idx)
            wrist = get_landmark(wrist_idx)

            # Rule A: wrist->elbow atan2（始终 Pose）
            dx = wrist[0] - elbow[0]
            dy = wrist[1] - elbow[1]
            ea = math.degrees(math.atan2(dy, dx))

            # Rule B: palm direction atan2（Hands wrist + index_tip，缺一不可）
            wa = None
            if fine_wrist[side] is not None and fine_index[side] is not None:
                dx = fine_index[side][0] - fine_wrist[side][0]
                dy = fine_index[side][1] - fine_wrist[side][1]
                wa = math.degrees(math.atan2(dy, dx))

            return ea, wa

        # 单侧独立计算角度
        elbow_l, wrist_l = _compute_side_angles(
            "left", LEFT_ELBOW, LEFT_WRIST)
        elbow_r, wrist_r = _compute_side_angles(
            "right", RIGHT_ELBOW, RIGHT_WRIST)

        # 静态手势分类需要这些关键点全部高可见
        key_points = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
                      LEFT_WRIST, RIGHT_WRIST]
        if not all(is_visible(i) for i in key_points):
            return "unknown", 0.0, elbow_l, elbow_r, wrist_l, wrist_r

        left_shoulder = get_landmark(LEFT_SHOULDER)
        right_shoulder = get_landmark(RIGHT_SHOULDER)
        left_elbow = get_landmark(LEFT_ELBOW)
        right_elbow = get_landmark(RIGHT_ELBOW)
        left_wrist = get_landmark(LEFT_WRIST)
        right_wrist = get_landmark(RIGHT_WRIST)

        shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder)
        if shoulder_dist < 1e-6:
            return "unknown", 0.0, elbow_l, elbow_r, wrist_l, wrist_r

        MARGIN = shoulder_dist * 0.2
        T_STATIC = 50  # 度²，比 T_FOREARM(300) 低一个数量级

        # 举手判断：三规则
        def is_arm_raised(wrist, elbow, shoulder, side):
            """三规则举手检测。返回 (raised_bool, info_str)。"""
            # 规则 1: 手腕接近或超过肩膀高度
            rule1 = wrist[1] <= shoulder[1] + MARGIN
            if not rule1:
                return False, "rule1_fail"

            # 规则 2: 前臂竖直（肘→腕方向接近向上）
            horiz_ok = abs(wrist[0] - elbow[0]) < shoulder_dist * 0.3
            vert_ok = wrist[1] < elbow[1]
            if not (horiz_ok and vert_ok):
                return False, "rule2_fail"

            # 规则 3: 15 帧内无旋转（排除挥手）
            if person_id is not None:
                fa_var = self.ring_buffer.get_forearm_variance(
                    person_id, side, window=15)
                wr_var = self.ring_buffer.get_wrist_variance(
                    person_id, side, window=15)
                # Hands 无数据时只检查前臂方差
                wr_ok = (wr_var < T_STATIC) if wr_var > 0 else True
                if fa_var >= T_STATIC or not wr_ok:
                    return False, "rule3_fail"
            # person_id=None → 单帧模式，跳过规则 3
            return True, "ok"

        T_ARROW_Y = shoulder_dist * 0.25  # 箭身三点 y 对齐容差

        # 举手检测
        left_raised, _ = is_arm_raised(left_wrist, left_elbow, left_shoulder, "left")
        right_raised, _ = is_arm_raised(right_wrist, right_elbow, right_shoulder, "right")

        # 指向判断：三规则
        def is_pointing(wrist, elbow, index_pose, side):
            """
            规则 1: 箭身水平 (elbow→wrist→index 三点 y 接近)
            规则 2: 手指姿势 (Hands: index 伸出 + thumb/middle/ring/pinky 缩回)
            规则 3: 15 帧静止 (方差 < T_STATIC)
            返回 (pointing_bool, direction_str)。direction: "left"|"right"|None
            """
            # 规则 1: 箭身水平（elbow, wrist, index_mcp 三点 y 对齐）
            rule1 = (abs(elbow[1] - wrist[1]) < T_ARROW_Y and
                     abs(wrist[1] - index_pose[1]) < T_ARROW_Y)
            if not rule1:
                return False, None

            # 规则 2: 手指姿势（需要 Hands 模型）
            hands_ok = False
            if hands_data:
                hd = hands_data.get("Left" if side == "left" else "Right")
                if hd and hd.get("landmarks"):
                    lm = hd["landmarks"]
                    if len(lm) >= 21:
                        # 计算每指 TIP-MCP 距离
                        def tip_mcp_dist(mcp_idx):
                            tip = np.array(lm[mcp_idx + 3])  # TIP = MCP+3
                            mcp = np.array(lm[mcp_idx])
                            return np.linalg.norm(tip - mcp)
                        # 参考长度：index 的 PIP-MCP × 2.5
                        pip = np.array(lm[6])
                        mcp = np.array(lm[5])
                        ref_len = np.linalg.norm(pip - mcp) * 2.5
                        if ref_len < 1e-6:
                            ref_len = 0.1  # fallback

                        index_dist = tip_mcp_dist(5)
                        middle_dist = tip_mcp_dist(9)
                        ring_dist = tip_mcp_dist(13)
                        pinky_dist = tip_mcp_dist(17)
                        thumb_dist = np.linalg.norm(
                            np.array(lm[4]) - np.array(lm[2]))  # thumb TIP-MCP

                        index_ext = index_dist > ref_len * 0.75
                        middle_ret = middle_dist < ref_len * 0.50
                        ring_ret = ring_dist < ref_len * 0.50
                        pinky_ret = pinky_dist < ref_len * 0.50
                        thumb_ret = thumb_dist < ref_len * 0.50

                        # 模式 A: index伸出 + 其余缩回
                        # 模式 B: index+middle伸出 + thumb/ring/pinky缩回
                        pattern_a = index_ext and middle_ret and ring_ret and pinky_ret and thumb_ret
                        pattern_b = index_ext and ring_ret and pinky_ret and thumb_ret
                        hands_ok = pattern_a or pattern_b
            else:
                # 无 Hands 数据 → 规则 2 不通过
                return False, None

            if not hands_ok:
                return False, None

            # 规则 3: 15 帧静止
            if person_id is not None:
                fa_var = self.ring_buffer.get_forearm_variance(
                    person_id, side, window=15)
                wr_var = self.ring_buffer.get_wrist_variance(
                    person_id, side, window=15)
                wr_ok = (wr_var < T_STATIC) if wr_var > 0 else True
                if fa_var >= T_STATIC or not wr_ok:
                    return False, None

            # 方向: elbow.x < wrist.x → pointing_right (摄像机视角)
            direction = "right" if elbow[0] < wrist[0] else "left"
            return True, direction

        # Pose index_mcp 坐标
        def get_index_pose(side):
            idx = LEFT_INDEX if side == "left" else RIGHT_INDEX
            if idx < len(landmarks) and landmarks[idx][2] > 0.3:
                return get_landmark(idx)
            return None

        left_index = get_index_pose("left")
        right_index = get_index_pose("right")

        left_pointing, l_dir = (is_pointing(left_wrist, left_elbow, left_index, "left")
                                if left_index else (False, None))
        right_pointing, r_dir = (is_pointing(right_wrist, right_elbow, right_index, "right")
                                 if right_index else (False, None))

        # 分类：raising > pointing 互斥（优先级：raising 先判；waving 在 _apply_temporal 中时序覆盖）
        if left_raised and not right_raised:
            return "raising_left_arm", 0.85, elbow_l, elbow_r, wrist_l, wrist_r
        elif right_raised and not left_raised:
            return "raising_right_arm", 0.85, elbow_l, elbow_r, wrist_l, wrist_r

        if left_pointing and not right_pointing:
            return f"pointing_{l_dir}", 0.75, elbow_l, elbow_r, wrist_l, wrist_r
        elif right_pointing and not left_pointing:
            return f"pointing_{r_dir}", 0.75, elbow_l, elbow_r, wrist_l, wrist_r
        elif left_pointing and right_pointing:
            return "pointing_both", 0.75, elbow_l, elbow_r, wrist_l, wrist_r
        elif left_raised and right_raised:
            return "raising_both_arms", 0.80, elbow_l, elbow_r, wrist_l, wrist_r
        else:
            return "none", 0.7, elbow_l, elbow_r, wrist_l, wrist_r

    @staticmethod
    def _unknown_result():
        return {
            "landmarks": [],
            "posture": "unknown",
            "gesture": "unknown",
            "confidence": 0.0,
            "elbow_angle_left": None,
            "elbow_angle_right": None,
            "wrist_angle_left": None,
            "wrist_angle_right": None,
        }
