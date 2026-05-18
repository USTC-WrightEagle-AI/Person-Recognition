# CADE Vision

CADE 机器人的**纯视觉感知节点** — YOLO-World 目标检测 + MediaPipe 姿态手势 + 3D 定位。

**设计原则：纯特征提取器，只做感知，不做控制。** 节点绝对不包含底盘控制、LLM 调用、模型状态修改。职责边界：接收 `/cade/task_cmd` 指令 → 驱动视觉模型 → 发布 `/vision/detections_3d`。

---

## 系统要求

| 依赖 | 说明 |
|------|------|
| ROS Noetic (或 Melodic) | catkin 构建 + 话题通信 |
| Python 3.8+ | 推荐 conda 环境 |
| CUDA (可选) | GPU 加速 YOLO 推理 |

Python 依赖 (`pip install`)：

```
ultralytics opencv-python numpy scipy mediapipe
```

RealSense 用户额外安装 `pyrealsense2`。

---

## 快速开始：克隆 → 构建 → 运行

### 1. 克隆到 ROS workspace

```bash
# 如果你已有 catkin workspace
cd ~/catkin_ws/src
git clone <repo-url> cade_vision

# 或者新建一个
mkdir -p ~/cade_ws/src && cd ~/cade_ws/src
git clone <repo-url> cade_vision
```

### 2. 构建

```bash
cd ~/cade_ws          # 或你的 workspace 根目录
catkin_make
source devel/setup.bash
```

### 3. 运行

```bash
# 启动 roscore（新终端）
roscore

# 启动 vision node（新终端）
source ~/cade_ws/devel/setup.bash
rosrun cade_vision open_vision_node.py --device cuda
```

---

## 图像源模式

通过 `--image-source` 选择输入源：

### File 模式（视频文件测试）

```bash
rosrun cade_vision open_vision_node.py \
    --image-source file \
    --image-path /path/to/video.mp4 \
    --model /path/to/yolov8x-worldv2.pt \
    --device cuda \
    --loop --playback-speed 0.2
```

| 专用参数 | 说明 |
|----------|------|
| `--image-path PATH` | 图片或视频文件路径（自动检测类型） |
| `--loop` | 视频播完自动从头循环 |
| `--playback-speed N` | 播放速率。`1.0`=原速，`0.2`=0.2x 慢速，不加=满速 |

### USB 摄像头模式

```bash
rosrun cade_vision open_vision_node.py \
    --image-source usb_cam --usb-cam-id 0 \
    --model /path/to/yolov8x-worldv2.pt --device cuda
```

| 专用参数 | 说明 |
|----------|------|
| `--usb-cam-id N` | 摄像头设备 ID，默认 `0` |
| `--width W` | 分辨率，默认 `640` |
| `--height H` | 分辨率，默认 `480` |

### RealSense 模式（默认，Jetson 部署）

```bash
rosrun cade_vision open_vision_node.py \
    --model /path/to/yolov8x-worldv2.pt --device cuda
```

| 专用参数 | 说明 |
|----------|------|
| `--serial-number S` | RealSense 序列号 |

---

## 全部命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `yolo11x-seg.pt` | YOLO 模型路径 |
| `--conf` | `0.25` | 检测置信度阈值 |
| `--iou` | `0.5` | NMS IoU 阈值 |
| `--device` | `cuda` | 推理设备：`cuda` / `cpu` |
| `--image-source` | `realsense` | 图像源：`realsense` / `file` / `usb_cam` |
| `--image-path` | — | file 模式下的图片/视频路径 |
| `--loop` | 关闭 | 视频播完自动循环（仅 file 视频模式） |
| `--playback-speed` | — | 播放速率（仅 file 视频模式） |
| `--usb-cam-id` | `0` | USB 摄像头设备 ID |
| `--width` | `640` | USB 摄像头宽度 |
| `--height` | `480` | USB 摄像头高度 |
| `--serial-number` | — | RealSense 序列号 |
| `--display` / `--no-display` | 开启 | 显示/隐藏检测窗口 |

按 `q` 退出显示窗口。

---

## ROS API

### Subscriber（输入）

| Topic | 类型 | 说明 |
|-------|------|------|
| `/cade/task_cmd` | `std_msgs/String` | 大脑发出的 JSON 任务指令 |

### Publisher（输出）

| Topic | 类型 | 说明 |
|-------|------|------|
| `/vision/detections_3d` | `std_msgs/String` (JSON) | 每帧检测结果（持续发布） |
| `/cade/task_status` | `std_msgs/String` (JSON) | 任务执行结果（按需发布） |
| `/object_3d_position` | `geometry_msgs/PointStamped` | 目标 3D 位置（兼容旧版） |

### 任务指令格式

向 `/cade/task_cmd` 发布 JSON：

```json
{
  "action": "find_person",
  "target": "person",
  "timeout": 30.0,
  "attributes": {
    "posture": "sitting",
    "gesture": "waving"
  }
}
```

支持的任务类型：

| action | 说明 | 可选 attributes |
|--------|------|-----------------|
| `find_person` | 搜索人物，返回最匹配的 1 个 | posture, gesture |
| `find_object` | 搜索物体 | — |
| `count_people` | 统计人物数量 | posture, gesture |
| `count_objects` | 统计物体数量 | — |
| `get_person_info` | 返回当前帧所有检测 | — |

### 检测结果格式

节点持续发布到 `/vision/detections_3d`：

```json
{
  "type": "object_detection",
  "name": "person",
  "confidence": 0.95,
  "position_3d": [0.15, -0.23, 0.85],
  "bbox": [120, 80, 400, 600],
  "posture": "standing",
  "gesture": "waving",
  "landmarks": [["x,y,score", ...], ...]
}
```

`position_3d` 仅在 RealSense 模式下有效，其余模式为 `null`。

### 任务状态格式

任务完成后发布到 `/cade/task_status`：

```json
{"status": "SUCCESS", "result": { ... }}
{"status": "FAILED", "error": "Target 'apple' not found"}
```

---

## 架构

```
                 /cade/task_cmd (ROS String/JSON)
                          |
                   OpenVisionNode
         +-------------+----+------------+
         |             |       |            |
    Image Capture   YOLO   MediaPipe    CadeTracker
    (RealSense/     World   Pose+Hands  (跨帧追踪)
     file/USB cam)          (并行线程池)
         |             |       |            |
         +------+------+-------+------------+
                |
          Analyzer (按 posture/gesture 过滤)
                |
         +------+-------+
         |              |
   /vision/detections_3d  /cade/task_status
```

### 模块

| 文件 | 职责 |
|------|------|
| `scripts/open_vision_node.py` | 主节点：帧循环、ROS 通信、任务调度 |
| `src/cade_vision/posture_gesture.py` | MediaPipe Pose + Hands，姿态/手势分类，时序挥手检测 |
| `src/cade_vision/analyzer.py` | 按属性代价顺序过滤检测结果 |
| `src/cade_vision/tracker.py` | 跨帧追踪：3D 距离 → IOU → 匈牙利匹配 |

### 手势检测能力

**静态手势**：`raising_left_arm` / `raising_right_arm` / `raising_both_arms` / `pointing_left` / `pointing_right` / `pointing_both` / `none`

**时序挥手** (RingBuffer 30 帧)：两路 OR 检测
- Rule A: 前臂摆动方差 > `T_FOREARM`（默认 300 deg²）
- Rule B: 手掌方向方差 > `T_WRIST`（默认 500 deg²）
- 辅助条件: 食指不低于肩膀过多

---

## 手动测试

详见 [TEST_GUIDE.md](./TEST_GUIDE.md)。快速流程：

```bash
# 终端 1：启动 roscore
roscore

# 终端 2：启动 vision node（视频循环慢播）
rosrun cade_vision open_vision_node.py \
    --image-source file --image-path test.mp4 \
    --model yolov8x-worldv2.pt --device cpu \
    --loop --playback-speed 0.2 --display

# 终端 3：发送任务指令
rostopic pub /cade/task_cmd std_msgs/String \
    "data: '{\"action\": \"find_person\", \"target\": \"person\"}'" -1

# 终端 4：查看检测结果
rostopic echo /vision/detections_3d

# 终端 5：查看任务状态
rostopic echo /cade/task_status
```

## ROS 环境常见问题

`source /opt/ros/noetic/setup.bash` 报 `no such file or directory`：

如果你的 ROS 是通过 **conda/mamba（robostack）** 安装的，激活 conda 环境后 ROS 命令已自动可用，**不需要额外 source**。验证：

```bash
conda activate CADE
which roscore   # 显示 conda 环境内路径 = OK
```

如果不是 conda 安装，用 `find / -name "setup.bash" -path "*/ros/*"` 找到实际路径。

---

## License

MIT
