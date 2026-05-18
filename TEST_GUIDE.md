# CADE Vision Node 独立测试指南

用视频文件替代摄像头，手动发布 ROS 话题来验证 `open_vision_node.py` 的端到端功能。

---

## 前置条件

- ROS 已安装且 workspace 已编译（`catkin_make` 无报错）
- CADE conda 环境 `/home/huyanshen/miniforge3/envs/CADE/bin/python`
- 一个测试视频文件（如 `tests/data/videos/test_video_waving.mp4`）
- YOLO-World 模型（`models/yolov8x-worldv2.pt`）

## 环境激活

ROS 的 setup 文件路径取决于你的 ROS 发行版和安装方式。**如果 `/opt/ros/noetic/setup.xxx` 不存在**，说明 ROS 不在 `/opt/ros/noetic/` 下。先找到它：

```bash
# 搜索 ROS 安装位置
find / -name "setup.bash" -path "*/ros/*" 2>/dev/null
```

常见的非标准安装路径包括：

| 情况 | 正确路径示例 |
|------|-------------|
| 通过 conda/mamba 装的 ROS | `$CONDA_PREFIX/setup.bash` |
| 自己编译的 ROS | `/home/xxx/ros_catkin_ws/install/setup.bash` |
| Docker 容器里的 ROS | `/ros_entrypoint.sh` |
| robostack (conda-forge ROS) | `$CONDA_PREFIX/setup.sh` |

找到后记下路径。激活 CADE workspace 的正确顺序：

```bash
# 1. 激活 conda 环境（如果需要）
conda activate CADE

# 2. source ROS 环境（用你找到的实际路径替换下面这行）
source <你的 ROS setup.bash 路径>

# 3. source CADE workspace
cd ~/HysProjects/CADE/cade_ws
source devel/setup.bash
```

> **常见错误说明**：如果你运行 `source /opt/ros/noetic/setup.bash` 报 `no such file or directory`，
> 说明 ROS **根本没装在** `/opt/ros/noetic/` 下。官方 apt 安装才会在那个路径，conda/mamba 安装
> 的 ROS 环境在 conda 目录内，不需要额外 source（激活 conda 环境后 ROS 命令自动可用）。

---

## 步骤 1：启动 roscore

```bash
# 终端 1：先确保环境激活
conda activate CADE
# 如果 ROS 通过 conda 安装在 CADE 环境中，直接：
roscore
# 如果 roscore 找不到，检查 ROS 是否正确 source
```

---

## 步骤 2：启动 Vision Node

```bash
# 终端 2
conda activate CADE
cd ~/HysProjects/CADE/cade_ws
source devel/setup.bash

rosrun cade_vision open_vision_node.py \
    --image-source file \
    --image-path ~/HysProjects/CADE/tests/data/videos/test_video_waving.mp4 \
    --model ~/HysProjects/CADE/models/yolov8x-worldv2.pt \
    --device cpu \
    --display \
    --loop \
    --playback-speed 0.2
```

参数说明：

| 参数 | 含义 |
|------|------|
| `--image-source file` | 从文件读取（自动识别图片/视频） |
| `--image-path ...` | 图片或视频文件路径 |
| `--model ...` | YOLO-World 模型路径 |
| `--device cpu` | CPU 推理（无 GPU 时用） |
| `--display` | 显示检测窗口（按 `q` 退出） |
| `--loop` | 视频播完自动从头循环 |
| `--playback-speed SPEED` | 播放速率：`1.0`=原速，`0.2`=0.2x 慢速。不加=满速 |

> **推荐**：测试时用 `--loop --playback-speed 0.2`，视频循环慢播，给你充足时间在终端 3 里手动 `rostopic pub` 发指令。

启动后应看到：

```
OpenVisionNode initialized
  Image Source: file
  YOLO: Available
  MediaPipe: Available
  Analyzer: Available
  ROS: Available
File source (video): ...  464x848, 30fps, 231 frames
CADE Vision Node running...
Listening on /cade/task_cmd
Publishing detections to /vision/detections_3d
```

---

## 步骤 3：手动发送任务指令

用 `rostopic pub` 向 `/cade/task_cmd` 发布 JSON 指令。

### 3.1 基础查找任务

```bash
# 终端 3：查找 person
rostopic pub /cade/task_cmd std_msgs/String "data: '{\"action\": \"find_person\", \"target\": \"person\"}'" -1
```

### 3.2 带属性过滤（posture / gesture）

```bash
# 查找坐着的人
rostopic pub /cade/task_cmd std_msgs/String \
    "data: '{\"action\": \"find_person\", \"target\": \"person\", \"attributes\": {\"posture\": \"sitting\"}}'" -1

# 查找挥手的人
rostopic pub /cade/task_cmd std_msgs/String \
    "data: '{\"action\": \"find_person\", \"target\": \"person\", \"attributes\": {\"gesture\": \"waving\"}}'" -1
```

### 3.3 查找物体

```bash
# 查找 apple
rostopic pub /cade/task_cmd std_msgs/String "data: '{\"action\": \"find_object\", \"target\": \"apple\"}'" -1
```

### 3.4 计数任务

```bash
# 统计 person 数量
rostopic pub /cade/task_cmd std_msgs/String "data: '{\"action\": \"count_people\", \"category\": \"person\"}'" -1
```

---

## 步骤 4：监控输出话题

### 4.1 监听 3D 检测结果

```bash
# 终端 4
rostopic echo /vision/detections_3d
```

正常输出示例：

```json
{
  "type": "object_detection",
  "name": "person",
  "confidence": 0.95,
  "position_3d": null,
  "bbox": [222, 406, 345, 860]
}
```

### 4.2 监听任务状态

```bash
# 终端 5
rostopic echo /cade/task_status
```

成功输出示例：

```json
{"status": "SUCCESS", "result": {"type": "object_detection", "name": "person", ...}}
```

超时输出示例：

```json
{"status": "FAILED", "error": "Target 'apple' not found"}
```

---

## 步骤 5：验证要点

| 检查项 | 预期行为 |
|--------|----------|
| `rostopic pub` 发送后 | 终端 2 打印 `[Vision Task] find_person: ...` |
| 检测到目标 | `/cade/task_status` 收到 `SUCCESS` |
| 未检测到目标 | 30 秒后 `/cade/task_status` 收到 `FAILED` |
| display 窗口 | 显示检测框 + 姿态/手势标签 + 手部关键点 |
| `/vision/detections_3d` | 每帧持续发布检测结果 JSON |

---

## 步骤 6：停止

```bash
# 终端 2：按 q 退出 display 窗口，或 Ctrl+C
# 终端 1：Ctrl+C 停 roscore
```

---

## 常见问题

**Q: `source /opt/ros/noetic/setup.bash` 报 `no such file or directory`？**

A: ROS 不在 `/opt/ros/noetic/` 下。如果你的 ROS 是通过 conda/mamba 安装的（robostack），
激活 conda 环境后 ROS 命令就已自动可用，不需要额外 `source`。验证方法：

```bash
conda activate CADE
which roscore    # 如果显示 conda 环境内的路径，说明 ROS 已就绪
roscore          # 能启动就说明没问题
```

如果不是 conda 安装的，用 `find / -name "setup.bash" -path "*/ros/*" 2>/dev/null` 找到实际路径。

**Q: `pyrealsense2` 报错但没用 realsense？**
A: 确认 `--image-source file` 已指定，节点会自动跳过 RealSense 初始化。

**Q: MediaPipe 报错无法加载？**
A: 确认 conda 环境中 `mediapipe` 已安装：
```bash
/home/huyanshen/miniforge3/envs/CADE/bin/pip list | grep mediapipe
```

**Q: 视频播完不想重启节点？**
A: 加 `--loop` 参数，视频播完自动从头循环，同时重置追踪器和 RingBuffer。

---

## 快捷测试脚本

也可以直接用 Python 发送 ROS 消息：

```python
#!/usr/bin/env python3
"""快捷测试：向 /cade/task_cmd 发送 find_person 指令"""
import rospy
from std_msgs.msg import String
import json

rospy.init_node('test_vision', anonymous=True)
pub = rospy.Publisher('/cade/task_cmd', String, queue_size=10)
rospy.sleep(0.5)  # 等 publisher 注册

cmd = json.dumps({"action": "find_person", "target": "person"})
pub.publish(String(data=cmd))
print(f"Sent: {cmd}")

# 等结果
try:
    msg = rospy.wait_for_message('/cade/task_status', String, timeout=30)
    print(f"Received: {msg.data}")
except rospy.ROSException:
    print("Timeout: no response")
```
