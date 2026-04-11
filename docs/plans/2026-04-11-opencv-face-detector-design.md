# OpenCV Haar 人脸检测替代 MediaPipe 设计

## 背景

MediaPipe 官方不提供 linux/arm64 (aarch64) 预编译包，Raspberry Pi 5 (Debian 13 aarch64) 上 FaceDetector/HandTracker/ObjectDetector 均运行在 NoOp 模式。使用 OpenCV Haar 级联替代 MediaPipe 实现人脸检测，同时移除手势识别功能。

## 方案

### FaceDetector — OpenCV Haar 级联

- 使用 `cv2.CascadeClassifier` + OpenCV 内置 `haarcascade_frontalface_default.xml`
- 保持公共接口完全不变：`detect(frame)` 返回格式、`FaceInfo` 数据类、`presence_callback` 防抖逻辑
- OpenCV 不可用时 NoOp 降级

### HandTracker — 始终 NoOp

- 移除 MediaPipe 依赖，`_noop = True`
- 保持接口兼容，调用方无需改动

### ObjectDetector — 始终 NoOp

- 移除 MediaPipe 依赖，`_noop = True`
- 保持接口兼容

### 依赖变更

- `pyproject.toml` vision extra：移除 mediapipe，opencv-python-headless 保留（已是 Pi 上的依赖）
- `__init__.py`：更新模块文档

## 改动文件

| 文件 | 操作 |
|------|------|
| `lelamp/edge/face_detector.py` | 重写：OpenCV Haar 替代 MediaPipe |
| `lelamp/edge/hand_tracker.py` | 简化：移除 MediaPipe，始终 NoOp |
| `lelamp/edge/object_detector.py` | 简化：移除 MediaPipe，始终 NoOp |
| `lelamp/edge/__init__.py` | 更新模块文档 |
| `pyproject.toml` | vision extra 移除 mediapipe |

## 不改动

- `hybrid_vision.py` — 接口不变
- `proactive_vision_monitor.py` — 接口不变
- `edge_vision_tools.py` — 接口不变
- `lelamp_agent.py` — 无需改动
