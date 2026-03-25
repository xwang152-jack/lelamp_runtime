# 边缘视觉模型下载与配置指南

**最后更新**: 2026-03-25
**适用版本**: LeLamp Runtime v0.1.0

---

## 📋 目录

1. [模型说明](#模型说明)
2. [下载模型](#下载模型)
3. [配置模型](#配置模型)
4. [验证安装](#验证安装)
5. [常见问题](#常见问题)

---

## 模型说明

LeLamp 边缘视觉使用 **MediaPipe** 和 **EfficientDet-Lite** 模型进行本地 AI 推理。

### 支持的功能

| 功能 | 模型 | 延迟 | 用途 |
|------|------|------|------|
| **人脸检测** | MediaPipe Face Detection | < 50ms | 用户在场检测 |
| **手势追踪** | MediaPipe Hands | < 100ms | 手势控制 |
| **物体检测** | EfficientDet-Lite0 | < 300ms | 物体识别（80类COCO） |

### 模型文件

- **人脸检测**: 内置于 MediaPipe，无需额外下载
- **手势追踪**: 内置于 MediaPipe，无需额外下载
- **物体检测**: 需要下载 `efficientdet_lite0.tflite` 模型文件

### 平台支持

| 平台 | 架构 | MediaPipe 支持 | 物体检测 |
|------|------|----------------|----------|
| macOS | ARM64 (Apple Silicon) | ✅ | ✅ |
| Linux | x86_64 | ✅ | ✅ |
| Windows | AMD64 | ✅ | ✅ |
| **Raspberry Pi** | **ARM64/ARMv7** | ❌ | ❌ |

---

## 下载模型

### 方法 1: 自动下载（推荐）

LeLamp 启动时会自动检测模型是否已下载，如果未下载会提示下载地址。

### 方法 2: 手动下载

#### 步骤 1: 下载物体检测模型

```bash
# 创建模型目录
mkdir -p ~/lelamp_runtime/models

# 下载 EfficientDet-Lite0 模型
cd ~/lelamp_runtime/models
curl -L -o efficientdet_lite0.tflite \
  "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite"
```

#### 步骤 2: 验证下载

```bash
# 检查文件大小（应该约为 20MB）
ls -lh efficientdet_lite0.tflite

# 计算文件 SHA256（可选）
shasum efficientdet_lite0.tflite
```

**预期文件大小**: 约 20MB (19.8 MB - 21.5 MB)

### 方法 3: 使用 Python 脚本下载

创建下载脚本：

```bash
#!/bin/bash
# download_models.sh

MODEL_DIR="${1:-$HOME/lelamp_runtime/models}"
MODEL_URL="https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite"
MODEL_FILE="$MODEL_DIR/efficientdet_lite0.tflite"

# 创建目录
mkdir -p "$MODEL_DIR"

# 下载模型
echo "正在下载 EfficientDet-Lite0 模型..."
curl -L -o "$MODEL_FILE" "$MODEL_URL"

# 验证下载
if [ -f "$MODEL_FILE" ]; then
    SIZE=$(du -h "$MODEL_FILE" | cut -f1)
    echo "✅ 模型下载成功: $MODEL_FILE ($SIZE)"
else
    echo "❌ 模型下载失败"
    exit 1
fi
```

使用方法：

```bash
chmod +x download_models.sh
./download_models.sh
```

---

## 配置模型

### 步骤 1: 设置环境变量

编辑 `.env` 文件：

```bash
# 启用边缘视觉
LELAMP_EDGE_VISION_ENABLED=true

# 模型目录（默认值，通常不需要修改）
LELAMP_EDGE_VISION_MODEL_DIR=/home/pi/lelamp_runtime/models

# 置信度阈值（0.0-1.0，默认 0.5）
LELAMP_EDGE_VISION_CONFIDENCE_THRESHOLD=0.5

# 帧率限制（默认 15 FPS）
LELAMP_EDGE_VISION_FPS=15

# 缓冲区大小（默认 1）
LELAMP_EDGE_VISION_BUFFER_SIZE=1
```

### 步骤 2: 放置模型文件

将下载的模型文件放置在以下位置之一（按优先级排序）：

```bash
# 1. 项目 models 目录（推荐）
~/lelamp_runtime/models/efficientdet_lite0.tflite

# 2. 用户主目录
~/.mediapipe/models/efficientdet_lite0.tflite

# 3. 临时目录
/tmp/mediapipe/efficientdet_lite0.tflite
```

### 步骤 3: 设置文件权限

```bash
# 确保模型文件可读
chmod 644 ~/lelamp_runtime/models/efficientdet_lite0.tflite
```

---

## 验证安装

### 方法 1: 运行测试

```bash
cd ~/lelamp_runtime

# 运行边缘视觉测试
uv run pytest tests/test_edge_vision.py -v
```

**预期输出**:
```
===== test session starts =====
collected 25 items

test_edge_vision.py::TestFaceDetector::test_init_no_mediapipe PASSED
test_edge_vision.py::TestHandTracker::test_init_no_mediapipe PASSED
test_edge_vision.py::TestObjectDetector::test_init_no_mediapipe PASSED
...
===== 25 passed in 2.34s =====
```

### 方法 2: Python 交互测试

```bash
cd ~/lelamp_runtime

# 启动 Python REPL
uv run python

# 在 Python 中测试
>>> from lelamp.edge.object_detector import ObjectDetector
>>> detector = ObjectDetector()
>>> stats = detector.get_stats()
>>> print(stats)
{'noop_mode': False, 'total_detections': 0, ...}
```

### 方法 3: 查看 Agent 日志

启动 LeLamp Agent 后，查看日志：

```bash
# 查看边缘视觉初始化日志
tail -f /var/log/lelamp.log | grep -E "EdgeVision|ObjectDetector"
```

**预期日志**:
```
INFO:lelamp.edge.object:ObjectDetector initialized (MediaPipe mode)
INFO:lelamp.edge.object:ObjectDetector using tasks API with model: /home/pi/lelamp_runtime/models/efficientdet_lite0.tflite
INFO:lelamp.edge.hybrid:HybridVisionService initialized with face=True, hand=True, object=True
```

---

## 常见问题

### Q1: 模型下载失败？

**解决方案**:

1. **使用镜像源**:
   ```bash
   # 尝试使用镜像（如果有）
   curl -L -o efficientdet_lite0.tflite \
     "https://ghproxy.com/https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite"
   ```

2. **手动下载**:
   - 在浏览器中访问: https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite
   - 下载后通过 SCP 传输到树莓派:
     ```bash
     scp efficientdet_lite0.tflite pi@192.168.1.100:~/lelamp_runtime/models/
     ```

### Q2: 模型文件损坏？

**检查方法**:
```bash
# 检查文件大小（应该约为 20MB）
ls -lh efficientdet_lite0.tflite

# 如果文件太小（< 10MB），可能是下载不完整，重新下载
```

### Q3: MediaPipe 不可用？

**错误信息**:
```
WARNING: MediaPipe not available, ObjectDetector will run in NoOp mode
```

**解决方案**:

1. **安装 MediaPipe**:
   ```bash
   # 标准平台
   uv sync --extra vision

   # 树莓派（需要手动编译）
   # 参考 FEATURES.md 中的手动安装方法
   ```

2. **检查 Python 版本**:
   ```bash
   # MediaPipe 需要 Python 3.8+
   python --version
   ```

3. **系统会自动降级**:
   - 即使 MediaPipe 不可用，系统仍能正常工作
   - 自动降级到 NoOp 模式
   - 回退到云端 Qwen VL 视觉识别

### Q4: 模型检测不到物体？

**检查清单**:

1. ✅ 模型文件已下载且完整（~20MB）
2. ✅ 环境变量 `LELAMP_EDGE_VISION_ENABLED=true`
3. ✅ 模型路径正确
4. ✅ 置信度阈值设置合理（默认 0.5）
5. ✅ 光线充足，物体清晰可见

**调试方法**:
```python
# 测试物体检测
from lelamp.edge.object_detector import ObjectDetector
import cv2

detector = ObjectDetector(score_threshold=0.3)  # 降低阈值
frame = cv2.imread("test_image.jpg")
result = detector.detect(frame)
print(result["summary"])
```

### Q5: Raspberry Pi 上无法使用？

**现状**:
- MediaPipe 官方不支持 ARM 架构（Raspberry Pi）
- 无法直接通过 `pip install mediapipe` 安装

**替代方案**:

1. **使用云端视觉识别**（推荐）
   - 设置 `LELAMP_EDGE_VISION_ENABLED=false`
   - 自动使用 Qwen VL 云端视觉
   - 功能完整，延迟略高（3-8秒）

2. **手动编译 MediaPipe**（高级）
   - 参考 MediaPipe GitHub 文档
   - 需要交叉编译环境
   - 复杂度高，不推荐

3. **使用其他边缘推理框架**
   - OpenCV DNN + TFLite
   - NCNN（腾讯开源）
   - 需要自行集成

### Q6: 内存不足？

**症状**: 系统变慢或崩溃

**解决方案**:

1. **降低 FPS**:
   ```bash
   LELAMP_EDGE_VISION_FPS=10  # 默认 15
   ```

2. **减小缓冲区**:
   ```bash
   LELAMP_EDGE_VISION_BUFFER_SIZE=0  # 默认 1
   ```

3. **仅启用必要功能**:
   ```python
   # 在代码中配置
   service = HybridVisionService(
       enable_face=True,   # 需要
       enable_hand=False,  # 禁用
       enable_object=False  # 禁用
   )
   ```

---

## 性能优化

### 降低延迟

```bash
# 降低分辨率
LELAMP_CAMERA_WIDTH=640
LELAMP_CAMERA_HEIGHT=480

# 降低 FPS
LELAMP_EDGE_VISION_FPS=10

# 提高置信度阈值（减少误检）
LELAMP_EDGE_VISION_CONFIDENCE_THRESHOLD=0.7
```

### 提高准确率

```bash
# 提高分辨率
LELAMP_CAMERA_WIDTH=1280
LELAMP_CAMERA_HEIGHT=720

# 降低置信度阈值（检测更多物体）
LELAMP_EDGE_VISION_CONFIDENCE_THRESHOLD=0.3

# 提高 FPS（更快响应）
LELAMP_EDGE_VISION_FPS=20
```

---

## 相关文档

- [FEATURES.md - 边缘推理功能](FEATURES.md)
- [ARCHITECTURE.md - 系统架构](ARCHITECTURE.md)
- [SETUP_GUIDE.md - 完整设置指南](SETUP_GUIDE.md)
- [USER_GUIDE_QUICK.md - 用户快速指南](USER_GUIDE_QUICK.md)

---

**文档版本**: v0.1.0
**最后更新**: 2026-03-25
**作者**: LeLamp 开发团队
