# 树莓派边缘视觉测试指南

**适用版本**: LeLamp Runtime v0.1.0
**测试平台**: Raspberry Pi 4B+
**最后更新**: 2026-03-25

---

## 📋 测试前准备

### 1. SSH 连接到树莓派

```bash
# 替换为您的树莓派 IP 地址
ssh pi@192.168.1.100
```

### 2. 进入项目目录

```bash
cd ~/lelamp_runtime
```

### 3. 确认项目状态

```bash
# 查看项目结构
ls -la

# 检查 Python 版本（需要 3.12+）
python3 --version

# 检查虚拟环境
ls -la .venv/
```

---

## 🚀 快速测试（推荐）

### 使用自动化测试脚本

```bash
cd ~/lelamp_runtime

# 运行边缘视觉测试脚本
bash scripts/test_edge_vision_on_pi.sh
```

**脚本会自动测试**：
- ✅ MediaPipe 安装状态
- ✅ OpenCV 安装状态
- ✅ 边缘视觉模块导入
- ✅ 人脸检测器初始化
- ✅ 手势追踪器初始化
- ✅ 物体检测器初始化
- ✅ 混合视觉服务初始化
- ✅ 图像处理功能

**预期输出**:

```
======================================
  LeLamp 边缘视觉功能测试
======================================

📋 系统信息:
   平台: aarch64
   系统: Linux
   Python: Python 3.12.0

📁 项目目录: /home/pi/lelamp_runtime

======================================
  测试 1: 检查 MediaPipe 安装
======================================

✅ MediaPipe 已安装
   版本: 0.10.9

======================================
  测试 2: 检查 OpenCV 安装
======================================

✅ OpenCV 已安装
   版本: 4.9.0

======================================
  测试 3: 测试边缘视觉模块
======================================

正在导入边缘视觉模块...
✅ 所有边缘视觉模块导入成功

==================================================
测试人脸检测器
==================================================
人脸检测器状态: NoOp 模式
总检测次数: 0

==================================================
测试手势追踪器
==================================================
手势追踪器状态: NoOp 模式
总追踪次数: 0

==================================================
测试物体检测器
==================================================
物体检测器状态: NoOp 模式
总检测次数: 0

==================================================
测试混合视觉服务
==================================================
混合视觉服务状态: 正常
...
```

---

## 🧪 手动测试步骤

如果自动化脚本无法运行，可以手动测试：

### 步骤 1: 测试 MediaPipe 安装

```bash
cd ~/lelamp_runtime

# 测试 MediaPipe 导入
uv run python -c "import mediapipe; print('MediaPipe version:', mediapipe.__version__)"
```

**预期结果**:
- ✅ 成功: 显示 MediaPipe 版本号
- ❌ 失败: `ImportError: No module named 'mediapipe'`

**如果失败**:
```bash
# 安装 MediaPipe
uv sync --extra vision

# 或手动安装
uv pip install mediapipe opencv-python-headless
```

### 步骤 2: 测试边缘视觉模块

```bash
# 测试人脸检测器
uv run python -c "
from lelamp.edge.face_detector import FaceDetector
detector = FaceDetector()
print('FaceDetector:', detector.get_stats())
"

# 测试手势追踪器
uv run python -c "
from lelamp.edge.hand_tracker import HandTracker
tracker = HandTracker()
print('HandTracker:', tracker.get_stats())
"

# 测试物体检测器
uv run python -c "
from lelamp.edge.object_detector import ObjectDetector
detector = ObjectDetector()
print('ObjectDetector:', detector.get_stats())
"
```

### 步骤 3: 测试图像处理

```bash
uv run python << 'EOF'
import numpy as np
import cv2
from lelamp.edge.object_detector import ObjectDetector

# 创建测试图像
frame = np.zeros((480, 640, 3), dtype=np.uint8)

# 画一个红色的圆
cv2.circle(frame, (320, 240), 100, (0, 0, 255), -1)

# 测试物体检测
detector = ObjectDetector()
result = detector.detect(frame)

print("检测结果:", result['summary'])
print("检测数量:", result['count'])
print("检测到的物体:", result['labels'])
EOF
```

### 步骤 4: 测试混合视觉服务

```bash
uv run python << 'EOF'
from lelamp.edge.hybrid_vision import HybridVisionService

# 初始化混合视觉服务
service = HybridVisionService(
    enable_face=True,
    enable_hand=True,
    enable_object=True,
)

# 获取统计信息
stats = service.get_stats()
print("统计信息:", stats)

# 测试查询复杂度分析
complexity = service.analyze_query("这是什么")
print("查询复杂度:", complexity)
EOF
```

---

## 📊 结果解读

### 情况 1: MediaPipe 不可用（NoOp 模式）

**症状**:
```
人脸检测器状态: NoOp 模式
手势追踪器状态: NoOp 模式
物体检测器状态: NoOp 模式
```

**原因**:
- Raspberry Pi ARM 架构官方不支持 MediaPipe
- MediaPipe 未正确安装

**解决方案**:
1. **使用云端视觉识别**（推荐）:
   ```bash
   # 在 .env 中禁用边缘视觉
   LELAMP_EDGE_VISION_ENABLED=false
   ```

2. **尝试手动编译 MediaPipe**（高级用户）:
   - 参考 MediaPipe GitHub 文档
   - 需要交叉编译环境
   - 复杂度高，不推荐

3. **系统会自动降级**:
   - 检测到 MediaPipe 不可用
   - 自动切换到 NoOp 模式
   - 使用云端 Qwen VL 视觉识别
   - 功能完全正常

### 情况 2: MediaPipe 可用（正常模式）

**症状**:
```
人脸检测器状态: 正常模式
手势追踪器状态: 正常模式
物体检测器状态: 正常模式
```

**下一步**:
```bash
# 1. 启用边缘视觉
nano .env

# 添加或修改以下配置
LELAMP_EDGE_VISION_ENABLED=true

# 2. 重启服务
sudo systemctl restart lelamp-*

# 3. 查看日志
tail -f /var/log/lelamp.log | grep -E "EdgeVision|MediaPipe"
```

---

## 🔍 详细诊断

### 查看 MediaPipe 安装详情

```bash
# 查看 MediaPipe 包信息
uv run pip show mediapipe

# 查看 MediaPipe 文件位置
uv run python -c "import mediapipe; import os; print(os.path.dirname(mediapipe.__file__))"
```

### 查看系统信息

```bash
# 系统架构
uname -m

# Python 信息
uv run python --version
uv run python -c "import sys; print('Python path:', sys.path)"

# 依赖检查
uv run pip list | grep -E "mediapipe|opencv"
```

### 测试摄像头

```bash
# 测试摄像头是否正常
uv run python << 'EOF'
import cv2

# 打开摄像头（0 通常是 /dev/video0）
cap = cv2.VideoCapture(0)

if cap.isOpened():
    print("✅ 摄像头正常")
    ret, frame = cap.read()
    if ret:
        print(f"✅ 图像采集成功: {frame.shape}")
    else:
        print("❌ 无法采集图像")
    cap.release()
else:
    print("❌ 摄像头无法打开")
EOF
```

---

## 📝 测试清单

完成以下测试项，确认边缘视觉功能状态：

- [ ] MediaPipe 已安装
- [ ] OpenCV 已安装
- [ ] 人脸检测器初始化成功
- [ ] 手势追踪器初始化成功
- [ ] 物体检测器初始化成功
- [ ] 混合视觉服务初始化成功
- [ ] 图像处理功能正常
- [ ] 摄像头采集正常

---

## 🚨 常见问题

### Q1: ImportError: No module named 'mediapipe'

**解决方案**:
```bash
# 安装 vision 依赖
uv sync --extra vision

# 或手动安装
uv pip install mediapipe
```

### Q2: RuntimeError: MediaPipe not available

**原因**: Raspberry Pi ARM 架构不支持

**解决方案**:
```bash
# 禁用边缘视觉，使用云端识别
LELAMP_EDGE_VISION_ENABLED=false
```

### Q3: 摄像头无法打开

**解决方案**:
```bash
# 检查摄像头设备
ls -l /dev/video*

# 添加用户到 video 组
sudo usermod -a -G video $USER

# 重新登录生效
sudo reboot
```

### Q4: 测试脚本无法运行

**解决方案**:
```bash
# 确保脚本有执行权限
chmod +x scripts/test_edge_vision_on_pi.sh

# 确保在项目目录中运行
cd ~/lelamp_runtime
bash scripts/test_edge_vision_on_pi.sh
```

---

## ✅ 测试完成后

### 如果测试成功（MediaPipe 可用）

```bash
# 1. 启用边缘视觉
nano .env
# 添加: LELAMP_EDGE_VISION_ENABLED=true

# 2. 配置模型目录（可选）
# LELAMP_EDGE_VISION_MODEL_DIR=/home/pi/lelamp_runtime/models

# 3. 重启服务
sudo systemctl restart lelamp-*

# 4. 验证日志
tail -f /var/log/lelamp.log | grep -E "EdgeVision|MediaPipe"
```

### 如果测试失败（MediaPipe 不可用）

```bash
# 1. 确认使用云端视觉识别
nano .env
# 确保: LELAMP_EDGE_VISION_ENABLED=false

# 2. 重启服务
sudo systemctl restart lelamp-*

# 3. 系统会自动使用云端 Qwen VL 视觉识别
# 功能完全正常，无需担心
```

---

## 📚 相关文档

- [边缘视觉配置指南](EDGE_VISION_SETUP.md)
- [边缘视觉快速参考](EDGE_VISION_QUICK_REF.md)
- [边缘视觉模型信息](EDGE_VISION_MODEL_INFO.md)
- [功能特性说明](FEATURES.md)

---

**测试脚本**: `scripts/test_edge_vision_on_pi.sh`
**文档版本**: v0.1.0
**最后更新**: 2026-03-25
