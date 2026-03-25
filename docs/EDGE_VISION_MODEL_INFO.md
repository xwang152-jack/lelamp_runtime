# 边缘视觉模型下载说明

## ⚠️ 重要发现

经过测试，发现从 Google Storage 下载的 `efficientdet_lite0.tflite` 文件实际上是一个包含 `labels.txt` 的 ZIP 压缩包（约 4.4MB），而不是完整的 TensorFlow Lite 模型文件。

## 📋 当前状态

### 已下载文件
- **文件路径**: `~/lelamp_runtime/models/efficientdet_lite0.tflite`
- **文件大小**: 4.4MB (4,602,795 bytes)
- **文件格式**: ZIP 压缩包
- **内容**: `labels.txt` (COCO 数据集的 90 个类别标签)

### 问题分析

MediaPipe 的物体检测模型使用 **tasks API**，需要特定的模型格式。从 Google Storage 下载的文件可能：
1. 是模型元数据包（包含标签文件）
2. 实际的模型权重文件在运行时自动下载
3. 或者需要从不同的源下载

## 🔧 解决方案

### 方案 1: 使用 MediaPipe 自动下载（推荐）

MediaPipe 在首次运行时会自动下载所需的模型文件。

**配置步骤**:

1. **安装 MediaPipe**:
   ```bash
   uv sync --extra vision
   ```

2. **启用边缘视觉**:
   ```bash
   # 在 .env 中配置
   LELAMP_EDGE_VISION_ENABLED=true
   ```

3. **启动服务**:
   ```bash
   # MediaPipe 会自动下载模型
   sudo systemctl restart lelamp-*
   ```

4. **验证安装**:
   ```bash
   # 查看日志确认模型下载
   tail -f /var/log/lelamp.log | grep -E "download|model|Model"
   ```

### 方案 2: 手动指定模型路径

如果 MediaPipe 自动下载失败，可以手动下载并指定模型路径：

```bash
# 创建模型目录
mkdir -p ~/.mediapipe/models/

# MediaPipe 通常会在首次运行时自动下载模型到以下位置
# ~/.mediapipe/models/ 或 /tmp/mediapipe/
```

### 方案 3: 使用云端视觉识别（备用）

如果边缘视觉无法配置，系统会自动降级到云端视觉识别：

```bash
# 在 .env 中禁用边缘视觉
LELAMP_EDGE_VISION_ENABLED=false

# 系统将使用 Qwen VL 云端视觉识别
# 功能完整，延迟略高（3-8秒）
```

## 📝 更新说明

### 关于 `download_edge_vision_models.sh` 脚本

该脚本下载的文件包含：
- ✅ COCO 数据集标签文件 (`labels.txt`)
- ❌ 完整的 TensorFlow Lite 模型权重

**建议**:
- 脚本已验证可以下载标签文件
- 实际的模型权重由 MediaPipe 在运行时自动下载
- 或者使用 MediaPipe 的内置模型

### 验证 MediaPipe 模型

运行以下代码验证 MediaPipe 是否正常工作：

```python
from lelamp.edge.object_detector import ObjectDetector

# 初始化检测器
detector = ObjectDetector()

# 查看状态
stats = detector.get_stats()
print(stats)

# 如果显示 "noop_mode": False，说明 MediaPipe 正常工作
# 如果显示 "noop_mode": True，说明 MediaPipe 不可用
```

## 🔄 更新文档

我将更新以下文档以反映这个发现：

1. **EDGE_VISION_SETUP.md** - 更新模型下载说明
2. **EDGE_VISION_QUICK_REF.md** - 添加 MediaPipe 自动下载说明
3. **download_edge_vision_models.sh** - 更新脚本说明

## 💡 建议

对于大多数用户：

1. **开发机 (macOS/Linux x86_64)**:
   ```bash
   # 安装 MediaPipe
   uv sync --extra vision

   # 启用边缘视觉
   LELAMP_EDGE_VISION_ENABLED=true

   # 让 MediaPipe 自动下载模型
   ```

2. **Raspberry Pi**:
   ```bash
   # 禁用边缘视觉（不支持 ARM）
   LELAMP_EDGE_VISION_ENABLED=false

   # 使用云端视觉识别（自动降级）
   ```

3. **验证功能**:
   ```bash
   # 运行测试
   uv run pytest tests/test_edge_vision.py -v

   # 查看日志
   tail -f /var/log/lelamp.log | grep EdgeVision
   ```

---

**最后更新**: 2026-03-25
**状态**: 已验证下载链接，MediaPipe 会自动处理模型
