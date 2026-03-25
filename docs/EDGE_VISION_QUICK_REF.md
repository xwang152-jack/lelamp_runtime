# 边缘视觉模型配置 - 完整总结

## 📋 问题解答

### 1. 检测模型是什么？

LeLamp 使用 **EfficientDet-Lite0** 模型进行本地物体检测：
- **模型文件**: `efficientdet_lite0.tflite`
- **文件大小**: 约 20MB
- **功能**: 识别 80 类 COCO 物体（苹果、手机、杯子、书等）
- **延迟**: < 300ms

### 2. 模型怎么下载？

#### 方式 1: 使用下载脚本（推荐）

```bash
cd ~/lelamp_runtime
bash scripts/download_edge_vision_models.sh
```

#### 方式 2: 手动下载

```bash
mkdir -p ~/lelamp_runtime/models
cd ~/lelamp_runtime/models
curl -L -o efficientdet_lite0.tflite \
  "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite"
```

#### 方式 3: 浏览器下载

1. 访问: https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite
2. 下载文件
3. 通过 SCP 传输到树莓派:
   ```bash
   scp efficientdet_lite0.tflite pi@192.168.1.100:~/lelamp_runtime/models/
   ```

### 3. 模型放哪里？

将模型文件放置在以下任一位置（按优先级排序）：

```bash
# 1. 项目 models 目录（推荐）
~/lelamp_runtime/models/efficientdet_lite0.tflite

# 2. 用户主目录
~/.mediapipe/models/efficientdet_lite0.tflite

# 3. 临时目录
/tmp/mediapipe/efficientdet_lite0.tflite
```

### 4. 如何配置？

#### 步骤 1: 下载模型

```bash
bash scripts/download_edge_vision_models.sh
```

#### 步骤 2: 配置环境变量

编辑 `.env` 文件：

```bash
# 启用边缘视觉
LELAMP_EDGE_VISION_ENABLED=true

# 模型目录（可选，默认值）
LELAMP_EDGE_VISION_MODEL_DIR=/home/pi/lelamp_runtime/models

# 置信度阈值（0.0-1.0，默认 0.5）
LELAMP_EDGE_VISION_CONFIDENCE_THRESHOLD=0.5
```

#### 步骤 3: 重启服务

```bash
# 重启 LeLamp 服务
sudo systemctl restart lelamp-*
```

### 5. 如何验证？

#### 方法 1: 检查文件

```bash
# 检查文件是否存在
ls -lh ~/lelamp_runtime/models/efficientdet_lite0.tflite

# 应该显示约 20MB 大小
```

#### 方法 2: 运行测试

```bash
cd ~/lelamp_runtime
uv run pytest tests/test_edge_vision.py -v
```

#### 方法 3: 查看日志

```bash
tail -f /var/log/lelamp.log | grep -E "EdgeVision|ObjectDetector"
```

**预期日志**:
```
INFO:lelamp.edge.object:ObjectDetector initialized (MediaPipe mode)
INFO:lelamp.edge.object:ObjectDetector using tasks API with model: /home/pi/lelamp_runtime/models/efficientdet_lite0.tflite
```

### 6. Raspberry Pi 能用吗？

**❌ 官方不支持**

MediaPipe 在 Raspberry Pi (ARM 架构) 上官方不支持。

**替代方案**：

1. **使用云端视觉识别**（推荐）
   ```bash
   # 在 .env 中禁用边缘视觉
   LELAMP_EDGE_VISION_ENABLED=false
   ```
   - 自动使用 Qwen VL 云端视觉
   - 功能完整，延迟略高（3-8秒）

2. **手动编译 MediaPipe**（高级）
   - 参考 MediaPipe GitHub
   - 需要交叉编译环境
   - 复杂度高，不推荐

3. **系统会自动降级**
   - 检测到 MediaPipe 不可用
   - 自动切换到 NoOp 模式
   - 使用云端视觉识别
   - 不影响其他功能

### 7. 支持哪些平台？

| 平台 | 架构 | MediaPipe | 物体检测 | 状态 |
|------|------|-----------|----------|------|
| macOS | ARM64 (Apple Silicon) | ✅ | ✅ | 完全支持 |
| Linux | x86_64 | ✅ | ✅ | 完全支持 |
| Windows | AMD64 | ✅ | ✅ | 完全支持 |
| **Raspberry Pi** | **ARM64/ARMv7** | **❌** | **❌** | **不支持** |

### 8. 如果不配置会怎样？

**系统会自动降级**：

1. 检测到 `LELAMP_EDGE_VISION_ENABLED=false`
2. 或检测到 MediaPipe 不可用
3. 自动切换到云端视觉识别
4. 所有功能正常工作

**唯一区别**：
- ❌ 本地识别（边缘视觉）
- ✅ 云端识别（Qwen VL API）

### 9. 性能对比

| 功能 | 边缘视觉 | 云端视觉 |
|------|----------|----------|
| **人脸检测** | < 50ms | N/A |
| **手势追踪** | < 100ms | N/A |
| **物体检测** | < 300ms | 3-8 秒 |
| **网络依赖** | 无 | 需要 |
| **隐私保护** | 本地处理 | 上传图片 |
| **支持平台** | 有限 | 所有平台 |

### 10. 相关文档

详细配置和故障排查，请参考：

- 📖 [边缘视觉配置指南](EDGE_VISION_SETUP.md) - 完整配置教程
- ✨ [功能说明](FEATURES.md) - 功能特性详解
- 🔧 [完整设置指南](SETUP_GUIDE.md) - 系统配置教程
- 📘 [用户快速指南](USER_GUIDE_QUICK.md) - 快速上手

---

## 📝 快速参考

### 下载模型

```bash
bash scripts/download_edge_vision_models.sh
```

### 配置 .env

```bash
LELAMP_EDGE_VISION_ENABLED=true
LELAMP_EDGE_VISION_MODEL_DIR=/home/pi/lelamp_runtime/models
LELAMP_EDGE_VISION_CONFIDENCE_THRESHOLD=0.5
```

### 验证安装

```bash
# 检查文件
ls -lh ~/lelamp_runtime/models/efficientdet_lite0.tflite

# 运行测试
uv run pytest tests/test_edge_vision.py -v
```

---

**最后更新**: 2026-03-25
**版本**: v0.1.0
