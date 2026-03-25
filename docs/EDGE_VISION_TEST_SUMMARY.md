# 树莓派边缘视觉测试 - 完整指南

## ✅ 已创建的测试资源

我已经为您创建了完整的边缘视觉测试系统，包括自动化脚本、手动测试指南和快速参考文档。

---

## 🚀 开始测试（3步完成）

### 步骤 1: SSH 连接到树莓派

```bash
ssh pi@<您的树莓派IP地址>
# 例如: ssh pi@192.168.1.100
```

### 步骤 2: 运行自动化测试脚本

```bash
cd ~/lelamp_runtime
bash scripts/test_edge_vision_on_pi.sh
```

### 步骤 3: 根据测试结果配置

**如果测试成功**（MediaPipe 可用）:
```bash
# 启用边缘视觉
nano .env
# 添加: LELAMP_EDGE_VISION_ENABLED=true

# 重启服务
sudo systemctl restart lelamp-*
```

**如果测试失败**（MediaPipe 不可用）:
```bash
# 无需额外配置
# 系统会自动使用云端视觉识别
# 功能完全正常
```

---

## 📁 创建的文件清单

### 1. 测试脚本
- **scripts/test_edge_vision_on_pi.sh** - 自动化测试脚本
  - 检查 MediaPipe 安装
  - 检查 OpenCV 安装
  - 测试所有边缘视觉模块
  - 生成详细测试报告

### 2. 测试文档
- **docs/EDGE_VISION_TEST_PI.md** - 完整测试指南
  - 详细测试步骤
  - 手动测试方法
  - 结果解读
  - 故障排查

- **docs/EDGE_VISION_SSH_QUICK_REF.md** - SSH 快速参考
  - 一键测试命令
  - 快速判断结果
  - 常用故障排查命令

### 3. 配置文档（之前创建）
- **docs/EDGE_VISION_SETUP.md** - 配置指南
- **docs/EDGE_VISION_QUICK_REF.md** - 快速参考
- **docs/EDGE_VISION_MODEL_INFO.md** - 模型信息

---

## 🧪 测试脚本功能

### 自动测试项目

| 测试项 | 说明 | 预期结果 |
|--------|------|----------|
| **系统信息** | 平台、Python 版本 | 显示系统信息 |
| **MediaPipe 检查** | 是否安装 MediaPipe | 显示版本号或报错 |
| **OpenCV 检查** | 是否安装 OpenCV | 显示版本号 |
| **模块导入** | 边缘视觉模块 | 所有模块导入成功 |
| **人脸检测器** | FaceDetector 初始化 | 显示状态和统计 |
| **手势追踪器** | HandTracker 初始化 | 显示状态和统计 |
| **物体检测器** | ObjectDetector 初始化 | 显示状态和统计 |
| **混合视觉服务** | HybridVisionService | 显示服务状态 |
| **图像处理** | 创建测试图像并检测 | 显示检测结果 |

### 测试输出示例

#### 成功案例（MediaPipe 可用）
```
======================================
  LeLamp 边缘视觉功能测试
======================================

📋 系统信息:
   平台: x86_64
   系统: Linux
   Python: Python 3.12.0

✅ MediaPipe 已安装
   版本: 0.10.9

✅ OpenCV 已安装
   版本: 4.9.0

✅ 所有边缘视觉模块导入成功

==================================================
测试物体检测器
==================================================
物体检测器状态: 正常模式
总检测次数: 0

==================================================
测试混合视觉服务
==================================================
混合视觉服务状态: 正常
服务组件:
  - 人脸检测: True
  - 手势追踪: True
  - 物体检测: True

✅ 边缘视觉功能正常工作
```

#### NoOp 模式（MediaPipe 不可用）
```
❌ MediaPipe 未安装

⚠️  边缘视觉功能处于 NoOp 模式
   这通常意味着 MediaPipe 不可用
   系统会自动降级到云端视觉识别
```

---

## 🎯 预期结果

### Raspberry Pi 4B (ARM64)

**最可能的结果**: NoOp 模式
- 原因: MediaPipe 官方不支持 ARM 架构
- 影响: 无影响，系统自动降级
- 备用方案: 云端 Qwen VL 视觉识别

**建议配置**:
```bash
# 在 .env 中
LELAMP_EDGE_VISION_ENABLED=false
```

### 开发机 (macOS/Linux x86_64)

**最可能的结果**: 正常模式
- 条件: 正确安装 MediaPipe
- 功能: 完整的边缘视觉能力
- 性能: 人脸检测 < 50ms, 物体检测 < 300ms

**建议配置**:
```bash
# 在 .env 中
LELAMP_EDGE_VISION_ENABLED=true
LELAMP_EDGE_VISION_MODEL_DIR=~/lelamp_runtime/models
LELAMP_EDGE_VISION_CONFIDENCE_THRESHOLD=0.5
```

---

## 📝 测试后配置

### 情况 1: 测试成功，启用边缘视觉

```bash
# 1. 编辑配置
nano .env

# 2. 添加配置
LELAMP_EDGE_VISION_ENABLED=true
LELAMP_EDGE_VISION_MODEL_DIR=/home/pi/lelamp_runtime/models
LELAMP_EDGE_VISION_CONFIDENCE_THRESHOLD=0.5

# 3. 重启服务
sudo systemctl restart lelamp-api
sudo systemctl restart lelamp-livekit

# 4. 查看日志
tail -f /var/log/lelamp.log | grep -E "EdgeVision|MediaPipe|ObjectDetector"

# 5. 验证功能
# 通过 Web 界面或语音命令测试视觉识别
```

### 情况 2: 测试失败，使用云端视觉

```bash
# 1. 确认配置
nano .env

# 2. 确保禁用边缘视觉
LELAMP_EDGE_VISION_ENABLED=false

# 3. 重启服务
sudo systemctl restart lelamp-api
sudo systemctl restart lelamp-livekit

# 4. 系统会自动使用云端 Qwen VL 视觉识别
# 无需其他配置
```

---

## 🔍 手动验证

### 方法 1: Python 交互测试

```bash
cd ~/lelamp_runtime
uv run python
```

```python
# 在 Python REPL 中测试
from lelamp.edge.hybrid_vision import HybridVisionService
import numpy as np

# 创建测试图像
frame = np.zeros((480, 640, 3), dtype=np.uint8)

# 初始化服务
service = HybridVisionService(enable_object=True)

# 测试检测
result = service.detect_objects(frame)
print(result)

# 查看统计
stats = service.get_stats()
print(stats)
```

### 方法 2: 运行单元测试

```bash
cd ~/lelamp_runtime

# 运行边缘视觉测试
uv run pytest tests/test_edge_vision.py -v

# 查看测试报告
uv run pytest tests/test_edge_vision.py --cov=lelamp.edge --cov-report=html
```

### 方法 3: 查看实时日志

```bash
# 查看 API 服务日志
tail -f /var/log/lelamp.log | grep -E "EdgeVision|MediaPipe|ObjectDetector"

# 查看 systemd 服务日志
sudo journalctl -u lelamp-api -f | grep -E "EdgeVision|MediaPipe"
```

---

## 📚 完整文档索引

### 测试相关
1. **EDGE_VISION_TEST_PI.md** - 完整测试指南（本文档）
2. **EDGE_VISION_SSH_QUICK_REF.md** - SSH 快速参考

### 配置相关
3. **EDGE_VISION_SETUP.md** - 详细配置指南
4. **EDGE_VISION_QUICK_REF.md** - 快速参考
5. **EDGE_VISION_MODEL_INFO.md** - 模型信息

### 功能说明
6. **FEATURES.md** - 边缘视觉功能说明

### 系统文档
7. **SETUP_GUIDE.md** - 系统设置指南
8. **ARCHITECTURE.md** - 系统架构文档

---

## 🎬 测试流程图

```
SSH 连接到树莓派
        ↓
cd ~/lelamp_runtime
        ↓
bash scripts/test_edge_vision_on_pi.sh
        ↓
    ┌─────────────────┐
    │ MediaPipe 可用？  │
    └─────────────────┘
         ↓
    是 │         │ 否
       ↓         ↓
正常模式   NoOp模式
       ↓         ↓
启用边缘视觉  使用云端视觉
       ↓         ↓
配置 .env    配置 .env
       ↓         ↓
重启服务    重启服务
       ↓         ↓
本地AI推理   云端AI推理
```

---

## ⚡ 快速命令参考

```bash
# === 测试命令 ===
bash scripts/test_edge_vision_on_pi.sh              # 自动化测试
uv run pytest tests/test_edge_vision.py -v          # 单元测试

# === 检查命令 ===
uv run python -c "import mediapipe; print('OK')"    # 检查 MediaPipe
uv run python -c "import cv2; print('OK')"           # 检查 OpenCV
uv run python -c "from lelamp.edge.hybrid_vision import HybridVisionService; print('OK')"

# === 配置命令 ===
nano .env                                          # 编辑配置
sudo systemctl restart lelamp-*                    # 重启服务
tail -f /var/log/lelamp.log | grep EdgeVision        # 查看日志

# === 故障排查 ===
uv pip list | grep -E mediapipe|opencv              # 查看已安装包
ls -l /dev/video*                                  # 检查摄像头
groups                                            # 检查用户组
```

---

## 📞 获取帮助

如果测试过程中遇到问题：

1. **查看测试脚本输出** - 脚本会显示详细的错误信息
2. **查看完整测试指南** - `docs/EDGE_VISION_TEST_PI.md`
3. **查看故障排查章节** - 包含常见问题和解决方案
4. **查看系统日志** - `/var/log/lelamp.log`
5. **运行手动测试** - 验证具体哪个环节出错

---

**测试脚本**: `scripts/test_edge_vision_on_pi.sh`
**完整指南**: [EDGE_VISION_TEST_PI.md](EDGE_VISION_TEST_PI.md)
**快速参考**: [EDGE_VISION_SSH_QUICK_REF.md](EDGE_VISION_SSH_QUICK_REF.md)

**版本**: v0.1.0
**最后更新**: 2026-03-25
**作者**: LeLamp 开发团队
