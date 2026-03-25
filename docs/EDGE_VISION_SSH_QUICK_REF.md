# 🌿 SSH 测试边缘视觉 - 快速参考

## 🚀 一键测试（推荐）

```bash
# 1. SSH 连接到树莓派
ssh pi@<您的树莓派IP>

# 2. 进入项目目录
cd ~/lelamp_runtime

# 3. 运行测试脚本
bash scripts/test_edge_vision_on_pi.sh
```

---

## 🧪 手动快速测试

### 测试 MediaPipe
```bash
uv run python -c "import mediapipe; print('✅ MediaPipe:', mediapipe.__version__)"
```

### 测试 OpenCV
```bash
uv run python -c "import cv2; print('✅ OpenCV:', cv2.__version__)"
```

### 测试边缘视觉模块
```bash
uv run python -c "
from lelamp.edge.object_detector import ObjectDetector
detector = ObjectDetector()
stats = detector.get_stats()
print('✅ 检测器状态:', '正常' if not stats.get('noop_mode') else 'NoOp模式')
"
```

---

## 📊 结果判断

### ✅ 测试成功
```
✅ MediaPipe: 0.10.9
✅ OpenCV: 4.9.0
✅ 检测器状态: 正常
```
→ **边缘视觉可用！**

### ⚠️ NoOp 模式
```
❌ MediaPipe: ImportError
✅ OpenCV: 4.9.0
⚠️ 检测器状态: NoOp模式
```
→ **边缘视觉不可用，系统会自动降级到云端识别**

---

## ⚙️ 配置边缘视觉

### 如果测试成功
```bash
# 1. 编辑配置
nano .env
# 添加: LELAMP_EDGE_VISION_ENABLED=true

# 2. 重启服务
sudo systemctl restart lelamp-*

# 3. 查看日志
tail -f /var/log/lelamp.log | grep EdgeVision
```

### 如果测试失败
```bash
# 无需额外配置
# 系统会自动使用云端 Qwen VL 视觉识别
# 功能完全正常
```

---

## 🔍 故障排查

### MediaPipe 安装失败
```bash
# 尝试安装
uv sync --extra vision

# 或手动安装
uv pip install mediapipe opencv-python-headless
```

### 摄像头问题
```bash
# 检查摄像头
ls -l /dev/video*

# 添加权限
sudo usermod -a -G video $USER
sudo reboot
```

---

## 📚 详细文档

```bash
# 查看完整测试指南
cat docs/EDGE_VISION_TEST_PI.md

# 查看配置指南
cat docs/EDGE_VISION_SETUP.md

# 查看快速参考
cat docs/EDGE_VISION_QUICK_REF.md
```

---

**快速命令**: `bash scripts/test_edge_vision_on_pi.sh`
**详细指南**: [EDGE_VISION_TEST_PI.md](EDGE_VISION_TEST_PI.md)
