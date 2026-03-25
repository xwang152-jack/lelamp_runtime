# 🖐️ 手部追踪功能使用指南

## 🎯 功能概述

LeLamp 支持手势识别功能，可以通过摄像头检测和识别多种手势，实现非接触式交互。

### 支持的手势

- **👍 点赞** (Thumb Up)
- **👎 踩** (Thumb Down)
- **✌️ 耶/胜利** (Victory)
- **👋 挥手** (Wave)
- **✊ 握拳** (Closed Fist)
- **👆 指向** (Pointing Up)

## 🚀 启用步骤

### 1. 确认边缘视觉已启用

手部追踪功能依赖边缘视觉系统，确保以下配置已设置：

```bash
# 在树莓派上检查配置
cd ~/lelamp_runtime
cat .env | grep EDGE_VISION
```

应该看到：
```
LELAMP_EDGE_VISION_ENABLED=true
LELAMP_EDGE_VISION_MODEL_DIR=/home/pi/lelamp_runtime/models
LELAMP_EDGE_VISION_CONFIDENCE_THRESHOLD=0.5
LELAMP_EDGE_VISION_FPS=15
```

### 2. 重启服务

如果修改了配置，需要重启服务：

```bash
sudo systemctl restart lelamp-livekit
```

### 3. 验证功能

运行验证脚本确认手部追踪已启用：

```bash
cd ~/lelamp_runtime
uv run python scripts/verify_edge_vision.py
```

应该看到：
```
HandTracker: ✅ 正常模式
```

## 💬 使用方式

### 方式 1: 语音命令（推荐）

**直接对 LeLamp 说话，系统会自动检测手势：**

```
"检测一下手势"
"我做了什么手势？"
"看看我的手势"
```

### 方式 2: API 调用

**通过 Web API 调用手部追踪：**

```bash
# 调用手势检测
curl -X POST http://192.168.0.106:8000/agent/functions/detect_gesture \
  -H "Content-Type: application/json"
```

### 方式 3: Web 界面

**通过 Web 控制面板：**

1. 访问 http://192.168.0.106:8000/docs
2. 找到 `detect_gesture` 函数
3. 点击 "Try it out"
4. 点击 "Execute"

## 🎮 使用示例

### 日常交互场景

**1. 点赞手势**
- 👍 对着摄像头竖起大拇指
- 说："检测手势"
- LeLamp 回复："检测到手势：点赞"

**2. 挥手打招呼**
- 👋 对着摄像头挥手
- 说："我做了什么手势？"
- LeLamp 回复："检测到手势：挥手"

**3. 握拳确认**
- ✊ 对着摄像头握拳
- 说："看看我的手势"
- LeLamp 回复："检测到手势：握拳"

## 🔧 技术细节

### 工作原理

1. **图像采集**: 摄像头持续采集画面
2. **手部检测**: MediaPipe 检测手部位置
3. **手势识别**: 分析手指关节角度和位置
4. **结果返回**: 返回识别到的手势类型

### 性能指标

- **检测速度**: ~136ms 平均响应时间
- **准确率**: 在良好光线条件下 >90%
- **支持距离**: 0.5m - 2m 最佳
- **支持角度**: 正面角度效果最好

### 最佳实践

**光线条件:**
- ✅ 充足的室内光线
- ✅ 避免强光直射摄像头
- ✅ 避免背光拍摄

**拍摄角度:**
- ✅ 手部正对摄像头
- ✅ 完整显示手掌和手指
- ✅ 手势保持 1-2 秒

**避免情况:**
- ❌ 手部移动过快
- ❌ 手指部分遮挡
- ❌ 光线太暗或太亮

## 🎯 高级功能

### 手势触发动作

系统可以配置特定手势触发特定动作：

```python
# 在系统中配置（开发功能）
gesture_actions = {
    "Thumb_Up": "play_recording(happy)",      # 👍 点赞 → 开心动作
    "Wave": "play_recording(greeting)",       # 👋 挥手 → 问候动作
    "Closed_Fist": "rgb_effect_stop",         # ✊ 握拳 → 停止灯效
}
```

### 实时手势追踪

**持续监控手势：**

```bash
# 查看实时手势状态
curl http://192.168.0.106:8000/api/vision/hand-tracking
```

## 📊 调试和监控

### 查看手部追踪状态

```bash
# 获取边缘视觉统计
curl http://192.168.0.106:8000/agent/functions/get_edge_vision_stats
```

### 查看详细日志

```bash
# 查看实时日志
sudo journalctl -u lelamp-livekit -f | grep -E "hand|Hand|gesture|Gesture"
```

### 测试脚本

**使用测试脚本验证功能：**

```bash
cd ~/lelamp_runtime
uv run python scripts/test_real_camera_tracking.py
```

## ❓ 常见问题

### Q: 手势检测不工作？

**A:** 检查以下几点：
1. 边缘视觉是否启用：`grep EDGE_VISION .env`
2. 摄像头是否正常：`ls /dev/video*`
3. 服务是否运行：`systemctl status lelamp-livekit`

### Q: 检测准确率低？

**A:** 改善检测环境：
1. 提高室内光线亮度
2. 手部完整显示在摄像头画面中
3. 保持手势稳定 1-2 秒
4. 手部正对摄像头

### Q: 支持自定义手势吗？

**A:** 当前支持预设的 6 种手势，自定义手势需要开发。

### Q: 可以同时检测多只手吗？

**A:** 支持，最多同时检测 2 只手。

## 🎉 开始使用

**最简单的使用方式：**

1. **确保边缘视觉已启用** ✅（当前状态）
2. **走到摄像头前**
3. **做一个手势**（👍、👋、✊等）
4. **说："检测手势"**
5. **LeLamp 会告诉你检测到的手势**

**就这么简单！手部追踪功能已完全就绪！** 🚀