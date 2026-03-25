# 🎉 LeLamp 系统当前状态和使用方法

## 🚀 系统运行状态

### ✅ 正在运行的服务

1. **Console 模式** (语音交互)
   - 状态: ✅ 运行中
   - 功能: 本地语音对话
   - 使用方式: **直接对设备说话**

2. **FastAPI 服务** (Web 控制面板)
   - 状态: ✅ 运行中
   - 地址: http://192.168.0.106:8000
   - 功能: Web 控制面板、API 接口

3. **Web 前端** (Vue 界面)
   - 状态: ✅ 运行中
   - 地址: http://192.168.0.106:5173
   - 注意: 需要 LiveKit 服务器才能完全功能

## 🎤 如何使用 LeLamp

### 方式 1: 直接语音对话（推荐！）

**最简单的方式 - 直接说话即可！**

1. **确保设备在听你说话**
   - LED 显示白色 = 空闲状态，可以说话

2. **直接对设备说话**
   - 无需唤醒词，无需按钮
   - 直接说："你好"、"看看这是什么"、"向左转"

3. **等待回复**
   - 蓝色 = 正在听
   - 紫色 = 正在思考
   - 彩色动画 = 正在回复

### 方式 2: Web 控制面板

**访问 Web 界面进行控制：**

1. **打开浏览器访问**: http://192.168.0.106:8000/docs
2. **可以看到 API 文档和测试界面**
3. **也可以访问**: http://192.168.0.106:5173 (Vue 前端)

### 方式 3: API 接口调用

**通过编程方式控制：**

```bash
# 健康检查
curl http://192.168.0.106:8000/health

# 获取设备状态
curl http://192.168.0.106:8000/api/devices/lelamp/state

# 执行动作
curl -X POST http://192.168.0.106:8000/api/devices/lelamp/action \
  -H "Content-Type: application/json" \
  -d '{"action": "move_joint", "joint": "base_yaw", "position": 30}'
```

## 🎯 边缘视觉功能状态

### ✅ 已启用并正常运行

```
边缘视觉配置: LELAMP_EDGE_VISION_ENABLED=true ✅

人脸识别: ✅ 正常模式 (本地 AI 推理，14ms 响应)
手部追踪: ✅ 正常模式 (本地 AI 推理，136ms 响应)
物体检测: ✅ 云端备用 (Qwen VL 智能切换)
```

### 视觉功能使用示例

- "看看这是谁" → 本地人脸识别
- "挥挥手" → 本地手势追踪
- "这是什么？" → 云端物体识别

## 📊 系统监控

### 查看服务状态

```bash
# 查看所有服务
systemctl status lelamp-*

# 查看 Console 模式日志
sudo journalctl -u lelamp-livekit -f

# 查看 API 服务日志
sudo journalctl -u lelamp-api -f

# 查看前端服务日志
sudo journalctl -u lelamp-frontend -f
```

### 检查边缘视觉状态

```bash
cd ~/lelamp_runtime
uv run python scripts/verify_edge_vision.py
```

## 🔧 故障排查

### 问题 1: 语音识别不工作

**解决方案:**
```bash
# 检查音量设置
amixer sset Line 100%
amixer sset 'Line DAC' 100%
amixer sset HP 100%

# 重启 Console 模式
sudo systemctl restart lelamp-livekit
```

### 问题 2: Web 界面无法访问

**解决方案:**
```bash
# 检查服务状态
systemctl status lelamp-api lelamp-frontend

# 重启服务
sudo systemctl restart lelamp-api
sudo systemctl restart lelamp-frontend
```

### 问题 3: 边缘视觉不工作

**解决方案:**
```bash
# 检查配置
cd ~/lelamp_runtime
grep EDGE_VISION .env

# 重新启用
echo "LELAMP_EDGE_VISION_ENABLED=true" >> .env
sudo systemctl restart lelamp-livekit
```

## 💡 推荐使用方式

**日常使用：直接语音对话**
- 最自然、最简单
- 无需任何界面或设备
- 边缘视觉功能已自动启用

**开发调试：Web 控制面板**
- 查看 API 文档: http://192.168.0.106:8000/docs
- 测试各种功能
- 监控系统状态

**系统管理：命令行**
- 通过 SSH 连接
- 查看日志和服务状态
- 配置系统参数

---

**🎉 享受与 LeLamp 的互动吧！直接说话即可，无需任何复杂操作！**