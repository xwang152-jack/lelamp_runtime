# LeLamp 使用指南 - 如何启动和唤醒功能

## 🚀 系统启动流程

### 1. 自动启动（推荐）
设备启动后，LeLamp 会自动启动以下服务：

```bash
# 查看服务状态
systemctl status lelamp-livekit.service     # LiveKit 语音交互服务
systemctl status lelamp-frontend.service    # Web 前端界面
systemctl status lelamp-api.service         # API 服务（如有）
```

### 2. 手动启动
如果需要手动启动服务：

```bash
# 启动 LiveKit 语音交互服务
sudo systemctl restart lelamp-livekit

# 启动 Web 前端
sudo systemctl restart lelamp-frontend

# 启动 API 服务
sudo systemctl restart lelamp-api
```

---

## 🎤 如何唤醒和使用 LeLamp

### 方式 1: Web 界面（推荐）

1. **访问 Web 界面**
   - 在浏览器中打开：`http://lelamp.local:5173` 或 `http://<树莓派IP>:5173`
   - 界面会显示连接状态和控制按钮

2. **连接到 LeLamp**
   - 点击 "连接" 按钮
   - 允许浏览器访问麦克风权限
   - 等待连接建立成功

3. **开始对话**
   - 直接说话，LeLamp 会自动检测语音
   - 无需特定唤醒词，系统会自动识别
   - 说完话后稍作停顿，LeLamp 会自动回复

### 方式 2: LiveKit Console 模式

系统已经在后台以 Console 模式运行：

```bash
# 查看 LiveKit 运行状态
sudo tmux attach -t livekit

# 分离会话（不停止服务）
按 Ctrl+B，然后按 D
```

### 方式 3: 命令行直接运行

```bash
cd ~/lelamp_runtime
uv run python main.py console
```

---

## 💬 交互功能说明

### 自动语音检测
LeLamp 使用 **VAD（Voice Activity Detection）** 自动检测语音：

- **无需唤醒词**：直接说话即可
- **自动识别**：系统会自动检测你开始说话
- **智能停顿**：说话结束后稍作停顿，系统会自动处理

### 对话示例

```
你: "你好"
LeLamp: "你好呀！有什么我可以帮你的吗？"

你: "看看这是什么"
LeLamp: [使用视觉识别物体] "这是一个..."

你: "帮我检查作业"
LeLamp: [使用视觉检查作业] "好的，让我看看..."

你: "向左转30度"
LeLamp: [控制电机转动] "好的，向左转30度"
```

---

## 🎯 支持的功能

### 语音交互
- ✅ 自由对话，无需唤醒词
- ✅ 自动语音检测（VAD）
- ✅ 中文语音识别和合成
- ✅ 降噪处理

### 视觉功能
- ✅ 物体识别："这是什么？"
- ✅ 作业检查："帮我检查作业"
- ✅ 人脸识别：自动检测人脸
- ✅ 手部追踪：识别手势动作

### 动作控制
- ✅ 电机控制："向左转"、"向上看"
- ✅ 录制动作回放：播放预设动作
- ✅ 关节控制：精确控制各个关节

### 灯光效果
- ✅ 颜色控制："变成红色"
- ✅ 灯光特效：呼吸、闪烁等
- ✅ 状态指示：对话状态通过灯光显示

### 系统功能
- ✅ 网络搜索：实时信息查询
- ✅ 系统设置：音量、配置等
- ✅ OTA 更新：远程固件升级

---

## 🔧 配置和调试

### 查看日志

```bash
# LiveKit 服务日志
sudo journalctl -u lelamp-livekit -f

# API 服务日志
sudo journalctl -u lelamp-api -f

# 前端服务日志
sudo journalctl -u lelamp-frontend -f
```

### 检查连接状态

```bash
# 检查进程
ps aux | grep -E 'lelamp|livekit'

# 检查端口
ss -tulpn | grep -E '5173|7681|8000'

# 检查 tmux 会话
sudo tmux ls
```

### 常见问题

**Q: 无法连接到 LeLamp？**
- 检查服务是否运行：`sudo systemctl status lelamp-livekit`
- 检查网络连接：`ping lelamp.local`
- 检查浏览器麦克风权限

**Q: 语音识别不工作？**
- 检查麦克风权限
- 检查音量设置：`amixer sset Line 100%`
- 查看日志是否有错误

**Q: 视觉功能不工作？**
- 确认摄像头已连接：`ls /dev/video*`
- 检查权限：`groups`（应在 video 组）
- 确认视觉功能已启用：`.env` 中 `LELAMP_VISION_ENABLED=true`

---

## 🎨 LED 状态指示

LeLamp 通过 LED 颜色显示当前状态：

- **暖白色** (255, 244, 229): 空闲状态
- **蓝色** (0, 140, 255): 正在听你说话
- **紫色** (180, 0, 255): 正在思考
- **彩色动画**: 正在说话
- **红色呼吸**: 摄像头正在使用

---

## 🌐 访问地址

- **Web 界面**: http://lelamp.local:5173
- **API 文档**: http://lelamp.local:8000/docs
- **设备主页**: http://lelamp.local:8080 (首次设置)

---

## 📝 配置文件

主要配置在 `.env` 文件中：

```bash
# 视觉功能
LELAMP_VISION_ENABLED=true           # 启用视觉
LELAMP_CAMERA_INDEX_OR_PATH=0        # 摄像头设备

# 边缘视觉
LELAMP_EDGE_VISION_ENABLED=true      # 启用边缘视觉
LELAMP_EDGE_VISION_FPS=15            # 帧率

# 语音交互
LELAMP_VAD_MIN_SPEECH_DURATION=0.1   # 最小说话时长
LELAMP_VAD_MIN_SILENCE_DURATION=0.8  # 最小静音时长

# 欢迎语
LELAMP_GREETING_TEXT="你好！我是小宝贝，很高兴见到你！"
```

---

**🎉 享受与 LeLamp 的互动吧！**

直接说话，无需唤醒词，LeLamp 会自动响应你的需求！