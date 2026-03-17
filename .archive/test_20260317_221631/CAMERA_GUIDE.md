# 📹 摄像头画面开启指南

## 🎯 快速答案

**摄像头的画面来自后端服务，不是前端！**

要看到摄像头画面，你需要：

1. ✅ 启动 **LeLamp 后端服务**（`sudo uv run main.py console`）
2. ✅ 后端会启动摄像头并推送视频流
3. ✅ 前端连接到后端后，会自动显示视频

---

## 🚀 完整步骤

### 方式 1: 使用自动化脚本（推荐）

```bash
./test_e2e.sh
```

这个脚本会自动：
- ✅ 检查环境配置
- ✅ 生成客户端 Token
- ✅ 启动后端服务
- ✅ 启动前端服务
- ✅ 打开浏览器
- ✅ 指导连接步骤

### 方式 2: 手动操作

#### 步骤 1: 启动后端服务

```bash
cd /Users/jackwang/lelamp_runtime

# 启动后端（需要 sudo 权限访问硬件）
sudo uv run main.py console
```

**后端启动后会做什么**：
- 🔌 连接到 LiveKit 服务器
- 📹 启动摄像头
- 🎤 启动麦克风
- 🌐 通过 WebRTC 推送音视频流
- 🤖 处理语音对话
- 💡 控制 RGB 灯光
- ⚙️ 控制电机动作

#### 步骤 2: 生成客户端 Token

在**另一个终端**运行：

```bash
python3 scripts/generate_client_token.py
```

输出类似：
```
LiveKit URL: wss://your-livekit-server.livekit.cloud
Token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

#### 步骤 3: 连接前端

1. 访问: http://localhost:5173
2. 输入 Server URL（从步骤 2）
3. 输入 Token（从步骤 2）
4. 点击"连接设备"

#### 步骤 4: 查看摄像头画面

连接成功后，在控制台页面中央会看到：
- 📹 **实时摄像头画面**
- 🔴 隐私指示器显示"摄像头已开启"
- 💡 LED 指示灯变为红色

---

## 🔍 故障排除

### 问题 1: 看不到摄像头画面

**可能原因**：

1. **后端服务未启动**
   ```bash
   # 检查后端是否运行
   ps aux | grep "python.*main.py"

   # 如果没有运行，启动它
   sudo uv run main.py console
   ```

2. **摄像头被其他程序占用**
   ```bash
   # 检查摄像头使用情况
   lsof /dev/video0

   # 关闭其他使用摄像头的程序（如 Zoom、Teams）
   ```

3. **权限不足**
   ```bash
   # 确保使用 sudo 启动后端
   sudo uv run main.py console
   ```

4. **LiveKit 连接失败**
   ```bash
   # 检查 .env 配置
   cat .env | grep LIVEKIT

   # 查看 LiveKit URL 是否正确
   ```

### 问题 2: 连接失败

**检查清单**：

- [ ] Server URL 正确（通常是 `wss://...`）
- [ ] Token 正确（完整的 JWT token）
- [ ] 后端服务正在运行
- [ ] 前端开发服务器正在运行
- [ ] 网络连接正常

### 问题 3: 后端启动失败

**查看日志**：

```bash
# 查看后端输出
# 后端启动时会显示详细的日志信息

# 常见错误：
# - 摄像头不存在 → 检查 /dev/video*
# - 权限不足 → 使用 sudo
# - LiveKit 连接失败 → 检查 .env 配置
```

---

## 📊 系统架构图

```
┌─────────────────┐
│  Web 前端       │
│  (localhost:5173)│
└────────┬─────────┘
         │
         │ LiveKit WebSocket
         │ (Video + Audio + Data)
         │
┌────────▼─────────┐
│  LiveKit Cloud   │
│  (信令服务器)     │
└────────┬─────────┘
         │
         │ LiveKit WebSocket
         │
┌────────▼─────────┐
│  LeLamp 后端     │
│  (main.py)       │
│                  │
│  📹 摄像头 ◄──────┤
│  🎤 麦克风 ◄─────┤
│  💡 RGB 灯 ◄─────┤
│  ⚙️ 电机   ◄─────┤
└──────────────────┘
```

**关键点**：
- 前端**不直接访问**摄像头
- 摄像头由**后端服务**控制
- 视频通过 **LiveKit** 传输
- 前端只是**接收端**

---

## 🎯 核心概念

### 前端职责
- 🖥️ 提供 UI 界面
- 📡 接收视频流（显示摄像头画面）
- 📤 发送控制命令（灯光、动作）
- 💬 显示对话消息

### 后端职责
- 📹 **启动和管理摄像头**
- 🎤 **录制和处理音频**
- 🤖 **运行 AI 对话**
- 💡 **控制硬件设备**
- 📡 **推送音视频流**

### LiveKit 职责
- 🌐 提供 WebRTC 基础设施
- 🔄 转发音视频流
- 🔐 处理认证和授权
- 📡 管理实时连接

---

## 💡 快速测试命令

### 一键启动后端

```bash
cd /Users/jackwang/lelamp_runtime
sudo uv run main.py console
```

### 一键启动前端

```bash
cd /Users/jackwang/lelamp_runtime/web
pnpm dev
```

### 一键生成 Token

```bash
cd /Users/jackwang/lelamp_runtime
python3 scripts/generate_client_token.py
```

### 一键完整测试

```bash
cd /Users/jackwang/lelamp_runtime
./test_e2e.sh
```

---

## 🎉 成功标志

当一切正常时，你会看到：

### 后端终端
```
[INFO] LeLamp agent starting...
[INFO] Connected to LiveKit
[INFO] Camera initialized
[INFO] Microphone initialized
[INFO] Ready to accept connections
```

### 浏览器控制台页面
- 📹 **中央视频区域显示实时摄像头画面**
- 🔴 隐私指示器显示"摄像头已开启"
- 💡 LED 指示灯为红色（呼吸动画）
- ✅ 顶部状态栏显示"已连接"
- 🟢 状态点显示绿色

---

## 📞 获取帮助

如果仍有问题：

1. **查看后端日志**：后端终端会显示详细错误信息
2. **检查环境配置**：确认 `.env` 文件配置正确
3. **查看硬件状态**：`ls /dev/video*` 检查摄像头
4. **参考文档**：
   - `CLAUDE.md` - 项目架构说明
   - `docs/BACKEND_DEVELOPMENT_PLAN.md` - 后端开发计划
   - `web/TESTING_GUIDE.md` - 前端测试指南

---

**记住**：摄像头的开启和画面推送是由**后端服务**完成的，前端只是接收和显示！

**快速开始**：运行 `./test_e2e.sh` 即可自动完成所有步骤！🚀
