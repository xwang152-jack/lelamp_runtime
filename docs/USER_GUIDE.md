# LeLamp 完整使用指南 v0.1.0

**最后更新**: 2026-03-19
**适用版本**: LeLamp Runtime v0.1.0

**当前功能**:
- ✅ Web 设置页面 - 无需 SSH 即可配置系统
- ✅ WiFi 网络配置 - 扫描、连接、断开 WiFi
- ✅ 配置管理 API - RESTful API 用于系统配置
- ✅ 服务热重启 - 通过 Web 界面重启服务
- ✅ 开箱即用 AP 模式配置 - 无需技术背景即可配置 WiFi
- ✅ 自动设置检测 - 首次启动自动进入配置模式
- ✨ Captive Portal 支持 - 连接热点后自动弹出配置页面

---

## 📋 目录

1. [系统架构](#系统架构)
2. [快速开始](#快速开始)
3. [详细配置](#详细配置)
4. [功能使用](#功能使用)
5. [故障排查](#故障排查)
6. [最佳实践](#最佳实践)
7. [常见问题](#常见问题)

---

## 系统架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    用户设备 (User Device)                    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │          Web Browser / Mobile App                    │   │
│  │  - 视频预览                                          │   │
│  │  - 控制面板                                          │   │
│  │  - 文字聊天                                          │   │
│  │  - 系统设置                                          │   │
│  └──────────────────┬──────────────────────────────────┘   │
└─────────────────────┼──────────────────────────────────────┘
                      │ WebSocket (ws://IP:8000/api/ws/lelamp)
                      │ HTTP/REST API
                      │
                      ↓
┌─────────────────────────────────────────────────────────────┐
│          Raspberry Pi (LeLamp Runtime)                      │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  FastAPI Server (端口 8000) - 主服务                 │   │
│  │  - WebSocket: /api/ws/{lamp_id}                      │   │
│  │  - /api/devices/* (设备控制)                         │   │
│  │  - /api/chat/* (文字聊天)                            │   │
│  │  - /api/settings/* (配置管理)                        │   │
│  │  - /api/system/* (WiFi、重启、系统信息)               │   │
│  └──────────────────┬──────────────────────────────────┘   │
│                     │                                        │
│  ┌──────────────────┴─────────────────────────────────────┐ │
│  │  MotorsService → 电机硬件                              │ │
│  │  RGBService → LED 灯光                                 │ │
│  │  VisionService → 摄像头                                │ │
│  │  LeLamp Agent → 状态管理、消息广播                     │ │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  LiveKit Agent (main.py) - 可选，语音功能             │   │
│  │  - DeepSeek LLM                                      │   │
│  │  - Qwen VL (视觉识别)                                │   │
│  │  - Baidu Speech (STT/TTS)                            │   │
│  │  - LiveKit Agents SDK                                │   │
│  └─────────────────────────────────────────────────────┘   │
│            │                 │                 │            │
│  ┌─────────┴─────────────────┴─────────────────┴───────┐   │
│  │  Hardware (硬件层)                                   │   │
│  │  - Feetech 伺服电机 (5 个关节)                       │   │
│  │  - WS2812B LED 矩阵 (8x8, 64 颗)                     │   │
│  │  - USB 摄像头 (1024x768)                             │   │
│  │  - USB 声卡 (麦克风 + 扬声器)                         │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**服务说明**:
- **FastAPI 服务** (必需) - 设备控制、文字聊天、系统配置
- **LiveKit Agent** (可选) - 语音对话功能，需要 LiveKit 配置

### 通信协议

#### 1. WebRTC (音视频流)
- **视频编码**: H.264 / VP8
- **音频编码**: Opus
- **传输协议**: DTLS/SRTP (加密传输)
- **分辨率**: 1024x768 @ 30fps

#### 2. Data Channel (控制指令)
- **传输模式**: 可靠传输 (RELIABLE)
- **消息格式**: JSON

**消息类型**:
- `chat`: 文字聊天
- `command`: 控制指令 (动作/灯光)
- `vision_result`: 视觉识别结果
- `camera_status`: 摄像头状态

#### 3. REST API (配置管理 - 新增)
- **协议**: HTTP/HTTPS
- **数据格式**: JSON
- **认证**: (可扩展) Token 认证

**API 端点**:
- `GET /api/system/wifi/status` - 获取 WiFi 状态
- `GET /api/system/wifi/scan` - 扫描可用网络
- `POST /api/system/wifi/connect` - 连接 WiFi
- `DELETE /api/system/wifi/disconnect` - 断开 WiFi
- `GET /api/settings` - 获取系统配置
- `PUT /api/settings` - 更新系统配置
- `POST /api/system/restart` - 重启服务
- `GET /api/system/info` - 获取系统信息

---

## 快速开始

### 前置要求

#### 硬件
- ✅ Raspberry Pi 4B+ (推荐 4GB RAM)
- ✅ LeLamp 硬件套件 (电机、LED、摄像头)
- ✅ 网络连接 (Wi-Fi / 有线)

#### 软件
- ✅ Python 3.12+ (已安装)
- ✅ UV Package Manager

#### 服务
- ✅ **DeepSeek API Key** (必需 - LLM 对话)
- ⚪ **Baidu Speech API Key** (语音功能需要)
- ⚪ **LiveKit Account** (语音功能需要)
- ⚪ **ModelScope API Key** (视觉识别 - 可选)

### 5 分钟快速启动

#### Step 1: 克隆项目
```bash
cd ~
git clone https://github.com/xwang152-jack/lelamp_runtime.git
cd lelamp_runtime
```

#### Step 2: 安装依赖
```bash
# 在 Raspberry Pi 上 (包含硬件依赖)
uv sync --extra hardware

# 在开发机上 (仅电机控制,无硬件依赖)
uv sync
```

#### Step 3: 配置环境变量
```bash
cp .env.example .env
nano .env
```

**最小配置** (必填项 - 设备控制 + 文字聊天):
```bash
# DeepSeek (LLM - 必需)
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# 开发模式（跳过授权检查）
LELAMP_DEV_MODE=1
```

**本地语音功能配置** (Console 模式):
```bash
# Baidu Speech (语音服务 - Console 模式需要)
BAIDU_SPEECH_API_KEY=your_baidu_api_key_here
BAIDU_SPEECH_SECRET_KEY=your_baidu_secret_key_here
```

**远程访问配置** (可选 - Room 模式/手机 App):
```bash
# LiveKit (远程语音对话 - 可选)
LIVEKIT_URL=wss://your-livekit-server.livekit.cloud
LIVEKIT_API_KEY=your_api_key_here
LIVEKIT_API_SECRET=your_api_secret_here
```

**其他可选配置**:
```bash
# ModelScope (视觉识别 - 可选)
MODELSCOPE_API_KEY=your_modelscope_api_key_here
```

#### Step 4: 启动 FastAPI 服务
```bash
# 启动 FastAPI 服务（需要 sudo 权限用于 LED 控制）
sudo uv run uvicorn lelamp.api.app:app --host 0.0.0.0 --port 8000
```

**预期输出**:
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**说明**: FastAPI 服务提供设备控制、文字聊天和系统配置功能。

#### Step 5: 启动 Console 模式 (本地语音测试)

**前置条件**: 需要配置 Baidu Speech API Key

```bash
# 在另一个终端窗口（如果需要语音功能）
cd ~/lelamp_runtime
sudo uv run main.py console
```

**预期输出**:
```
Agents   Starting console mode 🚀
INFO     livekit.agents starting worker
DEBUG    livekit.agents using audio io: `Console` -> `AgentSession` -> `Console`
INFO     lelamp config ready: lamp_id=lelamp port=/dev/ttyACM0 vision=True camera=0
INFO     service.motors MotorsService started
INFO     service.rgb RGBService started
INFO     service.vision VisionService started
```

**使用方式**:
- 直接对着电脑麦克风说话
- 台灯会通过系统扬声器回应
- **不需要 LiveKit 配置**

**注意**: 如果只需要文字聊天和设备控制，可以跳过此步骤。

#### Step 6: 打开 Web Frontend
```bash
# 启动前端开发服务器（允许局域网访问）
cd web
pnpm dev --host 0.0.0.0 --port 5173

# 访问:
# - 树莓派本机: http://localhost:5173
# - 局域网设备: http://<树莓派IP>:5173
```

**前端环境变量配置** (可选):
```bash
# web/.env.development
VITE_API_BASE_URL=http://your-raspberry-pi:8000  # API 服务器地址
```

**在 Web Client 中**:
1. 输入 API 地址：`http://<树莓派IP>:8000`
2. 点击 "连接设备"
3. 开始使用！🎉

**访问设置页面**:
- 连接后点击右上角 "设置" 按钮
- 或直接访问 `http://<树莓派IP>:5173/settings`

---

## 详细配置

### .env 配置详解

#### LiveKit 配置
```bash
# LiveKit Server WebSocket URL (可选 - 仅远程语音访问需要)
LIVEKIT_URL=wss://your-livekit-server.livekit.cloud

# LiveKit API 凭证 (用于 Token 生成)
LIVEKIT_API_KEY=your_api_key_here
LIVEKIT_API_SECRET=your_api_secret_here
```

**获取方式**:
```bash
# 从 LiveKit Cloud Dashboard 复制
# https://cloud.livekit.io/projects/YOUR_PROJECT/settings
```

**两种语音模式**:
| 模式 | 命令 | 是否需要 LiveKit |
|------|------|------------------|
| **Console 模式** | `main.py console` | ❌ 不需要 (本地测试) |
| **Room 模式** | `main.py dev` | ✅ 需要 (远程访问) |

**说明**:
- **本地语音测试** (Console 模式)：不需要 LiveKit 配置
- **远程语音访问** (手机 App)：需要 LiveKit 配置

#### DeepSeek LLM 配置
```bash
# DeepSeek API Key (必需)
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# 模型名称 (可选,默认 deepseek-chat)
DEEPSEEK_MODEL=deepseek-chat

# API 地址 (可选,默认官方地址)
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

#### Baidu Speech 配置
```bash
# 百度语音 API 凭证 (必需)
BAIDU_SPEECH_API_KEY=your_baidu_api_key_here
BAIDU_SPEECH_SECRET_KEY=your_baidu_secret_key_here

# 设备标识 (可选,默认 lelamp)
BAIDU_SPEECH_CUID=lelamp

# TTS 发音人 (可选,默认 4 = 度丫丫)
# 可选值: 0=度小美, 1=度小宇, 3=度逍遥, 4=度丫丫, 106=博文男声, 110=度小童, 111=度小萌
BAIDU_SPEECH_TTS_PER=4
```

#### ModelScope 视觉配置
```bash
# 是否启用视觉功能 (可选,默认 true)
LELAMP_VISION_ENABLED=true

# ModelScope API Key (启用视觉时必需)
MODELSCOPE_API_KEY=your_modelscope_api_key_here

# 模型名称 (可选,默认 Qwen/Qwen3-VL-235B-A22B-Instruct)
MODELSCOPE_MODEL=Qwen/Qwen3-VL-235B-A22B-Instruct

# API 地址 (可选,默认 https://api-inference.modelscope.cn/v1)
MODELSCOPE_BASE_URL=https://api-inference.modelscope.cn/v1

# 请求超时 (秒,可选,默认 60.0)
MODELSCOPE_TIMEOUT_S=60.0
```

#### 摄像头配置
```bash
# 摄像头设备索引或路径 (可选,默认 0)
LELAMP_CAMERA_INDEX_OR_PATH=0

# 分辨率 (可选,默认 1024x768)
LELAMP_CAMERA_WIDTH=1024
LELAMP_CAMERA_HEIGHT=768

# 旋转角度 (可选,默认 0)
# 可选值: 0, 90, 180, 270
LELAMP_CAMERA_ROTATE_DEG=0

# 翻转 (可选,默认 none)
# 可选值: none, horizontal, vertical, both
LELAMP_CAMERA_FLIP=none

# 视觉采集间隔 (秒,可选,默认 2.5)
LELAMP_VISION_CAPTURE_INTERVAL_S=2.5

# JPEG 压缩质量 (可选,默认 92)
LELAMP_VISION_JPEG_QUALITY=92

# 视觉帧最大存活时间 (秒,可选,默认 15.0)
LELAMP_VISION_MAX_AGE_S=15.0
```

#### 硬件配置
```bash
# 串口端口 (可选,默认 /dev/ttyACM0)
LELAMP_PORT=/dev/ttyACM0

# 设备 ID (可选,默认 lelamp)
LELAMP_ID=lelamp

# LED 亮度 (可选,默认 25)
# 取值范围: 0-255
LELAMP_LED_BRIGHTNESS=25
```

#### 行为配置
```bash
# 启动问候语 (可选)
LELAMP_GREETING_TEXT=Hello! 小宝贝上线了.

# 启动动画 (可选,默认 1)
LELAMP_BOOT_ANIMATION=1

# 噪声消除 (可选,默认 true)
LELAMP_NOISE_CANCELLATION=true

# STT 输入增益 (可选,默认 3.0)
LELAMP_STT_INPUT_GAIN=3.0

# 动作冷却时间 (秒,可选,默认 2)
LELAMP_MOTION_COOLDOWN_S=2

# 灯光覆盖时长 (秒,可选,默认 10)
LELAMP_LIGHT_OVERRIDE_S=10

# 灯光指令后抑制动作时长 (秒,可选,默认 2)
LELAMP_SUPPRESS_MOTION_AFTER_LIGHT_S=2
```

---

## 功能使用

### 1. 语音对话

LeLamp 支持两种语音模式：

#### Console 模式（本地测试）

**前置条件**:
- 配置 Baidu Speech API Key
- 运行 `sudo uv run main.py console`

**使用方式**: 直接对着电脑麦克风说话，台灯会通过系统扬声器回应
- ✅ 不需要 LiveKit 配置
- ✅ 适合本地开发测试

#### Room 模式（远程访问）

**前置条件**:
- 配置 Baidu Speech API Key
- 配置 LiveKit URL、API Key、Secret
- 运行 `sudo uv run main.py dev`

**使用方式**: 通过手机 App 远程访问
- ✅ 支持远程语音对话
- ✅ 支持 App 控制

**示例对话**:
```
用户: "你好"
台灯: "你好呀，小主人！本灯又来陪你啦~"

用户: "现在几点了"
台灯: "现在是下午 3:45，怎么，还没写完作业就想着玩了？"

用户: "讲个笑话"
台灯: "好吧...为什么程序员总是搞混万圣节和圣诞节？因为 Oct 31 == Dec 25！[摇头]"
```

**对话状态指示** (RGB 灯光):
- 🤍 **白色**: 空闲状态 (Idle)
- 🔵 **蓝色**: 正在倾听 (Listening)
- 🟣 **紫色**: 思考中 (Thinking)
- 🌈 **随机彩色**: 正在说话 (Speaking)

### 2. 视觉识别

#### 2.1 拍照识别
**功能**: 识别物体、场景、文字

**使用方式**:
1. 将物体放在摄像头前
2. 语音: "这是什么？"
3. 或在 Web Client 点击 "🔍 拍照识别"

**示例**:
```
用户: "这是什么？" (举起一个苹果)
台灯: "这是一个红苹果，看起来很新鲜。话说，你不会是想问本灯要不要吃吧？"
```

#### 2.2 检查作业
**功能**: AI 批改数学题、语文题

**使用方式**:
1. 将作业本放在摄像头前
2. 语音: "帮我检查作业"
3. 或在 Web Client 点击 "📚 检查作业"

**示例**:
```
用户: "帮我检查作业"
台灯: "好的，让我看看..." (拍照 + AI 分析)

结果:
✅ 第1题: 5 + 3 = 8 (正确)
✅ 第2题: 12 - 7 = 5 (正确)
❌ 第3题: 6 × 4 = 32 (错误，应该是 24)
✅ 第4题: 18 ÷ 3 = 6 (正确)

正确率: 75% (3/4)

台灯评价: "还不错嘛，但第 3 题算错了哦！6 乘以 4 等于 24，不是 32。再检查一遍吧！"
```

#### 2.3 推送飞书
**功能**: 拍照并发送到飞书群组

**前置条件**:
```bash
# 在 .env 中配置飞书 API
FEISHU_APP_ID=your_feishu_app_id_here
FEISHU_APP_SECRET=your_feishu_app_secret_here
FEISHU_RECEIVE_ID=your_receive_id_here
```

**使用方式**:
1. 语音: "拍照发送到飞书"
2. 或在 Web Client 点击 "✈️ 推送飞书"

### 3. 首次开箱配置（AP 模式）

> **适用场景**: 商业产品用户首次使用，或设备重置后

LeLamp 支持开箱即用配置，无需 SSH 或命令行操作：

#### 3.1 自动进入设置模式

当设备首次启动或检测到未配置 WiFi 时：

1. **自动启动 AP 热点** `LeLamp-Setup`
   - 密码: `lelamp123`
   - IP 地址: `192.168.4.1`

2. **LED 指示灯** 呈蓝色闪烁

3. **用户手机/电脑连接热点**

4. **浏览器自动弹出** 配置页面（或访问 `http://192.168.4.1/setup`）

#### 3.2 设置向导流程

```
步骤 1: 欢迎页面
       ↓
步骤 2: 扫描并选择 WiFi 网络
       ↓
步骤 3: 输入 WiFi 密码
       ↓
步骤 4: 连接验证
       ↓
步骤 5: 配置完成，设备自动重启
```

#### 3.3 手动触发设置模式

如果需要重新配置 WiFi：

**方法 1: Web 界面**
```bash
# 在设置页面点击 "重新配置" 或 "重置网络"
```

**方法 2: API 调用**
```bash
curl -X POST http://localhost:8000/api/setup/reset
```

#### 3.4 设置模式 API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/setup/status` | GET | 获取设置状态 |
| `/api/setup/ap/start` | POST | 启动 AP 模式 |
| `/api/setup/ap/stop` | POST | 停止 AP 模式 |
| `/api/setup/complete` | POST | 完成配置并重启 |
| `/api/setup/reset` | POST | 重置配置 |
| `/api/setup/ap/clients` | GET | 获取已连接的客户端 |

#### 3.5 配置状态文件

```bash
# 状态文件位置
/var/lib/lelamp/setup_status.json

# 查看状态
cat /var/lib/lelamp/setup_status.json
```

```json
{
  "setup_completed": true,
  "setup_completed_at": "2026-03-18T12:00:00",
  "wifi_ssid": "MyHomeWiFi",
  "last_mode": "client",
  "last_updated": "2026-03-18T12:00:00"
}
```

#### 3.6 配置完成后获取 IP 地址

> **为什么需要 IP 地址？**
> - 访问 Web 设置页面 `http://树莓派IP:8000`
> - 进行网络故障排查
> - SSH 远程连接（开发者用户）

AP 模式配置完成后，设备会自动连接到您指定的 WiFi 网络。此时需要获取设备在家庭网络中的 IP 地址。

**方法 1: 路由器管理页面（最简单）**

适合所有用户，无需额外工具：

1. 登录路由器管理页面
   - 通常地址为 `192.168.0.1`、`192.168.1.1` 或 `192.168.31.1`
   - 查看路由器底部标签获取默认地址和密码

2. 找到"已连接设备"或"DHCP 客户端列表"
   - 不同路由器位置不同：
     - TP-Link: 网络设置 → DHCP 客户端列表
     - 小米/华硕: 终端设备
     - 网件: 连接的设备

3. 查找名为以下之一的设备：
   - `lelamp-local`
   - `raspberrypi`
   - 或不认识的设备（制造商显示为 Raspberry Pi Foundation）

**方法 2: 网络扫描 App（手机用户）**

推荐使用 **Fing App**（iOS/Android 通用）：

1. 在 App Store 或 Google Play 下载 "Fing - Network Tools"
2. 连接到与树莓派相同的 WiFi
3. 打开 App，点击"扫描网络"
4. 在设备列表中查找名为 `lelamp-local` 或 `raspberrypi` 的设备
5. 记录显示的 IP 地址（如 `192.168.0.100`）

**方法 3: 直接连接显示器（最直接）**

需要显示器和键盘：

1. 将显示器连接到树莓派的 HDMI 接口
2. 连接 USB 键盘
3. 启动树莓派，登录系统
4. 在终端输入：
   ```bash
   hostname -I
   ```
5. 输出即为 IP 地址，例如：`192.168.0.100`

**验证连接成功**：
```bash
# 在电脑或手机上测试
ping 192.168.0.100

# 访问 Web 设置页面
# http://192.168.0.100:8000
```

### 4. 动作表情

#### 预设动作列表

| 动作 | 说明 | 录制文件 | 使用场景 |
|------|------|---------|---------|
| 👍 点头 (nod) | 上下点头表示同意 | nod.csv | 肯定回答、表示赞同 |
| 👎 摇头 (shake) | 左右摇头表示否定 | shake.csv | 否定回答、表示不同意 |
| 🎉 兴奋 (excited) | 快速摆动表示兴奋 | excited.csv | 听到好消息、庆祝 |
| 😴 睡觉 (sleep) | 缓慢低头表示困倦 | sleep.csv | 晚上睡前、表示疲惫 |
| 💃 跳舞 (dance) | 有节奏的摆动 | dance.csv | 听音乐、愉快心情 |
| 🤔 思考 (think) | 缓慢转动表示思考 | think.csv | 回答问题前、沉思 |

**使用方式 1: 语音指令**
```
用户: "点个头"
台灯: (执行点头动作)

用户: "跳个舞"
台灯: (执行跳舞动作) "嘿嘿，本灯的舞姿还不错吧？"
```

**使用方式 2: Web Client 按钮**
在 "🎭 动作表情" Tab 中点击对应按钮

**使用方式 3: 自动触发**
LeLamp 会根据对话内容自动做出表情：
```
用户: "你真棒！"
台灯: "那是当然啦~" (自动执行 excited 动作)

用户: "我做错了..."
台灯: "没关系，下次注意就好。" (自动执行 nod 动作)
```

### 4. 灯光控制

#### 4.1 纯色灯光
**使用方式 1: 语音指令**
```
用户: "打开红色灯光"
台灯: (RGB 变为红色)

用户: "变成蓝色"
台灯: (RGB 变为蓝色)

用户: "关灯"
台灯: (RGB 熄灭)
```

**使用方式 2: Web Client 调色盘**
1. 切换到 "💡 灯光魔法" Tab
2. 使用颜色选择器选择任意颜色
3. 点击 "设置" 按钮

**使用方式 3: 快速颜色**
点击预设颜色色块快速切换：
- 🔴 暖红
- 💖 粉红
- 🟠 橙色
- 🟡 金黄
- 🟢 浅绿
- 🔵 天蓝
- 🟣 紫色
- ⚪ 暖白 (护眼模式)

#### 4.2 灯效动画

| 灯效 | 说明 | 使用场景 | 触发方式 |
|------|------|---------|---------|
| 💗 呼吸灯 (breathing) | 灯光缓慢呼吸 | 睡前使用、放松 | 语音/按钮 |
| 🌈 彩虹 (rainbow) | 彩虹流动效果 | 活泼欢快、庆祝 | 语音/按钮 |
| 🌊 波浪 (wave) | 波浪起伏效果 | 平静舒缓 | 语音/按钮 |
| 🔥 火焰 (fire) | 火焰跳动效果 | 温暖热情 | 语音/按钮 |
| 🎆 烟花 (fireworks) | 烟花绽放效果 | 节日庆祝 | 语音/按钮 |
| ⭐ 星空 (starry) | 星空闪烁效果 | 浪漫梦幻 | 语音/按钮 |

**语音触发**:
```
用户: "来个彩虹灯效"
台灯: (启动彩虹效果)

用户: "停止灯效"
台灯: (停止当前效果)
```

**Web Client 触发**:
在 "💡 灯光魔法" Tab 中点击对应灯效按钮

### 5. 联网搜索

**功能**: 查询实时信息 (新闻、天气、百科)

**前置条件**:
```bash
# 在 .env 中配置博查 API Key
BOCHA_API_KEY=your_bocha_api_key_here
```

**使用方式**:
```
用户: "今天天气怎么样？"
台灯: (调用博查 API) "今天北京晴，最高温度 15℃，最低 5℃。适合出门玩耍哦！"

用户: "LeLamp 是什么？"
台灯: (调用搜索 API) "LeLamp 是一个开源的机器人台灯项目..."
```

**速率限制**:
- 每秒最多 2 次请求
- 最大缓存 5 个请求令牌

### 6. 系统管理

#### 6.1 Web 设置页面

**新增功能**: 通过 Web 前端进行系统配置，无需编辑 `.env` 文件或 SSH 到树莓派。

**访问方式**:
1. 在 Web 控制台点击右上角 "设置" 按钮
2. 或直接访问 `http://your-web-client-url/settings`

**设置分类**:

| 分类 | 功能 | 说明 |
|------|------|------|
| 📶 WiFi 网络 | 扫描/连接 WiFi | 无线网络配置 |
| 🤖 LLM 模型 | DeepSeek 配置 | API Key、模型名称、API 地址 |
| 👁️ 视觉识别 | ModelScope 配置 | API Key、模型、超时设置 |
| 📹 摄像头 | 分辨率/旋转/翻转 | 图像参数调整 |
| 🎤 语音配置 | TTS 音调/语速 | 百度语音参数 |
| ⚙️ 硬件配置 | LED 亮度/串口/设备 ID | 硬件参数设置 |
| 🎭 行为配置 | 问候语/噪音消除/冷却时间 | 行为模式调整 |
| 🎨 界面设置 | 主题/语言/通知/音量 | UI 偏好设置 |

**配置流程**:
1. 在设置页面修改参数
2. 点击 "保存配置"
3. 点击 "立即重启" 使配置生效
4. 服务重启后自动应用新配置

**注意事项**:
- ⚠️ 配置修改后需要重启服务才能生效
- ⚠️ WiFi 连接时请勿断开当前网络，否则无法访问设置页面
- 🔒 API Key 显示时会自动隐藏 (如 `sk-***abc123`)

#### 6.2 WiFi 配置详解

**扫描网络**:
1. 进入 "WiFi 网络" 设置页
2. 点击 "扫描网络" 按钮
3. 等待扫描完成，显示附近可用网络

**连接新网络**:
1. 在网络列表中找到目标网络
2. 点击 "连接" 按钮
3. 如果需要密码，输入 WiFi 密码
4. 等待连接成功

**网络信息**:
- **信号强度**: 0-100%，颜色区分 (绿色=强，黄色=中，红色=弱)
- **加密方式**: WPA2/WPA3/WEP/open
- **频段**: 2.4GHz 或 5GHz

**断开连接**:
- 点击 "断开连接" 按钮可断开当前 WiFi

#### 6.3 服务重启

**重启方式**:
1. 在设置页面修改配置后，顶部会显示警告提示
2. 点击 "立即重启" 按钮
3. 确认重启操作
4. 服务将在 3 秒后自动重启

**重启后**:
- Web 客户端会自动跳转到连接页面
- 重新连接后新配置即生效

#### 6.4 音量控制
```
用户: "把音量调到 80"
台灯: "好的，已设置音量为 80%"
```

#### 6.2 OTA 更新
```
用户: "检查更新"
台灯: "发现新版本 1.2.0！\n更新内容：修复作业检查 Bug\n请问是否需要现在更新？"

用户: "确认更新"
台灯: (执行更新) "更新成功！服务将在 5 秒后重启。"
```

---

## 故障排查

### 问题 1: 连接失败

**症状**: Web Client 显示 "连接失败: Connection timeout"

**可能原因**:
1. API 地址错误
2. FastAPI 服务未启动
3. 网络连接问题

**解决方案**:

**步骤 1: 检查 FastAPI 服务状态**
```bash
# 查看服务是否在运行
ps aux | grep "uvicorn"

# 如果没有运行,启动服务
cd ~/lelamp_runtime
sudo uv run uvicorn lelamp.api.app:app --host 0.0.0.0 --port 8000
```

**步骤 2: 检查 API 地址**
```bash
# 确认地址格式正确
# 应该是: http://<树莓派IP>:8000
# 例如: http://192.168.1.100:8000
```

**步骤 3: 验证服务可访问**
```bash
# 在浏览器访问
curl http://localhost:8000/health
# 或访问 API 文档
http://localhost:8000/docs
```

**步骤 4: 检查网络连接**
```bash
# 测试网络连通性
ping <树莓派IP>

# 检查防火墙
sudo ufw status
```

### 问题 2: 视频不显示

**症状**: 连接成功但看不到视频画面

**可能原因**:
1. 摄像头权限未授予
2. 摄像头设备不存在
3. VisionService 未启动

**解决方案**:

**步骤 1: 检查摄像头设备**
```bash
# 列出摄像头设备
ls -l /dev/video*

# 如果没有 /dev/video0,检查 USB 连接
lsusb | grep -i camera
```

**步骤 2: 检查权限**
```bash
# 将 pi 用户添加到 video 组
sudo usermod -a -G video pi

# 重新登录生效
```

**步骤 3: 检查 VisionService 日志**
```bash
# 查看 Agent 日志
tail -f /var/log/lelamp.log | grep "VisionService"

# 预期输出:
# INFO:root:VisionService started
```

**步骤 4: 测试摄像头**
```bash
# 使用 ffmpeg 测试摄像头
ffplay /dev/video0

# 或使用 Python 测试
python3 -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
# 输出: True
```

### 问题 3: 电机不动

**症状**: 点击动作按钮但电机没有反应

**可能原因**:
1. 电机未校准
2. 串口权限不足
3. 电机电源未接通
4. 录制文件不存在

**解决方案**:

**步骤 1: 检查串口连接**
```bash
# 查找串口设备
ls -l /dev/ttyACM*

# 输出示例:
# crw-rw---- 1 root dialout 166, 0 Mar 16 10:00 /dev/ttyACM0
```

**步骤 2: 检查串口权限**
```bash
# 将 pi 用户添加到 dialout 组
sudo usermod -a -G dialout pi

# 重新登录生效
```

**步骤 3: 校准电机**
```bash
# 查找串口
uv run lerobot-find-port

# 设置电机 ID
uv run -m lelamp.setup_motors --id lelamp --port /dev/ttyACM0

# 校准电机
sudo uv run -m lelamp.calibrate --id lelamp --port /dev/ttyACM0
```

**步骤 4: 检查录制文件**
```bash
# 列出所有录制文件
uv run -m lelamp.list_recordings --id lelamp

# 输出示例:
# Available recordings for lamp_id=lelamp:
# - nod.csv (188 rows)
# - shake.csv (31 rows)
# - excited.csv (456 rows)
# ...
```

### 问题 4: 语音识别不工作

**症状**: 说话后没有反应或识别不准确

**可能原因**:
1. 麦克风未正确连接
2. 音量太小
3. Baidu Speech API 配置错误
4. 网络连接问题

**解决方案**:

**步骤 1: 测试麦克风**
```bash
# 录音测试 (5 秒)
arecord -d 5 test.wav

# 播放录音
aplay test.wav

# 检查音量
alsamixer
```

**步骤 2: 检查 Baidu API 配置**
```bash
# 验证 API Key
grep "BAIDU_SPEECH_API_KEY" .env

# 测试 API 连接
curl -X POST "https://vop.baidu.com/server_api" \
  -d "format=pcm&rate=16000&channel=1&cuid=lelamp&token=YOUR_TOKEN" \
  --data-binary @test.pcm
```

**步骤 3: 调整输入增益**
```bash
# 在 .env 中增加增益
LELAMP_STT_INPUT_GAIN=5.0  # 默认 3.0,可调到 5.0
```

**步骤 4: 检查噪声消除**
```bash
# 如果环境安静,可禁用噪声消除
LELAMP_NOISE_CANCELLATION=false
```

### 问题 5: LED 灯不亮

**症状**: 灯光指令无效,LED 保持黑色

**可能原因**:
1. LED 硬件未连接
2. GPIO 权限不足
3. 亮度设置为 0

**解决方案**:

**步骤 1: 检查 GPIO 权限**
```bash
# 运行测试脚本 (需要 sudo)
sudo uv run -m lelamp.test.test_rgb

# 预期输出:
# Testing RGB LED...
# Setting color to red (255, 0, 0)
# ✅ RGB test passed
```

**步骤 2: 检查硬件连接**
- LED 数据线连接到 GPIO 12 (PWM0)
- 电源线连接到 5V
- 地线连接到 GND

**步骤 3: 调整亮度**
```bash
# 在 .env 中增加亮度
LELAMP_LED_BRIGHTNESS=50  # 默认 25,可调到 50
```

---

## 最佳实践

### 1. 性能优化

#### 降低延迟
```bash
# 增加摄像头 FPS (默认 30)
LELAMP_MATRIX_FPS=60

# 减少视觉采集间隔 (默认 2.5 秒)
LELAMP_VISION_CAPTURE_INTERVAL_S=1.5

# 提高 STT 输入增益 (更快识别)
LELAMP_STT_INPUT_GAIN=4.0
```

#### 降低功耗
```bash
# 降低 LED 亮度
LELAMP_LED_BRIGHTNESS=15

# 禁用启动动画
LELAMP_BOOT_ANIMATION=0

# 增加视觉采集间隔
LELAMP_VISION_CAPTURE_INTERVAL_S=5.0
```

### 2. 安全建议

#### 隐私保护
```bash
# 限制视觉帧存活时间 (默认 15 秒)
LELAMP_VISION_MAX_AGE_S=10.0

# 降低 JPEG 质量 (默认 92)
LELAMP_VISION_JPEG_QUALITY=80

# 启用摄像头 LED 指示灯 (已默认启用)
```

#### API 安全
- ✅ 使用环境变量存储 API Key
- ✅ 不要提交 `.env` 文件到 Git
- ✅ 生产环境配置 HTTPS
- ✅ 限制 API 访问权限

#### 设备授权
```bash
# 生产环境必须配置授权密钥
LELAMP_LICENSE_KEY=your_license_key_here

# 许可证签名密钥（生产环境必需，务必保密）
LELAMP_LICENSE_SECRET=your_strong_random_secret_here

# 开发环境可跳过 (不推荐)
LELAMP_DEV_MODE=1
```

### 3. 生产部署

#### Systemd 服务
```bash
# 创建服务文件
sudo nano /etc/systemd/system/lelamp.service
```

**服务配置**:
```ini
[Unit]
Description=Lelamp Runtime Service
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/lelamp_runtime
ExecStart=/usr/bin/sudo uv run main.py console
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

**启动服务**:
```bash
sudo systemctl daemon-reload
sudo systemctl enable lelamp.service
sudo systemctl start lelamp.service

# 查看状态
sudo systemctl status lelamp.service

# 查看日志
sudo journalctl -u lelamp.service -f
```

#### 自动 OTA 更新
```bash
# 配置 OTA 服务器
LELAMP_OTA_URL=https://api.lelamp.com/ota/check

# 台灯会自动检查更新 (每 24 小时)
# 或通过语音触发: "检查更新"
```

---

## 常见问题

### Q1: 需要启动哪些服务？
**A**: LeLamp 有三种运行模式：

| 模式 | 命令 | 功能 | 是否必需 |
|------|------|------|----------|
| **FastAPI 服务** | `uvicorn lelamp.api.app:app` | Web 控制面板、设备控制、文字聊天 | ✅ 必需 |
| **Console 模式** | `main.py console` | 本地语音测试（系统麦克风/扬声器） | ⚪ 可选 |
| **Room 模式** | `main.py dev` | 远程语音访问（手机 App） | ⚪ 可选 |

**推荐**：
- 开发测试：FastAPI + Console 模式
- 生产部署：FastAPI + Room 模式

**注意**：
- Console 模式**不需要 LiveKit 配置**
- Room 模式需要 LiveKit 配置

### Q2: 可以同时连接多个客户端吗？
**A**: 可以。多个浏览器可以同时连接到 FastAPI 服务，每个客户端都能：
- 控制设备
- 发送文字聊天
- 查看设备状态
- 接收状态更新（WebSocket 广播）

### Q3: 如何录制新动作？
**A**: 使用录制功能：
```bash
# 进入录制模式
uv run -m lelamp.record --id lelamp --port /dev/ttyACM0 --name my_action

# 手动摆动台灯到想要的动作
# 按 Ctrl+C 停止录制

# 录制文件保存在 lelamp/recordings/my_action.csv
```

### Q4: 视觉识别准确率如何提高？
**A**: 优化技巧：
1. 光线充足 (自然光最佳)
2. 物体距离摄像头 20-30cm
3. 背景简洁 (避免杂物)
4. 作业本平放 (避免倾斜)
5. 增加分辨率 (如果性能允许):
   ```bash
   LELAMP_CAMERA_WIDTH=1920
   LELAMP_CAMERA_HEIGHT=1080
   ```

### Q5: 如何禁用某些功能？
**A**: 通过环境变量禁用：
```bash
# 禁用视觉功能
LELAMP_VISION_ENABLED=false

# 禁用启动动画
LELAMP_BOOT_ANIMATION=0

# 禁用噪声消除
LELAMP_NOISE_CANCELLATION=false
```

### Q6: 支持哪些浏览器？
**A**: 支持现代浏览器 (WebRTC):
- ✅ Chrome 74+
- ✅ Firefox 66+
- ✅ Safari 12.1+
- ✅ Edge 79+
- ❌ IE (不支持 WebRTC)

### Q7: 移动端如何使用？
**A**: Web Client 支持移动端浏览器：
- 📱 在手机浏览器访问 Web Client URL
- 📱 自动适配响应式布局
- 📱 支持触摸操作

**未来**: 计划开发 Flutter 移动 App (iOS/Android)

### Q8: 如何贡献代码？
**A**: 欢迎贡献！
1. Fork 项目: https://github.com/xwang152-jack/lelamp_runtime
2. 创建功能分支: `git checkout -b feature/AmazingFeature`
3. 提交更改: `git commit -m 'Add some AmazingFeature'`
4. 推送到分支: `git push origin feature/AmazingFeature`
5. 创建 Pull Request

---

## 附录

### A. 录制文件格式

**CSV 格式**:
```csv
timestamp,base_yaw.pos,base_pitch.pos,elbow_pitch.pos,wrist_roll.pos,wrist_pitch.pos
0.000,0.000,-20.000,60.000,0.000,0.000
0.033,5.000,-18.000,58.000,2.000,0.000
0.067,10.000,-16.000,56.000,4.000,0.000
...
```

**字段说明**:
- `timestamp`: 时间戳 (秒)
- `base_yaw.pos`: 底座水平旋转角度 (-180 ~ 180°)
- `base_pitch.pos`: 底座俯仰角度 (-90 ~ 90°)
- `elbow_pitch.pos`: 肘部俯仰角度 (-150 ~ 150°)
- `wrist_roll.pos`: 腕部滚转角度 (-180 ~ 180°)
- `wrist_pitch.pos`: 灯头俯仰角度 (-90 ~ 90°)

### B. 环境变量完整列表

参见 `.env.example` 文件获取完整的环境变量配置说明

### C. 错误代码对照表

| 错误代码 | 说明 | 解决方案 |
|---------|------|---------|
| E001 | 串口连接失败 | 检查 USB 连接和权限 |
| E002 | 摄像头未找到 | 检查摄像头连接 |
| E003 | API Key 无效 | 检查 .env 配置 |
| E004 | 网络连接超时 | 检查网络连接 |
| E005 | 授权验证失败 | 检查 LELAMP_LICENSE_KEY |
| E006 | OTA 更新失败 | 检查更新服务器连接 |

### D. 性能基准

| 指标 | 数值 | 说明 |
|------|------|------|
| 视频延迟 | < 200ms | LiveKit WebRTC |
| 语音识别延迟 | < 500ms | Baidu Speech |
| 视觉识别延迟 | 3-8 秒 | Qwen VL API |
| 动作响应时间 | < 100ms | 本地电机控制 |
| 灯光响应时间 | < 50ms | 本地 LED 控制 |
| CPU 占用率 | 30-50% | Raspberry Pi 4B |
| 内存占用 | 800MB - 1.2GB | Python + Services |

---

**文档版本**: v0.1.0
**最后更新**: 2026-03-19
**作者**: LeLamp 开发团队
**许可证**: 参见主项目许可证
