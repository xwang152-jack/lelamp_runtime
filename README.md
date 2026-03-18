# LeLamp Runtime v2.0

![](./assets/images/Banner.png)

**LeLamp Runtime** 是一个完整的 Python 控制系统，为 [LeLamp 机器人台灯](https://github.com/humancomputerlab/LeLamp)提供对话式 AI、视觉识别、动作表情、灯光效果等功能。基于 [Apple 的 Elegnt 研究](https://machinelearning.apple.com/research/elegnt-expressive-functional-movement)，由 [[Human Computer Lab]](https://www.humancomputerlab.com/) 开发。

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-See%20LeLamp%20Repo-green.svg)](https://github.com/humancomputerlab/LeLamp)
[![UV](https://img.shields.io/badge/package%20manager-UV-orange.svg)](https://github.com/astral-sh/uv)

> 📖 **新用户?** 查看 [用户使用指南](USER_GUIDE.md) | **开发部署?** 查看 [部署指南](DEPLOYMENT_GUIDE.md)

---

## ✨ 主要特性

### 🎙️ 对话式 AI
- **语音交互**: 基于 LiveKit 的实时语音对话
- **中文支持**: 百度语音识别和合成
- **LLM 驱动**: DeepSeek 大语言模型提供智能对话
- **状态指示**: LED 灯光指示倾听、思考、说话状态

### 👀 视觉识别
- **物体识别**: 识别日常物品、场景、文字
- **作业检查**: AI 批改数学题、语文题
- **飞书推送**: 拍照并发送到飞书群组
- **隐私保护**: LED 指示灯 + 用户同意机制

### 🎭 动作表情
- **预设动作**: 点头、摇头、兴奋、睡觉、跳舞、思考
- **录制回放**: 录制自定义动作并回放
- **自动触发**: 根据对话内容自动做出表情
- **动作冷却**: 防止过度运动保护硬件

### 💡 灯光效果
- **纯色控制**: 调色盘选择任意颜色
- **灯效动画**: 呼吸、彩虹、波浪、火焰、烟花、星空
- **状态指示**: 对话状态自动切换颜色
- **隐私指示**: 摄像头激活时红色警示

### 🌐 Web 客户端
- **浏览器控制**: 现代化 Web 界面
- **实时视频**: WebRTC 视频流
- **双向音频**: 语音通话功能
- **全功能面板**: 视觉、动作、灯光、聊天全覆盖

### 🚀 RESTful API 系统 ⭐ NEW
- **完整的 REST API**: 9 个端点，完整的 CRUD 操作
- **实时 WebSocket 推送**: 13 种消息类型，频道订阅
- **数据持久化**: SQLite/PostgreSQL 支持，4 个 ORM 模型
- **自动 API 文档**: Swagger UI + ReDoc
- **99% 测试覆盖率**: 79 个测试全部通过

### 🔐 企业级功能
- **设备授权**: License Key 保护
- **OTA 更新**: 远程固件升级
- **舵机健康监控**: 温度、电压、负载实时监控 ⭐ NEW
- **远程 PID 调参**: 无需上门即可优化动作性能 ⭐ NEW
- **隐私保护**: 摄像头使用同意机制
- **速率限制**: API 调用保护
- **响应缓存**: TTL 缓存减少重复调用

---

## 📊 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    用户设备 (User Device)                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │          Web Browser / Mobile App                    │   │
│  │  - 视频预览 | 双向语音 | 控制面板 | 实时对话          │   │
│  └──────────────────┬──────────────────────────────────┘   │
└─────────────────────┼──────────────────────────────────────┘
                      │ WebSocket + WebRTC
                      ↓
┌─────────────────────────────────────────────────────────────┐
│               LiveKit Cloud / Self-hosted                   │
│  - SFU (Selective Forwarding Unit)                          │
│  - 实时音视频流转发 | Data Channel 消息传输                  │
└──────────────────┬──────────────────────────────────────────┘
                   │ WebSocket + DTLS/SRTP
                   ↓
┌─────────────────────────────────────────────────────────────┐
│          Raspberry Pi (LeLamp Runtime)                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  LeLamp Agent (main.py)                             │   │
│  │  - DeepSeek LLM (对话引擎)                           │   │
│  │  - Qwen VL (视觉识别)                                │   │
│  │  - Baidu Speech (STT/TTS)                            │   │
│  │  - LiveKit Agents SDK (实时通信)                     │   │
│  └─────────┬───────────────────────────────────────────┘   │
│            │ Priority-based Event Dispatch                  │
│  ┌─────────┴───────────────────────────────────────────┐   │
│  │  Services (多线程架构)                               │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────┐  │   │
│  │  │ MotorsService│  │ RGBService   │  │ Vision   │  │   │
│  │  │ (5轴电机)    │  │ (64颗LED)    │  │ Service  │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────┘  │   │
│  └─────────────────────────────────────────────────────┘   │
│            │                 │                 │            │
│  ┌─────────┴─────────────────┴─────────────────┴───────┐   │
│  │  Hardware (硬件层)                                   │   │
│  │  - Feetech 伺服电机 | WS2812B LED | USB 摄像头       │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**核心技术栈**:
- 🐍 Python 3.12+
- 🎙️ LiveKit (实时通信)
- 🤖 DeepSeek (大语言模型)
- 👁️ Qwen VL (视觉识别)
- 🗣️ Baidu Speech (语音服务)
- 🔧 UV (包管理器)

---

## 🚀 快速开始

### 前置要求

#### 硬件
- ✅ Raspberry Pi 4B+ (推荐 4GB RAM)
- ✅ LeLamp 硬件套件 (电机、LED、摄像头)
- ✅ 网络连接 (Wi-Fi / 有线)

#### 软件
- ✅ Python 3.12+
- ✅ UV Package Manager
- ✅ LiveKit Account (或自托管 LiveKit Server)

#### 服务
- ✅ DeepSeek API Key
- ✅ ModelScope API Key (视觉功能)
- ✅ Baidu Speech API Key (语音服务)

### 安装步骤

#### 1. 克隆项目
```bash
git clone https://github.com/humancomputerlab/lelamp_runtime.git
cd lelamp_runtime
```

#### 2. 安装 UV (如果未安装)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 3. 安装依赖
```bash
# Raspberry Pi (包含硬件依赖)
uv sync --extra hardware

# 开发机 (仅电机控制,无硬件依赖)
uv sync
```

**提示**: 如果遇到 LFS 问题:
```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

#### 4. 配置环境变量
```bash
cp .env.example .env
nano .env
```

**最小配置** (必填):
```bash
# LiveKit (实时通信)
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret

# DeepSeek (LLM)
DEEPSEEK_API_KEY=your_deepseek_key

# Baidu Speech (语音服务)
BAIDU_SPEECH_API_KEY=your_baidu_api_key
BAIDU_SPEECH_SECRET_KEY=your_baidu_secret_key

# ModelScope (视觉识别 - 可选)
MODELSCOPE_API_KEY=your_modelscope_key
```

#### 5. 硬件校准 (首次使用)
```bash
# 查找串口
uv run lerobot-find-port

# 设置电机 ID
uv run -m lelamp.setup_motors --id lelamp --port /dev/ttyACM0

# 校准电机
sudo uv run -m lelamp.calibrate --id lelamp --port /dev/ttyACM0
```

#### 6. 初始化数据库 (可选，用于设置界面)
```bash
# 创建数据库表
uv run python -c "
from lelamp.database.session import engine
from lelamp.database.models import Base
Base.metadata.create_all(bind=engine)
print('✅ 数据库初始化完成')
"
```

**注意**: 如果不使用 Web 设置界面，可以跳过此步骤。

#### 7. 启动 LeLamp Agent
```bash
sudo uv run main.py console
```

**预期输出**:
```
INFO:root:config ready: lamp_id=lelamp port=/dev/ttyACM0 vision=True
INFO:root:MotorsService started
INFO:root:RGBService started
INFO:root:VisionService started
INFO:livekit:Connected to LiveKit
```

#### 7. 启动 API 服务器
```bash
# 方式1: 使用 UV (推荐)
uv run uvicorn lelamp.api.app:app --host 0.0.0.0 --port 8000 --reload

# 方式2: 直接使用 Python
.venv/bin/python -m uvicorn lelamp.api.app:app --host 0.0.0.0 --port 8000

# 生产环境 (多进程)
.venv/bin/python -m uvicorn lelamp.api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

**API 端点**:
- 健康检查: `http://localhost:8000/health`
- API 文档: `http://localhost:8000/docs`
- WebSocket: `ws://localhost:8000/api/ws/{lamp_id}`

#### 8. 启动 Web Client
```bash
# 开发模式 (热更新)
cd web
npm run dev

# 生产构建
npm run build

# 访问: http://localhost:5173
```

在 Web Client 中输入服务器地址 `http://localhost:8000`，点击"连接设备"即可开始使用！🎉

---

## 📚 核心功能

### 1. 语音对话

**使用方式**: 直接对着台灯说话

**对话状态指示**:
- 🤍 **白色**: 空闲状态 (Idle)
- 🔵 **蓝色**: 正在倾听 (Listening)
- 🟣 **紫色**: 思考中 (Thinking)
- 🌈 **随机彩色**: 正在说话 (Speaking)

**示例对话**:
```
用户: "你好"
台灯: "你好呀，小主人！本灯又来陪你啦~"

用户: "现在几点了"
台灯: "现在是下午 3:45，怎么，还没写完作业就想着玩了？"

用户: "讲个笑话"
台灯: "好吧...为什么程序员总是搞混万圣节和圣诞节？因为 Oct 31 == Dec 25！"
```

---

### 2. 视觉识别

#### 拍照识别
```
用户: "这是什么？" (举起一个苹果)
台灯: "这是一个红苹果，看起来很新鲜。"
```

#### 检查作业
```
用户: "帮我检查作业"
台灯: "好的，让我看看..."

结果:
✅ 第1题: 5 + 3 = 8 (正确)
✅ 第2题: 12 - 7 = 5 (正确)
❌ 第3题: 6 × 4 = 32 (错误，应该是 24)

正确率: 67% (2/3)
```

#### 推送飞书
```
用户: "拍照发送到飞书"
台灯: "好的，正在拍照..." (拍照并发送到飞书群组)
```

---

### 3. 动作表情

**6 个预设动作**:
- 👍 **点头** (nod): 表示同意
- 👎 **摇头** (shake): 表示否定
- 🎉 **兴奋** (excited): 快速摆动
- 😴 **睡觉** (sleep): 缓慢低头
- 💃 **跳舞** (dance): 有节奏摆动
- 🤔 **思考** (think): 缓慢转动

**使用方式**:
```bash
# 语音指令
用户: "点个头"
台灯: (执行点头动作)

# Web Client 按钮
点击 "🎭 动作表情" Tab 中的按钮

# 自动触发
用户: "你真棒！"
台灯: (自动执行 excited 动作)
```

**录制自定义动作**:
```bash
# 录制新动作
uv run -m lelamp.record --id lelamp --port /dev/ttyACM0 --name my_action

# 回放动作
uv run -m lelamp.replay --id lelamp --port /dev/ttyACM0 --name my_action

# 列出所有录制
uv run -m lelamp.list_recordings --id lelamp
```

---

### 4. 灯光效果

#### 纯色灯光
- 🔴 暖红
- 💖 粉红
- 🟠 橙色
- 🟡 金黄
- 🟢 浅绿
- 🔵 天蓝
- 🟣 紫色
- ⚪ 暖白 (护眼模式)

#### 灯效动画
- 💗 **呼吸灯**: 缓慢呼吸
- 🌈 **彩虹**: 彩虹流动
- 🌊 **波浪**: 波浪起伏
- 🔥 **火焰**: 火焰跳动
- 🎆 **烟花**: 烟花绽放
- ⭐ **星空**: 星空闪烁

**使用方式**:
```bash
# 语音指令
用户: "打开红色灯光"
台灯: (RGB 变为红色)

用户: "来个彩虹灯效"
台灯: (启动彩虹效果)

# Web Client 控制
在 "💡 灯光魔法" Tab 中使用调色盘或点击灯效按钮
```

---

## 🧪 硬件测试

### RGB LED 测试
```bash
sudo uv run -m lelamp.test.test_rgb
```

### 音频系统测试
```bash
uv run -m lelamp.test.test_audio
```

### 电机测试
```bash
uv run -m lelamp.test.test_motors --id lelamp --port /dev/ttyACM0
```

---

## ⚙️ 配置说明

### 环境变量

完整的环境变量配置参见 `.env.example`。以下是核心配置：

#### LiveKit 配置
```bash
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=APIxxxxxxxxxxxx
LIVEKIT_API_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**获取方式**:
```bash
lk app env -w
cat .env.local
```

#### DeepSeek LLM
```bash
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
DEEPSEEK_MODEL=deepseek-chat  # 可选
```

#### Baidu Speech
```bash
BAIDU_SPEECH_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxx
BAIDU_SPEECH_SECRET_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
BAIDU_SPEECH_TTS_PER=4  # 0=度小美, 1=度小宇, 4=度丫丫
```

#### ModelScope 视觉
```bash
LELAMP_VISION_ENABLED=true
MODELSCOPE_API_KEY=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
MODELSCOPE_MODEL=Qwen/Qwen3-VL-235B-A22B-Instruct
```

#### 摄像头配置
```bash
LELAMP_CAMERA_INDEX_OR_PATH=0
LELAMP_CAMERA_WIDTH=1024
LELAMP_CAMERA_HEIGHT=768
LELAMP_CAMERA_ROTATE_DEG=0  # 0, 90, 180, 270
LELAMP_CAMERA_FLIP=none  # none, horizontal, vertical, both
```

#### 硬件配置
```bash
LELAMP_PORT=/dev/ttyACM0
LELAMP_ID=lelamp
LELAMP_LED_BRIGHTNESS=25  # 0-255
```

#### 行为配置
```bash
LELAMP_GREETING_TEXT="Hello! 小宝贝上线了."
LELAMP_BOOT_ANIMATION=1  # 启动动画
LELAMP_MOTION_COOLDOWN_S=2  # 动作冷却时间
LELAMP_LIGHT_OVERRIDE_S=10  # 灯光覆盖时长
```

#### 商业化配置
```bash
# 设备授权
LELAMP_LICENSE_KEY=your_license_key_here
LELAMP_DEV_MODE=1  # 开发模式 (跳过授权检查)

# OTA 更新
LELAMP_OTA_URL=https://api.lelamp.com/ota/check

# 舵机健康监控 ⭐ NEW
LELAMP_MOTOR_HEALTH_CHECK_ENABLED=true
LELAMP_MOTOR_HEALTH_CHECK_INTERVAL_S=300.0  # 5分钟检查一次
LELAMP_MOTOR_TEMP_WARNING_C=65.0            # 温度警告阈值
LELAMP_MOTOR_TEMP_CRITICAL_C=75.0           # 温度危险阈值
LELAMP_MOTOR_LOAD_STALL=0.95                # 堵转检测阈值
```

📖 **详细文档**: [舵机健康监控使用指南](./docs/MOTOR_HEALTH_MONITORING.md)

### 安全注意事项

⚠️ **重要**:
- 切勿提交 `.env` 文件到 Git
- 使用 `.env.example` 作为模板
- `LELAMP_LICENSE_SECRET` 必须是强随机密钥 (生产环境必需)
- 所有外部 API URL 会被验证防止 SSRF 攻击

---

## 🏭 生产部署

### Systemd 服务

创建服务文件:
```bash
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

### OTA 更新

配置 OTA 服务器:
```bash
LELAMP_OTA_URL=https://api.lelamp.com/ota/check
```

**检查更新**:
```
用户: "检查更新"
台灯: "发现新版本 1.2.0！是否现在更新？"

用户: "确认更新"
台灯: "更新成功！服务将在 5 秒后重启。"
```

---

## 📖 文档

### 🌟 **新用户文档**
- 📘 [用户使用指南](USER_GUIDE.md) - 完整的用户使用教程
- 🚀 [部署指南](DEPLOYMENT_GUIDE.md) - 详细的部署和配置说明
- 🔧 [设置快速参考](SETTINGS_QUICK_REFERENCE.md) - 设置界面快速参考

### 开发文档
- 🔧 [开发者指南](CLAUDE.md) - 代码架构和开发规范
- 🌐 [Web 前端](./web/) - Vue 3 前端应用
- ✅ [测试清单](./docs/TESTING_CHECKLIST.md) - 系统化测试流程

### 产品文档
- 📊 [产品评审报告](./docs/PRODUCT_APP_REVIEW.md) - 完整产品评审
- 🎨 [用户体验设计](./docs/PRODUCT_UX_JOURNEY.md) - UX Journey Map
- 🗺️ [技术实现路线图](./docs/PRODUCT_IMPLEMENTATION_ROADMAP.md) - 开发规划

### 技术文档
- 🐍 [Python 升级指南](./docs/PYTHON312_UPGRADE.md) - Python 3.12 升级
- 📦 [依赖管理](./pyproject.toml) - 项目配置
- 🔐 [安全指南](./docs/SECURITY_GUIDE.md) - 安全最佳实践

---

## 🏗️ 项目结构

```
lelamp_runtime/
├── main.py                      # 主入口 (LeLamp Agent)
├── pyproject.toml              # 项目配置和依赖
├── .env.example                # 环境变量模板
├── VERSION                     # 版本号
├── CLAUDE.md                   # 开发者指南
├── README.md                   # 本文档
│
├── docs/                       # 文档目录
│   ├── USER_GUIDE.md           # 完整使用指南
│   ├── TESTING_CHECKLIST.md   # 测试清单
│   ├── PRODUCT_*.md            # 产品评审文档
│   └── PYTHON*.md              # Python 升级文档
│
├── web/                       # Vue 3 前端应用
│   ├── index.html              # 主页面
│   ├── style.css               # 样式表
│   ├── app.js                  # 功能实现
│   └── README.md               # Web Client 文档
│
├── lelamp/                     # 核心包
│   ├── config.py               # 配置管理
│   ├── setup_motors.py         # 电机配置
│   ├── calibrate.py            # 电机校准
│   ├── record.py               # 动作录制
│   ├── replay.py               # 动作回放
│   ├── list_recordings.py      # 列出录制文件
│   │
│   ├── service/                # 服务架构
│   │   ├── base.py             # 服务基类
│   │   ├── motors.py           # 电机服务
│   │   ├── rgb.py              # RGB 服务
│   │   └── vision/             # 视觉服务
│   │       ├── service.py      # 视觉服务
│   │       └── privacy.py      # 隐私保护
│   │
│   ├── integrations/           # 外部服务集成
│   │   ├── baidu_speech.py     # 百度语音
│   │   ├── qwen_vl.py          # Qwen 视觉
│   │   ├── exceptions.py       # 异常处理
│   │   └── bocha.py            # 博查搜索
│   │
│   ├── utils/                  # 工具函数
│   │   ├── rate_limiter.py     # 速率限制
│   │   ├── security.py         # 设备授权
│   │   ├── url_validation.py   # URL 验证
│   │   └── ota.py              # OTA 更新
│   │
│   ├── cache/                  # 响应缓存
│   │   └── cache_manager.py    # 缓存管理
│   │
│   ├── recordings/             # 动作录制文件
│   │   ├── nod.csv             # 点头
│   │   ├── shake.csv           # 摇头
│   │   ├── excited.csv         # 兴奋
│   │   ├── sleep.csv           # 睡觉
│   │   ├── dance.csv           # 跳舞
│   │   ├── think.csv           # 思考
│   │   └── ...                 # 其他动作
│   │
│   ├── follower/               # Follower 模式配置
│   ├── leader/                 # Leader 模式配置
│   └── test/                   # 硬件测试模块
│       ├── test_rgb.py         # RGB 测试
│       ├── test_audio.py       # 音频测试
│       └── test_motors.py      # 电机测试
│
├── scripts/                    # 构建和工具脚本
│   ├── generate_client_token.py  # Token 生成
│   └── build_dist.sh           # 构建脚本
│
└── assets/                     # 资源文件
    └── images/                 # 图片资源
```

---

## 🔧 故障排查

### 问题 1: 连接失败

**症状**: Web Client 显示 "连接失败"

**解决方案**:
```bash
# 1. 检查 Agent 是否运行
ps aux | grep "main.py"

# 2. 检查 LiveKit URL
echo $LIVEKIT_URL

# 3. 重新生成 Token
uv run python scripts/generate_client_token.py --room lelamp-room --user test

# 4. 检查网络连接
curl -v https://your-project.livekit.cloud
```

### 问题 2: 视频不显示

**症状**: 连接成功但无视频

**解决方案**:
```bash
# 1. 检查摄像头设备
ls -l /dev/video*

# 2. 检查权限
sudo usermod -a -G video pi

# 3. 测试摄像头
ffplay /dev/video0
```

### 问题 3: 电机不动

**症状**: 动作按钮无响应

**解决方案**:
```bash
# 1. 检查串口连接
ls -l /dev/ttyACM*

# 2. 检查权限
sudo usermod -a -G dialout pi

# 3. 校准电机
sudo uv run -m lelamp.calibrate --id lelamp --port /dev/ttyACM0

# 4. 测试电机
uv run -m lelamp.test.test_motors --id lelamp --port /dev/ttyACM0
```

### 问题 4: LED 不亮

**症状**: 灯光指令无效

**解决方案**:
```bash
# 1. 测试 LED (需要 sudo)
sudo uv run -m lelamp.test.test_rgb

# 2. 调整亮度
# 在 .env 中设置:
LELAMP_LED_BRIGHTNESS=50
```

更多故障排查，请参见 [完整使用指南](./docs/USER_GUIDE.md)。

---

## 📊 性能指标

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

## 🤝 贡献

这是一个由 Human Computer Lab 开发的开源项目。欢迎通过 GitHub 仓库贡献代码。

### 贡献方式
1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

---

## 👥 维护者

由 [Human Computer Lab](https://www.humancomputerlab.com) 维护。

---

## 🙏 致谢与赞助

- 贡献者列表: [CONTRIBUTORS.md](./CONTRIBUTORS.md)
- 赞助商致谢: [SPONSORS.md](./SPONSORS.md)

---

## 📄 许可证

查看主 [LeLamp 仓库](https://github.com/humancomputerlab/LeLamp) 获取许可证信息。

---

## 🔗 相关链接

- 🏠 [LeLamp 主仓库](https://github.com/humancomputerlab/LeLamp)
- 📖 [LeLamp 控制教程](https://github.com/humancomputerlab/LeLamp/blob/master/docs/5.%20LeLamp%20Control.md)
- 🌐 [Human Computer Lab 官网](https://www.humancomputerlab.com)
- 📚 [LiveKit 文档](https://docs.livekit.io/agents/start/voice-ai/)
- 🤖 [DeepSeek API 文档](https://api.deepseek.com)

---

## ⭐ Star History

如果这个项目对你有帮助，请给我们一个 Star！⭐

---

**版本**: v2.0
**最后更新**: 2026-03-16
**Python 版本**: 3.12+
