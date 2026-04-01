# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
使用中文交流。

## Project Overview

LeLamp Runtime 是 LeLamp 机器人台灯的 Python 控制系统。基于 LiveKit 实时语音对话，集成 DeepSeek LLM、百度语音、Qwen VL 视觉、舵机控制、RGB LED 灯效。运行在 Raspberry Pi 上，macOS 可开发调试（无硬件功能降级为 NoOp）。

## Commands

```bash
# 安装依赖
uv sync                              # 基础依赖
uv sync --extra dev                  # + 测试/lint 工具
uv sync --extra api                  # + FastAPI/数据库
uv sync --extra vision               # + MediaPipe（仅 arm64 macOS / x86_64 Linux / AMD64 Windows）
GIT_LFS_SKIP_SMUDGE=1 uv sync       # LFS 问题时使用

# 运行 Agent
sudo uv run main.py console          # 本地文字对话（无需 LiveKit 客户端）
sudo uv run main.py start            # 启动 LiveKit Worker（需 LiveKit 服务端）

# 生成 LiveKit 客户端 Token（用于 Web/Mobile 客户端连接）
uv run python scripts/tools/generate_client_token.py --room lelamp-room --user user-app

# 测试
uv run pytest tests/ -v                              # 所有测试
uv run pytest tests/test_memory.py -v                # 记忆系统测试
uv run pytest tests/ -k "test_setup" -v              # 按名称匹配
uv run pytest tests/ --cov=lelamp --cov-report=html  # 覆盖率

# Lint & Format
uv run ruff check lelamp/
uv run ruff check --fix lelamp/
uv run ruff format lelamp/

# API 服务器
uv run uvicorn lelamp.api.app:app --host 0.0.0.0 --port 8000 --reload

# 硬件测试（需 Raspberry Pi）
sudo uv run -m tests.hardware.test_rgb
uv run -m tests.hardware.test_audio
uv run -m tests.hardware.test_motors --id <lamp_id> --port <port>

# 电机工具
uv run -m lelamp.list_recordings --id <lamp_id>
uv run -m lelamp.record --id <lamp_id> --port <port> --name <name>
uv run -m lelamp.replay --id <lamp_id> --port <port> --name <name>

# Web 前端（独立部署，不使用 LiveKit）
cd web && pnpm install && pnpm dev
```

## Architecture

### 核心分层

```
main.py                    # LiveKit Agent 启动入口（console/start 两种模式）
lelamp/
├── agent/                 # Agent 核心
│   ├── lelamp_agent.py    #   LeLamp(Agent) 主类 + @function_tool 定义
│   ├── states.py          #   对话状态机（IDLE/LISTENING/THINKING/SPEAKING）
│   └── tools/             #   按领域拆分的 function_tool（motor/rgb/vision/memory/system）
├── service/               # 优先级事件分发服务（每服务独立 daemon thread）
│   ├── base.py            #   ServiceBase + heapq 优先队列
│   ├── motors/            #   舵机控制 + 健康监控
│   ├── rgb/               #   8x8 LED 矩阵
│   └── vision/            #   摄像头采集 + 隐私保护 + 主动监控
├── edge/                  # 本地 AI 推理（MediaPipe Tasks API）
│   ├── face_detector.py   #   人脸检测 → 存在感知
│   ├── hand_tracker.py    #   21 点手部追踪 + 8 种手势识别
│   ├── object_detector.py #   COCO 80 类物体检测
│   └── hybrid_vision.py   #   智能路由：本地 vs 云端（Qwen VL）
├── memory/                # 跨会话记忆系统（SQLite）
├── integrations/          # 外部 AI 客户端（百度语音、Qwen VL）+ 统一异常处理
├── api/                   # FastAPI REST + WebSocket（独立于 Agent 运行）
├── database/              # SQLAlchemy ORM（SQLite/PostgreSQL）
├── config.py              # 冻结 dataclass 配置，环境变量加载
├── cache/                 # TTL 缓存（视觉/搜索 API）
└── utils/                 # 限流、安全（License/SSRF/OTA）、URL 校验
web/                       # Vue 3 灯控面板（自定义 WebSocket，非 LiveKit 客户端）
```

### 两个独立入口

1. **`main.py`** — LiveKit 语音 Agent（`console` 文字模式 / `start` Worker 模式）
2. **`lelamp/api/app.py`** — FastAPI 服务器（REST API + WebSocket，独立启动）

两者共享数据库和配置，但运行时互不依赖。

### 关键架构模式

**并发模型** — Agent 运行在 LiveKit asyncio 事件循环中，Service 运行在独立 daemon 线程中：
- 跨线程共享状态 → `threading.Lock`（不是 `asyncio.Lock`）
- 异步子进程 → `asyncio.create_subprocess_exec()`（不是 `subprocess.run`）
- 时间戳临界区（`_light_override_until_ts` 等）→ 必须用 `_timestamps_lock` 保护

**Service 事件系统** — `ServiceBase` 基于 heapq 的优先级队列：
- 优先级：CRITICAL(0) > HIGH(1) > NORMAL(2) > LOW(3)
- 队列满时高优先级事件可替换低优先级事件
- 各 Service 的 `dispatch_event()` 是线程安全的

**边缘视觉降级** — MediaPipe 不可用时自动进入 NoOp 模式（`_noop = True`），不崩溃。
检查点：`hasattr(mp, 'tasks')` at import time。

**记忆系统初始化顺序** — 必须在 `init_db()` 之前 `import lelamp.memory` 注册模型到 SQLAlchemy Base。
`main.py` 的 `entrypoint` 中已有此顺序。

**LiveKit SDK 回调签名** — `data_received` 事件回调接收 `DataPacket` 对象（含 `.data` 和 `.participant`），不是独立的 `(data, participant)` 参数。

**配置管理** — 所有配置通过 `lelamp/config.py` 的冻结 dataclass 加载，运行时不可变。

## Critical Patterns

### 舵机安全
所有 `move_joint` 调用必须验证 `SAFE_JOINT_RANGES`：
```python
SAFE_JOINT_RANGES = {
    "base_yaw": (-180, 180), "base_pitch": (-90, 90),
    "elbow_pitch": (-150, 150), "wrist_roll": (-180, 180), "wrist_pitch": (-90, 90),
}
```

### 外部 API 调用
- 异常处理：`lelamp/integrations/exceptions.py` 统一异常层次 + `@retry_on_error` 指数退避
- 限流：`lelamp/utils/rate_limiter.py` 令牌桶算法
- URL 校验：`lelamp/utils/url_validation.py` SSRF 防护
- 缓存：`lelamp/cache/cache_manager.py` TTL + LRU

### 记忆系统（`lelamp/memory/`）
- `MemoryStore`：`threading.Lock` 保护的 CRUD，LIKE 搜索，token 预算控制（~400 tokens，~0.67 chars/token 中文）
- `MemoryConsolidator`：对话 10 轮后触发 LLM 提取记忆，5 分钟冷却，`asyncio.create_task()` 非阻塞
- 6 层降级：init 失败 → DB 错误 → LLM 失败 → JSON 解析失败 → tool 错误 → prompt 构建失败，均不影响核心对话
- 禁用调试：`LELAMP_MEMORY_ENABLED=0`

### 数据库初始化
- Agent 入口（`main.py entrypoint`）：调用 `init_db()` 创建所有表
- API 入口（`lelamp/api/app.py lifespan`）：同样调用 `init_db()`
- 新增 ORM 模型时必须在对应入口 `init_db()` 之前 import 以注册到 Base

## Environment Variables

核心必需变量（`.env` 文件）：
- `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` — 语音基础设施
- `DEEPSEEK_API_KEY` — LLM 对话
- `BAIDU_SPEECH_API_KEY`, `BAIDU_SPEECH_SECRET_KEY` — 语音识别/合成

常用可选变量：
- `LELAMP_DEV_MODE=1` — 跳过 License 验证（开发用）
- `LELAMP_VISION_ENABLED` (default: true) — 云端视觉
- `LELAMP_EDGE_VISION_ENABLED` (default: false) — 本地边缘视觉
- `LELAMP_PROACTIVE_MONITOR` (default: true) — 主动视觉监控
- `LELAMP_MEMORY_ENABLED` (default: 1) — 记忆系统
- `MODELSCOPE_API_KEY` — Qwen VL 视觉模型
- `LOG_LEVEL` (default: "INFO") — 日志级别

完整变量列表见 `lelamp/config.py` 和各模块文档。

## Hardware Notes

- **RGB LED**：`rpi-ws281x`（GPIO 12，8x8 矩阵），macOS 不可用
- **舵机**：Feetech servo SDK，串口通信 `/dev/ttyACM0`，macOS 无硬件时 NoOp
- **摄像头**：支持旋转/翻转（`LELAMP_CAMERA_ROTATE_DEG`/`LELAMP_CAMERA_FLIP`），隐私保护 LED 指示
- **音量**：`amixer` 控制 Line/Line DAC/HP 输出

## Security

- 永远不提交 `.env` 文件，使用 `.env.example` 模板
- `LELAMP_LICENSE_SECRET` 生产环境必须使用强随机密钥
- OTA 更新强制 HTTPS + SHA256 校验
- 外部 API URL 经过域名白名单 + 私有 IP 阻断
