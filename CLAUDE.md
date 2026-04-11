# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
使用中文交流。

## Project Overview

LeLamp Runtime 是 LeLamp 机器人台灯的 Python 控制系统。基于 LiveKit 实时语音对话，集成 DeepSeek LLM、百度语音、Qwen VL 视觉、舵机控制、RGB LED 灯效。运行在 Raspberry Pi 上，macOS 可开发调试（无硬件功能降级为 NoOp）。Python 版本约束：`>=3.12,<3.14`。

## Commands

```bash
# 安装依赖
uv sync                              # 基础依赖
uv sync --extra dev                  # + 测试/lint 工具
uv sync --extra api                  # + FastAPI/数据库
uv sync --extra vision               # + MediaPipe + OpenCV（仅 arm64 macOS / x86_64 Linux / AMD64 Windows）
uv sync --extra dev --extra api --extra vision  # 完整开发依赖
uv sync --extra hardware             # + rpi-ws281x 等硬件依赖（仅 Raspberry Pi）
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

# 数据库迁移（Alembic）
uv run alembic revision --autogenerate -m "描述"  # 生成迁移（模型变更后执行）
uv run alembic upgrade head                       # 应用所有迁移
uv run alembic downgrade -1                       # 回退一步
uv run alembic history                            # 查看迁移历史
uv run alembic current                            # 查看当前版本

# Lint & Format
uv run ruff check lelamp/
uv run ruff check --fix lelamp/
uv run ruff format lelamp/

# API 服务器（同时托管 Vue 前端，单端口访问）
uv run uvicorn lelamp.api.app:app --host 0.0.0.0 --port 8000 --reload

# 构建前端（部署到 Pi 前执行）
bash scripts/build_web.sh

# 硬件测试（需 Raspberry Pi）
sudo uv run -m tests.hardware.test_rgb
uv run -m tests.hardware.test_audio
uv run -m tests.hardware.test_motors --id <lamp_id> --port <port>

# 出厂预配置（需 Raspberry Pi + sudo）
sudo bash scripts/factory/prepare_factory_env.sh

# 电机工具
uv run -m lelamp.list_recordings --id <lamp_id>
uv run -m lelamp.record --id <lamp_id> --port <port> --name <name>
uv run -m lelamp.replay --id <lamp_id> --port <port> --name <name>

# Web 前端（开发模式）
cd web && pnpm install && pnpm dev

# 访问设备（mDNS 自动发现）
# macOS: open http://lelamp.local:8000
# 其他: http://<Pi-IP>:8000
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
├── api/                   # FastAPI REST + WebSocket + Vue 静态托管
│   ├── app.py             #   FastAPI 主应用（含 SPA 静态文件托管）
│   ├── routes/            #   REST 端点（auth/livekit/system/wifi/websocket，多数需认证）
│   ├── services/          #   业务服务（auth/wifi/ap/onboarding/mdns）
│   └── middleware/         #   JWT 认证中间件（get_current_user/get_current_admin/get_current_user_optional）
├── database/              # SQLAlchemy ORM（SQLite/PostgreSQL）
├── config.py              # 冻结 dataclass 配置，环境变量加载（load_config + load_config_strict）
├── cache/                 # TTL 缓存（视觉/搜索 API）
└── utils/                 # 限流、安全（License/SSRF/OTA）、URL 校验
web/                       # Vue 3 灯控面板（构建后由 FastAPI 托管，单端口访问）
```

### 两个独立入口

1. **`main.py`** — LiveKit 语音 Agent（`console` 文字模式 / `start` Worker 模式）
2. **`lelamp/api/app.py`** — FastAPI 服务器（REST API + WebSocket，独立启动）

两者共享数据库和配置，但运行时互不依赖。配置加载策略不同：`main.py` 使用 `load_config_strict()` 强制校验 LiveKit/DeepSeek/百度密钥（缺少时报错退出），API 入口使用 `load_config()` 更宽容（未设置时使用默认值）。两者均定义在 `lelamp/config.py` 中。

**单端口部署** — API 服务器通过 `StaticFiles` 托管 Vue 构建产物（`web/dist/`），用户通过 `http://<device>:8000` 同时访问 API 和前端。构建产物不存在时自动跳过静态文件托管（开发模式不受影响）。环境变量 `LELAMP_WEB_DIST` 控制构建产物路径。

**mDNS 设备发现** — API 启动时通过 zeroconf 注册 `_http._tcp` 服务，用户可通过 `http://<device_id>.local:8000` 访问设备，无需记忆 IP。mDNS 不可用时静默降级，不影响核心功能。

### 关键架构模式

**并发模型** — Agent 运行在 LiveKit asyncio 事件循环中，Service 运行在独立 daemon 线程中：
- 跨线程共享状态 → `threading.Lock`（不是 `asyncio.Lock`）
- 异步子进程 → `asyncio.create_subprocess_exec()`（不是 `subprocess.run`）
- 时间戳临界区（`_light_override_until_ts` 等）→ 必须用 `_timestamps_lock` 保护
- **后台任务追踪**：所有 `asyncio.create_task()` 必须追踪，确保 `shutdown()` 时能正确取消。Agent 内使用 `self._track_task(coro)`；API 路由模块使用模块级 `_background_tasks` 集合 + `_track_background_task()` 辅助函数；工具类（如 `SystemTools`）使用 `self._tasks` 集合

**Service 事件系统** — `ServiceBase` 基于 heapq 的优先级队列：
- 优先级：CRITICAL(0) > HIGH(1) > NORMAL(2) > LOW(3)
- 队列满时高优先级事件可替换低优先级事件
- 各 Service 的 `dispatch_event()` 是线程安全的

**边缘视觉降级** — MediaPipe 不可用时自动进入 NoOp 模式（`_noop = True`），不崩溃。
检查点：`hasattr(mp, 'tasks')` at import time。

**记忆系统初始化顺序** — 必须在 `init_db()` 之前 `import lelamp.memory` 注册模型到 SQLAlchemy Base。
`main.py` 的 `entrypoint` 中已有此顺序。

**LiveKit SDK 回调签名** — `data_received` 事件回调接收 `DataPacket` 对象（含 `.data` 和 `.participant`），不是独立的 `(data, participant)` 参数。

**LiveKit 1.5+ turn_handling API** — `AgentSession` 使用 `turn_handling` 参数配置打断和结束点检测：
```python
session = AgentSession(
    turn_handling={
        "interruption": {"mode": "adaptive"},  # ML 区分真正打断 vs 假阳性
        "endpointing": {"mode": "dynamic"},    # 自适应沉默阈值
    },
)
```
单次 `session.say()` 使用 `allow_interruptions=False/True` 控制打断行为。

**配置管理** — 所有配置通过 `lelamp/config.py` 的冻结 dataclass 加载，运行时不可变。提供两个入口：`load_config()`（宽容模式，API 使用）和 `load_config_strict()`（严格模式，Agent 使用，强制校验关键密钥）。`main.py` 不再重复定义配置加载逻辑。

**API 硬件降级** — API 入口（`app.py lifespan`）对硬件服务做降级：MotorsService 失败 → `NoOpMotorsService`，RGBService 失败 → `NoOpRGBService`。VisionService 在 API 模式下不启动（避免与 LiveKit Agent 争抢摄像头）。

**数据库模型注册** — ORM 模型分布在 3 个文件，必须在 `init_db()` 之前全部 import 以注册到 `Base`：
- `lelamp/database/models.py` — Conversation, OperationLog, DeviceState, UserSettings（内部 import models_auth）
- `lelamp/database/models_auth.py` — User, DeviceBinding, RefreshToken
- `lelamp/memory/models.py` — Memory, ConversationSummary

**Follower/Leader 模式** — `lelamp/follower/` 和 `lelamp/leader/` 集成 HuggingFace LeRobot 框架：
- `LeLampFollower`：5 轴舵机机器人控制类（FeetechMotorsBus + STS3215）
- `LeLampLeader`：遥操作端，读取关节位置
- 两者都注册到 LeRobot 配置注册表（`"lelamp_follower"` / `"lelamp_leader"`）

## Critical Patterns

### Agent Tool 定义（LiveKit 1.5+ 最佳实践）

所有 `@function_tool` 方法必须包含 `context: RunContext` 参数：
```python
from livekit.agents import function_tool, RunContext

@function_tool()
async def play_recording(
    self,
    context: RunContext,  # 必须
    recording_name: str,
) -> str:
    ...
```
Data Channel 命令执行（WebSocket → `_execute_command`）使用轻量级 `_DataContext` 作为 context 占位（不导入 `unittest.mock`）。

长时间运行的工具使用 `_tool_with_timeout()` 包装：
```python
async def vision_answer(self, context: RunContext, question: str) -> str:
    return await self._tool_with_timeout(
        self._do_vision_answer(question),
        timeout_seconds=30.0,
        error_message="视觉识别超时，请稍后重试",
    )
```
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
- URL 校验：`lelamp/utils/url_validation.py` SSRF 防护（HTTPS 强制 + 域名白名单 + 私有/保留/组播 IP 阻断）
- 缓存：`lelamp/cache/cache_manager.py` TTL + LRU

### 时间戳规范
- 所有代码使用 `datetime.now(UTC)`，禁止 `datetime.utcnow()`（Python 3.12+ 已废弃）
- SQLAlchemy `default=` / `onupdate=` 使用 `lambda: datetime.now(UTC)`（可调用对象）
- Pydantic `default_factory=` 同上
- 确保 `from datetime import datetime, UTC`

### 记忆系统（`lelamp/memory/`）
- `MemoryStore`：`threading.Lock` 保护的 CRUD，LIKE 搜索，token 预算控制（~400 tokens，~0.67 chars/token 中文）
- `MemoryConsolidator`：对话 10 轮后触发 LLM 提取记忆，5 分钟冷却，`asyncio.create_task()` 非阻塞
- 6 层降级：init 失败 → DB 错误 → LLM 失败 → JSON 解析失败 → tool 错误 → prompt 构建失败，均不影响核心对话
- 禁用调试：`LELAMP_MEMORY_ENABLED=0`

### 数据库初始化
- `init_db()` 优先使用 Alembic 迁移（`alembic upgrade head`），Alembic 不可用时 fallback 到 `create_all()`
- Agent 入口（`main.py entrypoint`）和 API 入口（`lelamp/api/app.py lifespan`）均调用 `init_db()`
- 新增 ORM 模型时必须在对应入口 `init_db()` 之前 import 以注册到 Base
- 同时需要在 `alembic/env.py` 中导入新模型（用于 autogenerate 检测）
- Schema 变更流程：修改模型 → `alembic revision --autogenerate -m "描述"` → 检查生成结果 → `alembic upgrade head`
- SQLite 使用 WAL 模式，`LELAMP_DATABASE_URL` 支持切换到 PostgreSQL
- 数据库连接：`get_db()` 用于 FastAPI 依赖注入，`get_db_context()` 用于手动管理

### 设备绑定与认证（`lelamp/api/services/`）
- `device_secret`：首次 WiFi 配置时自动生成（`secrets.token_hex(8)`），持久化到 `setup_status.json`
- `bind_device()`：验证密钥使用 `hmac.compare_digest()`（恒定时间比较），密钥不存入数据库
- `auto_bind_device()`：自动绑定（Setup Wizard 使用），从 `setup_status.json` 读取密钥，已绑定时幂等返回
- 密钥读取优先级：`setup_status.json` > `LELAMP_DEVICE_SECRET` 环境变量 > 不验证（开发模式）
- `GET /api/system/device`：公开端点，返回设备基本信息（不返回 `device_secret`），已认证时返回绑定状态
- `POST /api/auth/auto-bind`：自动绑定端点（需 JWT 认证），Setup Wizard 注册/登录后调用
- JWT `SECRET_KEY`：从 `LELAMP_JWT_SECRET` 环境变量读取，未设置时随机生成并警告

### LiveKit Token 管理（`lelamp/api/routes/livekit.py`）
- `POST /api/livekit/token`：需 JWT 认证，强制使用认证用户身份（不可伪造）
- 房间名通过 Pydantic Field 校验（`[a-zA-Z0-9_-]+`，1-128 字符）
- 替代了旧的命令行脚本 `scripts/tools/generate_client_token.py`

## Environment Variables

核心必需变量（`.env` 文件）：
- `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` — 语音基础设施
- `LLM_API_KEY` — LLM 对话（fallback: `DEEPSEEK_API_KEY`）
- `LLM_MODEL` — LLM 模型名（默认: `MiniMax-M2.7`，fallback: `DEEPSEEK_MODEL`）
- `LLM_BASE_URL` — LLM API 地址（默认: `https://api.minimaxi.com/v1`，fallback: `DEEPSEEK_BASE_URL`）
- `BAIDU_SPEECH_API_KEY`, `BAIDU_SPEECH_SECRET_KEY` — 语音识别/合成

常用可选变量：
- `LELAMP_DEV_MODE=1` — 跳过 License 验证（开发用）
- `LELAMP_LICENSE_KEY` — 设备授权码（生产必需）
- `LELAMP_DATABASE_URL` (default: sqlite:///./lelamp.db) — 数据库连接字符串
- `LELAMP_VISION_ENABLED` (default: true) — 云端视觉
- `LELAMP_EDGE_VISION_ENABLED` (default: false) — 本地边缘视觉
- `LELAMP_PROACTIVE_MONITOR` (default: true) — 主动视觉监控
- `LELAMP_MEMORY_ENABLED` (default: 1) — 记忆系统
- `LELAMP_JWT_SECRET` — JWT 签名密钥（未设置时随机生成，重启失效）
- `LELAMP_DEVICE_SECRET` — 设备绑定密钥（优先从 setup_status.json 读取）
- `LELAMP_WEB_DIST` (default: web/dist) — Vue 前端构建产物路径
- `LELAMP_CORS_ORIGINS` — 自定义 CORS 允许的源列表（逗号分隔，开发环境内网 IP 通过此变量配置，不要硬编码）
- `MODELSCOPE_API_KEY` — Qwen VL 视觉模型
- `FEISHU_APP_ID`, `FEISHU_APP_SECRET`, `FEISHU_RECEIVE_ID` — 飞书通知集成
- `TAVILY_API_KEY` — Tavily 搜索
- `LELAMP_VOLUME_LEVEL` (default: "100") — 系统音量 (0-100)，通过 Web 界面调节后自动保存到 .env
- `LOG_LEVEL` (default: "INFO") — 日志级别
- `LELAMP_LOG_TO_FILE` (default: false) — 启用文件日志
- `LELAMP_VAD_MIN_SPEECH_DURATION` — VAD 最小语音时长（秒）
- `LELAMP_VAD_MIN_SILENCE_DURATION` — VAD 最小沉默时长（秒）
- `LELAMP_VAD_PREFIX_PADDING_DURATION` — VAD 前缀填充时长（秒）
- `LELAMP_VAD_ACTIVATION_THRESHOLD` — VAD 激活阈值

完整变量列表见 `lelamp/config.py` 和各模块文档。

## Hardware Notes

- **RGB LED**：`rpi-ws281x`（GPIO 12，8x8 矩阵），macOS 不可用
- **舵机**：Feetech servo SDK，串口通信 `/dev/ttyACM0`，macOS 无硬件时 NoOp
- **摄像头**：支持旋转/翻转（`LELAMP_CAMERA_ROTATE_DEG`/`LELAMP_CAMERA_FLIP`），隐私保护 LED 指示
- **音量**：`amixer` 控制 Line/Line DAC/HP 输出

## Testing

测试使用 pytest，共享 fixtures 在 `tests/conftest.py` 中定义：
- `mock_config()` — 返回完整的 `AppConfig` 测试实例
- `mock_motors_service()` / `mock_rgb_service()` / `mock_vision_service()` — Mock 硬件服务

```bash
# Agent 行为测试（含 tool delegation、timeout、background task）
uv run pytest tests/test_agent_behavior.py -v
# 记忆系统测试
uv run pytest tests/test_memory.py -v
# 按名称匹配
uv run pytest tests/ -k "test_setup" -v
# 带覆盖率
uv run pytest tests/ --cov=lelamp --cov-report=html
```

## API & WebSocket

### API 路由（前缀 `/api`）
`/auth`（认证）、`/devices`（设备管理，需认证）、`/history`（历史记录，需认证）、`/system`（系统信息）、`/settings`（设置，需认证）、`/livekit`（Token）、`/ws/{lamp_id}`（设备状态 WebSocket）、`/ws/setup`（配网进度 WebSocket）

### API 认证级别
- **`Depends(get_current_user)`**：settings（全部端点）、devices（command/conversations/operations/statistics/list）、history（全部端点）
- **`Depends(get_current_admin)`**：system（restart/restart/cancel）
- **`Depends(get_current_user_optional)`**：devices（state/health，匿名可查看基本状态）
- **无认证**：system（setup/*、wifi/*、device、info、health）、auth（register/login）

### 中间件栈（按添加顺序）
SecurityHeadersMiddleware → CaptivePortalMiddleware → GZipMiddleware → CORSMiddleware

### WebSocket 端点
1. **设备状态** (`/api/ws/{lamp_id}`)：支持 JWT 认证（通过 query 参数 `token`），匿名连接仅允许只读操作（ping/pong、subscribe）。硬件控制命令（chat/RGB/电机/摄像头）需要有效 JWT token，匿名连接会收到 `AUTH_REQUIRED` 错误。频道订阅系统支持 state/events/logs/notifications/conversations/health。
2. **配网进度** (`/api/ws/setup`)：使用 `SetupEventBus` asyncio 广播推送 WiFi 配网事件（wifi_connecting/connected/failed 等）

### 配置同步
`ConfigSyncService` 将数据库中的 UserSettings 同步回 `.env` 文件。API 的 settings 端点修改设置后会写回 `.env`，重启后生效。

## Security

- 永远不提交 `.env` 文件，使用 `.env.example` 模板
- `LELAMP_LICENSE_SECRET` 生产环境必须使用强随机密钥
- `LELAMP_JWT_SECRET` 生产环境必须设置为固定强随机密钥
- OTA 更新强制 HTTPS + SHA256 校验
- 外部 API URL 经过域名白名单 + 私有/保留/组播 IP 阻断（`is_private`/`is_loopback`/`is_link_local`/`is_reserved`/`is_multicast`/`is_unspecified`）
- 所有密钥比较（License 验证、设备绑定）必须使用 `hmac.compare_digest()`（防时序攻击），禁止 `==`/`!=`
- API 异常处理器不向客户端返回原始异常详情（`str(exc)`），仅返回通用错误消息，详细信息记录到服务端日志
- AP 热点密码每次启动时随机生成（`secrets.token_urlsafe(6)`），不再使用固定密码
- LiveKit Token 端点强制使用认证用户身份，禁止用户指定 `identity`

## Systemd Services

生产环境使用 4 个 systemd 服务，启动链：`lelamp-setup` → `lelamp-setup-ap` → `lelamp-captive-portal` → `lelamp-livekit` + `lelamp-api`。工作目录 `/home/pi/lelamp_runtime`。

```bash
# 查看状态
sudo systemctl status lelamp-{livekit,api}.service
# 重启
sudo systemctl restart lelamp-{livekit,api}.service
# 查看日志
sudo journalctl -u lelamp-{livekit,api}.service -f
```
