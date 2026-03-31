# 系统架构文档

## 🏗️ 概述

LeLamp Runtime 采用模块化、分层架构设计,将对话式 AI、硬件控制、API 服务和数据持久化无缝集成。

---

## 📐 整体架构

### 架构分层

```
┌─────────────────────────────────────────────────────────────────┐
│                         客户端层                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Web 客户端   │  │  LiveKit App │  │  第三方集成   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    LiveKit Agent 层                             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Voice Pipeline                                          │    │
│  │  - Silero VAD    - BVC Noise Cancellation               │    │
│  │  - Baidu STT     - DeepSeek LLM     - Baidu TTS         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              ↓                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  LeLamp Agent (livekit.agents.Agent)                     │    │
│  │  - 状态管理 (idle/listening/thinking/speaking)            │    │
│  │  - Function Tools (motor/RGB/vision/system/memory)        │    │
│  │  - 记忆系统 (Memory/ConversationSummary)                 │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI 服务层                              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  中间件层                                                │    │
│  │  - CORS        - Rate Limit      - Security Headers     │    │
│  │  - Cache       - Compression      - Logging             │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              ↓                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  路由层                                                  │    │
│  │  - /api/auth/*  - /api/devices/*  - /api/settings/*     │    │
│  │  - /api/system  - /api/ws/*                             │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              ↓                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  服务层 (Business Logic)                                 │    │
│  │  - AuthService  - DeviceService  - SettingsService       │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                         硬件服务层                               │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐   │
│  │  MotorsService   │  │   RGBService     │  │ VisionService│   │
│  │  (5轴舵机控制)    │  │  (8x8 LED矩阵)   │  │ (摄像头+隐私) │   │
│  └──────────────────┘  └──────────────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                         数据访问层                               │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Database│  │Cache Manager │  │ MemoryStore │          │
│  │  - User │  │- API Response│  │- 长期记忆  │          │
│  │  - Token│  │- Vision API  │  │- 对话摘要  │          │
│  └──────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       数据持久化层                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   SQLite     │  │ PostgreSQL   │  │  File System │          │
│  │  (开发环境)    │  │  (生产环境)    │  │  (录音/日志)  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔌 LiveKit 集成

### LiveKit 在项目中的角色

LiveKit **不是**用于 STT/TTS（项目使用 Baidu Speech），而是提供以下核心功能：

| 功能 | LiveKit 的作用 |
|------|---------------|
| 实时语音通信 | WebRTC 语音管道 |
| Agent 框架 | `livekit.agents.Agent` 基类 |
| VAD | Silero 语音活动检测 |
| 噪音消除 | BVC (Background Voice Cancellation) |
| Function Tools | 让 LLM 能调用硬件控制函数 |

### 入口点流程 (`main.py`)

```
CLI 启动 → WorkerOptions → JobContext.connect() → AgentSession.start() → LeLamp Agent
```

**关键代码流程：**
```python
# 1. LiveKit CLI 启动
cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))

# 2. 连接 LiveKit 房间
await ctx.connect()

# 3. 创建 AgentSession
session = AgentSession(
    vad=_build_vad(),              # Silero VAD
    stt=BaiduShortSpeechSTT(...),  # 百度 STT
    llm=deepseek_llm,              # DeepSeek LLM
    tts=BaiduTTS(...),             # 百度 TTS
)

# 4. 启动会话
await session.start(agent=agent, room=ctx.room)
```

### VAD 配置

使用 `livekit.plugins.silero.VAD` 进行语音活动检测：

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `LELAMP_VAD_MIN_SPEECH_DURATION` | - | 最小语音时长 |
| `LELAMP_VAD_MIN_SILENCE_DURATION` | - | 最小静音时长 |
| `LELAMP_VAD_PREFIX_PADDING_DURATION` | - | 前缀填充时长 |
| `LELAMP_VAD_ACTIVATION_THRESHOLD` | - | 激活阈值 |

### 噪音消除 (BVC)

```python
start_kwargs["room_input_options"] = RoomInputOptions(
    noise_cancellation=noise_cancellation.BVC(),
)
```

通过 `LELAMP_NOISE_CANCELLATION` 环境变量控制（默认启用）。

---

## 🧩 核心组件

### 1. Agent 架构 (`lelamp/agent/`)

**设计理念**: 模块化、可扩展的对话式 Agent

```
lelamp/agent/
├── lelamp_agent.py       # 主 Agent 类
├── states.py             # 状态管理
└── tools/                # 功能工具
    ├── motor_tools.py    # 电机控制
    ├── rgb_tools.py      # RGB 灯光
    ├── vision_tools.py   # 视觉识别
    ├── memory_tools.py   # 记忆工具
    └── system_tools.py   # 系统功能
```

#### LeLamp Agent 类

继承自 LiveKit 的 `Agent` 类,提供核心对话能力:

```python
class LeLamp(Agent):
    """
    LeLamp 对话式 Agent

    功能:
    - 对话状态管理 (idle, listening, thinking, speaking)
    - 功能工具暴露 (motor, RGB, vision, system)
    - 运动和灯光的冷却与覆盖系统
    - 安全范围验证
    """
```

**关键特性**:
- **对话状态**: 使用 `StateManager` 管理状态转换
- **状态颜色**: 通过 `StateColors` 映射 LED 颜色
- **冷却系统**: 防止过度运动和灯光闪烁
- **安全验证**: 所有关节运动在 `SAFE_JOINT_RANGES` 内

#### 状态管理 (`states.py`)

```python
class ConversationState(Enum):
    IDLE = "idle"           # 空闲 - 暖白光
    LISTENING = "listening" # 听取 - 蓝光
    THINKING = "thinking"   # 思考 - 紫光
    SPEAKING = "speaking"   # 说话 - 随机动画色

class StateManager:
    """
    线程安全的状态管理器

    功能:
    - 状态转换验证
    - 回调通知
    - 历史记录
    """
```

#### 功能工具 (`tools/`)

每个工具模块使用 `@function_tool` 装饰器暴露给 LLM:

**motor_tools.py**:
```python
@function_tool()
def play_recording(name: str) -> str:
    """播放录制的电机动画"""

@function_tool()
def move_joint(joint: str, position: float) -> str:
    """移动指定关节到位置"""

@function_tool()
def get_joint_positions() -> dict:
    """获取所有关节的当前位置"""

@function_tool()
def get_motor_health(motor_name: str = None) -> dict:
    """获取电机健康状态（商用功能）"""
```

**安全范围限制：**
```python
SAFE_JOINT_RANGES = {
    "base_yaw": (-180, 180),
    "base_pitch": (-90, 90),
    "elbow_pitch": (-150, 150),
    "wrist_roll": (-180, 180),
    "wrist_pitch": (-90, 90),
}
```

**rgb_tools.py**:
```python
@function_tool()
def set_rgb_solid(r: int, g: int, b: int) -> str:
    """设置纯色"""

@function_tool()
def paint_rgb_pattern(pattern: str) -> str:
    """绘制预定义图案 (heart, smile, arrow)"""

@function_tool()
def rgb_effect_rainbow(speed: float = 1.0) -> str:
    """彩虹效果"""

@function_tool()
def rgb_effect_breathing(r: int, g: int, b: int) -> str:
    """呼吸效果"""

@function_tool()
def rgb_effect_random_animation() -> str:
    """随机颜色动画（speaking 状态）"""
```

**vision_tools.py**:
```python
@function_tool()
def vision_answer(question: str) -> str:
    """使用视觉 AI 回答问题"""

@function_tool()
def check_homework() -> str:
    """检查作业（数学、口算等）"""

@function_tool()
def capture_to_feishu() -> str:
    """拍照并推送到飞书"""
```

**system_tools.py**:
```python
@function_tool()
def get_available_recordings() -> list:
    """获取可用的录制动作列表"""

@function_tool()
def tune_motor_pid(motor: str, p: float, i: float, d: float) -> str:
    """远程调整舵机 PID 参数（商用）"""

@function_tool()
def reset_motor_health_stats(motor_name: str = None) -> str:
    """重置健康统计数据"""

@function_tool()
def set_rgb_brightness(percent: int) -> str:
    """调节灯光亮度 (0-100)"""

@function_tool()
def rgb_effect_wave(...) -> str:
    """波纹/呼吸波动效果"""

@function_tool()
def rgb_effect_fire(...) -> str:
    """火焰动态效果"""

@function_tool()
def rgb_effect_emoji(emoji: str) -> str:
    """表情动画 (smile/sad/wink/angry/heart)"""

@function_tool()
def stop_rgb_effect() -> str:
    """停止所有动态灯效"""

@function_tool()
def set_volume(volume_percent: int) -> str:
    """控制系统音量"""

@function_tool()
def get_rate_limit_stats() -> dict:
    """获取 API 速率限制统计"""

@function_tool()
def web_search(query: str) -> str:
    """联网搜索（实时信息）"""

@function_tool()
def check_for_updates() -> str:
    """检查 OTA 更新"""

@function_tool()
def perform_ota_update() -> str:
    """执行 OTA 更新"""
```

---

### 2. API 服务架构 (`lelamp/api/`)

**设计理念**: 分层、解耦、可测试

```
lelamp/api/
├── app.py                 # FastAPI 应用
├── routes/                # 路由处理器
│   ├── auth.py           # 认证端点
│   ├── devices.py        # 设备管理
│   ├── settings.py       # 设置管理
│   ├── system.py         # 系统端点
│   └── websocket.py      # WebSocket 端点
├── models/                # Pydantic 模型
│   └── auth_models.py    # 认证模型
├── services/              # 业务逻辑
│   └── auth_service.py   # 认证服务
└── middleware/            # 中间件
    ├── auth.py           # 认证依赖
    ├── rate_limit.py     # 速率限制
    └── cache.py          # API 缓存
```

#### 中间件层

**认证中间件** (`middleware/auth.py`):
```python
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """
    验证 JWT 令牌并返回当前用户

    - 解析令牌
    - 查询数据库
    - 返回用户对象
    """
```

**速率限制中间件** (`middleware/rate_limit.py`):
```python
class RateLimiter:
    """
    滑动窗口速率限制器

    算法:
    1. 维护请求时间戳队列
    2. 删除过期请求
    3. 计算当前窗口请求数
    4. 超限时拒绝请求
    """
```

**缓存中间件** (`middleware/cache.py`):
```python
def cache_response(ttl_seconds: int = 60):
    """
    API 响应缓存装饰器

    功能:
    - 仅缓存 GET 请求
    - 基于 URL 生成缓存键
    - 可配置 TTL
    - 自动过期
    """
```

#### 服务层

**认证服务** (`services/auth_service.py`):
```python
class AuthService:
    """
    认证业务逻辑

    方法:
    - register_user(): 用户注册
    - authenticate_user(): 用户认证
    - create_access_token(): 创建访问令牌
    - create_refresh_token(): 创建刷新令牌
    - verify_token(): 验证令牌
    """
```

---

### 3. 数据库架构 (`lelamp/database/`)

**设计理念**: ORM、类型安全、关系完整

#### 数据模型

**认证模型** (`models_auth.py`):
```python
class User(Base):
    """
    用户表

    索引:
    - ix_users_username (unique)
    - ix_users_email (unique)
    """
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True)
    email: Mapped[str] = mapped_column(String(100), unique=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime)

class RefreshToken(Base):
    """
    刷新令牌表

    索引:
    - ix_refresh_tokens_token (unique)
    - ix_refresh_tokens_user_id
    - ix_refresh_tokens_expires_at
    """
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    token: Mapped[str] = mapped_column(String(500), unique=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"))
    expires_at: Mapped[datetime] = mapped_column(DateTime)

class DeviceBinding(Base):
    """
    设备绑定表

    关系:
    - User (many-to-one)
    """
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"))
    device_id: Mapped[str] = mapped_column(String(100))
    permission_level: Mapped[str] = mapped_column(String(50))
    bound_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
```

**性能优化索引** (`models.py`):
```python
class OperationLog(Base):
    """
    操作日志表

    性能索引:
    - lamp_id + timestamp (复合索引)
    - operation_type
    - success
    - lamp_id + success + timestamp (复合索引)
    """
    __table_args__ = (
        Index("ix_operation_logs_lamp_id_timestamp", "lamp_id", "timestamp"),
        Index("ix_operation_logs_operation_type", "operation_type"),
        Index("ix_operation_logs_success", "success"),
        Index("ix_operation_logs_lamp_id_success_timestamp", "lamp_id", "success", "timestamp"),
    )

class DeviceState(Base):
    """
    设备状态表

    性能索引:
    - lamp_id + conversation_state (复合索引)
    - updated_at
    """
    __table_args__ = (
        Index("ix_device_states_lamp_id_conversation_state", "lamp_id", "conversation_state"),
        Index("ix_device_states_updated_at", "updated_at"),
    )
```

---

### 4. 服务架构 (`lelamp/service/`)

**设计理念**: 事件驱动、优先级队列、线程安全

#### 服务基类

```python
class ServiceBase(threading.Thread):
    """
    服务基类

    功能:
    - 优先级事件队列
    - 后台线程处理
    - 优雅关闭
    - 统计信息
    """
```

#### 优先级系统

```python
class EventPriority(IntEnum):
    CRITICAL = 0  # 关键事件 (立即处理)
    HIGH = 1      # 高优先级 (优先处理)
    NORMAL = 2    # 普通事件 (默认)
    LOW = 3       # 低优先级 (后台处理)
```

#### 事件处理流程

```
事件调度
   ↓
优先级队列 (heapq)
   ↓
工作线程
   ↓
事件处理器
   ↓
回调通知
```

---

### 5. 边缘推理层 (`lelamp/edge/`) 🆕

**设计理念**: 本地优先、低延迟、优雅降级

```
lelamp/edge/
├── __init__.py            # 模块入口
├── face_detector.py       # 人脸检测 → 用户在场
├── hand_tracker.py        # 手势追踪 → 手势控制
├── object_detector.py     # 物体检测 → 本地识别
└── hybrid_vision.py       # 混合推理路由
```

#### 核心组件

**FaceDetector** - 人脸检测服务
```python
class FaceDetector:
    """
    基于 MediaPipe 的人脸检测
    
    功能:
    - 用户在场检测 → 自动唤醒/休眠
    - 多人检测 → 区分主用户
    - 防抖机制 → 可配置阈值
    """
```

**HandTracker** - 手势追踪服务
```python
class HandTracker:
    """
    基于 MediaPipe 的手势追踪
    
    支持手势:
    - 👍 点赞 → 触发 nod 动作
    - 👎 踩 → 触发 shake 动作
    - ✌️ 耶 → 触发 excited 动作
    - 👋 挥手 → 开关灯
    - ✊ 握拳 → 静音/取消静音
    - 👆 指向 → 台灯看向指定方向
    """
```

**ObjectDetector** - 物体检测服务
```python
class ObjectDetector:
    """
    基于 MediaPipe EfficientDet 的物体检测
    
    特性:
    - 80 类 COCO 物体
    - 中文标签映射
    - 低延迟 (100-300ms)
    """
```

**HybridVisionService** - 混合推理路由
```python
class HybridVisionService:
    """
    智能路由：简单任务本地推理，复杂任务云端推理
    
    路由策略:
    - 简单查询 ("这是什么") → 本地物体检测
    - 复杂查询 ("检查作业") → 云端 Qwen VL
    - 中等查询 → 混合推理
    """
```

#### 混合推理流程

```
用户问："这是什么？"
    ↓
HybridVisionService.analyze_query()
    ↓
QueryComplexity.SIMPLE
    ↓
本地 MediaPipe 物体检测 (< 200ms)
    ↓
能识别？ ──是──→ 直接回答 "这是苹果"
    │
    否
    ↓
云端 Qwen VL (3-8s) → 详细回答
```

#### 配置选项

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `LELAMP_EDGE_VISION_ENABLED` | `0` | 启用边缘视觉 |
| `LELAMP_EDGE_VISION_PREFER_LOCAL` | `1` | 优先本地推理 |
| `LELAMP_EDGE_VISION_LOCAL_THRESHOLD` | `0.7` | 本地置信度阈值 |

#### Agent 工具集成

```python
@function_tool()
async def quick_identify() -> str:
    """快速识别物体（本地推理）"""

@function_tool()
async def detect_gesture() -> str:
    """检测手势"""

@function_tool()
async def check_presence() -> str:
    """检测用户在场"""

@function_tool()
async def get_edge_vision_stats() -> str:
    """获取边缘视觉统计"""
```

---

### 6. 记忆系统 (`lelamp/memory/`) 🆕

**设计理念**: 双层记忆、LLM 驱动、优雅降级。灵感来自 [nanobot](https://github.com/HKUDS/nanobot) 的记忆架构。

```
lelamp/memory/
├── __init__.py            # 模块入口，导出核心类
├── models.py              # 数据模型 (Memory, ConversationSummary)
├── config.py              # 配置 (MemoryConfig, frozen dataclass)
├── store.py               # 记忆存储 (MemoryStore, CRUD + 线程安全)
└── consolidator.py        # LLM 记忆整合 (MemoryConsolidator)
```

#### 数据模型 (`models.py`)

```python
class Memory(Base):
    """
    长期记忆表

    分类: preference(偏好), fact(事实), relationship(关系),
          context(上下文), general(通用)

    索引:
    - (lamp_id, category)
    - (lamp_id, is_active)
    - (lamp_id, importance)
    """
    id, lamp_id, category, content, importance, source,
    access_count, last_accessed, created_at, updated_at, is_active

class ConversationSummary(Base):
    """
    对话摘要表

    索引:
    - (lamp_id, ended_at)
    """
    id, lamp_id, session_id, summary, key_topics(JSON),
    message_count, started_at, ended_at, created_at
```

#### 记忆存储 (`store.py`)

```python
class MemoryStore:
    """
    线程安全的记忆 CRUD 操作 (threading.Lock)

    核心方法:
    - add_memory()            → 新增记忆 (importance 1-10 钳制)
    - get_active_memories()   → 按 importance 降序 + token 预算控制
    - search_memories()       → LIKE 关键词搜索
    - update_memory()         → 更新内容/重要性
    - deactivate_memory()     → 软删除
    - deactivate_by_content_hint() → 按关键词批量软删除
    - save_summary()          → 保存对话摘要
    - get_recent_summaries()  → 获取近期摘要 (按小时过滤)
    - search_summaries()      → 搜索摘要内容

    Token 预算: 中文字符 ~0.67 chars/token
    Session 安全: db.expunge() 返回对象防止 DetachedInstanceError
    """
```

#### LLM 记忆整合 (`consolidator.py`)

```python
class MemoryConsolidator:
    """
    LLM 驱动的记忆提取和整合

    流程:
    1. should_consolidate() → 检查轮数 (>=10) + 冷却时间 (5min)
    2. consolidate() → 调用 DeepSeek API 提取记忆
    3. _deduplicate_memories() → bigram Jaccard 去重 (阈值 80%)
    4. 保存新记忆 + 对话摘要
    """
```

#### Agent 工具集成 (`agent/tools/memory_tools.py`)

```python
class MemoryTools:
    """
    记忆 Agent 工具 (由 LeLamp Agent 通过 @function_tool 暴露)

    方法:
    - save_memory(content, category)    → 保存记忆
    - recall_memory(query)              → 搜索记忆
    - forget_memory(content_hint)       → 删除记忆
    """

# LeLamp Agent 中的 @function_tool 装饰:
@function_tool()
async def save_memory(self, content: str, category: str = "general") -> str:
    """记住一个重要信息（用户偏好、事实、上下文等）"""

@function_tool()
async def recall_memory(self, query: str) -> str:
    """搜索你的记忆，查找相关信息"""

@function_tool()
async def forget_memory(self, content_hint: str) -> str:
    """删除一条记忆（当信息过时或不再相关时使用）"""
```

#### 记忆系统数据流

```
用户对话持续进行...
   ↓
note_user_text() → 追加到 _conversation_turns (硬上限 200 轮)
   ↓
对话轮次达到 10 轮?
   ├─ 否 → 继续对话
   └─ 是 ↓
      冷却时间检查 (5 分钟)
         ├─ 冷却中 → 跳过
         └─ 可执行 ↓
            asyncio.create_task() 后台整合 (不阻塞对话)
               ↓
            DeepSeek LLM 分析对话:
              1. 提取值得记住的信息 → 写入长期记忆
              2. 生成对话摘要 → 写入对话摘要
              3. 提取话题标签 → 用于后续搜索
               ↓
            新记忆去重 (bigram Jaccard 相似度 > 80% 跳过)
               ↓
            保存记忆 + 摘要
               ↓
            update_instructions() → 下次对话自动生效
```

#### 动态 System Prompt

```python
def _build_dynamic_instructions(self) -> str:
    base = self._INSTRUCTIONS
    memories = self._memory_store.get_active_memories(
        self._lamp_id, max_tokens=token_budget
    )
    if memories:
        base += "\n\n# Memory\nYou remember the following about this user:\n"
        for m in memories:
            base += f"- [{m.category}] {m.content}\n"
    return base
```

#### 配置选项

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `LELAMP_MEMORY_ENABLED` | `1` | 启用/禁用记忆系统 |
| `LELAMP_MEMORY_TOKEN_BUDGET` | `400` | system prompt 中记忆最大 token 数 |
| `LELAMP_MEMORY_CONSOLIDATION_MIN_TURNS` | `10` | 触发整合的最少对话轮数 |
| `LELAMP_MEMORY_CONSOLIDATION_COOLDOWN_S` | `300` | 整合冷却时间 (秒) |
| `LELAMP_MEMORY_MAX_CONTENT_LENGTH` | `500` | 单条记忆最大字符数 |

#### 优雅降级策略

| 层级 | 场景 | 降级行为 |
|------|------|----------|
| 1 | 初始化失败 | 记录警告，`_memory_initialized=False`，Agent 正常运行 |
| 2 | 数据库操作失败 | MemoryStore 捕获异常，返回空结果 |
| 3 | 整合 API 不可用 | `asyncio.create_task` 静默失败，对话不中断 |
| 4 | LLM 返回格式错误 | JSON 解析失败返回 None，不写入脏数据 |
| 5 | 工具调用失败 | 检查 `_memory_tools is None`，返回友好提示 |
| 6 | Prompt 构建失败 | try/except 回退到静态 `_INSTRUCTIONS` |

---

### 7. 集成层 (`lelamp/integrations/`)

**设计理念**: 统一错误处理、重试机制、降级策略

#### 百度语音集成 (`baidu_speech.py`)

**架构：**
```python
BaiduShortSpeechSTT 继承自 livekit.agents.stt.STT
BaiduTTS 继承自 livekit.agents.tts.TT
```

**状态回调机制：**
```python
async def _on_state(state: str) -> None:
    """会话状态变化回调"""
    await agent.set_conversation_state(state)

async def _on_transcript(text: str) -> None:
    """语音转录文本回调"""
    await agent.note_user_text(text)
```

**音频处理流程：**
1. **STT**: AudioFrame → PCM16 单声道 16kHz → Base64 编码 → 百度 API
2. **TTS**: 文本 → 分块 → 百度 API → PCM 音频流

**认证管理：**
使用共享的 `BaiduAuth` 类管理 OAuth 令牌，支持自动刷新。

#### 错误处理

```python
class IntegrationError(Exception):
    """
    集成错误基类

    属性:
    - retryable: 是否可重试
    - provider: 服务提供商
    """
```

#### 重试装饰器

```python
@retry_on_error(
    max_attempts=3,
    base_delay=1.0,
    max_delay=10.0,
    exponential_base=2
)
async def call_api():
    """
    自动重试的 API 调用

    策略:
    - 指数退避
    - 最大重试次数
    - 最大延迟限制
    """
```

---

## 🔒 隐私保护

### CameraPrivacyManager (`lelamp/service/vision/privacy.py`)

摄像头隐私管理，保护用户隐私：

| 状态 | 说明 | LED 指示 |
|------|------|----------|
| IDLE | 空闲状态 | 关闭 |
| ACTIVE | 正在使用摄像头 | 红色呼吸 |
| PAUSED | 暂停使用 | 关闭 |
| CONSENT_REQUIRED | 需要用户同意 | 黄色闪烁 |

**关键特性：**
- 用户同意系统，TTL 默认 1 小时
- 使用统计追踪（会话次数、总使用时长）
- 线程安全的状态管理（`threading.Lock`）
- `PrivacyGuard` 上下文管理器用于自动激活/停用

---

## 💎 商业化功能

### 设备授权

```bash
LELAMP_LICENSE_KEY      # 设备授权
LELAMP_LICENSE_SECRET   # 签名密钥（生产必需）
LELAMP_DEV_MODE         # 开发模式跳过授权
```

### 电机健康监控 (`lelamp/service/motors/health_monitor.py`)

**监控指标：**
- 温度监控 (`LELAMP_MOTOR_TEMP_WARNING_C`, `LELAMP_MOTOR_TEMP_CRITICAL_C`)
- 电压监控 (`LELAMP_MOTOR_VOLTAGE_MIN_V`, `LELAMP_MOTOR_VOLTAGE_MAX_V`)
- 负载监控 (`LELAMP_MOTOR_LOAD_WARNING`, `LELAMP_MOTOR_LOAD_STALL`)
- 位置误差 (`LELAMP_MOTOR_POSITION_ERROR_DEG`)

**健康状态：**
| 状态 | 说明 | 动作 |
|------|------|------|
| HEALTHY | 正常运行 | - |
| WARNING | 指标异常警告 | 记录日志 |
| CRITICAL | 临界状态 | 准备保护 |
| STALLED | 检测到堵转 | 停止播放 |

**自动保护：**
- 后台健康检查线程（默认 5 分钟间隔）
- 堵转时自动停止播放
- 健康历史追踪和统计

### OTA 更新 (`lelamp/utils/ota.py`)

**OTAManager 功能：**
- 版本检查（语义化版本）
- 安全下载（强制 SHA256 验证）
- HTTPS 强制（SSL 证书验证）
- 自动回滚（更新失败时）
- 线程安全更新操作

---

## 🔄 核心数据流

### 语音对话流程

```
用户语音
   ↓
WebRTC (LiveKit)
   ↓
Silero VAD (语音活动检测)
   ↓
BVC (噪音消除)
   ↓
BaiduShortSpeechSTT (语音识别)
   ↓
DeepSeek LLM (意图理解)
   ↓ ← 记忆注入 (动态 system prompt)
Function Tools 调用
   ├─ MotorsService (舵机控制)
   ├─ RGBService (灯光效果)
   ├─ VisionService (视觉识别)
   ├─ MemoryTools (记忆工具: save/recall/forget)
   └─ SystemTools (系统功能)
   ↓
BaiduTTS (语音合成)
   ↓
WebRTC → 用户
   ↓
后台: 对话轮次达 10 轮 → 记忆整合 (DeepSeek API) → 持久化
```

### API 服务数据流

### 1. 用户注册流程

```
客户端
   ↓
POST /api/auth/register
   ↓
路由处理器 (auth.py)
   ↓
服务层 (AuthService.register_user)
   ↓
密码哈希 (bcrypt)
   ↓
数据库插入 (User)
   ↓
生成令牌 (access_token + refresh_token)
   ↓
返回响应
```

### 2. 设备控制流程

```
客户端 (WebSocket)
   ↓
WebSocket 连接 (/api/ws/{lamp_id})
   ↓
可选 JWT 认证
   ↓
连接建立
   ↓
接收实时消息
   - state_changed
   - motor_moved
   - rgb_changed
   - operation_log
   ↓
客户端更新 UI
```

### 3. API 缓存流程

```
GET 请求
   ↓
缓存中间件
   ↓
检查缓存
   ├─ 命中 → 返回缓存数据
   └─ 未命中 ↓
         ↓
      执行处理器
         ↓
      返回响应
         ↓
      写入缓存 (TTL)
         ↓
      返回客户端
```

---

## 🛡️ 安全架构

### 1. 认证流程

```
用户凭证
   ↓
AuthService.authenticate_user
   ↓
验证密码 (bcrypt)
   ↓
生成 JWT (access_token + refresh_token)
   ↓
返回令牌
   ↓
客户端存储
   ↓
后续请求携带令牌
   ↓
get_current_user 验证
   ↓
允许访问
```

### 2. 速率限制流程

```
API 请求
   ↓
提取标识符 (user_id 或 IP)
   ↓
查询请求历史
   ↓
清理过期请求
   ↓
计算当前窗口请求数
   ↓
判断是否超限
   ├─ 超限 → 返回 429
   └─ 未超限 ↓
         ↓
      记录请求
         ↓
      处理请求
```

### 3. 输入验证流程

```
请求数据
   ↓
Pydantic 模型验证
   ├─ 类型验证
   ├─ 范围验证
   └─ 格式验证 (EmailStr)
   ↓
验证失败 → 返回 400
   ↓
验证通过
   ↓
业务逻辑处理
```

---

## 🚀 性能优化

### 1. 数据库优化

**索引策略**:
- 单列索引: `username`, `email`, `operation_type`
- 复合索引: `(lamp_id, timestamp)`, `(lamp_id, success, timestamp)`
- 查询性能提升: 50-70%

**查询优化**:
```python
# ❌ 低效查询 (全表扫描)
logs = db.query(OperationLog).all()

# ✅ 高效查询 (使用索引)
logs = db.query(OperationLog)\
    .filter(OperationLog.lamp_id == lamp_id)\
    .order_by(OperationLog.timestamp.desc())\
    .limit(10)\
    .all()
```

### 2. API 缓存

**缓存策略**:
- 仅缓存 GET 请求
- 默认 TTL: 60 秒
- 基于 URL 的缓存键
- 自动过期清理

**缓存命中率**:
- 设备状态查询: 70-80%
- 设置查询: 60-70%
- 历史记录: 50-60%

### 3. 并发处理

**异步 I/O**:
- FastAPI 异步端点
- 异步数据库操作 (SQLAlchemy 2.0)
- 异步 WebSocket 连接

**线程安全**:
- `threading.Lock` 保护共享状态
- 优先级队列线程安全
- 数据库会话隔离

---

## 📊 监控与日志

### 日志系统

**日志级别**:
- DEBUG: 详细调试信息
- INFO: 一般信息
- WARNING: 警告信息
- ERROR: 错误信息
- CRITICAL: 严重错误

**日志配置**:
```python
LOG_LEVEL=INFO
LELAMP_LOG_TO_FILE=false
LELAMP_LOG_DIR=logs
LELAMP_LOG_JSON=false
```

### 监控指标

**API 性能**:
- 请求响应时间
- 请求成功率
- 缓存命中率
- 速率限制触发率

**数据库性能**:
- 查询执行时间
- 慢查询日志
- 连接池使用率
- 索引使用率

---

## 🧪 测试架构

### 测试分层

```
单元测试 (Unit Tests)
   ├─ 模型测试 (test_auth_models.py)
   ├─ 服务测试 (test_auth_service.py)
   └─ 工具测试 (test_rate_limit.py)

集成测试 (Integration Tests)
   ├─ API 路由测试 (test_auth_routes.py)
   ├─ 中间件测试 (test_auth_middleware.py)
   ├─ WebSocket 测试 (test_websocket_auth.py)
   └─ 数据库测试 (test_database_performance.py)

端到端测试 (E2E Tests)
   └─ 完整用户流程测试
```

### 测试覆盖率

**目标**: 60%+ (已达成 61%)

**覆盖范围**:
- 认证系统: 90%+
- API 端点: 75%+
- 中间件: 70%+
- 数据库: 60%+

---

## 📦 部署架构

### 开发环境

```
localhost
├── FastAPI Server (uvicorn --reload)
├── SQLite Database
└── File System (logs/)
```

### 生产环境

```
负载均衡器 (Nginx)
   ↓
FastAPI Workers (4+ instances)
   ├─ Gunicorn + Uvicorn workers
   ├─ PostgreSQL Database
   ├─ Redis Cache (可选)
   └─ File System (logs/ + recordings/)
```

---

## 📚 相关文档

- [API 文档](API.md) - API 使用指南
- [功能说明](FEATURES.md) - 功能特性介绍
- [安全指南](SECURITY.md) - 安全最佳实践
- [部署指南](DEPLOYMENT_GUIDE.md) - 生产部署说明
- [设置指南](SETUP_GUIDE.md) - 开发环境配置
- [用户指南](USER_GUIDE.md) - 用户使用手册

---

## 📄 关键文件路径

| 文件 | 功能 |
|------|------|
| `main.py` | LiveKit Agent 入口点 |
| `lelamp/agent/lelamp_agent.py` | LeLamp Agent 主类 |
| `lelamp/agent/states.py` | 状态管理 |
| `lelamp/agent/tools/*.py` | Function Tools |
| `lelamp/integrations/baidu_speech.py` | 百度语音适配器 |
| `lelamp/service/base.py` | 服务基类 |
| `lelamp/service/motors/motors_service.py` | 电机服务 |
| `lelamp/service/rgb/rgb_service.py` | RGB 服务 |
| `lelamp/service/vision/vision_service.py` | 视觉服务 |
| `lelamp/service/vision/privacy.py` | 隐私保护管理 |
| `lelamp/edge/face_detector.py` | 人脸检测服务 |
| `lelamp/edge/hand_tracker.py` | 手势追踪服务 |
| `lelamp/edge/object_detector.py` | 物体检测服务 |
| `lelamp/edge/hybrid_vision.py` | 混合推理路由 |
| `lelamp/agent/tools/edge_vision_tools.py` | 边缘视觉工具 |
| `lelamp/agent/tools/memory_tools.py` | 记忆工具 (save/recall/forget) |
| `lelamp/memory/models.py` | 记忆数据模型 |
| `lelamp/memory/store.py` | 记忆存储 (CRUD + 线程安全) |
| `lelamp/memory/consolidator.py` | LLM 记忆整合 |
| `lelamp/memory/config.py` | 记忆系统配置 |
| `lelamp/config.py` | 配置管理 |
| `lelamp/utils/rate_limiter.py` | 速率限制 |
| `lelamp/utils/ota.py` | OTA 更新 |
| `lelamp/api/app.py` | FastAPI 应用 |

---

**最后更新**: 2026-03-31
**版本**: v3.2 (新增记忆系统模块)
