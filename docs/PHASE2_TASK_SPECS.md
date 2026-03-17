# Phase 2 剩余任务详细规范

**生成时间**: 2026-03-17
**参考**: Task 2.1-2.4 的执行模式和代码风格

---

## Task 2.5: 创建 vision_tools.py 和 system_tools.py

### 目标

从 `main.py` 中提取视觉相关和系统相关的 `@function_tool` 方法，保持与 `motor_tools.py` 和 `rgb_tools.py` 一致的代码风格。

### 2.5.1: 创建 lelamp/agent/tools/vision_tools.py

**文件路径**: `lelamp/agent/tools/vision_tools.py`

**必需功能**:
1. `VisionTools` 类，接受 `vision_service`, `qwen_client`, `rgb_service`, `state_manager` 初始化
2. 三个 `@function_tool` 方法：
   - `vision_answer(question: str) -> str`
   - `check_homework() -> str`
   - `capture_to_feishu() -> str`

**关键规范**:
- 使用 `TYPE_CHECKING` 进行类型提示，避免循环导入
- 日志使用 `logging.getLogger("lelamp.agent.tools.vision")`
- 遵循 `main.py` 中原有的业务逻辑（速率限制、灯光状态覆盖/恢复、飞书推送）
- 验证服务是否初始化（`if not self._vision_service` 等）
- 使用 `threading.Lock` 保护时间戳变量（与 main.py 一致）

**代码结构参考**:
```python
"""
视觉工具模块
"""
import logging
import base64
import json
import time
import uuid
import urllib.request
import asyncio
from typing import TYPE_CHECKING

from livekit.agents import function_tool

if TYPE_CHECKING:
    from lelamp.service.vision.vision_service import VisionService
    from lelamp.integrations.qwen_vl import Qwen3VLClient
    from lelamp.service.rgb.rgb_service import RGBService
    from lelamp.agent.states import StateManager

class VisionTools:
    """视觉控制工具类 - 管理摄像头、视觉问答和照片推送"""

    def __init__(
        self,
        vision_service: "VisionService",
        qwen_client: "Qwen3VLClient",
        rgb_service: "RGBService",
        state_manager: "StateManager",
    ):
        # ... 初始化代码

    @function_tool
    async def vision_answer(self, question: str) -> str:
        """..."""

    @function_tool
    async def check_homework(self) -> str:
        """..."""

    @function_tool
    async def capture_to_feishu(self) -> str:
        """..."""
```

**线程安全要求**:
- `vision_answer` 和 `check_homework` 需要保护 `light_override_until_ts`
- 使用 `state_manager.set_light_override()` 而不是直接操作时间戳

**测试文件**: `lelamp/test/integration/test_vision_tools.py`
- 8-10 个集成测试
- 测试视觉问答、作业检查、飞书推送
- Mock `vision_service`, `qwen_client`, `rgb_service`

---

### 2.5.2: 创建 lelamp/agent/tools/system_tools.py

**文件路径**: `lelamp/agent/tools/system_tools.py`

**必需功能**:
1. `SystemTools` 类，接受 `motors_service`, `rgb_service`, `ota_manager`, `state_manager` 等初始化
2. 多个 `@function_tool` 方法（按功能分组）

**工具方法列表**:

| 方法名 | 功能 | 优先级 |
|--------|------|--------|
| `get_available_recordings()` | 获取可用录制列表 | HIGH |
| `set_rgb_brightness(percent: int)` | 设置灯光亮度 | HIGH |
| `rgb_effect_wave(...)` | 波纹效果 | NORMAL |
| `rgb_effect_fire(...)` | 火焰效果 | NORMAL |
| `rgb_effect_emoji(...)` | 表情动画 | NORMAL |
| `stop_rgb_effect()` | 停止灯效 | NORMAL |
| `set_volume(volume_percent: int)` | 系统音量 | HIGH |
| `get_rate_limit_stats()` | API 速率统计 | LOW |
| `web_search(query: str)` | 联网搜索 | HIGH |
| `check_for_updates()` | 检查更新 | NORMAL |
| `perform_ota_update()` | OTA 更新 | NORMAL |
| `tune_motor_pid(...)` | 电机 PID 调参 | LOW |
| `reset_motor_health_stats(motor_name: str)` | 重置健康统计 | LOW |

**关键规范**:
- 保持与 `main.py` 完全一致的业务逻辑
- `rgb_effect_*` 方法使用 `state_manager.set_light_override()`
- `web_search` 需要验证 URL（使用 `validate_external_url`）
- `tune_motor_pid` 需要访问 `motors_service.robot.bus`
- OTA 相关方法需要处理重启逻辑

**代码结构参考**:
```python
"""
系统工具模块
"""
import asyncio
import json
import logging
import os
import sys
import threading
import time
import urllib.request
from typing import TYPE_CHECKING, Optional

from livekit.agents import function_tool

from lelamp.service import Priority
from lelamp.utils import get_all_rate_limiter_stats
from lelamp.utils.url_validation import validate_external_url, ALLOWED_API_DOMAINS

if TYPE_CHECKING:
    from lelamp.service.motors.motors_service import MotorsService
    from lelamp.service.rgb.rgb_service import RGBService
    from lelamp.agent.states import StateManager

# 复用 main.py 中的 SAFE_JOINT_RANGES
from lelamp.agent.tools.motor_tools import SAFE_JOINT_RANGES

class SystemTools:
    """系统工具类 - 音量、搜索、更新、健康监控等功能"""

    def __init__(
        self,
        motors_service: "MotorsService",
        rgb_service: "RGBService",
        ota_manager,  # 类型取决于 OTAManager 实现
        ota_url: str,
        state_manager: "StateManager",
    ):
        # ... 初始化代码

    # 电机相关
    @function_tool
    async def get_available_recordings(self) -> str:
        """..."""

    @function_tool
    async def tune_motor_pid(self, motor_name: str, p_coefficient: int, ...) -> str:
        """..."""

    @function_tool
    async def reset_motor_health_stats(self, motor_name: Optional[str] = None) -> str:
        """..."""

    # RGB 效果扩展
    @function_tool
    async def set_rgb_brightness(self, percent: int) -> str:
        """..."""

    @function_tool
    async def rgb_effect_wave(self, red: int = 60, ...) -> str:
        """..."""

    # ... 其他 rgb_effect_* 方法

    # 系统控制
    @function_tool
    async def set_volume(self, volume_percent: int) -> str:
        """..."""

    @function_tool
    async def web_search(self, query: str) -> str:
        """..."""

    # OTA 更新
    @function_tool
    async def check_for_updates(self) -> str:
        """..."""

    @function_tool
    async def perform_ota_update(self) -> str:
        """..."""
```

**测试文件**: `lelamp/test/integration/test_system_tools.py`
- 15-20 个集成测试
- 测试每个主要功能组
- Mock 所有外部依赖

---

### 2.5.3: 更新 lelamp/agent/tools/__init__.py

**目标**: 导出新的工具类

```python
"""Agent 工具模块"""

from lelamp.agent.tools.motor_tools import MotorTools, SAFE_JOINT_RANGES
from lelamp.agent.tools.rgb_tools import RGBTools
from lelamp.agent.tools.vision_tools import VisionTools
from lelamp.agent.tools.system_tools import SystemTools

__all__ = [
    "MotorTools",
    "SAFE_JOINT_RANGES",
    "RGBTools",
    "VisionTools",
    "SystemTools",
]
```

---

## Task 2.6: 创建 lelamp/agent/lelamp_agent.py

### 目标

将 `LeLamp` 类从 `main.py` 提取到独立模块，实现核心代理逻辑。

**文件路径**: `lelamp/agent/lelamp_agent.py`

### 必需功能

1. `LeLamp` 类继承 `Agent`
2. 保留核心方法：
   - `__init__`: 初始化服务和工具
   - `note_user_text(text: str)`: 记录用户输入
   - `set_conversation_state(state: str)`: 设置会话状态并更新灯光
   - `_set_system_volume(volume_percent: int)`: 异步音量控制
   - Data Channel 处理方法：
     - `handle_data_message(data: bytes, participant)`
     - `_execute_command(action: str, params: dict) -> str`
     - `_execute_rgb_effect(effect_name: str) -> str`
     - `_send_chat_message(content: str)`
     - `_send_vision_result(result: str, image_base64: str = None)`
     - `_update_camera_status(active: bool)`

### 关键规范

- 保持与原 `main.py` 相同的初始化流程
- 集成 `StateManager` 进行状态管理
- 集成所有工具类（MotorTools, RGBTools, VisionTools, SystemTools）
- 保留 Data Channel 消息处理逻辑（Web Client v2.0 支持）
- 保留 `_restart_later()` 方法

### 代码结构

```python
"""
LeLamp 代理主类
"""
import asyncio
import json
import logging
import os
import sys
 threading
import time
import base64
from typing import TYPE_CHECKING, Optional

from livekit.agents import Agent

if TYPE_CHECKING:
    from lelamp.service.motors.motors_service import MotorsService
    from lelamp.service.rgb.rgb_service import RGBService
    from lelamp.service.vision.vision_service import VisionService
    from lelamp.integrations.qwen_vl import Qwen3VLClient

from lelamp.agent.states import StateManager, StateColors
from lelamp.agent.tools import MotorTools, RGBTools, VisionTools, SystemTools
from lelamp.service import Priority

class LeLamp(Agent):
    """LeLamp 机器人台灯代理"""

    # 人格设定
    instructions = """..."""  # 保持原有内容

    def __init__(
        self,
        port: str,
        lamp_id: str,
        vision_service: "VisionService",
        qwen_client: "Qwen3VLClient",
        ota_url: str,
        motor_config,
    ):
        super().__init__()
        # ... 初始化代码

        # 创建状态管理器
        self.state_manager = StateManager()

        # 创建工具实例
        self.motor_tools = MotorTools(self.motors_service, self.state_manager)
        self.rgb_tools = RGBTools(self.rgb_service, self.state_manager)
        self.vision_tools = VisionTools(
            self._vision_service, self._qwen_client,
            self.rgb_service, self.state_manager
        )
        self.system_tools = SystemTools(
            self.motors_service, self.rgb_service,
            self._ota_manager, ota_url, self.state_manager
        )

    # 工具方法通过工具类暴露
    # 将 MotorTools, RGBTools, VisionTools, SystemTools 的 @function_tool 方法注册到代理

    async def note_user_text(self, text: str) -> None:
        """..."""

    async def set_conversation_state(self, state: str) -> None:
        """..."""

    # Data Channel 方法
    async def handle_data_message(self, data: bytes, participant):
        """..."""
```

### 注册工具方法

LeLamp 类需要将工具类的方法注册为代理的 function_tool。有几种方式：

**方式 A**: 直接引用工具方法（推荐）
```python
# 在 LeLamp.__init__ 中
self.play_recording = self.motor_tools.play_recording
self.move_joint = self.motor_tools.move_joint
# ... 对所有工具方法
```

**方式 B**: 使用 `@function_tool` 装饰器包装
```python
from livekit.agents import function_tool

@function_tool
async def play_recording(self, recording_name: str) -> str:
    return await self.motor_tools.play_recording(recording_name)
```

### 测试文件

`lelamp/test/integration/test_lelamp_agent.py`
- 测试状态切换
- 测试 Data Channel 消息处理
- Mock 所有服务依赖

---

## Task 2.7: 简化 main.py

### 目标

将 `main.py` 简化为入口文件，只保留配置加载和代理启动逻辑。

### 必需功能

1. 导入语句
2. 日志设置
3. `_load_config()` 函数（从 main.py 提取）
4. `_setup_logging()` 函数
5. `_build_vad()` 函数
6. `entrypoint(ctx: JobContext)` 异步函数
7. `if __name__ == "__main__"` 块

### 目标长度

< 100 行（不包括配置加载辅助函数）

### 代码结构

```python
"""LeLamp Runtime - 主入口文件"""
import sys
import logging
from livekit.agents import cli, JobContext, WorkerOptions, RoomInputOptions
from livekit.plugins import noise_cancellation, openai, silero
from dotenv import load_dotenv

from lelamp.config import AppConfig, load_motor_config
from lelamp.agent.lelamp_agent import LeLamp
from lelamp.service.vision.vision_service import VisionService
from lelamp.integrations.qwen_vl import Qwen3VLClient
from lelamp.integrations.baidu_speech import BaiduShortSpeechSTT, BaiduTTS
from lelamp.utils.logging import setup_logging
from lelamp.utils.security import verify_license

load_dotenv()
logger = logging.getLogger("lelamp")

# 读取版本号
try:
    with open("VERSION", "r") as f:
        LELAMP_VERSION = f.read().strip()
except FileNotFoundError:
    LELAMP_VERSION = "0.0.0"

def _load_config() -> AppConfig:
    """加载应用配置"""
    return AppConfig.from_env()

def _setup_logging():
    """设置日志系统"""
    setup_logging()

def _build_vad():
    """构建 VAD 实例"""
    import os
    return silero.VAD.load(
        min_speech_duration=float(os.getenv("LELAMP_VAD_MIN_SPEECH_DURATION", "0.5")),
        min_silence_duration=float(os.getenv("LELAMP_VAD_MIN_SILENCE_DURATION", "0.8")),
        prefix_padding_duration=float(os.getenv("LELAMP_VAD_PREFIX_PADDING_DURATION", "0.3")),
        activation_threshold=float(os.getenv("LELAMP_VAD_ACTIVATION_THRESHOLD", "0.5")),
    )

async def entrypoint(ctx: JobContext):
    """LiveKit 代理入口点"""
    config = _load_config()
    await ctx.connect()

    # 创建 LLM
    deepseek_llm = openai.LLM(
        model=config.deepseek_model,
        base_url=config.deepseek_base_url,
        api_key=config.deepseek_api_key,
    )

    # 创建视觉客户端
    qwen_client = Qwen3VLClient(
        base_url=config.modelscope_base_url,
        api_key=config.modelscope_api_key,
        model=config.modelscope_model,
        timeout_s=config.modelscope_timeout_s,
    )

    # 创建并启动视觉服务
    vision_service = VisionService(
        enabled=config.vision_enabled,
        index_or_path=config.camera_index_or_path,
        width=config.camera_width,
        height=config.camera_height,
        capture_interval_s=config.vision_capture_interval_s,
        jpeg_quality=config.vision_jpeg_quality,
        max_age_s=config.vision_max_age_s,
        rotate_deg=config.camera_rotate_deg,
        flip=config.camera_flip,
    )
    vision_service.start()

    # 加载电机配置
    motor_config = load_motor_config()

    logger.info(
        "config ready: lamp_id=%s port=%s vision=%s",
        config.lamp_id, config.lamp_port, config.vision_enabled,
    )

    # 创建代理
    agent = LeLamp(
        port=config.lamp_port,
        lamp_id=config.lamp_id,
        vision_service=vision_service,
        qwen_client=qwen_client,
        ota_url=config.ota_url,
        motor_config=motor_config,
    )

    # 会话状态回调
    async def _on_state(state: str) -> None:
        await agent.set_conversation_state(state)

    async def _on_transcript(text: str) -> None:
        await agent.note_user_text(text)

    # 创建会话
    from livekit.agents import AgentSession
    session = AgentSession(
        vad=_build_vad(),
        stt=BaiduShortSpeechSTT(
            api_key=config.baidu_api_key,
            secret_key=config.baidu_secret_key,
            cuid=config.baidu_cuid,
            state_cb=_on_state,
            transcript_cb=_on_transcript,
        ),
        llm=deepseek_llm,
        tts=BaiduTTS(
            api_key=config.baidu_api_key,
            secret_key=config.baidu_secret_key,
            cuid=config.baidu_cuid,
            per=config.baidu_tts_per,
            state_cb=_on_state,
        ),
    )

    # 噪声抑制
    start_kwargs: dict[str, object] = {}
    if config.noise_cancellation_enabled:
        start_kwargs["room_input_options"] = RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        )

    # Data Channel 监听
    @ctx.room.on("data_received")
    def on_data_received(data: bytes, participant):
        asyncio.create_task(agent.handle_data_message(data, participant))

    try:
        await session.start(agent=agent, room=ctx.room, **start_kwargs)
        if config.greeting_text:
            await session.say(config.greeting_text, allow_interruptions=False)
    finally:
        vision_service.stop()

if __name__ == "__main__":
    _setup_logging()

    # 商业化保护：启动时校验设备授权
    if not verify_license():
        logger.fatal("设备授权校验失败。请检查 LELAMP_LICENSE_KEY 配置。")
        sys.exit(1)

    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

### 测试

- 运行 `uv run main.py console` 验证代理正常启动
- 验证所有功能工具仍然可用

---

## 质量标准（与 Task 2.1-2.4 保持一致）

### 代码质量
- 类型提示完整（使用 `TYPE_CHECKING` 避免循环导入）
- 日志使用规范命名（`lelamp.agent.tools.*`）
- 错误处理完整
- 输入验证

### 线程安全
- 使用 `StateManager` 管理状态和时间戳
- 跨线程访问使用 `threading.Lock`

### 测试要求
- 集成测试使用 `pytest` 和 `pytest-asyncio`
- Mock 所有外部依赖
- 覆盖率目标：80%+

### Git 提交
- 每个任务一个独立的 commit
- Commit message 格式：`refactor: extract vision tools to dedicated module`

---

## 执行顺序

1. **Task 2.5.1**: 创建 `vision_tools.py` + 测试
2. **Task 2.5.2**: 创建 `system_tools.py` + 测试
3. **Task 2.5.3**: 更新 `tools/__init__.py`
4. **Task 2.6**: 创建 `lelamp_agent.py` + 测试
5. **Task 2.7**: 简化 `main.py` + 验证

---

**预计工作量**:
- Task 2.5: ~2-3 小时（两个工具类）
- Task 2.6: ~1-2 小时（代理类提取）
- Task 2.7: ~1 小时（简化入口）

**总计**: ~4-6 小时
