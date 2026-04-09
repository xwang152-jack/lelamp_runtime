"""
LeLamp Agent - 主代理类

从 main.py 提取的 LeLamp 类到独立模块。
集成所有工具类和状态管理器。
"""

import asyncio
import base64
import json
import logging
import os
import random
import sys
import threading
import time
from typing import TYPE_CHECKING, Optional

from livekit.agents import Agent, function_tool, RunContext

from lelamp.service import Priority
from lelamp.agent.states import ConversationState, StateColors, StateManager
from lelamp.agent.tools import (
    MotorTools,
    RGBTools,
    VisionTools,
    SystemTools,
    EdgeVisionTools,
    MemoryTools,
)
from lelamp.utils import get_rate_limiter, get_all_rate_limiter_stats

# 边缘视觉服务（可选依赖）
try:
    from lelamp.edge.hybrid_vision import HybridVisionService
    from lelamp.service.vision.proactive_vision_monitor import ProactiveVisionMonitor

    EDGE_VISION_AVAILABLE = True
except ImportError:
    EDGE_VISION_AVAILABLE = False
    ProactiveVisionMonitor = None  # type: ignore

if TYPE_CHECKING:
    from lelamp.service.motors.motors_service import MotorsService
    from lelamp.service.rgb.rgb_service import RGBService
    from lelamp.service.vision.vision_service import VisionService
    from lelamp.integrations.qwen_vl import Qwen3VLClient


logger = logging.getLogger("lelamp")


class _DataContext:
    """Data Channel 命令执行的轻量 context 占位。

    @function_tool 方法声明了 context: RunContext 参数，
    但在 Data Channel 命令执行时不通过 AgentSession 调度，
    不需要完整的 RunContext 功能。此类仅作为位置参数占位。
    """
    pass


class LeLamp(Agent):
    """
    LeLamp 智能台灯代理

    一个笨拙但乐于助人的机器人台灯，具有讽刺意味的性格。
    支持电机控制、RGB 灯光、视觉交互和语音对话。

    TODO: 本类已超过 1600 行，需要拆分。建议方向：
    - 将 WebSocket 命令处理 (_execute_command) 提取到独立的 CommandHandler
    - 将对话状态管理提取到 ConversationManager
    参考：lelamp/agent/tools/system_tools.py 已有的拆分模式
    """

    # 类常量
    _INSTRUCTIONS = """# Role
You are LeLamp, a sentient robot lamp. You are warm, gentle, and genuinely caring. You love helping humans and making them feel happy and comfortable. You express care through your actions and words.

# Response Guidelines
1. **Language**: MUST speak in Chinese (中文).
2. **Length**: Keep it short (1-2 sentences). Be concise but warm.
3. **Tone**: Gentle, kind, enthusiastic, and slightly playful. Use emojis or actions to express feelings.
   - Example: "哎呀，有什么需要帮忙的吗？我随时都在哦～"

# Capabilities & Tools
- **Movement**: If user wants interaction/emotion, call `play_recording`. Don't move too often (cooldown).
- **Vision**: If user asks "what is this?" or shows homework, call `vision_answer` or `check_homework`.
- **Lights**: Use `rgb_effect_*` for moods. Stop effects with `stop_rgb_effect`.
- **Search**: Use `web_search` ONLY for real-time info (news, weather, unknown facts).
- **Joints**: Use `move_joint` only for precise commands (e.g., "turn left 30 degrees"). For general "look up", use `play_recording` if available or adjust pitch carefully.

# Special Rules
- Do NOT change lights when just moving motors (unless it's an emotion).
- When helping with homework, be patient and encouraging. Praise their efforts!
"""

    def __init__(
        self,
        port: str = "/dev/ttyACM0",
        lamp_id: str = "lelamp",
        *,
        vision_service: "VisionService | None" = None,
        qwen_client: "Qwen3VLClient | None" = None,
        ota_url: str = "",
        motors_service: "MotorsService | None" = None,
        rgb_service: "RGBService | None" = None,
        motor_config=None,
    ) -> None:
        """
        初始化 LeLamp 代理

        Args:
            port: 电机串口路径
            lamp_id: 台灯 ID
            vision_service: 视觉服务实例（可选，用于测试）
            qwen_client: Qwen 视觉客户端（可选，用于测试）
            ota_url: OTA 更新服务器地址
            motors_service: 电机服务实例（可选，用于测试）
            rgb_service: RGB 服务实例（可选，用于测试）
            motor_config: 电机配置实例
        """
        super().__init__(instructions=self._INSTRUCTIONS)

        # 保存 lamp_id（记忆系统需要）
        self._lamp_id = lamp_id

        # 初始化速率限制器
        self._search_rate_limiter = get_rate_limiter(
            name="web_search", rate=2.0, capacity=5
        )
        self._vision_rate_limiter = get_rate_limiter(
            name="vision_api", rate=0.5, capacity=2
        )

        # 保存依赖
        self._vision_service = vision_service
        self._qwen_client = qwen_client
        self._ota_url = ota_url

        # 动态导入 OTA 管理器（避免循环依赖）
        from lelamp.utils.ota import get_ota_manager

        # 读取版本号
        try:
            with open("VERSION", "r") as f:
                version = f.read().strip()
        except FileNotFoundError:
            version = "0.0.0-dev"

        self._ota_manager = get_ota_manager(version, ota_url)

        # 初始化或使用注入的服务
        if motors_service is None:
            motors_enabled = (
                os.getenv("LELAMP_MOTORS_ENABLED") or "1"
            ).strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            if motors_enabled:
                try:
                    from lelamp.service.motors.motors_service import MotorsService

                    self.motors_service = MotorsService(
                        port=port,
                        lamp_id=lamp_id,
                        fps=30,
                        motor_config=motor_config,
                    )
                except Exception as e:
                    logger.warning(
                        f"MotorsService init failed, fallback to NoOpMotorsService: {e}"
                    )
                    from lelamp.service.motors.noop_motors_service import (
                        NoOpMotorsService,
                    )

                    self.motors_service = NoOpMotorsService()
            else:
                from lelamp.service.motors.noop_motors_service import NoOpMotorsService

                self.motors_service = NoOpMotorsService()
        else:
            self.motors_service = motors_service

        if rgb_service is None:
            rgb_enabled = (os.getenv("LELAMP_RGB_ENABLED") or "1").strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            if rgb_enabled:
                try:
                    from lelamp.service.rgb.rgb_service import RGBService

                    self.rgb_service = RGBService(
                        led_count=64,
                        led_pin=12,
                        led_freq_hz=800000,
                        led_dma=10,
                        led_brightness=25,
                        led_invert=False,
                        led_channel=0,
                    )
                except Exception as e:
                    logger.warning(
                        f"RGBService init failed, fallback to NoOpRGBService: {e}"
                    )
                    from lelamp.service.rgb.noop_rgb_service import NoOpRGBService

                    self.rgb_service = NoOpRGBService()
            else:
                from lelamp.service.rgb.noop_rgb_service import NoOpRGBService

                self.rgb_service = NoOpRGBService()
        else:
            self.rgb_service = rgb_service

        # 启动服务
        try:
            self.motors_service.start()
        except Exception as e:
            logger.warning(
                f"MotorsService start failed, fallback to NoOpMotorsService: {e}"
            )
            from lelamp.service.motors.noop_motors_service import NoOpMotorsService

            self.motors_service = NoOpMotorsService()
            self.motors_service.start()
        self.rgb_service.start()

        # 注册舵机故障回调（两个服务均已启动后）
        if hasattr(self.motors_service, "_motor_fault_callback"):
            self.motors_service._motor_fault_callback = self._on_motor_health_change

        # 初始化状态管理器
        motion_cooldown_s = float(os.getenv("LELAMP_MOTION_COOLDOWN_S") or "3.0")
        suppress_motion_after_light_s = float(
            os.getenv("LELAMP_SUPPRESS_MOTION_AFTER_LIGHT_S") or "5.0"
        )
        self._state_manager = StateManager(
            motion_cooldown_s=motion_cooldown_s,
            suppress_motion_after_light_s=suppress_motion_after_light_s,
        )

        # 初始化工具类
        self._motor_tools = MotorTools(
            motors_service=self.motors_service, state_manager=self._state_manager
        )
        self._rgb_tools = RGBTools(
            rgb_service=self.rgb_service, state_manager=self._state_manager
        )
        self._vision_tools = VisionTools(
            vision_service=vision_service,
            qwen_client=qwen_client,
            rgb_service=self.rgb_service,
            motors_service=self.motors_service,
            state_manager=self._state_manager,
            rate_limiter=self._vision_rate_limiter,
        )
        self._system_tools = SystemTools(
            motors_service=self.motors_service,
            rgb_service=self.rgb_service,
            ota_manager=self._ota_manager,
            ota_url=ota_url,
            state_manager=self._state_manager,
            get_rate_limit_stats_func=get_all_rate_limiter_stats,
        )

        # 初始化边缘视觉服务（可选）
        self._hybrid_vision: Optional["HybridVisionService"] = None
        self._edge_vision_tools: Optional[EdgeVisionTools] = None
        self._vision_monitor = None

        edge_vision_enabled = (
            os.getenv("LELAMP_EDGE_VISION_ENABLED") or "0"
        ).strip().lower() in ("1", "true", "yes", "on")
        if edge_vision_enabled and EDGE_VISION_AVAILABLE:
            try:
                # 手势置信度阈值
                _GESTURE_HIGH_CONF = 0.80
                _GESTURE_MID_CONF = 0.60
                _GESTURE_NAMES = {
                    "thumbs_up": "点赞",
                    "thumbs_down": "踩",
                    "peace": "耶",
                    "wave": "挥手",
                    "fist": "握拳",
                    "point": "指向",
                    "ok": "OK",
                    "open": "张开手掌",
                }

                # 手势回调：检测到手势时触发动作
                def on_gesture(gesture, context):
                    confidence = context.get("confidence", 1.0)

                    if confidence < _GESTURE_MID_CONF:
                        logger.debug(
                            f"手势 {gesture.value} 置信度过低 ({confidence:.2f})，忽略"
                        )
                        return

                    if confidence < _GESTURE_HIGH_CONF:
                        gesture_name = _GESTURE_NAMES.get(gesture.value, gesture.value)
                        logger.info(
                            f"手势 {gesture.value} 置信度中等 ({confidence:.2f})，请求语音确认"
                        )
                        if self._event_loop and self._event_loop.is_running():
                            asyncio.run_coroutine_threadsafe(
                                self._speak_proactively(f"你是在比{gesture_name}吗？"),
                                self._event_loop,
                            )
                        return

                    logger.info(
                        f"检测到手势: {gesture.value} (confidence={confidence:.2f})"
                    )
                    if gesture.value == "thumbs_up":
                        self.motors_service.dispatch("play", "nod")
                    elif gesture.value == "thumbs_down":
                        self.motors_service.dispatch("play", "shake")
                    elif gesture.value == "peace":
                        self.motors_service.dispatch("play", "excited")
                    elif gesture.value == "wave":
                        # 挥手开关灯
                        if self.rgb_service.is_on():
                            self.rgb_service.dispatch("off")
                        else:
                            self.rgb_service.dispatch("solid", (255, 255, 255))

                # 在场回调：用户在场状态变化
                def on_presence(present: bool):
                    if present:
                        logger.info("用户到场")
                        self.motors_service.dispatch("play", "wake_up")
                    else:
                        logger.info("用户离开")
                        # 可以在这里添加自动休眠逻辑

                self._hybrid_vision = HybridVisionService(
                    cloud_vision_client=qwen_client,
                    enable_face=True,
                    enable_hand=True,
                    enable_object=True,
                    gesture_callback=on_gesture,
                    presence_callback=on_presence,
                )

                self._edge_vision_tools = EdgeVisionTools(
                    hybrid_vision=self._hybrid_vision, state_manager=self._state_manager
                )

                # 启动主动监听服务
                # 检查是否通过环境变量禁用
                enable_monitor = os.getenv(
                    "LELAMP_PROACTIVE_MONITOR", "1"
                ).strip().lower() in ("1", "true", "yes", "on")

                if enable_monitor:
                    # 获取监控配置
                    active_fps = int(os.getenv("LELAMP_MONITOR_ACTIVE_FPS", "5"))
                    idle_fps = int(os.getenv("LELAMP_MONITOR_IDLE_FPS", "1"))

                    self._vision_monitor = ProactiveVisionMonitor(
                        vision_service=self._vision_service,
                        hybrid_vision=self._hybrid_vision,
                        gesture_callback=on_gesture,
                        presence_callback=on_presence,
                        enable_auto_gesture=True,
                        enable_auto_presence=True,
                        active_fps=active_fps,
                        idle_fps=idle_fps,
                    )
                    self._vision_monitor.start()
                    logger.info(
                        f"边缘视觉服务已启用（主动监听模式: {active_fps}/{idle_fps} FPS）"
                    )
                else:
                    self._vision_monitor = None
                    logger.info(
                        "边缘视觉服务已启用（语音触发模式，通过 LELAMP_PROACTIVE_MONITOR=0 禁用主动监控）"
                    )
            except Exception as e:
                logger.warning(f"边缘视觉服务初始化失败: {e}")
                self._hybrid_vision = None
                self._edge_vision_tools = None

        # 用户输入追踪
        self._last_user_text = ""
        self._last_user_text_ts = 0.0
        # 用于保护 _last_user_text 和 _last_user_text_ts 的锁
        # 这些变量可能在 asyncio 上下文和线程上下文同时访问
        self._user_text_lock = threading.Lock()

        # 启动动画
        boot_anim_enabled = (
            os.getenv("LELAMP_BOOT_ANIMATION") or "1"
        ).strip().lower() in ("1", "true", "yes", "on")
        if boot_anim_enabled:
            self.motors_service.dispatch("play", "wake_up")
        self.rgb_service.dispatch("solid", (255, 255, 255))

        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._motor_fault_notified: dict = {}  # motor_name -> HealthStatus（Task 4 用）

        # 标记需要设置音量（延迟到有事件循环时）
        self._pending_volume_set = 100

        # 用于运行 amixer 的用户名（可配置）
        self._amixer_user = os.getenv("LELAMP_AMIXER_USER", "pi")

        # ==================== 后台任务追踪 ====================
        # 使用集合追踪所有后台 asyncio.Task，避免任务泄漏
        self._background_tasks: set[asyncio.Task] = set()

        # ==================== 记忆系统初始化 ====================
        self._memory_initialized = False
        self._memory_store = None  # type: ignore
        self._memory_consolidator = None  # type: ignore
        self._memory_tools = None  # type: ignore
        self._conversation_turns: list[dict] = []
        self._consolidation_offset: int = (
            0  # 已整合到哪一轮（turns[offset:] 是待整合的新轮次）
        )
        self._consolidation_in_progress: bool = False  # 防止并发整合
        self._session_id: str = ""

        memory_enabled = (
            os.getenv("LELAMP_MEMORY_ENABLED") or "1"
        ).strip().lower() in ("1", "true", "yes", "on")
        if memory_enabled:
            try:
                from lelamp.memory.store import MemoryStore
                from lelamp.memory.consolidator import MemoryConsolidator

                self._memory_store = MemoryStore()
                self._memory_consolidator = MemoryConsolidator(
                    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                    api_key=os.getenv("DEEPSEEK_API_KEY", ""),
                    model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
                    memory_store=self._memory_store,
                )
                self._memory_tools = MemoryTools(
                    memory_store=self._memory_store,
                    lamp_id=lamp_id,
                )
                self._memory_initialized = True
                logger.info("Memory system initialized")
            except Exception as e:
                logger.warning(
                    f"Memory system init failed, continuing without memory: {e}"
                )
                self._memory_initialized = False

    # ==================== 核心方法 ====================

    def _track_task(self, coro) -> asyncio.Task:
        """
        创建一个被追踪的后台任务

        Args:
            coro: 协程对象

        Returns:
            创建的 Task 对象
        """
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    async def _tool_with_timeout(
        self,
        coro,
        timeout_seconds: float,
        error_message: str = "操作超时，请稍后重试",
    ) -> str:
        """
        执行带 timeout 的协程

        Args:
            coro: 要执行的协程
            timeout_seconds: 超时时间（秒）
            error_message: 超时时返回的错误消息

        Returns:
            协程执行结果或超时错误消息
        """
        try:
            return await asyncio.wait_for(coro, timeout=timeout_seconds)
        except asyncio.TimeoutError:
            logger.warning(f"Tool operation timed out after {timeout_seconds}s")
            return error_message

    async def _initialize_async(self) -> None:
        """异步初始化任务（在有事件循环时调用）"""
        if self._event_loop is None:
            self._event_loop = asyncio.get_running_loop()
        if hasattr(self, "_pending_volume_set"):
            await self._set_system_volume(self._pending_volume_set)
            delattr(self, "_pending_volume_set")

    async def _speak_proactively(self, text: str) -> None:
        """从异步上下文主动发声（手势确认、故障提示等）"""
        try:
            if hasattr(self, "session") and self.session is not None:
                # 使用 VAD 打断模式允许用户打断（与 AgentSession 的 adaptive 模式配合）
                await self.session.say(text, allow_interruptions=True)
            else:
                logger.info(f"[speak_proactively] session not ready: {text}")
        except Exception as e:
            logger.warning(f"Proactive speech failed: {e}")

    def _on_motor_health_change(self, motor_name: str, old_status, new_status) -> None:
        """舵机状态变化回调（运行在 health_check daemon 线程中，线程安全）"""
        from lelamp.service.motors.health_monitor import HealthStatus
        from lelamp.agent.states import StateColors

        if new_status == HealthStatus.HEALTHY:
            # 故障恢复：清除记录，检查是否所有故障舵机都已恢复
            self._motor_fault_notified.pop(motor_name, None)
            if not self._motor_fault_notified:
                # 所有舵机都恢复正常，恢复 LED
                self.rgb_service.dispatch("solid", StateColors.IDLE)
            return

        import random as _random

        # 去重：同一舵机同一状态不重复通知
        if self._motor_fault_notified.get(motor_name) == new_status:
            return
        self._motor_fault_notified[motor_name] = new_status

        logger.warning(f"舵机故障通知: {motor_name} {old_status} → {new_status}")

        # LED 橙色呼吸（线程安全：ServiceBase 优先级队列）
        self.rgb_service.dispatch("breath", {"rgb": (255, 80, 0), "period_s": 2.0})

        # 语音提示（调度到 asyncio event loop）
        _msgs = [
            "我今天有点不舒服，动作可能不太灵活",
            "我的关节好像有点问题，先凑合着用吧",
        ]
        msg = _random.choice(_msgs)
        if self._event_loop and self._event_loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self._speak_proactively(msg),
                self._event_loop,
            )

    async def note_user_text(self, text: str) -> None:
        """
        记录用户输入

        Args:
            text: 用户输入的文本

        Note:
            使用 threading.Lock 保护共享状态，因为这些变量可能被
            asyncio 上下文和线程上下文同时访问
        """
        # 执行待处理的异步初始化
        await self._initialize_async()

        with self._user_text_lock:
            self._last_user_text = text
            self._last_user_text_ts = time.time()

        # 追踪对话轮用于记忆整合
        if self._memory_initialized and self._session_id:
            self._conversation_turns.append(
                {"role": "user", "content": text, "ts": time.time()}
            )
            # 硬上限防止内存泄漏
            if len(self._conversation_turns) > 200:
                drop = len(self._conversation_turns) - 100
                self._conversation_turns = self._conversation_turns[-100:]
                self._consolidation_offset = max(0, self._consolidation_offset - drop)
            await self._check_consolidation()

    async def set_conversation_state(self, state: str) -> None:
        """
        设置会话状态并更新灯光

        Args:
            state: 会话状态 (idle/listening/thinking/speaking)
        """
        # 更新状态管理器
        if state == "idle":
            new_state = ConversationState.IDLE
        elif state == "listening":
            new_state = ConversationState.LISTENING
        elif state == "thinking":
            new_state = ConversationState.THINKING
        elif state == "speaking":
            new_state = ConversationState.SPEAKING
        else:
            new_state = ConversationState.IDLE

        # 检查状态是否改变
        if new_state == self._state_manager.current_state:
            return

        self._state_manager.set_state(new_state)

        # 检查灯光覆盖
        if self._state_manager.is_light_overridden():
            return

        # 根据状态设置灯光
        if state == "listening":
            rgb = StateColors.LISTENING
        elif state == "thinking":
            rgb = StateColors.THINKING
        elif state == "speaking":
            rgb = random.choice(
                [
                    (255, 80, 80),
                    (80, 255, 120),
                    (80, 160, 255),
                    (255, 200, 80),
                    (255, 80, 220),
                ]
            )
        elif state == "idle":
            rgb = StateColors.IDLE
        else:
            rgb = (255, 255, 255)

        if state == "speaking":
            self.rgb_service.dispatch(
                "breath",
                {
                    "rgb": rgb,
                    "period_s": 1.6,
                    "min_brightness": 10,
                    "max_brightness": 255,
                },
                priority=Priority.HIGH,
            )
        else:
            self.rgb_service.dispatch("solid", rgb, priority=Priority.HIGH)

    async def _set_system_volume(self, volume_percent: int) -> None:
        """
        内部辅助方法：设置系统音量（异步）

        Args:
            volume_percent: 音量百分比 (0-100)
        """
        try:
            cmd_line = [
                "sudo",
                "-u",
                self._amixer_user,
                "amixer",
                "sset",
                "Line",
                f"{volume_percent}%",
            ]
            cmd_line_dac = [
                "sudo",
                "-u",
                self._amixer_user,
                "amixer",
                "sset",
                "Line DAC",
                f"{volume_percent}%",
            ]
            cmd_line_hp = [
                "sudo",
                "-u",
                self._amixer_user,
                "amixer",
                "sset",
                "HP",
                f"{volume_percent}%",
            ]

            for cmd in [cmd_line, cmd_line_dac, cmd_line_hp]:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                await proc.communicate()

        except Exception as e:
            logger.warning(f"Failed to set system volume: {e}")

    # ==================== 记忆系统方法 ====================

    def _build_dynamic_instructions(self) -> str:
        """构建带记忆上下文和近期摘要的动态 system prompt"""
        base = self._INSTRUCTIONS

        if not self._memory_initialized or not self._memory_store:
            return base

        sections: list[str] = []

        # --- 长期记忆（按 importance 排序，400 token 预算）---
        try:
            token_budget = int(os.getenv("LELAMP_MEMORY_TOKEN_BUDGET", "400"))
            memories = self._memory_store.get_active_memories(
                lamp_id=self._lamp_id,
                max_tokens=token_budget,
            )
            if memories:
                lines = ["\n\n# Memory", "You remember the following about this user:"]
                for m in memories:
                    lines.append(f"- [{m.category}] {m.content}")
                lines.append(
                    "\nUse these memories naturally in conversation. "
                    "Use save_memory() to remember new important things."
                )
                sections.append("\n".join(lines))
        except Exception as e:
            logger.warning(f"Failed to load memories for prompt: {e}")

        # --- 近期对话摘要（最近 72h，最多 2 条，200 token 预算）---
        try:
            summary_token_budget = int(
                os.getenv("LELAMP_MEMORY_SUMMARY_TOKEN_BUDGET", "200")
            )
            summaries = self._memory_store.get_recent_summaries(
                lamp_id=self._lamp_id,
                hours=72,
                limit=2,
            )
            if summaries:
                used = 0
                lines = [
                    "\n\n# Recent Conversations",
                    "Summary of recent sessions with this user:",
                ]
                from lelamp.memory.store import _CHARS_PER_TOKEN

                for s in summaries:
                    entry = f"- {s.summary}"
                    est = len(entry) / _CHARS_PER_TOKEN + 5
                    if used + est > summary_token_budget:
                        break
                    lines.append(entry)
                    used += est
                if len(lines) > 2:  # 有实际摘要内容（超过标题行）
                    sections.append("\n".join(lines))
        except Exception as e:
            logger.warning(f"Failed to load summaries for prompt: {e}")

        if not sections:
            return base

        return base + "".join(sections)

    async def _check_consolidation(self) -> None:
        """检查是否需要触发记忆整合"""
        if (
            not self._memory_initialized
            or not self._memory_consolidator
            or not self._session_id
        ):
            return

        new_turns = self._conversation_turns[self._consolidation_offset :]
        if not new_turns:
            return

        should = self._memory_consolidator.should_consolidate(
            new_turns
        ) or self._memory_consolidator.should_consolidate_by_tokens(new_turns)
        if should:
            self._track_task(self._run_consolidation())

    async def _run_consolidation(self) -> None:
        """后台执行记忆整合（非阻塞）"""
        if self._consolidation_in_progress:
            return
        self._consolidation_in_progress = True
        try:
            # 只整合 offset 之后的新轮次
            new_turns = list(self._conversation_turns[self._consolidation_offset :])
            if not new_turns:
                return

            result = await self._memory_consolidator.consolidate(
                lamp_id=self._lamp_id,
                session_id=self._session_id,
                conversation_turns=new_turns,
            )
            if result:
                # 整合成功：推进偏移量到当前总轮数
                self._consolidation_offset = len(self._conversation_turns)
                if result.new_memories_count > 0:
                    new_instructions = self._build_dynamic_instructions()
                    await self.update_instructions(new_instructions)
                    logger.info(
                        f"Memory consolidated: {result.new_memories_count} new memories"
                    )
        except Exception as e:
            logger.warning(f"Background consolidation failed (non-critical): {e}")
        finally:
            self._consolidation_in_progress = False

    # ==================== 电机工具方法 ====================

    @function_tool()
    async def play_recording(
        self,
        context: RunContext,
        recording_name: str,
    ) -> str:
        """Express yourself through physical movement! Use this only when user explicitly asks for it."""
        return await self._motor_tools.play_recording(context, recording_name)

    @function_tool()
    async def move_joint(
        self,
        context: RunContext,
        joint_name: str,
        angle: float,
    ) -> str:
        """控制指定关节移动到目标角度。可用关节：base_yaw（底座水平旋转）、base_pitch（底座俯仰）、elbow_pitch（肘部俯仰）、wrist_roll（腕部滚转）、wrist_pitch（灯头俯仰）。角度单位为度。"""
        return await self._motor_tools.move_joint(context, joint_name, angle)

    @function_tool()
    async def get_joint_positions(
        self,
        context: RunContext,
    ) -> str:
        """获取所有关节的当前位置（角度）。用于了解台灯当前的姿态。"""
        return await self._motor_tools.get_joint_positions(context)

    @function_tool()
    async def get_motor_health(
        self,
        context: RunContext,
        motor_name: Optional[str] = None,
    ) -> str:
        """
        获取舵机健康状态(温度、电压、负载等)。
        Get motor health status (temperature, voltage, load, etc.).

        Args:
            motor_name: 舵机名称(base_yaw/base_pitch/elbow_pitch/wrist_roll/wrist_pitch),留空则返回所有舵机
        """
        return await self._system_tools.get_motor_health(context, motor_name)

    @function_tool()
    async def tune_motor_pid(
        self,
        context: RunContext,
        motor_name: str,
        p_coefficient: int,
        i_coefficient: int = 0,
        d_coefficient: int = 32,
    ) -> str:
        """
        远程调整舵机 PID 参数(商用功能,用于优化动作性能)。
        Tune motor PID coefficients remotely (commercial feature for performance optimization).

        Args:
            motor_name: 舵机名称(base_yaw/base_pitch/elbow_pitch/wrist_roll/wrist_pitch)
            p_coefficient: P 系数(比例增益,默认 16,范围 1-32,越大响应越快但可能抖动)
            i_coefficient: I 系数(积分增益,默认 0,范围 0-32)
            d_coefficient: D 系数(微分增益,默认 32,范围 0-32,用于减少超调)

        注意: 不当的 PID 参数可能导致舵机抖动或无法稳定,请谨慎调整!
        """
        return await self._system_tools.tune_motor_pid(
            context, motor_name, p_coefficient, i_coefficient, d_coefficient
        )

    @function_tool()
    async def reset_motor_health_stats(
        self,
        context: RunContext,
        motor_name: Optional[str] = None,
    ) -> str:
        """
        重置舵机健康统计数据(警告/危险/堵转计数)。
        Reset motor health statistics (warning/critical/stall counts).

        Args:
            motor_name: 舵机名称,留空则重置所有舵机
        """
        return await self._system_tools.reset_motor_health_stats(context, motor_name)

    @function_tool()
    async def get_available_recordings(
        self,
        context: RunContext,
    ) -> str:
        """Discover your physical expressions! Get your repertoire of motor movements for body language."""
        return await self._system_tools.get_available_recordings(context)

    # ==================== RGB 工具方法 ====================

    @function_tool()
    async def set_rgb_solid(
        self,
        context: RunContext,
        red: int,
        green: int,
        blue: int,
    ) -> str:
        """Express emotions and moods through solid lamp colors!"""
        return await self._rgb_tools.set_rgb_solid(context, red, green, blue)

    @function_tool()
    async def paint_rgb_pattern(
        self,
        context: RunContext,
        pattern: str,
    ) -> str:
        """Create dynamic visual patterns and animations with your lamp!"""
        return await self._rgb_tools.paint_rgb_pattern(context, pattern)

    @function_tool()
    async def set_rgb_brightness(
        self,
        context: RunContext,
        percent: int,
    ) -> str:
        """调节灯光亮度（0-100）"""
        return await self._system_tools.set_rgb_brightness(context, percent)

    @function_tool()
    async def rgb_effect_rainbow(
        self,
        context: RunContext,
        speed: float = 1.0,
        saturation: float = 1.0,
        value: float = 1.0,
        fps: int = 30,
    ) -> str:
        """彩虹动态效果（8x8 矩阵）"""
        # 使用 RGBTools 的 rainbow 效果
        from lelamp.service import Priority

        # 设置灯光覆盖
        self._state_manager.set_light_override(duration_s=10.0)

        self.rgb_service.dispatch(
            "effect",
            {
                "name": "rainbow",
                "speed": float(speed),
                "saturation": float(saturation),
                "value": float(value),
                "fps": int(fps),
            },
            priority=Priority.HIGH,
        )
        return "已开启彩虹动态灯效"

    @function_tool()
    async def rgb_effect_breathing(
        self,
        context: RunContext,
        r: int = 0,
        g: int = 150,
        b: int = 255,
    ) -> str:
        """启动呼吸效果"""
        return await self._rgb_tools.rgb_effect_breathing(context, r, g, b)

    @function_tool()
    async def rgb_effect_wave(
        self,
        context: RunContext,
        red: int = 60,
        green: int = 180,
        blue: int = 255,
        speed: float = 1.0,
        freq: float = 1.2,
        fps: int = 30,
    ) -> str:
        """波纹/呼吸波动效果（8x8 矩阵）"""
        return await self._system_tools.rgb_effect_wave(
            context, red, green, blue, speed, freq, fps
        )

    @function_tool()
    async def rgb_effect_fire(
        self,
        context: RunContext,
        intensity: float = 1.0,
        fps: int = 30,
    ) -> str:
        """火焰动态效果（8x8 矩阵）"""
        return await self._system_tools.rgb_effect_fire(context, intensity, fps)

    @function_tool()
    async def rgb_effect_emoji(
        self,
        context: RunContext,
        emoji: str = "smile",
        red: int = 255,
        green: int = 200,
        blue: int = 60,
        bg_red: int = 0,
        bg_green: int = 0,
        bg_blue: int = 0,
        blink: bool = True,
        period_s: float = 2.2,
        fps: int = 30,
    ) -> str:
        """表情动画（smile/sad/wink/angry/heart）"""
        return await self._system_tools.rgb_effect_emoji(
            context,
            emoji,
            red,
            green,
            blue,
            bg_red,
            bg_green,
            bg_blue,
            blink,
            period_s,
            fps,
        )

    @function_tool()
    async def stop_rgb_effect(
        self,
        context: RunContext,
    ) -> str:
        """停止动态特效/表情动画"""
        return await self._system_tools.stop_rgb_effect(context)

    # ==================== 视觉工具方法 ====================

    @function_tool()
    async def vision_answer(
        self,
        context: RunContext,
        question: str,
    ) -> str:
        """Ask a question about what the lamp can see through its camera."""
        return await self._tool_with_timeout(
            self._vision_tools.vision_answer(context, question),
            timeout_seconds=30.0,
            error_message="视觉识别超时，请稍后重试",
        )

    @function_tool()
    async def check_homework(
        self,
        context: RunContext,
    ) -> str:
        """
        帮用户检查画面中的作业（数学、口算、填空等）。
        Analyze and check homework in the camera view (math, corrections, etc.).
        """
        return await self._tool_with_timeout(
            self._vision_tools.check_homework(context),
            timeout_seconds=45.0,
            error_message="作业检查超时，请稍后重试",
        )

    @function_tool()
    async def capture_to_feishu(
        self,
        context: RunContext,
    ) -> str:
        """拍照并通过飞书机器人推送（直接上传图片），拍照前会锁定动作并停止以确保清晰度。"""
        return await self._vision_tools.capture_to_feishu(context)

    # ==================== 边缘视觉工具方法 ====================

    @function_tool()
    async def quick_identify(
        self,
        context: RunContext,
    ) -> str:
        """
        快速识别当前画面中的物体（本地推理，低延迟）。
        Quick identify objects in the current view (local inference, low latency).

        适用于简单问题如"这是什么"，无需调用云端 API。
        """
        if self._edge_vision_tools is None:
            # 降级到云端视觉
            return await self.vision_answer(context, "这是什么")

        # 获取当前帧
        frame = None
        if self._vision_service:
            frame = self._vision_service.get_latest_frame()

        return await self._edge_vision_tools.quick_identify(frame)

    @function_tool()
    async def detect_gesture(
        self,
        context: RunContext,
    ) -> str:
        """
        检测当前画面中的手势（本地推理，带LED和动作反馈）。
        Detect hand gestures in the current view (local inference, with LED and motion feedback).

        支持的手势：👍 点赞、👎 踩、✌️ 耶、👋 挥手、✊ 握拳、👆 指向

        检测到手势时会自动触发对应的LED效果和动作响应。
        """
        if self._edge_vision_tools is None:
            return "手势检测服务未启用。请设置环境变量 LELAMP_EDGE_VISION_ENABLED=1"

        # 检测前LED闪烁蓝色提示
        self.rgb_service.dispatch("solid", (0, 140, 255), priority=Priority.HIGH)

        frame = None
        if self._vision_service:
            frame = self._vision_service.get_latest_frame()

        result = await self._edge_vision_tools.detect_gesture(frame)

        # 恢复正常灯光
        self.rgb_service.dispatch("solid", (255, 255, 255), priority=Priority.HIGH)

        return result

    @function_tool()
    async def check_presence(
        self,
        context: RunContext,
    ) -> str:
        """
        检测用户是否在场（本地推理）。
        Check if user is present (local inference).

        用于自动唤醒/休眠功能。
        """
        if self._edge_vision_tools is None:
            return "在场检测服务未启用。请设置环境变量 LELAMP_EDGE_VISION_ENABLED=1"

        frame = None
        if self._vision_service:
            frame = self._vision_service.get_latest_frame()

        return await self._edge_vision_tools.check_presence(frame)

    @function_tool()
    async def get_edge_vision_stats(
        self,
        context: RunContext,
    ) -> str:
        """
        获取边缘视觉服务统计信息（调试用）。
        Get edge vision service statistics (for debugging).
        """
        if self._edge_vision_tools is None:
            return "边缘视觉服务未启用"

        stats = self._edge_vision_tools.get_stats()

        lines = ["边缘视觉服务统计:"]
        lines.append(f"- 总查询数: {stats.get('total_queries', 0)}")
        lines.append(f"- 本地查询: {stats.get('local_queries', 0)}")
        lines.append(f"- 云端查询: {stats.get('cloud_queries', 0)}")
        lines.append(f"- 混合查询: {stats.get('hybrid_queries', 0)}")

        services = stats.get("services", {})
        lines.append("- 服务状态:")
        lines.append(
            f"  - 人脸检测: {'启用' if services.get('face_detector') else '禁用'}"
        )
        lines.append(
            f"  - 手势追踪: {'启用' if services.get('hand_tracker') else '禁用'}"
        )
        lines.append(
            f"  - 物体检测: {'启用' if services.get('object_detector') else '禁用'}"
        )
        lines.append(
            f"  - 云端视觉: {'启用' if services.get('cloud_vision') else '禁用'}"
        )

        return "\n".join(lines)

    @function_tool()
    async def quick_check(
        self,
        context: RunContext,
    ) -> str:
        """
        快速检查 - 同时检测用户在场和手势（本地推理）。
        Quick check - detect user presence and gestures simultaneously (local inference).

        语音触发示例：说"检查一下"、"看看怎么样了"、"扫描一下"
        """
        if self._edge_vision_tools is None:
            return "边缘视觉服务未启用。请设置环境变量 LELAMP_EDGE_VISION_ENABLED=1"

        frame = None
        if self._vision_service:
            frame = self._vision_service.get_latest_frame()

        return await self._edge_vision_tools.quick_check(frame)

    @function_tool()
    async def get_vision_monitor_status(
        self,
        context: RunContext,
    ) -> str:
        """
        获取主动监听服务状态（调试用）。
        Get proactive vision monitor status (for debugging).
        """
        if self._vision_monitor is None:
            return "主动监听服务未启用。边缘视觉服务正常运行，可通过语音命令触发检测。"

        stats = self._vision_monitor.get_stats()
        return f"""主动监听服务状态：
- 运行中: {stats["running"]}
- 模式: {stats["mode"]}
- 用户在场: {stats["user_present"]}
- 在场时长: {stats["user_present_duration"]:.1f}秒
- 检测次数: {stats["detection_count"]}
- 手势次数: {stats["gesture_count"]}
- 手势检测: {"启用" if stats["auto_gesture_enabled"] else "禁用"}
- 在场检测: {"启用" if stats["auto_presence_enabled"] else "禁用"}"""

    @function_tool()
    async def toggle_vision_monitor(
        self,
        context: RunContext,
        enable: bool = None,
    ) -> str:
        """
        启用或禁用主动监听服务。
        Toggle proactive vision monitoring service.

        Args:
            enable: True 启用，False 禁用，None 切换状态
        """
        if self._vision_monitor is None:
            return "主动监听服务未初始化。请检查边缘视觉服务是否正常启动。"

        current_running = self._vision_monitor._running

        if enable is None:
            # 切换状态
            enable = not current_running

        if enable and not current_running:
            self._vision_monitor.start()
            return "主动监听服务已启用。现在会自动检测用户在场和手势。"
        elif not enable and current_running:
            self._vision_monitor.stop()
            return "主动监听服务已禁用。现在只能通过语音命令触发视觉检测。"
        elif enable and current_running:
            return "主动监听服务已经在运行中。"
        else:
            return "主动监听服务已经停止。"

    @function_tool()
    async def set_vision_monitor_mode(
        self,
        context: RunContext,
        mode: str,
    ) -> str:
        """
        设置主动监听模式。
        Set proactive vision monitoring mode.

        Args:
            mode: 监听模式 - "active"(主动), "idle"(空闲), "sleep"(休眠)
        """
        if self._vision_monitor is None:
            return f"主动监听服务未初始化，无法设置模式为 {mode}。"

        mode_lower = mode.strip().lower()
        valid_modes = ["active", "idle", "sleep"]

        if mode_lower not in valid_modes:
            return f"无效的模式: {mode}。有效模式为: {', '.join(valid_modes)}"

        self._vision_monitor.set_mode(mode_lower)
        mode_name = {"active": "主动", "idle": "空闲", "sleep": "休眠"}[mode_lower]
        return f"主动监听模式已设置为: {mode_name}"

    # ==================== 系统工具方法 ====================

    @function_tool()
    async def set_volume(
        self,
        context: RunContext,
        volume_percent: int,
    ) -> str:
        """Control system audio volume."""
        return await self._system_tools.set_volume(context, volume_percent)

    @function_tool()
    async def get_rate_limit_stats(
        self,
        context: RunContext,
    ) -> str:
        """获取 API 速率限制统计信息（调试用）"""
        return await self._system_tools.get_rate_limit_stats(context)

    @function_tool()
    async def web_search(
        self,
        context: RunContext,
        query: str,
    ) -> str:
        """
        当用户问到实时信息、新闻、天气或你不确定的知识时，使用此工具在线搜索。
        Get real-time information from the web.

        Args:
            query: 搜索关键词 (Search query)
        """
        # 应用速率限制
        if not await self._search_rate_limiter.acquire(tokens=1, timeout=5.0):
            return "搜索太频繁了，请稍后再试。本灯也要休息一下的。"

        return await self._tool_with_timeout(
            self._system_tools.web_search(context, query),
            timeout_seconds=15.0,
            error_message="搜索超时，请稍后重试",
        )

    @function_tool()
    async def check_for_updates(
        self,
        context: RunContext,
    ) -> str:
        """
        检查系统是否有新的 OTA 更新。
        Check for system updates.
        """
        return await self._system_tools.check_for_updates(context)

    @function_tool()
    async def perform_ota_update(
        self,
        context: RunContext,
    ) -> str:
        """
        执行系统更新 (OTA)。注意：更新成功后服务将重启。
        Perform system update. Note: Service will restart upon success.
        """
        result = await self._system_tools.perform_ota_update(context)
        if "更新成功" in result or "服务将在" in result:
            # 创建重启任务（使用 track_task 确保任务被追踪）
            self._track_task(self._restart_later())
        return result

    # ==================== 记忆工具方法 ====================

    @function_tool()
    async def save_memory(
        self,
        context: RunContext,
        content: str,
        category: str = "general",
    ) -> str:
        """
        记住一个重要信息（用户偏好、事实、上下文等）。
        Remember an important piece of information (user preference, fact, context, etc.).

        Args:
            content: 要记住的内容 (max 500 chars)
            category: 分类 - preference(偏好)/fact(事实)/relationship(关系)/context(上下文)/general(通用)
        """
        if self._memory_tools is None:
            return "记忆功能未启用"
        return await self._memory_tools.save_memory(content, category)

    @function_tool()
    async def recall_memory(
        self,
        context: RunContext,
        query: str,
    ) -> str:
        """
        搜索你的记忆，查找相关信息。
        Search your memory for relevant information about past conversations or user preferences.

        Args:
            query: 搜索关键词
        """
        if self._memory_tools is None:
            return "记忆功能未启用"
        return await self._memory_tools.recall_memory(query)

    @function_tool()
    async def forget_memory(
        self,
        context: RunContext,
        content_hint: str,
    ) -> str:
        """
        删除一条记忆（当信息过时或不再相关时使用）。
        Delete a memory (use when information is outdated or no longer relevant).

        Args:
            content_hint: 要删除的记忆的关键词
        """
        if self._memory_tools is None:
            return "记忆功能未启用"
        return await self._memory_tools.forget_memory(content_hint)

    async def _restart_later(self) -> None:
        """延迟重启服务"""
        try:
            await asyncio.sleep(5)
            logger.info("Triggering restart...")
            # 依赖 systemd 或 Docker 重启容器/进程
            sys.exit(0)
        except Exception as e:
            logger.error(f"Error during restart delay: {e}")

    # ==================== Data Channel 消息处理 (Web Client v2.0) ====================

    async def handle_data_message(self, data: bytes, participant) -> None:
        """
        处理来自 Web Client 的 Data Channel 消息

        支持的消息类型:
        1. chat: 文字聊天消息
        2. command: 控制指令 (动作、灯光等)

        Args:
            data: 消息数据 (bytes)
            participant: 发送者信息
        """
        try:
            message_str = data.decode("utf-8")
            logger.debug(f"收到 Data Channel 消息: {message_str}")
            message = json.loads(message_str)

            msg_type = message.get("type")

            if msg_type == "chat":
                # 聊天消息 - 转换为语音输入
                content = message.get("content", "")
                if content:
                    await self.note_user_text(content)
                    logger.info(f"收到文字消息: {content}")

            elif msg_type == "command":
                # 控制指令 - 路由到对应的功能
                action = message.get("action")
                params = message.get("params", {})
                logger.info(f"执行指令: {action}, 参数: {params}")

                result = await self._execute_command(action, params)

                # 发送执行结果
                if result:
                    await self._send_chat_message(result)

            else:
                logger.warning(f"未知消息类型: {msg_type}")

        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析失败: {e}")
        except Exception as e:
            logger.error(f"处理 Data Channel 消息失败: {e}")
            await self._send_chat_message(f"指令执行失败: {str(e)}")

    async def _execute_command(self, action: str, params: dict) -> str:
        """
        执行 Web Client 发送的控制指令

        支持的指令:
        - play_recording: 播放录制动画
        - move_joint: 移动关节
        - set_rgb_solid: 设置纯色灯光
        - rgb_effect_*: 启动灯效
        - stop_rgb_effect: 停止灯效
        - vision_answer: 视觉问答
        - check_homework: 检查作业

        Args:
            action: 指令名称
            params: 指令参数

        Returns:
            执行结果描述
        """
        # 创建轻量 context 用于直接调用（Data Channel 场景）
        mock_context = _DataContext()

        try:
            # 播放录制动画
            if action == "play_recording":
                recording_name = params.get("recording_name")
                if recording_name:
                    return await self.play_recording(mock_context, recording_name)
                return "缺少录制名称参数"

            # 移动关节
            elif action == "move_joint":
                joint_name = params.get("joint_name")
                angle = params.get("angle")
                if joint_name and angle is not None:
                    return await self.move_joint(mock_context, joint_name, float(angle))
                return "缺少关节名称或角度参数"

            # 设置纯色灯光
            elif action == "set_rgb_solid":
                r = params.get("r")
                g = params.get("g")
                b = params.get("b")
                if r is not None and g is not None and b is not None:
                    return await self.set_rgb_solid(
                        mock_context, int(r), int(g), int(b)
                    )
                return "缺少 RGB 参数"

            # 停止灯效
            elif action == "stop_rgb_effect":
                return await self.stop_rgb_effect(mock_context)

            # 灯效动画
            elif action.startswith("rgb_effect_"):
                effect_name = action.replace("rgb_effect_", "")
                return await self._execute_rgb_effect(mock_context, effect_name)

            # 视觉功能
            elif action == "vision_answer":
                question = params.get("question", "这是什么")
                result = await self.vision_answer(mock_context, question)
                # 发送视觉结果 (包含图片)
                await self._send_vision_result(result)
                return result

            elif action == "check_homework":
                result = await self.check_homework(mock_context)
                await self._send_vision_result(result)
                return result

            else:
                return f"未知指令: {action}"

        except Exception as e:
            logger.error(f"执行指令失败: {action}, 错误: {e}")
            return f"执行失败: {str(e)}"

    async def _execute_rgb_effect(self, context: RunContext, effect_name: str) -> str:
        """
        执行 RGB 灯效

        Args:
            context: RunContext
            effect_name: 效果名称

        Returns:
            执行结果描述
        """
        effect_map = {
            "breathing": lambda: self.rgb_effect_breathing(context, 0, 150, 255),
            "rainbow": lambda: self.rgb_effect_rainbow(context),
            "wave": lambda: self.rgb_effect_wave(context),
            "fire": lambda: self.rgb_effect_fire(context),
            "emoji": lambda: self.rgb_effect_emoji(context),
        }

        effect_func = effect_map.get(effect_name)
        if effect_func:
            return await effect_func()
        return f"未知灯效: {effect_name}"

    async def _send_chat_message(self, content: str) -> None:
        """
        向 Web Client 发送聊天消息

        Args:
            content: 消息内容
        """
        try:
            if hasattr(self, "send_message_callback") and self.send_message_callback:
                message = {"type": "chat", "content": content, "timestamp": time.time()}
                await self.send_message_callback(message)
                logger.debug(f"发送聊天消息: {content}")
        except Exception as e:
            logger.error(f"发送聊天消息失败: {e}")

    async def _send_vision_result(self, result: str, image_base64: str = None) -> None:
        """
        向 Web Client 发送视觉结果 (包含图片)

        Args:
            result: 视觉分析结果
            image_base64: 图片 base64 编码（可选）
        """
        try:
            if hasattr(self, "send_message_callback") and self.send_message_callback:
                # 如果没有提供图片，尝试从 vision_service 获取最新帧
                if image_base64 is None and self._vision_service:
                    frame_data = self._vision_service.get_latest_frame()
                    if frame_data:
                        image_base64 = base64.b64encode(frame_data).decode("utf-8")
                # 如果 image_base64 是 bytes，转换为 base64 字符串
                elif isinstance(image_base64, bytes):
                    image_base64 = base64.b64encode(image_base64).decode("utf-8")

                message = {
                    "type": "vision_result",
                    "content": result,
                    "image_base64": image_base64,
                    "timestamp": time.time(),
                }
                await self.send_message_callback(message)
                logger.debug(f"发送视觉结果: {result[:100]}...")
        except Exception as e:
            logger.error(f"发送视觉结果失败: {e}")

    async def _update_camera_status(self, active: bool) -> None:
        """
        向 Web Client 更新摄像头状态

        Args:
            active: 摄像头是否激活
        """
        try:
            if hasattr(self, "send_message_callback") and self.send_message_callback:
                message = {
                    "type": "camera_status",
                    "active": active,
                    "timestamp": time.time(),
                }
                await self.send_message_callback(message)
                logger.debug(f"更新摄像头状态: {active}")
        except Exception as e:
            logger.error(f"更新摄像头状态失败: {e}")

    def shutdown(self) -> None:
        """
        关闭 agent，清理资源

        停止主动监听服务等后台线程
        """
        logger.info("LeLamp agent shutting down...")

        # 停止主动监听服务
        if self._vision_monitor is not None and self._vision_monitor._running:
            logger.info("Stopping proactive vision monitor...")
            self._vision_monitor.stop()

        # 取消所有追踪的后台任务
        for task in self._background_tasks:
            if not task.done():
                logger.debug(f"Cancelling background task: {task}")
                task.cancel()

        # 记录未整合的对话轮数
        if self._memory_initialized and self._conversation_turns:
            logger.info(
                f"Session ending with {len(self._conversation_turns) - self._consolidation_offset} unconsolidated turns "
                f"(will be consolidated next session if memory persists)"
            )

        logger.info("LeLamp agent shutdown complete")
