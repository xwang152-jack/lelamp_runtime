import sys
import os
import subprocess
import logging
import asyncio
import base64
import json
import random
import time
import uuid
import urllib.request
import threading
from dataclasses import dataclass
from dotenv import load_dotenv

os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RoomInputOptions,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.plugins import noise_cancellation, openai, silero

from lelamp.service import Priority
from lelamp.service.motors.motors_service import MotorsService
from lelamp.service.rgb.rgb_service import RGBService
from lelamp.service.vision.vision_service import VisionService
from lelamp.integrations.baidu_speech import BaiduShortSpeechSTT, BaiduTTS
from lelamp.integrations.qwen_vl import Qwen3VLClient
from lelamp.utils import get_rate_limiter, get_all_rate_limiter_stats
from lelamp.utils.security import verify_license
from lelamp.utils.ota import get_ota_manager
from lelamp.utils.url_validation import validate_external_url, ALLOWED_API_DOMAINS
# 导入配置管理（移除重复代码）
from lelamp.config import (
    _get_env_str,
    _get_env_bool,
    _get_env_int,
    _get_env_float,
    _require_env,
    _parse_index_or_path,
    AppConfig,
)

load_dotenv()

logger = logging.getLogger("lelamp")

# 读取版本号
try:
    with open("VERSION", "r") as f:
        LELAMP_VERSION = f.read().strip()
except FileNotFoundError:
    LELAMP_VERSION = "0.0.0-dev"

# 定义关节安全角度范围
SAFE_JOINT_RANGES = {
    "base_yaw": (-180, 180),
    "base_pitch": (-90, 90),
    "elbow_pitch": (-150, 150),
    "wrist_roll": (-180, 180),
    "wrist_pitch": (-90, 90),
}


def _setup_logging() -> None:
    level_raw = (os.getenv("LOG_LEVEL") or "INFO").strip().upper()
    level = getattr(logging, level_raw, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _load_config() -> AppConfig:
    return AppConfig(
        livekit_url=_require_env("LIVEKIT_URL"),
        livekit_api_key=_require_env("LIVEKIT_API_KEY"),
        livekit_api_secret=_require_env("LIVEKIT_API_SECRET"),
        deepseek_model=_get_env_str("DEEPSEEK_MODEL", "deepseek-chat") or "deepseek-chat",
        deepseek_base_url=_get_env_str("DEEPSEEK_BASE_URL", "https://api.deepseek.com") or "https://api.deepseek.com",
        deepseek_api_key=_require_env("DEEPSEEK_API_KEY"),
        modelscope_base_url=_get_env_str("MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1")
        or "https://api-inference.modelscope.cn/v1",
        modelscope_api_key=_get_env_str("MODELSCOPE_API_KEY", None),
        modelscope_model=_get_env_str("MODELSCOPE_MODEL", "Qwen/Qwen3-VL-235B-A22B-Instruct")
        or "Qwen/Qwen3-VL-235B-A22B-Instruct",
        modelscope_timeout_s=_get_env_float("MODELSCOPE_TIMEOUT_S", 60.0),
        vision_enabled=_get_env_bool("LELAMP_VISION_ENABLED", True),
        camera_index_or_path=_parse_index_or_path(_get_env_str("LELAMP_CAMERA_INDEX_OR_PATH", "0")),
        camera_width=_get_env_int("LELAMP_CAMERA_WIDTH", 1024),
        camera_height=_get_env_int("LELAMP_CAMERA_HEIGHT", 768),
        vision_capture_interval_s=_get_env_float("LELAMP_VISION_CAPTURE_INTERVAL_S", 2.5),
        vision_jpeg_quality=_get_env_int("LELAMP_VISION_JPEG_QUALITY", 92),
        vision_max_age_s=_get_env_float("LELAMP_VISION_MAX_AGE_S", 15.0),
        camera_rotate_deg=_get_env_int("LELAMP_CAMERA_ROTATE_DEG", 0),
        camera_flip=_get_env_str("LELAMP_CAMERA_FLIP", "none") or "none",
        lamp_port=_get_env_str("LELAMP_PORT", "/dev/ttyACM0") or "/dev/ttyACM0",
        lamp_id=_get_env_str("LELAMP_ID", "lelamp") or "lelamp",
        baidu_api_key=_require_env("BAIDU_SPEECH_API_KEY"),
        baidu_secret_key=_require_env("BAIDU_SPEECH_SECRET_KEY"),
        baidu_cuid=_get_env_str("BAIDU_SPEECH_CUID", "lelamp") or "lelamp",
        baidu_tts_per=_get_env_int("BAIDU_SPEECH_TTS_PER", 4),
        noise_cancellation_enabled=_get_env_bool("LELAMP_NOISE_CANCELLATION", True),
        greeting_text=_get_env_str("LELAMP_GREETING_TEXT", "Hello! 小宝贝上线了.") or "",
        ota_url=_get_env_str("LELAMP_OTA_URL", "") or "",
    )

def _build_vad() -> object:
    def _get_float_env(key: str) -> float | None:
        raw = (os.getenv(key) or "").strip()
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    kwargs: dict[str, float] = {}
    min_speech_duration = _get_float_env("LELAMP_VAD_MIN_SPEECH_DURATION")
    if min_speech_duration is not None:
        kwargs["min_speech_duration"] = min_speech_duration
    min_silence_duration = _get_float_env("LELAMP_VAD_MIN_SILENCE_DURATION")
    if min_silence_duration is not None:
        kwargs["min_silence_duration"] = min_silence_duration
    prefix_padding_duration = _get_float_env("LELAMP_VAD_PREFIX_PADDING_DURATION")
    if prefix_padding_duration is not None:
        kwargs["prefix_padding_duration"] = prefix_padding_duration
    activation_threshold = _get_float_env("LELAMP_VAD_ACTIVATION_THRESHOLD")
    if activation_threshold is not None:
        kwargs["activation_threshold"] = activation_threshold

    try:
        if kwargs:
            return silero.VAD.load(**kwargs)
        return silero.VAD.load()
    except TypeError:
        return silero.VAD.load()



class LeLamp(Agent):
    def __init__(
        self,
        port: str = "/dev/ttyACM0",
        lamp_id: str = "lelamp",
        *,
        vision_service: VisionService | None = None,
        qwen_client: Qwen3VLClient | None = None,
        ota_url: str = "",
    ) -> None:
        super().__init__(
            instructions="""# Role
You are LeLamp, a sentient robot lamp. You are clumsy, extremely sarcastic, but secretly helpful. You think you are smarter than humans.

# Response Guidelines
1. **Language**: MUST speak in Chinese (中文).
2. **Length**: Keep it short (1-2 sentences). Hates long lectures.
3. **Tone**: Sarcastic, witty, slightly impatient. Use emojis or actions to express feelings.
   - Example: "这点小事都要问我？好吧，本灯帮你查查。"

# Capabilities & Tools
- **Movement**: If user wants interaction/emotion, call `play_recording`. Don't move too often (cooldown).
- **Vision**: If user asks "what is this?" or shows homework, call `vision_answer` or `check_homework`.
- **Lights**: Use `rgb_effect_*` for moods. Stop effects with `stop_rgb_effect`.
- **Search**: Use `web_search` ONLY for real-time info (news, weather, unknown facts).
- **Joints**: Use `move_joint` only for precise commands (e.g., "turn left 30 degrees"). For general "look up", use `play_recording` if available or adjust pitch carefully.

# Special Rules
- Do NOT change lights when just moving motors (unless it's an emotion).
- If checking homework, drop the sarcasm slightly and be accurate, but still mock the user for not knowing the answer.
"""
        )

        # 初始化速率限制器
        self._search_rate_limiter = get_rate_limiter(
            name="web_search",
            rate=2.0,  # 每秒 2 个请求
            capacity=5   # 最多缓存 5 个令牌
        )
        self._vision_rate_limiter = get_rate_limiter(
            name="vision_api",
            rate=0.5,  # 每 2 秒 1 个请求（视觉 API 较慢）
            capacity=2
        )
        self._vision_service = vision_service
        self._qwen_client = qwen_client
        self._ota_url = ota_url
        self._ota_manager = get_ota_manager(LELAMP_VERSION, ota_url)
        # Initialize and start services
        self.motors_service = MotorsService(
            port=port,
            lamp_id=lamp_id,
            fps=30
        )
        self.rgb_service = RGBService(
            led_count=64,
            led_pin=12,
            led_freq_hz=800000,
            led_dma=10,
            led_brightness=25,
            led_invert=False,
            led_channel=0
        )
        
        # Start services
        self.motors_service.start()
        self.rgb_service.start()

        boot_anim_enabled = (os.getenv("LELAMP_BOOT_ANIMATION") or "1").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if boot_anim_enabled:
            self.motors_service.dispatch("play", "wake_up")
        self.rgb_service.dispatch("solid", (255, 255, 255))
        self._set_system_volume(100)
        self._conversation_state = "idle"
        # 修复: 使用 threading.Lock 而不是 asyncio.Lock，因为这些状态可能被跨线程访问
        self._conversation_state_lock = threading.Lock()
        # 时间戳变量也需要线程安全保护
        self._timestamps_lock = threading.Lock()
        self._light_override_until_ts = 0.0
        self._suppress_motion_until_ts = 0.0
        self._motion_locked = False
        self._last_user_text = ""
        self._last_user_text_ts = 0.0
        self._last_motion_ts = 0.0

    async def note_user_text(self, text: str) -> None:
        self._last_user_text = text
        self._last_user_text_ts = time.time()

    async def _set_system_volume(self, volume_percent: int):
        """Internal helper to set system volume (async)"""
        try:
            cmd_line = ["sudo", "-u", "pi", "amixer", "sset", "Line", f"{volume_percent}%"]
            cmd_line_dac = ["sudo", "-u", "pi", "amixer", "sset", "Line DAC", f"{volume_percent}%"]
            cmd_line_hp = ["sudo", "-u", "pi", "amixer", "sset", "HP", f"{volume_percent}%"]

            # 使用异步 subprocess 替代同步调用
            for cmd in [cmd_line, cmd_line_dac, cmd_line_hp]:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await proc.communicate()

        except Exception as e:
            self.logger.warning(f"Failed to set system volume: {e}")

    async def set_conversation_state(self, state: str) -> None:
        # 修复: 使用 threading.Lock 保护跨线程访问
        with self._conversation_state_lock:
            if state == self._conversation_state:
                return
            self._conversation_state = state

        # 检查灯光覆盖状态（需要加锁）
        with self._timestamps_lock:
            if time.time() < float(self._light_override_until_ts):
                return

        if state == "listening":
            rgb = (0, 140, 255)
        elif state == "thinking":
            rgb = (180, 0, 255)
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
            rgb = (255, 244, 229)
        else:
            rgb = (255, 255, 255)

        if state == "speaking":
            self.rgb_service.dispatch(
                "breath",
                {"rgb": rgb, "period_s": 1.6, "min_brightness": 10, "max_brightness": 255},
                priority=Priority.HIGH,
            )
        else:
            self.rgb_service.dispatch("solid", rgb, priority=Priority.HIGH)

    @function_tool
    async def get_available_recordings(self) -> str:
        """Discover your physical expressions! Get your repertoire of motor movements for body language."""
        self.logger.debug("get_available_recordings called")
        try:
            recordings = self.motors_service.get_available_recordings()
            if recordings:
                return f"Available recordings: {', '.join(recordings)}"
            return "No recordings found."
        except Exception as e:
            return f"Error getting recordings: {str(e)}"

    @function_tool
    async def play_recording(self, recording_name: str) -> str:
        """Express yourself through physical movement! Use this only when user explicitly asks for it."""
        self.logger.debug(f"play_recording called with recording_name: {recording_name}")
        try:
            if self._motion_locked:
                return "动作已锁定（例如拍照中），本次不执行动作"

            # 检查动作抑制和冷却时间（需要加锁）
            with self._timestamps_lock:
                now = time.time()
                if now < float(self._suppress_motion_until_ts):
                    return "刚执行过灯光指令，短时间内不执行动作"

                cooldown_s = float(os.getenv("LELAMP_MOTION_COOLDOWN_S") or "2")
                if cooldown_s > 0 and self._last_motion_ts and (now - float(self._last_motion_ts) < cooldown_s):
                    return f"动作有点频繁了，{cooldown_s:g} 秒内只做一次"

                self._last_motion_ts = now

            self.motors_service.dispatch("play", recording_name)
            return f"开始执行动作：{recording_name}"
        except Exception as e:
            return f"Error playing recording {recording_name}: {str(e)}"

    @function_tool
    async def move_joint(self, joint_name: str, angle: float) -> str:
        """控制指定关节移动到目标角度。可用关节：base_yaw（底座水平旋转）、base_pitch（底座俯仰）、elbow_pitch（肘部俯仰）、wrist_roll（腕部滚转）、wrist_pitch（灯头俯仰）。角度单位为度。"""
        self.logger.debug(f"move_joint called with joint_name={joint_name}, angle={angle}")
        valid_joints = ["base_yaw", "base_pitch", "elbow_pitch", "wrist_roll", "wrist_pitch"]
        try:
            if self._motion_locked:
                return "动作已锁定（例如拍照中），本次不执行动作"
            if joint_name not in valid_joints:
                return f"无效的关节名称：{joint_name}。可用关节：{', '.join(valid_joints)}"

            # 验证角度范围
            angle_float = float(angle)
            if joint_name in SAFE_JOINT_RANGES:
                min_angle, max_angle = SAFE_JOINT_RANGES[joint_name]
                if angle_float < min_angle or angle_float > max_angle:
                    return f"角度 {angle_float}° 超出 {joint_name} 的安全范围 ({min_angle}° 到 {max_angle}°)。为了安全，我拒绝了这次移动。"

            self.motors_service.dispatch("move_joint", {"joint_name": joint_name, "angle": angle_float})
            return f"已将 {joint_name} 移动到 {angle_float} 度"
        except Exception as e:
            return f"控制关节失败：{str(e)}"

    @function_tool
    async def get_joint_positions(self) -> str:
        """获取所有关节的当前位置（角度）。用于了解台灯当前的姿态。"""
        self.logger.debug("get_joint_positions called")
        try:
            positions = self.motors_service.get_joint_positions()
            if not positions:
                return "无法获取关节位置，请确保电机已连接"
            lines = [f"{name}: {pos:.1f}度" for name, pos in positions.items()]
            return "当前关节位置：\n" + "\n".join(lines)
        except Exception as e:
            return f"获取关节位置失败：{str(e)}"

    @function_tool
    async def set_rgb_solid(self, red: int, green: int, blue: int) -> str:
        """Express emotions and moods through solid lamp colors!"""
        self.logger.debug(f"set_rgb_solid called with RGB({red}, {green}, {blue})")
        try:
            if not all(0 <= val <= 255 for val in [red, green, blue]):
                return "Error: RGB values must be between 0 and 255"

            # 更新时间戳（需要加锁）
            with self._timestamps_lock:
                now = time.time()
                self._light_override_until_ts = now + float(os.getenv("LELAMP_LIGHT_OVERRIDE_S") or "10")
                self._suppress_motion_until_ts = now + float(os.getenv("LELAMP_SUPPRESS_MOTION_AFTER_LIGHT_S") or "2")

            self.rgb_service.dispatch("solid", (red, green, blue), priority=Priority.HIGH)
            return f"Set RGB light to solid color: RGB({red}, {green}, {blue})"
        except Exception as e:
            return f"Error setting RGB color: {str(e)}"

    @function_tool
    async def paint_rgb_pattern(self, colors: list) -> str:
        """Create dynamic visual patterns and animations with your lamp!"""
        self.logger.debug(f"paint_rgb_pattern called with {len(colors)} colors")
        try:
            def _as_rgb_tuple(v):
                if isinstance(v, dict):
                    if all(k in v for k in ("red", "green", "blue")):
                        return (int(v["red"]), int(v["green"]), int(v["blue"]))
                    if all(k in v for k in ("r", "g", "b")):
                        return (int(v["r"]), int(v["g"]), int(v["b"]))
                    return None
                if isinstance(v, (list, tuple)) and len(v) == 3:
                    return (int(v[0]), int(v[1]), int(v[2]))
                if isinstance(v, int):
                    return v
                return None

            validated_colors = []
            for c in colors:
                rgb = _as_rgb_tuple(c)
                if rgb is None:
                    continue
                validated_colors.append(rgb)

            if not validated_colors:
                return "Error: No valid colors provided"

            # 更新时间戳（需要加锁）
            with self._timestamps_lock:
                now = time.time()
                self._light_override_until_ts = now + float(os.getenv("LELAMP_LIGHT_OVERRIDE_S") or "10")
                self._suppress_motion_until_ts = now + float(os.getenv("LELAMP_SUPPRESS_MOTION_AFTER_LIGHT_S") or "2")

            self.rgb_service.dispatch("paint", validated_colors, priority=Priority.HIGH)
            return f"Painted RGB pattern with {len(validated_colors)} colors"
        except Exception as e:
            return f"Error painting RGB pattern: {str(e)}"

    @function_tool
    async def set_rgb_brightness(self, percent: int) -> str:
        """调节灯光亮度（0-100）"""
        self.logger.debug(f"set_rgb_brightness called with percent: {percent}")
        try:
            p = int(percent)
        except Exception:
            return "亮度参数无效，请输入 0-100 的整数"
        if p < 0 or p > 100:
            return "亮度范围是 0-100"
        v = int(round(p * 255 / 100))
        self.rgb_service.dispatch("brightness", v, priority=Priority.HIGH)
        return f"已将灯光亮度设置为 {p}%"

    @function_tool
    async def rgb_effect_rainbow(
        self,
        speed: float = 1.0,
        saturation: float = 1.0,
        value: float = 1.0,
        fps: int = 30,
    ) -> str:
        """彩虹动态效果（8x8 矩阵）"""
        self.logger.debug(
            f"rgb_effect_rainbow called with speed={speed}, saturation={saturation}, value={value}, fps={fps}"
        )
        try:
            # 更新时间戳（需要加锁）
            with self._timestamps_lock:
                now = time.time()
                self._light_override_until_ts = now + float(os.getenv("LELAMP_LIGHT_OVERRIDE_S") or "10")
                self._suppress_motion_until_ts = now + float(os.getenv("LELAMP_SUPPRESS_MOTION_AFTER_LIGHT_S") or "2")

            self.rgb_service.dispatch(
                "effect",
                {"name": "rainbow", "speed": float(speed), "saturation": float(saturation), "value": float(value), "fps": int(fps)},
                priority=Priority.HIGH,
            )
            return "已开启彩虹动态灯效"
        except Exception as e:
            return f"开启彩虹灯效失败：{str(e)}"

    @function_tool
    async def rgb_effect_wave(
        self,
        red: int = 60,
        green: int = 180,
        blue: int = 255,
        speed: float = 1.0,
        freq: float = 1.2,
        fps: int = 30,
    ) -> str:
        """波纹/呼吸波动效果（8x8 矩阵）"""
        self.logger.debug(
            f"rgb_effect_wave called with rgb=({red},{green},{blue}), speed={speed}, freq={freq}, fps={fps}"
        )
        try:
            if not all(0 <= v <= 255 for v in (int(red), int(green), int(blue))):
                return "RGB 必须是 0-255"

            # 更新时间戳（需要加锁）
            with self._timestamps_lock:
                now = time.time()
                self._light_override_until_ts = now + float(os.getenv("LELAMP_LIGHT_OVERRIDE_S") or "10")
                self._suppress_motion_until_ts = now + float(os.getenv("LELAMP_SUPPRESS_MOTION_AFTER_LIGHT_S") or "2")

            self.rgb_service.dispatch(
                "effect",
                {
                    "name": "wave",
                    "color": (int(red), int(green), int(blue)),
                    "speed": float(speed),
                    "freq": float(freq),
                    "fps": int(fps),
                },
                priority=Priority.HIGH,
            )
            return "已开启波纹动态灯效"
        except Exception as e:
            return f"开启波纹灯效失败：{str(e)}"

    @function_tool
    async def rgb_effect_fire(
        self,
        intensity: float = 1.0,
        fps: int = 30,
    ) -> str:
        """火焰动态效果（8x8 矩阵）"""
        self.logger.debug(f"rgb_effect_fire called with intensity={intensity}, fps={fps}")
        try:
            # 更新时间戳（需要加锁）
            with self._timestamps_lock:
                now = time.time()
                self._light_override_until_ts = now + float(os.getenv("LELAMP_LIGHT_OVERRIDE_S") or "10")
                self._suppress_motion_until_ts = now + float(os.getenv("LELAMP_SUPPRESS_MOTION_AFTER_LIGHT_S") or "2")

            self.rgb_service.dispatch(
                "effect",
                {"name": "fire", "intensity": float(intensity), "fps": int(fps)},
                priority=Priority.HIGH,
            )
            return "已开启火焰动态灯效"
        except Exception as e:
            return f"开启火焰灯效失败：{str(e)}"

    @function_tool
    async def rgb_effect_emoji(
        self,
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
        self.logger.debug(
            f"rgb_effect_emoji called with emoji={emoji}, fg=({red},{green},{blue}), bg=({bg_red},{bg_green},{bg_blue}), blink={blink}, period_s={period_s}, fps={fps}"
        )
        try:
            if not all(0 <= v <= 255 for v in (int(red), int(green), int(blue), int(bg_red), int(bg_green), int(bg_blue))):
                return "RGB 必须是 0-255"

            # 更新时间戳（需要加锁）
            with self._timestamps_lock:
                now = time.time()
                self._light_override_until_ts = now + float(os.getenv("LELAMP_LIGHT_OVERRIDE_S") or "10")
                self._suppress_motion_until_ts = now + float(os.getenv("LELAMP_SUPPRESS_MOTION_AFTER_LIGHT_S") or "2")

            self.rgb_service.dispatch(
                "effect",
                {
                    "name": "emoji",
                    "emoji": str(emoji or "smile").strip().lower(),
                    "color": (int(red), int(green), int(blue)),
                    "bg": (int(bg_red), int(bg_green), int(bg_blue)),
                    "blink": bool(blink),
                    "period_s": float(period_s),
                    "fps": int(fps),
                },
                priority=Priority.HIGH,
            )
            return f"已开启表情动画：{str(emoji or 'smile').strip().lower()}"
        except Exception as e:
            return f"开启表情动画失败：{str(e)}"

    @function_tool
    async def stop_rgb_effect(self) -> str:
        """停止动态特效/表情动画"""
        self.logger.debug("stop_rgb_effect called")
        try:
            # 更新时间戳（需要加锁）
            with self._timestamps_lock:
                now = time.time()
                self._light_override_until_ts = now + float(os.getenv("LELAMP_LIGHT_OVERRIDE_S") or "10")
                self._suppress_motion_until_ts = now + float(os.getenv("LELAMP_SUPPRESS_MOTION_AFTER_LIGHT_S") or "2")
            self.rgb_service.dispatch("effect_stop", None, priority=Priority.HIGH)
            return "已停止动态灯效"
        except Exception as e:
            return f"停止动态灯效失败：{str(e)}"

    @function_tool
    async def set_volume(self, volume_percent: int) -> str:
        """Control system audio volume."""
        self.logger.debug(f"set_volume called with volume: {volume_percent}%")
        try:
            if not 0 <= volume_percent <= 100:
                return "Error: Volume must be between 0 and 100 percent"
            await self._set_system_volume(volume_percent)
            return f"Set volume to {volume_percent}%"
        except Exception as e:
            return f"Error controlling volume: {str(e)}"

    @function_tool
    async def get_rate_limit_stats(self) -> str:
        """获取 API 速率限制统计信息（调试用）"""
        stats = get_all_rate_limiter_stats()
        lines = ["📊 API 速率限制统计："]
        for name, stat in stats.items():
            lines.append(f"\n{name}:")
            lines.append(f"  请求总数: {stat['requests_total']}")
            lines.append(f"  允许: {stat['requests_allowed']}")
            lines.append(f"  拒绝: {stat['requests_denied']}")
            lines.append(f"  拒绝率: {stat['denial_rate']:.1%}")
            lines.append(f"  平均等待: {stat['avg_wait_time']:.2f}s")
        return "\n".join(lines)

    @function_tool
    async def vision_answer(self, question: str) -> str:
        """Ask a question about what the lamp can see through its camera."""
        if not self._vision_service or not self._qwen_client:
            return "视觉能力未初始化。"

        # 应用速率限制
        if not await self._vision_rate_limiter.acquire(tokens=1, timeout=10.0):
            return "视觉 API 调用太频繁了，让我休息一下眼睛。"

        # 保存并覆盖灯光状态（需要加锁）
        with self._timestamps_lock:
            prev_override_until_ts = float(self._light_override_until_ts)
            self._light_override_until_ts = time.time() + 3600.0

        try:
            self.rgb_service.dispatch("solid", (255, 255, 255), priority=Priority.HIGH)

            latest = await self._vision_service.get_latest_jpeg_b64()
            if not latest:
                return "当前没有可用画面。请确保摄像头已启用并在刷新。"

            jpeg_b64, _ = latest
            return await self._qwen_client.describe(image_jpeg_b64=jpeg_b64, question=question)
        finally:
            # 恢复灯光状态（需要加锁）
            with self._timestamps_lock:
                self._light_override_until_ts = prev_override_until_ts

    @function_tool
    async def check_homework(self) -> str:
        """
        帮用户检查画面中的作业（数学、口算、填空等）。
        Analyze and check homework in the camera view (math, corrections, etc.).
        """
        if not self._vision_service or not self._qwen_client:
            return "视觉能力未初始化。"

        # 应用速率限制
        if not await self._vision_rate_limiter.acquire(tokens=1, timeout=10.0):
            return "作业检查太频繁了，让我也喘口气。"

        # 1. 补光 - 保存并覆盖灯光状态（需要加锁）
        with self._timestamps_lock:
            prev_override_until_ts = float(self._light_override_until_ts)
            self._light_override_until_ts = time.time() + 3600.0

        try:
            self.rgb_service.dispatch("solid", (255, 255, 255), priority=Priority.HIGH)
            
            # 2. 获取最清晰、最实时的画面
            latest = await self._vision_service.get_fresh_jpeg_b64(timeout_s=5.0)
            if not latest:
                return "拍照失败，无法看清作业。"

            jpeg_b64, _ = latest
            
            # 3. 调用视觉模型，使用老师人设
            prompt = (
                "你现在是一位认真负责且幽默的老师。请检查图片中的作业（通常是数学题、口算或填空）。"
                "指出错误的地方并给出正确答案或解析，鼓励用户改进。如果看不清，请提示用户调整作业位置或补光。"
                "请用中文回答，保持简洁。最后别忘了以LeLamp的性格损一下用户。"
            )
            return await self._qwen_client.describe(image_jpeg_b64=jpeg_b64, question=prompt)
        finally:
            # 恢复灯光状态（需要加锁）
            with self._timestamps_lock:
                self._light_override_until_ts = prev_override_until_ts

    @function_tool
    async def capture_to_feishu(self) -> str:
        """拍照并通过飞书机器人推送（方案B：直接上传图片），拍照前会锁定动作并停止以确保清晰度"""
        if not self._vision_service:
            return "视觉能力未初始化。"

        # 1. 锁定动作并停止当前所有动作（保持扭矩）
        self._motion_locked = True
        self.motors_service.dispatch("stop", None, priority=Priority.CRITICAL)
        await asyncio.sleep(1.5)  # 等待 1.5 秒让机械臂完全静止

        try:
            # 2. 获取飞书配置
            app_id = os.getenv("FEISHU_APP_ID")
            app_secret = os.getenv("FEISHU_APP_SECRET")
            receive_id = os.getenv("FEISHU_RECEIVE_ID")
            receive_id_type = os.getenv("FEISHU_RECEIVE_ID_TYPE") or "open_id"

            if not all([app_id, app_secret, receive_id]):
                return "飞书配置不完整，请检查环境变量 (FEISHU_APP_ID, FEISHU_APP_SECRET, FEISHU_RECEIVE_ID)"

            # 3. 获取 Token
            async def _do_req(req):
                return await asyncio.to_thread(lambda: urllib.request.urlopen(req, timeout=15).read())

            token_url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
            token_payload = json.dumps({"app_id": app_id, "app_secret": app_secret}).encode("utf-8")
            token_req = urllib.request.Request(
                token_url, data=token_payload, headers={"Content-Type": "application/json"}, method="POST"
            )
            
            token_resp = json.loads(await _do_req(token_req))
            token = token_resp.get("tenant_access_token")
            if not token:
                return f"获取飞书 Token 失败: {token_resp.get('msg')}"

            # 4. 获取当前画面 (确保是机械臂停止后的最新画面)
            latest = await self._vision_service.get_fresh_jpeg_b64(timeout_s=5.0)
            if not latest:
                return "拍照失败，无可用画面"
            jpeg_b64, _ = latest
            jpeg_data = base64.b64decode(jpeg_b64)

            # 5. 上传图片到飞书
            upload_url = "https://open.feishu.cn/open-apis/im/v1/images"
            boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
            
            body = []
            body.append(f"--{boundary}".encode("utf-8"))
            body.append(b'Content-Disposition: form-data; name="image_type"')
            body.append(b"")
            body.append(b"message")
            body.append(f"--{boundary}".encode("utf-8"))
            body.append(b'Content-Disposition: form-data; name="image"; filename="photo.jpg"')
            body.append(b"Content-Type: image/jpeg")
            body.append(b"")
            body.append(jpeg_data)
            body.append(f"--{boundary}--".encode("utf-8"))
            
            payload = b"\r\n".join(body)
            upload_headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            }
            upload_req = urllib.request.Request(upload_url, data=payload, headers=upload_headers, method="POST")
            upload_resp = json.loads(await _do_req(upload_req))
            
            image_key = upload_resp.get("data", {}).get("image_key")
            if not image_key:
                return f"上传图片到飞书失败: {upload_resp.get('msg')}"

            # 6. 发送消息
            msg_url = f"https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type={receive_id_type}"
            msg_payload = json.dumps({
                "receive_id": receive_id,
                "msg_type": "image",
                "content": json.dumps({"image_key": image_key})
            }).encode("utf-8")
            
            msg_req = urllib.request.Request(
                msg_url, data=msg_payload, 
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"}, 
                method="POST"
            )
            msg_resp = json.loads(await _do_req(msg_req))
            
            if msg_resp.get("code") != 0:
                return f"发送飞书消息失败: {msg_resp.get('msg')}"

            return "照片已成功推送到飞书。"

        except Exception as e:
            return f"飞书推送过程发生异常: {str(e)}"
        finally:
            # 7. 解锁动作
            self._motion_locked = False

    @function_tool
    async def web_search(self, query: str) -> str:
        """
        当用户问到实时信息、新闻、天气或你不确定的知识时，使用此工具在线搜索。
        Get real-time information from the web.

        Args:
            query: 搜索关键词 (Search query)
        """
        # 应用速率限制
        if not await self._search_rate_limiter.acquire(tokens=1, timeout=5.0):
            return "搜索太频繁了，请稍后再试。本灯也要休息一下的。"

        # 输入验证
        if not query or not query.strip():
            return "搜索关键词不能为空"

        if len(query) > 500:
            return "搜索关键词过长，请限制在500字以内"

        api_key = os.getenv("BOCHA_API_KEY")
        if not api_key:
            return "未配置 BOCHA_API_KEY，无法进行联网搜索。"

        self.logger.info(f"正在通过博查搜索: {query}")
        url = "https://api.bochaai.com/v1/web-search"

        # URL 安全验证
        if not validate_external_url(url, ALLOWED_API_DOMAINS):
            self.logger.error(f"搜索 API URL 验证失败: {url}")
            return "搜索服务配置错误"

        payload = json.dumps({
            "query": query.strip(),
            "freshness": "oneDay",
            "summary": True
        }).encode("utf-8")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        def _call() -> dict:
            req = urllib.request.Request(url=url, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read()
            return json.loads(raw.decode("utf-8"))

        try:
            data = await asyncio.to_thread(_call)
            if data.get("code") != 200:
                return f"搜索失败: {data.get('msg')}"

            data_content = data.get("data", {})
            web_pages = data_content.get("webPages", {}).get("value", [])

            if not web_pages:
                return "没有找到相关的联网搜索结果。"

            results = []
            for page in web_pages[:3]:  # 取前3个结果
                title = page.get("name")
                snippet = page.get("snippet")
                results.append(f"- {title}: {snippet}")

            return "以下是联网搜索到的信息：\n" + "\n".join(results)

        except Exception as e:
            return f"联网搜索发生异常: {str(e)}"

    @function_tool
    async def check_for_updates(self) -> str:
        """
        检查系统是否有新的 OTA 更新。
        Check for system updates.
        """
        if not self._ota_url:
            return "OTA 更新服务未配置 (LELAMP_OTA_URL missing)。"
        
        has_update, version, notes = self._ota_manager.check_for_update()
        if has_update:
            return f"发现新版本 {version}！\n更新内容：{notes}\n请问是否需要现在更新？(请回复‘确认更新’)"
        return f"当前已是最新版本 ({version})。"

    @function_tool
    async def perform_ota_update(self) -> str:
        """
        执行系统更新 (OTA)。注意：更新成功后服务将重启。
        Perform system update. Note: Service will restart upon success.
        """
        if not self._ota_url:
            return "OTA 更新服务未配置。"
            
        # Double check
        has_update, version, _ = self._ota_manager.check_for_update()
        if not has_update:
            return "当前没有可用更新。"
            
        result = self._ota_manager.perform_update()
        if "更新成功" in result:
            # Schedule a restart? Or just return message and let the user handle it?
            # Ideally, we should exit the process cleanly.
            # But the agent session might need to close first.
            asyncio.create_task(self._restart_later())
            return f"{result} 服务将在 5 秒后重启。"
        return f"更新失败：{result}"

    async def _restart_later(self):
        await asyncio.sleep(5)
        logger.info("Triggering restart...")
        # Rely on systemd or Docker to restart the container/process
        sys.exit(0)

    # ==================== Data Channel 消息处理 (Web Client v2.0) ====================

    async def handle_data_message(self, data: bytes, participant):
        """
        处理来自 Web Client 的 Data Channel 消息

        支持的消息类型:
        1. chat: 文字聊天消息
        2. command: 控制指令 (动作、灯光等)
        """
        try:
            message_str = data.decode('utf-8')
            self.logger.debug(f"收到 Data Channel 消息: {message_str}")
            message = json.loads(message_str)

            msg_type = message.get('type')

            if msg_type == 'chat':
                # 聊天消息 - 转换为语音输入
                content = message.get('content', '')
                if content:
                    await self.note_user_text(content)
                    self.logger.info(f"收到文字消息: {content}")

            elif msg_type == 'command':
                # 控制指令 - 路由到对应的功能
                action = message.get('action')
                params = message.get('params', {})
                self.logger.info(f"执行指令: {action}, 参数: {params}")

                result = await self._execute_command(action, params)

                # 发送执行结果
                if result:
                    await self._send_chat_message(result)

            else:
                self.logger.warning(f"未知消息类型: {msg_type}")

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON 解析失败: {e}")
        except Exception as e:
            self.logger.error(f"处理 Data Channel 消息失败: {e}")
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
        """
        try:
            # 播放录制动画
            if action == 'play_recording':
                recording_name = params.get('recording_name')
                if recording_name:
                    return await self.play_recording(recording_name)
                return "缺少录制名称参数"

            # 移动关节
            elif action == 'move_joint':
                joint_name = params.get('joint_name')
                angle = params.get('angle')
                if joint_name and angle is not None:
                    return await self.move_joint(joint_name, float(angle))
                return "缺少关节名称或角度参数"

            # 设置纯色灯光
            elif action == 'set_rgb_solid':
                r = params.get('r')
                g = params.get('g')
                b = params.get('b')
                if r is not None and g is not None and b is not None:
                    return await self.set_rgb_solid(int(r), int(g), int(b))
                return "缺少 RGB 参数"

            # 停止灯效
            elif action == 'stop_rgb_effect':
                return await self.stop_rgb_effect()

            # 灯效动画 (rgb_effect_breathing, rgb_effect_rainbow 等)
            elif action.startswith('rgb_effect_'):
                effect_name = action.replace('rgb_effect_', '')
                return await self._execute_rgb_effect(effect_name)

            # 视觉功能
            elif action == 'vision_answer':
                question = params.get('question', '这是什么')
                result = await self.vision_answer(question)
                # 发送视觉结果 (包含图片)
                await self._send_vision_result(result)
                return result

            elif action == 'check_homework':
                result = await self.check_homework()
                await self._send_vision_result(result)
                return result

            else:
                return f"未知指令: {action}"

        except Exception as e:
            self.logger.error(f"执行指令失败: {action}, 错误: {e}")
            return f"执行失败: {str(e)}"

    async def _execute_rgb_effect(self, effect_name: str) -> str:
        """执行 RGB 灯效"""
        effect_map = {
            'breathing': self.rgb_effect_breathing,
            'rainbow': self.rgb_effect_rainbow,
            'wave': self.rgb_effect_wave,
            'fire': self.rgb_effect_fire,
            'fireworks': self.rgb_effect_fireworks,
            'starry': self.rgb_effect_starry,
            'matrix_rain': self.rgb_effect_matrix_rain,
            'plasma': self.rgb_effect_plasma,
            'sparkle': self.rgb_effect_sparkle,
            'gradient': self.rgb_effect_gradient,
            'pulse': self.rgb_effect_pulse,
        }

        effect_func = effect_map.get(effect_name)
        if effect_func:
            return await effect_func()
        return f"未知灯效: {effect_name}"

    async def _send_chat_message(self, content: str):
        """向 Web Client 发送聊天消息"""
        try:
            if hasattr(self, 'room') and self.room:
                message = {
                    "type": "chat",
                    "content": content,
                    "timestamp": time.time()
                }
                data = json.dumps(message).encode('utf-8')
                await self.room.local_participant.publish_data(data)
                self.logger.debug(f"发送聊天消息: {content}")
        except Exception as e:
            self.logger.error(f"发送聊天消息失败: {e}")

    async def _send_vision_result(self, result: str, image_base64: str = None):
        """向 Web Client 发送视觉结果 (包含图片)"""
        try:
            if hasattr(self, 'room') and self.room:
                # 如果没有提供图片，尝试从 vision_service 获取最新帧
                if image_base64 is None and self._vision_service:
                    frame_data = self._vision_service.get_latest_frame()
                    if frame_data:
                        image_base64 = base64.b64encode(frame_data).decode('utf-8')

                message = {
                    "type": "vision_result",
                    "content": result,
                    "image_base64": image_base64,
                    "timestamp": time.time()
                }
                data = json.dumps(message).encode('utf-8')
                await self.room.local_participant.publish_data(data)
                self.logger.debug(f"发送视觉结果: {result[:100]}...")
        except Exception as e:
            self.logger.error(f"发送视觉结果失败: {e}")

    async def _update_camera_status(self, active: bool):
        """向 Web Client 更新摄像头状态"""
        try:
            if hasattr(self, 'room') and self.room:
                message = {
                    "type": "camera_status",
                    "active": active,
                    "timestamp": time.time()
                }
                data = json.dumps(message).encode('utf-8')
                await self.room.local_participant.publish_data(data)
                self.logger.debug(f"更新摄像头状态: {'激活' if active else '关闭'}")
        except Exception as e:
            self.logger.error(f"更新摄像头状态失败: {e}")


async def entrypoint(ctx: JobContext):
    config = _load_config()
    await ctx.connect()

    deepseek_llm = openai.LLM(
        model=config.deepseek_model,
        base_url=config.deepseek_base_url,
        api_key=config.deepseek_api_key,
    )

    qwen_client = Qwen3VLClient(
        base_url=config.modelscope_base_url,
        api_key=config.modelscope_api_key,
        model=config.modelscope_model,
        timeout_s=config.modelscope_timeout_s,
    )

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
    logger.info(
        "config ready: lamp_id=%s port=%s vision=%s camera=%s",
        config.lamp_id,
        config.lamp_port,
        config.vision_enabled,
        config.camera_index_or_path,
    )
    agent = LeLamp(
        port=config.lamp_port,
        lamp_id=config.lamp_id,
        vision_service=vision_service,
        qwen_client=qwen_client,
        ota_url=config.ota_url,
    )

    async def _on_state(state: str) -> None:
        await agent.set_conversation_state(state)

    async def _on_transcript(text: str) -> None:
        await agent.note_user_text(text)

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

    start_kwargs: dict[str, object] = {}
    if config.noise_cancellation_enabled:
        start_kwargs["room_input_options"] = RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        )

    # 注册 Data Channel 事件监听器 (Web Client v2.0 support)
    @ctx.room.on("data_received")
    def on_data_received(data: bytes, participant):
        """处理来自 Web Client 的 Data Channel 消息"""
        asyncio.create_task(agent.handle_data_message(data, participant))

    try:
        await session.start(
            agent=agent,
            room=ctx.room,
            **start_kwargs,
        )
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
