import os
import subprocess
import logging
import asyncio
import base64
import json
import random
import time
import uuid
import mimetypes
import urllib.error
import urllib.parse
import urllib.request
from array import array
from dataclasses import dataclass
from dotenv import load_dotenv
from typing import Awaitable, Callable, Optional

os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RoomInputOptions,
    WorkerOptions,
    cli,
    function_tool,
    tts,
)
from livekit.agents._exceptions import APIError
from livekit.agents.stt import STT, STTCapabilities, SpeechData, SpeechEvent, SpeechEventType
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer
from livekit.agents.utils.audio import merge_frames
from livekit.plugins import noise_cancellation, openai, silero

from lelamp.service import Priority
from lelamp.service.motors.motors_service import MotorsService
from lelamp.service.rgb.rgb_service import RGBService
from lelamp.service.vision.vision_service import VisionService
from lelamp.integrations.baidu_speech import BaiduShortSpeechSTT, BaiduTTS
from lelamp.integrations.qwen_vl import Qwen3VLClient

load_dotenv()

logger = logging.getLogger("lelamp")






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
        self._vision_service = vision_service
        self._qwen_client = qwen_client
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
        self._conversation_state_lock = asyncio.Lock()
        self._light_override_until_ts = 0.0
        self._suppress_motion_until_ts = 0.0
        self._motion_locked = False
        self._last_user_text = ""
        self._last_user_text_ts = 0.0
        self._last_motion_ts = 0.0

    async def note_user_text(self, text: str) -> None:
        self._last_user_text = text
        self._last_user_text_ts = time.time()

    def _set_system_volume(self, volume_percent: int):
        """Internal helper to set system volume"""
        try:
            cmd_line = ["sudo", "-u", "pi", "amixer", "sset", "Line", f"{volume_percent}%"]
            cmd_line_dac = ["sudo", "-u", "pi", "amixer", "sset", "Line DAC", f"{volume_percent}%"]
            cmd_line_hp = ["sudo", "-u", "pi", "amixer", "sset", "HP", f"{volume_percent}%"]
            
            subprocess.run(cmd_line, capture_output=True, text=True, timeout=5)
            subprocess.run(cmd_line_dac, capture_output=True, text=True, timeout=5)
            subprocess.run(cmd_line_hp, capture_output=True, text=True, timeout=5)
        except Exception:
            pass

    async def set_conversation_state(self, state: str) -> None:
        async with self._conversation_state_lock:
            if state == self._conversation_state:
                return
            self._conversation_state = state

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
        print("LeLamp: get_available_recordings function called")
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
        print(f"LeLamp: play_recording function called with recording_name: {recording_name}")
        try:
            if self._motion_locked:
                return "动作已锁定（例如拍照中），本次不执行动作"
            if time.time() < float(self._suppress_motion_until_ts):
                return "刚执行过灯光指令，短时间内不执行动作"
            now = time.time()
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
        print(f"LeLamp: move_joint function called with joint_name={joint_name}, angle={angle}")
        valid_joints = ["base_yaw", "base_pitch", "elbow_pitch", "wrist_roll", "wrist_pitch"]
        try:
            if self._motion_locked:
                return "动作已锁定（例如拍照中），本次不执行动作"
            if joint_name not in valid_joints:
                return f"无效的关节名称：{joint_name}。可用关节：{', '.join(valid_joints)}"
            self.motors_service.dispatch("move_joint", {"joint_name": joint_name, "angle": float(angle)})
            return f"已将 {joint_name} 移动到 {angle} 度"
        except Exception as e:
            return f"控制关节失败：{str(e)}"

    @function_tool
    async def get_joint_positions(self) -> str:
        """获取所有关节的当前位置（角度）。用于了解台灯当前的姿态。"""
        print("LeLamp: get_joint_positions function called")
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
        print(f"LeLamp: set_rgb_solid function called with RGB({red}, {green}, {blue})")
        try:
            if not all(0 <= val <= 255 for val in [red, green, blue]):
                return "Error: RGB values must be between 0 and 255"
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
        print(f"LeLamp: paint_rgb_pattern function called with {len(colors)} colors")
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
        print(f"LeLamp: set_rgb_brightness function called with percent: {percent}")
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
        print(
            f"LeLamp: rgb_effect_rainbow function called with speed={speed}, saturation={saturation}, value={value}, fps={fps}"
        )
        try:
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
        print(
            f"LeLamp: rgb_effect_wave function called with rgb=({red},{green},{blue}), speed={speed}, freq={freq}, fps={fps}"
        )
        try:
            if not all(0 <= v <= 255 for v in (int(red), int(green), int(blue))):
                return "RGB 必须是 0-255"
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
        print(f"LeLamp: rgb_effect_fire function called with intensity={intensity}, fps={fps}")
        try:
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
        print(
            f"LeLamp: rgb_effect_emoji function called with emoji={emoji}, fg=({red},{green},{blue}), bg=({bg_red},{bg_green},{bg_blue}), blink={blink}, period_s={period_s}, fps={fps}"
        )
        try:
            if not all(0 <= v <= 255 for v in (int(red), int(green), int(blue), int(bg_red), int(bg_green), int(bg_blue))):
                return "RGB 必须是 0-255"
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
        print("LeLamp: stop_rgb_effect function called")
        try:
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
        print(f"LeLamp: set_volume function called with volume: {volume_percent}%")
        try:
            if not 0 <= volume_percent <= 100:
                return "Error: Volume must be between 0 and 100 percent"
            self._set_system_volume(volume_percent)
            return f"Set volume to {volume_percent}%"
        except Exception as e:
            return f"Error controlling volume: {str(e)}"

    @function_tool
    async def vision_answer(self, question: str) -> str:
        """Ask a question about what the lamp can see through its camera."""
        if not self._vision_service or not self._qwen_client:
            return "视觉能力未初始化。"

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
            self._light_override_until_ts = prev_override_until_ts

    @function_tool
    async def check_homework(self) -> str:
        """
        帮用户检查画面中的作业（数学、口算、填空等）。
        Analyze and check homework in the camera view (math, corrections, etc.).
        """
        if not self._vision_service or not self._qwen_client:
            return "视觉能力未初始化。"

        # 1. 补光
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
        api_key = os.getenv("BOCHA_API_KEY")
        if not api_key:
            return "未配置 BOCHA_API_KEY，无法进行联网搜索。"

        print(f"LeLamp: 正在通过博查搜索: {query}")
        url = "https://api.bochaai.com/v1/web-search"
        payload = json.dumps({
            "query": query,
            "freshness": "oneDay", # 可选：noLimit, oneDay, oneWeek, oneMonth, oneYear
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
            for page in web_pages[:3]: # 取前3个结果
                title = page.get("name")
                snippet = page.get("snippet")
                results.append(f"- {title}: {snippet}")
            
            return "以下是联网搜索到的信息：\n" + "\n".join(results)

        except Exception as e:
            return f"联网搜索发生异常: {str(e)}"

async def entrypoint(ctx: JobContext):
    await ctx.connect()
    
    # Use ModelScope DeepSeek as the LLM (OpenAI-compatible)
    deepseek_llm = openai.LLM(
        model=os.getenv("DEEPSEEK_MODEL") or "deepseek-ai/DeepSeek-V3.2",
        base_url=os.getenv("DEEPSEEK_BASE_URL") or "https://api-inference.modelscope.cn/v1",
        api_key=os.getenv("MODELSCOPE_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    )
    
    qwen_client = Qwen3VLClient(
        base_url=os.getenv("MODELSCOPE_BASE_URL") or "https://api-inference.modelscope.cn/v1",
        api_key=os.getenv("MODELSCOPE_API_KEY"),
        model=os.getenv("MODELSCOPE_MODEL") or "Qwen/Qwen3-VL-235B-A22B-Instruct",
        timeout_s=float(os.getenv("MODELSCOPE_TIMEOUT_S") or "60"),
    )

    vision_enabled = (os.getenv("LELAMP_VISION_ENABLED") or "1").strip().lower() in ("1", "true", "yes", "on")
    idx_or_path_raw = (os.getenv("LELAMP_CAMERA_INDEX_OR_PATH") or "0").strip()
    try:
        index_or_path: int | str = int(idx_or_path_raw)
    except ValueError:
        index_or_path = idx_or_path_raw

    vision_service = VisionService(
        enabled=vision_enabled,
        index_or_path=index_or_path,
        width=int(os.getenv("LELAMP_CAMERA_WIDTH") or "1024"),
        height=int(os.getenv("LELAMP_CAMERA_HEIGHT") or "768"),
        capture_interval_s=float(os.getenv("LELAMP_VISION_CAPTURE_INTERVAL_S") or "2.5"),
        jpeg_quality=int(os.getenv("LELAMP_VISION_JPEG_QUALITY") or "92"),
        max_age_s=float(os.getenv("LELAMP_VISION_MAX_AGE_S") or "15"),
        rotate_deg=int(os.getenv("LELAMP_CAMERA_ROTATE_DEG") or "0"),
        flip=os.getenv("LELAMP_CAMERA_FLIP") or "none",
    )
    vision_service.start()

    port = os.getenv("LELAMP_PORT") or "/dev/ttyACM0"
    lamp_id = os.getenv("LELAMP_ID") or "lelamp"
    agent = LeLamp(port=port, lamp_id=lamp_id, vision_service=vision_service, qwen_client=qwen_client)

    # --- MCP Integration ---
    from lelamp.mcp import McpManager
    mcp_manager = McpManager()
    agent.mcp_manager = mcp_manager # Keep reference
    
    mcp_cmd = os.getenv("LELAMP_MCP_CMD")
    if mcp_cmd:
        try:
            mcp_args = (os.getenv("LELAMP_MCP_ARGS") or "").split()
            logger.info(f"Connecting to MCP server: {mcp_cmd} {mcp_args}")
            await mcp_manager.connect_stdio("default", mcp_cmd, mcp_args)
            
            bridged_tools = await mcp_manager.get_bridged_tools()
            for name, func in bridged_tools.items():
                if not hasattr(agent, name):
                    setattr(agent, name, func)
                    logger.info(f"Registered MCP tool: {name}")
                else:
                    logger.warning(f"MCP tool {name} conflicts with existing agent method, skipping.")
        except Exception as e:
            logger.error(f"Failed to initialize MCP: {e}")
    # -----------------------

    async def _on_state(state: str) -> None:
        await agent.set_conversation_state(state)

    async def _on_transcript(text: str) -> None:
        await agent.note_user_text(text)

    session = AgentSession(
        vad=_build_vad(),
        stt=BaiduShortSpeechSTT(
            api_key=os.getenv("BAIDU_SPEECH_API_KEY"),
            secret_key=os.getenv("BAIDU_SPEECH_SECRET_KEY"),
            cuid=os.getenv("BAIDU_SPEECH_CUID") or "lelamp",
            state_cb=_on_state,
            transcript_cb=_on_transcript,
        ),
        llm=deepseek_llm,
        tts=BaiduTTS(
            api_key=os.getenv("BAIDU_SPEECH_API_KEY"),
            secret_key=os.getenv("BAIDU_SPEECH_SECRET_KEY"),
            cuid=os.getenv("BAIDU_SPEECH_CUID") or "lelamp",
            per=4,
            state_cb=_on_state,
        ),
    )

    start_kwargs: dict[str, object] = {}
    noise_cancellation_enabled = (os.getenv("LELAMP_NOISE_CANCELLATION") or "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if noise_cancellation_enabled:
        start_kwargs["room_input_options"] = RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        )

    await session.start(
        agent=agent,
        room=ctx.room,
        **start_kwargs,
    )
    
    # Optional: Initial greeting
    await session.say("Hello! 小宝贝上线了.", allow_interruptions=False)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
