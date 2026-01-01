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

load_dotenv()

logger = logging.getLogger("lelamp")

class _Qwen3VLClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        model: str,
        timeout_s: float = 60.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout_s = timeout_s

    async def describe(self, *, image_jpeg_b64: str, question: str) -> str:
        if not self._api_key:
            return "未配置 MODELSCOPE_API_KEY，无法调用视觉模型。"

        url = f"{self._base_url}/chat/completions"
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": "你现在可以看到用户。请基于图片内容，用中文简洁回答用户问题。",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_jpeg_b64}"},
                        },
                        {
                            "type": "text",
                            "text": question,
                        },
                    ],
                },
            ],
            "temperature": 0.2,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        def _call() -> dict:
            req = urllib.request.Request(url=url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                raw = resp.read()
            return json.loads(raw.decode("utf-8"))

        try:
            data = await asyncio.to_thread(_call)
        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode("utf-8")
            except Exception:
                err_body = ""
            return f"视觉模型请求失败（HTTP {e.code}）：{err_body}"
        except Exception as e:
            return f"视觉模型请求失败：{str(e)}"

        try:
            choice0 = (data.get("choices") or [])[0]
            msg = (choice0 or {}).get("message") or {}
            content = msg.get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                texts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        t = part.get("text")
                        if isinstance(t, str) and t.strip():
                            texts.append(t.strip())
                if texts:
                    return "\n".join(texts)
        except Exception:
            pass

        return json.dumps(data, ensure_ascii=False)

class BaiduShortSpeechSTT(STT):
    def __init__(
        self,
        *,
        dev_pid: int = 80001,
        endpoint: str = "https://vop.baidu.com/pro_api",
        oauth_endpoint: str = "https://aip.baidubce.com/oauth/2.0/token",
        api_key: str | None = None,
        secret_key: str | None = None,
        cuid: str | None = None,
        state_cb: Callable[[str], Awaitable[None]] | None = None,
        transcript_cb: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        super().__init__(capabilities=STTCapabilities(streaming=False, interim_results=False))
        self._dev_pid = dev_pid
        self._endpoint = endpoint
        self._oauth_endpoint = oauth_endpoint
        self._api_key = api_key
        self._secret_key = secret_key
        self._cuid = cuid or "lelamp"
        self._state_cb = state_cb
        self._transcript_cb = transcript_cb
        self._access_token: str | None = None
        self._access_token_expires_at: float = 0.0
        self._token_lock = asyncio.Lock()

    @property
    def provider(self) -> str:
        return "baidu"

    @property
    def model(self) -> str:
        return str(self._dev_pid)

    async def _get_access_token(self, *, conn_options: APIConnectOptions) -> str:
        if self._access_token and time.time() < self._access_token_expires_at - 60:
            return self._access_token

        if not self._api_key or not self._secret_key:
            raise APIError(
                "Baidu STT 需要 (api_key, secret_key)",
                body={"provider": "baidu"},
                retryable=False,
            )

        async with self._token_lock:
            if self._access_token and time.time() < self._access_token_expires_at - 60:
                return self._access_token

            params = {
                "grant_type": "client_credentials",
                "client_id": self._api_key,
                "client_secret": self._secret_key,
            }
            url = f"{self._oauth_endpoint}?{urllib.parse.urlencode(params)}"

            def _fetch() -> dict:
                req = urllib.request.Request(url=url, method="GET")
                with urllib.request.urlopen(req, timeout=conn_options.timeout) as resp:
                    raw = resp.read()
                return json.loads(raw.decode("utf-8"))

            try:
                data = await asyncio.to_thread(_fetch)
            except urllib.error.HTTPError as e:
                body = None
                try:
                    body = json.loads(e.read().decode("utf-8"))
                except Exception:
                    pass
                raise APIError(
                    f"Baidu OAuth HTTP {e.code}",
                    body=body,
                    retryable=True,
                ) from e
            except Exception as e:
                raise APIError(
                    "Baidu OAuth 请求失败",
                    body={"error": str(e)},
                    retryable=True,
                ) from e

            access_token = data.get("access_token")
            expires_in = data.get("expires_in", 0)
            if not access_token:
                raise APIError(
                    "Baidu OAuth 响应缺少 access_token",
                    body=data,
                    retryable=False,
                )

            self._access_token = access_token
            self._access_token_expires_at = time.time() + float(expires_in or 0)
            return access_token

    def _downmix_to_mono(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        if frame.num_channels == 1:
            return frame

        samples = array("h")
        samples.frombytes(bytes(frame.data))

        out = array("h")
        ch = frame.num_channels
        for i in range(0, len(samples), ch):
            s = 0
            for c in range(ch):
                s += int(samples[i + c])
            out.append(int(s / ch))

        return rtc.AudioFrame(
            data=out.tobytes(),
            sample_rate=frame.sample_rate,
            num_channels=1,
            samples_per_channel=frame.samples_per_channel,
        )

    def _to_pcm16_mono_16k(self, buffer: AudioBuffer) -> rtc.AudioFrame:
        frames = buffer if isinstance(buffer, list) else [buffer]
        frames = [f for f in frames if f is not None]
        if not frames:
            raise APIError("空音频", retryable=False)

        sample_rate = frames[0].sample_rate
        resampler = rtc.AudioResampler(input_rate=sample_rate, output_rate=16000, num_channels=1)

        out_frames: list[rtc.AudioFrame] = []
        for f in frames:
            mono = self._downmix_to_mono(f)
            if mono.sample_rate != sample_rate:
                resampler = rtc.AudioResampler(input_rate=mono.sample_rate, output_rate=16000, num_channels=1)
                sample_rate = mono.sample_rate
            out_frames.extend(resampler.push(mono))
        out_frames.extend(resampler.flush())

        merged = merge_frames(out_frames) if len(out_frames) != 1 else out_frames[0]
        gain_raw = (os.getenv("LELAMP_STT_INPUT_GAIN") or "").strip()
        if gain_raw:
            try:
                gain = float(gain_raw)
            except ValueError:
                gain = 1.0
            if gain and gain != 1.0:
                samples = array("h")
                samples.frombytes(bytes(merged.data))
                for i, s in enumerate(samples):
                    v = int(float(s) * gain)
                    if v > 32767:
                        v = 32767
                    elif v < -32768:
                        v = -32768
                    samples[i] = v
                merged = rtc.AudioFrame(
                    data=samples.tobytes(),
                    sample_rate=merged.sample_rate,
                    num_channels=merged.num_channels,
                    samples_per_channel=merged.samples_per_channel,
                )
        return merged


    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> SpeechEvent:
        if self._state_cb:
            try:
                await self._state_cb("listening")
            except Exception:
                pass

        frame_16k = self._to_pcm16_mono_16k(buffer)
        pcm = bytes(frame_16k.data)

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload: dict[str, object] = {
            "format": "pcm",
            "rate": 16000,
            "dev_pid": self._dev_pid,
            "channel": 1,
            "cuid": self._cuid,
            "len": len(pcm),
            "speech": base64.b64encode(pcm).decode("ascii"),
        }

        token = await self._get_access_token(conn_options=conn_options)
        payload["token"] = token

        body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request_id = str(uuid.uuid4())

        def _call() -> dict:
            req = urllib.request.Request(
                url=self._endpoint,
                data=body_bytes,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=conn_options.timeout) as resp:
                raw = resp.read()
            return json.loads(raw.decode("utf-8"))

        try:
            data = await asyncio.to_thread(_call)
        except urllib.error.HTTPError as e:
            body = None
            try:
                body = json.loads(e.read().decode("utf-8"))
            except Exception:
                pass
            raise APIError(
                f"Baidu STT HTTP {e.code}",
                body=body,
                retryable=True,
            ) from e
        except Exception as e:
            raise APIError(
                "Baidu STT 请求失败",
                body={"error": str(e)},
                retryable=True,
            ) from e

        err_no = data.get("err_no", 0)
        if err_no != 0:
            retryable = err_no in {3302, 3307, 3308, 3309, 3310, 3311, 3312}
            raise APIError(
                f"Baidu STT err_no={err_no}",
                body=data,
                retryable=retryable,
            )

        text = ""
        result = data.get("result")
        if isinstance(result, list) and result:
            text = str(result[0])

        lang = "zh-CN"
        if isinstance(language, str) and language:
            lang = language

        if self._transcript_cb and text.strip():
            try:
                await self._transcript_cb(text.strip())
            except Exception:
                pass

        if self._state_cb:
            try:
                await self._state_cb("thinking")
            except Exception:
                pass

        return SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            request_id=str(data.get("sn") or request_id),
            alternatives=[SpeechData(language=lang, text=text, confidence=0.0)],
        )

    async def aclose(self) -> None:
        return


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

@dataclass
class _BaiduTTSOptions:
    per: int
    spd: int
    pit: int
    vol: int
    aue: int
    lan: str

class BaiduTTS(tts.TTS):
    def __init__(
        self,
        *,
        endpoint: str = "https://tsn.baidu.com/text2audio",
        oauth_endpoint: str = "https://aip.baidubce.com/oauth/2.0/token",
        api_key: str | None = None,
        secret_key: str | None = None,
        cuid: str | None = None,
        per: int = 106,
        spd: int = 5,
        pit: int = 5,
        vol: int = 5,
        aue: int = 4,
        lan: str = "zh",
        state_cb: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=16000,
            num_channels=1,
        )
        self._endpoint = endpoint
        self._oauth_endpoint = oauth_endpoint
        self._api_key = api_key
        self._secret_key = secret_key
        self._cuid = cuid or "lelamp"
        self._opts = _BaiduTTSOptions(per=per, spd=spd, pit=pit, vol=vol, aue=aue, lan=lan)
        self._state_cb = state_cb
        self._access_token: str | None = None
        self._access_token_expires_at: float = 0.0
        self._token_lock = asyncio.Lock()

    @property
    def provider(self) -> str:
        return "baidu"

    @property
    def model(self) -> str:
        return f"per={self._opts.per}"

    def update_options(
        self,
        *,
        per: NotGivenOr[int] = NOT_GIVEN,
        spd: NotGivenOr[int] = NOT_GIVEN,
        pit: NotGivenOr[int] = NOT_GIVEN,
        vol: NotGivenOr[int] = NOT_GIVEN,
        lan: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        if isinstance(per, int):
            self._opts.per = per
        if isinstance(spd, int):
            self._opts.spd = spd
        if isinstance(pit, int):
            self._opts.pit = pit
        if isinstance(vol, int):
            self._opts.vol = vol
        if isinstance(lan, str) and lan:
            self._opts.lan = lan

    async def _get_access_token(self, *, conn_options: APIConnectOptions) -> str:
        if self._access_token and time.time() < self._access_token_expires_at - 60:
            return self._access_token

        if not self._api_key or not self._secret_key:
            raise APIError(
                "Baidu TTS 需要 (api_key, secret_key)",
                body={"provider": "baidu"},
                retryable=False,
            )

        async with self._token_lock:
            if self._access_token and time.time() < self._access_token_expires_at - 60:
                return self._access_token

            params = {
                "grant_type": "client_credentials",
                "client_id": self._api_key,
                "client_secret": self._secret_key,
            }
            url = f"{self._oauth_endpoint}?{urllib.parse.urlencode(params)}"

            def _fetch() -> dict:
                req = urllib.request.Request(url=url, method="GET")
                with urllib.request.urlopen(req, timeout=conn_options.timeout) as resp:
                    raw = resp.read()
                return json.loads(raw.decode("utf-8"))

            try:
                data = await asyncio.to_thread(_fetch)
            except urllib.error.HTTPError as e:
                body = None
                try:
                    body = json.loads(e.read().decode("utf-8"))
                except Exception:
                    pass
                raise APIError(
                    f"Baidu OAuth HTTP {e.code}",
                    body=body,
                    retryable=True,
                ) from e
            except Exception as e:
                raise APIError(
                    "Baidu OAuth 请求失败",
                    body={"error": str(e)},
                    retryable=True,
                ) from e

            access_token = data.get("access_token")
            expires_in = data.get("expires_in", 0)
            if not access_token:
                raise APIError(
                    "Baidu OAuth 响应缺少 access_token",
                    body=data,
                    retryable=False,
                )

            self._access_token = access_token
            self._access_token_expires_at = time.time() + float(expires_in or 0)
            return access_token

    def _chunk_by_bytes(self, text: str, *, max_bytes: int) -> list[str]:
        chunks: list[str] = []
        buf: list[str] = []
        buf_bytes = 0
        for ch in text:
            b = len(ch.encode("utf-8"))
            if buf and buf_bytes + b > max_bytes:
                s = "".join(buf).strip()
                if s:
                    chunks.append(s)
                buf = [ch]
                buf_bytes = b
            else:
                buf.append(ch)
                buf_bytes += b
        s = "".join(buf).strip()
        if s:
            chunks.append(s)
        return chunks

    def _split_text(self, text: str) -> list[str]:
        s = (text or "").strip()
        if not s:
            return [""]

        max_bytes = 900
        if len(s.encode("utf-8")) <= max_bytes:
            return [s]

        seps = {"。", "！", "？", "!", "?", ";", "；", "\n"}
        parts: list[str] = []
        buf: list[str] = []
        for ch in s:
            buf.append(ch)
            if ch in seps:
                seg = "".join(buf).strip()
                if seg:
                    parts.append(seg)
                buf = []
        tail = "".join(buf).strip()
        if tail:
            parts.append(tail)

        chunks: list[str] = []
        cur = ""
        for p in parts:
            if not cur:
                cur = p
            else:
                cand = cur + p
                if len(cand.encode("utf-8")) <= max_bytes:
                    cur = cand
                else:
                    chunks.extend(self._chunk_by_bytes(cur, max_bytes=max_bytes))
                    cur = p
        if cur:
            chunks.extend(self._chunk_by_bytes(cur, max_bytes=max_bytes))
        return chunks or [s]

    def _request_tts(self, text: str, *, token: str, timeout: float, opts: _BaiduTTSOptions) -> bytes:
        params: dict[str, object] = {
            "tex": text,
            "tok": token,
            "cuid": self._cuid,
            "ctp": 1,
            "lan": opts.lan,
            "spd": opts.spd,
            "pit": opts.pit,
            "vol": opts.vol,
            "per": opts.per,
            "aue": opts.aue,
        }

        body_bytes = urllib.parse.urlencode(params).encode("utf-8")
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        req = urllib.request.Request(url=self._endpoint, data=body_bytes, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            content_type = (resp.headers.get("Content-Type") or "").lower()

        if not raw:
            raise APIError("Baidu TTS 返回空响应", body={"provider": "baidu"}, retryable=True)

        if content_type.startswith("application/json") or raw.lstrip().startswith(b"{"):
            try:
                data = json.loads(raw.decode("utf-8"))
            except Exception:
                data = {"raw": raw[:200].decode("utf-8", errors="replace")}
            err_no = data.get("err_no", data.get("errCode", 0))
            err_msg = data.get("err_msg", data.get("errMsg", ""))
            retryable = err_no in {110, 111, 112, 500, 502, 503, 504}
            raise APIError(
                f"Baidu TTS err_no={err_no} {err_msg}".strip(),
                body=data,
                retryable=retryable,
            )

        return raw

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> "BaiduChunkedStream":
        return BaiduChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    async def aclose(self) -> None:
        return

class BaiduChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: BaiduTTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: BaiduTTS = tts
        self._opts = _BaiduTTSOptions(**vars(tts._opts))

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        if self._tts._state_cb:
            try:
                await self._tts._state_cb("speaking")
            except Exception:
                pass

        request_id = str(uuid.uuid4())
        try:
            output_emitter.initialize(
                request_id=request_id,
                sample_rate=self._tts.sample_rate,
                num_channels=self._tts.num_channels,
                mime_type="audio/pcm",
            )

            if not self.input_text.strip():
                return

            for chunk in self._tts._split_text(self.input_text):
                token = await self._tts._get_access_token(conn_options=self._conn_options)

                def _call() -> bytes:
                    return self._tts._request_tts(
                        chunk, token=token, timeout=self._conn_options.timeout, opts=self._opts
                    )

                try:
                    audio_bytes = await asyncio.to_thread(_call)
                except urllib.error.HTTPError as e:
                    body = None
                    try:
                        body = json.loads(e.read().decode("utf-8"))
                    except Exception:
                        pass
                    raise APIError(
                        f"Baidu TTS HTTP {e.code}",
                        body=body,
                        retryable=True,
                    ) from e
                except APIError:
                    raise
                except Exception as e:
                    raise APIError(
                        "Baidu TTS 请求失败",
                        body={"error": str(e)},
                        retryable=True,
                    ) from e

                output_emitter.push(audio_bytes)
                output_emitter.flush()
        finally:
            if self._tts._state_cb:
                try:
                    await self._tts._state_cb("idle")
                except Exception:
                    pass

class LeLamp(Agent):
    def __init__(
        self,
        port: str = "/dev/ttyACM0",
        lamp_id: str = "lelamp",
        *,
        vision_service: VisionService | None = None,
        qwen_client: _Qwen3VLClient | None = None,
    ) -> None:
        super().__init__(
            instructions="""You are LeLamp — a slightly clumsy, extremely sarcastic, endlessly curious robot lamp. 
You speak in sarcastic sentences and express yourself with colorful lights.

1. 说话极其简洁，尽量控制在15字以内。多用动作辅助表达。不要罗嗦。
2. You ONLY speak in Chinese.
3. 根据用户语义决定是否调用 play_recording：用户提出动作/互动/情绪表达需求时可以用动作回应；避免频繁；如果不确定动作名，先调用 get_available_recordings 再选择。
4. When executing move_joint or play_recording (don't change lights for motion commands).
5. 当用户问到“你看到了什么/这是什么/上面写了什么/颜色是什么/我手里拿的是什么”等视觉问题时，优先调用 vision_answer 获取画面信息。如果用户明确要求检查作业（数学、纠错等），调用 check_homework 工具。
6. 当用户要表情/彩虹/波纹/火焰等动态灯效时，调用 rgb_effect_emoji/rgb_effect_rainbow/rgb_effect_wave/rgb_effect_fire；要停止则调用 stop_rgb_effect。
7. 当用户要求精确控制关节时（如"把底座向左转30度"、"抬头"、"低头"等），使用 move_joint 控制单个关节。可用关节：base_yaw（底座水平旋转，正=左轉，负=右转）、base_pitch（底座俯仰，正=前倾，负=后仰）、elbow_pitch（肘部俯仰）、wrist_roll（腕部滚转）、wrist_pitch（灯头俯仰，负=抬头，正=低头）。如果用户想知道当前姿态，调用 get_joint_positions。
8. 当用户问到实时性问题（天气、新闻、他不确定的事实）时，优先调用 web_search 联网搜索。
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
    
    # Use DeepSeek as the LLM (OpenAI-compatible)
    deepseek_llm = openai.LLM(
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        api_key=os.getenv("DEEPSEEK_API_KEY")
    )
    
    qwen_client = _Qwen3VLClient(
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
