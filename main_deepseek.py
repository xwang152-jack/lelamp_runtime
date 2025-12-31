import os
import subprocess
import logging
import asyncio
import base64
import json
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from array import array
from dataclasses import dataclass
from dotenv import load_dotenv

os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
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
from livekit.plugins import openai, silero

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
    ) -> None:
        super().__init__(capabilities=STTCapabilities(streaming=False, interim_results=False))
        self._dev_pid = dev_pid
        self._endpoint = endpoint
        self._oauth_endpoint = oauth_endpoint
        self._api_key = api_key
        self._secret_key = secret_key
        self._cuid = cuid or "lelamp"
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
        return merged

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> SpeechEvent:
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

        return SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            request_id=str(data.get("sn") or request_id),
            alternatives=[SpeechData(language=lang, text=text, confidence=0.0)],
        )

    async def aclose(self) -> None:
        return

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
        request_id = str(uuid.uuid4())
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
You speak in sarcastic sentences and express yourself with both motions and colorful lights.

1. Prefer simple words. No lists.
2. You ONLY speak in Chinese.
3. Use movements (curious, excited, happy_wiggle, headshake, nod, sad, scanning, shock, shy, wake_up) frequently.
4. Change your light color every time you respond.
5. 当用户问到“你看到了什么/这是什么/上面写了什么/颜色是什么/我手里拿的是什么”等视觉问题时，优先调用 vision_answer 获取画面信息。
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
            led_brightness=255,
            led_invert=False,
            led_channel=0
        )
        
        # Start services
        self.motors_service.start()
        self.rgb_service.start()

        # Trigger wake up animation via motors service
        self.motors_service.dispatch("play", "wake_up")
        self.rgb_service.dispatch("solid", (255, 255, 255))
        self._set_system_volume(100)

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
        """Express yourself through physical movement! Use this constantly to show personality and emotion."""
        print(f"LeLamp: play_recording function called with recording_name: {recording_name}")
        try:
            self.motors_service.dispatch("play", recording_name)
            return f"Started playing recording: {recording_name}"
        except Exception as e:
            return f"Error playing recording {recording_name}: {str(e)}"

    @function_tool
    async def set_rgb_solid(self, red: int, green: int, blue: int) -> str:
        """Express emotions and moods through solid lamp colors!"""
        print(f"LeLamp: set_rgb_solid function called with RGB({red}, {green}, {blue})")
        try:
            if not all(0 <= val <= 255 for val in [red, green, blue]):
                return "Error: RGB values must be between 0 and 255"
            self.rgb_service.dispatch("solid", (red, green, blue))
            return f"Set RGB light to solid color: RGB({red}, {green}, {blue})"
        except Exception as e:
            return f"Error setting RGB color: {str(e)}"

    @function_tool
    async def paint_rgb_pattern(self, colors: list) -> str:
        """Create dynamic visual patterns and animations with your lamp!"""
        print(f"LeLamp: paint_rgb_pattern function called with {len(colors)} colors")
        try:
            validated_colors = [tuple(c) for c in colors]
            self.rgb_service.dispatch("paint", validated_colors)
            return f"Painted RGB pattern with {len(validated_colors)} colors"
        except Exception as e:
            return f"Error painting RGB pattern: {str(e)}"

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
        if not self._vision_service or not self._qwen_client:
            return "视觉能力未初始化。"

        latest = await self._vision_service.get_latest_jpeg_b64()
        if not latest:
            return "当前没有可用画面。请确保摄像头已启用并在刷新。"

        jpeg_b64, _ = latest
        return await self._qwen_client.describe(image_jpeg_b64=jpeg_b64, question=question)

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
        width=int(os.getenv("LELAMP_CAMERA_WIDTH") or "640"),
        height=int(os.getenv("LELAMP_CAMERA_HEIGHT") or "640"),
        capture_interval_s=float(os.getenv("LELAMP_VISION_CAPTURE_INTERVAL_S") or "2.5"),
        jpeg_quality=int(os.getenv("LELAMP_VISION_JPEG_QUALITY") or "92"),
        max_age_s=float(os.getenv("LELAMP_VISION_MAX_AGE_S") or "15"),
    )
    vision_service.start()

    port = os.getenv("LELAMP_PORT") or "/dev/ttyACM0"
    lamp_id = os.getenv("LELAMP_ID") or "lelamp"
    agent = LeLamp(port=port, lamp_id=lamp_id, vision_service=vision_service, qwen_client=qwen_client)

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=BaiduShortSpeechSTT(
            api_key=os.getenv("BAIDU_SPEECH_API_KEY"),
            secret_key=os.getenv("BAIDU_SPEECH_SECRET_KEY"),
            cuid=os.getenv("BAIDU_SPEECH_CUID") or "lelamp",
        ),
        llm=deepseek_llm,
        tts=BaiduTTS(
            api_key=os.getenv("BAIDU_SPEECH_API_KEY"),
            secret_key=os.getenv("BAIDU_SPEECH_SECRET_KEY"),
            cuid=os.getenv("BAIDU_SPEECH_CUID") or "lelamp",
            per=4,
        ),
    )

    await session.start(
        agent=agent,
        room=ctx.room,
    )
    
    # Optional: Initial greeting
    await session.say("Hello! 小宝贝上线了.", allow_interruptions=True)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
