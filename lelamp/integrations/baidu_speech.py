import os
import json
import time
import uuid
import asyncio
import base64
import logging
import httpx
from array import array
from typing import Callable, Awaitable, List
from dataclasses import dataclass

from livekit import rtc
from livekit.agents import stt, tts
from livekit.agents._exceptions import APIError
from livekit.agents.stt import STTCapabilities, SpeechData, SpeechEvent, SpeechEventType
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer
from livekit.agents.utils.audio import merge_frames

logger = logging.getLogger("lelamp.integrations.baidu")

@dataclass
class _BaiduTTSOptions:
    per: int
    spd: int
    pit: int
    vol: int
    aue: int
    lan: str

class BaiduShortSpeechSTT(stt.STT):
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
            
            try:
                async with httpx.AsyncClient(timeout=conn_options.timeout) as client:
                    resp = await client.get(self._oauth_endpoint, params=params)
                    resp.raise_for_status()
                    data = resp.json()
            except httpx.HTTPStatusError as e:
                body = None
                try:
                    body = e.response.json()
                except Exception:
                    pass
                raise APIError(
                    f"Baidu OAuth HTTP {e.response.status_code}",
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

        request_id = str(uuid.uuid4())

        try:
            async with httpx.AsyncClient(timeout=conn_options.timeout) as client:
                resp = await client.post(self._endpoint, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as e:
            body = None
            try:
                body = e.response.json()
            except Exception:
                pass
            raise APIError(
                f"Baidu STT HTTP {e.response.status_code}",
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
            
            try:
                async with httpx.AsyncClient(timeout=conn_options.timeout) as client:
                    resp = await client.get(self._oauth_endpoint, params=params)
                    resp.raise_for_status()
                    data = resp.json()
            except httpx.HTTPStatusError as e:
                body = None
                try:
                    body = e.response.json()
                except Exception:
                    pass
                raise APIError(
                    f"Baidu OAuth HTTP {e.response.status_code}",
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

    async def _request_tts(self, text: str, *, token: str, timeout: float, opts: _BaiduTTSOptions) -> bytes:
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

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(self._endpoint, data=params, headers={"Content-Type": "application/x-www-form-urlencoded"})
                # Note: Baidu TTS sometimes returns 200 with error message in body, so we need to check content-type
                raw = resp.content
                content_type = resp.headers.get("Content-Type", "").lower()
        except Exception as e:
            raise APIError("Baidu TTS 请求失败", body={"error": str(e)}, retryable=True) from e

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

                try:
                    # Now calling async method directly
                    audio_bytes = await self._tts._request_tts(
                        chunk, token=token, timeout=self._conn_options.timeout, opts=self._opts
                    )
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
