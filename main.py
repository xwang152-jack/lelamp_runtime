import sys
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")

from livekit.agents import (
    JobContext,
    RoomInputOptions,
    WorkerOptions,
    cli,
)
from livekit.plugins import noise_cancellation, openai, silero

from lelamp.service.vision.vision_service import VisionService
from lelamp.integrations.baidu_speech import BaiduShortSpeechSTT, BaiduTTS
from lelamp.integrations.qwen_vl import Qwen3VLClient
from lelamp.utils.security import verify_license
from lelamp.utils.logging import setup_logging as setup_enhanced_logging
from lelamp.config import (
    _get_env_str,
    _get_env_bool,
    AppConfig,
    load_motor_config,
)

# 导入 LeLamp 代理类
from lelamp.agent.lelamp_agent import LeLamp

# 注册记忆模型到 SQLAlchemy Base（确保 create_all 包含记忆表）
import lelamp.memory  # noqa: F401

load_dotenv()

logger = logging.getLogger("lelamp")


def _setup_logging() -> None:
    """配置日志系统，支持文件日志和日志轮转"""
    level_raw = (os.getenv("LOG_LEVEL") or "INFO").strip().upper()
    log_to_file = _get_env_bool("LELAMP_LOG_TO_FILE", False)

    log_dir = None
    if log_to_file:
        log_dir_str = _get_env_str("LELAMP_LOG_DIR", "logs")
        log_dir = Path(log_dir_str)

    enable_json = _get_env_bool("LELAMP_LOG_JSON", False)

    setup_enhanced_logging(
        log_level=level_raw,
        log_dir=log_dir,
        enable_json=enable_json,
    )


def _load_config() -> AppConfig:
    from lelamp.config import load_config_strict
    return load_config_strict()


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


async def entrypoint(ctx: JobContext):
    from livekit.agents import AgentSession
    import lelamp.database.models  # 注册所有 ORM 模型到 Base
    from lelamp.database.base import init_db

    # 初始化数据库（确保记忆表等已创建）
    init_db()

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

    # 注意：对于个人设备，隐私保护默认禁用（用户已在设置时授权）
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
        enable_privacy_protection=False,  # 个人设备无需每次请求同意
    )
    vision_service.start()

    # 加载电机配置(包含健康监控设置)
    motor_config = load_motor_config()

    logger.info(
        "config ready: lamp_id=%s port=%s vision=%s camera=%s motor_health_check=%s",
        config.lamp_id,
        config.lamp_port,
        config.vision_enabled,
        config.camera_index_or_path,
        motor_config.health_check_enabled,
    )

    # 创建 LeLamp 代理实例
    agent = LeLamp(
        port=config.lamp_port,
        lamp_id=config.lamp_id,
        vision_service=vision_service,
        qwen_client=qwen_client,
        ota_url=config.ota_url,
        motors_service=None,  # 使用默认创建的服务
        rgb_service=None,  # 使用默认创建的服务
        motor_config=motor_config,
    )

    # 注入记忆上下文到 system prompt
    if agent._memory_initialized:
        import uuid as _uuid

        agent._session_id = str(_uuid.uuid4())[:8]
        await agent.update_instructions(agent._build_dynamic_instructions())
        logger.info(f"Memory context injected, session_id={agent._session_id}")

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
        # 使用新的 turn_handling API (LiveKit 1.5+)
        # adaptive: ML 模型区分真正的打断 vs 假阳性（咳嗽、背景音等）
        # dynamic: 自适应沉默阈值，根据对话节奏动态调整
        turn_handling={
            "interruption": {"mode": "adaptive"},
            "endpointing": {"mode": "dynamic"},
        },
    )

    start_kwargs: dict[str, object] = {}
    if config.noise_cancellation_enabled:
        start_kwargs["room_input_options"] = RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        )

    # 注册 Data Channel 事件监听器 (Web Client v2.0 support)
    @ctx.room.on("data_received")
    def on_data_received(data_packet):
        """处理来自 Web Client 的 Data Channel 消息"""
        # 使用 agent 的 _track_task 方法追踪后台任务
        agent._track_task(
            agent.handle_data_message(data_packet.data, data_packet.participant)
        )

    try:
        await session.start(
            agent=agent,
            room=ctx.room,
            **start_kwargs,
        )
        if config.greeting_text:
            # greeting 期间禁用打断
            await session.say(config.greeting_text, allow_interruptions=False)
    except Exception as e:
        logger.error(f"Session error: {e}")
    finally:
        # 注意：不再停止 vision_service，因为它应该在进程生命周期中持续运行
        # 当进程退出时，vision_service 线程会自动终止
        logger.info("Session ended")


if __name__ == "__main__":
    _setup_logging()

    # 商业化保护：启动时校验设备授权
    if not verify_license():
        logger.fatal("设备授权校验失败。请检查 LELAMP_LICENSE_KEY 配置。")
        sys.exit(1)

    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
