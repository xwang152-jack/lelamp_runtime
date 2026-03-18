"""
配置管理模块
集中管理所有环境变量和配置
"""

import os
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Any

# 加载 .env 文件
load_dotenv()


def _get_env_str(key: str, default: str | None = None) -> str | None:
    """获取字符串环境变量"""
    raw = os.getenv(key)
    if raw is None:
        return default
    value = raw.strip()
    return value if value != "" else default


def _get_env_bool(key: str, default: bool = False) -> bool:
    """获取布尔环境变量"""
    raw = _get_env_str(key, None)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


def _get_env_int(key: str, default: int) -> int:
    """获取整数环境变量"""
    raw = _get_env_str(key, None)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_env_float(key: str, default: float) -> float:
    """获取浮点数环境变量"""
    raw = _get_env_str(key, None)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _require_env(key: str) -> str:
    """获取必需的环境变量"""
    value = _get_env_str(key)
    if not value:
        raise RuntimeError(f"缺少环境变量：{key}")
    return value


def _parse_index_or_path(raw: str | None) -> int | str:
    """解析摄像头索引或路径"""
    if raw is None:
        return 0
    raw = raw.strip()
    if raw == "":
        return 0
    try:
        return int(raw)
    except ValueError:
        return raw


@dataclass(frozen=True)
class AppConfig:
    """应用配置"""
    # LiveKit
    livekit_url: str
    livekit_api_key: str
    livekit_api_secret: str

    # LLM
    deepseek_model: str
    deepseek_base_url: str
    deepseek_api_key: str

    # Vision
    modelscope_base_url: str
    modelscope_api_key: str | None
    modelscope_model: str
    modelscope_timeout_s: float
    vision_enabled: bool
    camera_index_or_path: int | str
    camera_width: int
    camera_height: int
    vision_capture_interval_s: float
    vision_jpeg_quality: int
    vision_max_age_s: float
    camera_rotate_deg: int
    camera_flip: str

    # Speech (Baidu)
    baidu_api_key: str
    baidu_secret_key: str
    baidu_cuid: str
    baidu_tts_per: int

    # Hardware
    lamp_port: str
    lamp_id: str

    # Features
    noise_cancellation_enabled: bool
    greeting_text: str

    # OTA (Over-The-Air Update)
    ota_url: str | None


@dataclass
class MotorConfig:
    """电机配置"""
    port: str
    lamp_id: str
    fps: int = 30
    # 健康监控配置
    health_check_enabled: bool = True
    health_check_interval_s: float = 300.0  # 5分钟检查一次
    temp_warning_c: float = 65.0            # 温度警告阈值
    temp_critical_c: float = 75.0           # 温度危险阈值
    voltage_min_v: float = 11.0             # 最低电压
    voltage_max_v: float = 13.0             # 最高电压
    load_warning: float = 0.8               # 负载警告阈值
    load_stall: float = 0.95                # 堵转阈值
    position_error_deg: float = 5.0         # 位置误差阈值


@dataclass
class RGBConfig:
    """RGB LED 配置"""
    led_count: int = 64
    led_pin: int = 12
    led_freq_hz: int = 800000
    led_dma: int = 10
    led_brightness: int = 25
    led_invert: bool = False
    led_channel: int = 0


@dataclass
class VisionConfig:
    """视觉服务配置"""
    enabled: bool = True
    index_or_path: int | str = 0
    width: int = 1024
    height: int = 768
    capture_interval_s: float = 2.5
    jpeg_quality: int = 92
    max_age_s: float = 15.0
    rotate_deg: int = 0
    flip: str = "none"
    enable_privacy_protection: bool = True


# 关节安全角度范围
SAFE_JOINT_RANGES = {
    "base_yaw": (-180, 180),
    "base_pitch": (-90, 90),
    "elbow_pitch": (-150, 150),
    "wrist_roll": (-180, 180),
    "wrist_pitch": (-90, 90),
}


def load_config() -> AppConfig:
    """加载应用配置"""
    return AppConfig(
        # LiveKit (不再强制要求)
        livekit_url=_get_env_str("LIVEKIT_URL", ""),
        livekit_api_key=_get_env_str("LIVEKIT_API_KEY", ""),
        livekit_api_secret=_get_env_str("LIVEKIT_API_SECRET", ""),

        # LLM
        deepseek_model=_get_env_str("DEEPSEEK_MODEL", "deepseek-chat") or "deepseek-chat",
        deepseek_base_url=_get_env_str("DEEPSEEK_BASE_URL", "https://api.deepseek.com") or "https://api.deepseek.com",
        deepseek_api_key=_get_env_str("DEEPSEEK_API_KEY", "dummy"),

        # Vision
        modelscope_base_url=_get_env_str("MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1") or "https://api-inference.modelscope.cn/v1",
        modelscope_api_key=_get_env_str("MODELSCOPE_API_KEY"),
        modelscope_model=_get_env_str("MODELSCOPE_MODEL", "Qwen/Qwen3-VL-235B-A22B-Instruct") or "Qwen/Qwen3-VL-235B-A22B-Instruct",
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

        # Speech
        baidu_api_key=_get_env_str("BAIDU_SPEECH_API_KEY"),
        baidu_secret_key=_get_env_str("BAIDU_SPEECH_SECRET_KEY"),
        baidu_cuid=_get_env_str("BAIDU_SPEECH_CUID", "lelamp") or "lelamp",
        baidu_tts_per=_get_env_int("BAIDU_SPEECH_TTS_PER", 4),

        # Hardware
        lamp_port=_get_env_str("LELAMP_PORT", "/dev/ttyACM0") or "/dev/ttyACM0",
        lamp_id=_get_env_str("LELAMP_ID", "lelamp") or "lelamp",

        # Features
        noise_cancellation_enabled=_get_env_bool("LELAMP_NOISE_CANCELLATION", True),
        greeting_text=_get_env_str("LELAMP_GREETING_TEXT", "你好！我是 LeLamp，你的智能台灯。") or "你好！我是 LeLamp，你的智能台灯。",

        # OTA (Over-The-Air Update)
        ota_url=_get_env_str("LELAMP_OTA_URL", "") or None,
    )


def load_motor_config() -> MotorConfig:
    """加载电机配置"""
    config = load_config()
    return MotorConfig(
        port=config.lamp_port,
        lamp_id=config.lamp_id,
        fps=30,
        # 健康监控配置
        health_check_enabled=_get_env_bool("LELAMP_MOTOR_HEALTH_CHECK_ENABLED", True),
        health_check_interval_s=_get_env_float("LELAMP_MOTOR_HEALTH_CHECK_INTERVAL_S", 300.0),
        temp_warning_c=_get_env_float("LELAMP_MOTOR_TEMP_WARNING_C", 65.0),
        temp_critical_c=_get_env_float("LELAMP_MOTOR_TEMP_CRITICAL_C", 75.0),
        voltage_min_v=_get_env_float("LELAMP_MOTOR_VOLTAGE_MIN_V", 11.0),
        voltage_max_v=_get_env_float("LELAMP_MOTOR_VOLTAGE_MAX_V", 13.0),
        load_warning=_get_env_float("LELAMP_MOTOR_LOAD_WARNING", 0.8),
        load_stall=_get_env_float("LELAMP_MOTOR_LOAD_STALL", 0.95),
        position_error_deg=_get_env_float("LELAMP_MOTOR_POSITION_ERROR_DEG", 5.0),
    )


def load_rgb_config() -> RGBConfig:
    """加载 RGB 配置"""
    return RGBConfig()


def load_vision_config() -> VisionConfig:
    """加载视觉配置"""
    config = load_config()
    return VisionConfig(
        enabled=config.vision_enabled,
        index_or_path=config.camera_index_or_path,
        width=config.camera_width,
        height=config.camera_height,
        capture_interval_s=config.vision_capture_interval_s,
        jpeg_quality=config.vision_jpeg_quality,
        max_age_s=config.vision_max_age_s,
        rotate_deg=config.camera_rotate_deg,
        flip=config.camera_flip,
        enable_privacy_protection=True,
    )
