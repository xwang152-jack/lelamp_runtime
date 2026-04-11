"""
配置同步服务

负责在数据库和环境变量之间同步配置
"""
import logging
import os
from pathlib import Path
from typing import Optional, Any

from sqlalchemy.orm import Session

from lelamp.database.models import UserSettings
from lelamp.config import AppConfig

logger = logging.getLogger(__name__)


class ConfigSyncService:
    """
    配置同步服务

    将数据库中的配置同步到 .env 文件，使配置在重启后生效
    """

    # 配置字段映射：数据库字段 -> 环境变量名
    FIELD_ENV_MAP = {
        # LLM
        "deepseek_model": "DEEPSEEK_MODEL",
        "deepseek_base_url": "DEEPSEEK_BASE_URL",
        "deepseek_api_key": "DEEPSEEK_API_KEY",

        # Vision
        "vision_enabled": "LELAMP_VISION_ENABLED",
        "modelscope_model": "MODELSCOPE_MODEL",
        "modelscope_api_key": "MODELSCOPE_API_KEY",
        "modelscope_timeout_s": "MODELSCOPE_TIMEOUT_S",

        # Camera
        "camera_width": "LELAMP_CAMERA_WIDTH",
        "camera_height": "LELAMP_CAMERA_HEIGHT",
        "camera_rotate_deg": "LELAMP_CAMERA_ROTATE_DEG",
        "camera_flip": "LELAMP_CAMERA_FLIP",

        # Speech
        "baidu_tts_per": "BAIDU_SPEECH_TTS_PER",

        # Hardware
        "led_brightness": "LELAMP_LED_BRIGHTNESS",
        "lamp_port": "LELAMP_PORT",
        "lamp_id": "LELAMP_ID",

        # Behavior
        "greeting_text": "LELAMP_GREETING_TEXT",
        "noise_cancellation": "LELAMP_NOISE_CANCELLATION",
        "motion_cooldown_s": "LELAMP_MOTION_COOLDOWN_S",

        # Audio
        "volume_level": "LELAMP_VOLUME_LEVEL",
    }

    def __init__(self, env_file_path: Optional[Path] = None):
        """
        初始化配置同步服务

        Args:
            env_file_path: .env 文件路径，默认为项目根目录的 .env
        """
        if env_file_path is None:
            # 默认 .env 文件位置
            project_root = Path(__file__).parent.parent.parent.parent
            env_file_path = project_root / ".env"

        self.env_file_path = env_file_path

    def get_current_config(self, db: Session, lamp_id: str) -> dict:
        """
        获取当前配置（优先从数据库读取，回退到环境变量）

        Args:
            db: 数据库会话
            lamp_id: 设备 ID

        Returns:
            配置字典
        """
        settings = db.query(UserSettings).filter(
            UserSettings.lamp_id == lamp_id
        ).first()

        # 辅助函数：从环境变量获取默认值
        def get_env_bool(key: str, default: bool = False) -> bool:
            return os.getenv(key, str(default)).lower() in ("true", "1", "yes")

        # 构建配置字典
        config = {
            # UI Settings
            "theme": settings.theme if settings else "light",
            "language": settings.language if settings else "zh",
            "notifications_enabled": settings.notifications_enabled if settings else True,
            "brightness_level": settings.brightness_level if settings else 25,
            "volume_level": settings.volume_level if settings else int(os.getenv("LELAMP_VOLUME_LEVEL", "50")),

            # LLM
            "deepseek_model": settings.deepseek_model if settings and settings.deepseek_model else os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            "deepseek_base_url": settings.deepseek_base_url if settings and settings.deepseek_base_url else os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            "deepseek_api_key_configured": bool(
                (settings and settings.deepseek_api_key) or os.getenv("DEEPSEEK_API_KEY")
            ),
            "deepseek_api_key_masked": self._mask_api_key(
                settings.deepseek_api_key if settings else None
            ) or self._mask_api_key(os.getenv("DEEPSEEK_API_KEY")),

            # Vision
            "vision_enabled": settings.vision_enabled if settings else get_env_bool("LELAMP_VISION_ENABLED", True),
            "modelscope_model": settings.modelscope_model if settings and settings.modelscope_model else os.getenv("MODELSCOPE_MODEL", "Qwen/Qwen3-VL-235B-A22B-Instruct"),
            "modelscope_api_key_configured": bool(
                (settings and settings.modelscope_api_key) or os.getenv("MODELSCOPE_API_KEY")
            ),
            "modelscope_api_key_masked": self._mask_api_key(
                settings.modelscope_api_key if settings else None
            ) or self._mask_api_key(os.getenv("MODELSCOPE_API_KEY")),
            "modelscope_timeout_s": settings.modelscope_timeout_s if settings else float(os.getenv("MODELSCOPE_TIMEOUT_S", "60.0")),

            # Camera
            "camera_width": settings.camera_width if settings else int(os.getenv("LELAMP_CAMERA_WIDTH", "1024")),
            "camera_height": settings.camera_height if settings else int(os.getenv("LELAMP_CAMERA_HEIGHT", "768")),
            "camera_rotate_deg": settings.camera_rotate_deg if settings else int(os.getenv("LELAMP_CAMERA_ROTATE_DEG", "0")),
            "camera_flip": settings.camera_flip if settings else os.getenv("LELAMP_CAMERA_FLIP", "none"),

            # Speech
            "baidu_tts_per": settings.baidu_tts_per if settings else int(os.getenv("BAIDU_SPEECH_TTS_PER", "4")),

            # Hardware
            "led_brightness": settings.led_brightness if settings else int(os.getenv("LELAMP_LED_BRIGHTNESS", "25")),
            "lamp_port": settings.lamp_port if settings else os.getenv("LELAMP_PORT", "/dev/ttyACM0"),
            "lamp_id": settings.lamp_id if settings else os.getenv("LELAMP_ID", "lelamp"),

            # Behavior
            "greeting_text": settings.greeting_text if settings and settings.greeting_text else os.getenv("LELAMP_GREETING_TEXT", "你好！我是 LeLamp，你的智能台灯。"),
            "noise_cancellation": settings.noise_cancellation if settings else get_env_bool("LELAMP_NOISE_CANCELLATION", True),
            "motion_cooldown_s": settings.motion_cooldown_s if settings else float(os.getenv("LELAMP_MOTION_COOLDOWN_S", "2.0")),

            # Edge Vision (新增)
            "edge_vision_enabled": settings.edge_vision_enabled if settings else get_env_bool("LELAMP_EDGE_VISION_ENABLED", False),
            "edge_vision_prefer_local": settings.edge_vision_prefer_local if settings else get_env_bool("LELAMP_EDGE_VISION_PREFER_LOCAL", True),
            "edge_vision_local_threshold": settings.edge_vision_local_threshold if settings else float(os.getenv("LELAMP_EDGE_VISION_CONFIDENCE_THRESHOLD", "0.7")),

            # Metadata
            "requires_restart": False,
            "last_updated": settings.updated_at.isoformat() if settings and settings.updated_at else None,
        }

        # 检查是否需要重启（数据库配置与环境变量不一致）
        config["requires_restart"] = self._check_requires_restart(db, lamp_id)

        return config

    def update_settings(
        self,
        db: Session,
        lamp_id: str,
        updates: dict
    ) -> tuple[dict, bool]:
        """
        更新配置

        Args:
            db: 数据库会话
            lamp_id: 设备 ID
            updates: 更新的字段字典

        Returns:
            (更新后的配置, 是否需要重启)
        """
        # 获取或创建设置
        settings = db.query(UserSettings).filter(
            UserSettings.lamp_id == lamp_id
        ).first()

        if settings is None:
            settings = UserSettings(lamp_id=lamp_id)
            db.add(settings)

        # 更新字段
        for key, value in updates.items():
            if hasattr(settings, key):
                setattr(settings, key, value)

        db.commit()
        db.refresh(settings)

        # 将配置写入 .env 文件（下次启动生效）
        requires_restart = self._sync_to_env_file(updates)

        return self.get_current_config(db, lamp_id), requires_restart

    def _sync_to_env_file(self, updates: dict) -> bool:
        """
        将配置更新同步到 .env 文件

        Args:
            updates: 更新的字段字典

        Returns:
            是否需要重启服务
        """
        if not self.env_file_path.exists():
            logger.warning(f"Env file not found: {self.env_file_path}")
            # 如果 .env 不存在，创建一个
            self.env_file_path.write_text("")

        # 读取现有 .env 内容
        lines = self.env_file_path.read_text().splitlines()
        env_dict = {}
        comments = []

        for line in lines:
            stripped = line.strip()
            # 保存注释
            if stripped.startswith('#'):
                comments.append(line)
                continue
            if stripped and '=' in stripped:
                key, value = stripped.split('=', 1)
                env_dict[key.strip()] = value.strip()

        # 更新环境变量
        requires_restart = False
        for field, value in updates.items():
            if field in self.FIELD_ENV_MAP:
                env_key = self.FIELD_ENV_MAP[field]

                # 转换值类型
                if isinstance(value, bool):
                    value = "1" if value else "0"
                elif isinstance(value, str):
                    # 字符串值不需要引号，但保留空格
                    pass
                else:
                    value = str(value)

                env_dict[env_key] = value
                requires_restart = True

        # 写回文件
        output_lines = []
        # 添加注释
        for comment in comments:
            output_lines.append(comment)
        # 添加环境变量
        for key, value in env_dict.items():
            output_lines.append(f"{key}={value}")

        self.env_file_path.write_text('\n'.join(output_lines) + '\n')
        logger.info(f"Synced {len(updates)} settings to {self.env_file_path}")

        return requires_restart

    def _check_requires_restart(self, db: Session, lamp_id: str) -> bool:
        """
        检查配置是否与运行时配置一致

        Args:
            db: 数据库会话
            lamp_id: 设备 ID

        Returns:
            是否需要重启
        """
        # 简化实现：如果数据库中有任何自定义配置，则认为需要重启
        # 可以扩展为比较数据库配置与实际运行时环境变量
        settings = db.query(UserSettings).filter(
            UserSettings.lamp_id == lamp_id
        ).first()

        if not settings:
            return False

        # 检查是否有任何非默认配置
        non_default_fields = [
            "deepseek_model", "deepseek_base_url", "deepseek_api_key",
            "modelscope_model", "modelscope_api_key",
            "greeting_text"
        ]

        for field in non_default_fields:
            value = getattr(settings, field, None)
            if value is not None:
                return True

        return False

    def _mask_api_key(self, api_key: Optional[str]) -> Optional[str]:
        """
        隐藏 API Key 的敏感部分

        Args:
            api_key: 原始 API Key

        Returns:
            隐藏后的 API Key (如: sk-***abc123)
        """
        if not api_key:
            return None

        api_key_str = str(api_key)
        if len(api_key_str) <= 8:
            return "***"

        # 保留前3个和后4个字符
        return f"{api_key_str[:3]}***{api_key_str[-4:]}"


# 全局单例实例
config_sync_service = ConfigSyncService()
