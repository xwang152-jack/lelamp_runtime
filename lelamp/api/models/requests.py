"""
API 请求模型
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any, Literal
import uuid


class MotorControlRequest(BaseModel):
    """电机控制请求"""
    joint_name: str = Field(..., description="关节名称")
    position: float = Field(..., description="目标位置")
    speed: Optional[int] = Field(50, description="移动速度")


class RGBColorRequest(BaseModel):
    """RGB 颜色控制请求"""
    r: int = Field(..., ge=0, le=255, description="红色值 (0-255)")
    g: int = Field(..., ge=0, le=255, description="绿色值 (0-255)")
    b: int = Field(..., ge=0, le=255, description="蓝色值 (0-255)")


class CommandRequest(BaseModel):
    """设备命令请求"""
    type: str = Field(..., description="命令类型", min_length=1, max_length=50)
    action: str = Field(..., description="命令动作", min_length=1, max_length=100)
    params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="命令参数")

    @field_validator("type")
    @classmethod
    def validate_command_type(cls, v):
        """验证命令类型"""
        # 允许的命令类型
        allowed_types = {
            "motor_move",
            "rgb_set",
            "rgb_effect",
            "vision_capture",
            "play_recording",
            "set_volume",
            "system_command",
        }
        if v not in allowed_types:
            # 不阻止未知类型，但记录警告
            pass
        return v


# ============================================================================
# WiFi 请求模型
# ============================================================================

class WiFiConnectRequest(BaseModel):
    """WiFi 连接请求"""
    ssid: str = Field(..., min_length=1, max_length=100, description="网络名称")
    password: Optional[str] = Field(None, max_length=200, description="WiFi 密码（开放网络可省略）")
    hidden: bool = Field(False, description="是否为隐藏网络")


# ============================================================================
# 设置请求模型
# ============================================================================

class AppSettingsUpdateRequest(BaseModel):
    """应用配置更新请求"""
    # LLM 配置
    deepseek_model: Optional[str] = Field(None, max_length=100)
    deepseek_base_url: Optional[str] = Field(None, max_length=200)
    deepseek_api_key: Optional[str] = Field(None, min_length=1, max_length=200)

    # Vision 配置
    vision_enabled: Optional[bool] = None
    modelscope_model: Optional[str] = Field(None, max_length=200)
    modelscope_api_key: Optional[str] = Field(None, min_length=1, max_length=200)
    modelscope_timeout_s: Optional[float] = Field(None, ge=1.0, le=300.0)

    # Edge Vision 边缘视觉配置
    edge_vision_enabled: Optional[bool] = None
    edge_vision_prefer_local: Optional[bool] = None
    edge_vision_local_threshold: Optional[float] = Field(None, ge=0.1, le=1.0)

    # 摄像头配置
    camera_width: Optional[int] = Field(None, ge=320, le=1920)
    camera_height: Optional[int] = Field(None, ge=240, le=1080)
    camera_rotate_deg: Optional[int] = Field(None, ge=0, le=360)
    camera_flip: Optional[Literal["none", "horizontal", "vertical", "both"]] = None

    # 语音配置
    baidu_tts_per: Optional[int] = Field(None, ge=0, le=500)

    # 硬件配置
    led_brightness: Optional[int] = Field(None, ge=0, le=100)
    lamp_port: Optional[str] = Field(None, max_length=50)
    lamp_id: Optional[str] = Field(None, min_length=1, max_length=50)

    # 行为配置
    greeting_text: Optional[str] = Field(None, max_length=500)
    noise_cancellation: Optional[bool] = None
    motion_cooldown_s: Optional[float] = Field(None, ge=0.5, le=30.0)

    # UI 配置
    theme: Optional[str] = Field(None, max_length=20)
    language: Optional[str] = Field(None, max_length=10)
    notifications_enabled: Optional[bool] = None
    brightness_level: Optional[int] = Field(None, ge=0, le=100)
    volume_level: Optional[int] = Field(None, ge=0, le=100)


# ============================================================================
# 系统请求模型
# ============================================================================

class RestartRequest(BaseModel):
    """重启请求"""
    delay_seconds: int = Field(3, ge=0, le=60, description="延迟重启秒数")
    reason: Optional[str] = Field(None, max_length=200, description="重启原因")
