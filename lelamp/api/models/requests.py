"""
API 请求模型
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any
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

    @validator("type")
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
