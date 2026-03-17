"""
API 请求模型
"""
from pydantic import BaseModel, Field
from typing import Optional

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
