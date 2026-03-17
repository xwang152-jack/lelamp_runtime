"""
API 响应模型
"""
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class DeviceStateResponse(BaseModel):
    """设备状态响应"""
    lamp_id: str
    status: str
    timestamp: datetime
    motor_positions: dict
    light_color: dict
    camera_active: bool

class ConversationResponse(BaseModel):
    """对话记录响应"""
    id: int
    timestamp: datetime
    user_message: str
    agent_response: str
    duration_ms: int

class HealthResponse(BaseModel):
    """健康状态响应"""
    lamp_id: str
    overall_status: str
    motors: list
    last_check: datetime
