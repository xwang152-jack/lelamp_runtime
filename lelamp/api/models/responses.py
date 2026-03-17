"""
API 响应模型
"""
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime


class DeviceStateResponse(BaseModel):
    """设备状态响应"""
    lamp_id: str
    status: str
    conversation_state: str
    timestamp: datetime
    motor_positions: Dict[str, Any]
    light_color: Dict[str, Any]
    camera_active: bool
    uptime_seconds: Optional[int] = None


class ConversationResponse(BaseModel):
    """对话记录响应"""
    id: int
    timestamp: datetime
    lamp_id: str
    user_input: Optional[str]
    ai_response: Optional[str]
    duration: Optional[int]
    messages: List[Dict[str, Any]]


class OperationResponse(BaseModel):
    """操作日志响应"""
    id: int
    timestamp: datetime
    lamp_id: str
    operation_type: str
    action: str
    params: Dict[str, Any]
    success: bool
    error_message: Optional[str]
    duration_ms: Optional[int]


class HealthResponse(BaseModel):
    """健康状态响应"""
    lamp_id: str
    overall_status: str
    motors: List[Dict[str, Any]]
    last_check: datetime


class ConversationListResponse(BaseModel):
    """对话列表响应"""
    total: int
    conversations: List[ConversationResponse]


class OperationListResponse(BaseModel):
    """操作日志列表响应"""
    total: int
    operations: List[OperationResponse]


class DeviceListResponse(BaseModel):
    """设备列表响应"""
    devices: List["DeviceInfoResponse"]


class DeviceInfoResponse(BaseModel):
    """设备信息响应"""
    lamp_id: str
    last_seen: Optional[datetime]
    state: Optional[str]  # idle/listening/thinking/speaking


class CommandResponse(BaseModel):
    """命令执行响应"""
    success: bool
    command_id: str
    message: str
    timestamp: datetime


class StatisticsResponse(BaseModel):
    """设备统计响应"""
    lamp_id: str
    period_days: int
    total_operations: int
    success_rate: float
    operation_counts: Dict[str, int]
    avg_duration_ms: Optional[float]
    most_common_operation: str


# 更新前向引用
DeviceListResponse.model_rebuild()
