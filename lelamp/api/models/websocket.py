"""
WebSocket 消息模型

定义 WebSocket 通信中使用的消息类型和格式。
"""
from pydantic import BaseModel, Field
from typing import Literal, Optional, Any, Dict, List
from datetime import datetime


# =============================================================================
# 基础消息模型
# =============================================================================


class WSMessage(BaseModel):
    """WebSocket 基础消息"""
    type: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# 客户端消息类型
# =============================================================================


class WSPing(BaseModel):
    """客户端心跳消息"""
    type: Literal["ping"]


class WSPong(BaseModel):
    """服务端心跳响应"""
    type: Literal["pong"]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class WSSubscribe(BaseModel):
    """订阅频道"""
    type: Literal["subscribe"]
    channels: List[str] = Field(
        ...,
        description="订阅的频道列表: state, events, logs, notifications"
    )


class WSUnsubscribe(BaseModel):
    """取消订阅"""
    type: Literal["unsubscribe"]
    channels: List[str] = Field(
        ...,
        description="取消订阅的频道列表"
    )


class WSClientCommand(BaseModel):
    """客户端命令"""
    type: Literal["command"]
    action: str
    params: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# 服务端消息类型
# =============================================================================


class WSSubscriptionConfirmed(BaseModel):
    """订阅确认"""
    type: Literal["subscription_confirmed"]
    channels: List[str]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class WSStateUpdate(BaseModel):
    """状态更新消息"""
    type: Literal["state_update"]
    data: Dict[str, Any] = Field(
        ...,
        description="设备状态数据"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class WSEvent(BaseModel):
    """事件消息"""
    type: Literal["event"]
    event_type: str = Field(
        ...,
        description="事件类型: motor_move, rgb_set, vision_capture, etc."
    )
    data: Dict[str, Any] = Field(
        ...,
        description="事件数据"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class WSLog(BaseModel):
    """日志消息"""
    type: Literal["log"]
    log_entry: Dict[str, Any] = Field(
        ...,
        description="日志条目"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class WSNotification(BaseModel):
    """通知消息"""
    type: Literal["notification"]
    message: str
    level: Literal["info", "warning", "error"] = "info"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = None


class WSError(BaseModel):
    """错误消息"""
    type: Literal["error"]
    message: str
    code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# 聊天和对话相关消息
# =============================================================================


class WSConversationUpdate(BaseModel):
    """对话更新消息"""
    type: Literal["conversation_update"]
    data: Dict[str, Any] = Field(
        ...,
        description="对话数据"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class WSUserMessage(BaseModel):
    """用户消息"""
    type: Literal["user_message"]
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AIMessage(BaseModel):
    """AI 响应消息"""
    type: Literal["ai_message"]
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# 摄像头相关消息
# =============================================================================


class WSCameraFrame(BaseModel):
    """摄像头帧消息"""
    type: Literal["camera_frame"]
    frame_b64: str = Field(
        ...,
        description="Base64 编码的 JPEG 图像数据"
    )
    width: Optional[int] = Field(None, description="图像宽度")
    height: Optional[int] = Field(None, description="图像高度")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class WSCameraStatus(BaseModel):
    """摄像头状态消息"""
    type: Literal["camera_status"]
    active: bool = Field(
        ...,
        description="摄像头是否激活"
    )
    privacy_granted: Optional[bool] = Field(None, description="隐私同意状态")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# 健康和诊断相关消息
# =============================================================================


class WSHealthUpdate(BaseModel):
    """健康状态更新"""
    type: Literal["health_update"]
    data: Dict[str, Any] = Field(
        ...,
        description="健康状态数据"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class WSWarning(BaseModel):
    """警告消息"""
    type: Literal["warning"]
    message: str
    warning_type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# 连接状态消息
# =============================================================================


class WSConnected(BaseModel):
    """连接确认"""
    type: Literal["connected"]
    lamp_id: str
    server_time: datetime = Field(default_factory=datetime.utcnow)
    message: str = "WebSocket connection established"


class WSDisconnected(BaseModel):
    """断开通知（仅用于服务端内部）"""
    type: Literal["disconnected"]
    lamp_id: str
    reason: Optional[str] = None


# =============================================================================
# 消息联合类型
# =============================================================================


ClientMessage = (
    WSPing |
    WSSubscribe |
    WSUnsubscribe |
    WSClientCommand
)

ServerMessage = (
    WSPong |
    WSSubscriptionConfirmed |
    WSStateUpdate |
    WSEvent |
    WSLog |
    WSNotification |
    WSConversationUpdate |
    WSHealthUpdate |
    WSWarning |
    WSError |
    WSConnected |
    WSCameraFrame |
    WSCameraStatus
)


# =============================================================================
# 验证辅助函数
# =============================================================================


def validate_client_message(data: dict) -> Optional[ClientMessage]:
    """
    验证客户端消息格式

    Args:
        data: 消息数据

    Returns:
        验证后的消息对象，如果无效则返回 None
    """
    message_type = data.get("type")

    try:
        if message_type == "ping":
            return WSPing(**data)
        elif message_type == "subscribe":
            return WSSubscribe(**data)
        elif message_type == "unsubscribe":
            return WSUnsubscribe(**data)
        elif message_type == "command":
            return WSClientCommand(**data)
        else:
            return None
    except Exception:
        return None


def is_valid_channel(channel: str) -> bool:
    """
    验证频道名称是否有效

    Args:
        channel: 频道名称

    Returns:
        是否有效
    """
    valid_channels = {
        "state",
        "events",
        "logs",
        "notifications",
        "conversations",
        "health"
    }
    return channel in valid_channels
