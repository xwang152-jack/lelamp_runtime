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


# ============================================================================
# WiFi 响应模型
# ============================================================================

class WiFiNetworkInfo(BaseModel):
    """WiFi 网络信息"""
    ssid: str
    bssid: str
    signal_strength: int  # 0-100
    security: str  # WPA2, WPA3, WEP, open
    frequency: str  # 2.4GHz / 5GHz
    is_hidden: bool


class WiFiNetworkListResponse(BaseModel):
    """WiFi 网络列表响应"""
    networks: List[WiFiNetworkInfo]
    scan_time: datetime
    total: int


class WiFiStatusResponse(BaseModel):
    """WiFi 状态响应"""
    connected: bool
    ssid: Optional[str]
    signal_strength: Optional[int]
    ip_address: Optional[str]
    gateway: Optional[str]
    dns_servers: List[str]


class WiFiConnectResponse(BaseModel):
    """WiFi 连接响应"""
    success: bool
    message: str
    ssid: str


# ============================================================================
# 设置响应模型
# ============================================================================

class AppSettingsResponse(BaseModel):
    """应用配置响应"""
    # UI
    theme: str
    language: str
    notifications_enabled: bool
    brightness_level: int
    volume_level: int

    # LLM
    deepseek_model: str
    deepseek_base_url: str
    deepseek_api_key_configured: bool
    deepseek_api_key_masked: Optional[str]

    # Vision
    vision_enabled: bool
    modelscope_model: str
    modelscope_api_key_configured: bool
    modelscope_api_key_masked: Optional[str]
    modelscope_timeout_s: float

    # Edge Vision 边缘视觉
    edge_vision_enabled: bool
    edge_vision_prefer_local: bool
    edge_vision_local_threshold: float

    # Camera
    camera_width: int
    camera_height: int
    camera_rotate_deg: int
    camera_flip: str

    # Speech
    baidu_tts_per: int

    # Hardware
    led_brightness: int
    lamp_port: str
    lamp_id: str

    # Behavior
    greeting_text: str
    noise_cancellation: bool
    motion_cooldown_s: float

    # Metadata
    requires_restart: bool
    last_updated: Optional[datetime]


# ============================================================================
# 系统响应模型
# ============================================================================

class RestartResponse(BaseModel):
    """重启响应"""
    scheduled: bool
    restart_at: datetime
    delay_seconds: int
    message: str


class SystemInfoResponse(BaseModel):
    """系统信息响应"""
    hostname: str
    uptime_seconds: int
    cpu_usage_percent: float
    memory_usage_percent: float
    disk_usage_percent: float
    wifi_available: bool


# ============================================================================
# 设置模式响应模型 (Setup Mode / Onboarding)
# ============================================================================

class SetupStatusResponse(BaseModel):
    """设置状态响应"""
    is_configured: bool  # 是否已完成配置
    configured_wifi: Optional[str]  # 配置的 WiFi SSID
    current_mode: str  # 当前模式 (ap/client/unknown)
    needs_setup: bool  # 是否需要进入设置模式
    can_exit_setup: bool  # 是否可以退出设置模式
    is_ap_mode: bool  # 是否当前在 AP 模式
    connected_clients: List[Dict[str, Any]]  # AP 模式下已连接的客户端


class APModeStartResponse(BaseModel):
    """AP 模式启动响应"""
    success: bool
    message: str
    ssid: str  # 热点 SSID
    password: str  # 热点密码
    ip_address: str  # AP IP 地址


class SetupCompleteResponse(BaseModel):
    """配置完成响应"""
    success: bool
    message: str
    restart_at: str  # ISO 格式重启时间
    delay_seconds: int  # 延迟秒数


class APClientsResponse(BaseModel):
    """AP 客户端列表响应"""
    clients: List[Dict[str, Any]]
    total: int


# 更新前向引用
DeviceListResponse.model_rebuild()
