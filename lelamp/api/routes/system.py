"""
系统管理 API 路由

提供 WiFi 配置、系统信息、服务重启、首次配置设置等功能
"""
import asyncio
import signal
import os
import logging
import socket
import re
from datetime import datetime, timedelta, UTC
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from lelamp.database.session import get_db
from lelamp.api.services.wifi_manager import wifi_manager, WiFiNetwork
from lelamp.api.services.ap_manager import ap_manager, ClientInfo
from lelamp.api.services.onboarding import onboarding_manager
from lelamp.api.models.requests import WiFiConnectRequest, RestartRequest
from lelamp.api.models.responses import (
    WiFiNetworkListResponse,
    WiFiNetworkInfo,
    WiFiStatusResponse,
    WiFiConnectResponse,
    RestartResponse,
    SystemInfoResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# 重启管理器状态
_restart_task: asyncio.Task | None = None
_restart_scheduled_at: datetime | None = None


# ============================================================================
# 设置模式端点（Setup Mode / Onboarding）
# ============================================================================

class APStartRequest(BaseModel):
    """启动 AP 模式请求"""
    ssid: Optional[str] = Field(None, min_length=1, max_length=32, description="热点 SSID (1-32 字符)")
    password: Optional[str] = Field(None, min_length=8, max_length=63, description="热点密码 (8-63 字符)")
    channel: Optional[int] = Field(None, ge=1, le=14, description="WiFi 频道 (1-14)")

    @field_validator('ssid', 'password')
    @classmethod
    def validate_no_special_chars(cls, v: Optional[str], info) -> Optional[str]:
        if v is not None:
            # 不允许引号和反斜杠，避免配置文件解析问题
            if any(c in v for c in '"\\\''):
                raise ValueError("不能包含引号或反斜杠")
        return v


class SetupCompleteRequest(BaseModel):
    """完成配置请求"""
    wifi_ssid: str = Field(..., min_length=1, max_length=32, description="配置的 WiFi SSID")
    restart_delay: int = Field(default=5, ge=0, le=60, description="重启延迟秒数 (0-60)")


@router.get("/setup/status")
async def get_setup_status() -> dict:
    """
    获取设备配置状态

    Returns:
        配置状态字典，包含:
        - is_configured: 是否已完成配置
        - configured_wifi: 配置的 WiFi SSID
        - current_mode: 当前模式 (ap/client)
        - needs_setup: 是否需要进入设置模式
        - can_exit_setup: 是否可以退出设置模式
    """
    try:
        status = await onboarding_manager.get_configuration_summary()

        # 检查是否在 AP 模式
        is_ap_mode = await ap_manager.is_in_ap_mode()
        status["is_ap_mode"] = is_ap_mode

        # 获取已连接的客户端（AP 模式下）
        if is_ap_mode:
            clients = await ap_manager.get_connected_clients()
            status["connected_clients"] = [
                {"mac": c.mac, "ip": c.ip} for c in clients
            ]
        else:
            status["connected_clients"] = []

        return status
    except Exception as e:
        logger.error(f"Get setup status error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取配置状态失败: {str(e)}"
        )


@router.post("/setup/ap/start")
async def start_ap_mode(request: APStartRequest = APStartRequest()) -> dict:
    """
    启动 AP 模式（设置模式）

    创建 WiFi 热点供用户连接配置

    Args:
        request: AP 配置请求（可选）

    Returns:
        操作结果，包含 SSID、密码、IP 地址
    """
    try:
        # 检查是否已在运行
        if await ap_manager.is_in_ap_mode():
            return {
                "success": True,
                "message": "AP 模式已在运行",
                "ssid": "LeLamp-Setup",
                "password": "lelamp123",
                "ip_address": "192.168.4.1"
            }

        # 标记进入 AP 模式
        await onboarding_manager.mark_ap_mode_entered()

        # 启动 AP 模式
        result = await ap_manager.start_ap_mode()

        if result.get("success"):
            logger.info("AP mode started successfully")
        else:
            logger.error(f"Failed to start AP mode: {result.get('message')}")

        return result
    except Exception as e:
        logger.error(f"Start AP mode error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"启动 AP 模式失败: {str(e)}"
        )


@router.post("/setup/ap/stop")
async def stop_ap_mode() -> dict:
    """
    停止 AP 模式

    Returns:
        操作结果
    """
    try:
        success = await ap_manager.stop_ap_mode()
        return {
            "success": success,
            "message": "AP 模式已停止" if success else "停止 AP 模式失败"
        }
    except Exception as e:
        logger.error(f"Stop AP mode error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"停止 AP 模式失败: {str(e)}"
        )


@router.post("/setup/complete")
async def complete_setup(request: SetupCompleteRequest) -> dict:
    """
    完成配置并重启

    Args:
        request: 配置完成请求，包含 WiFi SSID 和重启延迟

    Returns:
        操作结果
    """
    try:
        global _restart_task, _restart_scheduled_at

        # 1. 标记配置完成
        await onboarding_manager.mark_setup_complete(request.wifi_ssid)

        # 2. 停止 AP 模式
        await ap_manager.stop_ap_mode()

        # 3. 安排重启
        restart_at = datetime.now(UTC) + timedelta(seconds=request.restart_delay)
        _restart_scheduled_at = restart_at
        _restart_task = asyncio.create_task(
            _execute_restart(request.restart_delay, "Setup completed")
        )

        logger.info(f"Setup completed for WiFi: {request.wifi_ssid}, restarting in {request.restart_delay}s")

        return {
            "success": True,
            "message": "配置完成，设备即将重启",
            "restart_at": restart_at.isoformat(),
            "delay_seconds": request.restart_delay
        }
    except Exception as e:
        logger.error(f"Complete setup error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"完成配置失败: {str(e)}"
        )


@router.post("/setup/reset")
async def reset_setup() -> dict:
    """
    重置配置（强制重新配置）

    Returns:
        操作结果
    """
    try:
        success = await onboarding_manager.mark_setup_required()
        return {
            "success": success,
            "message": "配置已重置，设备将进入设置模式"
        }
    except Exception as e:
        logger.error(f"Reset setup error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"重置配置失败: {str(e)}"
        )


@router.get("/setup/ap/clients")
async def get_ap_clients() -> dict:
    """
    获取 AP 模式下已连接的客户端

    Returns:
        客户端列表
    """
    try:
        clients = await ap_manager.get_connected_clients()
        return {
            "clients": [
                {"mac": c.mac, "ip": c.ip, "connected_at": c.connected_at}
                for c in clients
            ],
            "total": len(clients)
        }
    except Exception as e:
        logger.error(f"Get AP clients error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取客户端列表失败: {str(e)}"
        )


# ============================================================================
# WiFi 端点
# ============================================================================

@router.get("/wifi/status", response_model=WiFiStatusResponse)
async def get_wifi_status() -> WiFiStatusResponse:
    """
    获取当前 WiFi 连接状态

    Returns:
        WiFiStatusResponse: 当前连接状态信息
    """
    try:
        status_data = await wifi_manager.get_status()
        return WiFiStatusResponse(**status_data)
    except Exception as e:
        logger.error(f"Get WiFi status error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取 WiFi 状态失败: {str(e)}"
        )


@router.get("/wifi/scan", response_model=WiFiNetworkListResponse)
async def scan_wifi_networks() -> WiFiNetworkListResponse:
    """
    扫描可用 WiFi 网络

    需要 sudo 权限执行 nmcli 命令

    Returns:
        WiFiNetworkListResponse: 扫描到的网络列表
    """
    try:
        networks = await wifi_manager.scan_networks()

        # 转换为响应模型
        network_infos = [
            WiFiNetworkInfo(
                ssid=n.ssid,
                bssid=n.bssid,
                signal_strength=n.signal_strength,
                security=n.security,
                frequency=n.frequency,
                is_hidden=n.is_hidden
            )
            for n in networks
        ]

        return WiFiNetworkListResponse(
            networks=network_infos,
            scan_time=datetime.now(UTC),
            total=len(network_infos)
        )
    except Exception as e:
        logger.error(f"WiFi scan error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"WiFi 扫描失败: {str(e)}"
        )


@router.post("/wifi/connect", response_model=WiFiConnectResponse)
async def connect_wifi(request: WiFiConnectRequest) -> WiFiConnectResponse:
    """
    连接到指定 WiFi 网络

    需要 sudo 权限执行 nmcli 命令

    Args:
        request: WiFi 连接请求，包含 SSID 和密码

    Returns:
        WiFiConnectResponse: 连接结果
    """
    try:
        result = await wifi_manager.connect(request.ssid, request.password)
        return WiFiConnectResponse(**result)
    except Exception as e:
        logger.error(f"WiFi connect error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"WiFi 连接失败: {str(e)}"
        )


@router.delete("/wifi/disconnect")
async def disconnect_wifi() -> dict:
    """
    断开当前 WiFi 连接

    需要 sudo 权限执行 nmcli 命令

    Returns:
        操作结果
    """
    try:
        success = await wifi_manager.disconnect()
        return {"success": success, "message": "已断开连接" if success else "断开连接失败"}
    except Exception as e:
        logger.error(f"WiFi disconnect error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"断开 WiFi 连接失败: {str(e)}"
        )


# ============================================================================
# 系统信息端点
# ============================================================================

@router.get("/info", response_model=SystemInfoResponse)
async def get_system_info() -> SystemInfoResponse:
    """
    获取系统信息

    Returns:
        SystemInfoResponse: 系统信息
    """
    try:
        import psutil
        import platform

        # WiFi 可用性检查
        wifi_available = await wifi_manager.check_available()

        return SystemInfoResponse(
            hostname=socket.gethostname(),
            uptime_seconds=int(psutil.boot_time()),
            cpu_usage_percent=psutil.cpu_percent(interval=0.1),
            memory_usage_percent=psutil.virtual_memory().percent,
            disk_usage_percent=psutil.disk_usage('/').percent,
            wifi_available=wifi_available
        )
    except ImportError:
        # psutil 未安装，返回基本信息
        return SystemInfoResponse(
            hostname=socket.gethostname(),
            uptime_seconds=0,
            cpu_usage_percent=0.0,
            memory_usage_percent=0.0,
            disk_usage_percent=0.0,
            wifi_available=False
        )
    except Exception as e:
        logger.error(f"Get system info error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取系统信息失败: {str(e)}"
        )


# ============================================================================
# 重启端点
# ============================================================================

async def _execute_restart(delay: int, reason: str | None):
    """执行重启的内部协程"""
    await asyncio.sleep(delay)

    logger.info(f"Restarting service. Reason: {reason or 'Manual restart'}")

    # 发送 SIGTERM 给自身，让 systemd 重启服务
    try:
        os.kill(os.getpid(), signal.SIGTERM)
    except Exception as e:
        logger.error(f"Failed to send SIGTERM: {e}")


@router.post("/restart", response_model=RestartResponse)
async def trigger_restart(request: RestartRequest) -> RestartResponse:
    """
    触发服务重启

    重启会延迟执行，给客户端时间响应。
    建议前端在收到响应后显示倒计时。

    Args:
        request: 重启请求，包含延迟秒数和原因

    Returns:
        RestartResponse: 重排信息
    """
    global _restart_task, _restart_scheduled_at

    if _restart_task is not None and not _restart_task.done():
        return RestartResponse(
            scheduled=False,
            restart_at=_restart_scheduled_at or datetime.now(UTC),
            delay_seconds=0,
            message="重启已在进行中"
        )

    restart_at = datetime.now(UTC) + timedelta(seconds=request.delay_seconds)
    _restart_scheduled_at = restart_at

    # 创建延迟重启任务
    _restart_task = asyncio.create_task(
        _execute_restart(request.delay_seconds, request.reason)
    )

    return RestartResponse(
        scheduled=True,
        restart_at=restart_at,
        delay_seconds=request.delay_seconds,
        message=f"服务将在 {request.delay_seconds} 秒后重启"
    )


@router.post("/restart/cancel")
async def cancel_restart() -> dict:
    """
    取消计划的重启

    Returns:
        操作结果
    """
    global _restart_task, _restart_scheduled_at

    if _restart_task is None or _restart_task.done():
        return {"success": False, "message": "没有正在进行的重启"}

    _restart_task.cancel()
    _restart_task = None
    _restart_scheduled_at = None

    return {"success": True, "message": "已取消重启"}
