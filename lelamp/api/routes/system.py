"""
系统管理 API 路由

提供 WiFi 配置、系统信息、服务重启等功能
"""
import asyncio
import signal
import os
import logging
import socket
from datetime import datetime, timedelta, UTC
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from lelamp.database.session import get_db
from lelamp.api.services.wifi_manager import wifi_manager, WiFiNetwork
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
