"""
设备相关 API 路由

提供设备状态查询、命令发送、对话记录、操作日志等端点。
所有端点都使用数据库进行数据持久化。
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from typing import Dict, List, Optional
from datetime import datetime
import logging
import uuid
import re

from lelamp.database.session import get_db
from lelamp.database import crud
from lelamp.api.models.responses import (
    DeviceStateResponse,
    ConversationListResponse,
    ConversationResponse,
    OperationListResponse,
    OperationResponse,
    HealthResponse,
    DeviceListResponse,
    DeviceInfoResponse,
    CommandResponse,
    StatisticsResponse,
)

logger = logging.getLogger("lelamp.api.devices")

router = APIRouter()


# =============================================================================
# 验证辅助函数
# =============================================================================


def validate_lamp_id(lamp_id: str) -> bool:
    """
    验证 lamp_id 格式

    Args:
        lamp_id: 设备ID

    Returns:
        是否有效

    规则: 只允许字母、数字、连字符和下划线
    """
    pattern = r'^[a-zA-Z0-9_-]+$'
    return bool(re.match(pattern, lamp_id))


def validate_pagination(skip: int, limit: int) -> tuple[int, int]:
    """
    验证并规范化分页参数

    Args:
        skip: 跳过记录数
        limit: 返回记录数

    Returns:
        (skip, limit) 规范化后的值

    Raises:
        HTTPException: 参数无效时
    """
    if skip < 0:
        raise HTTPException(status_code=400, detail="skip parameter must be >= 0")

    if limit < 1:
        raise HTTPException(status_code=400, detail="limit parameter must be >= 1")

    if limit > 100:
        logger.warning(f"limit {limit} exceeds maximum, using 100")
        limit = 100

    return skip, limit


# =============================================================================
# 设备状态端点
# =============================================================================


@router.get("/{lamp_id}/state", response_model=DeviceStateResponse)
async def get_device_state(
    lamp_id: str,
    db: Session = Depends(get_db)
) -> DeviceStateResponse:
    """
    获取设备当前状态

    从数据库查询最新的设备状态记录。如果不存在，返回默认的离线状态。

    Args:
        lamp_id: 设备ID
        db: 数据库会话

    Returns:
        DeviceStateResponse: 设备状态信息

    Raises:
        HTTPException 400: lamp_id 格式无效
    """
    if not validate_lamp_id(lamp_id):
        raise HTTPException(status_code=400, detail="Invalid lamp_id format")

    # 查询最新状态
    state = crud.get_latest_device_state(db, lamp_id)

    if state:
        return DeviceStateResponse(
            lamp_id=state.lamp_id,
            status="online",
            conversation_state=state.conversation_state,
            timestamp=state.timestamp,
            motor_positions=state.motor_positions,
            light_color=state.light_color,
            camera_active=False,  # 从 conversation_state 推断
            uptime_seconds=state.uptime_seconds,
        )
    else:
        # 返回默认离线状态
        return DeviceStateResponse(
            lamp_id=lamp_id,
            status="offline",
            conversation_state="unknown",
            timestamp=datetime.utcnow(),
            motor_positions={},
            light_color={"r": 0, "g": 0, "b": 0},
            camera_active=False,
            uptime_seconds=0,
        )


@router.post("/{lamp_id}/command", response_model=CommandResponse)
async def send_device_command(
    lamp_id: str,
    command: Dict,
    db: Session = Depends(get_db)
) -> CommandResponse:
    """
    发送设备命令

    接收设备命令并创建操作日志记录。
    注意: 实际命令执行将在后续任务中实现（当与 agent 集成时）。

    Args:
        lamp_id: 设备ID
        command: 命令对象，包含 type, action, params
        db: 数据库会话

    Returns:
        CommandResponse: 命令确认信息

    Raises:
        HTTPException 400: 命令格式无效
    """
    if not validate_lamp_id(lamp_id):
        raise HTTPException(status_code=400, detail="Invalid lamp_id format")

    # 验证命令格式
    if "type" not in command or "action" not in command:
        raise HTTPException(status_code=400, detail="Command must include 'type' and 'action'")

    command_type = command.get("type")
    action = command.get("action")
    params = command.get("params", {})

    # 生成命令ID
    command_id = str(uuid.uuid4())

    # 创建操作日志
    try:
        crud.create_operation_log(
            db,
            lamp_id=lamp_id,
            operation_type=command_type,
            action=action,
            params=params,
            success=True,
            duration_ms=None,
        )
    except Exception as e:
        logger.error(f"Failed to create operation log: {e}")
        # 不阻止命令发送，只记录错误

    logger.info(f"Command {command_id} sent to device {lamp_id}: {command_type}/{action}")

    return CommandResponse(
        success=True,
        command_id=command_id,
        message="Command received",
        timestamp=datetime.utcnow(),
    )


# =============================================================================
# 对话记录端点
# =============================================================================


@router.get("/{lamp_id}/conversations", response_model=ConversationListResponse)
async def get_conversations(
    lamp_id: str,
    skip: int = Query(0, ge=0, description="跳过记录数"),
    limit: int = Query(50, ge=1, le=100, description="返回记录数"),
    db: Session = Depends(get_db)
) -> ConversationListResponse:
    """
    获取设备对话历史

    查询指定设备的对话记录，支持分页。

    Args:
        lamp_id: 设备ID
        skip: 跳过记录数（默认0）
        limit: 返回记录数（默认50，最大100）
        db: 数据库会话

    Returns:
        ConversationListResponse: 对话记录列表和总数

    Raises:
        HTTPException 400: lamp_id 格式无效
    """
    if not validate_lamp_id(lamp_id):
        raise HTTPException(status_code=400, detail="Invalid lamp_id format")

    skip, limit = validate_pagination(skip, limit)

    # 查询对话记录
    conversations = crud.get_conversations_by_lamp_id(db, lamp_id, skip=skip, limit=limit)

    # 查询总数
    from sqlalchemy import func
    from lelamp.database.models import Conversation
    total = (
        db.query(func.count(Conversation.id))
        .filter(Conversation.lamp_id == lamp_id)
        .scalar()
    )

    return ConversationListResponse(
        total=total,
        conversations=[
            ConversationResponse(
                id=conv.id,
                timestamp=conv.timestamp,
                lamp_id=conv.lamp_id,
                user_input=conv.user_input,
                ai_response=conv.ai_response,
                duration=conv.duration,
                messages=conv.messages,
            )
            for conv in conversations
        ],
    )


# =============================================================================
# 操作日志端点
# =============================================================================


@router.get("/{lamp_id}/operations", response_model=OperationListResponse)
async def get_operations(
    lamp_id: str,
    skip: int = Query(0, ge=0, description="跳过记录数"),
    limit: int = Query(50, ge=1, le=100, description="返回记录数"),
    hours: int = Query(24, ge=1, le=168, description="时间窗口（小时）"),
    db: Session = Depends(get_db)
) -> OperationListResponse:
    """
    获取设备操作日志

    查询指定设备的操作日志，支持分页和时间过滤。

    Args:
        lamp_id: 设备ID
        skip: 跳过记录数（默认0）
        limit: 返回记录数（默认50，最大100）
        hours: 时间窗口小时数（默认24小时）
        db: 数据库会话

    Returns:
        OperationListResponse: 操作日志列表和总数

    Raises:
        HTTPException 400: lamp_id 格式无效
    """
    if not validate_lamp_id(lamp_id):
        raise HTTPException(status_code=400, detail="Invalid lamp_id format")

    skip, limit = validate_pagination(skip, limit)

    # 查询操作日志（带时间过滤）
    operations = crud.get_recent_operation_logs(db, lamp_id, hours=hours, limit=limit + skip)

    # 应用分页
    operations = operations[skip:skip + limit]

    # 查询总数
    from sqlalchemy import func
    from lelamp.database.models import OperationLog
    from datetime import datetime, timedelta

    time_threshold = datetime.utcnow() - timedelta(hours=hours)
    total = (
        db.query(func.count(OperationLog.id))
        .filter(
            (OperationLog.lamp_id == lamp_id) &
            (OperationLog.timestamp >= time_threshold)
        )
        .scalar()
    )

    return OperationListResponse(
        total=total or 0,
        operations=[
            OperationResponse(
                id=op.id,
                timestamp=op.timestamp,
                lamp_id=op.lamp_id,
                operation_type=op.operation_type,
                action=op.action,
                params=op.params,
                success=op.success,
                error_message=op.error_message,
                duration_ms=op.duration_ms,
            )
            for op in operations
        ],
    )


# =============================================================================
# 设备健康端点
# =============================================================================


@router.get("/{lamp_id}/health", response_model=HealthResponse)
async def get_device_health(
    lamp_id: str,
    db: Session = Depends(get_db)
) -> HealthResponse:
    """
    获取设备健康状态

    从最新设备状态中提取健康信息。

    Args:
        lamp_id: 设备ID
        db: 数据库会话

    Returns:
        HealthResponse: 设备健康状态

    Raises:
        HTTPException 400: lamp_id 格式无效
    """
    if not validate_lamp_id(lamp_id):
        raise HTTPException(status_code=400, detail="Invalid lamp_id format")

    # 查询最新状态
    state = crud.get_latest_device_state(db, lamp_id)

    if state and state.health_status:
        health_data = state.health_status
        overall_status = health_data.get("overall", "unknown")
        motors = health_data.get("motors", [])
        last_check = state.timestamp

        return HealthResponse(
            lamp_id=lamp_id,
            overall_status=overall_status,
            motors=motors,
            last_check=last_check,
        )
    else:
        # 无健康数据
        return HealthResponse(
            lamp_id=lamp_id,
            overall_status="unknown",
            motors=[],
            last_check=datetime.utcnow(),
        )


# =============================================================================
# 设备统计端点
# =============================================================================


@router.get("/{lamp_id}/statistics")
async def get_device_statistics(
    lamp_id: str,
    days: int = Query(7, ge=1, le=30, description="统计天数"),
    db: Session = Depends(get_db)
) -> Dict:
    """
    获取设备统计数据

    查询指定时间窗口内的操作统计信息。

    Args:
        lamp_id: 设备ID
        days: 统计天数（默认7天）
        db: 数据库会话

    Returns:
        Dict: 统计信息，包括总操作数、成功率、操作计数等

    Raises:
        HTTPException 400: lamp_id 格式无效
    """
    if not validate_lamp_id(lamp_id):
        raise HTTPException(status_code=400, detail="Invalid lamp_id format")

    # 查询统计数据
    stats = crud.get_operation_statistics(db, lamp_id, days=days)

    if stats:
        # 找出最常见的操作
        operation_counts = stats["operation_counts"]
        most_common = max(operation_counts.items(), key=lambda x: x[1])[0] if operation_counts else "none"

        return {
            "lamp_id": lamp_id,
            "period_days": days,
            "total_operations": stats["total_operations"],
            "success_rate": stats["success_rate"],
            "operation_counts": stats["operation_counts"],
            "avg_duration_ms": stats["average_duration_ms"],
            "most_common_operation": most_common,
        }
    else:
        # 无数据
        return {
            "lamp_id": lamp_id,
            "period_days": days,
            "total_operations": 0,
            "success_rate": 0.0,
            "operation_counts": {},
            "avg_duration_ms": None,
            "most_common_operation": "none",
        }


# =============================================================================
# 设备列表端点
# =============================================================================


@router.get("", response_model=DeviceListResponse)
async def list_devices(
    db: Session = Depends(get_db)
) -> DeviceListResponse:
    """
    获取所有设备列表

    从对话记录表中查询所有出现过的设备ID，及其最后活跃时间。

    Args:
        db: 数据库会话

    Returns:
        DeviceListResponse: 设备列表
    """
    from sqlalchemy import func
    from lelamp.database.models import Conversation, DeviceState

    # 查询所有对话中的 lamp_id 及其最新时间戳
    subquery = (
        db.query(
            Conversation.lamp_id,
            func.max(Conversation.timestamp).label("last_seen")
        )
        .group_by(Conversation.lamp_id)
        .subquery()
    )

    results = db.query(subquery).all()

    devices = []
    for lamp_id, last_seen in results:
        # 查询最新状态
        state = crud.get_latest_device_state(db, lamp_id)
        conversation_state = state.conversation_state if state else None

        devices.append(
            DeviceInfoResponse(
                lamp_id=lamp_id,
                last_seen=last_seen,
                state=conversation_state,
            )
        )

    return DeviceListResponse(devices=devices)
