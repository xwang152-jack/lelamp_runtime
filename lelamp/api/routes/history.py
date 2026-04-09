"""
历史记录 API 路由

提供单个对话记录和操作日志的详细查询端点。
"""
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
import logging

from lelamp.database.session import get_db
from lelamp.database.models_auth import User
from lelamp.api.middleware.auth import get_current_user
from lelamp.database import crud
from lelamp.api.models.responses import (
    ConversationResponse,
    OperationResponse,
)

logger = logging.getLogger("lelamp.api.history")

router = APIRouter()


# =============================================================================
# 对话历史端点
# =============================================================================


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation_by_id(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> ConversationResponse:
    """
    获取单个对话记录详情

    通过对话ID查询完整的对话记录信息。

    Args:
        conversation_id: 对话记录ID
        db: 数据库会话

    Returns:
        ConversationResponse: 对话记录详情

    Raises:
        HTTPException 404: 对话记录不存在
    """
    conversation = crud.get_conversation_by_id(db, conversation_id)

    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation {conversation_id} not found"
        )

    return ConversationResponse(
        id=conversation.id,
        timestamp=conversation.timestamp,
        lamp_id=conversation.lamp_id,
        user_input=conversation.user_input,
        ai_response=conversation.ai_response,
        duration=conversation.duration,
        messages=conversation.messages,
    )


# =============================================================================
# 操作历史端点
# =============================================================================


@router.get("/operations/{operation_id}", response_model=OperationResponse)
async def get_operation_by_id(
    operation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> OperationResponse:
    """
    获取单个操作日志详情

    通过操作ID查询完整的操作日志信息。

    Args:
        operation_id: 操作日志ID
        db: 数据库会话

    Returns:
        OperationResponse: 操作日志详情

    Raises:
        HTTPException 404: 操作日志不存在
    """
    # 查询操作日志
    from lelamp.database.models import OperationLog

    operation = (
        db.query(OperationLog)
        .filter(OperationLog.id == operation_id)
        .first()
    )

    if operation is None:
        raise HTTPException(
            status_code=404,
            detail=f"Operation {operation_id} not found"
        )

    return OperationResponse(
        id=operation.id,
        timestamp=operation.timestamp,
        lamp_id=operation.lamp_id,
        operation_type=operation.operation_type,
        action=operation.action,
        params=operation.params,
        success=operation.success,
        error_message=operation.error_message,
        duration_ms=operation.duration_ms,
    )
