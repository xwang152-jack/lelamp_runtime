"""
CRUD operations for LeLamp database models.

Provides database operations for Conversation, OperationLog, DeviceState,
and UserSettings models with proper error handling and type hints.
"""
import logging
from datetime import datetime, timedelta, UTC
from typing import Any, Dict, List, Optional

from sqlalchemy import func, and_
from sqlalchemy.orm import Session

from lelamp.database.models import (
    Conversation,
    OperationLog,
    DeviceState,
    UserSettings,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Conversation CRUD Operations
# =============================================================================

def create_conversation(
    db: Session,
    lamp_id: str,
    messages: List[Dict[str, Any]],
    duration: Optional[int] = None,
    user_input: Optional[str] = None,
    ai_response: Optional[str] = None,
) -> Conversation:
    """
    Create a new conversation record.

    Args:
        db: Database session
        lamp_id: Device identifier
        messages: List of conversation messages (role, content)
        duration: Conversation duration in seconds
        user_input: User's input text
        ai_response: AI's response text

    Returns:
        Created Conversation object

    Raises:
        sqlalchemy_exc.SQLAlchemyError: On database errors
    """
    conversation = Conversation(
        lamp_id=lamp_id,
        messages=messages,
        duration=duration,
        user_input=user_input,
        ai_response=ai_response,
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    logger.info(f"Created conversation {conversation.id} for lamp {lamp_id}")
    return conversation


def get_conversation_by_id(db: Session, conversation_id: int) -> Optional[Conversation]:
    """
    Retrieve a conversation by ID.

    Args:
        db: Database session
        conversation_id: Conversation primary key

    Returns:
        Conversation object if found, None otherwise
    """
    return db.query(Conversation).filter(Conversation.id == conversation_id).first()


def get_conversations_by_lamp_id(
    db: Session, lamp_id: str, skip: int = 0, limit: int = 100
) -> List[Conversation]:
    """
    Retrieve conversations for a specific lamp with pagination.

    Args:
        db: Database session
        lamp_id: Device identifier
        skip: Number of records to skip (pagination offset)
        limit: Maximum number of records to return

    Returns:
        List of Conversation objects ordered by timestamp descending
    """
    return (
        db.query(Conversation)
        .filter(Conversation.lamp_id == lamp_id)
        .order_by(Conversation.timestamp.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def get_recent_conversations(
    db: Session, lamp_id: str, hours: int = 24, limit: int = 50
) -> List[Conversation]:
    """
    Retrieve recent conversations within a time window.

    Args:
        db: Database session
        lamp_id: Device identifier
        hours: Time window in hours (default: 24)
        limit: Maximum number of records to return

    Returns:
        List of Conversation objects within the time window
    """
    time_threshold = datetime.now(UTC) - timedelta(hours=hours)
    return (
        db.query(Conversation)
        .filter(
            and_(
                Conversation.lamp_id == lamp_id,
                Conversation.timestamp >= time_threshold,
            )
        )
        .order_by(Conversation.timestamp.desc())
        .limit(limit)
        .all()
    )


def delete_old_conversations(db: Session, days: int = 30) -> int:
    """
    Delete conversations older than specified days.

    Args:
        db: Database session
        days: Delete conversations older than this many days

    Returns:
        Number of conversations deleted
    """
    time_threshold = datetime.now(UTC) - timedelta(days=days)
    count = (
        db.query(Conversation)
        .filter(Conversation.timestamp < time_threshold)
        .delete()
    )
    db.commit()
    logger.info(f"Deleted {count} old conversations (older than {days} days)")
    return count


# =============================================================================
# OperationLog CRUD Operations
# =============================================================================

def create_operation_log(
    db: Session,
    lamp_id: str,
    operation_type: str,
    action: str,
    params: Optional[Dict[str, Any]] = None,
    success: bool = True,
    error_message: Optional[str] = None,
    duration_ms: Optional[int] = None,
) -> OperationLog:
    """
    Create a new operation log record.

    Args:
        db: Database session
        lamp_id: Device identifier
        operation_type: Type of operation (motor_move, rgb_set, vision_capture, etc.)
        action: Action name
        params: Operation parameters as dictionary
        success: Whether the operation succeeded
        error_message: Error message if operation failed
        duration_ms: Operation duration in milliseconds

    Returns:
        Created OperationLog object
    """
    log = OperationLog(
        lamp_id=lamp_id,
        operation_type=operation_type,
        action=action,
        params=params or {},
        success=success,
        error_message=error_message,
        duration_ms=duration_ms,
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    logger.debug(
        f"Logged operation {operation_type}/{action} for lamp {lamp_id} "
        f"(success={success})"
    )
    return log


def get_operation_logs_by_lamp_id(
    db: Session, lamp_id: str, skip: int = 0, limit: int = 100
) -> List[OperationLog]:
    """
    Retrieve operation logs for a specific lamp with pagination.

    Args:
        db: Database session
        lamp_id: Device identifier
        skip: Number of records to skip (pagination offset)
        limit: Maximum number of records to return

    Returns:
        List of OperationLog objects ordered by timestamp descending
    """
    return (
        db.query(OperationLog)
        .filter(OperationLog.lamp_id == lamp_id)
        .order_by(OperationLog.timestamp.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def get_recent_operation_logs(
    db: Session, lamp_id: str, hours: int = 24, limit: int = 100
) -> List[OperationLog]:
    """
    Retrieve recent operation logs within a time window.

    Args:
        db: Database session
        lamp_id: Device identifier
        hours: Time window in hours (default: 24)
        limit: Maximum number of records to return

    Returns:
        List of OperationLog objects within the time window
    """
    time_threshold = datetime.now(UTC) - timedelta(hours=hours)
    return (
        db.query(OperationLog)
        .filter(
            and_(
                OperationLog.lamp_id == lamp_id,
                OperationLog.timestamp >= time_threshold,
            )
        )
        .order_by(OperationLog.timestamp.desc())
        .limit(limit)
        .all()
    )


def get_operation_statistics(
    db: Session, lamp_id: str, days: int = 7
) -> Optional[Dict[str, Any]]:
    """
    Get aggregated operation statistics for a lamp.

    Args:
        db: Database session
        lamp_id: Device identifier
        days: Statistics time window in days (default: 7)

    Returns:
        Dictionary with statistics:
        - total_operations: Total number of operations
        - successful_operations: Number of successful operations
        - failed_operations: Number of failed operations
        - success_rate: Success rate as float (0-1)
        - operation_counts: Dict with count per operation_type
        - average_duration_ms: Average operation duration in ms

        Returns None if no operations found in the time window
    """
    time_threshold = datetime.now(UTC) - timedelta(days=days)

    # Get all operations in time window
    operations = (
        db.query(OperationLog)
        .filter(
            and_(
                OperationLog.lamp_id == lamp_id,
                OperationLog.timestamp >= time_threshold,
            )
        )
        .all()
    )

    if not operations:
        return None

    # Calculate statistics
    total = len(operations)
    successful = sum(1 for op in operations if op.success)
    failed = total - successful
    success_rate = successful / total if total > 0 else 0

    # Count operations by type
    operation_counts: Dict[str, int] = {}
    for op in operations:
        operation_counts[op.operation_type] = operation_counts.get(op.operation_type, 0) + 1

    # Calculate average duration (only for successful operations with duration)
    durations = [
        op.duration_ms
        for op in operations
        if op.success and op.duration_ms is not None
    ]
    average_duration = sum(durations) / len(durations) if durations else None

    return {
        "total_operations": total,
        "successful_operations": successful,
        "failed_operations": failed,
        "success_rate": success_rate,
        "operation_counts": operation_counts,
        "average_duration_ms": average_duration,
    }


# =============================================================================
# DeviceState CRUD Operations
# =============================================================================

def create_device_state(
    db: Session,
    lamp_id: str,
    motor_positions: Dict[str, Any],
    health_status: Dict[str, Any],
    light_color: Dict[str, Any],
    conversation_state: str,
    uptime_seconds: int,
) -> DeviceState:
    """
    Create a new device state record.

    Args:
        db: Database session
        lamp_id: Device identifier
        motor_positions: Motor joint positions dictionary
        health_status: Motor health data dictionary
        light_color: RGB color dictionary
        conversation_state: Current conversation state
        uptime_seconds: Device uptime in seconds

    Returns:
        Created DeviceState object
    """
    state = DeviceState(
        lamp_id=lamp_id,
        motor_positions=motor_positions,
        health_status=health_status,
        light_color=light_color,
        conversation_state=conversation_state,
        uptime_seconds=uptime_seconds,
    )
    db.add(state)
    db.commit()
    db.refresh(state)
    logger.debug(f"Created device state {state.id} for lamp {lamp_id}")
    return state


def get_latest_device_state(db: Session, lamp_id: str) -> Optional[DeviceState]:
    """
    Retrieve the latest device state for a lamp.

    Args:
        db: Database session
        lamp_id: Device identifier

    Returns:
        Latest DeviceState object if found, None otherwise
    """
    return (
        db.query(DeviceState)
        .filter(DeviceState.lamp_id == lamp_id)
        .order_by(DeviceState.timestamp.desc())
        .first()
    )


def get_device_state_history(
    db: Session, lamp_id: str, hours: int = 24, limit: int = 100
) -> List[DeviceState]:
    """
    Retrieve device state history within a time window.

    Args:
        db: Database session
        lamp_id: Device identifier
        hours: Time window in hours (default: 24)
        limit: Maximum number of records to return

    Returns:
        List of DeviceState objects within the time window
    """
    time_threshold = datetime.now(UTC) - timedelta(hours=hours)
    return (
        db.query(DeviceState)
        .filter(
            and_(
                DeviceState.lamp_id == lamp_id,
                DeviceState.timestamp >= time_threshold,
            )
        )
        .order_by(DeviceState.timestamp.desc())
        .limit(limit)
        .all()
    )


def delete_old_device_states(db: Session, days: int = 7) -> int:
    """
    Delete device states older than specified days.

    Args:
        db: Database session
        days: Delete states older than this many days

    Returns:
        Number of device states deleted
    """
    time_threshold = datetime.now(UTC) - timedelta(days=days)
    count = (
        db.query(DeviceState)
        .filter(DeviceState.timestamp < time_threshold)
        .delete()
    )
    db.commit()
    logger.info(f"Deleted {count} old device states (older than {days} days)")
    return count


# =============================================================================
# UserSettings CRUD Operations
# =============================================================================

def get_or_create_user_settings(
    db: Session, lamp_id: str
) -> UserSettings:
    """
    Get existing user settings or create with defaults.

    Args:
        db: Database session
        lamp_id: Device identifier

    Returns:
        UserSettings object (existing or newly created)
    """
    settings = db.query(UserSettings).filter(UserSettings.lamp_id == lamp_id).first()

    if settings is None:
        settings = UserSettings(lamp_id=lamp_id)
        db.add(settings)
        db.commit()
        db.refresh(settings)
        logger.info(f"Created default user settings for lamp {lamp_id}")

    return settings


def get_user_settings(db: Session, lamp_id: str) -> Optional[UserSettings]:
    """
    Retrieve user settings for a lamp.

    Args:
        db: Database session
        lamp_id: Device identifier

    Returns:
        UserSettings object if found, None otherwise
    """
    return db.query(UserSettings).filter(UserSettings.lamp_id == lamp_id).first()


def update_user_settings(
    db: Session,
    lamp_id: str,
    **kwargs: Any,
) -> UserSettings:
    """
    Update user settings for a lamp.

    Args:
        db: Database session
        lamp_id: Device identifier
        **kwargs: Settings fields to update (theme, language, notifications_enabled,
                  brightness_level, volume_level)

    Returns:
        Updated UserSettings object

    Raises:
        ValueError: If settings not found for the lamp_id
    """
    settings = get_user_settings(db, lamp_id)

    if settings is None:
        raise ValueError(f"User settings not found for lamp {lamp_id}")

    # Update specified fields
    for key, value in kwargs.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
        else:
            logger.warning(f"Invalid UserSettings field: {key}")

    db.commit()
    db.refresh(settings)
    logger.info(f"Updated user settings for lamp {lamp_id}")
    return settings
