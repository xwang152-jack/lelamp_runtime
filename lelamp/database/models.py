"""
SQLAlchemy ORM models for LeLamp runtime.

Defines database models for:
- Conversation: Chat conversation history
- OperationLog: System operation logs
- DeviceState: Device state snapshots
- UserSettings: User preferences and settings
"""
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Integer,
    String,
    DateTime,
    JSON,
    Index,
)
from sqlalchemy.orm import Mapped, mapped_column

from lelamp.database.base import Base


class Conversation(Base):
    """
    Conversation model for storing chat history.

    Attributes:
        id: Primary key
        timestamp: Conversation timestamp
        lamp_id: Device identifier (indexed)
        messages: JSON field storing conversation messages
        duration: Conversation duration in seconds
        user_input: User's input text
        ai_response: AI's response text
    """

    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False, index=True
    )
    lamp_id: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True
    )
    messages: Mapped[dict] = mapped_column(JSON, nullable=False, default=list)
    duration: Mapped[int] = mapped_column(Integer, nullable=True)
    user_input: Mapped[str] = mapped_column(String(500), nullable=True)
    ai_response: Mapped[str] = mapped_column(String(2000), nullable=True)

    __table_args__ = (
        Index("ix_conversations_lamp_id_timestamp", "lamp_id", "timestamp"),
    )

    def __repr__(self) -> str:
        return f"<Conversation(id={self.id}, lamp_id={self.lamp_id}, timestamp={self.timestamp})>"


class OperationLog(Base):
    """
    Operation log model for tracking system operations.

    Attributes:
        id: Primary key
        timestamp: Operation timestamp
        lamp_id: Device identifier (indexed)
        operation_type: Type of operation (motor_move, rgb_set, vision_capture, etc.)
        action: Action name
        params: JSON field storing operation parameters
        success: Whether the operation succeeded
        error_message: Error message if operation failed
        duration_ms: Operation duration in milliseconds
    """

    __tablename__ = "operation_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False, index=True
    )
    lamp_id: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True
    )
    operation_type: Mapped[str] = mapped_column(String(50), nullable=False)
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    params: Mapped[dict] = mapped_column(JSON, nullable=True, default=dict)
    success: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    error_message: Mapped[str | None] = mapped_column(String(500), nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    __table_args__ = (
        Index("ix_operation_logs_lamp_id_timestamp", "lamp_id", "timestamp"),
        Index("ix_operation_logs_operation_type", "operation_type"),
    )

    def __repr__(self) -> str:
        return (
            f"<OperationLog(id={self.id}, lamp_id={self.lamp_id}, "
            f"operation_type={self.operation_type}, success={self.success})>"
        )


class DeviceState(Base):
    """
    Device state model for storing device state snapshots.

    Attributes:
        id: Primary key
        timestamp: State snapshot timestamp
        lamp_id: Device identifier (indexed)
        motor_positions: JSON field storing motor joint positions
        health_status: JSON field storing motor health data
        light_color: JSON field storing current RGB color
        conversation_state: Current conversation state (idle/listening/thinking/speaking)
        uptime_seconds: Device uptime in seconds
    """

    __tablename__ = "device_states"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False, index=True
    )
    lamp_id: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True
    )
    motor_positions: Mapped[dict] = mapped_column(
        JSON, nullable=False, default=dict
    )
    health_status: Mapped[dict] = mapped_column(
        JSON, nullable=False, default=dict
    )
    light_color: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    conversation_state: Mapped[str] = mapped_column(String(50), nullable=False)
    uptime_seconds: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (
        Index("ix_device_states_lamp_id_timestamp", "lamp_id", "timestamp"),
    )

    def __repr__(self) -> str:
        return (
            f"<DeviceState(id={self.id}, lamp_id={self.lamp_id}, "
            f"conversation_state={self.conversation_state})>"
        )


class UserSettings(Base):
    """
    User settings model for storing user preferences.

    Attributes:
        id: Primary key
        lamp_id: Device identifier (unique, indexed)
        theme: UI theme (light/dark)
        language: Language setting (zh/en)
        notifications_enabled: Whether notifications are enabled
        brightness_level: LED brightness level (0-100)
        volume_level: System volume level (0-100)
        created_at: Settings creation timestamp
        updated_at: Settings last update timestamp
    """

    __tablename__ = "user_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    lamp_id: Mapped[str] = mapped_column(
        String(50), unique=True, nullable=False, index=True
    )
    theme: Mapped[str] = mapped_column(String(20), default="light", nullable=False)
    language: Mapped[str] = mapped_column(String(10), default="zh", nullable=False)
    notifications_enabled: Mapped[bool] = mapped_column(
        Boolean, default=True, nullable=False
    )
    brightness_level: Mapped[int] = mapped_column(
        Integer, default=25, nullable=False
    )
    volume_level: Mapped[int] = mapped_column(Integer, default=50, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    def __repr__(self) -> str:
        return f"<UserSettings(id={self.id}, lamp_id={self.lamp_id}, theme={self.theme})>"
