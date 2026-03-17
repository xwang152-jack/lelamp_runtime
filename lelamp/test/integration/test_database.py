"""
Integration tests for database persistence layer.

Tests cover:
- Database initialization
- CRUD operations for all models
- Data cleanup utilities
- Statistics and aggregation queries
"""
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Generator

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker, Session

from lelamp.database.base import Base, get_db
from lelamp.database.models import (
    Conversation,
    OperationLog,
    DeviceState,
    UserSettings,
)
from lelamp.database import crud


@pytest.fixture
def test_db_url() -> str:
    """Create a temporary database file for testing."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    temp_file.close()
    # Use aiosqlite for async compatibility
    return f"sqlite+aiosqlite:///{temp_file.name}"


@pytest.fixture
def test_engine(test_db_url: str):
    """Create test database engine."""
    engine = create_engine(
        test_db_url.replace("+aiosqlite", ""),  # Use regular sqlite for tests
        echo=False,
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    engine.dispose()

    # Cleanup temp file
    db_path = test_db_url.replace("sqlite:///", "").replace("sqlite+aiosqlite:///", "")
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def test_session(test_engine) -> Generator[Session, None, None]:
    """Create test database session."""
    TestingSessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=test_engine
    )
    session = TestingSessionLocal()
    yield session
    session.close()


class TestDatabaseInitialization:
    """Test database initialization and setup."""

    def test_database_initialization(self, test_engine):
        """Test that database tables are created correctly."""
        # Check that all tables exist
        inspector = inspect(test_engine)
        tables = inspector.get_table_names()

        assert "conversations" in tables
        assert "operation_logs" in tables
        assert "device_states" in tables
        assert "user_settings" in tables

    def test_database_indexes(self, test_engine):
        """Test that required indexes are created."""
        inspector = inspect(test_engine)

        # Check conversations table indexes
        conversation_indexes = [
            idx["name"] for idx in inspector.get_indexes("conversations")
        ]
        assert any("lamp_id" in idx for idx in conversation_indexes)

        # Check operation_logs table indexes
        operation_log_indexes = [
            idx["name"] for idx in inspector.get_indexes("operation_logs")
        ]
        assert any("lamp_id" in idx for idx in operation_log_indexes)

        # Check device_states table indexes
        device_state_indexes = [
            idx["name"] for idx in inspector.get_indexes("device_states")
        ]
        assert any("lamp_id" in idx for idx in device_state_indexes)

        # Check user_settings table indexes
        user_settings_indexes = [
            idx["name"] for idx in inspector.get_indexes("user_settings")
        ]
        assert any("lamp_id" in idx for idx in user_settings_indexes)


class TestConversationCRUD:
    """Test CRUD operations for Conversation model."""

    def test_create_conversation(self, test_session: Session):
        """Test creating a new conversation record."""
        messages = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什么我可以帮助你的吗？"},
        ]

        conversation = crud.create_conversation(
            db=test_session,
            lamp_id="test_lamp",
            messages=messages,
            duration=15,
            user_input="你好",
            ai_response="你好！有什么我可以帮助你的吗？",
        )

        assert conversation.id is not None
        assert conversation.lamp_id == "test_lamp"
        assert conversation.messages == messages
        assert conversation.duration == 15
        assert conversation.user_input == "你好"
        assert conversation.ai_response == "你好！有什么我可以帮助你的吗？"
        assert conversation.timestamp is not None

    def test_get_conversation_by_id(self, test_session: Session):
        """Test retrieving a conversation by ID."""
        # Create a conversation
        created = crud.create_conversation(
            db=test_session,
            lamp_id="test_lamp",
            messages=[{"role": "user", "content": "test"}],
            duration=10,
            user_input="test",
            ai_response="response",
        )

        # Retrieve by ID
        retrieved = crud.get_conversation_by_id(
            db=test_session, conversation_id=created.id
        )

        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.lamp_id == "test_lamp"
        assert retrieved.user_input == "test"

    def test_get_conversations_by_lamp_id(self, test_session: Session):
        """Test retrieving conversations by lamp ID with pagination."""
        lamp_id = "test_lamp"

        # Create multiple conversations
        for i in range(5):
            crud.create_conversation(
                db=test_session,
                lamp_id=lamp_id,
                messages=[{"role": "user", "content": f"test {i}"}],
                duration=10 + i,
                user_input=f"test {i}",
                ai_response=f"response {i}",
            )

        # Retrieve with pagination
        conversations = crud.get_conversations_by_lamp_id(
            db=test_session, lamp_id=lamp_id, skip=0, limit=3
        )

        assert len(conversations) == 3
        assert all(c.lamp_id == lamp_id for c in conversations)

    def test_get_recent_conversations(self, test_session: Session):
        """Test retrieving recent conversations within a time window."""
        lamp_id = "test_lamp"

        # Create conversations at different times
        now = datetime.utcnow()

        # Create old conversation (25 hours ago)
        old_conv = crud.create_conversation(
            db=test_session,
            lamp_id=lamp_id,
            messages=[{"role": "user", "content": "old"}],
            duration=10,
            user_input="old",
            ai_response="old response",
        )
        old_conv.timestamp = now - timedelta(hours=25)

        # Create recent conversation (1 hour ago)
        recent_conv = crud.create_conversation(
            db=test_session,
            lamp_id=lamp_id,
            messages=[{"role": "user", "content": "recent"}],
            duration=10,
            user_input="recent",
            ai_response="recent response",
        )
        recent_conv.timestamp = now - timedelta(hours=1)

        test_session.commit()

        # Get recent conversations (last 24 hours)
        recent = crud.get_recent_conversations(
            db=test_session, lamp_id=lamp_id, hours=24, limit=10
        )

        assert len(recent) == 1
        assert recent[0].id == recent_conv.id

    def test_delete_old_conversations(self, test_session: Session):
        """Test deleting conversations older than specified days."""
        lamp_id = "test_lamp"
        now = datetime.utcnow()

        # Create old conversation (35 days ago)
        old_conv = crud.create_conversation(
            db=test_session,
            lamp_id=lamp_id,
            messages=[{"role": "user", "content": "old"}],
            duration=10,
            user_input="old",
            ai_response="old response",
        )
        old_conv.timestamp = now - timedelta(days=35)

        # Create recent conversation (10 days ago)
        recent_conv = crud.create_conversation(
            db=test_session,
            lamp_id=lamp_id,
            messages=[{"role": "user", "content": "recent"}],
            duration=10,
            user_input="recent",
            ai_response="recent response",
        )
        recent_conv.timestamp = now - timedelta(days=10)

        test_session.commit()

        # Delete conversations older than 30 days
        deleted_count = crud.delete_old_conversations(db=test_session, days=30)

        assert deleted_count == 1

        # Verify old conversation is deleted
        remaining = crud.get_conversations_by_lamp_id(
            db=test_session, lamp_id=lamp_id
        )
        assert len(remaining) == 1
        assert remaining[0].id == recent_conv.id


class TestOperationLogCRUD:
    """Test CRUD operations for OperationLog model."""

    def test_create_operation_log(self, test_session: Session):
        """Test creating a new operation log."""
        params = {"joint": "base_yaw", "position": 90}

        log = crud.create_operation_log(
            db=test_session,
            lamp_id="test_lamp",
            operation_type="motor_move",
            action="move_joint",
            params=params,
            success=True,
            error_message=None,
            duration_ms=150,
        )

        assert log.id is not None
        assert log.lamp_id == "test_lamp"
        assert log.operation_type == "motor_move"
        assert log.action == "move_joint"
        assert log.params == params
        assert log.success is True
        assert log.duration_ms == 150

    def test_get_operation_logs_by_lamp_id(self, test_session: Session):
        """Test retrieving operation logs by lamp ID."""
        lamp_id = "test_lamp"

        # Create multiple logs
        for i in range(3):
            crud.create_operation_log(
                db=test_session,
                lamp_id=lamp_id,
                operation_type="motor_move",
                action=f"move_{i}",
                params={"index": i},
                success=True,
                duration_ms=100,
            )

        # Retrieve logs
        logs = crud.get_operation_logs_by_lamp_id(
            db=test_session, lamp_id=lamp_id, skip=0, limit=10
        )

        assert len(logs) == 3
        assert all(log.lamp_id == lamp_id for log in logs)

    def test_get_recent_operation_logs(self, test_session: Session):
        """Test retrieving recent operation logs."""
        lamp_id = "test_lamp"
        now = datetime.utcnow()

        # Create old log (25 hours ago)
        old_log = crud.create_operation_log(
            db=test_session,
            lamp_id=lamp_id,
            operation_type="motor_move",
            action="old_action",
            params={},
            success=True,
            duration_ms=100,
        )
        old_log.timestamp = now - timedelta(hours=25)

        # Create recent log (1 hour ago)
        recent_log = crud.create_operation_log(
            db=test_session,
            lamp_id=lamp_id,
            operation_type="rgb_set",
            action="set_color",
            params={"color": "red"},
            success=True,
            duration_ms=50,
        )
        recent_log.timestamp = now - timedelta(hours=1)

        test_session.commit()

        # Get recent logs
        recent = crud.get_recent_operation_logs(
            db=test_session, lamp_id=lamp_id, hours=24, limit=10
        )

        assert len(recent) == 1
        assert recent[0].id == recent_log.id

    def test_get_operation_statistics(self, test_session: Session):
        """Test getting aggregated operation statistics."""
        lamp_id = "test_lamp"
        now = datetime.utcnow()

        # Create logs for different operation types
        for i in range(5):
            crud.create_operation_log(
                db=test_session,
                lamp_id=lamp_id,
                operation_type="motor_move",
                action="move_joint",
                params={},
                success=True,
                duration_ms=100 + i * 10,
            )

        for i in range(3):
            crud.create_operation_log(
                db=test_session,
                lamp_id=lamp_id,
                operation_type="rgb_set",
                action="set_color",
                params={},
                success=True,
                duration_ms=50,
            )

        # Create one failed operation
        crud.create_operation_log(
            db=test_session,
            lamp_id=lamp_id,
            operation_type="vision_capture",
            action="capture",
            params={},
            success=False,
            error_message="Camera not available",
            duration_ms=0,
        )

        test_session.commit()

        # Get statistics
        stats = crud.get_operation_statistics(db=test_session, lamp_id=lamp_id, days=7)

        assert stats is not None
        assert stats["total_operations"] == 9
        assert stats["successful_operations"] == 8
        assert stats["failed_operations"] == 1
        assert stats["success_rate"] == 8 / 9
        assert "operation_counts" in stats
        assert stats["operation_counts"]["motor_move"] == 5
        assert stats["operation_counts"]["rgb_set"] == 3


class TestDeviceStateCRUD:
    """Test CRUD operations for DeviceState model."""

    def test_create_device_state(self, test_session: Session):
        """Test creating a new device state record."""
        motor_positions = {
            "base_yaw": 90,
            "base_pitch": 45,
            "elbow_pitch": 30,
        }
        health_status = {
            "motor1": {"temperature": 45.0, "voltage": 12.5},
            "motor2": {"temperature": 50.0, "voltage": 12.3},
        }
        light_color = {"r": 255, "g": 0, "b": 0}

        state = crud.create_device_state(
            db=test_session,
            lamp_id="test_lamp",
            motor_positions=motor_positions,
            health_status=health_status,
            light_color=light_color,
            conversation_state="idle",
            uptime_seconds=3600,
        )

        assert state.id is not None
        assert state.lamp_id == "test_lamp"
        assert state.motor_positions == motor_positions
        assert state.health_status == health_status
        assert state.light_color == light_color
        assert state.conversation_state == "idle"
        assert state.uptime_seconds == 3600

    def test_get_latest_device_state(self, test_session: Session):
        """Test retrieving the latest device state."""
        lamp_id = "test_lamp"

        # Create multiple states
        state1 = crud.create_device_state(
            db=test_session,
            lamp_id=lamp_id,
            motor_positions={},
            health_status={},
            light_color={},
            conversation_state="idle",
            uptime_seconds=1000,
        )

        state2 = crud.create_device_state(
            db=test_session,
            lamp_id=lamp_id,
            motor_positions={},
            health_status={},
            light_color={},
            conversation_state="thinking",
            uptime_seconds=2000,
        )

        # Get latest
        latest = crud.get_latest_device_state(db=test_session, lamp_id=lamp_id)

        assert latest is not None
        assert latest.id == state2.id
        assert latest.conversation_state == "thinking"

    def test_get_device_state_history(self, test_session: Session):
        """Test retrieving device state history."""
        lamp_id = "test_lamp"

        # Create multiple states
        for i in range(5):
            crud.create_device_state(
                db=test_session,
                lamp_id=lamp_id,
                motor_positions={"pos": i},
                health_status={},
                light_color={},
                conversation_state="idle",
                uptime_seconds=i * 100,
            )

        # Get history
        history = crud.get_device_state_history(
            db=test_session, lamp_id=lamp_id, hours=24, limit=10
        )

        assert len(history) == 5
        assert all(s.lamp_id == lamp_id for s in history)

    def test_delete_old_device_states(self, test_session: Session):
        """Test deleting old device states."""
        lamp_id = "test_lamp"
        now = datetime.utcnow()

        # Create old state (10 days ago) - manually set timestamp
        old_state = crud.create_device_state(
            db=test_session,
            lamp_id=lamp_id,
            motor_positions={},
            health_status={},
            light_color={},
            conversation_state="idle",
            uptime_seconds=1000,
        )
        # Manually update timestamp and commit
        old_state.timestamp = now - timedelta(days=10)
        test_session.add(old_state)
        test_session.commit()

        # Create recent state (1 day ago) - manually set timestamp
        recent_state = crud.create_device_state(
            db=test_session,
            lamp_id=lamp_id,
            motor_positions={},
            health_status={},
            light_color={},
            conversation_state="idle",
            uptime_seconds=2000,
        )
        # Manually update timestamp and commit
        recent_state.timestamp = now - timedelta(days=1)
        test_session.add(recent_state)
        test_session.commit()

        # Delete states older than 7 days
        deleted_count = crud.delete_old_device_states(db=test_session, days=7)

        assert deleted_count == 1

        # Verify recent state remains (query with 48 hours to include our 1-day-old state)
        remaining = crud.get_device_state_history(
            db=test_session, lamp_id=lamp_id, hours=48, limit=10
        )
        assert len(remaining) == 1
        assert remaining[0].id == recent_state.id


class TestUserSettingsCRUD:
    """Test CRUD operations for UserSettings model."""

    def test_get_or_create_user_settings(self, test_session: Session):
        """Test getting or creating user settings."""
        lamp_id = "test_lamp"

        # First call should create with defaults
        settings = crud.get_or_create_user_settings(db=test_session, lamp_id=lamp_id)

        assert settings.id is not None
        assert settings.lamp_id == lamp_id
        assert settings.theme == "light"
        assert settings.language == "zh"
        assert settings.notifications_enabled is True
        assert settings.brightness_level == 25
        assert settings.volume_level == 50

        # Second call should return existing settings
        settings2 = crud.get_or_create_user_settings(
            db=test_session, lamp_id=lamp_id
        )

        assert settings2.id == settings.id

    def test_update_user_settings(self, test_session: Session):
        """Test updating user settings."""
        lamp_id = "test_lamp"

        # Create settings
        settings = crud.get_or_create_user_settings(db=test_session, lamp_id=lamp_id)

        # Update settings
        updated = crud.update_user_settings(
            db=test_session,
            lamp_id=lamp_id,
            theme="dark",
            brightness_level=50,
            volume_level=75,
        )

        assert updated.theme == "dark"
        assert updated.brightness_level == 50
        assert updated.volume_level == 75
        assert updated.language == "zh"  # Unchanged

    def test_get_user_settings(self, test_session: Session):
        """Test retrieving user settings."""
        lamp_id = "test_lamp"

        # Create settings
        crud.get_or_create_user_settings(db=test_session, lamp_id=lamp_id)

        # Get settings
        settings = crud.get_user_settings(db=test_session, lamp_id=lamp_id)

        assert settings is not None
        assert settings.lamp_id == lamp_id

    def test_get_user_settings_not_found(self, test_session: Session):
        """Test retrieving non-existent user settings."""
        settings = crud.get_user_settings(db=test_session, lamp_id="nonexistent")

        assert settings is None
