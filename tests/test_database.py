"""
测试数据库模块
"""
import pytest
from unittest.mock import patch
from datetime import datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from lelamp.database.base import Base


# --- 内存 SQLite fixtures ---

@pytest.fixture(scope="module")
def db_engine():
    """创建内存 SQLite 引擎"""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    # 导入所有模型以注册到 Base.metadata
    from lelamp.database import models  # noqa: F401
    from lelamp.database import models_auth  # noqa: F401
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session(db_engine):
    """创建数据库 session，每个测试后回滚"""
    from sqlalchemy.orm import sessionmaker
    SessionLocal = sessionmaker(bind=db_engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.mark.unit
class TestDatabaseModels:
    """测试数据库模型"""

    def test_conversation_model(self, db_session):
        """测试对话模型"""
        from lelamp.database.models import Conversation

        conv = Conversation(
            lamp_id="test_lamp",
            messages=[{"role": "user", "content": "你好"}],
            duration=1000,
            user_input="你好",
            ai_response="你好！",
        )
        db_session.add(conv)
        db_session.commit()
        db_session.refresh(conv)

        assert conv.id is not None
        assert conv.lamp_id == "test_lamp"
        assert conv.user_input == "你好"
        assert conv.ai_response == "你好！"

    def test_operation_log_model(self, db_session):
        """测试操作日志模型"""
        from lelamp.database.models import OperationLog

        log = OperationLog(
            lamp_id="test_lamp",
            operation_type="motor_move",
            action="move_joint",
            params={"joint": "base_yaw", "angle": 30},
            success=True,
        )
        db_session.add(log)
        db_session.commit()
        db_session.refresh(log)

        assert log.id is not None
        assert log.lamp_id == "test_lamp"
        assert log.operation_type == "motor_move"
        assert log.success is True

    def test_device_state_model(self, db_session):
        """测试设备状态模型"""
        from lelamp.database.models import DeviceState

        state = DeviceState(
            lamp_id="test_lamp",
            motor_positions={"base_yaw": 0},
            health_status={"status": "healthy"},
            light_color={"r": 255, "g": 0, "b": 0},
            conversation_state="idle",
            uptime_seconds=3600,
        )
        db_session.add(state)
        db_session.commit()
        db_session.refresh(state)

        assert state.id is not None
        assert state.lamp_id == "test_lamp"
        assert state.conversation_state == "idle"

    def test_user_settings_model(self, db_session):
        """测试用户设置模型"""
        from lelamp.database.models import UserSettings

        settings = UserSettings(lamp_id="test_lamp")
        db_session.add(settings)
        db_session.commit()
        db_session.refresh(settings)

        assert settings.id is not None
        assert settings.lamp_id == "test_lamp"
        assert settings.theme == "light"
        assert settings.language == "zh"
        assert settings.led_brightness == 25


@pytest.mark.unit
class TestAuthModels:
    """测试认证模型"""

    def test_user_model(self, db_session):
        """测试用户模型"""
        from lelamp.database.models_auth import User

        user = User(
            username="test_user",
            email="test@example.com",
            hashed_password="hashed",
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)

        assert user.id is not None
        assert user.username == "test_user"
        assert user.email == "test@example.com"
        assert user.is_active is True
        assert user.is_admin is False

    def test_refresh_token_model(self, db_session):
        """测试刷新令牌模型"""
        from lelamp.database.models_auth import User, RefreshToken

        user = User(username="token_user", email="token@test.com", hashed_password="h")
        db_session.add(user)
        db_session.commit()

        token = RefreshToken(
            token="test_token_unique",
            user_id=user.id,
            expires_at=datetime.now() + timedelta(days=7),
        )
        db_session.add(token)
        db_session.commit()
        db_session.refresh(token)

        assert token.id is not None
        assert token.token == "test_token_unique"
        assert token.user_id == user.id

    def test_device_binding_model(self, db_session):
        """测试设备绑定模型"""
        from lelamp.database.models_auth import User, DeviceBinding

        user = User(username="bind_user", email="bind@test.com", hashed_password="h")
        db_session.add(user)
        db_session.commit()

        binding = DeviceBinding(
            user_id=user.id,
            device_id="test_device",
            device_secret="secret123",
        )
        db_session.add(binding)
        db_session.commit()
        db_session.refresh(binding)

        assert binding.id is not None
        assert binding.user_id == user.id
        assert binding.device_id == "test_device"
        assert binding.permission_level == "member"


@pytest.mark.unit
class TestDatabaseSession:
    """测试数据库会话"""

    def test_get_db(self):
        """测试获取数据库会话生成器"""
        from lelamp.database.session import get_db

        gen = get_db()
        assert gen is not None


@pytest.mark.unit
class TestDatabaseCRUD:
    """测试数据库 CRUD 操作"""

    def test_create_and_get_conversation(self, db_session):
        """测试创建和获取对话"""
        from lelamp.database.crud import create_conversation, get_conversation_by_id

        conv = create_conversation(
            db=db_session,
            lamp_id="test_lamp",
            messages=[{"role": "user", "content": "hello"}],
            user_input="hello",
            ai_response="hi there",
        )
        assert conv.id is not None

        fetched = get_conversation_by_id(db_session, conv.id)
        assert fetched is not None
        assert fetched.user_input == "hello"

    def test_get_conversations_by_lamp_id(self, db_session):
        """测试按 lamp_id 获取对话列表"""
        from lelamp.database.crud import create_conversation, get_conversations_by_lamp_id

        create_conversation(db=db_session, lamp_id="lamp_a", messages=[])
        create_conversation(db=db_session, lamp_id="lamp_a", messages=[])
        create_conversation(db=db_session, lamp_id="lamp_b", messages=[])

        results = get_conversations_by_lamp_id(db_session, "lamp_a")
        assert len(results) == 2

    def test_get_latest_device_state(self, db_session):
        """测试获取最新设备状态"""
        from lelamp.database.crud import create_device_state, get_latest_device_state

        create_device_state(
            db=db_session, lamp_id="test_lamp",
            motor_positions={}, health_status={}, light_color={},
            conversation_state="idle", uptime_seconds=100,
        )
        create_device_state(
            db=db_session, lamp_id="test_lamp",
            motor_positions={}, health_status={}, light_color={},
            conversation_state="speaking", uptime_seconds=200,
        )

        latest = get_latest_device_state(db_session, "test_lamp")
        assert latest is not None
        assert latest.conversation_state == "speaking"

    def test_get_latest_device_state_empty(self, db_session):
        """测试空数据库获取最新设备状态"""
        from lelamp.database.crud import get_latest_device_state

        result = get_latest_device_state(db_session, "nonexistent")
        assert result is None

    def test_get_user_settings(self, db_session):
        """测试获取用户设置"""
        from lelamp.database.crud import get_or_create_user_settings, get_user_settings

        settings = get_or_create_user_settings(db_session, "test_lamp")
        assert settings.lamp_id == "test_lamp"

        fetched = get_user_settings(db_session, "test_lamp")
        assert fetched is not None
        assert fetched.id == settings.id

    def test_update_user_settings(self, db_session):
        """测试更新用户设置"""
        from lelamp.database.crud import get_or_create_user_settings, update_user_settings

        get_or_create_user_settings(db_session, "test_lamp")

        updated = update_user_settings(db_session, "test_lamp", theme="dark", volume_level=80)
        assert updated.theme == "dark"
        assert updated.volume_level == 80

    def test_create_operation_log(self, db_session):
        """测试创建操作日志"""
        from lelamp.database.crud import create_operation_log, get_operation_logs_by_lamp_id

        create_operation_log(
            db=db_session, lamp_id="test_lamp_unique",
            operation_type="rgb_set", action="set_color",
            params={"color": "red"}, success=True, duration_ms=50,
        )

        logs = get_operation_logs_by_lamp_id(db_session, "test_lamp_unique")
        assert len(logs) == 1
        assert logs[0].operation_type == "rgb_set"
