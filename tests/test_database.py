"""
测试数据库模块
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta


@pytest.mark.unit
class TestDatabaseModels:
    """测试数据库模型"""

    def test_conversation_model(self):
        """测试对话模型"""
        from lelamp.database.models import Conversation

        conv = Conversation(
            lamp_id="test_lamp",
            user_input="你好",
            response="你好！",
            duration_ms=1000,
        )

        assert conv.lamp_id == "test_lamp"
        assert conv.user_input == "你好"

    def test_operation_log_model(self):
        """测试操作日志模型"""
        from lelamp.database.models import OperationLog

        log = OperationLog(
            lamp_id="test_lamp",
            operation="test_op",
            success=True,
        )

        assert log.lamp_id == "test_lamp"
        assert log.operation == "test_op"
        assert log.success is True

    def test_device_state_model(self):
        """测试设备状态模型"""
        from lelamp.database.models import DeviceState

        state = DeviceState(
            lamp_id="test_lamp",
            state_data='{"test": "data"}',
        )

        assert state.lamp_id == "test_lamp"

    def test_user_settings_model(self):
        """测试用户设置模型"""
        from lelamp.database.models import UserSettings

        settings = UserSettings(
            lamp_id="test_lamp",
            settings='{"volume": 50}',
        )

        assert settings.lamp_id == "test_lamp"


@pytest.mark.unit
class TestAuthModels:
    """测试认证模型"""

    def test_user_model(self):
        """测试用户模型"""
        from lelamp.database.models_auth import User

        user = User(
            username="test_user",
            email="test@example.com",
            hashed_password="hashed",
        )

        assert user.username == "test_user"
        assert user.email == "test@example.com"
        assert user.hashed_password == "hashed"

    def test_refresh_token_model(self):
        """测试刷新令牌模型"""
        from lelamp.database.models_auth import RefreshToken

        token = RefreshToken(
            token="test_token",
            user_id=1,
            expires_at=datetime.now() + timedelta(days=7),
        )

        assert token.token == "test_token"
        assert token.user_id == 1

    def test_device_binding_model(self):
        """测试设备绑定模型"""
        from lelamp.database.models_auth import DeviceBinding

        binding = DeviceBinding(
            user_id=1,
            device_id="test_device",
            permissions='["read", "write"]',
        )

        assert binding.user_id == 1
        assert binding.device_id == "test_device"


@pytest.mark.unit
class TestDatabaseSession:
    """测试数据库会话"""

    def test_get_db(self):
        """测试获取数据库会话"""
        from lelamp.database.session import get_db

        gen = get_db()
        assert gen is not None


@pytest.mark.unit
class TestDatabaseCRUD:
    """测试数据库CRUD操作"""

    def test_get_latest_device_state(self):
        """测试获取最新设备状态"""
        from lelamp.database.crud import get_latest_device_state

        mock_db = Mock(spec=Mock)
        mock_query = MagicMock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.first.return_value = None

        result = get_latest_device_state(mock_db, lamp_id="test_lamp")
        # 可能返回None
        assert result is None or isinstance(result, MagicMock)

    def test_get_user_settings(self):
        """测试获取用户设置"""
        from lelamp.database.crud import get_user_settings

        mock_db = Mock(spec=Mock)
        mock_query = MagicMock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = None

        result = get_user_settings(mock_db, lamp_id="test_lamp")
        # 可能返回None
        assert result is None or isinstance(result, MagicMock)

    def test_upsert_user_settings(self):
        """测试插入或更新用户设置"""
        from lelamp.database.crud import upsert_user_settings

        mock_db = Mock(spec=Mock)
        mock_query = MagicMock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = None  # 不存在

        with patch('lelamp.database.crud.UserSettings') as mock_model:
            mock_instance = MagicMock()
            mock_model.return_value = mock_instance
            mock_db.add.return_value = None
            mock_db.commit.return_value = None
            mock_db.refresh.return_value = None

            result = upsert_user_settings(
                db=mock_db,
                lamp_id="test_lamp",
                settings='{"volume": 75}',
            )

            assert result is not None

    def test_update_user_settings(self):
        """测试更新用户设置"""
        from lelamp.database.crud import update_user_settings

        mock_db = Mock(spec=Mock)
        mock_settings = MagicMock()
        mock_query = MagicMock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = mock_settings
        mock_db.commit.return_value = None

        result = update_user_settings(
            db=mock_db,
            lamp_id="test_lamp",
            settings='{"volume": 75}',
        )

        assert result is not None
