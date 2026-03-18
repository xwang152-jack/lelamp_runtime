"""
Unit tests for authentication models.

Tests User, DeviceBinding, and RefreshToken models.
"""
import pytest
from datetime import datetime, timedelta
from lelamp.database.models_auth import User, DeviceBinding, RefreshToken
from lelamp.database.base import Base, engine


@pytest.fixture(scope="function")
def setup_auth_db():
    """创建认证相关表"""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


def test_create_user(setup_auth_db):
    """测试创建用户"""
    from lelamp.database.session import get_db_session
    db = get_db_session()

    user = User(
        username="testuser",
        email="test@example.com",
        hashed_password="hashed_password_here"
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    assert user.id is not None
    assert user.username == "testuser"
    assert user.email == "test@example.com"
    assert user.is_active is True
    assert user.is_admin is False
    assert user.created_at is not None
    assert user.last_login is None
    db.close()


def test_user_unique_constraints(setup_auth_db):
    """测试用户唯一性约束"""
    from lelamp.database.session import get_db_session
    from sqlalchemy.exc import IntegrityError
    db = get_db_session()

    # 创建第一个用户
    user1 = User(
        username="testuser",
        email="test@example.com",
        hashed_password="hashed_password_here"
    )
    db.add(user1)
    db.commit()

    # 尝试创建重复用户名的用户
    user2 = User(
        username="testuser",  # 重复用户名
        email="test2@example.com",
        hashed_password="hashed_password_here"
    )
    db.add(user2)
    with pytest.raises(IntegrityError):
        db.commit()

    db.close()


def test_device_binding_relationship(setup_auth_db):
    """测试设备绑定关系"""
    from lelamp.database.session import get_db_session
    db = get_db_session()

    # 创建用户
    user = User(
        username="testuser",
        email="test@example.com",
        hashed_password="hashed_password_here"
    )
    db.add(user)
    db.flush()

    # 创建设备绑定
    binding = DeviceBinding(
        user_id=user.id,
        device_id="lelamp_001",
        device_secret="secret_123",
        permission_level="admin"
    )
    db.add(binding)
    db.commit()

    # 验证关系
    assert len(user.devices) == 1
    assert user.devices[0].device_id == "lelamp_001"
    assert binding.user.username == "testuser"
    assert binding.permission_level == "admin"
    db.close()


def test_refresh_token_relationship(setup_auth_db):
    """测试刷新令牌关系"""
    from lelamp.database.session import get_db_session
    db = get_db_session()

    # 创建用户
    user = User(
        username="testuser",
        email="test@example.com",
        hashed_password="hashed_password_here"
    )
    db.add(user)
    db.flush()

    # 创建刷新令牌
    from datetime import timedelta
    token = RefreshToken(
        user_id=user.id,
        token="sample_refresh_token_123",
        expires_at=datetime.utcnow() + timedelta(days=7)
    )
    db.add(token)
    db.commit()

    # 验证关系
    assert len(user.refresh_tokens) == 1
    assert user.refresh_tokens[0].token == "sample_refresh_token_123"
    assert token.user.username == "testuser"
    assert token.revoked is False
    db.close()


def test_user_repr(setup_auth_db):
    """测试用户模型的 __repr__ 方法"""
    from lelamp.database.session import get_db_session
    db = get_db_session()

    user = User(
        username="testuser",
        email="test@example.com",
        hashed_password="hashed_password_here"
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    repr_str = repr(user)
    assert "User(" in repr_str
    assert "testuser" in repr_str
    db.close()


def test_device_binding_default_values(setup_auth_db):
    """测试设备绑定默认值"""
    from lelamp.database.session import get_db_session
    db = get_db_session()

    # 创建用户
    user = User(
        username="testuser",
        email="test@example.com",
        hashed_password="hashed_password_here"
    )
    db.add(user)
    db.flush()

    # 创建设备绑定（不指定可选参数）
    binding = DeviceBinding(
        user_id=user.id,
        device_id="lelamp_001",
        device_secret="secret_123"
    )
    db.add(binding)
    db.commit()
    db.refresh(binding)

    assert binding.permission_level == "member"  # 默认值
    assert binding.is_active is True  # 默认值
    assert binding.bound_at is not None  # 自动设置
    db.close()
