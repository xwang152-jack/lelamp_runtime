"""
Integration tests for authentication service.

Tests JWT token generation, user authentication, registration, and device binding.
"""
import pytest
from lelamp.api.services.auth_service import AuthService
from lelamp.database.models_auth import User
from lelamp.database.session import get_db_session
from lelamp.database.base import Base, engine


@pytest.fixture(scope="function")
def setup_db():
    """创建测试数据库"""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


def test_password_hashing():
    """测试密码哈希"""
    password = "test_password_123"
    hashed = AuthService.hash_password(password)

    assert hashed != password
    assert len(hashed) == 60  # bcrypt 哈希长度
    assert AuthService.verify_password(password, hashed) is True
    assert AuthService.verify_password("wrong_password", hashed) is False


def test_user_registration(setup_db):
    """测试用户注册"""
    db = get_db_session()
    user = AuthService.register_user(
        db,
        username="testuser",
        email="test@example.com",
        password="testpass123"
    )

    assert user.id is not None
    assert user.username == "testuser"
    assert user.email == "test@example.com"
    assert user.is_active is True
    assert user.is_admin is False
    db.close()


def test_user_registration_duplicate_username(setup_db):
    """测试重复用户名注册"""
    db = get_db_session()

    # 第一个用户
    AuthService.register_user(
        db,
        username="duplicate",
        email="user1@example.com",
        password="testpass123"
    )

    # 第二个用户（相同用户名）
    with pytest.raises(ValueError, match="Username already exists"):
        AuthService.register_user(
            db,
            username="duplicate",
            email="user2@example.com",
            password="testpass123"
        )

    db.close()


def test_user_registration_duplicate_email(setup_db):
    """测试重复邮箱注册"""
    db = get_db_session()

    # 第一个用户
    AuthService.register_user(
        db,
        username="user1",
        email="duplicate@example.com",
        password="testpass123"
    )

    # 第二个用户（相同邮箱）
    with pytest.raises(ValueError, match="Email already exists"):
        AuthService.register_user(
            db,
            username="user2",
            email="duplicate@example.com",
            password="testpass123"
        )

    db.close()


def test_token_creation_and_verification():
    """测试令牌创建和验证"""
    data = {"user_id": 1, "username": "testuser"}
    token = AuthService.create_access_token(data)

    assert token is not None
    assert isinstance(token, str)

    # 验证令牌
    payload = AuthService.verify_token(token, "access")
    assert payload is not None
    assert payload["user_id"] == 1
    assert payload["username"] == "testuser"
    assert payload["type"] == "access"


def test_token_verification_invalid_token():
    """测试无效令牌验证"""
    payload = AuthService.verify_token("invalid_token_12345", "access")
    assert payload is None


def test_token_verification_wrong_type():
    """测试错误令牌类型验证"""
    data = {"user_id": 1, "username": "testuser"}
    access_token = AuthService.create_access_token(data)

    # 尝试作为 refresh token 验证
    payload = AuthService.verify_token(access_token, "refresh")
    assert payload is None


def test_user_authentication(setup_db):
    """测试用户认证"""
    db = get_db_session()

    # 先注册用户
    user = AuthService.register_user(
        db,
        username="authuser",
        email="auth@example.com",
        password="authpass123"
    )

    # 测试正确密码
    auth_user = AuthService.authenticate_user(db, "authuser", "authpass123")
    assert auth_user is not None
    assert auth_user.username == "authuser"
    assert auth_user.email == "auth@example.com"

    # 测试错误密码
    auth_user = AuthService.authenticate_user(db, "authuser", "wrongpassword")
    assert auth_user is None

    # 测试不存在的用户
    auth_user = AuthService.authenticate_user(db, "nonexistent", "password")
    assert auth_user is None

    db.close()


def test_refresh_token_creation(setup_db):
    """测试刷新令牌创建"""
    from datetime import timedelta
    db = get_db_session()

    # 创建用户
    user = AuthService.register_user(
        db,
        username="refreshuser",
        email="refresh@example.com",
        password="refreshpass123"
    )

    # 创建刷新令牌
    token = AuthService.create_refresh_token(user.id, db)

    assert token is not None
    assert isinstance(token, str)

    # 验证刷新令牌
    payload = AuthService.verify_token(token, "refresh")
    assert payload is not None
    assert payload["user_id"] == user.id
    assert payload["type"] == "refresh"

    # 检查数据库中的令牌记录
    from lelamp.database.models_auth import RefreshToken
    refresh_token_record = db.query(RefreshToken).filter(
        RefreshToken.token == token
    ).first()
    assert refresh_token_record is not None
    assert refresh_token_record.user_id == user.id
    assert refresh_token_record.revoked is False

    db.close()


def test_device_binding(setup_db):
    """测试设备绑定"""
    db = get_db_session()

    # 创建用户
    user = AuthService.register_user(
        db,
        username="deviceuser",
        email="device@example.com",
        password="devicepass123"
    )

    # 绑定设备
    binding = AuthService.bind_device(
        db,
        user_id=user.id,
        device_id="lelamp_001",
        device_secret="secret_123"
    )

    assert binding.id is not None
    assert binding.device_id == "lelamp_001"
    assert binding.permission_level == "admin"
    assert binding.is_active is True

    db.close()


def test_device_binding_duplicate(setup_db):
    """测试重复设备绑定"""
    db = get_db_session()

    # 创建用户
    user = AuthService.register_user(
        db,
        username="dupuser",
        email="dup@example.com",
        password="duppass123"
    )

    # 第一次绑定
    AuthService.bind_device(
        db,
        user_id=user.id,
        device_id="lelamp_002",
        device_secret="secret_456"
    )

    # 第二次绑定相同设备
    with pytest.raises(ValueError, match="Device already bound"):
        AuthService.bind_device(
            db,
            user_id=user.id,
            device_id="lelamp_002",
            device_secret="secret_789"
        )

    db.close()
