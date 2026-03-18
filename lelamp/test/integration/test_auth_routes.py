"""
Integration tests for authentication API routes.

Tests user registration, login, token refresh, and device binding endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from lelamp.api.app import app
from lelamp.database.session import get_db_session
from lelamp.database.base import Base, engine
from lelamp.api.services.auth_service import AuthService

client = TestClient(app)

@pytest.fixture(scope="function")
def setup_db():
    """创建测试数据库"""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def auth_token(setup_db):
    """创建认证令牌"""
    # 注册用户
    user = AuthService.register_user(
        get_db_session(),
        username="apitest",
        email="api@example.com",
        password="apipass123"
    )

    # 创建令牌
    token = AuthService.create_access_token(
        data={"sub": "apitest", "user_id": user.id}
    )
    return token


def test_register_user(setup_db):
    """测试用户注册"""
    response = client.post(
        "/api/auth/register",
        json={
            "username": "testuser",
            "email": "test@example.com",
            "password": "testpass123"
        }
    )

    assert response.status_code == 201
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"


def test_register_duplicate_username(setup_db):
    """测试重复用户名注册"""
    # 第一次注册
    client.post(
        "/api/auth/register",
        json={
            "username": "duplicate",
            "email": "user1@example.com",
            "password": "testpass123"
        }
    )

    # 第二次注册相同用户名
    response = client.post(
        "/api/auth/register",
        json={
            "username": "duplicate",
            "email": "user2@example.com",
            "password": "testpass123"
        }
    )

    assert response.status_code == 400
    assert "already exists" in response.json()["detail"]


def test_login_user(setup_db):
    """测试用户登录"""
    # 先注册
    client.post(
        "/api/auth/register",
        json={
            "username": "loginuser",
            "email": "login@example.com",
            "password": "loginpass123"
        }
    )

    # 登录
    response = client.post(
        "/api/auth/login",
        data={
            "username": "loginuser",
            "password": "loginpass123"
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data


def test_login_wrong_password(setup_db):
    """测试错误密码登录"""
    # 先注册
    client.post(
        "/api/auth/register",
        json={
            "username": "wrongpass",
            "email": "wrong@example.com",
            "password": "correctpass"
        }
    )

    # 错误密码登录
    response = client.post(
        "/api/auth/login",
        data={
            "username": "wrongpass",
            "password": "wrongpass"
        }
    )

    assert response.status_code == 401


def test_get_current_user(auth_token, setup_db):
    """测试获取当前用户信息"""
    response = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {auth_token}"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["username"] == "apitest"
    assert data["email"] == "api@example.com"
    assert data["is_active"] is True
    assert data["is_admin"] is False


def test_get_current_user_no_token():
    """测试无认证获取用户信息"""
    response = client.get("/api/auth/me")

    assert response.status_code == 401  # 未授权


def test_get_current_user_invalid_token():
    """测试无效令牌获取用户信息"""
    response = client.get(
        "/api/auth/me",
        headers={"Authorization": "Bearer invalid_token_12345"}
    )

    assert response.status_code == 401


def test_bind_device(auth_token, setup_db):
    """测试设备绑定"""
    response = client.post(
        "/api/auth/bind-device",
        headers={"Authorization": f"Bearer {auth_token}"},
        json={
            "device_id": "lelamp_test_001",
            "device_secret": "secret_test_123"
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert data["device_id"] == "lelamp_test_001"
    assert data["permission_level"] == "admin"


def test_bind_device_no_token(setup_db):
    """测试无认证绑定设备"""
    response = client.post(
        "/api/auth/bind-device",
        json={
            "device_id": "lelamp_test_002",
            "device_secret": "secret_test_456"
        }
    )

    assert response.status_code == 401  # 未授权


def test_bind_device_duplicate(auth_token, setup_db):
    """测试重复绑定设备"""
    # 第一次绑定
    client.post(
        "/api/auth/bind-device",
        headers={"Authorization": f"Bearer {auth_token}"},
        json={
            "device_id": "lelamp_duplicate",
            "device_secret": "secret_dup_123"
        }
    )

    # 第二次绑定相同设备
    response = client.post(
        "/api/auth/bind-device",
        headers={"Authorization": f"Bearer {auth_token}"},
        json={
            "device_id": "lelamp_duplicate",
            "device_secret": "secret_dup_456"
        }
    )

    assert response.status_code == 400
    assert "already bound" in response.json()["detail"]


def test_refresh_token_valid(setup_db):
    """测试有效的刷新令牌"""
    from lelamp.database.models_auth import RefreshToken
    db = get_db_session()

    # 创建用户和刷新令牌
    user = AuthService.register_user(
        db,
        username="refreshuser",
        email="refresh@example.com",
        password="refreshpass123"
    )
    refresh_token = AuthService.create_refresh_token(user.id, db)

    # 使用刷新令牌获取新令牌
    response = client.post(
        "/api/auth/refresh-token",
        json={"refresh_token": refresh_token}
    )

    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"

    # 验证旧令牌已被撤销
    old_token_record = db.query(RefreshToken).filter(
        RefreshToken.token == refresh_token
    ).first()
    assert old_token_record.revoked is True

    db.close()


def test_refresh_token_invalid():
    """测试无效的刷新令牌"""
    response = client.post(
        "/api/auth/refresh-token",
        json={"refresh_token": "invalid_refresh_token_123"}
    )

    assert response.status_code == 401
