"""
Integration tests for authentication middleware.
"""
import pytest
from fastapi.testclient import TestClient
from lelamp.api.app import app
from lelamp.database.base import Base, engine
from lelamp.database.session import get_db_session
from lelamp.api.services.auth_service import AuthService

client = TestClient(app)

@pytest.fixture(scope="function")
def setup_db():
    """创建测试数据库"""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def auth_tokens(setup_db):
    """创建认证令牌"""
    db = get_db_session()
    user = AuthService.register_user(
        db,
        username="testuser",
        email="test@example.com",
        password="testpass123"
    )

    access_token = AuthService.create_access_token(
        data={"sub": "testuser", "user_id": user.id}
    )
    db.close()
    return {"access_token": access_token, "user": user}


def test_get_current_user_valid_token(auth_tokens, setup_db):
    """测试有效令牌获取用户"""
    response = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {auth_tokens['access_token']}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["username"] == "testuser"


def test_get_current_user_no_token():
    """测试无令牌访问"""
    response = client.get("/api/auth/me")
    assert response.status_code == 401


def test_get_current_user_invalid_token():
    """测试无效令牌"""
    response = client.get(
        "/api/auth/me",
        headers={"Authorization": "Bearer invalid_token"}
    )
    assert response.status_code == 401
