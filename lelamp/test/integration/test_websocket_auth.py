"""
Integration tests for WebSocket authentication.
"""
import pytest
from lelamp.api.app import app
from lelamp.database.base import Base, engine
from lelamp.database.session import get_db_session
from lelamp.api.services.auth_service import AuthService


@pytest.fixture(scope="function")
def setup_db():
    """创建测试数据库"""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def ws_token(setup_db):
    """创建 WebSocket 认证令牌"""
    db = get_db_session()
    user = AuthService.register_user(
        db, username="wsuser", email="ws@example.com", password="wspass123"
    )
    token = AuthService.create_access_token(
        data={"sub": "wsuser", "user_id": user.id}
    )
    db.close()
    return token


@pytest.mark.asyncio
async def test_websocket_with_valid_token(ws_token):
    """测试有效令牌连接 WebSocket"""
    from fastapi.testclient import TestClient

    client = TestClient(app)

    with client.websocket_connect(
        "/api/ws/test_lamp?token=" + ws_token
    ) as websocket:
        # 等待欢迎消息
        data = websocket.receive_json()
        assert data["type"] == "connected"
        assert data["lamp_id"] == "test_lamp"


@pytest.mark.asyncio
async def test_websocket_without_token():
    """测试无令牌连接 WebSocket (匿名连接)"""
    from fastapi.testclient import TestClient

    client = TestClient(app)

    # 允许匿名连接,但不会有用户信息
    with client.websocket_connect("/api/ws/test_lamp") as websocket:
        data = websocket.receive_json()
        assert data["type"] == "connected"
        assert data["lamp_id"] == "test_lamp"


@pytest.mark.asyncio
async def test_websocket_with_invalid_token():
    """测试无效令牌连接 WebSocket (应该允许匿名连接)"""
    from fastapi.testclient import TestClient

    client = TestClient(app)

    # 无效令牌应该允许匿名连接
    with client.websocket_connect(
        "/api/ws/test_lamp?token=invalid_token_12345"
    ) as websocket:
        data = websocket.receive_json()
        assert data["type"] == "connected"
        assert data["lamp_id"] == "test_lamp"
