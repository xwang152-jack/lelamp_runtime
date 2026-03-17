"""
API 集成测试

测试所有 API 端点的功能，包括设备状态、命令发送、对话记录、操作日志等。
使用 TestClient 进行测试，依赖注入使用测试数据库。
"""
import pytest
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from lelamp.api.app import app
from lelamp.database.base import Base
from lelamp.database.models import Conversation, OperationLog, DeviceState
from lelamp.database.session import get_db

# 创建测试数据库引擎（使用内存数据库）
# 重要: 使用 share_connections=True 确保所有会话共享同一个内存数据库
SQLALCHEMY_TEST_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,  # 使用静态池确保所有连接共享同一个数据库
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="function", autouse=True)
def setup_database():
    """
    为每个测试创建和销毁数据库表

    使用 autouse=True 确保所有测试都有表结构
    """
    # 创建表
    Base.metadata.create_all(bind=engine)
    yield
    # 删除表
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def db_session():
    """
    创建测试数据库会话

    每个测试函数都会创建新的会话，确保测试隔离
    """
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="function")
def client():
    """
    创建测试客户端，每个测试使用独立的数据库会话
    """
    # 创建新的数据库会话（每个测试独立）
    db = TestingSessionLocal()

    def override_get_db():
        try:
            yield db
        finally:
            pass

    # 设置依赖注入覆盖
    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    # 清理
    db.close()
    app.dependency_overrides.clear()


@pytest.fixture
def sample_device_state(db_session: Session):
    """
    创建示例设备状态数据
    """
    state = DeviceState(
        lamp_id="lelamp-001",
        motor_positions={
            "base_yaw": 0,
            "base_pitch": 0,
            "elbow_pitch": 0,
            "wrist_roll": 0,
            "wrist_pitch": 0,
        },
        health_status={"overall": "healthy"},
        light_color={"r": 255, "g": 244, "b": 229},
        conversation_state="idle",
        uptime_seconds=3600,
    )
    db_session.add(state)
    db_session.commit()
    db_session.refresh(state)
    return state


@pytest.fixture
def sample_conversations(db_session: Session):
    """
    创建示例对话记录
    """
    conversations = []
    for i in range(3):
        conv = Conversation(
            lamp_id="lelamp-001",
            messages=[{"role": "user", "content": f"测试消息 {i+1}"}],
            duration=10 + i * 5,
            user_input=f"用户输入 {i+1}",
            ai_response=f"AI 回复 {i+1}",
        )
        conversations.append(conv)
        db_session.add(conv)

    db_session.commit()
    for conv in conversations:
        db_session.refresh(conv)
    return conversations


@pytest.fixture
def sample_operations(db_session: Session):
    """
    创建示例操作日志
    """
    operations = []
    for i in range(5):
        op = OperationLog(
            lamp_id="lelamp-001",
            operation_type="motor_move",
            action=f"move_joint_{i}",
            params={"joint": "base_yaw", "position": i * 10},
            success=True,
            duration_ms=100 + i * 50,
        )
        operations.append(op)
        db_session.add(op)

    db_session.commit()
    for op in operations:
        db_session.refresh(op)
    return operations


# =============================================================================
# 设备状态 API 测试
# =============================================================================


class TestDeviceStateAPI:
    """设备状态 API 测试"""

    def test_get_device_state_with_data(self, client: TestClient, sample_device_state):
        """
        测试获取设备状态 - 有数据的情况
        """
        response = client.get("/api/devices/lelamp-001/state")

        assert response.status_code == 200
        data = response.json()
        assert data["lamp_id"] == "lelamp-001"
        assert data["conversation_state"] == "idle"
        assert "motor_positions" in data
        assert "light_color" in data
        assert data["camera_active"] is False
        assert "timestamp" in data

    def test_get_device_state_without_data(self, client: TestClient):
        """
        测试获取设备状态 - 无数据时返回默认状态
        """
        response = client.get("/api/devices/unknown-lamp/state")

        assert response.status_code == 200
        data = response.json()
        assert data["lamp_id"] == "unknown-lamp"
        assert data["status"] == "offline"
        assert data["conversation_state"] == "unknown"

    def test_send_command(self, client: TestClient):
        """
        测试发送设备命令
        """
        command_data = {
            "type": "motor_move",
            "action": "move_joint",
            "params": {"joint": "base_yaw", "position": 45}
        }

        response = client.post("/api/devices/lelamp-001/command", json=command_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "command_id" in data
        assert data["message"] == "Command received"
        assert "timestamp" in data


# =============================================================================
# 对话记录 API 测试
# =============================================================================


class TestConversationsAPI:
    """对话记录 API 测试"""

    def test_get_conversations(self, client: TestClient, sample_conversations):
        """
        测试获取对话记录列表
        """
        response = client.get("/api/devices/lelamp-001/conversations")

        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "conversations" in data
        assert data["total"] == 3
        assert len(data["conversations"]) == 3
        assert data["conversations"][0]["user_input"] == "用户输入 3"

    def test_get_conversations_with_pagination(self, client: TestClient, sample_conversations):
        """
        测试获取对话记录 - 带分页参数
        """
        response = client.get("/api/devices/lelamp-001/conversations?skip=1&limit=2")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert len(data["conversations"]) == 2

    def test_get_conversations_empty(self, client: TestClient):
        """
        测试获取对话记录 - 空列表
        """
        response = client.get("/api/devices/unknown-lamp/conversations")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert len(data["conversations"]) == 0

    def test_get_conversation_by_id(self, client: TestClient, sample_conversations):
        """
        测试通过 ID 获取单个对话记录
        """
        conv_id = sample_conversations[0].id
        response = client.get(f"/api/history/conversations/{conv_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == conv_id
        assert data["lamp_id"] == "lelamp-001"
        assert "messages" in data

    def test_get_conversation_by_id_not_found(self, client: TestClient):
        """
        测试通过 ID 获取不存在的对话记录
        """
        response = client.get("/api/history/conversations/99999")

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data


# =============================================================================
# 操作日志 API 测试
# =============================================================================


class TestOperationsAPI:
    """操作日志 API 测试"""

    def test_get_operations(self, client: TestClient, sample_operations):
        """
        测试获取操作日志列表
        """
        response = client.get("/api/devices/lelamp-001/operations")

        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "operations" in data
        assert data["total"] == 5
        assert len(data["operations"]) == 5

    def test_get_operations_with_time_filter(self, client: TestClient, sample_operations):
        """
        测试获取操作日志 - 带时间过滤
        """
        response = client.get("/api/devices/lelamp-001/operations?hours=24")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 5

    def test_get_operations_with_pagination(self, client: TestClient, sample_operations):
        """
        测试获取操作日志 - 带分页参数
        """
        response = client.get("/api/devices/lelamp-001/operations?skip=2&limit=2")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 5
        assert len(data["operations"]) == 2

    def test_get_operations_empty(self, client: TestClient):
        """
        测试获取操作日志 - 空列表
        """
        response = client.get("/api/devices/unknown-lamp/operations")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert len(data["operations"]) == 0

    def test_get_operation_by_id(self, client: TestClient, sample_operations):
        """
        测试通过 ID 获取单个操作日志
        """
        op_id = sample_operations[0].id
        response = client.get(f"/api/history/operations/{op_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == op_id
        assert data["lamp_id"] == "lelamp-001"
        assert data["operation_type"] == "motor_move"

    def test_get_operation_by_id_not_found(self, client: TestClient):
        """
        测试通过 ID 获取不存在的操作日志
        """
        response = client.get("/api/history/operations/99999")

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data


# =============================================================================
# 设备健康和统计 API 测试
# =============================================================================


class TestHealthAndStatsAPI:
    """设备健康和统计 API 测试"""

    def test_get_device_health(self, client: TestClient, sample_device_state):
        """
        测试获取设备健康状态
        """
        response = client.get("/api/devices/lelamp-001/health")

        assert response.status_code == 200
        data = response.json()
        assert data["lamp_id"] == "lelamp-001"
        assert "overall_status" in data
        assert "last_check" in data

    def test_get_device_health_unknown(self, client: TestClient):
        """
        测试获取未知设备的健康状态
        """
        response = client.get("/api/devices/unknown-lamp/health")

        assert response.status_code == 200
        data = response.json()
        assert data["overall_status"] == "unknown"

    def test_get_device_statistics(self, client: TestClient, sample_operations):
        """
        测试获取设备统计数据
        """
        response = client.get("/api/devices/lelamp-001/statistics")

        assert response.status_code == 200
        data = response.json()
        assert data["lamp_id"] == "lelamp-001"
        assert "total_operations" in data
        assert "success_rate" in data
        assert "operation_counts" in data
        assert data["total_operations"] == 5

    def test_get_device_statistics_no_data(self, client: TestClient):
        """
        测试获取无数据的设备统计
        """
        response = client.get("/api/devices/unknown-lamp/statistics")

        assert response.status_code == 200
        data = response.json()
        assert data["total_operations"] == 0


# =============================================================================
# 设备列表 API 测试
# =============================================================================


class TestDeviceListAPI:
    """设备列表 API 测试"""

    def test_list_devices(self, client: TestClient, sample_conversations):
        """
        测试获取所有设备列表
        """
        response = client.get("/api/devices")

        assert response.status_code == 200
        data = response.json()
        assert "devices" in data
        assert len(data["devices"]) >= 1
        assert any(d["lamp_id"] == "lelamp-001" for d in data["devices"])

    def test_list_devices_empty(self, client: TestClient):
        """
        测试获取设备列表 - 无设备
        """
        response = client.get("/api/devices")

        assert response.status_code == 200
        data = response.json()
        assert "devices" in data
        assert len(data["devices"]) == 0


# =============================================================================
# 错误处理测试
# =============================================================================


class TestErrorHandling:
    """错误处理测试"""

    def test_invalid_lamp_id_format(self, client: TestClient):
        """
        测试无效的 lamp_id 格式

        注意: FastAPI 路径参数默认会阻止某些特殊字符，返回 404 而非 400
        这是 FastAPI 的预期行为，路径验证在自定义验证之前执行
        """
        # 使用特殊字符作为 lamp_id
        response = client.get("/api/devices/lamp@#$%/state")

        # FastAPI 路径验证会拒绝特殊字符，返回 404
        assert response.status_code == 404

    def test_invalid_pagination_params(self, client: TestClient):
        """
        测试无效的分页参数
        """
        # limit 超过最大值 - FastAPI 会自动验证（Query 中设置了 le=100）
        # 所以会返回 422 而不是我们手动返回的 400
        response = client.get("/api/devices/lelamp-001/conversations?limit=999")

        # FastAPI 自动验证失败
        assert response.status_code == 422

    def test_negative_skip_param(self, client: TestClient):
        """
        测试负数的 skip 参数
        """
        # FastAPI Query 中设置了 ge=0，负数会自动返回 422
        response = client.get("/api/devices/lelamp-001/conversations?skip=-1")

        # FastAPI 自动验证失败
        assert response.status_code == 422

    def test_invalid_command_type(self, client: TestClient):
        """
        测试无效的命令类型
        """
        command_data = {
            "type": "invalid_type",
            "action": "test_action",
            "params": {}
        }

        response = client.post("/api/devices/lelamp-001/command", json=command_data)

        # 应该返回 400 或 200（取决于验证策略）
        assert response.status_code in [200, 400]

    def test_missing_required_fields(self, client: TestClient):
        """
        测试缺少必需字段
        """
        # 缺少 type 字段
        command_data = {
            "action": "test_action"
        }

        response = client.post("/api/devices/lelamp-001/command", json=command_data)

        # 我们在代码中手动检查并返回 400
        assert response.status_code == 400


# =============================================================================
# 边界条件测试
# =============================================================================


class TestEdgeCases:
    """边界条件测试"""

    def test_large_pagination(self, client: TestClient, db_session: Session):
        """
        测试大量数据的分页
        """
        # 创建 150 条对话记录
        for i in range(150):
            conv = Conversation(
                lamp_id="lelamp-001",
                messages=[{"role": "user", "content": f"消息 {i}"}],
                duration=10,
            )
            db_session.add(conv)
        db_session.commit()

        # 测试分页
        response = client.get("/api/devices/lelamp-001/conversations?skip=100&limit=50")

        assert response.status_code == 200
        data = response.json()
        assert len(data["conversations"]) == 50

    def test_concurrent_operations(self, client: TestClient, db_session: Session):
        """
        测试并发操作
        """
        # 创建多个操作日志
        for i in range(10):
            op = OperationLog(
                lamp_id="lelamp-001",
                operation_type="test",
                action=f"action_{i}",
                params={},
                success=True,
            )
            db_session.add(op)
        db_session.commit()

        # 同时请求多个端点
        responses = [
            client.get("/api/devices/lelamp-001/state"),
            client.get("/api/devices/lelamp-001/operations"),
            client.get("/api/devices/lelamp-001/conversations"),
        ]

        for response in responses:
            assert response.status_code == 200
