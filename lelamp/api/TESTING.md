# LeLamp API 测试文档

本文档提供 LeLamp API 的测试指南，包括单元测试、集成测试、端到端测试和性能测试。

---

## 目录

- [运行测试](#运行测试)
- [单元测试](#单元测试)
- [集成测试](#集成测试)
- [端到端测试](#端到端测试)
- [API 测试](#api-测试)
- [性能测试](#性能测试)
- [手动测试](#手动测试)
- [测试覆盖率](#测试覆盖率)

---

## 运行测试

### 安装测试依赖

```bash
# 安装测试依赖
uv sync --extra test

# 或安装所有依赖
uv sync --all-extras
```

### 运行所有测试

```bash
# 运行所有测试
uv run pytest

# 运行测试并显示覆盖率
uv run pytest --cov=lelamp --cov-report=html

# 运行测试并生成详细报告
uv run pytest -v --tb=short
```

### 运行特定测试

```bash
# 运行特定测试文件
uv run pytest lelamp/test/integration/test_e2e.py

# 运行特定测试类
uv run pytest lelamp/test/integration/test_e2e.py::TestDeviceLifecycle

# 运行特定测试函数
uv run pytest lelamp/test/integration/test_e2e.py::TestDeviceLifecycle::test_complete_lifecycle

# 运行匹配模式的测试
uv run pytest -k "test_device"
```

### 测试选项

```bash
# 并行运行测试（需要 pytest-xdist）
uv run pytest -n auto

# 只运行失败的测试
uv run pytest --lf

# 遇到第一个失败时停止
uv run pytest -x

# 显示详细输出
uv run pytest -vv

# 显示打印输出
uv run pytest -s
```

---

## 单元测试

单元测试测试单个函数或类的功能。

### 创建单元测试

```python
# tests/unit/test_models.py
import pytest
from lelamp.api.models.requests import CommandRequest

def test_command_request_valid():
    """测试有效的命令请求"""
    cmd = CommandRequest(
        type="motor_move",
        action="move_joint",
        params={"joint_name": "base_yaw", "position": 45.0}
    )
    assert cmd.type == "motor_move"
    assert cmd.action == "move_joint"

def test_command_request_invalid_type():
    """测试无效的命令类型"""
    with pytest.raises(ValueError):
        CommandRequest(
            type="invalid_type",
            action="test"
        )

def test_command_request_missing_fields():
    """测试缺少必需字段"""
    with pytest.raises(ValueError):
        CommandRequest(type="motor_move")  # 缺少 action
```

### 运行单元测试

```bash
# 运行所有单元测试
uv run pytest lelamp/test/unit/

# 运行特定单元测试
uv run pytest lelamp/test/unit/test_models.py
```

---

## 集成测试

集成测试测试多个组件之间的交互。

### 数据库集成测试

```python
# tests/integration/test_database.py
import pytest
from lelamp.database import crud
from lelamp.database.models import Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

@pytest.fixture(scope="function")
def db_session():
    """创建测试数据库会话"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)

    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    yield session

    session.close()
    Base.metadata.drop_all(bind=engine)

def test_create_conversation(db_session):
    """测试创建对话记录"""
    conv = crud.create_conversation(
        db_session,
        lamp_id="test_lamp",
        messages=[{"role": "user", "content": "test"}],
        duration=1000
    )

    assert conv.id is not None
    assert conv.lamp_id == "test_lamp"
    assert conv.duration == 1000

def test_get_conversations_by_lamp_id(db_session):
    """测试查询对话记录"""
    # 创建测试数据
    crud.create_conversation(
        db_session,
        lamp_id="test_lamp",
        messages=[],
        duration=1000
    )

    # 查询
    conversations = crud.get_conversations_by_lamp_id(db_session, "test_lamp")

    assert len(conversations) == 1
    assert conversations[0].lamp_id == "test_lamp"
```

### API 集成测试

```python
# tests/integration/test_api.py
from fastapi.testclient import TestClient
from lelamp.api.app import app
import pytest

@pytest.fixture(scope="function")
def client():
    """创建测试客户端"""
    return TestClient(app)

def test_get_devices(client):
    """测试获取设备列表"""
    response = client.get("/api/devices")
    assert response.status_code == 200

    data = response.json()
    assert "devices" in data
    assert isinstance(data["devices"], list)

def test_get_device_state(client):
    """测试获取设备状态"""
    response = client.get("/api/devices/test_lamp/state")
    assert response.status_code == 200

    data = response.json()
    assert data["lamp_id"] == "test_lamp"
    assert "status" in data

def test_send_command(client):
    """测试发送命令"""
    response = client.post(
        "/api/devices/test_lamp/command",
        json={
            "type": "motor_move",
            "action": "move_joint",
            "params": {"joint_name": "base_yaw", "position": 45.0}
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "command_id" in data
```

### 运行集成测试

```bash
# 运行所有集成测试
uv run pytest lelamp/test/integration/

# 运行特定集成测试
uv run pytest lelamp/test/integration/test_api.py
```

---

## 端到端测试

端到端测试测试完整的用户场景。

### E2E 测试场景

```python
# tests/integration/test_e2e.py
import pytest
from fastapi.testclient import TestClient
from lelamp.api.app import app
from lelamp.database import crud

@pytest.fixture(scope="function")
def client_with_db(db_session):
    """创建带数据库的测试客户端"""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()

def test_complete_device_lifecycle(client_with_db, db_session):
    """测试完整设备生命周期"""
    lamp_id = "test_lamp_001"

    # 1. 创建设备状态
    state = crud.create_device_state(
        db_session,
        lamp_id=lamp_id,
        conversation_state="idle",
        motor_positions={},
        light_color={"r": 255, "g": 244, "b": 229},
        health_status={"overall": "healthy", "motors": []},
        uptime_seconds=3600
    )
    assert state is not None

    # 2. 发送命令
    response = client_with_db.post(
        f"/api/devices/{lamp_id}/command",
        json={
            "type": "motor_move",
            "action": "move_joint",
            "params": {"joint_name": "base_yaw", "position": 45.0}
        }
    )
    assert response.status_code == 200

    # 3. 查询设备状态
    response = client_with_db.get(f"/api/devices/{lamp_id}/state")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"

    # 4. 获取操作日志
    response = client_with_db.get(f"/api/devices/{lamp_id}/operations")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] >= 1

def test_multi_device_isolation(client_with_db, db_session):
    """测试多设备隔离"""
    # 创建多个设备
    lamp_ids = ["lamp_001", "lamp_002", "lamp_003"]

    for lamp_id in lamp_ids:
        crud.create_conversation(
            db_session,
            lamp_id=lamp_id,
            messages=[],
            duration=1000
        )

    # 验证设备隔离
    for lamp_id in lamp_ids:
        response = client_with_db.get(f"/api/devices/{lamp_id}/conversations")
        assert response.status_code == 200

        data = response.json()
        for conv in data["conversations"]:
            assert conv["lamp_id"] == lamp_id

def test_error_handling(client_with_db):
    """测试错误处理"""
    # 无效的 lamp_id
    response = client_with_db.get("/api/devices/invalid id/state")
    assert response.status_code == 400

    # 不存在的设备
    response = client_with_db.get("/api/devices/nonexistent_lamp/state")
    assert response.status_code == 200  # 返回默认离线状态

    # 格式错误的命令
    response = client_with_db.post(
        "/api/devices/test_lamp/command",
        json={"type": "test"}  # 缺少 action
    )
    assert response.status_code == 400
```

### 运行 E2E 测试

```bash
# 运行 E2E 测试
uv run pytest lelamp/test/integration/test_e2e.py -v

# 运行特定场景
uv run pytest lelamp/test/integration/test_e2e.py::TestDeviceLifecycle -v
```

---

## API 测试

### 使用 Postman

#### 导入集合

1. 下载 Postman 集合：`lelamp-api.postman_collection.json`
2. 在 Postman 中导入：File > Import > 选择文件
3. 设置环境变量：
   - `base_url`: `http://localhost:8000/api`
   - `lamp_id`: `lelamp_001`

#### 测试集合

```
LeLamp API Collection
├── Health Check
│   └── GET /health
├── Devices
│   ├── GET /api/devices
│   ├── GET /api/devices/{{lamp_id}}/state
│   ├── POST /api/devices/{{lamp_id}}/command
│   ├── GET /api/devices/{{lamp_id}}/conversations
│   ├── GET /api/devices/{{lamp_id}}/operations
│   ├── GET /api/devices/{{lamp_id}}/health
│   └── GET /api/devices/{{lamp_id}}/statistics
└── History
    ├── GET /api/history/conversations/{id}
    └── GET /api/history/operations/{id}
```

#### Postman 测试脚本

```javascript
// Tests 标签页
pm.test("Status code is 200", function () {
    pm.response.to.have.status(200);
});

pm.test("Response has correct structure", function () {
    var jsonData = pm.response.json();
    pm.expect(jsonData).to.have.property("devices");
    pm.expect(jsonData.devices).to.be.an("array");
});

// 设置环境变量
const lampId = "lelamp_001";
pm.environment.set("lamp_id", lampId);

// 从响应中提取数据
const responseJson = pm.response.json();
const commandId = responseJson.command_id;
pm.environment.set("command_id", commandId);
```

### 使用 Insomnia

#### 创建工作空间

1. 创建新的工作空间：`LeLamp API`
2. 创建环境：`Development`
3. 设置环境变量：
   - `base_url`: `http://localhost:8000/api`
   - `lamp_id`: `lelamp_001`

#### 请求模板

```yaml
# 获取设备列表
GET: {{base_url}}/devices

# 获取设备状态
GET: {{base_url}}/devices/{{lamp_id}}/state

# 发送命令
POST: {{base_url}}/devices/{{lamp_id}}/command
Content-Type: application/json

{
  "type": "motor_move",
  "action": "move_joint",
  "params": {
    "joint_name": "base_yaw",
    "position": 45.0
  }
}
```

---

## 性能测试

### 使用 Locust

#### 安装 Locust

```bash
pip install locust
# 或
uv add --dev locust
```

#### 创建 Locust 测试

```python
# locustfile.py
from locust import HttpUser, task, between

class LeLampUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        """每个用户启动时执行"""
        self.client.get("/health")

    @task(3)
    def get_device_state(self):
        """获取设备状态（高频率）"""
        self.client.get("/api/devices/lelamp_001/state")

    @task(2)
    def get_devices(self):
        """获取设备列表（中频率）"""
        self.client.get("/api/devices")

    @task(1)
    def send_command(self):
        """发送命令（低频率）"""
        self.client.post(
            "/api/devices/lelamp_001/command",
            json={
                "type": "motor_move",
                "action": "move_joint",
                "params": {"joint_name": "base_yaw", "position": 45.0}
            }
        )

    @task(1)
    def get_conversations(self):
        """获取对话记录（低频率）"""
        self.client.get("/api/devices/lelamp_001/conversations")
```

#### 运行 Locust

```bash
# 启动 Locust Web UI
uv run locust -f locustfile.py

# 或直接运行（无 Web UI）
uv run locust -f locustfile.py --headless \
  --users 100 \
  --spawn-rate 10 \
  --run-time 1m \
  --host http://localhost:8000
```

### 使用 Apache Bench

```bash
# 安装
sudo apt install apache2-utils

# 测试 GET 请求
ab -n 1000 -c 10 http://localhost:8000/api/devices

# 测试 POST 请求
ab -n 1000 -c 10 -p command.json -T application/json \
  http://localhost:8000/api/devices/lelamp_001/command
```

### 使用 wrk

```bash
# 安装
sudo apt install wrk

# 运行测试
wrk -t4 -c100 -d30s http://localhost:8000/api/devices
```

---

## 手动测试

### 测试检查清单

#### 功能测试

- [ ] 获取设备列表
- [ ] 获取设备状态
- [ ] 发送设备命令
- [ ] 获取对话记录
- [ ] 获取操作日志
- [ ] 获取健康状态
- [ ] 获取统计数据
- [ ] WebSocket 连接
- [ ] WebSocket 订阅
- [ ] 实时消息推送

#### 错误处理

- [ ] 无效的 lamp_id 格式
- [ ] 不存在的设备
- [ ] 格式错误的命令
- [ ] 无效的分页参数
- [ ] 无效的时间参数
- [ ] 数据库连接失败
- [ ] WebSocket 连接失败

#### 性能测试

- [ ] 高并发请求
- [ ] 大数据量查询
- [ ] 长时间运行
- [ ] 内存泄漏检测
- [ ] 响应时间监控

### 测试脚本

```bash
#!/bin/bash
# test_api.sh - API 测试脚本

BASE_URL="http://localhost:8000/api"
LAMP_ID="lelamp_001"

echo "开始 API 测试..."

# 1. 健康检查
echo "1. 健康检查"
curl -s -X GET "http://localhost:8000/health" | jq '.'

# 2. 获取设备列表
echo "2. 获取设备列表"
curl -s -X GET "$BASE_URL/devices" | jq '.'

# 3. 获取设备状态
echo "3. 获取设备状态"
curl -s -X GET "$BASE_URL/devices/$LAMP_ID/state" | jq '.'

# 4. 发送命令
echo "4. 发送命令"
curl -s -X POST "$BASE_URL/devices/$LAMP_ID/command" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "motor_move",
    "action": "move_joint",
    "params": {"joint_name": "base_yaw", "position": 45.0}
  }' | jq '.'

# 5. 获取对话记录
echo "5. 获取对话记录"
curl -s -X GET "$BASE_URL/devices/$LAMP_ID/conversations" | jq '.'

# 6. 获取操作日志
echo "6. 获取操作日志"
curl -s -X GET "$BASE_URL/devices/$LAMP_ID/operations" | jq '.'

# 7. 获取健康状态
echo "7. 获取健康状态"
curl -s -X GET "$BASE_URL/devices/$LAMP_ID/health" | jq '.'

# 8. 获取统计数据
echo "8. 获取统计数据"
curl -s -X GET "$BASE_URL/devices/$LAMP_ID/statistics" | jq '.'

echo "API 测试完成"
```

---

## 测试覆盖率

### 生成覆盖率报告

```bash
# 生成 HTML 覆盖率报告
uv run pytest --cov=lelamp --cov-report=html --cov-report=term

# 查看报告
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### 覆盖率目标

| 模块 | 目标覆盖率 | 当前覆盖率 |
|------|-----------|-----------|
| API 路由 | 80% | - |
| 数据模型 | 90% | - |
| 数据库 CRUD | 85% | - |
| WebSocket | 75% | - |
| 整体 | 70% | - |

### 提高覆盖率

1. **识别未测试的代码**：
```bash
uv run pytest --cov=lelamp --cov-report=term-missing
```

2. **为未覆盖的代码添加测试**

3. **使用分支覆盖**：
```bash
uv run pytest --cov=lelamp --cov-branch --cov-report=html
```

---

## 持续集成

### GitHub Actions

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.12'

    - name: Install UV
      run: curl -LsSf https://astral.sh/uv/install.sh | sh

    - name: Install dependencies
      run: uv sync --all-extras

    - name: Run tests
      run: uv run pytest --cov=lelamp --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v2
      with:
        file: ./coverage.xml
```

### 本地 CI 测试

```bash
# 模拟 CI 环境
uv run pytest --cov=lelamp --cov-report=xml --junitxml=test-results.xml

# 运行 linter
uv run pylint lelamp/api/

# 运行 type checker
uv run mypy lelamp/api/

# 运行 formatter
uv run black --check lelamp/api/
```

---

## 测试最佳实践

### 1. 测试命名

```python
# 好的命名
def test_get_device_state_returns_online_status():
    pass

# 不好的命名
def test_device():
    pass
```

### 2. 测试组织

```python
class TestDeviceState:
    """设备状态测试"""

    def test_get_device_state_online(self):
        """测试在线设备状态"""
        pass

    def test_get_device_state_offline(self):
        """测试离线设备状态"""
        pass
```

### 3. 使用 Fixture

```python
@pytest.fixture
def sample_device_state():
    """示例设备状态"""
    return {
        "lamp_id": "test_lamp",
        "status": "online",
        "conversation_state": "idle"
    }

def test_with_fixture(sample_device_state):
    assert sample_device_state["status"] == "online"
```

### 4. 测试隔离

```python
# 每个测试应该独立
def test_create_device(db_session):
    device = crud.create_device(db_session, ...)
    assert device is not None

def test_query_device(db_session):
    # 不依赖前面的测试
    device = crud.create_device(db_session, ...)
    result = crud.get_device(db_session, device.id)
    assert result is not None
```

### 5. 清理资源

```python
@pytest.fixture
def temp_file():
    """创建临时文件"""
    import tempfile
    fd, path = tempfile.mkstemp()
    yield path
    # 清理
    os.close(fd)
    os.unlink(path)
```

---

## 故障排除

### 常见问题

#### 1. 测试数据库锁定

```bash
# 错误：sqlite3.OperationalError: database is locked

# 解决：使用内存数据库或文件数据库
TEST_DATABASE_URL = "sqlite:///:memory:"
```

#### 2. 异步测试超时

```python
# 增加超时时间
@pytest.mark.asyncio(timeout=10.0)
async def test_slow_operation():
    await slow_function()
```

#### 3. Import 错误

```bash
# 错误：ModuleNotFoundError: No module named 'lelamp'

# 解决：确保 PYTHONPATH 正确
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### 4. Fixture 未找到

```bash
# 错误：fixture 'db_session' not found

# 解决：确保 fixture 在 conftest.py 中定义
# 或使用正确的路径导入
```

---

## 更多资源

- [Pytest 文档](https://docs.pytest.org/)
- [FastAPI 测试指南](https://fastapi.tiangolo.com/tutorial/testing/)
- [Locust 文档](https://locust.io/)
- [API 文档](./API_DOCUMENTATION.md)
- [部署指南](./DEPLOYMENT_GUIDE.md)
