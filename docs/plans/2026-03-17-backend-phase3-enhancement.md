# LeLamp 后端 Phase 3 功能增强实施计划

**日期**: 2026-03-17
**版本**: 1.0
**状态**: 待审批

---

## 1. Phase 3 概述

### 1.1 目标

在 Phase 2 模块化架构的基础上，添加关键的后端功能：
- **FastAPI 服务器** - RESTful API 和实时通信
- **数据持久化** - 对话历史、操作日志、设备状态
- **WebSocket 实时推送** - 设备状态实时同步
- **API 文档** - 自动生成的交互式文档

### 1.2 技术栈

| 类别 | 技术 | 版本 | 用途 |
|------|------|------|------|
| Web 框架 | FastAPI | ^0.110+ | 高性能 API 框架 |
| 数据验证 | Pydantic | ^2.0+ | 数据验证和序列化 |
| 数据库 | SQLite | 3.x+ | 本地轻量级数据库 |
| ORM | SQLAlchemy | ^2.0+ | Python ORM |
| WebSocket | FastAPI WebSocket | 内置 | 实时通信 |
| 异步支持 | asyncio | 内置 | 异步 I/O |
| API 文档 | Swagger/OpenAPI | 内置 | 自动文档生成 |

---

## 2. 任务分解

### Task 3.1: 创建 FastAPI 应用基础

**文件**: `lelamp/api/__init__.py`, `lelamp/api/app.py`

**目标**: 创建 FastAPI 应用，配置基础路由和中间件

**步骤**:

#### 3.1.1: 创建 API 目录结构

```bash
lelamp/
├── api/
│   ├── __init__.py
│   ├── app.py              # FastAPI 应用主文件
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── devices.py       # 设备相关路由
│   │   └── websocket.py     # WebSocket 路由
│   ├── models/
│   │   ├── __init__.py
│   │   ├── requests.py      # 请求模型
│   │   └── responses.py     # 响应模型
│   ├── dependencies.py      # 依赖注入
│   └── database.py          # 数据库配置
```

#### 3.1.2: 创建 FastAPI 应用

**文件**: `lelamp/api/app.py`

```python
"""
FastAPI 应用主文件
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger("lelamp.api")

# 创建 FastAPI 应用
app = FastAPI(
    title="LeLamp API",
    description="LeLamp 智能台灯 RESTful API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 生命周期事件
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时执行
    logger.info("LeLamp API 启动")
    yield
    # 关闭时执行
    logger.info("LeLamp API 关闭")

app.router.lifespan_context = lifespan

# 健康检查
@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "service": "lelamp-api"}

# 包含路由
from lelamp.api.routes import devices, websocket
app.include_router(devices.router, prefix="/api/devices", tags=["devices"])
app.include_router(websocket.router, prefix="/ws", tags=["websocket"])

# WebSocket 连接管理
active_connections: list[WebSocket] = []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 端点 - 实时推送设备状态"""
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # 处理客户端消息
            await websocket.send_text(f"Echo: {data}")
    except WebSocketDisconnect:
        active_connections.remove(websocket)

async def broadcast_to_websockets(message: dict):
    """向所有 WebSocket 连接广播消息"""
    for connection in active_connections:
        try:
            await connection.send_json(message)
        except:
            active_connections.remove(connection)

# 导出 broadcast 函数供其他模块使用
__all__ = ["app", "broadcast_to_websockets"]
```

#### 3.1.3: 创建请求和响应模型

**文件**: `lelamp/api/models/requests.py`

```python
"""
API 请求模型
"""
from pydantic import BaseModel, Field
from typing import Optional

class MotorControlRequest(BaseModel):
    """电机控制请求"""
    joint_name: str = Field(..., description="关节名称")
    position: float = Field(..., description="目标位置")
    speed: Optional[int] = Field(50, description="移动速度")

class RGBColorRequest(BaseModel):
    """RGB 颜色控制请求"""
    r: int = Field(..., ge=0, le=255, description="红色值 (0-255)")
    g: int = Field(..., ge=0, le=255, description="绿色值 (0-255)")
    b: int = Field(..., ge=0, le=255, description="蓝色值 (0-255)")
```

**文件**: `lelamp/api/models/responses.py`

```python
"""
API 响应模型
"""
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class DeviceStateResponse(BaseModel):
    """设备状态响应"""
    lamp_id: str
    status: str
    timestamp: datetime
    motor_positions: dict
    light_color: dict
    camera_active: bool

class ConversationResponse(BaseModel):
    """对话记录响应"""
    id: int
    timestamp: datetime
    user_message: str
    agent_response: str
    duration_ms: int

class HealthResponse(BaseModel):
    """健康状态响应"""
    lamp_id: str
    overall_status: str
    motors: list
    last_check: datetime
```

#### 3.1.4: 创建设备路由

**文件**: `lelamp/api/routes/devices.py`

```python
"""
设备相关 API 路由
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict
import logging

logger = logging.getLogger("lelamp.api.devices")

router = APIRouter()

# 依赖注入：获取设备实例
# 实际实现中需要从服务容器获取
def get_device_service():
    """获取设备服务实例（待实现）"""
    # TODO: 从服务容器获取实际的 MotorsService, RGBService 等
    return None

@router.get("/{lamp_id}/state")
async def get_device_state(lamp_id: str) -> Dict:
    """获取设备状态"""
    # TODO: 从服务获取实际状态
    return {
        "lamp_id": lamp_id,
        "status": "online",
        "motor_positions": {},
        "light_color": {"r": 255, "g": 244, "b": 229},
        "camera_active": False
    }

@router.post("/{lamp_id}/command")
async def send_device_command(
    lamp_id: str,
    command: dict,
    service = Depends(get_device_service)
) -> Dict:
    """发送设备命令"""
    # TODO: 实现命令发送逻辑
    logger.info(f"Sending command to {lamp_id}: {command}")
    return {"status": "success", "command": command}
```

#### 3.1.5: 添加依赖到 pyproject.toml

```toml
[project.optional-dependencies]
api = [
    "fastapi>=0.110.0",
    "uvicorn[standard]>=0.29.0",
    "pydantic>=2.0.0",
    "sqlalchemy>=2.0.0",
    "aiosqlite>=0.19.0",
]
```

#### 3.1.6: 创建 API 启动脚本

**文件**: `scripts/run_api_server.py`

```python
"""
FastAPI 服务器启动脚本
"""
import uvicorn
from lelamp.api.app import app

if __name__ == "__main__":
    uvicorn.run(
        "lelamp.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # 开发模式
        log_level="info"
    )
```

#### 3.1.7: 测试 API

```bash
# 安装 API 依赖
uv sync --extra api

# 启动 API 服务器
uv run python scripts/run_api_server.py

# 访问 API 文档
open http://localhost:8000/docs
```

---

### Task 3.2: 实现数据持久化

**目标**: 使用 SQLite + SQLAlchemy 实现数据存储

#### 3.2.1: 创建数据库模型

**文件**: `lelamp/database/models.py`

```python
"""
数据库 ORM 模型
"""
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from typing import Optional

Base = declarative_base()

class Conversation(Base):
    """对话记录"""
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    user_message = Column(Text, nullable=False)
    agent_response = Column(Text, nullable=False)
    duration_ms = Column(Integer, nullable=True)
    session_id = Column(String, nullable=True)  # LiveKit session ID

class OperationLog(Base):
    """操作日志"""
    __tablename__ = "operation_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    operation_type = Column(String(50), nullable=False)  # motor, rgb, vision, etc.
    action = Column(String(100), nullable=False)
    params = Column(Text)  # JSON string
    success = Column(Boolean, default=True, nullable=False)
    error_message = Column(Text, nullable=True)
    session_id = Column(String, nullable=True)

class DeviceState(Base):
    """设备状态快照"""
    __tablename__ = "device_states"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    lamp_id = Column(String(100), nullable=False, index=True)
    status = Column(String(20), nullable=False)  # online, offline, error
    motor_positions = Column(Text)  # JSON string
    health_status = Column(Text)  # JSON string
    light_color = Column(Text)  # JSON string
    camera_active = Column(Boolean, default=False)

class UserSettings(Base):
    """用户设置"""
    __tablename__ = "user_settings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(100), unique=True, nullable=False)
    theme = Column(String(20), default="light")  # light, dark, auto
    language = Column(String(10), default="zh")  # zh, en
    notifications_enabled = Column(Boolean, default=True)
    preferred_device = Column(String(100))
```

#### 3.2.2: 创建数据库会话

**文件**: `lelamp/database/session.py`

```python
"""
数据库会话管理
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
from pathlib import Path
import logging

logger = logging.getLogger("lelamp.database")

# 数据库文件路径
DB_PATH = Path("lelamp.db")

# 创建引擎
engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,  # SQLite 单线程
    echo=False,  # 设置 True 打印 SQL
)

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@contextmanager
def get_db():
    """获取数据库会话"""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

def init_db():
    """初始化数据库表"""
    from lelamp.database.models import Base

    logger.info(f"初始化数据库: {DB_PATH.absolute()}")
    Base.metadata.create_all(bind=engine)
    logger.info("数据库表创建完成")
```

#### 3.2.3: 创建 CRUD 操作

**文件**: `lelamp/database/crud.py`

```python
"""
CRUD 操作
"""
from sqlalchemy.orm import Session
from datetime import datetime
from typing import List, Optional
import json
import logging

from lelamp.database.models import Conversation, OperationLog, DeviceState, UserSettings

logger = logging.getLogger("lelamp.database")

class ConversationCRUD:
    """对话记录 CRUD"""

    @staticmethod
    def create(
        db: Session,
        user_message: str,
        agent_response: str,
        duration_ms: int,
        session_id: Optional[str] = None
    ) -> Conversation:
        """创建对话记录"""
        conversation = Conversation(
            user_message=user_message,
            agent_response=agent_response,
            duration_ms=duration_ms,
            session_id=session_id
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        return conversation

    @staticmethod
    def get_recent(db: Session, limit: int = 100) -> List[Conversation]:
        """获取最近的对话记录"""
        return db.query(Conversation)\
            .order_by(Conversation.timestamp.desc())\
            .limit(limit)\
            .all()

class OperationLogCRUD:
    """操作日志 CRUD"""

    @staticmethod
    def create(
        db: Session,
        operation_type: str,
        action: str,
        params: dict,
        success: bool = True,
        error_message: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> OperationLog:
        """创建操作日志"""
        log = OperationLog(
            operation_type=operation_type,
            action=action,
            params=json.dumps(params) if params else None,
            success=success,
            error_message=error_message,
            session_id=session_id
        )
        db.add(log)
        db.commit()
        db.refresh(log)
        return log

class DeviceStateCRUD:
    """设备状态 CRUD"""

    @staticmethod
    def save_state(
        db: Session,
        lamp_id: str,
        status: str,
        motor_positions: dict,
        health_status: dict,
        light_color: dict,
        camera_active: bool
    ) -> DeviceState:
        """保存设备状态快照"""
        state = DeviceState(
            lamp_id=lamp_id,
            status=status,
            motor_positions=json.dumps(motor_positions),
            health_status=json.dumps(health_status),
            light_color=json.dumps(light_color),
            camera_active=camera_active
        )
        db.add(state)
        db.commit()
        db.refresh(state)
        return state

    @staticmethod
    def get_latest(db: Session, lamp_id: str, limit: int = 100) -> List[DeviceState]:
        """获取最新的设备状态"""
        return db.query(DeviceState)\
            .filter(DeviceState.lamp_id == lamp_id)\
            .order_by(DeviceState.timestamp.desc())\
            .limit(limit)\
            .all()
```

#### 3.2.4: 集成到 LeLamp Agent

修改 `lelamp/agent/lelamp_agent.py`，添加数据持久化：

```python
# 在 LeLamp 类中添加
from lelamp.database.session import get_db, init_db
from lelamp.database.crud import ConversationCRUD, OperationLogCRUD, DeviceStateCRUD

class LeLamp(Agent):
    def __init__(self, ...):
        # ... 现有代码

        # 初始化数据库
        init_db()

    async def note_user_text(self, text: str) -> None:
        """记录用户输入（增强版 - 持久化）"""
        # 原有逻辑...

        # 新增：保存到数据库
        with get_db() as db:
            # 这里需要获取实际的响应，暂时记录用户输入
            ConversationCRUD.create(
                db=db,
                user_message=text,
                agent_response="",  # 待更新
                duration_ms=0,
                session_id=self.session_id
            )
```

---

### Task 3.3: 实现设备状态 API

**目标**: 提供设备状态查询和历史记录接口

#### 3.3.1: 扩展设备路由

**文件**: `lelamp/api/routes/devices.py`（扩展）

```python
"""
设备相关 API 路由（扩展版）
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta
import logging

from lelamp.database.session import get_db
from lelamp.database.models import DeviceState, Conversation
from lelamp.database.crud import DeviceStateCRUD, ConversationCRUD

logger = logging.getLogger("lelamp.api.devices")

router = APIRouter()

@router.get("/{lamp_id}/state")
async def get_device_state(
    lamp_id: str,
    db: Session = Depends(get_db)
):
    """获取设备当前状态"""
    # 获取最新状态
    states = DeviceStateCRUD.get_latest(db, lamp_id, limit=1)

    if not states:
        raise HTTPException(status_code=404, detail="设备状态未找到")

    state = states[0]

    # 解析 JSON 数据
    import json
    return {
        "lamp_id": lamp_id,
        "status": state.status,
        "timestamp": state.timestamp,
        "motor_positions": json.loads(state.motor_positions),
        "health_status": json.loads(state.health_status),
        "light_color": json.loads(state.light_color),
        "camera_active": state.camera_active
    }

@router.get("/{lamp_id}/history")
async def get_device_history(
    lamp_id: str,
    hours: int = Query(24, description="时间范围（小时）"),
    db: Session = Depends(get_db)
):
    """获取设备历史状态"""
    start_time = datetime.utcnow() - timedelta(hours=hours)

    states = db.query(DeviceState)\
        .filter(
            DeviceState.lamp_id == lamp_id,
            DeviceState.timestamp >= start_time
        )\
        .order_by(DeviceState.timestamp.desc())\
        .all()

    return {
        "lamp_id": lamp_id,
        "time_range_hours": hours,
        "states": [
            {
                "timestamp": s.timestamp,
                "status": s.status,
                "motor_positions": s.motor_positions,
                "health_status": s.health_status,
            }
            for s in states
        ]
    }

@router.get("/{lamp_id}/conversations")
async def get_conversations(
    lamp_id: str,
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """获取对话历史记录"""
    conversations = ConversationCRUD.get_recent(db, limit=limit)

    return {
        "lamp_id": lamp_id,
        "conversations": [
            {
                "id": c.id,
                "timestamp": c.timestamp,
                "user_message": c.user_message,
                "agent_response": c.agent_response,
                "duration_ms": c.duration_ms
            }
            for c in conversations
        ]
    }
```

---

### Task 3.4: 集成 WebSocket 实时推送

**目标**: 当设备状态变化时，实时推送到前端

#### 3.4.1: 扩展 WebSocket 路由

**文件**: `lelamp/api/routes/websocket.py`

```python
"""
WebSocket 路由 - 实时设备状态推送
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Set
import json
import logging
import asyncio

logger = logging.getLogger("lelamp.api.websocket")

router = APIRouter()

# 连接管理
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, lamp_id: str):
        await websocket.accept()
        if lamp_id not in self.active_connections:
            self.active_connections[lamp_id] = set()
        self.active_connections[lamp_id].add(websocket)
        logger.info(f"WebSocket 连接: {lamp_id}, 总连接数: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket, lamp_id: str):
        self.active_connections[lamp_id].discard(websocket)
        if not self.active_connections[lamp_id]:
            del self.active_connections[lamp_id]
        logger.info(f"WebSocket 断开: {lamp_id}")

    async def broadcast_to_device(self, lamp_id: str, message: dict):
        """向特定设备的所有连接广播消息"""
        if lamp_id in self.active_connections:
            for connection in self.active_connections[lamp_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"WebSocket 发送失败: {e}")
                    self.disconnect(connection, lamp_id)

manager = ConnectionManager()

@router.websocket("/state/{lamp_id}")
async def websocket_device_state(websocket: WebSocket, lamp_id: str):
    """设备状态 WebSocket 端点"""
    await manager.connect(websocket, lamp_id)

    try:
        while True:
            # 保持连接，接收客户端消息（如心跳）
            data = await websocket.receive_text()

            # 处理心跳消息
            if data == "ping":
                await websocket.send_text("pong")

    except WebSocketDisconnect:
        manager.disconnect(websocket, lamp_id)
    except Exception as e:
        logger.error(f"WebSocket 错误: {e}")
        manager.disconnect(websocket, lamp_id)

# 导出管理器实例
__all__ = ["manager"]
```

#### 3.4.2: 更新主应用

**文件**: `lelamp/api/app.py`（更新）

```python
# 导入 WebSocket 管理器
from lelamp.api.routes.websocket import manager

# 更新健康检查
@app.get("/health")
async def health_check():
    """健康检查端点"""
    connection_count = sum(
        len(conns) for conns in manager.active_connections.values()
    )
    return {
        "status": "healthy",
        "service": "lelamp-api",
        "active_connections": connection_count
    }
```

---

### Task 3.5: API 测试和文档

#### 3.5.1: 创建 API 测试

**文件**: `lelamp/test/api/test_devices_api.py`

```python
"""
设备 API 测试
"""
from fastapi.testclient import TestClient
from lelamp.api.app import app
from lelamp.database.session import init_db
import pytest

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_database():
    """初始化测试数据库"""
    init_db()

def test_health_check():
    """测试健康检查"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_get_device_state():
    """测试获取设备状态"""
    response = client.get("/api/devices/test-lamp/state")
    # 由于没有实际数据，可能返回 404
    assert response.status_code in [200, 404]
```

---

## 3. 实施步骤总结

### 第 1 步：基础 API 服务器（1-2 小时）
- [ ] 创建 FastAPI 应用结构
- [ ] 配置 CORS 和中间件
- [ ] 添加健康检查端点
- [ ] 测试服务器启动

### 第 2 步：数据持久化（2-3 小时）
- [ ] 创建数据库模型
- [ ] 实现 CRUD 操作
- [ ] 初始化数据库
- [ ] 集成到 LeLamp Agent

### 第 3 步：设备状态 API（1-2 小时）
- [ ] 实现状态查询接口
- [ ] 实现历史记录接口
- [ ] 实现对话记录接口

### 第 4 步：WebSocket 实时推送（2-3 小时）
- [ ] 创建 WebSocket 路由
- [ ] 实现连接管理
- [ ] 集成状态广播逻辑
- [ ] 测试实时推送

### 第 5 步：测试和文档（1 小时）
- [ ] 编写 API 测试
- [ ] 验证 API 文档
- [ ] 手动测试所有接口
- [ ] 性能基准测试

**总计**: 7-11 小时

---

## 4. 验收标准

### 功能完整性
- [ ] API 服务器正常启动
- [ ] Swagger 文档可访问
- [ ] 数据库正常创建
- [ ] 对话记录正常保存
- [ ] 设备状态正常记录
- [ ] WebSocket 连接正常
- [ ] 实时推送正常工作

### 代码质量
- [ ] 类型检查通过（mypy）
- [ ] 所有端点有文档
- [ ] 错误处理完整
- [ ] 日志记录清晰

### 性能
- [ ] API 响应时间 < 100ms（P95）
- [ ] 支持并发连接
- [ ] 内存使用稳定

---

## 5. 下一步

Phase 3 完成后，可以考虑：

1. **Phase 4.1**: 性能优化
2. **Phase 4.2**: 安全加固
3. **Phase 4.3**: CI/CD 集成
4. **Phase 4.4**: 监控和运维

---

**文档版本**: 1.0
**创建日期**: 2026-03-17
**预计工时**: 7-11 小时
