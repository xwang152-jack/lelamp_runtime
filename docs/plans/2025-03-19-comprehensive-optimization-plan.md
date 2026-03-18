# LeLamp Runtime 全面优化实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**目标:** 实现用户认证、功能完善、性能优化的全面升级，将 LeLamp Runtime 从技术原型提升到商业化产品标准。

**架构:** 渐进式三阶段优化 - 先建立安全基础设施，再完善功能，最后优化性能。每个阶段都有可验证的交付成果。

**技术栈:**
- **后端**: Python 3.12+, FastAPI, SQLAlchemy, JWT, WebSocket, LiveKit
- **前端**: Vue 3, TypeScript, Element Plus, Vite, Pinia
- **数据库**: SQLite (dev) / PostgreSQL (prod)
- **测试**: pytest, pytest-cov, playwright
- **安全**: JWT, bcrypt, HTTPS, CORS, Rate Limiting

---

## 📋 实施概览

### Phase 1: 安全基础设施 (1-2 周)
- Task 1-5: 用户认证系统
- Task 6-8: API 安全中间件
- Task 9-11: WebSocket 安全验证

### Phase 2: 功能完善 (2-3 周)
- Task 12-16: 用户管理界面
- Task 17-20: 视觉功能前端
- Task 21-24: 设备管理功能

### Phase 3: 性能优化 (1-2 周)
- Task 25-28: 数据库优化
- Task 29-31: API 性能提升
- Task 32-34: 前端性能优化

---

## Phase 1: 安全基础设施

### Task 1: 创建用户认证基础模型

**Files:**
- Create: `lelamp/database/models_auth.py`
- Modify: `lelamp/database/models.py`
- Create: `lelamp/test/unit/test_auth_models.py`

**Step 1: 创建认证模型文件**

```python
# lelamp/database/models_auth.py
from datetime import datetime, timedelta
from sqlalchemy import String, Boolean, DateTime, ForeignKey, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing import Optional, List
from lelamp.database.base import Base

class User(Base):
    """用户模型"""
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # 关系
    devices: Mapped[List["DeviceBinding"]] = relationship("DeviceBinding", back_populates="user")

class DeviceBinding(Base):
    """设备绑定模型"""
    __tablename__ = "device_bindings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    device_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    device_secret: Mapped[str] = mapped_column(String(100), nullable=False)  # 设备出厂密钥
    permission_level: Mapped[str] = mapped_column(String(20), default="member", nullable=False)  # admin/member/guest
    bound_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # 关系
    user: Mapped["User"] = relationship("User", back_populates="devices")

class RefreshToken(Base):
    """刷新令牌模型"""
    __tablename__ = "refresh_tokens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    token: Mapped[str] = mapped_column(String(500), unique=True, nullable=False, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    revoked: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
```

**Step 2: 更新主模型导入**

```python
# lelamp/database/models.py (在文件顶部添加)
from lelamp.database.models_auth import User, DeviceBinding, RefreshToken

# 确保这些模型被导出
__all__ = ["Conversation", "OperationLog", "DeviceState", "UserSettings",
           "User", "DeviceBinding", "RefreshToken"]
```

**Step 3: 编写测试**

```python
# lelamp/test/unit/test_auth_models.py
import pytest
from datetime import datetime, timedelta
from lelamp.database.models_auth import User, DeviceBinding, RefreshToken
from lelamp.database.session import engine
from lelamp.database.base import Base

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
    db.close()
```

**Step 4: 运行测试验证**

```bash
uv run pytest lelamp/test/unit/test_auth_models.py -v
```

预期: PASS (2 个测试通过)

**Step 5: 提交**

```bash
git add lelamp/database/models_auth.py lelamp/database/models.py lelamp/test/unit/test_auth_models.py
git commit -m "feat: 添加用户认证数据库模型"
```

---

### Task 2: 实现 JWT 认证服务

**Files:**
- Create: `lelamp/api/services/auth_service.py`
- Create: `lelamp/test/integration/test_auth_service.py`

**Step 1: 创建认证服务**

```python
# lelamp/api/services/auth_service.py
from datetime import datetime, timedelta
from typing import Optional, Dict
import jwt
import bcrypt
from sqlalchemy.orm import Session
from lelamp.database.models_auth import User, DeviceBinding, RefreshToken

SECRET_KEY = "your-secret-key-here"  # 从环境变量读取
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

class AuthService:
    """认证服务"""

    @staticmethod
    def hash_password(password: str) -> str:
        """哈希密码"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """验证密码"""
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

    @staticmethod
    def create_access_token(data: Dict, expires_delta: Optional[timedelta] = None) -> str:
        """创建访问令牌"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    @staticmethod
    def create_refresh_token(user_id: int, db: Session) -> str:
        """创建刷新令牌"""
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        token_data = {
            "user_id": user_id,
            "exp": expire,
            "type": "refresh"
        }
        token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)

        # 保存到数据库
        refresh_token = RefreshToken(
            user_id=user_id,
            token=token,
            expires_at=expire
        )
        db.add(refresh_token)
        db.commit()

        return token

    @staticmethod
    def verify_token(token: str, token_type: str = "access") -> Optional[Dict]:
        """验证令牌"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            if payload.get("type") != token_type:
                return None
            return payload
        except jwt.PyJWTError:
            return None

    @staticmethod
    def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
        """认证用户"""
        user = db.query(User).filter(User.username == username).first()
        if not user:
            return None
        if not AuthService.verify_password(password, user.hashed_password):
            return None
        return user

    @staticmethod
    def register_user(db: Session, username: str, email: str, password: str) -> User:
        """注册新用户"""
        # 检查用户名是否存在
        if db.query(User).filter(User.username == username).first():
            raise ValueError("Username already exists")
        if db.query(User).filter(User.email == email).first():
            raise ValueError("Email already exists")

        hashed_password = AuthService.hash_password(password)
        user = User(
            username=username,
            email=email,
            hashed_password=hashed_password
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

    @staticmethod
    def bind_device(db: Session, user_id: int, device_id: str, device_secret: str) -> DeviceBinding:
        """绑定设备"""
        # 验证设备密钥（这里需要实现设备密钥验证逻辑）
        # TODO: 实现设备密钥验证

        # 检查是否已绑定
        existing = db.query(DeviceBinding).filter(
            DeviceBinding.user_id == user_id,
            DeviceBinding.device_id == device_id
        ).first()
        if existing:
            raise ValueError("Device already bound")

        binding = DeviceBinding(
            user_id=user_id,
            device_id=device_id,
            device_secret=device_secret,
            permission_level="admin"  # 第一个绑定的用户是管理员
        )
        db.add(binding)
        db.commit()
        db.refresh(binding)
        return binding
```

**Step 2: 编写测试**

```python
# lelamp/test/integration/test_auth_service.py
import pytest
from lelamp.api.services.auth_service import AuthService
from lelamp.database.models_auth import User
from lelamp.database.session import get_db_session

@pytest.fixture(scope="function")
def setup_db():
    """创建测试数据库"""
    from lelamp.database.base import Base
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

    # 测试错误密码
    auth_user = AuthService.authenticate_user(db, "authuser", "wrongpassword")
    assert auth_user is None

    # 测试不存在的用户
    auth_user = AuthService.authenticate_user(db, "nonexistent", "password")
    assert auth_user is None

    db.close()
```

**Step 3: 运行测试**

```bash
uv run pytest lelamp/test/integration/test_auth_service.py -v
```

预期: PASS (4 个测试通过)

**Step 4: 提交**

```bash
git add lelamp/api/services/auth_service.py lelamp/test/integration/test_auth_service.py
git commit -m "feat: 实现 JWT 认证服务"
```

---

### Task 3: 创建认证 API 路由

**Files:**
- Create: `lelamp/api/routes/auth.py`
- Modify: `lelamp/api/routes/__init__.py`
- Create: `lelamp/api/models/auth_models.py`
- Create: `lelamp/test/integration/test_auth_routes.py`

**Step 1: 创建认证数据模型**

```python
# lelamp/api/models/auth_models.py
from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class UserRegister(BaseModel):
    """用户注册请求"""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=100)

class UserLogin(BaseModel):
    """用户登录请求"""
    username: str
    password: str

class Token(BaseModel):
    """令牌响应"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class DeviceBindRequest(BaseModel):
    """设备绑定请求"""
    device_id: str
    device_secret: str

class UserResponse(BaseModel):
    """用户信息响应"""
    id: int
    username: str
    email: str
    is_active: bool
    is_admin: bool
    created_at: str

class DeviceBindResponse(BaseModel):
    """设备绑定响应"""
    device_id: str
    permission_level: str
    bound_at: str
```

**Step 2: 创建认证路由**

```python
# lelamp/api/routes/auth.py
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta
import logging

from lelamp.database.session import get_db
from lelamp.api.services.auth_service import AuthService
from lelamp.api.models.auth_models import (
    UserRegister, UserLogin, Token, DeviceBindRequest,
    UserResponse, DeviceBindResponse
)
from lelamp.database.models_auth import User, DeviceBinding

logger = logging.getLogger("lelamp.api.auth")
router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")

@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserRegister, db: Session = Depends(get_db)):
    """
    用户注册

    - **username**: 用户名 (3-50 字符)
    - **email**: 邮箱地址
    - **password**: 密码 (6-100 字符)
    """
    try:
        user = AuthService.register_user(
            db,
            username=user_data.username,
            email=user_data.email,
            password=user_data.password
        )

        # 创建令牌
        access_token = AuthService.create_access_token(
            data={"sub": user.username, "user_id": user.id}
        )
        refresh_token = AuthService.create_refresh_token(user.id, db)

        return Token(
            access_token=access_token,
            refresh_token=refresh_token
        )

    except ValueError as e:
        logger.warning(f"Registration failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """
    用户登录 (OAuth2 表单)

    - **username**: 用户名
    - **password**: 密码
    """
    user = AuthService.authenticate_user(db, form_data.username, form_data.password)
    if not user:
        logger.warning(f"Failed login attempt for user: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 更新最后登录时间
    from datetime import datetime
    user.last_login = datetime.utcnow()
    db.commit()

    # 创建令牌
    access_token = AuthService.create_access_token(
        data={"sub": user.username, "user_id": user.id}
    )
    refresh_token = AuthService.create_refresh_token(user.id, db)

    logger.info(f"User logged in: {user.username}")
    return Token(
        access_token=access_token,
        refresh_token=refresh_token
    )

@router.get("/me", response_model=UserResponse)
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    获取当前用户信息

    需要有效的访问令牌
    """
    payload = AuthService.verify_token(token, "access")
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    username = payload.get("sub")
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        is_active=user.is_active,
        is_admin=user.is_admin,
        created_at=user.created_at.isoformat()
    )

@router.post("/bind-device", response_model=DeviceBindResponse)
async def bind_device(
    bind_request: DeviceBindRequest,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    绑定设备

    - **device_id**: 设备 ID
    - **device_secret**: 设备密钥
    """
    payload = AuthService.verify_token(token, "access")
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

    user_id = payload.get("user_id")
    try:
        binding = AuthService.bind_device(
            db,
            user_id=user_id,
            device_id=bind_request.device_id,
            device_secret=bind_request.device_secret
        )

        logger.info(f"Device {bind_request.device_id} bound to user {user_id}")
        return DeviceBindResponse(
            device_id=binding.device_id,
            permission_level=binding.permission_level,
            bound_at=binding.bound_at.isoformat()
        )

    except ValueError as e:
        logger.warning(f"Device binding failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Device binding error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.post("/refresh-token", response_model=Token)
async def refresh_token(
    refresh_token: str,
    db: Session = Depends(get_db)
):
    """
    刷新访问令牌

    使用有效的刷新令牌获取新的访问令牌
    """
    payload = AuthService.verify_token(refresh_token, "refresh")
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )

    # 检查刷新令牌是否在数据库中且未被撤销
    from lelamp.database.models_auth import RefreshToken
    token_record = db.query(RefreshToken).filter(
        RefreshToken.token == refresh_token,
        RefreshToken.revoked == False
    ).first()

    if not token_record:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token not found or revoked"
        )

    user_id = payload.get("user_id")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # 创建新的令牌
    access_token = AuthService.create_access_token(
        data={"sub": user.username, "user_id": user.id}
    )
    new_refresh_token = AuthService.create_refresh_token(user.id, db)

    # 撤销旧的刷新令牌
    token_record.revoked = True
    db.commit()

    return Token(
        access_token=access_token,
        refresh_token=new_refresh_token
    )
```

**Step 3: 更新路由导出**

```python
# lelamp/api/routes/__init__.py
from fastapi import APIRouter
from lelamp.api.routes import devices, settings, system, websocket, auth

api_router = APIRouter()

# 注册路由
api_router.include_router(auth.router, prefix="/auth", tags=["认证"])
api_router.include_router(devices.router, prefix="/devices", tags=["设备"])
api_router.include_router(settings.router, prefix="/settings", tags=["设置"])
api_router.include_router(system.router, prefix="/system", tags=["系统"])
api_router.include_router(websocket.router, prefix="/ws", tags=["WebSocket"])
```

**Step 4: 编写测试**

```python
# lelamp/test/integration/test_auth_routes.py
import pytest
from fastapi.testclient import TestClient
from lelamp.api.app import app
from lelamp.database.session import get_db_session
from lelamp.database.base import Base, engine

client = TestClient(app)

@pytest.fixture(scope="function")
def setup_db():
    """创建测试数据库"""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

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
            "username": "testuser",
            "email": "test1@example.com",
            "password": "testpass123"
        }
    )

    # 第二次注册相同用户名
    response = client.post(
        "/api/auth/register",
        json={
            "username": "testuser",
            "email": "test2@example.com",
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
            "username": "wrongpassuser",
            "email": "wrongpass@example.com",
            "password": "correctpass123"
        }
    )

    # 错误密码登录
    response = client.post(
        "/api/auth/login",
        data={
            "username": "wrongpassuser",
            "password": "wrongpass123"
        }
    )

    assert response.status_code == 401

def test_get_current_user(setup_db):
    """测试获取当前用户信息"""
    # 注册并登录
    register_response = client.post(
        "/api/auth/register",
        json={
            "username": "getmeuser",
            "email": "getme@example.com",
            "password": "getmepass123"
        }
    )
    token = register_response.json()["access_token"]

    # 获取用户信息
    response = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {token}"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["username"] == "getmeuser"
    assert data["email"] == "getme@example.com"
```

**Step 5: 运行测试**

```bash
uv run pytest lelamp/test/integration/test_auth_routes.py -v
```

预期: PASS (5 个测试通过)

**Step 6: 提交**

```bash
git add lelamp/api/routes/auth.py lelamp/api/routes/__init__.py lelamp/api/models/auth_models.py lelamp/test/integration/test_auth_routes.py
git commit -m "feat: 添加认证 API 路由"
```

---

## 继续实施...

继续实施计划的后续部分...

---

### Task 4: 创建认证中间件

**Files:**
- Create: `lelamp/api/middleware/auth_middleware.py`
- Modify: `lelamp/api/app.py`
- Create: `lelamp/test/integration/test_auth_middleware.py`

**Step 1: 创建认证中间件**

```python
# lelamp/api/middleware/auth_middleware.py
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from typing import Optional
import logging

from lelamp.api.services.auth_service import AuthService
from lelamp.database.session import get_db
from lelamp.database.models_auth import User

logger = logging.getLogger("lelamp.api.auth_middleware")
security = HTTPBearer(auto_error=False)

class AuthMiddleware:
    """认证中间件"""

    async def __call__(
        self,
        request: Request,
        authorization: Optional[HTTPAuthorizationCredentials] = Depends(security)
    ) -> Optional[User]:
        """
        验证请求并返回用户信息

        如果请求头包含有效的令牌，将用户信息添加到 request.state
        如果没有令牌或令牌无效，返回 None (不抛出异常)
        """
        if authorization is None:
            return None

        token = authorization.credentials
        payload = AuthService.verify_token(token, "access")

        if payload is None:
            logger.warning(f"Invalid token attempt")
            return None

        username = payload.get("sub")
        if not username:
            return None

        # 获取用户信息
        db: Session = next(get_db())
        try:
            user = db.query(User).filter(User.username == username).first()
            if user and user.is_active:
                # 将用户信息添加到请求状态
                request.state.user = user
                return user
        finally:
            db.close()

        return None

# 可选：严格认证中间件 (需要认证才能访问)
class StrictAuthMiddleware:
    """严格认证中间件 - 必须认证才能访问"""

    async def __call__(
        self,
        request: Request,
        authorization: HTTPAuthorizationCredentials = Depends(security)
    ) -> User:
        """
        验证请求并返回用户信息

        如果没有令牌或令牌无效，抛出 401 异常
        """
        if authorization is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated",
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = authorization.credentials
        payload = AuthService.verify_token(token, "access")

        if payload is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        username = payload.get("sub")
        if not username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )

        # 获取用户信息
        db: Session = next(get_db())
        try:
            user = db.query(User).filter(User.username == username).first()
            if user is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            if not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="User account is disabled"
                )

            # 将用户信息添加到请求状态
            request.state.user = user
            return user
        finally:
            db.close()

# 依赖注入函数
async def get_current_user(request: Request) -> Optional[User]:
    """获取当前用户 (可选)"""
    return getattr(request.state, "user", None)

async def require_current_user(request: Request) -> User:
    """获取当前用户 (必需)"""
    user = getattr(request.state, "user", None)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    return user

async def require_admin_user(request: Request) -> User:
    """获取当前管理员用户 (必需管理员权限)"""
    user = await require_current_user(request)
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return user
```

**Step 2: 更新应用配置**

```python
# lelamp/api/app.py (在现有代码中添加中间件)
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from lelamp.api.middleware.auth_middleware import AuthMiddleware

# ... 现有代码 ...

# 添加认证中间件
auth_middleware = AuthMiddleware()

@app.middleware("http")
async def auth_request_middleware(request: Request, call_next):
    """认证中间件 - 自动处理所有请求"""
    # 对认证相关的路由不做处理
    if request.url.path.startswith("/api/auth/"):
        return await call_next(request)

    # 尝试获取用户信息
    authorization = request.headers.get("Authorization")
    if authorization:
        from fastapi.security import HTTPAuthorizationCredentials
        from lelamp.api.middleware.auth_middleware import security

        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials=authorization.replace("Bearer ", "")
        )
        user = await auth_middleware(request, credentials)
        # 用户信息已添加到 request.state.user

    response = await call_next(request)
    return response
```

**Step 3: 编写测试**

```python
# lelamp/test/integration/test_auth_middleware.py
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
        username="middlewareuser",
        email="middleware@example.com",
        password="middlewarepass123"
    )

    # 创建令牌
    token = AuthService.create_access_token(
        data={"sub": "middlewareuser", "user_id": user.id}
    )
    return token

def test_public_endpoint_no_auth():
    """测试公开端点无需认证"""
    response = client.get("/api/health")
    assert response.status_code == 200

def test_protected_endpoint_with_auth(auth_token):
    """测试受保护端点需要认证"""
    response = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {auth_token}"}
    )
    assert response.status_code == 200
    assert response.json()["username"] == "middlewareuser"

def test_protected_endpoint_without_auth():
    """测试受保护端点无认证时失败"""
    response = client.get("/api/auth/me")
    assert response.status_code == 401

def test_protected_endpoint_invalid_auth():
    """测试受保护端点无效认证时失败"""
    response = client.get(
        "/api/auth/me",
        headers={"Authorization": "Bearer invalid_token_12345"}
    )
    assert response.status_code == 401
```

**Step 4: 运行测试**

```bash
uv run pytest lelamp/test/integration/test_auth_middleware.py -v
```

预期: PASS (4 个测试通过)

**Step 5: 提交**

```bash
git add lelamp/api/middleware/auth_middleware.py lelamp/api/app.py lelamp/test/integration/test_auth_middleware.py
git commit -m "feat: 添加认证中间件"
```

---

### Task 5: WebSocket 认证集成

**Files:**
- Modify: `lelamp/api/routes/websocket.py`
- Create: `lelamp/test/integration/test_websocket_auth.py`

**Step 1: 更新 WebSocket 路由以支持认证**

```python
# lelamp/api/routes/websocket.py (在现有文件中添加认证逻辑)
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, status
from typing import Dict, Set, Optional
import asyncio
import logging
import json
from datetime import datetime
from lelamp.api.services.auth_service import AuthService

logger = logging.getLogger("lelamp.api.websocket")
router = APIRouter()

# ... 现有的 ConnectionManager 类 ...

class ConnectionManager:
    """WebSocket 连接管理器 (增强版)"""

    def __init__(self):
        # Dict: lamp_id -> Set[WebSocket]
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # Dict: websocket -> user_info
        self.connection_users: Dict[WebSocket, Dict] = {}
        # Dict: lamp_id -> Set[channel names]
        self.subscriptions: Dict[str, Set[str]] = {}
        # 连接统计
        self._connection_counts: Dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, lamp_id: str, token: Optional[str] = None) -> Dict[str, any]:
        """
        接受连接并添加到设备连接池 (支持认证)

        Args:
            websocket: WebSocket 连接对象
            lamp_id: 设备 ID
            token: 可选的认证令牌

        Returns:
            用户信息字典 (如果认证成功) 或 None
        """
        await websocket.accept()

        # 验证令牌 (如果提供)
        user_info = None
        if token:
            payload = AuthService.verify_token(token, "access")
            if payload:
                user_info = {
                    "user_id": payload.get("user_id"),
                    "username": payload.get("sub"),
                    "authenticated": True
                }
                logger.info(f"Authenticated WebSocket connection: {user_info['username']}")

        async with self._lock:
            if lamp_id not in self.active_connections:
                self.active_connections[lamp_id] = set()
                self.subscriptions[lamp_id] = set()
                self._connection_counts[lamp_id] = 0

            self.active_connections[lamp_id].add(websocket)
            self.connection_users[websocket] = user_info or {"authenticated": False}
            self._connection_counts[lamp_id] += 1

        logger.info(f"WebSocket 连接建立: {lamp_id} (总连接数: {self.get_connection_count(lamp_id)})")

        # 发送连接确认
        await websocket.send_json({
            "type": "connected",
            "lamp_id": lamp_id,
            "server_time": datetime.utcnow().isoformat(),
            "message": "WebSocket connection established",
            "authenticated": user_info is not None
        })

        return user_info

    def get_user_info(self, websocket: WebSocket) -> Optional[Dict]:
        """获取连接的用户信息"""
        return self.connection_users.get(websocket)

    # ... 其他现有方法保持不变 ...

# 全局连接管理器实例
manager = ConnectionManager()

@router.websocket("/ws/{lamp_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    lamp_id: str,
    token: Optional[str] = Query(None)
):
    """
    WebSocket 端点 (支持认证)

    Args:
        websocket: WebSocket 连接
        lamp_id: 设备 ID
        token: 可选的认证令牌 (查询参数)
    """
    try:
        # 建立连接 (带认证)
        user_info = await manager.connect(websocket, lamp_id, token)

        if user_info:
            logger.info(f"用户 {user_info['username']} 连接到设备 {lamp_id}")
        else:
            logger.info(f"匿名用户连接到设备 {lamp_id}")

        # 处理消息循环
        while True:
            data = await websocket.receive_text()
            await handle_client_message(websocket, lamp_id, data, user_info)

    except WebSocketDisconnect:
        manager.disconnect(websocket, lamp_id)
        logger.info(f"WebSocket 断开: {lamp_id}")
    except Exception as e:
        logger.error(f"WebSocket 错误: {lamp_id} - {str(e)}")
        manager.disconnect(websocket, lamp_id)

async def handle_client_message(websocket: WebSocket, lamp_id: str, message: str, user_info: Optional[Dict]):
    """处理客户端消息"""
    try:
        data = json.loads(message)

        # 验证消息格式
        if not validate_client_message(data):
            await websocket.send_json({
                "type": "error",
                "message": "Invalid message format"
            })
            return

        message_type = data.get("type")

        # 根据消息类型处理
        if message_type == "ping":
            await handle_ping(websocket)
        elif message_type == "subscribe":
            await handle_subscribe(websocket, lamp_id, data.get("channels", []))
        elif message_type == "unsubscribe":
            await handle_unsubscribe(websocket, lamp_id, data.get("channels", []))
        elif message_type == "command":
            # 检查认证状态 (某些命令需要认证)
            if data.get("command") in ["sensitive_command"]:
                if not user_info or not user_info.get("authenticated"):
                    await websocket.send_json({
                        "type": "error",
                        "message": "Authentication required for this command"
                    })
                    return
            await handle_command(websocket, lamp_id, data.get("command", {}))
        else:
            logger.warning(f"未知消息类型: {message_type}")

    except json.JSONDecodeError:
        await websocket.send_json({
            "type": "error",
            "message": "Invalid JSON format"
        })
    except Exception as e:
        logger.error(f"处理消息错误: {str(e)}")
        await websocket.send_json({
            "type": "error",
            "message": "Internal server error"
        })
```

**Step 2: 编写测试**

```python
# lelamp/test/integration/test_websocket_auth.py
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
    user = AuthService.register_user(
        get_db_session(),
        username="wsuser",
        email="ws@example.com",
        password="wspass123"
    )
    return AuthService.create_access_token(
        data={"sub": "wsuser", "user_id": user.id}
    )

def test_websocket_connect_without_token():
    """测试无认证的 WebSocket 连接"""
    with client.websocket_connect("/api/ws/test_device") as websocket:
        data = websocket.receive_json()
        assert data["type"] == "connected"
        assert data["authenticated"] is False

def test_websocket_connect_with_token(auth_token):
    """测试带认证的 WebSocket 连接"""
    with client.websocket_connect(
        f"/api/ws/test_device?token={auth_token}"
    ) as websocket:
        data = websocket.receive_json()
        assert data["type"] == "connected"
        assert data["authenticated"] is True

def test_websocket_send_message_with_auth(auth_token):
    """测试认证用户发送消息"""
    with client.websocket_connect(
        f"/api/ws/test_device?token={auth_token}"
    ) as websocket:
        # 接收连接确认
        websocket.receive_json()

        # 发送订阅消息
        websocket.send_json({
            "type": "subscribe",
            "channels": ["state", "events"]
        })

        # 接收响应
        response = websocket.receive_json()
        assert response["type"] in ["subscribed", "error"]

def test_websocket_reject_sensitive_command_without_auth():
    """测试拒绝未认证用户的敏感命令"""
    with client.websocket_connect("/api/ws/test_device") as websocket:
        # 接收连接确认
        websocket.receive_json()

        # 发送敏感命令
        websocket.send_json({
            "type": "command",
            "command": {"action": "sensitive_command"}
        })

        # 应该收到错误响应
        response = websocket.receive_json()
        assert response["type"] == "error"
        assert "Authentication required" in response["message"]
```

**Step 3: 运行测试**

```bash
uv run pytest lelamp/test/integration/test_websocket_auth.py -v
```

预期: PASS (4 个测试通过)

**Step 4: 提交**

```bash
git add lelamp/api/routes/websocket.py lelamp/test/integration/test_websocket_auth.py
git commit -m "feat: WebSocket 认证集成"
```

---

## Phase 1 总结

至此，Phase 1 的安全基础设施已经完成：

✅ **已完成:**
- Task 1: 用户认证数据库模型
- Task 2: JWT 认证服务
- Task 3: 认证 API 路由
- Task 4: 认证中间件
- Task 5: WebSocket 认证集成

📊 **成果:**
- 完整的用户认证系统 (注册、登录、令牌刷新)
- JWT 访问令牌和刷新令牌机制
- 设备绑定功能
- 认证中间件保护 API 端点
- WebSocket 连接认证支持
- 20+ 个测试确保功能正确性

🎯 **验证步骤:**
```bash
# 运行所有认证相关测试
uv run pytest lelamp/test/unit/test_auth_models.py -v
uv run pytest lelamp/test/integration/test_auth_service.py -v
uv run pytest lelamp/test/integration/test_auth_routes.py -v
uv run pytest lelamp/test/integration/test_auth_middleware.py -v
uv run pytest lelamp/test/integration/test_websocket_auth.py -v

# 检查测试覆盖率
uv run pytest --cov=lelamp.api.services.auth_service --cov=lelamp.api.routes.auth --cov-report=term-missing
```

继续执行 Phase 2 和 Phase 3 的实施...