"""
Authentication service for user management and JWT token handling.

Provides:
- Password hashing and verification
- JWT access token and refresh token generation
- User authentication and registration
- Device binding management
"""
from datetime import datetime, timedelta
from typing import Optional, Dict
import os
import hmac
import secrets
import logging
import jwt
import bcrypt
import uuid
from sqlalchemy.orm import Session
from lelamp.database.models_auth import User, DeviceBinding, RefreshToken

logger = logging.getLogger("lelamp.api.auth")

# JWT 配置
_jwt_secret = os.getenv("LELAMP_JWT_SECRET")
if _jwt_secret:
    SECRET_KEY = _jwt_secret
else:
    SECRET_KEY = secrets.token_hex(32)
    logger.warning(
        "LELAMP_JWT_SECRET 未设置，使用随机密钥。重启后所有 Token 将失效。"
        "请在 .env 中设置 LELAMP_JWT_SECRET 以持久化密钥。"
    )
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
        jti = str(uuid.uuid4())  # JWT ID: 唯一标识符
        token_data = {
            "user_id": user_id,
            "exp": expire,
            "type": "refresh",
            "jti": jti
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
        # 验证设备密钥（恒定时间比较防止时序攻击）
        # 优先从 setup_status.json 读取，其次从环境变量
        expected_secret = os.getenv("LELAMP_DEVICE_SECRET")
        if not expected_secret:
            try:
                import json
                from pathlib import Path
                status_file = Path("/var/lib/lelamp/setup_status.json")
                if status_file.exists():
                    data = json.loads(status_file.read_text())
                    expected_secret = data.get("device_secret")
            except Exception:
                pass
        if expected_secret and not hmac.compare_digest(device_secret, expected_secret):
            raise PermissionError("Invalid device secret")

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
            device_secret="",  # 不存储密钥明文
            permission_level="admin"  # 第一个绑定的用户是管理员
        )
        db.add(binding)
        db.commit()
        db.refresh(binding)
        return binding