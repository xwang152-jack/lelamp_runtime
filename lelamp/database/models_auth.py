"""
SQLAlchemy ORM models for user authentication and device management.

Defines database models for:
- User: User accounts with authentication
- DeviceBinding: User-device relationships with permissions
- RefreshToken: JWT refresh token storage
"""
from datetime import datetime, timedelta, UTC
from sqlalchemy import String, Boolean, DateTime, ForeignKey, Integer
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
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC), nullable=False)
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # 关系
    devices: Mapped[List["DeviceBinding"]] = relationship("DeviceBinding", back_populates="user")
    refresh_tokens: Mapped[List["RefreshToken"]] = relationship("RefreshToken", back_populates="user")

    def __repr__(self) -> str:
        return f"<User(id={self.id}, username={self.username}, email={self.email})>"

class DeviceBinding(Base):
    """设备绑定模型"""
    __tablename__ = "device_bindings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    device_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    device_secret: Mapped[str] = mapped_column(String(100), nullable=False)  # 设备出厂密钥
    permission_level: Mapped[str] = mapped_column(String(20), default="member", nullable=False)  # admin/member/guest
    bound_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # 关系
    user: Mapped["User"] = relationship("User", back_populates="devices")

    # 唯一约束：一个用户只能绑定一次同一设备
    __table_args__ = (
        {"sqlite_autoincrement": True},  # SQLite 兼容性
    )

    def __repr__(self) -> str:
        return f"<DeviceBinding(id={self.id}, user_id={self.user_id}, device_id={self.device_id})>"

class RefreshToken(Base):
    """刷新令牌模型"""
    __tablename__ = "refresh_tokens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    token: Mapped[str] = mapped_column(String(500), unique=True, nullable=False, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC), nullable=False)
    revoked: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # 关系
    user: Mapped["User"] = relationship("User", back_populates="refresh_tokens")

    def __repr__(self) -> str:
        return f"<RefreshToken(id={self.id}, user_id={self.user_id}, revoked={self.revoked})>"
