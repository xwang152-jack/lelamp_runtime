"""
Pydantic models for authentication API requests and responses.

Defines request/response schemas for:
- User registration and login
- Token responses
- Device binding
- User information
"""
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


class RefreshTokenRequest(BaseModel):
    """刷新令牌请求"""
    refresh_token: str
