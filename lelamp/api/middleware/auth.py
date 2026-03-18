"""
Authentication middleware for FastAPI.

Provides:
- JWT token verification dependency
- Optional authentication (for public endpoints)
- Admin role verification
"""
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from typing import Optional
import logging

from lelamp.database.session import get_db
from lelamp.api.services.auth_service import AuthService
from lelamp.database.models_auth import User

logger = logging.getLogger("lelamp.api.middleware")
security = HTTPBearer(auto_error=False)


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """
    可选认证依赖 - 如果提供 token 则验证,否则返回 None

    用于需要支持匿名访问的端点
    """
    if credentials is None:
        return None

    try:
        token = credentials.credentials
        payload = AuthService.verify_token(token, "access")
        if payload is None:
            return None

        username = payload.get("sub")
        user = db.query(User).filter(User.username == username).first()
        return user

    except Exception as e:
        logger.warning(f"Optional auth failed: {str(e)}")
        return None


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """
    必须认证依赖 - 要求有效的 JWT token

    用于需要登录的端点
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
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

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled"
        )

    return user


async def get_current_admin(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    管理员权限依赖 - 要求用户具有管理员角色

    用于管理端点
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user
