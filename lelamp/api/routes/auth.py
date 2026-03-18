"""
Authentication API routes.

Provides endpoints for:
- User registration and login
- Token refresh
- User profile management
- Device binding
"""
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta
import logging

from lelamp.database.session import get_db
from lelamp.api.services.auth_service import AuthService
from lelamp.api.models.auth_models import (
    UserRegister, UserLogin, Token, DeviceBindRequest,
    UserResponse, DeviceBindResponse, RefreshTokenRequest
)
from lelamp.database.models_auth import User

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

        logger.info(f"用户注册成功: {user.username}")
        return Token(
            access_token=access_token,
            refresh_token=refresh_token
        )

    except ValueError as e:
        logger.warning(f"注册失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"注册错误: {str(e)}")
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
        logger.warning(f"登录失败: 用户名或密码错误 - {form_data.username}")
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

    logger.info(f"用户登录成功: {user.username}")
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

        logger.info(f"设备 {bind_request.device_id} 绑定到用户 {user_id}")
        return DeviceBindResponse(
            device_id=binding.device_id,
            permission_level=binding.permission_level,
            bound_at=binding.bound_at.isoformat()
        )

    except ValueError as e:
        logger.warning(f"设备绑定失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"设备绑定错误: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post("/refresh-token", response_model=Token)
async def refresh_token(
    request: RefreshTokenRequest,
    db: Session = Depends(get_db)
):
    """
    刷新访问令牌

    使用有效的刷新令牌获取新的访问令牌
    """
    payload = AuthService.verify_token(request.refresh_token, "refresh")
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )

    # 检查刷新令牌是否在数据库中且未被撤销
    from lelamp.database.models_auth import RefreshToken
    token_record = db.query(RefreshToken).filter(
        RefreshToken.token == request.refresh_token,
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

    logger.info(f"令牌刷新成功: 用户 {user.username}")
    return Token(
        access_token=access_token,
        refresh_token=new_refresh_token
    )
