"""
LiveKit Token 生成 API

提供经过认证的 LiveKit Token 生成端点，供前端直接获取 Token。
"""
import os
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from livekit import api

from lelamp.api.services.auth_service import AuthService

logger = logging.getLogger("lelamp.api.livekit")
router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")


class LiveKitTokenRequest(BaseModel):
    room: str = "lelamp-room"
    identity: str = "user-app"


class LiveKitTokenResponse(BaseModel):
    token: str
    room: str
    identity: str


@router.post("/token", response_model=LiveKitTokenResponse)
async def create_livekit_token(
    request: LiveKitTokenRequest,
    token: str = Depends(oauth2_scheme),
):
    """生成 LiveKit 访问令牌（需要认证）"""
    payload = AuthService.verify_token(token, "access")
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )

    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")
    if not api_key or not api_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LiveKit is not configured on this device",
        )

    # 使用用户身份作为 participant identity
    identity = request.identity or f"user-{payload.get('user_id', 'unknown')}"

    access_token = (
        api.AccessToken(api_key, api_secret)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=request.room,
                can_publish=True,
                can_subscribe=True,
            )
        )
    )

    jwt_token = access_token.to_jwt()
    logger.info(f"LiveKit token generated for {identity} in room {request.room}")

    return LiveKitTokenResponse(
        token=jwt_token,
        room=request.room,
        identity=identity,
    )
