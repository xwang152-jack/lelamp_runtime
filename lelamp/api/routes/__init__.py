from fastapi import APIRouter
from lelamp.api.routes import devices, history, websocket, system, settings

api_router = APIRouter(prefix="/api")

# 设备相关端点
api_router.include_router(devices.router, prefix="/devices", tags=["devices"])
api_router.include_router(history.router, prefix="/history", tags=["history"])
api_router.include_router(websocket.router, prefix="/ws", tags=["websocket"])

# 系统管理端点
api_router.include_router(system.router, prefix="/system", tags=["system"])

# 设置管理端点
api_router.include_router(settings.router, prefix="/settings", tags=["settings"])

__all__ = ["api_router"]
