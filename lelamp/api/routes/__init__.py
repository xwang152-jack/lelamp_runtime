from fastapi import APIRouter
from lelamp.api.routes import devices, history, websocket

api_router = APIRouter(prefix="/api")
api_router.include_router(devices.router, prefix="/devices", tags=["devices"])
api_router.include_router(history.router, prefix="/history", tags=["history"])

__all__ = ["api_router"]
