"""
FastAPI 应用主文件
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
import logging
import asyncio
from typing import Dict

from lelamp.api.routes import api_router
from lelamp.database.session import SessionLocal
from lelamp.database import crud
from lelamp.api.routes.websocket import (
    manager,
    push_state_update,
    push_notification,
)

logger = logging.getLogger("lelamp.api")

# 存储上一次的状态，用于检测变化
_last_state_cache: Dict[str, dict] = {}


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


# =============================================================================
# 后台任务
# =============================================================================


async def state_polling_task(interval_seconds: int = 5):
    """
    后台任务：定期轮询数据库状态变化并推送到 WebSocket 客户端

    Args:
        interval_seconds: 轮询间隔（秒）
    """
    global _last_state_cache

    logger.info(f"启动状态轮询任务，间隔: {interval_seconds} 秒")

    while True:
        try:
            db: Session = SessionLocal()
            try:
                # 获取所有有活跃连接的设备
                active_devices = manager.get_all_connection_counts()

                for lamp_id in active_devices.keys():
                    # 查询最新状态
                    state = crud.get_latest_device_state(db, lamp_id)

                    if state:
                        # 构建状态数据
                        current_state = {
                            "lamp_id": state.lamp_id,
                            "conversation_state": state.conversation_state,
                            "motor_positions": state.motor_positions,
                            "light_color": state.light_color,
                            "health_status": state.health_status,
                            "uptime_seconds": state.uptime_seconds,
                            "timestamp": state.timestamp.isoformat(),
                        }

                        # 检测状态变化
                        last_state = _last_state_cache.get(lamp_id)

                        if last_state != current_state:
                            # 状态有变化，推送给订阅者
                            await push_state_update(lamp_id, current_state)
                            _last_state_cache[lamp_id] = current_state
                            logger.debug(f"状态更新已推送: {lamp_id}")

            finally:
                db.close()

            # 等待下一次轮询
            await asyncio.sleep(interval_seconds)

        except Exception as e:
            logger.error(f"状态轮询任务错误: {e}", exc_info=True)
            await asyncio.sleep(interval_seconds)


# =============================================================================
# 生命周期事件
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("LeLamp API 启动")

    # 启动后台状态轮询任务
    polling_task = asyncio.create_task(state_polling_task())

    yield

    # 关闭时执行
    logger.info("LeLamp API 关闭")

    # 取消后台任务
    polling_task.cancel()
    try:
        await polling_task
    except asyncio.CancelledError:
        pass


app.router.lifespan_context = lifespan

# 包含 API 路由
app.include_router(api_router)

# 健康检查
@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "service": "lelamp-api",
        "active_connections": manager.get_all_connection_counts()
    }


# =============================================================================
# 异常处理器
# =============================================================================


@app.exception_handler(IntegrityError)
async def handle_integrity_error(request: Request, exc: IntegrityError):
    """
    处理数据库完整性错误（重复键、约束违反等）
    """
    logger.error(f"Database integrity error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "detail": "Database constraint violation",
            "error": str(exc.orig) if hasattr(exc, 'orig') else str(exc),
        },
    )


@app.exception_handler(ValueError)
async def handle_value_error(request: Request, exc: ValueError):
    """
    处理值错误（无效数据等）
    """
    logger.warning(f"Value error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)},
    )


@app.exception_handler(Exception)
async def handle_generic_exception(request: Request, exc: Exception):
    """
    处理未捕获的通用异常
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error", "error": str(exc)},
    )


# 导出关键组件
__all__ = ["app"]
