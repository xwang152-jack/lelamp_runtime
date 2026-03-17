"""
FastAPI 应用主文件
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError
import logging

from lelamp.api.routes import api_router

logger = logging.getLogger("lelamp.api")

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

# 生命周期事件
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时执行
    logger.info("LeLamp API 启动")
    yield
    # 关闭时执行
    logger.info("LeLamp API 关闭")

app.router.lifespan_context = lifespan

# 包含 API 路由
app.include_router(api_router)

# 健康检查
@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "service": "lelamp-api"}

# WebSocket 连接管理
active_connections: list[WebSocket] = []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 端点 - 实时推送设备状态"""
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # 处理客户端消息
            await websocket.send_text(f"Echo: {data}")
    except WebSocketDisconnect:
        active_connections.remove(websocket)

async def broadcast_to_websockets(message: dict):
    """向所有 WebSocket 连接广播消息"""
    for connection in active_connections:
        try:
            await connection.send_json(message)
        except:
            active_connections.remove(connection)


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
__all__ = ["app", "broadcast_to_websockets"]
