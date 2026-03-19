"""
FastAPI 应用主文件
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from starlette.middleware.base import BaseHTTPMiddleware
import logging
import asyncio
from typing import Dict
import os

from lelamp.api.routes import api_router
from lelamp.database.session import SessionLocal
from lelamp.database import crud
from lelamp.api.routes.websocket import (
    manager,
    push_state_update,
)
from lelamp.config import load_motor_config

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
# 开发环境：允许所有本地网络访问
# 生产环境：应该设置具体的域名
import os

# 从环境变量读取允许的源，如果没有则使用默认列表
allowed_origins = os.getenv("LELAMP_CORS_ORIGINS", "").split(",") if os.getenv("LELAMP_CORS_ORIGINS") else [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
    "http://127.0.0.1:3000",
    "http://192.168.0.104:5173",
    "http://192.168.0.104:3000",
    "http://192.168.0.105:5173",
    "http://192.168.0.105:3000",
]

# 开发模式下添加当前访问的 IP
if os.getenv("LELAMP_DEV_MODE", "0") == "1":
    # 开发模式：允许所有 localhost 和内网 IP
    allowed_origins.extend([
        "http://10.251.145.7:5173",
        "http://10.251.145.7:5174",
    ])
    # 可以通过环境变量添加更多源

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"],
    max_age=600,  # 预检请求缓存时间（秒）
)


# =============================================================================
# 安全头中间件
# =============================================================================


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    添加安全响应头的中间件

    添加常见的安全头以防止 XSS、点击劫持等攻击
    """

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # 添加安全头
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        # 内容安全策略 (可根据实际需要调整)
        response.headers["Content-Security-Policy"] = "default-src 'self'"

        # 推荐安全实践
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

        return response


# 添加安全头中间件
app.add_middleware(SecurityHeadersMiddleware)

# 添加 GZip 压缩中间件
app.add_middleware(GZipMiddleware, minimum_size=1000)


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

    # 初始化硬件服务
    try:
        from lelamp.service.motors.motors_service import MotorsService
        motor_config = load_motor_config()
        motors_service = MotorsService(port="/dev/ttyACM0", lamp_id="lelamp", motor_config=motor_config)
        motors_service.start()
        app.state.motors_service = motors_service
        logger.info("MotorsService started")
    except Exception as e:
        logger.error(f"MotorsService start failed: {e}, fallback to NoOpMotorsService")
        from lelamp.service.motors.noop_motors_service import NoOpMotorsService
        motors_service = NoOpMotorsService()
        motors_service.start()
        app.state.motors_service = motors_service

    try:
        from lelamp.service.rgb.rgb_service import RGBService
        rgb_service = RGBService()
        rgb_service.start()
        app.state.rgb_service = rgb_service
        logger.info("RGBService started")
    except Exception as e:
        logger.error(f"RGBService start failed: {e}, fallback to NoOpRGBService")
        from lelamp.service.rgb.noop_rgb_service import NoOpRGBService
        rgb_service = NoOpRGBService()
        rgb_service.start()
        app.state.rgb_service = rgb_service

    # 初始化 LeLamp Agent
    try:
        from lelamp.agent.lelamp_agent import LeLamp
        agent = LeLamp(
            port="/dev/ttyACM0",
            lamp_id="lelamp",
            motors_service=app.state.motors_service,
            rgb_service=app.state.rgb_service,
        )

        async def broadcast_callback(msg):
            await manager.broadcast_to_device("lelamp", msg)

        agent.send_message_callback = broadcast_callback
        app.state.agent = agent
        logger.info("LeLamp Agent initialized")
    except Exception as e:
        logger.error(f"LeLamp Agent init failed: {e}", exc_info=True)
        app.state.agent = None

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

    # 停止硬件服务
    if getattr(app.state, "motors_service", None):
        app.state.motors_service.stop()
    if getattr(app.state, "rgb_service", None):
        app.state.rgb_service.stop()


app.router.lifespan_context = lifespan

# 包含 API 路由
app.include_router(api_router)

# 静态文件服务（前端）
frontend_dist = os.path.join(os.path.dirname(__file__), "..", "..", "web", "dist")
if os.path.exists(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
    logger.info(f"Frontend static files mounted from: {frontend_dist}")
else:
    logger.warning(f"Frontend dist directory not found: {frontend_dist}")

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
