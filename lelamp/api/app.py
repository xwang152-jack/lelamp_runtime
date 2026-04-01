"""
FastAPI 应用主文件
"""
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, FileResponse
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

        # 内容安全策略 (允许 Vue/Element Plus 运行所需的内联脚本和样式)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; "
            "font-src 'self' data:;"
        )

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

    # 初始化数据库（确保表已创建）
    try:
        from lelamp.database.base import init_db
        init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Database init failed: {e}")

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

    # 初始化 VisionService（用于边缘视觉监控）
    vision_service = None
    try:
        from lelamp.service.vision.vision_service import VisionService
        from lelamp.config import load_vision_config

        vision_cfg = load_vision_config()
        if vision_cfg.enabled:
            vision_service = VisionService(
                enabled=vision_cfg.enabled,
                index_or_path=vision_cfg.index_or_path,
                width=vision_cfg.width,
                height=vision_cfg.height,
                capture_interval_s=0.5,  # API 模式下平衡性能（2fps捕获）
                jpeg_quality=85,  # 降低质量以减少带宽
                max_age_s=vision_cfg.max_age_s,
                rotate_deg=vision_cfg.rotate_deg,
                flip=vision_cfg.flip,
                enable_privacy_protection=False,  # API 模式下不需要隐私保护
            )
            vision_service.start()
            app.state.vision_service = vision_service
            logger.info("VisionService started")
    except Exception as e:
        logger.error(f"VisionService start failed: {e}")
        vision_service = None

    # 初始化 LeLamp Agent
    try:
        from lelamp.agent.lelamp_agent import LeLamp
        agent = LeLamp(
            port="/dev/ttyACM0",
            lamp_id="lelamp",
            motors_service=app.state.motors_service,
            rgb_service=app.state.rgb_service,
            vision_service=vision_service,  # 传入 vision_service
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

    # 启动摄像头流推送服务
    camera_stream_task = None
    if vision_service:
        try:
            from lelamp.api.services.camera_stream_service import get_camera_stream_service
            camera_stream = get_camera_stream_service("lelamp")
            camera_stream.set_vision_service(vision_service)

            # 注入主动视觉监控器（如有）
            if hasattr(app.state, 'agent') and app.state.agent:
                monitor = getattr(app.state.agent, '_vision_monitor', None)
                if monitor:
                    camera_stream.set_proactive_monitor(monitor)
                    logger.info("ProactiveVisionMonitor linked to camera stream")

            camera_stream_task = asyncio.create_task(camera_stream.start())
            logger.info("Camera stream service started")
        except Exception as e:
            logger.error(f"Camera stream service start failed: {e}")

    yield

    # 关闭时执行
    logger.info("LeLamp API 关闭")

    # 取消后台任务
    polling_task.cancel()
    try:
        await polling_task
    except asyncio.CancelledError:
        pass

    # 停止摄像头流推送服务
    if camera_stream_task:
        camera_stream_task.cancel()
        try:
            await camera_stream_task
        except asyncio.CancelledError:
            pass
        try:
            from lelamp.api.services.camera_stream_service import get_camera_stream_service
            camera_stream = get_camera_stream_service("lelamp")
            await camera_stream.stop()
        except Exception as e:
            logger.error(f"Camera stream stop failed: {e}")

    # 停止硬件服务
    if getattr(app.state, "motors_service", None):
        app.state.motors_service.stop()
    if getattr(app.state, "rgb_service", None):
        app.state.rgb_service.stop()
    if getattr(app.state, "vision_service", None):
        app.state.vision_service.stop()


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


# =============================================================================
# Vue 前端静态文件托管
# =============================================================================

_WEB_DIST = Path(os.getenv("LELAMP_WEB_DIST", "web/dist"))
_SPA_INDEX = _WEB_DIST / "index.html"


def _mount_vue_frontend():
    """挂载 Vue 构建产物，仅在 dist/index.html 存在时启用。"""
    if not _SPA_INDEX.is_file():
        logger.info(f"Vue 前端未构建（{_SPA_INDEX} 不存在），跳过静态文件托管")
        return

    assets_dir = _WEB_DIST / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="vue-assets")
        logger.info(f"已挂载静态资源目录: {assets_dir}")

    # SPA fallback：非 API 路径返回 index.html
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        return FileResponse(str(_SPA_INDEX))

    logger.info(f"Vue 前端已启用，静态文件目录: {_WEB_DIST}")


_mount_vue_frontend()
