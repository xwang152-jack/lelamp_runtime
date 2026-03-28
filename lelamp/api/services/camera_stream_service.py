"""
摄像头帧推送服务

定期从 VisionService 获取最新帧并通过 WebSocket 推送给客户端。
"""
import asyncio
import logging
from typing import Optional

logger = logging.getLogger("lelamp.api.camera_stream")


class CameraStreamService:
    """
    摄像头流推送服务

    功能：
    - 定期从 VisionService 获取最新帧
    - 通过 WebSocket 推送给订阅的客户端
    - 支持隐私保护（未激活时不推送）
    - 可配置推送帧率
    """

    def __init__(
        self,
        lamp_id: str = "lelamp",
        push_fps: int = 10,  # 推送帧率，默认 10fps
        enable_stream: bool = True,
    ):
        """
        初始化摄像头流推送服务

        Args:
            lamp_id: 设备 ID
            push_fps: 推送帧率（每秒推送帧数）
            enable_stream: 是否启用流推送
        """
        self.lamp_id = lamp_id
        self.push_fps = push_fps
        self.enable_stream = enable_stream

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._vision_service = None
        self._push_interval = 1.0 / push_fps if push_fps > 0 else 0.2

        logger.info(
            f"CameraStreamService initialized "
            f"(lamp_id={lamp_id}, fps={push_fps})"
        )

    def set_vision_service(self, vision_service):
        """设置视觉服务实例"""
        self._vision_service = vision_service
        logger.info("VisionService linked to CameraStreamService")

    async def start(self):
        """启动摄像头流推送"""
        if self._running:
            logger.warning("CameraStreamService already running")
            return

        self._running = True

        # 推送初始摄像头状态
        await self._push_initial_status()

        self._task = asyncio.create_task(self._stream_loop())
        logger.info(f"CameraStreamService started for {self.lamp_id}")

    async def _push_initial_status(self):
        """推送初始摄像头状态"""
        try:
            from lelamp.api.routes.websocket import push_camera_status

            # 检查视觉服务是否启用
            is_enabled = False
            is_active = False
            privacy_granted = False

            if self._vision_service:
                is_enabled = self._vision_service.enabled
                # 检查隐私管理器状态
                if hasattr(self._vision_service, 'privacy_manager'):
                    privacy_mgr = self._vision_service.privacy_manager
                    is_active = privacy_mgr.is_active
                    privacy_granted = privacy_mgr.has_consent
                else:
                    # 没有隐私保护，直接视为激活
                    is_active = is_enabled

            # 如果摄像头启用但未激活（隐私保护），自动激活
            if is_enabled and not is_active:
                logger.info("Camera enabled but not active (privacy protection), activating...")
                if self._vision_service and hasattr(self._vision_service, 'activate_camera'):
                    is_active = await self._vision_service.activate_camera()
                    if is_active:
                        privacy_granted = True

            # 推送状态
            await push_camera_status(
                self.lamp_id,
                is_active,
                privacy_granted if is_active else None
            )
            logger.info(f"Camera status pushed: enabled={is_enabled}, active={is_active}")

        except Exception as e:
            logger.error(f"Error pushing initial camera status: {e}")

    async def stop(self):
        """停止摄像头流推送"""
        if not self._running:
            return

        logger.info("Stopping CameraStreamService...")
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        logger.info("CameraStreamService stopped")

    async def _stream_loop(self):
        """流推送主循环"""
        logger.info("Camera stream loop started")

        try:
            from lelamp.api.routes.websocket import push_camera_frame, push_camera_status
            from lelamp.api.routes.websocket import manager

            # 检查是否需要激活摄像头
            activation_tried = False

            while self._running:
                start_time = asyncio.get_event_loop().time()

                # 检查是否启用流推送
                if not self.enable_stream:
                    await asyncio.sleep(self._push_interval)
                    continue

                # 检查视觉服务是否可用
                if not self._vision_service:
                    await asyncio.sleep(self._push_interval)
                    continue

                # 尝试激活摄像头（仅首次）
                if not activation_tried:
                    try:
                        if hasattr(self._vision_service, 'privacy_manager'):
                            privacy_mgr = self._vision_service.privacy_manager
                            if not privacy_mgr.is_active:
                                logger.info("Auto-activating camera for streaming...")
                                # 自动授予同意并激活
                                self._vision_service.grant_camera_consent()
                                result = await self._vision_service.activate_camera()
                                if result:
                                    await push_camera_status(self.lamp_id, True, True)
                                    logger.info("Camera auto-activated successfully")
                        activation_tried = True
                    except Exception as e:
                        logger.error(f"Error auto-activating camera: {e}")
                        activation_tried = True

                # 获取最新帧
                try:
                    frame_data = await self._vision_service.get_latest_jpeg_b64_with_privacy()
                    if frame_data:
                        frame_b64, timestamp = frame_data

                        # 推送帧
                        await push_camera_frame(
                            self.lamp_id,
                            frame_b64,
                            width=self._vision_service.width,
                            height=self._vision_service.height,
                        )

                        # 调试日志（每30帧打印一次）
                        if hasattr(self, '_frame_count'):
                            self._frame_count += 1
                        else:
                            self._frame_count = 1

                        if self._frame_count % 30 == 1:
                            logger.debug(
                                f"Pushed camera frame {self._frame_count} "
                                f"({len(frame_b64)} bytes)"
                            )

                except Exception as e:
                    logger.error(f"Error getting/pushing camera frame: {e}")

                # 计算剩余等待时间
                elapsed = asyncio.get_event_loop().time() - start_time
                wait_time = max(0, self._push_interval - elapsed)
                await asyncio.sleep(wait_time)

        except asyncio.CancelledError:
            logger.info("Camera stream loop cancelled")
        except Exception as e:
            logger.error(f"Error in camera stream loop: {e}", exc_info=True)

        logger.info("Camera stream loop ended")

    def set_push_fps(self, fps: int):
        """
        设置推送帧率

        Args:
            fps: 新的推送帧率
        """
        self.push_fps = max(1, min(30, fps))  # 限制在 1-30fps
        self._push_interval = 1.0 / self.push_fps
        logger.info(f"Camera stream FPS set to {self.push_fps}")

    def enable(self):
        """启用流推送"""
        self.enable_stream = True
        logger.info("Camera stream enabled")

    def disable(self):
        """禁用流推送"""
        self.enable_stream = False
        logger.info("Camera stream disabled")


# 全局服务实例字典
_camera_stream_services: dict[str, CameraStreamService] = {}


def get_camera_stream_service(lamp_id: str = "lelamp") -> CameraStreamService:
    """
    获取或创建摄像头流推送服务

    Args:
        lamp_id: 设备 ID

    Returns:
        CameraStreamService 实例
    """
    if lamp_id not in _camera_stream_services:
        _camera_stream_services[lamp_id] = CameraStreamService(lamp_id=lamp_id)
    return _camera_stream_services[lamp_id]
