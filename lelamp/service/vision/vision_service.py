import asyncio
import base64
import threading
import time
import numpy as np
from typing import Any, Callable, Optional

from ..base import ServiceBase
from .privacy import CameraPrivacyManager, PrivacyConfig, PrivacyGuard, CameraState


class VisionService(ServiceBase):
    def __init__(
        self,
        *,
        enabled: bool = True,
        index_or_path: int | str = 0,
        width: int = 1024,
        height: int = 768,
        capture_interval_s: float = 2.5,
        jpeg_quality: int = 92,
        max_age_s: float = 15.0,
        rotate_deg: int = 0,
        flip: str = "none",
        # 隐私保护配置
        enable_privacy_protection: bool = True,
        privacy_config: PrivacyConfig | None = None,
        rgb_setter: Callable[[tuple], None] | None = None,
        notification_shower: Callable[[str], None] | None = None,
    ):
        super().__init__("vision")
        self.enabled = enabled
        self.index_or_path = index_or_path
        self.width = width
        self.height = height
        self.capture_interval_s = capture_interval_s
        self.jpeg_quality = jpeg_quality
        self.max_age_s = max_age_s
        self.rotate_deg = int(rotate_deg)
        self.flip = str(flip or "none")

        self._camera = None
        self._camera_thread: threading.Thread | None = None
        self._camera_stop = threading.Event()

        self._frame_lock = threading.Lock()
        self._latest_jpeg_b64: str | None = None
        self._latest_ts: float = 0.0
        self._force_capture = False

        # 隐私保护
        self.enable_privacy_protection = enable_privacy_protection
        if privacy_config is None:
            privacy_config = PrivacyConfig()
        self._privacy_manager = CameraPrivacyManager(
            config=privacy_config,
            rgb_setter=rgb_setter,
            notification_shower=notification_shower,
        )

    def start(self):
        super().start()
        if self.enabled:
            self._privacy_manager.start()
            self._start_camera_thread()

    def stop(self, timeout: float = 5.0):
        self.enabled = False
        self._privacy_manager.stop()
        self._stop_camera_thread(timeout=timeout)
        super().stop(timeout)

    def handle_event(self, event_type: str, payload: Any):
        if event_type == "enable":
            self.enabled = True
            self._start_camera_thread()
            return
        if event_type == "disable":
            self.enabled = False
            self._stop_camera_thread(timeout=5.0)
            with self._frame_lock:
                self._latest_jpeg_b64 = None
                self._latest_ts = 0.0
            return
        if event_type == "config" and isinstance(payload, dict):
            self._apply_config(payload)
            return
        self.logger.warning(f"Unknown event type: {event_type}")

    def _apply_config(self, cfg: dict[str, Any]) -> None:
        if "enabled" in cfg:
            self.enabled = bool(cfg["enabled"])
        if "index_or_path" in cfg:
            self.index_or_path = cfg["index_or_path"]
        if "width" in cfg:
            self.width = int(cfg["width"])
        if "height" in cfg:
            self.height = int(cfg["height"])
        if "capture_interval_s" in cfg:
            self.capture_interval_s = float(cfg["capture_interval_s"])
        if "jpeg_quality" in cfg:
            self.jpeg_quality = int(cfg["jpeg_quality"])
        if "max_age_s" in cfg:
            self.max_age_s = float(cfg["max_age_s"])
        if "rotate_deg" in cfg:
            self.rotate_deg = int(cfg["rotate_deg"])
        if "flip" in cfg:
            self.flip = str(cfg["flip"] or "none")

        if self.enabled:
            self._start_camera_thread()
        else:
            self._stop_camera_thread(timeout=5.0)

    def _apply_transform(self, frame_bgr):
        import cv2

        deg = int(self.rotate_deg or 0) % 360
        if deg == 90:
            frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
        elif deg == 180:
            frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_180)
        elif deg == 270:
            frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

        flip = (self.flip or "none").strip().lower()
        if flip in ("h", "horizontal", "x"):
            frame_bgr = cv2.flip(frame_bgr, 1)
        elif flip in ("v", "vertical", "y"):
            frame_bgr = cv2.flip(frame_bgr, 0)
        elif flip in ("hv", "vh", "both", "xy"):
            frame_bgr = cv2.flip(frame_bgr, -1)

        return frame_bgr

    def _start_camera_thread(self):
        if self._camera_thread and self._camera_thread.is_alive():
            return
        self._camera_stop.clear()
        self._camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._camera_thread.start()

    def _stop_camera_thread(self, *, timeout: float):
        self._camera_stop.set()
        if self._camera_thread and self._camera_thread.is_alive():
            self._camera_thread.join(timeout=timeout)
        self._camera_thread = None

        cam = self._camera
        self._camera = None
        if cam is not None:
            try:
                cam.release()
            except Exception:
                pass

    def _open_camera(self):
        import cv2
        import time

        def _try_open(source):
            cap = cv2.VideoCapture(source)
            if not cap or not cap.isOpened():
                try:
                    cap.release()
                except Exception:
                    pass
                return None
            # 摄像头预热（macOS USB 摄像头需要额外初始化时间）
            time.sleep(0.3)
            # 验证可以读取帧
            for _ in range(3):
                ok, frame = cap.read()
                if ok and frame is not None:
                    break
                time.sleep(0.1)
            else:
                cap.release()
                return None
            # 设置分辨率
            if self.width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
            if self.height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
            return cap

        # 首选配置的索引或路径
        candidates: list[int | str] = [self.index_or_path]
        # 在 macOS 或多摄像头环境中，0 可能不可用，尝试备用索引
        if isinstance(self.index_or_path, int):
            for alt in (0, 1, 2, 3):
                if alt not in candidates:
                    candidates.append(alt)

        for cand in candidates:
            cap = _try_open(cand)
            if cap is not None:
                if cand != self.index_or_path:
                    self.logger.info(f"Camera auto-detected at index/path: {cand} (was {self.index_or_path})")
                    self.index_or_path = cand
                return cap

        raise RuntimeError(f"无法打开摄像头（尝试: {candidates}）")

    def _encode_bgr_to_jpeg_b64(self, frame_bgr) -> str:
        import cv2

        ok, buf = cv2.imencode(
            ".jpg",
            frame_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)],
        )
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def _camera_loop(self):
        self.logger.info("VisionService _camera_loop started")
        try:
            import cv2
        except Exception as e:
            self.logger.error(f"Vision 不可用：无法导入 OpenCV: {e}")
            return

        last_capture_ts = 0.0

        while not self._camera_stop.is_set():
            if not self.enabled:
                time.sleep(0.1)
                continue

            if self._camera is None:
                try:
                    self._camera = self._open_camera()
                except Exception as e:
                    self.logger.error(f"Vision 打开摄像头失败: {e}")
                    time.sleep(1.0)
                    continue

            now = time.time()
            if not self._force_capture and now - last_capture_ts < float(self.capture_interval_s):
                time.sleep(0.02)
                continue

            if self._camera is None or not self._camera.isOpened():
                self.logger.warning(f"Vision 摄像头未打开或已关闭, isOpened={self._camera.isOpened() if self._camera else 'camera is None'}")
                try:
                    if self._camera:
                        self._camera.release()
                except Exception:
                    pass
                self._camera = None
                time.sleep(0.2)
                continue

            self._force_capture = False

            # Flush camera buffer: grab several frames to get the latest one
            for _ in range(5):
                self._camera.grab()

            ok, frame = self._camera.read()
            if not ok or frame is None:
                self.logger.warning(f"Vision 读取帧失败: ok={ok}, frame={frame is None}, 重新打开摄像头")
                try:
                    self._camera.release()
                except Exception:
                    pass
                self._camera = None
                time.sleep(0.2)
                continue

            try:
                frame = self._apply_transform(frame)
                jpeg_b64 = self._encode_bgr_to_jpeg_b64(frame)
            except Exception as e:
                self.logger.warning(f"Vision 编码 JPEG 失败: {e}")
                time.sleep(0.1)
                continue

            self.logger.debug(f"Vision 成功捕获帧: {len(jpeg_b64)} bytes")
            with self._frame_lock:
                self._latest_jpeg_b64 = jpeg_b64
                self._latest_ts = now

            last_capture_ts = now
            time.sleep(0.01)

    async def get_latest_jpeg_b64(self) -> tuple[str, float] | None:
        def _get():
            with self._frame_lock:
                if not self._latest_jpeg_b64:
                    return None
                if time.time() - self._latest_ts > float(self.max_age_s):
                    return None
                return (self._latest_jpeg_b64, self._latest_ts)

        return await asyncio.to_thread(_get)

    async def get_fresh_jpeg_b64(self, timeout_s: float = 5.0) -> tuple[str, float] | None:
        """Wait for a new frame to be captured and return it."""
        start_time = time.time()
        self.trigger_capture()
        
        while time.time() - start_time < timeout_s:
            latest = await self.get_latest_jpeg_b64()
            if latest:
                _, ts = latest
                if ts >= start_time:
                    return latest
            await asyncio.sleep(0.1)
            
        return await self.get_latest_jpeg_b64()

    def trigger_capture(self):
        """Force the camera loop to capture a frame immediately."""
        self._force_capture = True

    # ========================================================================
    # 隐私保护相关方法
    # ========================================================================

    @property
    def privacy_manager(self) -> CameraPrivacyManager:
        """获取隐私管理器"""
        return self._privacy_manager

    async def get_latest_jpeg_b64_with_privacy(self) -> tuple[str, float] | None:
        """
        获取最新帧（带隐私保护）

        Returns:
            JPEG base64 数据和时间戳，如果摄像头未激活则返回 None
        """
        if not self.enable_privacy_protection:
            return await self.get_latest_jpeg_b64()

        # 检查摄像头是否激活
        if not self._privacy_manager.is_active:
            self.logger.debug("摄像头未激活，使用隐私保护")
            return None

        return await self.get_latest_jpeg_b64()

    async def activate_camera(self) -> bool:
        """
        激活摄像头（带用户同意）

        Returns:
            是否成功激活
        """
        if not self.enable_privacy_protection:
            self.logger.debug("隐私保护未启用，直接激活摄像头")
            return True

        success = await self._privacy_manager.activate_camera()
        if success:
            self.logger.info("摄像头已激活（通过隐私保护）")
        else:
            self.logger.warning("摄像头激活失败：用户未同意或超时")

        return success

    def deactivate_camera(self):
        """停用摄像头"""
        if not self.enable_privacy_protection:
            return

        self._privacy_manager.deactivate_camera()
        self.logger.info("摄像头已停用（通过隐私保护）")

    def grant_camera_consent(self):
        """
        授予摄像头使用同意

        可由外部触发（例如：按钮、语音命令）
        """
        self._privacy_manager.grant_consent()
        self.logger.info("用户授予摄像头使用同意")

    def revoke_camera_consent(self):
        """
        撤销摄像头使用同意

        可由外部触发（例如：按钮、语音命令）
        """
        self._privacy_manager.revoke_consent()
        self.logger.info("用户撤销摄像头使用同意")

    def get_latest_frame(self):
        """
        获取最新的原始帧（用于边缘视觉处理）

        Returns:
            numpy.ndarray 或 None：BGR 格式的图像帧
        """
        if not self.enabled:
            return None
        if not self._camera_thread:
            return None

        # 触发一次新的捕获
        self.trigger_capture()

        # 等待新帧可用（最多等待1秒）
        import time
        start_wait = time.time()
        timeout = 1.0

        while time.time() - start_wait < timeout:
            with self._frame_lock:
                if self._latest_jpeg_b64 and self._latest_ts > 0:
                    # 解码 JPEG 获取原始帧
                    import cv2
                    import base64
                    try:
                        jpeg_bytes = base64.b64decode(self._latest_jpeg_b64)
                        frame = cv2.imdecode(
                            np.frombuffer(jpeg_bytes, dtype=np.uint8),
                            cv2.IMREAD_COLOR
                        )
                        return frame
                    except Exception as e:
                        self.logger.warning(f"Failed to decode frame: {e}")
                        return None
            time.sleep(0.05)  # 短暂等待

        self.logger.debug(f"VisionService.get_latest_frame timeout: _latest_jpeg_b64={bool(self._latest_jpeg_b64)}, _latest_ts={self._latest_ts}")
        return None

    def get_camera_stats(self) -> dict:
        """
        获取摄像头使用统计

        Returns:
            包含使用统计的字典
        """
        stats = {
            "enabled": self.enabled,
            "privacy_protection_enabled": self.enable_privacy_protection,
        }

        if self.enable_privacy_protection:
            stats.update(self._privacy_manager.get_stats())

        return stats

    def create_privacy_guard(self):
        """
        创建隐私保护上下文管理器

        Example:
            async with vision_service.create_privacy_guard():
                frame = await vision_service.get_fresh_jpeg_b64()
        """
        return PrivacyGuard(self._privacy_manager, auto_activate=True)
