import asyncio
import base64
import threading
import time
from typing import Any

from ..base import ServiceBase


class VisionService(ServiceBase):
    def __init__(
        self,
        *,
        enabled: bool = True,
        index_or_path: int | str = 0,
        width: int = 640,
        height: int = 640,
        capture_interval_s: float = 2.5,
        jpeg_quality: int = 92,
        max_age_s: float = 15.0,
    ):
        super().__init__("vision")
        self.enabled = enabled
        self.index_or_path = index_or_path
        self.width = width
        self.height = height
        self.capture_interval_s = capture_interval_s
        self.jpeg_quality = jpeg_quality
        self.max_age_s = max_age_s

        self._camera = None
        self._camera_thread: threading.Thread | None = None
        self._camera_stop = threading.Event()

        self._frame_lock = threading.Lock()
        self._latest_jpeg_b64: str | None = None
        self._latest_ts: float = 0.0

    def start(self):
        super().start()
        if self.enabled:
            self._start_camera_thread()

    def stop(self, timeout: float = 5.0):
        self.enabled = False
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

        if self.enabled:
            self._start_camera_thread()
        else:
            self._stop_camera_thread(timeout=5.0)

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

        cap = cv2.VideoCapture(self.index_or_path)
        if self.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        if self.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        return cap

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
            if now - last_capture_ts < float(self.capture_interval_s):
                time.sleep(0.02)
                continue

            ok, frame = self._camera.read()
            if not ok or frame is None:
                try:
                    self._camera.release()
                except Exception:
                    pass
                self._camera = None
                time.sleep(0.2)
                continue

            try:
                jpeg_b64 = self._encode_bgr_to_jpeg_b64(frame)
            except Exception as e:
                self.logger.warning(f"Vision 编码 JPEG 失败: {e}")
                time.sleep(0.1)
                continue

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
