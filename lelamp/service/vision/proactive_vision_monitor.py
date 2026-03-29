"""
主动视觉监听服务

实现边缘视觉的主动服务模式：
- 持续监听用户手势，自动触发响应
- 持续检测用户在场，自动唤醒/休眠
- 智能采样，平衡性能和响应速度
"""
import asyncio
import threading
import time
import logging
from typing import Optional, Callable, Dict, Any, List
from enum import Enum

logger = logging.getLogger("lelamp.vision.monitor")


class MonitorMode(Enum):
    """监听模式"""
    ACTIVE = "active"      # 主动模式：高频率检测
    IDLE = "idle"         # 空闲模式：低频率检测
    SLEEP = "sleep"       # 休眠模式：暂停检测


class ProactiveVisionMonitor:
    """
    主动视觉监听服务

    功能：
    - 后台持续监听手势和用户
    - 检测到变化时自动触发回调
    - 智能采样频率调节
    - 线程安全的事件通知
    """

    def __init__(
        self,
        # 视觉检测服务
        hybrid_vision=None,
        vision_service=None,

        # 回调函数
        gesture_callback: Optional[Callable] = None,
        presence_callback: Optional[Callable] = None,

        # 配置
        enable_auto_gesture: bool = True,
        enable_auto_presence: bool = True,

        # 采样配置
        active_fps: int = 10,          # 主动模式 FPS
        idle_fps: int = 2,             # 空闲模式 FPS
        presence_check_interval_s: float = 1.0,  # 在场检测间隔

        # 智能调节
        auto_adjust_fps: bool = True,
        min_user_presence_time: float = 2.0,  # 确认用户在场的时间(秒)
    ):
        """
        初始化主动监听服务

        Args:
            hybrid_vision: 混合视觉服务实例
            vision_service: 摄像头服务实例
            gesture_callback: 手势检测回调
            presence_callback: 用户在场回调
            enable_auto_gesture: 启用自动手势检测
            enable_auto_presence: 启用自动在场检测
            active_fps: 主动模式帧率
            idle_fps: 空闲模式帧率
            presence_check_interval_s: 在场检测间隔
            auto_adjust_fps: 自动调节帧率
            min_user_presence_time: 确认用户在场最短时间
        """
        self._hybrid_vision = hybrid_vision
        self._vision_service = vision_service

        self._gesture_callback = gesture_callback
        self._presence_callback = presence_callback

        self._enable_auto_gesture = enable_auto_gesture
        self._enable_auto_presence = enable_auto_presence

        # 采样配置
        self._active_interval = 1.0 / active_fps if active_fps > 0 else 0.1
        self._idle_interval = 1.0 / idle_fps if idle_fps > 0 else 0.5
        self._presence_interval = presence_check_interval_s

        # 智能调节参数
        self._auto_adjust = auto_adjust_fps
        self._min_presence_time = min_user_presence_time

        # 状态
        self._mode = MonitorMode.IDLE
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # 统计
        self._user_present = False
        self._user_present_since = 0.0
        self._last_gesture_time = 0.0
        self._last_presence_check = 0.0
        self._detection_count = 0
        self._gesture_count = 0

        # 最新检测结果缓存（线程安全，供外部读取）
        self._last_face_result: Optional[Dict[str, Any]] = None
        self._last_hand_result: Optional[Dict[str, Any]] = None

        # 线程安全
        self._state_lock = threading.Lock()

        logger.info(
            f"ProactiveVisionMonitor initialized "
            f"(gesture={enable_auto_gesture}, presence={enable_auto_presence}, "
            f"active_fps={active_fps}, idle_fps={idle_fps})"
        )

    def start(self):
        """启动主动监听服务"""
        if self._running:
            logger.warning("ProactiveVisionMonitor already running")
            return

        self._running = True
        self._stop_event.clear()

        # 启动监听线程
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="VisionMonitor",
            daemon=True
        )
        self._monitor_thread.start()

        logger.info("ProactiveVisionMonitor started in background thread")

    def stop(self):
        """停止主动监听服务"""
        if not self._running:
            return

        logger.info("Stopping ProactiveVisionMonitor...")
        self._running = False
        self._stop_event.set()

        # 等待线程结束
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
            if self._monitor_thread.is_alive():
                logger.warning("Monitor thread did not stop gracefully")

        logger.info("ProactiveVisionMonitor stopped")

    def _monitor_loop(self):
        """监听主循环"""
        logger.info("Vision monitor loop started")

        try:
            while self._running and not self._stop_event.is_set():
                start_time = time.time()

                # 获取当前采样间隔
                interval = self._get_sampling_interval()

                # 执行检测
                self._perform_detection()

                # 计算剩余等待时间
                elapsed = time.time() - start_time
                wait_time = max(0, interval - elapsed)

                # 等待或响应停止事件
                if self._stop_event.wait(wait_time):
                    break

        except Exception as e:
            logger.error(f"Error in vision monitor loop: {e}", exc_info=True)

        logger.info("Vision monitor loop ended")

    def _get_sampling_interval(self) -> float:
        """获取当前采样间隔"""
        with self._state_lock:
            if self._mode == MonitorMode.ACTIVE:
                return self._active_interval
            elif self._mode == MonitorMode.IDLE:
                return self._idle_interval
            else:  # SLEEP
                return 1.0  # 休眠时每秒检查一次

    def _perform_detection(self):
        """执行检测任务"""
        if not self._running:
            return

        current_time = time.time()

        # 获取最新帧
        frame = None
        if self._vision_service:
            frame = self._vision_service.get_latest_frame()
        else:
            logger.warning("VisionMonitor: _vision_service is None")

        if frame is None:
            if self._detection_count % 20 == 0:
                logger.warning(f"VisionMonitor 未获取到帧（已检测 {self._detection_count} 次）")
            return

        self._detection_count += 1
        if self._detection_count % 20 == 1:
            logger.info(f"VisionMonitor 运行中（检测 {self._detection_count} 次，帧 shape={frame.shape}）")

        # 1. 用户在场检测（优先级高，频率低）
        if self._enable_auto_presence and \
           (current_time - self._last_presence_check) >= self._presence_interval:
            self._check_presence(frame, current_time)
            self._last_presence_check = current_time

        # 2. 手势检测（频率高，仅在用户在场时）
        if self._enable_auto_gesture and self._user_present:
            self._detect_gestures(frame, current_time)

    def _check_presence(self, frame, current_time: float):
        """检测用户在场"""
        try:
            if not self._hybrid_vision:
                return

            result = self._hybrid_vision.detect_faces(frame)
            present = bool(result.get("presence", False))

            # 缓存帧尺寸（供 get_latest_detections 归一化使用）
            if hasattr(frame, 'shape'):
                fh, fw = frame.shape[:2]
                result["width"] = fw
                result["height"] = fh

            # 缓存人脸检测结果
            with self._state_lock:
                self._last_face_result = result
            if not present:
                faces_val = result.get("faces", [])
                if isinstance(faces_val, list):
                    present = len(faces_val) > 0
                else:
                    count_val = result.get("count", 0)
                    try:
                        present = int(count_val) > 0
                    except (TypeError, ValueError):
                        present = False

            with self._state_lock:
                # 用户状态变化
                if present != self._user_present:
                    if present:
                        # 用户刚出现
                        self._user_present_since = current_time
                        logger.info("用户出现在视野中")
                    else:
                        # 用户刚离开
                        if self._user_present and \
                           (current_time - self._user_present_since) >= self._min_presence_time:
                            logger.info("用户离开视野")
                            self._set_mode(MonitorMode.IDLE)

                    self._user_present = present

                    # 触发在场回调
                    if self._presence_callback:
                        try:
                            self._presence_callback(present)
                        except Exception as e:
                            logger.error(f"Presence callback error: {e}")

                # 智能调节采样频率
                if self._auto_adjust:
                    if present and self._mode == MonitorMode.IDLE:
                        # 确认用户在场一定时间后，切换到主动模式
                        if (current_time - self._user_present_since) >= self._min_presence_time:
                            self._set_mode(MonitorMode.ACTIVE)
                            logger.info("切换到主动监听模式")
                    elif not present and self._mode == MonitorMode.ACTIVE:
                        # 用户离开后，切换到空闲模式
                        self._set_mode(MonitorMode.IDLE)
                        logger.info("切换到空闲监听模式")

        except Exception as e:
            logger.error(f"Presence detection error: {e}")

    def _detect_gestures(self, frame, current_time: float):
        """检测手势"""
        try:
            if not self._hybrid_vision:
                return

            result = self._hybrid_vision.track_hands(frame)
            gestures = result.get("gestures", [])

            # 缓存手势检测结果
            with self._state_lock:
                self._last_hand_result = result

            if gestures:
                self._gesture_count += 1
                self._last_gesture_time = current_time

                logger.info(f"检测到手势: {[g.value for g in gestures]}")

                # 触发手势回调
                if self._gesture_callback:
                    try:
                        for gesture in gestures:
                            context = {
                                "timestamp": current_time,
                                "user_present": self._user_present,
                                "detection_count": self._detection_count
                            }
                            self._gesture_callback(gesture, context)
                    except Exception as e:
                        logger.error(f"Gesture callback error: {e}")

        except Exception as e:
            logger.error(f"Gesture detection error: {e}")

    def _set_mode(self, mode: MonitorMode):
        """设置监听模式"""
        if self._mode != mode:
            old_mode = self._mode
            self._mode = mode
            logger.info(f"监听模式变化: {old_mode.value} → {mode.value}")

    def get_stats(self) -> Dict[str, Any]:
        """获取监听统计"""
        with self._state_lock:
            return {
                "running": self._running,
                "mode": self._mode.value,
                "user_present": self._user_present,
                "user_present_duration": time.time() - self._user_present_since if self._user_present else 0,
                "detection_count": self._detection_count,
                "gesture_count": self._gesture_count,
                "last_gesture_time": self._last_gesture_time,
                "auto_gesture_enabled": self._enable_auto_gesture,
                "auto_presence_enabled": self._enable_auto_presence,
            }

    def set_mode(self, mode: str):
        """外部设置监听模式"""
        with self._state_lock:
            if mode == "active":
                self._set_mode(MonitorMode.ACTIVE)
            elif mode == "idle":
                self._set_mode(MonitorMode.IDLE)
            elif mode == "sleep":
                self._set_mode(MonitorMode.SLEEP)

    def get_latest_detections(self) -> Dict[str, Any]:
        """
        获取最新检测结果（线程安全）

        返回归一化坐标 (0-1) 的检测结果，供前端绘制标注。
        - faces: 包含 bbox 矩形框归一化坐标
        - hands: 包含 21 个手部关键点归一化坐标
        """
        with self._state_lock:
            result: Dict[str, Any] = {
                "faces": [],
                "hands": [],
                "presence": self._user_present,
                "mode": self._mode.value,
            }

            # 人脸检测结果 — 返回归一化 bbox 矩形框
            if self._last_face_result:
                for face in self._last_face_result.get("faces", []):
                    if hasattr(face, "bbox") and face.bbox:
                        # bbox 是像素坐标 [x, y, w, h]，需要归一化
                        # 从 _last_face_result 中获取帧尺寸
                        fw = self._last_face_result.get("width", 1)
                        fh = self._last_face_result.get("height", 1)
                        if fw <= 0:
                            fw = 1
                        if fh <= 0:
                            fh = 1
                        result["faces"].append({
                            "x": face.bbox[0] / fw,
                            "y": face.bbox[1] / fh,
                            "w": face.bbox[2] / fw,
                            "h": face.bbox[3] / fh,
                            "confidence": face.confidence,
                        })

            # 手势检测结果 — 返回完整 21 个关键点
            if self._last_hand_result:
                for hand in self._last_hand_result.get("hands", []):
                    if hasattr(hand, "landmarks") and hand.landmarks:
                        # landmarks: [(x, y, z), ...] 共 21 个，归一化坐标
                        result["hands"].append({
                            "landmarks": [{"x": lm[0], "y": lm[1], "z": lm[2]} for lm in hand.landmarks],
                            "gesture": hand.gesture.value if hasattr(hand.gesture, "value") else str(hand.gesture),
                            "handedness": hand.handedness,
                            "confidence": hand.confidence,
                        })

            return result
