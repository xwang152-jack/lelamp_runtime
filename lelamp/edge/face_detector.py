"""
MediaPipe 人脸检测服务

用于用户在场检测、自动唤醒/休眠功能。
"""
import logging
import os
import time
from typing import Optional, Callable, List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger("lelamp.edge.face")

# MediaPipe 是可选依赖，优雅降级
MEDIAPIPE_AVAILABLE = False
try:
    import mediapipe as mp
    import cv2
    # 检查是否有 tasks API（新版 MediaPipe 0.10+）
    if hasattr(mp, 'tasks'):
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        MEDIAPIPE_AVAILABLE = True
    else:
        logger.warning("MediaPipe installed but 'tasks' API not available. "
                      "FaceDetector will run in NoOp mode.")
except ImportError:
    logger.warning("MediaPipe not available, FaceDetector will run in NoOp mode")


@dataclass
class FaceInfo:
    """人脸信息"""
    bbox: List[int]  # [x, y, width, height]
    confidence: float
    center: tuple  # (x, y) 归一化坐标


class FaceDetector:
    """
    基于 MediaPipe Tasks API 的人脸检测服务

    用途：
    - 用户在场检测 → 自动唤醒/休眠
    - 多人检测 → 区分主用户
    - 视线追踪 → 台灯跟随

    使用示例:
        detector = FaceDetector(
            presence_callback=lambda present: print(f"用户{'在场' if present else '离开'}")
        )

        while True:
            ret, frame = cap.read()
            result = detector.detect(frame)
            if result["presence"]:
                print(f"检测到 {result['count']} 个人")
    """

    def __init__(
        self,
        model_selection: int = 0,
        min_detection_confidence: float = 0.5,
        presence_callback: Optional[Callable[[bool], None]] = None,
        presence_threshold_s: float = 2.0,
        absence_threshold_s: float = 5.0,
    ):
        """
        初始化人脸检测器

        Args:
            model_selection: 模型选择
                0: 短距离模型（适合 2 米内）
                1: 远距离模型（适合 5 米内）
            min_detection_confidence: 最小检测置信度 (0.0-1.0)
            presence_callback: 在场状态变化回调
            presence_threshold_s: 持续检测到人脸的时长阈值（秒）
            absence_threshold_s: 持续未检测到人脸的时长阈值（秒）
        """
        self._noop = not MEDIAPIPE_AVAILABLE

        if not self._noop:
            model_path = self._get_model_path(model_selection)
            if model_path and os.path.exists(model_path):
                try:
                    base_options = python.BaseOptions(model_asset_path=model_path)
                    options = vision.FaceDetectorOptions(
                        base_options=base_options,
                        running_mode=vision.RunningMode.IMAGE,
                        min_detection_confidence=min_detection_confidence,
                    )
                    self.detector = vision.FaceDetector.create_from_options(options)
                except Exception as e:
                    logger.error(f"Failed to initialize FaceDetector: {e}")
                    self._noop = True
            else:
                logger.warning(f"Face detection model not found at {model_path}. "
                              "FaceDetector will run in NoOp mode.")
                self._noop = True

        self.presence_callback = presence_callback
        self._last_presence = False
        self._presence_threshold_s = presence_threshold_s
        self._absence_threshold_s = absence_threshold_s
        self._first_seen_time: Optional[float] = None
        self._last_seen_time: float = 0
        self._last_absent_time: float = time.time()

        # 统计信息
        self._total_detections = 0
        self._presence_changes = 0

        mode = "NoOp" if self._noop else "MediaPipe Tasks"
        logger.info(f"FaceDetector initialized ({mode} mode)")

    def _get_model_path(self, model_selection: int) -> Optional[str]:
        """获取模型文件路径"""
        model_files = {
            0: "blaze_face_short_range.tflite",
            1: "blaze_face_full_range.tflite",
        }
        model_file = model_files.get(model_selection, "blaze_face_full_range.tflite")

        possible_paths = [
            os.path.join(os.path.dirname(__file__), "..", "..", "models", model_file),
            os.path.expanduser(f"~/.mediapipe/models/{model_file}"),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        # 回退：尝试任何可用的人脸检测模型
        for path in [
            os.path.join(os.path.dirname(__file__), "..", "..", "models", "blaze_face_full_range.tflite"),
            os.path.join(os.path.dirname(__file__), "..", "..", "models", "blaze_face_short_range.tflite"),
        ]:
            if os.path.exists(path):
                return path

        return None

    def detect(self, frame) -> Dict[str, Any]:
        """
        检测帧中的人脸

        Args:
            frame: BGR 格式的图像帧 (OpenCV 格式)

        Returns:
            {
                "faces": [FaceInfo, ...],
                "count": int,
                "presence": bool,
                "main_face_center": (x, y) or None,
                "presence_duration": float  # 在场持续时间（秒）
            }
        """
        if self._noop:
            return self._noop_detect()

        try:
            # 转换为 RGB
            rgb_frame = frame
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 创建 MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # 执行检测
            results = self.detector.detect(mp_image)

            faces: List[FaceInfo] = []
            main_face_center: Optional[tuple] = None

            if results.detections:
                self._total_detections += 1
                h, w = frame.shape[:2]

                for i, detection in enumerate(results.detections):
                    # bounding_box 是像素坐标
                    bbox = detection.bounding_box
                    face_bbox = [
                        bbox.origin_x,
                        bbox.origin_y,
                        bbox.width,
                        bbox.height
                    ]
                    confidence = detection.categories[0].score

                    # 计算归一化中心坐标
                    center = (
                        (bbox.origin_x + bbox.width / 2) / w,
                        (bbox.origin_y + bbox.height / 2) / h,
                    )

                    faces.append(FaceInfo(
                        bbox=face_bbox,
                        confidence=confidence,
                        center=center
                    ))

                    # 主人脸（第一个检测到的）
                    if i == 0:
                        main_face_center = center

            presence = len(faces) > 0
            presence_duration = self._update_presence_state(presence)

            return {
                "faces": faces,
                "count": len(faces),
                "presence": presence,
                "main_face_center": main_face_center,
                "presence_duration": presence_duration
            }
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return self._noop_detect()

    def _update_presence_state(self, current_presence: bool) -> float:
        """
        更新在场状态，带防抖

        Returns:
            当前在场持续时间（秒）
        """
        now = time.time()
        presence_duration = 0.0

        if current_presence:
            if self._first_seen_time is None:
                self._first_seen_time = now
            self._last_seen_time = now
            presence_duration = now - self._first_seen_time

            # 持续检测到人脸超过阈值 → 触发在场回调
            if not self._last_presence:
                if presence_duration >= self._presence_threshold_s:
                    self._last_presence = True
                    self._presence_changes += 1
                    logger.info(f"用户在场 (持续 {presence_duration:.1f}s)")
                    if self.presence_callback:
                        try:
                            self.presence_callback(True)
                        except Exception as e:
                            logger.error(f"Presence callback error: {e}")
        else:
            self._first_seen_time = None
            absence_duration = now - self._last_seen_time
            presence_duration = 0.0

            # 持续未检测到人脸超过阈值 → 触发离场回调
            if self._last_presence:
                if absence_duration >= self._absence_threshold_s:
                    self._last_presence = False
                    self._presence_changes += 1
                    logger.info(f"用户离开 (离开 {absence_duration:.1f}s)")
                    if self.presence_callback:
                        try:
                            self.presence_callback(False)
                        except Exception as e:
                            logger.error(f"Presence callback error: {e}")

        return presence_duration

    def _noop_detect(self) -> Dict[str, Any]:
        """NoOp 模式的默认返回"""
        return {
            "faces": [],
            "count": 0,
            "presence": False,
            "main_face_center": None,
            "presence_duration": 0.0
        }

    @property
    def is_present(self) -> bool:
        """当前用户是否在场"""
        return self._last_presence

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_detections": self._total_detections,
            "presence_changes": self._presence_changes,
            "current_presence": self._last_presence,
            "noop_mode": self._noop
        }

    def reset_stats(self):
        """重置统计信息"""
        self._total_detections = 0
        self._presence_changes = 0

    def close(self):
        """释放资源"""
        if not self._noop and hasattr(self, 'detector'):
            self.detector.close()
            logger.info("FaceDetector closed")
