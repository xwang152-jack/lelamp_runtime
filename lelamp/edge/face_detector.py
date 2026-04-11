"""
基于 OpenCV Haar 级联的人脸检测服务

用于用户在场检测、自动唤醒/休眠功能。
无需 MediaPipe，支持 aarch64 (Raspberry Pi 5)。
"""
import logging
import os
import time
from typing import Optional, Callable, List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger("lelamp.edge.face")

# OpenCV 是可选依赖，优雅降级
CV2_AVAILABLE = False
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    logger.warning("OpenCV not available, FaceDetector will run in NoOp mode")


@dataclass
class FaceInfo:
    """人脸信息"""
    bbox: List[int]  # [x, y, width, height]
    confidence: float
    center: tuple  # (x, y) 归一化坐标


class FaceDetector:
    """
    基于 OpenCV Haar 级联的人脸检测服务

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
                0: 正面人脸（默认，适合台灯场景）
                1: 侧面人脸（适合侧面检测）
            min_detection_confidence: 最小检测置信度（映射到 scaleFactor 和 minNeighbors）
            presence_callback: 在场状态变化回调
            presence_threshold_s: 持续检测到人脸的时长阈值（秒）
            absence_threshold_s: 持续未检测到人脸的时长阈值（秒）
        """
        self._noop = not CV2_AVAILABLE

        if not self._noop:
            cascade_file = self._get_cascade_path(model_selection)
            if cascade_file:
                try:
                    self.cascade = cv2.CascadeClassifier(cascade_file)
                    if self.cascade.empty():
                        logger.warning(f"Failed to load cascade from {cascade_file}")
                        self._noop = True
                except Exception as e:
                    logger.error(f"Failed to initialize FaceDetector: {e}")
                    self._noop = True
            else:
                logger.warning("Haar cascade file not found. FaceDetector will run in NoOp mode.")
                self._noop = True

        # Haar 参数：置信度越高 → scaleFactor 越小、minNeighbors 越大
        if min_detection_confidence >= 0.7:
            self._scale_factor = 1.05
            self._min_neighbors = 5
        elif min_detection_confidence >= 0.5:
            self._scale_factor = 1.1
            self._min_neighbors = 4
        else:
            self._scale_factor = 1.2
            self._min_neighbors = 3

        self._min_size = (60, 60)  # 最小人脸尺寸（像素）
        self._smooth_absent_count = 0  # 连续未检测到人脸的帧数
        self._smooth_absent_threshold = 3  # 容忍连续3帧丢失（第4帧才判不在场）

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

        mode = "NoOp" if self._noop else "OpenCV Haar"
        logger.info(f"FaceDetector initialized ({mode} mode)")

    def _get_cascade_path(self, model_selection: int) -> Optional[str]:
        """获取 Haar 级联文件路径"""
        cascade_files = {
            0: "haarcascade_frontalface_default.xml",
            1: "haarcascade_profileface.xml",
        }
        cascade_file = cascade_files.get(model_selection, "haarcascade_frontalface_default.xml")

        # 1. 尝试 OpenCV 内置路径
        try:
            cv2_path = cv2.data.haarcascades + cascade_file
            if os.path.exists(cv2_path):
                return cv2_path
        except Exception:
            pass

        # 2. 尝试项目目录
        project_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "models", cascade_file
        )
        if os.path.exists(project_path):
            return project_path

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
                "presence_duration": float
            }
        """
        if self._noop:
            return self._noop_detect()

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 直方图均衡化，改善光照
            gray = cv2.equalizeHist(gray)

            faces_rects = self.cascade.detectMultiScale(
                gray,
                scaleFactor=self._scale_factor,
                minNeighbors=self._min_neighbors,
                minSize=self._min_size,
            )

            faces: List[FaceInfo] = []
            main_face_center: Optional[tuple] = None
            h, w = frame.shape[:2]

            for i, (x, y, fw, fh) in enumerate(faces_rects):
                # 取人脸区域面积占帧面积的比例作为置信度代理
                face_area = fw * fh
                frame_area = w * h
                confidence = min(face_area / frame_area * 10, 1.0)

                # 归一化中心坐标
                center = (
                    (x + fw / 2) / w,
                    (y + fh / 2) / h,
                )

                faces.append(FaceInfo(
                    bbox=[x, y, fw, fh],
                    confidence=confidence,
                    center=center,
                ))

                if i == 0:
                    main_face_center = center

            if faces:
                self._total_detections += 1
                self._smooth_absent_count = 0  # 检测到人脸，重置计数
            else:
                self._smooth_absent_count += 1

            # 帧平滑：连续 N 帧都没检测到才判为不在场
            presence = len(faces) > 0 or self._smooth_absent_count <= self._smooth_absent_threshold
            presence_duration = self._update_presence_state(presence)

            return {
                "faces": faces,
                "count": len(faces),
                "presence": presence,
                "main_face_center": main_face_center,
                "presence_duration": presence_duration,
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
            "presence_duration": 0.0,
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
            "noop_mode": self._noop,
        }

    def reset_stats(self):
        """重置统计信息"""
        self._total_detections = 0
        self._presence_changes = 0

    def close(self):
        """释放资源"""
        logger.info("FaceDetector closed")
