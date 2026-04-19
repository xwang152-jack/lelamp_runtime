"""
MediaPipe 人脸检测服务

用于用户在场检测、自动唤醒/休眠功能。
优先使用 MediaPipe Solutions API，不可用时回退到 OpenCV Haar 级联。
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
    if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_detection'):
        MEDIAPIPE_AVAILABLE = True
    else:
        logger.warning("MediaPipe installed but 'solutions' API not available. "
                      "FaceDetector will use OpenCV Haar fallback.")
except ImportError:
    logger.warning("MediaPipe not available, FaceDetector will use OpenCV Haar fallback.")

# OpenCV 是可选依赖，二级 fallback
CV2_AVAILABLE = False
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    pass


@dataclass
class FaceInfo:
    """人脸信息"""
    bbox: List[int]  # [x, y, width, height]
    confidence: float
    center: tuple  # (x, y) 归一化坐标


class FaceDetector:
    """
    人脸检测服务

    优先级：MediaPipe Solutions API > OpenCV Haar > NoOp

    用途：
    - 用户在场检测 → 自动唤醒/休眠
    - 多人检测 → 区分主用户
    - 视线追踪 → 台灯跟随
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
                0: 短距离模型（适合 2 米内，MediaPipe）
                1: 远距离模型（适合 5 米内，MediaPipe）
            min_detection_confidence: 最小检测置信度 (0.0-1.0)
            presence_callback: 在场状态变化回调
            presence_threshold_s: 持续检测到人脸的时长阈值（秒）
            absence_threshold_s: 持续未检测到人脸的时长阈值（秒）
        """
        self._use_mediapipe = False
        self._use_haar = False
        self._noop = True

        # 尝试 MediaPipe
        if MEDIAPIPE_AVAILABLE:
            try:
                self.detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=model_selection,
                    min_detection_confidence=min_detection_confidence,
                )
                self._use_mediapipe = True
                self._noop = False
            except Exception as e:
                logger.warning(f"MediaPipe FaceDetector init failed: {e}, falling back to OpenCV Haar")

        # 回退到 OpenCV Haar
        if self._noop and CV2_AVAILABLE:
            cascade_file = self._get_cascade_path(model_selection)
            if cascade_file:
                try:
                    self.cascade = cv2.CascadeClassifier(cascade_file)
                    if not self.cascade.empty():
                        self._use_haar = True
                        self._noop = False
                        if min_detection_confidence >= 0.7:
                            self._scale_factor = 1.05
                            self._min_neighbors = 5
                        elif min_detection_confidence >= 0.5:
                            self._scale_factor = 1.1
                            self._min_neighbors = 4
                        else:
                            self._scale_factor = 1.2
                            self._min_neighbors = 3
                        self._min_size = (60, 60)
                except Exception as e:
                    logger.warning(f"OpenCV Haar init failed: {e}")

        # 帧平滑防抖（Haar 模式使用）
        self._smooth_absent_count = 0
        self._smooth_absent_threshold = 3

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

        if self._use_mediapipe:
            logger.info("FaceDetector initialized (MediaPipe mode)")
        elif self._use_haar:
            logger.info("FaceDetector initialized (OpenCV Haar mode)")
        else:
            logger.info("FaceDetector initialized (NoOp mode)")

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

        if self._use_mediapipe:
            return self._detect_mediapipe(frame)
        else:
            return self._detect_haar(frame)

    def _detect_mediapipe(self, frame) -> Dict[str, Any]:
        """MediaPipe 人脸检测"""
        try:
            rgb_frame = frame
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = self.detector.process(rgb_frame)

            faces: List[FaceInfo] = []
            main_face_center: Optional[tuple] = None

            if results.detections:
                self._total_detections += 1
                h, w = frame.shape[:2]

                for i, detection in enumerate(results.detections):
                    bbox = detection.location_data.relative_bounding_box
                    face_bbox = [
                        int(bbox.xmin * w),
                        int(bbox.ymin * h),
                        int(bbox.width * w),
                        int(bbox.height * h),
                    ]

                    categories = detection.location_data.relative_keypoints
                    confidence = categories[0].score if categories else detection.score[0]

                    center = (
                        bbox.xmin + bbox.width / 2,
                        bbox.ymin + bbox.height / 2,
                    )

                    faces.append(FaceInfo(
                        bbox=face_bbox,
                        confidence=confidence,
                        center=center
                    ))

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
            logger.error(f"MediaPipe face detection error: {e}")
            return self._noop_detect()

    def _get_cascade_path(self, model_selection: int) -> Optional[str]:
        """获取 Haar 级联文件路径"""
        cascade_files = {
            0: "haarcascade_frontalface_default.xml",
            1: "haarcascade_profileface.xml",
        }
        cascade_file = cascade_files.get(model_selection, "haarcascade_frontalface_default.xml")

        try:
            cv2_path = cv2.data.haarcascades + cascade_file
            if os.path.exists(cv2_path):
                return cv2_path
        except Exception:
            pass

        project_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", cascade_file)
        if os.path.exists(project_path):
            return project_path

        return None

    def _detect_haar(self, frame) -> Dict[str, Any]:
        """OpenCV Haar 人脸检测（带帧平滑防抖）"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
                face_area = fw * fh
                frame_area = w * h
                confidence = min(face_area / frame_area * 10, 1.0)

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
                self._smooth_absent_count = 0
            else:
                self._smooth_absent_count += 1

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
            logger.error(f"Haar face detection error: {e}")
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
        if self._use_mediapipe and hasattr(self, 'detector'):
            self.detector.close()
        logger.info("FaceDetector closed")
