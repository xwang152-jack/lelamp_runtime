"""
MediaPipe 手势追踪服务

用于手势控制台灯功能。
基于 MediaPipe Solutions API。
"""
import logging
import time
from typing import Optional, Callable, List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("lelamp.edge.hand")

# MediaPipe 是可选依赖，优雅降级
MEDIAPIPE_AVAILABLE = False
try:
    import mediapipe as mp
    import cv2
    import numpy as np
    if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands'):
        MEDIAPIPE_AVAILABLE = True
    else:
        logger.warning("MediaPipe installed but 'solutions' API not available. "
                      "HandTracker will run in NoOp mode.")
except ImportError:
    logger.warning("MediaPipe not available, HandTracker will run in NoOp mode")


class Gesture(Enum):
    """手势类型"""
    OPEN = "open"           # 张开手掌
    FIST = "fist"           # 握拳
    POINT = "point"         # 指向 (食指)
    PEACE = "peace"         # 耶 ✌️
    THUMBS_UP = "thumbs_up" # 点赞 👍
    THUMBS_DOWN = "thumbs_down"  # 踩 👎
    OK = "ok"               # OK 手势
    WAVE = "wave"           # 挥手
    UNKNOWN = "unknown"     # 未知


@dataclass
class HandInfo:
    """手部信息"""
    landmarks: List[Tuple[float, float, float]]  # 21个关键点 (x, y, z)
    handedness: str  # "Left" 或 "Right"
    gesture: Gesture
    confidence: float


class HandTracker:
    """
    基于 MediaPipe Solutions API 的手势追踪服务

    支持的手势：
    - 👍 点赞 → 触发 nod 动作
    - 👎 踩 → 触发 shake 动作
    - ✌️ 耶 → 触发 excited 动作
    - 👋 挥手 → 开关灯
    - ✊ 握拳 → 静音/取消静音
    - 👆 指向 → 台灯看向指定方向
    """

    # MediaPipe 手部关键点索引
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        gesture_callback: Optional[Callable[[Gesture, Dict], None]] = None,
        wave_threshold: float = 0.15,
        wave_frames: int = 10,
        gesture_cooldown_s: float = 1.0,
    ):
        self._noop = not MEDIAPIPE_AVAILABLE

        if not self._noop:
            try:
                self.detector = mp.solutions.hands.Hands(
                    static_image_mode=True,
                    max_num_hands=max_num_hands,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
            except Exception as e:
                logger.error(f"Failed to initialize HandTracker: {e}")
                self._noop = True

        self.gesture_callback = gesture_callback
        self._wave_threshold = wave_threshold
        self._wave_frames = wave_frames
        self._gesture_cooldown_s = gesture_cooldown_s

        # 挥手检测状态
        self._hand_positions_history: List[Tuple[float, float]] = []

        # 手势冷却
        self._last_gesture_time: Dict[Gesture, float] = {}

        # 统计信息
        self._total_tracks = 0
        self._gesture_counts: Dict[Gesture, int] = {}

        mode = "NoOp" if self._noop else "MediaPipe"
        logger.info(f"HandTracker initialized ({mode} mode)")

    def track(self, frame) -> Dict[str, Any]:
        """
        追踪手势

        Args:
            frame: BGR 格式的图像帧

        Returns:
            {
                "hands": [HandInfo, ...],
                "gestures": [Gesture, ...],
                "count": int
            }
        """
        if self._noop:
            return self._noop_track()

        try:
            rgb_frame = frame
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = self.detector.process(rgb_frame)

            hands_data: List[HandInfo] = []
            gestures: List[Gesture] = []

            if results.multi_hand_landmarks:
                self._total_tracks += 1

                for i, landmarks in enumerate(results.multi_hand_landmarks):
                    landmark_list: List[Tuple[float, float, float]] = []
                    for lm in landmarks.landmark:
                        landmark_list.append((lm.x, lm.y, lm.z))

                    handedness_label = "Right"
                    confidence = 0.5
                    if results.multi_handedness and i < len(results.multi_handedness):
                        handedness_entry = results.multi_handedness[i]
                        if handedness_entry and handedness_entry.classification:
                            handedness_label = handedness_entry.classification[0].label
                            confidence = handedness_entry.classification[0].score

                    gesture = self._recognize_gesture(landmark_list, handedness_label)

                    if gesture != Gesture.UNKNOWN:
                        self._gesture_counts[gesture] = self._gesture_counts.get(gesture, 0) + 1

                    hands_data.append(HandInfo(
                        landmarks=landmark_list,
                        handedness=handedness_label,
                        gesture=gesture,
                        confidence=confidence
                    ))

                    if gesture not in gestures:
                        gestures.append(gesture)

                    if len(landmark_list) > 0:
                        self._detect_wave(landmark_list[self.WRIST][:2])

            self._trigger_gesture_callbacks(gestures, {"hands": hands_data})

            return {
                "hands": hands_data,
                "gestures": gestures,
                "count": len(hands_data)
            }
        except Exception as e:
            logger.error(f"Hand tracking error: {e}")
            return self._noop_track()

    def _recognize_gesture(self, landmarks: List[Tuple[float, float, float]], handedness: str) -> Gesture:
        """识别手势"""
        thumb_tip = landmarks[self.THUMB_TIP]
        thumb_ip = landmarks[self.THUMB_IP]
        thumb_mcp = landmarks[self.THUMB_MCP]

        index_tip = landmarks[self.INDEX_TIP]
        index_pip = landmarks[self.INDEX_PIP]
        index_mcp = landmarks[self.INDEX_MCP]

        middle_tip = landmarks[self.MIDDLE_TIP]
        middle_pip = landmarks[self.MIDDLE_PIP]

        ring_tip = landmarks[self.RING_TIP]
        ring_pip = landmarks[self.RING_PIP]

        pinky_tip = landmarks[self.PINKY_TIP]
        pinky_pip = landmarks[self.PINKY_PIP]

        wrist = landmarks[self.WRIST]

        if handedness == "Right":
            thumb_up = thumb_tip[0] < thumb_ip[0]
        else:
            thumb_up = thumb_tip[0] > thumb_ip[0]

        index_up = index_tip[1] < index_pip[1]
        middle_up = middle_tip[1] < middle_pip[1]
        ring_up = ring_tip[1] < ring_pip[1]
        pinky_up = pinky_tip[1] < pinky_pip[1]

        thumb_index_dist = self._distance(thumb_tip, index_tip)
        ok_gesture = thumb_index_dist < 0.05

        fingers_up = [thumb_up, index_up, middle_up, ring_up, pinky_up]
        count = sum(fingers_up)

        if ok_gesture and middle_up and ring_up and pinky_up:
            return Gesture.OK

        if count <= 1 and not index_up and not middle_up and not ring_up and not pinky_up:
            return Gesture.FIST

        if count >= 4:
            return Gesture.OPEN

        if index_up and not middle_up and not ring_up and not pinky_up:
            return Gesture.POINT

        if index_up and middle_up and not ring_up and not pinky_up:
            return Gesture.PEACE

        if thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
            if thumb_tip[1] < thumb_mcp[1]:
                return Gesture.THUMBS_UP
            else:
                return Gesture.THUMBS_DOWN

        return Gesture.UNKNOWN

    def _distance(self, p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
        """计算两点间的欧氏距离"""
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2) ** 0.5

    def _detect_wave(self, wrist_pos: Tuple[float, float]):
        """检测挥手动作"""
        self._hand_positions_history.append(wrist_pos)

        if len(self._hand_positions_history) > self._wave_frames:
            self._hand_positions_history.pop(0)

        if len(self._hand_positions_history) >= self._wave_frames:
            x_positions = [p[0] for p in self._hand_positions_history]
            x_range = max(x_positions) - min(x_positions)

            if x_range > self._wave_threshold:
                self._trigger_callback(Gesture.WAVE, {})
                self._hand_positions_history.clear()

    def _trigger_gesture_callbacks(self, gestures: List[Gesture], context: Dict):
        """触发手势回调"""
        now = time.time()

        for gesture in gestures:
            if gesture in [Gesture.OPEN, Gesture.UNKNOWN]:
                continue

            last_time = self._last_gesture_time.get(gesture, 0)
            if now - last_time < self._gesture_cooldown_s:
                continue

            self._last_gesture_time[gesture] = now
            self._trigger_callback(gesture, context)

    def _trigger_callback(self, gesture: Gesture, context: Dict):
        """触发单个回调"""
        if self.gesture_callback:
            try:
                self.gesture_callback(gesture, context)
            except Exception as e:
                logger.error(f"Gesture callback error: {e}")

    def _noop_track(self) -> Dict[str, Any]:
        """NoOp 模式的默认返回"""
        return {
            "hands": [],
            "gestures": [],
            "count": 0
        }

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_tracks": self._total_tracks,
            "gesture_counts": {g.value: c for g, c in self._gesture_counts.items()},
            "noop_mode": self._noop
        }

    def reset_stats(self):
        """重置统计信息"""
        self._total_tracks = 0
        self._gesture_counts = {}

    def close(self):
        """释放资源"""
        if not self._noop and hasattr(self, 'detector'):
            self.detector.close()
            logger.info("HandTracker closed")
