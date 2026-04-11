"""
手势追踪服务（NoOp 模式）

手势识别功能已移除。保留接口兼容性，所有调用返回空结果。
"""
import logging
from typing import Optional, Callable, List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("lelamp.edge.hand")


class Gesture(Enum):
    """手势类型"""
    OPEN = "open"
    FIST = "fist"
    POINT = "point"
    PEACE = "peace"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    OK = "ok"
    WAVE = "wave"
    UNKNOWN = "unknown"


@dataclass
class HandInfo:
    """手部信息"""
    landmarks: List[Tuple[float, float, float]]
    handedness: str
    gesture: Gesture
    confidence: float


class HandTracker:
    """
    手势追踪服务（NoOp 模式）

    手势识别已移除，所有方法返回空结果。
    保留类和接口以确保调用方兼容。
    """

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
        logger.info("HandTracker initialized (NoOp mode - gesture recognition removed)")

    def track(self, frame) -> Dict[str, Any]:
        """NoOp：返回空结果"""
        return {"hands": [], "gestures": [], "count": 0}

    def get_stats(self) -> Dict[str, Any]:
        return {"total_tracks": 0, "gesture_counts": {}, "noop_mode": True}

    def reset_stats(self):
        pass

    def close(self):
        logger.info("HandTracker closed")
