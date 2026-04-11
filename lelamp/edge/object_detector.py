"""
物体检测服务（NoOp 模式）

本地物体检测功能已移除。保留接口兼容性，所有调用返回空结果。
物体检测由云端视觉（Qwen VL）承担。
"""
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger("lelamp.edge.object")


@dataclass
class DetectedObject:
    """检测到的物体"""
    label: str
    label_en: str
    confidence: float
    bbox: List[int]
    category_id: int


class ObjectDetector:
    """
    物体检测服务（NoOp 模式）

    本地物体检测已移除，物体识别由云端视觉（Qwen VL）承担。
    保留类和接口以确保调用方兼容。
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        max_results: int = 5,
        score_threshold: float = 0.5,
    ):
        self._noop = True
        logger.info("ObjectDetector initialized (NoOp mode - using cloud vision)")

    def detect(self, frame) -> Dict[str, Any]:
        """NoOp：返回空结果"""
        return {
            "objects": [],
            "labels": [],
            "summary": "物体检测服务未启用",
            "count": 0,
        }

    def get_category(self, label: str) -> str:
        return "物品"

    def get_stats(self) -> Dict[str, Any]:
        return {"total_detections": 0, "object_counts": {}, "noop_mode": True}

    def reset_stats(self):
        pass

    def close(self):
        logger.info("ObjectDetector closed")
