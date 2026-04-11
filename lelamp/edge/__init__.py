"""
LeLamp 边缘推理模块

本地 AI 推理功能：
- Face Detection: 基于 OpenCV Haar 级联的用户在场检测（支持 aarch64）
- Hand Tracking: 已移除（保留 NoOp 接口兼容）
- Object Detection: 已移除（保留 NoOp 接口兼容，物体识别由云端视觉承担）
- Hybrid Vision: 混合推理路由（本地人脸 + 云端 Qwen VL）
"""

from .face_detector import FaceDetector
from .hand_tracker import HandTracker, Gesture
from .object_detector import ObjectDetector
from .hybrid_vision import HybridVisionService, QueryComplexity

__all__ = [
    "FaceDetector",
    "HandTracker",
    "ObjectDetector",
    "HybridVisionService",
    "QueryComplexity",
]
