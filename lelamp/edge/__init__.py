"""
LeLamp 边缘推理模块

基于 MediaPipe 的本地 AI 推理功能：
- Face Detection: 用户在场检测
- Hand Tracking: 手势控制
- Object Detection: 本地物体识别
- Hybrid Vision: 混合推理路由
"""

from .face_detector import FaceDetector
from .hand_tracker import HandTracker
from .object_detector import ObjectDetector
from .hybrid_vision import HybridVisionService, QueryComplexity

__all__ = [
    "FaceDetector",
    "HandTracker", 
    "ObjectDetector",
    "HybridVisionService",
    "QueryComplexity",
]