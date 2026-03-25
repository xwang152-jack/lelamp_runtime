"""
混合视觉推理服务

智能路由：简单任务本地推理，复杂任务云端推理。
"""
import logging
import time
from typing import Optional, Dict, Any, List, Callable
from enum import Enum
from dataclasses import dataclass

from .face_detector import FaceDetector
from .hand_tracker import HandTracker, Gesture
from .object_detector import ObjectDetector

logger = logging.getLogger("lelamp.edge.hybrid")


class QueryComplexity(Enum):
    """查询复杂度"""
    SIMPLE = "simple"      # 本地可处理（如"这是什么"）
    MODERATE = "moderate"  # 本地 + 云端补充
    COMPLEX = "complex"    # 需要云端（如"检查作业"）


@dataclass
class HybridResult:
    """混合推理结果"""
    answer: str
    source: str  # "local" 或 "cloud" 或 "hybrid"
    local_result: Optional[Dict] = None
    cloud_result: Optional[str] = None
    latency_ms: float = 0.0


class HybridVisionService:
    """
    混合视觉推理服务
    
    路由策略：
    - 简单查询（"这是什么"）→ 本地物体检测
    - 复杂查询（"检查作业"）→ 云端 Qwen VL
    - 手势控制 → 本地 MediaPipe
    - 用户在场 → 本地 MediaPipe
    
    使用示例:
        service = HybridVisionService(
            cloud_vision_client=qwen_client,
            gesture_callback=lambda g, ctx: handle_gesture(g),
            presence_callback=lambda p: handle_presence(p)
        )
        
        # 处理视觉问题
        result = await service.answer("这是什么？", frame)
        print(result.answer)
        
        # 检测手势
        gesture_result = service.track_hands(frame)
        
        # 检测用户在场
        face_result = service.detect_faces(frame)
    """
    
    # 简单查询模式（本地可处理）
    SIMPLE_PATTERNS = [
        "这是什么", "这是啥", "有什么", "识别一下",
        "看到了什么", "前面是什么", "那是什么",
        "what is this", "what's this", "what do you see"
    ]
    
    # 复杂查询模式（需要云端）
    COMPLEX_PATTERNS = [
        "检查作业", "批改", "这道题对吗", "做得对不对",
        "详细描述", "分析一下", "比较一下", "评价一下",
        "文字是什么", "读一下", "写的是什么",
        "讲个故事", "编一个", "想象一下",
        "describe in detail", "analyze", "compare"
    ]
    
    def __init__(
        self,
        # 云端视觉客户端（Qwen VL）
        cloud_vision_client=None,
        
        # 功能开关
        enable_face: bool = True,
        enable_hand: bool = True,
        enable_object: bool = True,
        
        # 回调函数
        gesture_callback: Optional[Callable[[Gesture, Dict], None]] = None,
        presence_callback: Optional[Callable[[bool], None]] = None,
        
        # 配置
        prefer_local: bool = True,
        local_confidence_threshold: float = 0.7,
    ):
        """
        初始化混合视觉服务
        
        Args:
            cloud_vision_client: 云端视觉客户端（Qwen3VLClient）
            enable_face: 启用人脸检测
            enable_hand: 启用手势追踪
            enable_object: 启用物体检测
            gesture_callback: 手势识别回调
            presence_callback: 用户在场状态变化回调
            prefer_local: 优先使用本地推理
            local_confidence_threshold: 本地推理的置信度阈值
        """
        self.cloud_vision = cloud_vision_client
        self.prefer_local = prefer_local
        self.local_confidence_threshold = local_confidence_threshold
        
        # 初始化本地服务
        self.face_detector: Optional[FaceDetector] = None
        self.hand_tracker: Optional[HandTracker] = None
        self.object_detector: Optional[ObjectDetector] = None
        
        if enable_face:
            self.face_detector = FaceDetector(
                presence_callback=presence_callback
            )
        
        if enable_hand:
            self.hand_tracker = HandTracker(
                gesture_callback=gesture_callback
            )
        
        if enable_object:
            self.object_detector = ObjectDetector()
        
        # 状态
        self._user_present = False
        self._last_local_result: Optional[Dict] = None
        self._last_frame = None
        
        # 统计
        self._total_queries = 0
        self._local_queries = 0
        self._cloud_queries = 0
        self._hybrid_queries = 0
        
        logger.info(
            f"HybridVisionService initialized "
            f"(face={enable_face}, hand={enable_hand}, object={enable_object})"
        )
    
    # ==================== 核心方法 ====================
    
    def analyze_query(self, question: str) -> QueryComplexity:
        """
        分析查询复杂度
        
        Args:
            question: 用户问题
            
        Returns:
            QueryComplexity 枚举值
        """
        question_lower = question.lower()
        
        # 检查复杂模式
        for pattern in self.COMPLEX_PATTERNS:
            if pattern in question_lower:
                return QueryComplexity.COMPLEX
        
        # 检查简单模式
        for pattern in self.SIMPLE_PATTERNS:
            if pattern in question_lower:
                return QueryComplexity.SIMPLE
        
        # 默认中等复杂度
        return QueryComplexity.MODERATE
    
    async def answer(self, question: str, frame=None) -> HybridResult:
        """
        回答视觉问题
        
        自动路由：
        - 简单问题 → 本地物体检测
        - 复杂问题 → 云端 Qwen VL
        - 中等问题 → 混合推理
        
        Args:
            question: 用户问题
            frame: 图像帧（可选，如果不提供则使用最后一帧）
            
        Returns:
            HybridResult 对象
        """
        start_time = time.time()
        self._total_queries += 1
        
        if frame is not None:
            self._last_frame = frame
        
        complexity = self.analyze_query(question)
        
        if complexity == QueryComplexity.SIMPLE and self.object_detector and self.prefer_local:
            # 本地推理
            result = self._local_answer(question, self._last_frame)
            self._local_queries += 1
            return result
        
        elif complexity == QueryComplexity.COMPLEX or not self.object_detector:
            # 云端推理
            result = await self._cloud_answer(question)
            self._cloud_queries += 1
            return result
        
        else:
            # 混合推理
            result = await self._hybrid_answer(question, self._last_frame)
            self._hybrid_queries += 1
            return result
    
    def _local_answer(self, question: str, frame) -> HybridResult:
        """本地推理回答"""
        start_time = time.time()
        
        if frame is None or self.object_detector is None:
            return HybridResult(
                answer="我没有看到任何东西，请让我看看。",
                source="local",
                latency_ms=(time.time() - start_time) * 1000
            )
        
        # 执行物体检测
        detect_result = self.object_detector.detect(frame)
        self._last_local_result = detect_result
        
        if detect_result["labels"]:
            # 构建回答
            objects_str = detect_result["summary"]
            confidence = detect_result["objects"][0].confidence if detect_result["objects"] else 0
            
            if confidence >= self.local_confidence_threshold:
                answer = f"我看到{objects_str}。需要我详细介绍一下吗？"
            else:
                answer = f"看起来{objects_str}，不过我不太确定。要让我仔细看看吗？"
        else:
            answer = "我没有识别到明显的物体，可能需要换个角度或者让我仔细看看。"
        
        return HybridResult(
            answer=answer,
            source="local",
            local_result=detect_result,
            latency_ms=(time.time() - start_time) * 1000
        )
    
    async def _cloud_answer(self, question: str) -> HybridResult:
        """云端推理回答"""
        start_time = time.time()
        
        if self.cloud_vision is None:
            return HybridResult(
                answer="云端视觉服务未配置，无法回答这个问题。",
                source="cloud",
                latency_ms=(time.time() - start_time) * 1000
            )
        
        try:
            # 调用云端视觉服务
            cloud_result = await self.cloud_vision.answer(question)
            
            return HybridResult(
                answer=cloud_result,
                source="cloud",
                cloud_result=cloud_result,
                latency_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            logger.error(f"Cloud vision error: {e}")
            return HybridResult(
                answer=f"云端服务出了一点问题：{str(e)}",
                source="cloud",
                latency_ms=(time.time() - start_time) * 1000
            )
    
    async def _hybrid_answer(self, question: str, frame) -> HybridResult:
        """混合推理回答"""
        start_time = time.time()
        
        # 先尝试本地推理
        local_result = None
        if self.object_detector and frame is not None:
            local_result = self.object_detector.detect(frame)
            self._last_local_result = local_result
        
        # 如果本地结果足够好，直接返回
        if local_result and local_result["labels"]:
            confidence = local_result["objects"][0].confidence if local_result["objects"] else 0
            if confidence >= self.local_confidence_threshold:
                objects_str = local_result["summary"]
                return HybridResult(
                    answer=f"我看到{objects_str}。需要我详细介绍一下吗？",
                    source="local",
                    local_result=local_result,
                    latency_ms=(time.time() - start_time) * 1000
                )
        
        # 否则调用云端
        if self.cloud_vision:
            try:
                cloud_result = await self.cloud_vision.answer(question)
                
                # 如果有本地结果，补充到回答中
                if local_result and local_result["labels"]:
                    answer = f"本地识别到{local_result['summary']}。{cloud_result}"
                else:
                    answer = cloud_result
                
                return HybridResult(
                    answer=answer,
                    source="hybrid",
                    local_result=local_result,
                    cloud_result=cloud_result,
                    latency_ms=(time.time() - start_time) * 1000
                )
            except Exception as e:
                logger.error(f"Cloud vision error in hybrid mode: {e}")
                # 降级到本地结果
                if local_result and local_result["labels"]:
                    return HybridResult(
                        answer=f"我看到{local_result['summary']}（云端服务暂时不可用）。",
                        source="local",
                        local_result=local_result,
                        latency_ms=(time.time() - start_time) * 1000
                    )
                return HybridResult(
                    answer=f"我看不太清楚，云端服务也出问题了：{str(e)}",
                    source="hybrid",
                    latency_ms=(time.time() - start_time) * 1000
                )
        
        # 没有云端服务，使用本地结果
        if local_result and local_result["labels"]:
            return HybridResult(
                answer=f"我看到{local_result['summary']}。",
                source="local",
                local_result=local_result,
                latency_ms=(time.time() - start_time) * 1000
            )
        
        return HybridResult(
            answer="我没有识别到任何东西。",
            source="local",
            local_result=local_result,
            latency_ms=(time.time() - start_time) * 1000
        )
    
    # ==================== 人脸检测 ====================
    
    def detect_faces(self, frame) -> Dict[str, Any]:
        """
        检测人脸
        
        Args:
            frame: 图像帧
            
        Returns:
            人脸检测结果
        """
        if self.face_detector is None:
            return {"faces": [], "count": 0, "presence": False}
        
        result = self.face_detector.detect(frame)
        self._user_present = result["presence"]
        return result
    
    @property
    def is_user_present(self) -> bool:
        """用户是否在场"""
        return self._user_present
    
    # ==================== 手势追踪 ====================
    
    def track_hands(self, frame) -> Dict[str, Any]:
        """
        追踪手势
        
        Args:
            frame: 图像帧
            
        Returns:
            手势追踪结果
        """
        if self.hand_tracker is None:
            return {"hands": [], "gestures": [], "count": 0}
        
        return self.hand_tracker.track(frame)
    
    # ==================== 物体检测 ====================
    
    def detect_objects(self, frame) -> Dict[str, Any]:
        """
        检测物体
        
        Args:
            frame: 图像帧
            
        Returns:
            物体检测结果
        """
        if self.object_detector is None:
            return {"objects": [], "labels": [], "summary": "物体检测未启用", "count": 0}
        
        result = self.object_detector.detect(frame)
        self._last_local_result = result
        return result
    
    # ==================== 统计信息 ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_queries": self._total_queries,
            "local_queries": self._local_queries,
            "cloud_queries": self._cloud_queries,
            "hybrid_queries": self._hybrid_queries,
            "user_present": self._user_present,
            "services": {
                "face_detector": self.face_detector is not None,
                "hand_tracker": self.hand_tracker is not None,
                "object_detector": self.object_detector is not None,
                "cloud_vision": self.cloud_vision is not None,
            }
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self._total_queries = 0
        self._local_queries = 0
        self._cloud_queries = 0
        self._hybrid_queries = 0
        
        if self.face_detector:
            self.face_detector.reset_stats()
        if self.hand_tracker:
            self.hand_tracker.reset_stats()
        if self.object_detector:
            self.object_detector.reset_stats()
    
    # ==================== 生命周期 ====================
    
    def close(self):
        """释放所有资源"""
        if self.face_detector:
            self.face_detector.close()
        if self.hand_tracker:
            self.hand_tracker.close()
        if self.object_detector:
            self.object_detector.close()
        
        logger.info("HybridVisionService closed")