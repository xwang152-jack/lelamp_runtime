"""
边缘推理模块测试
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# 测试 MediaPipe 是否可用
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


@pytest.fixture
def sample_frame():
    """创建测试用的图像帧"""
    # 创建一个 640x480 的黑色图像
    return np.zeros((480, 640, 3), dtype=np.uint8)


class TestFaceDetector:
    """人脸检测器测试"""
    
    def test_init_no_mediapipe(self):
        """测试 MediaPipe 不可用时的初始化"""
        with patch('lelamp.edge.face_detector.MEDIAPIPE_AVAILABLE', False):
            from lelamp.edge.face_detector import FaceDetector
            detector = FaceDetector()
            
            # 应该是 NoOp 模式
            assert detector._noop is True
            assert detector.get_stats()["noop_mode"] is True
    
    def test_noop_detect(self):
        """测试 NoOp 模式的检测"""
        with patch('lelamp.edge.face_detector.MEDIAPIPE_AVAILABLE', False):
            from lelamp.edge.face_detector import FaceDetector
            detector = FaceDetector()
            
            result = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
            
            assert result["count"] == 0
            assert result["presence"] is False
            assert result["faces"] == []
    
    def test_presence_callback(self):
        """测试在场状态回调"""
        with patch('lelamp.edge.face_detector.MEDIAPIPE_AVAILABLE', False):
            from lelamp.edge.face_detector import FaceDetector
            
            callback_mock = Mock()
            detector = FaceDetector(
                presence_callback=callback_mock,
                presence_threshold_s=0.0,  # 立即触发
                absence_threshold_s=0.0
            )
            
            # NoOp 模式下回调不会被触发
            assert callback_mock.call_count == 0
    
    def test_get_stats(self):
        """测试统计信息"""
        with patch('lelamp.edge.face_detector.MEDIAPIPE_AVAILABLE', False):
            from lelamp.edge.face_detector import FaceDetector
            detector = FaceDetector()
            
            stats = detector.get_stats()
            
            assert "total_detections" in stats
            assert "presence_changes" in stats
            assert "current_presence" in stats
            assert "noop_mode" in stats


class TestHandTracker:
    """手势追踪器测试"""
    
    def test_init_no_mediapipe(self):
        """测试 MediaPipe 不可用时的初始化"""
        with patch('lelamp.edge.hand_tracker.MEDIAPIPE_AVAILABLE', False):
            from lelamp.edge.hand_tracker import HandTracker
            tracker = HandTracker()
            
            assert tracker._noop is True
            assert tracker.get_stats()["noop_mode"] is True
    
    def test_noop_track(self):
        """测试 NoOp 模式的追踪"""
        with patch('lelamp.edge.hand_tracker.MEDIAPIPE_AVAILABLE', False):
            from lelamp.edge.hand_tracker import HandTracker
            tracker = HandTracker()
            
            result = tracker.track(np.zeros((480, 640, 3), dtype=np.uint8))
            
            assert result["count"] == 0
            assert result["gestures"] == []
            assert result["hands"] == []
    
    def test_gesture_enum(self):
        """测试手势枚举"""
        from lelamp.edge.hand_tracker import Gesture
        
        assert Gesture.OPEN.value == "open"
        assert Gesture.FIST.value == "fist"
        assert Gesture.POINT.value == "point"
        assert Gesture.PEACE.value == "peace"
        assert Gesture.THUMBS_UP.value == "thumbs_up"
        assert Gesture.WAVE.value == "wave"


class TestObjectDetector:
    """物体检测器测试"""
    
    def test_init_no_mediapipe(self):
        """测试 MediaPipe 不可用时的初始化"""
        with patch('lelamp.edge.object_detector.MEDIAPIPE_AVAILABLE', False):
            from lelamp.edge.object_detector import ObjectDetector
            detector = ObjectDetector()
            
            assert detector._noop is True
    
    def test_noop_detect(self):
        """测试 NoOp 模式的检测"""
        with patch('lelamp.edge.object_detector.MEDIAPIPE_AVAILABLE', False):
            from lelamp.edge.object_detector import ObjectDetector
            detector = ObjectDetector()
            
            result = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
            
            assert result["count"] == 0
            assert result["labels"] == []
            assert "未启用" in result["summary"]
    
    def test_coco_labels(self):
        """测试 COCO 标签映射"""
        from lelamp.edge.object_detector import ObjectDetector
        
        # 检查一些常用标签
        assert ObjectDetector.COCO_LABELS_ZH[48] == "苹果"
        assert ObjectDetector.COCO_LABELS_ZH[68] == "手机"
        assert ObjectDetector.COCO_LABELS_ZH[74] == "书"
    
    def test_get_category(self):
        """测试获取物体分类"""
        with patch('lelamp.edge.object_detector.MEDIAPIPE_AVAILABLE', False):
            from lelamp.edge.object_detector import ObjectDetector
            detector = ObjectDetector()
            
            assert detector.get_category("苹果") == "水果"
            assert detector.get_category("手机") == "电子产品"
            assert detector.get_category("未知物品") == "物品"


class TestHybridVisionService:
    """混合视觉服务测试"""
    
    def test_init(self):
        """测试初始化"""
        from lelamp.edge.hybrid_vision import HybridVisionService
        
        service = HybridVisionService(
            enable_face=True,
            enable_hand=True,
            enable_object=True,
        )
        
        assert service.face_detector is not None
        assert service.hand_tracker is not None
        assert service.object_detector is not None
    
    def test_init_partial(self):
        """测试部分功能初始化"""
        from lelamp.edge.hybrid_vision import HybridVisionService
        
        service = HybridVisionService(
            enable_face=True,
            enable_hand=False,
            enable_object=False,
        )
        
        assert service.face_detector is not None
        assert service.hand_tracker is None
        assert service.object_detector is None
    
    def test_analyze_query(self):
        """测试查询复杂度分析"""
        from lelamp.edge.hybrid_vision import HybridVisionService, QueryComplexity
        
        service = HybridVisionService(enable_face=False, enable_hand=False, enable_object=False)
        
        # 简单查询
        assert service.analyze_query("这是什么") == QueryComplexity.SIMPLE
        assert service.analyze_query("这是啥") == QueryComplexity.SIMPLE
        assert service.analyze_query("what is this") == QueryComplexity.SIMPLE
        
        # 复杂查询
        assert service.analyze_query("检查作业") == QueryComplexity.COMPLEX
        assert service.analyze_query("详细描述一下") == QueryComplexity.COMPLEX
        assert service.analyze_query("这道题对吗") == QueryComplexity.COMPLEX
        
        # 中等查询
        assert service.analyze_query("你好") == QueryComplexity.MODERATE
    
    @pytest.mark.asyncio
    async def test_local_answer_no_frame(self):
        """测试无图像帧时的本地回答"""
        from lelamp.edge.hybrid_vision import HybridVisionService
        
        service = HybridVisionService(
            enable_face=False,
            enable_hand=False,
            enable_object=True,
        )
        
        result = await service.answer("这是什么", frame=None)
        
        assert "没有看到" in result.answer
        assert result.source == "local"
    
    @pytest.mark.asyncio
    async def test_cloud_answer_no_client(self):
        """测试无云端客户端时的回答"""
        from lelamp.edge.hybrid_vision import HybridVisionService
        
        service = HybridVisionService(
            cloud_vision_client=None,
            enable_face=False,
            enable_hand=False,
            enable_object=False,
        )
        
        result = await service.answer("检查作业", frame=None)
        
        assert "未配置" in result.answer or "出了一点问题" in result.answer
    
    def test_detect_faces(self):
        """测试人脸检测接口"""
        from lelamp.edge.hybrid_vision import HybridVisionService
        
        service = HybridVisionService(enable_face=True, enable_hand=False, enable_object=False)
        
        result = service.detect_faces(np.zeros((480, 640, 3), dtype=np.uint8))
        
        assert "faces" in result
        assert "count" in result
        assert "presence" in result
    
    def test_track_hands(self):
        """测试手势追踪接口"""
        from lelamp.edge.hybrid_vision import HybridVisionService
        
        service = HybridVisionService(enable_face=False, enable_hand=True, enable_object=False)
        
        result = service.track_hands(np.zeros((480, 640, 3), dtype=np.uint8))
        
        assert "hands" in result
        assert "gestures" in result
        assert "count" in result
    
    def test_detect_objects(self):
        """测试物体检测接口"""
        from lelamp.edge.hybrid_vision import HybridVisionService
        
        service = HybridVisionService(enable_face=False, enable_hand=False, enable_object=True)
        
        result = service.detect_objects(np.zeros((480, 640, 3), dtype=np.uint8))
        
        assert "objects" in result
        assert "labels" in result
        assert "summary" in result
    
    def test_get_stats(self):
        """测试统计信息"""
        from lelamp.edge.hybrid_vision import HybridVisionService
        
        service = HybridVisionService()
        stats = service.get_stats()
        
        assert "total_queries" in stats
        assert "local_queries" in stats
        assert "cloud_queries" in stats
        assert "services" in stats
    
    def test_close(self):
        """测试资源释放"""
        from lelamp.edge.hybrid_vision import HybridVisionService
        
        service = HybridVisionService()
        # 不应该抛出异常
        service.close()


class TestIntegration:
    """集成测试"""
    
    @pytest.mark.skipif(not MEDIAPIPE_AVAILABLE, reason="MediaPipe not available")
    def test_full_pipeline(self, sample_frame):
        """测试完整流水线（需要 MediaPipe）"""
        from lelamp.edge.hybrid_vision import HybridVisionService
        
        service = HybridVisionService(
            enable_face=True,
            enable_hand=True,
            enable_object=False,  # 需要模型文件
        )
        
        # 人脸检测
        face_result = service.detect_faces(sample_frame)
        assert isinstance(face_result["count"], int)
        
        # 手势追踪
        hand_result = service.track_hands(sample_frame)
        assert isinstance(hand_result["count"], int)
        
        # 统计
        stats = service.get_stats()
        assert stats["services"]["face_detector"] is True
        assert stats["services"]["hand_tracker"] is True
        
        service.close()