"""
测试代理工具模块
"""
import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from lelamp.agent.states import (
    ConversationState,
    StateManager,
    StateColors,
)


@pytest.mark.unit
class TestConversationState:
    """测试对话状态枚举"""

    def test_state_values(self):
        """测试状态值"""
        assert ConversationState.IDLE == "idle"
        assert ConversationState.LISTENING == "listening"
        assert ConversationState.THINKING == "thinking"
        assert ConversationState.SPEAKING == "speaking"


@pytest.mark.unit
class TestStateColors:
    """测试状态颜色映射"""

    def test_state_colors(self):
        """测试状态颜色"""
        colors = StateColors
        assert colors.IDLE == (255, 244, 229)
        assert colors.LISTENING == (0, 140, 255)
        assert colors.THINKING == (180, 0, 255)


@pytest.mark.unit
class TestStateManager:
    """测试状态管理器"""

    def test_init(self):
        """测试初始化"""
        manager = StateManager()
        assert manager.current_state == ConversationState.IDLE

    def test_set_state(self):
        """测试设置状态"""
        manager = StateManager()
        manager.set_state(ConversationState.LISTENING)
        assert manager.current_state == ConversationState.LISTENING

    def test_can_execute_motion(self):
        """测试是否允许执行电机动作"""
        manager = StateManager(motion_cooldown_s=1.0)

        # 初始状态应该允许
        assert manager.can_execute_motion() is True

        # 记录动作后应该不允许
        manager.record_motion()
        assert manager.can_execute_motion() is False

        # 等待冷却后应该允许
        time.sleep(1.1)
        assert manager.can_execute_motion() is True

    def test_suppress_motion_after_light(self):
        """测试灯光后抑制动作"""
        manager = StateManager(suppress_motion_after_light_s=0.5)

        # 设置灯光覆盖应该抑制动作
        manager.set_light_override(1.0)
        assert manager.can_execute_motion() is False

        # 等待抑制期后应该允许
        time.sleep(0.6)
        assert manager.can_execute_motion() is True

    def test_light_override(self):
        """测试灯光覆盖"""
        manager = StateManager()

        assert manager.is_light_overridden() is False

        manager.set_light_override(0.1)
        assert manager.is_light_overridden() is True

        time.sleep(0.15)
        assert manager.is_light_overridden() is False

    def test_clear_light_override(self):
        """测试清除灯光覆盖"""
        manager = StateManager()

        manager.set_light_override(10.0)
        assert manager.is_light_overridden() is True

        manager.clear_light_override()
        assert manager.is_light_overridden() is False

    def test_save_and_restore_light_override(self):
        """测试保存和恢复灯光覆盖"""
        manager = StateManager()

        # 初始状态没有覆盖
        saved = manager.save_and_set_light_override(1.0)
        assert saved is None
        assert manager.is_light_overridden() is True

        # 保存当前覆盖状态
        saved = manager.save_and_set_light_override(2.0)
        assert saved is not None
        assert manager.is_light_overridden() is True

        # 恢复之前的覆盖状态
        manager.restore_light_override(saved)
        assert manager.is_light_overridden() is True


@pytest.mark.unit
class TestMotorTools:
    """测试电机工具"""

    def test_safe_joint_ranges(self):
        """测试关节安全范围"""
        from lelamp.config import SAFE_JOINT_RANGES

        assert "base_yaw" in SAFE_JOINT_RANGES
        assert "base_pitch" in SAFE_JOINT_RANGES
        assert "elbow_pitch" in SAFE_JOINT_RANGES
        assert "wrist_roll" in SAFE_JOINT_RANGES
        assert "wrist_pitch" in SAFE_JOINT_RANGES

    def test_joint_ranges_values(self):
        """测试安全范围值"""
        from lelamp.config import SAFE_JOINT_RANGES

        assert SAFE_JOINT_RANGES["base_yaw"] == (-180, 180)
        assert SAFE_JOINT_RANGES["base_pitch"] == (-90, 90)
        assert SAFE_JOINT_RANGES["elbow_pitch"] == (-150, 150)
        assert SAFE_JOINT_RANGES["wrist_roll"] == (-180, 180)
        assert SAFE_JOINT_RANGES["wrist_pitch"] == (-90, 90)


@pytest.mark.unit
class TestRGBTools:
    """测试RGB工具"""

    def test_rgb_config(self):
        """测试RGB配置"""
        from lelamp.config import RGBConfig, load_rgb_config

        config = load_rgb_config()
        assert config.led_count == 64
        assert config.led_pin == 12
        assert config.led_brightness == 25


@pytest.mark.unit
class TestVisionConfig:
    """测试视觉配置"""

    def test_vision_config(self):
        """测试视觉配置"""
        from lelamp.config import VisionConfig, load_vision_config

        config = load_vision_config()
        assert config.enabled is True
        assert config.width == 1024
        assert config.height == 768


@pytest.mark.unit
class TestEdgeVision:
    """测试边缘视觉"""

    def test_face_detector_no_mediapipe(self):
        """测试人脸检测器初始化"""
        from lelamp.edge.face_detector import FaceDetector

        detector = FaceDetector()
        assert detector is not None

        # detect 方法返回字典
        result = detector.detect(b"fake_frame")
        assert isinstance(result, dict)
        assert "faces" in result

    def test_face_detector_get_stats(self):
        """测试获取人脸检测统计"""
        from lelamp.edge.face_detector import FaceDetector

        detector = FaceDetector()
        stats = detector.get_stats()

        assert isinstance(stats, dict)
        assert "noop_mode" in stats

    def test_hand_tracker_no_mediapipe(self):
        """测试手势追踪器初始化"""
        from lelamp.edge.hand_tracker import HandTracker

        tracker = HandTracker()
        assert tracker is not None

        # track 方法返回字典
        result = tracker.track(b"fake_frame")
        assert isinstance(result, dict)
        assert "hands" in result

    def test_gesture_enum(self):
        """测试手势枚举"""
        from lelamp.edge.hand_tracker import Gesture

        # 测试实际的手势枚举值 - 使用 .value 比较
        assert Gesture.OPEN.value == "open"
        assert Gesture.FIST.value == "fist"
        assert Gesture.POINT.value == "point"
        assert Gesture.PEACE.value == "peace"
        assert Gesture.THUMBS_UP.value == "thumbs_up"
        assert Gesture.THUMBS_DOWN.value == "thumbs_down"
        assert Gesture.OK.value == "ok"
        assert Gesture.WAVE.value == "wave"
        assert Gesture.UNKNOWN.value == "unknown"

    def test_hand_tracker_get_stats(self):
        """测试获取手势追踪统计"""
        from lelamp.edge.hand_tracker import HandTracker

        tracker = HandTracker()
        stats = tracker.get_stats()

        assert isinstance(stats, dict)
        assert "noop_mode" in stats

    def test_object_detector_no_mediapipe(self):
        """测试物体检测器初始化"""
        from lelamp.edge.object_detector import ObjectDetector

        detector = ObjectDetector()
        assert detector is not None

        # detect 方法返回字典
        result = detector.detect(b"fake_frame")
        assert isinstance(result, dict)
        assert "objects" in result

    def test_object_detector_get_category(self):
        """测试获取物体类别"""
        from lelamp.edge.object_detector import ObjectDetector

        detector = ObjectDetector()

        # 测试获取类别 - NoOp 模式下所有输入都返回"物品"
        category = detector.get_category(0)
        assert category == "物品"

        category = detector.get_category(999)
        assert category == "物品"

    def test_object_detector_stats(self):
        """测试获取物体检测统计"""
        from lelamp.edge.object_detector import ObjectDetector

        detector = ObjectDetector()
        stats = detector.get_stats()

        assert isinstance(stats, dict)
        assert "noop_mode" in stats


@pytest.mark.unit
class TestHybridVision:
    """测试混合视觉服务"""

    def test_hybrid_vision_init(self):
        """测试混合视觉初始化"""
        from lelamp.edge.hybrid_vision import HybridVisionService

        service = HybridVisionService()

        assert service is not None

    def test_hybrid_vision_analyze_query(self):
        """测试分析查询"""
        from lelamp.edge.hybrid_vision import HybridVisionService, QueryComplexity

        service = HybridVisionService()

        # 测试视觉查询
        complexity = service.analyze_query("这是什么？")
        assert isinstance(complexity, QueryComplexity)

    def test_hybrid_vision_get_stats(self):
        """测试获取统计信息"""
        from lelamp.edge.hybrid_vision import HybridVisionService

        service = HybridVisionService()
        stats = service.get_stats()

        assert isinstance(stats, dict)
        assert "services" in stats


@pytest.mark.unit
class TestVisionPrivacy:
    """测试视觉隐私保护"""

    def test_privacy_manager_init(self):
        """测试隐私管理器初始化"""
        from lelamp.service.vision.privacy import CameraPrivacyManager, CameraState, PrivacyConfig

        config = PrivacyConfig()
        manager = CameraPrivacyManager(config)
        assert manager.state == CameraState.IDLE

    def test_camera_states(self):
        """测试摄像头状态"""
        from lelamp.service.vision.privacy import CameraState

        # CameraState 是枚举，比较时需要使用 .value
        assert CameraState.IDLE.value == "idle"
        assert CameraState.ACTIVE.value == "active"
        assert CameraState.PAUSED.value == "paused"
        assert CameraState.CONSENT_REQUIRED.value == "consent_required"

    @pytest.mark.asyncio
    async def test_privacy_guard(self):
        """测试隐私保护上下文管理器"""
        from lelamp.service.vision.privacy import PrivacyGuard, CameraPrivacyManager, PrivacyConfig

        config = PrivacyConfig()
        manager = CameraPrivacyManager(config)

        # PrivacyGuard 是异步上下文管理器
        async with PrivacyGuard(manager, auto_activate=False):
            pass

        # 验证状态变化
        assert manager.state.value == "idle"


@pytest.mark.unit
class TestServiceBase:
    """测试服务基类"""

    def test_service_import(self):
        """测试服务基类导入"""
        from lelamp.service.base import ServiceBase, Priority

        # ServiceBase 是抽象类，不能直接实例化
        assert ServiceBase is not None
        assert Priority is not None

    def test_priority_values(self):
        """测试优先级值"""
        from lelamp.service.base import Priority

        assert Priority.CRITICAL.value == 0
        assert Priority.HIGH.value == 1
        assert Priority.NORMAL.value == 2
        assert Priority.LOW.value == 3


@pytest.mark.unit
class TestRGBService:
    """测试RGB服务"""

    def test_rgb_service_import(self):
        """测试RGB服务导入"""
        from lelamp.service.rgb.noop_rgb_service import NoOpRGBService

        # 测试类存在
        assert NoOpRGBService is not None

    def test_noop_rgb_service(self):
        """测试noop RGB服务"""
        from lelamp.service.rgb.noop_rgb_service import NoOpRGBService

        service = NoOpRGBService()
        assert service is not None


@pytest.mark.unit
class TestMotorsService:
    """测试电机服务"""

    def test_motors_service_import(self):
        """测试电机服务导入"""
        from lelamp.service.motors.noop_motors_service import NoOpMotorsService

        # 测试类存在
        assert NoOpMotorsService is not None

    def test_noop_motors_service(self):
        """测试noop电机服务"""
        from lelamp.service.motors.noop_motors_service import NoOpMotorsService

        service = NoOpMotorsService()
        assert service is not None


@pytest.mark.unit
class TestVisionService:
    """测试视觉服务"""

    def test_vision_service_import(self):
        """测试视觉服务导入"""
        from lelamp.service.vision.vision_service import VisionService

        # 测试类存在
        assert VisionService is not None


@pytest.mark.unit
class TestHealthMonitor:
    """测试健康监控"""

    def test_health_monitor_import(self):
        """测试健康监控导入"""
        from lelamp.service.motors.health_monitor import (
            MotorHealthMonitor,
            HealthStatus,
        )

        # 测试类存在
        assert MotorHealthMonitor is not None

        # 测试健康状态枚举
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.WARNING.value == "warning"
        assert HealthStatus.CRITICAL.value == "critical"
        assert HealthStatus.STALLED.value == "stalled"
