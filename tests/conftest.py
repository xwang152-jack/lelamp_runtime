"""
Pytest 共享配置和 fixtures
"""
import pytest
from unittest.mock import Mock
from lelamp.config import AppConfig


@pytest.fixture
def mock_config():
    """提供测试用配置"""
    return AppConfig(
        livekit_url="wss://test.livekit.io",
        livekit_api_key="test_key",
        livekit_api_secret="test_secret",
        deepseek_model="deepseek-chat",
        deepseek_base_url="https://api.deepseek.com",
        deepseek_api_key="test_deepseek_key",
        modelscope_base_url="https://api-inference.modelscope.cn/v1",
        modelscope_api_key=None,
        modelscope_model="Qwen/Qwen3-VL-235B-A22B-Instruct",
        modelscope_timeout_s=60.0,
        vision_enabled=True,
        camera_index_or_path=0,
        camera_width=1024,
        camera_height=768,
        vision_capture_interval_s=2.5,
        vision_jpeg_quality=92,
        vision_max_age_s=15.0,
    )


@pytest.fixture
def mock_motors_service():
    """Mock MotorsService"""
    mock = Mock()
    mock.is_running = True
    mock.has_pending_event.return_value = False
    return mock


@pytest.fixture
def mock_rgb_service():
    """Mock RGBService"""
    mock = Mock()
    mock.is_running = True
    return mock


@pytest.fixture
def mock_vision_service():
    """Mock VisionService"""
    mock = Mock()
    mock.is_running = True
    return mock
