"""
Config 单元测试
"""
import pytest
import os
from unittest.mock import patch
from lelamp.config import (
    _get_env_str,
    _get_env_bool,
    _get_env_int,
    _get_env_float,
    _require_env,
    _parse_index_or_path,
    AppConfig,
)


@pytest.mark.unit
class TestConfigHelpers:
    """配置辅助函数测试"""

    def test_get_env_str_with_default(self):
        """测试字符串环境变量获取"""
        with patch.dict(os.environ, {}, clear=True):
            assert _get_env_str("MISSING_VAR", "default") == "default"

        with patch.dict(os.environ, {"TEST_VAR": "value"}, clear=True):
            assert _get_env_str("TEST_VAR", "default") == "value"

    def test_get_env_str_strips_whitespace(self):
        """测试字符串环境变量去除空白"""
        with patch.dict(os.environ, {"TEST_VAR": "  value  "}, clear=True):
            assert _get_env_str("TEST_VAR", "default") == "value"

    def test_get_env_str_empty_returns_default(self):
        """测试空字符串返回默认值"""
        with patch.dict(os.environ, {"TEST_VAR": ""}, clear=True):
            assert _get_env_str("TEST_VAR", "default") == "default"

        with patch.dict(os.environ, {"TEST_VAR": "   "}, clear=True):
            assert _get_env_str("TEST_VAR", "default") == "default"

    def test_get_env_bool(self):
        """测试布尔环境变量解析"""
        test_cases = {
            "1": True,
            "true": True,
            "True": True,
            "TRUE": True,
            "yes": True,
            "YES": True,
            "on": True,
            "ON": True,
            "0": False,
            "false": False,
            "False": False,
            "no": False,
            "anything": False,
        }

        for value, expected in test_cases.items():
            with patch.dict(os.environ, {"TEST_BOOL": value}, clear=True):
                assert _get_env_bool("TEST_BOOL", False) == expected, f"Failed for value: {value}"

    def test_get_env_bool_default(self):
        """测试布尔环境变量默认值"""
        with patch.dict(os.environ, {}, clear=True):
            assert _get_env_bool("MISSING_BOOL", True) is True
            assert _get_env_bool("MISSING_BOOL", False) is False

    def test_get_env_int(self):
        """测试整数环境变量解析"""
        with patch.dict(os.environ, {"TEST_INT": "42"}, clear=True):
            assert _get_env_int("TEST_INT", 0) == 42

        with patch.dict(os.environ, {"TEST_INT": "-100"}, clear=True):
            assert _get_env_int("TEST_INT", 0) == -100

    def test_get_env_int_invalid(self):
        """测试整数环境变量无效值回退到默认值"""
        with patch.dict(os.environ, {"TEST_INT": "invalid"}, clear=True):
            assert _get_env_int("TEST_INT", 10) == 10

        with patch.dict(os.environ, {"TEST_INT": "3.14"}, clear=True):
            assert _get_env_int("TEST_INT", 10) == 10

    def test_get_env_int_missing(self):
        """测试整数环境变量缺失返回默认值"""
        with patch.dict(os.environ, {}, clear=True):
            assert _get_env_int("MISSING_INT", 99) == 99

    def test_get_env_float(self):
        """测试浮点数环境变量解析"""
        with patch.dict(os.environ, {"TEST_FLOAT": "3.14"}, clear=True):
            assert _get_env_float("TEST_FLOAT", 0.0) == 3.14

        with patch.dict(os.environ, {"TEST_FLOAT": "-2.5"}, clear=True):
            assert _get_env_float("TEST_FLOAT", 0.0) == -2.5

        with patch.dict(os.environ, {"TEST_FLOAT": "42"}, clear=True):
            assert _get_env_float("TEST_FLOAT", 0.0) == 42.0

    def test_get_env_float_invalid(self):
        """测试浮点数环境变量无效值回退到默认值"""
        with patch.dict(os.environ, {"TEST_FLOAT": "invalid"}, clear=True):
            assert _get_env_float("TEST_FLOAT", 1.0) == 1.0

    def test_get_env_float_missing(self):
        """测试浮点数环境变量缺失返回默认值"""
        with patch.dict(os.environ, {}, clear=True):
            assert _get_env_float("MISSING_FLOAT", 2.71) == 2.71

    def test_require_env_present(self):
        """测试必需环境变量存在"""
        with patch.dict(os.environ, {"REQUIRED_VAR": "value"}, clear=True):
            assert _require_env("REQUIRED_VAR") == "value"

    def test_require_env_missing(self):
        """测试必需环境变量缺失抛出异常"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="缺少环境变量"):
                _require_env("MISSING_REQUIRED")

    def test_require_env_empty(self):
        """测试必需环境变量为空抛出异常"""
        with patch.dict(os.environ, {"EMPTY_VAR": ""}, clear=True):
            with pytest.raises(RuntimeError, match="缺少环境变量"):
                _require_env("EMPTY_VAR")

        with patch.dict(os.environ, {"EMPTY_VAR": "   "}, clear=True):
            with pytest.raises(RuntimeError, match="缺少环境变量"):
                _require_env("EMPTY_VAR")

    def test_parse_index_or_path_integer(self):
        """测试解析整数索引"""
        assert _parse_index_or_path("0") == 0
        assert _parse_index_or_path("2") == 2
        assert _parse_index_or_path("10") == 10

    def test_parse_index_or_path_string(self):
        """测试解析字符串路径"""
        assert _parse_index_or_path("/dev/video0") == "/dev/video0"
        assert _parse_index_or_path("rtsp://camera") == "rtsp://camera"
        assert _parse_index_or_path("/path/to/video.mp4") == "/path/to/video.mp4"

    def test_parse_index_or_path_none(self):
        """测试解析 None 返回默认值 0"""
        assert _parse_index_or_path(None) == 0

    def test_parse_index_or_path_empty(self):
        """测试解析空字符串返回默认值 0"""
        assert _parse_index_or_path("") == 0
        assert _parse_index_or_path("   ") == 0


@pytest.mark.unit
class TestAppConfig:
    """AppConfig 测试"""

    def test_app_config_immutable(self):
        """测试 AppConfig 是冻结的（不可变）"""
        config = AppConfig(
            livekit_url="wss://test",
            livekit_api_key="key",
            livekit_api_secret="secret",
            deepseek_model="model",
            deepseek_base_url="url",
            deepseek_api_key="key",
            modelscope_base_url="url",
            modelscope_api_key=None,
            modelscope_model="model",
            modelscope_timeout_s=60.0,
            vision_enabled=True,
            camera_index_or_path=0,
            camera_width=1024,
            camera_height=768,
            vision_capture_interval_s=2.5,
            vision_jpeg_quality=92,
            vision_max_age_s=15.0,
            camera_rotate_deg=0,
            camera_flip="none",
            baidu_api_key="baidu_key",
            baidu_secret_key="baidu_secret",
            baidu_cuid="lelamp",
            baidu_tts_per=4,
            lamp_port="/dev/ttyACM0",
            lamp_id="lelamp",
            noise_cancellation_enabled=True,
            greeting_text="Hello",
        )

        # 尝试修改应该失败
        with pytest.raises(AttributeError):
            config.livekit_url = "new_url"

        with pytest.raises(AttributeError):
            config.deepseek_api_key = "new_key"

    def test_app_config_creation(self):
        """测试 AppConfig 正常创建"""
        config = AppConfig(
            livekit_url="wss://test.livekit.cloud",
            livekit_api_key="test_key",
            livekit_api_secret="test_secret",
            deepseek_model="deepseek-chat",
            deepseek_base_url="https://api.deepseek.com",
            deepseek_api_key="deepseek_key",
            modelscope_base_url="https://api-inference.modelscope.cn/v1",
            modelscope_api_key="modelscope_key",
            modelscope_model="Qwen/Qwen3-VL-235B-A22B-Instruct",
            modelscope_timeout_s=60.0,
            vision_enabled=True,
            camera_index_or_path=0,
            camera_width=1920,
            camera_height=1080,
            vision_capture_interval_s=2.5,
            vision_jpeg_quality=95,
            vision_max_age_s=15.0,
            camera_rotate_deg=180,
            camera_flip="horizontal",
            baidu_api_key="baidu_key",
            baidu_secret_key="baidu_secret",
            baidu_cuid="test_device",
            baidu_tts_per=5,
            lamp_port="/dev/ttyUSB0",
            lamp_id="test_lamp",
            noise_cancellation_enabled=False,
            greeting_text="测试问候语",
        )

        # 验证所有字段
        assert config.livekit_url == "wss://test.livekit.cloud"
        assert config.livekit_api_key == "test_key"
        assert config.deepseek_model == "deepseek-chat"
        assert config.vision_enabled is True
        assert config.camera_width == 1920
        assert config.camera_height == 1080
        assert config.camera_rotate_deg == 180
        assert config.camera_flip == "horizontal"
        assert config.lamp_port == "/dev/ttyUSB0"
        assert config.greeting_text == "测试问候语"
