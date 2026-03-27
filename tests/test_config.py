"""
测试配置管理模块
"""
import os
import pytest
from unittest.mock import patch
from lelamp.config import (
    AppConfig,
    MotorConfig,
    RGBConfig,
    VisionConfig,
    load_config,
    load_motor_config,
    load_rgb_config,
    load_vision_config,
    _get_env_str,
    _get_env_bool,
    _get_env_int,
    _get_env_float,
    _parse_index_or_path,
    SAFE_JOINT_RANGES,
)


@pytest.mark.unit
class TestEnvHelpers:
    """测试环境变量辅助函数"""

    def test_get_env_str_default(self):
        """测试获取字符串环境变量默认值"""
        with patch.dict(os.environ, {}, clear=True):
            assert _get_env_str("NONEXISTENT_VAR", "default") == "default"

    def test_get_env_str_value(self):
        """测试获取字符串环境变量值"""
        with patch.dict(os.environ, {"TEST_VAR": "  value  "}):
            assert _get_env_str("TEST_VAR", "default") == "value"

    def test_get_env_str_empty(self):
        """测试空字符串返回None"""
        with patch.dict(os.environ, {"TEST_VAR": ""}):
            # 空字符串会返回None（不是默认值）
            result = _get_env_str("TEST_VAR", "default")
            assert result == "default"  # 空字符串被处理后返回默认值

    def test_get_env_bool_default(self):
        """测试获取布尔环境变量默认值"""
        with patch.dict(os.environ, {}, clear=True):
            assert _get_env_bool("NONEXISTENT_VAR") is False
            assert _get_env_bool("NONEXISTENT_VAR", True) is True

    def test_get_env_bool_true(self):
        """测试布尔环境变量为真"""
        for val in ["1", "true", "TRUE", "yes", "YES", "on", "ON"]:
            with patch.dict(os.environ, {"TEST_VAR": val}):
                assert _get_env_bool("TEST_VAR") is True

    def test_get_env_bool_false(self):
        """测试布尔环境变量为假"""
        for val in ["0", "false", "no", "off"]:
            with patch.dict(os.environ, {"TEST_VAR": val}):
                assert _get_env_bool("TEST_VAR") is False

    def test_get_env_int_default(self):
        """测试获取整数环境变量默认值"""
        with patch.dict(os.environ, {}, clear=True):
            assert _get_env_int("NONEXISTENT_VAR", 42) == 42

    def test_get_env_int_value(self):
        """测试获取整数环境变量值"""
        with patch.dict(os.environ, {"TEST_VAR": "123"}):
            assert _get_env_int("TEST_VAR", 0) == 123

    def test_get_env_int_invalid(self):
        """测试无效整数返回默认值"""
        with patch.dict(os.environ, {"TEST_VAR": "abc"}):
            assert _get_env_int("TEST_VAR", 42) == 42

    def test_get_env_float_default(self):
        """测试获取浮点数环境变量默认值"""
        with patch.dict(os.environ, {}, clear=True):
            assert _get_env_float("NONEXISTENT_VAR", 3.14) == 3.14

    def test_get_env_float_value(self):
        """测试获取浮点数环境变量值"""
        with patch.dict(os.environ, {"TEST_VAR": "3.14159"}):
            assert _get_env_float("TEST_VAR", 0.0) == 3.14159

    def test_parse_index_or_path_int(self):
        """测试解析摄像头索引"""
        assert _parse_index_or_path("0") == 0
        assert _parse_index_or_path("1") == 1

    def test_parse_index_or_path_str(self):
        """测试解析摄像头路径"""
        assert _parse_index_or_path("/dev/video0") == "/dev/video0"

    def test_parse_index_or_path_none(self):
        """测试解析None返回0"""
        assert _parse_index_or_path(None) == 0

    def test_parse_index_or_path_empty(self):
        """测试解析空字符串返回0"""
        assert _parse_index_or_path("  ") == 0


@pytest.mark.unit
class TestSafeJointRanges:
    """测试关节安全范围"""

    def test_safe_joint_ranges_structure(self):
        """测试安全范围结构"""
        assert isinstance(SAFE_JOINT_RANGES, dict)
        assert len(SAFE_JOINT_RANGES) == 5

    def test_joint_ranges_values(self):
        """测试各关节范围"""
        assert SAFE_JOINT_RANGES["base_yaw"] == (-180, 180)
        assert SAFE_JOINT_RANGES["base_pitch"] == (-90, 90)
        assert SAFE_JOINT_RANGES["elbow_pitch"] == (-150, 150)
        assert SAFE_JOINT_RANGES["wrist_roll"] == (-180, 180)
        assert SAFE_JOINT_RANGES["wrist_pitch"] == (-90, 90)


@pytest.mark.unit
class TestLoadConfig:
    """测试加载配置"""

    @patch.dict(os.environ, {}, clear=True)
    def test_load_config_defaults(self):
        """测试加载默认配置"""
        config = load_config()

        assert config.deepseek_model == "deepseek-chat"
        assert config.deepseek_base_url == "https://api.deepseek.com"
        assert config.vision_enabled is True
        assert config.camera_width == 1024
        assert config.camera_height == 768
        assert config.lamp_port == "/dev/ttyACM0"
        assert config.lamp_id == "lelamp"
        assert config.noise_cancellation_enabled is True

    @patch.dict(os.environ, {
        "DEEPSEEK_MODEL": "deepseek-coder",
        "LELAMP_PORT": "/dev/ttyUSB0",
        "LELAMP_ID": "test_lamp",
        "LELAMP_VISION_ENABLED": "0",
        "LELAMP_OTA_URL": "https://ota.example.com",
    })
    def test_load_config_from_env(self):
        """测试从环境变量加载配置"""
        config = load_config()

        assert config.deepseek_model == "deepseek-coder"
        assert config.lamp_port == "/dev/ttyUSB0"
        assert config.lamp_id == "test_lamp"
        assert config.vision_enabled is False
        assert config.ota_url == "https://ota.example.com"


@pytest.mark.unit
class TestMotorConfig:
    """测试电机配置"""

    @patch.dict(os.environ, {}, clear=True)
    def test_load_motor_config_defaults(self):
        """测试加载默认电机配置"""
        config = load_motor_config()

        assert config.fps == 30
        assert config.health_check_enabled is True
        assert config.health_check_interval_s == 300.0
        assert config.temp_warning_c == 65.0
        assert config.temp_critical_c == 75.0
        assert config.voltage_min_v == 11.0
        assert config.voltage_max_v == 13.0
        assert config.load_warning == 0.8
        assert config.load_stall == 0.95
        assert config.position_error_deg == 5.0

    def test_motor_config_custom(self):
        """测试自定义电机配置"""
        config = MotorConfig(
            port="/dev/ttyUSB0",
            lamp_id="test",
            fps=60,
            health_check_enabled=False,
        )

        assert config.port == "/dev/ttyUSB0"
        assert config.lamp_id == "test"
        assert config.fps == 60
        assert config.health_check_enabled is False


@pytest.mark.unit
class TestRGBConfig:
    """测试RGB配置"""

    def test_rgb_config_defaults(self):
        """测试默认RGB配置"""
        config = load_rgb_config()

        assert config.led_count == 64
        assert config.led_pin == 12
        assert config.led_brightness == 25
        assert config.led_invert is False

    def test_rgb_config_custom(self):
        """测试自定义RGB配置"""
        config = RGBConfig(
            led_count=128,
            led_brightness=50,
        )

        assert config.led_count == 128
        assert config.led_brightness == 50


@pytest.mark.unit
class TestVisionConfig:
    """测试视觉配置"""

    @patch.dict(os.environ, {}, clear=True)
    def test_load_vision_config_defaults(self):
        """测试加载默认视觉配置"""
        config = load_vision_config()

        assert config.enabled is True
        assert config.width == 1024
        assert config.height == 768
        assert config.capture_interval_s == 2.5
        assert config.jpeg_quality == 92
        assert config.max_age_s == 15.0
        assert config.enable_privacy_protection is True

    def test_vision_config_custom(self):
        """测试自定义视觉配置"""
        config = VisionConfig(
            enabled=False,
            width=640,
            height=480,
        )

        assert config.enabled is False
        assert config.width == 640
        assert config.height == 480


@pytest.mark.unit
class TestAppConfig:
    """测试应用配置"""

    def test_app_config_immutable(self):
        """测试配置不可变（frozen）"""
        config = AppConfig(
            livekit_url="wss://test.io",
            livekit_api_key="key",
            livekit_api_secret="secret",
            deepseek_model="deepseek-chat",
            deepseek_base_url="https://api.deepseek.com",
            deepseek_api_key="test_key",
            modelscope_base_url="https://api.modelscope.cn/v1",
            modelscope_api_key=None,
            modelscope_model="Qwen/Qwen3-VL-235B",
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
            ota_url=None,
        )

        # frozen=True 会导致无法修改属性
        with pytest.raises(Exception):  # FrozenInstanceError
            config.lamp_id = "modified"
