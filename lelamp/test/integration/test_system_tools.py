"""
SystemTools 集成测试

测试系统工具的各个功能，包括电机、RGB、系统控制和 OTA 更新。
"""
import asyncio
import os
import json
from unittest.mock import AsyncMock, MagicMock, patch, call
import pytest

# 模拟环境变量
os.environ.update({
    "LELAMP_LIGHT_OVERRIDE_S": "10",
    "LELAMP_SUPPRESS_MOTION_AFTER_LIGHT_S": "2",
    "BOCHA_API_KEY": "test-api-key",
})

from lelamp.agent.tools.system_tools import SystemTools


class TestSystemToolsMotor:
    """电机相关工具测试"""

    @pytest.fixture
    def motors_service(self):
        """模拟电机服务"""
        service = MagicMock()
        service.get_available_recordings.return_value = ["nod", "shake", "bow"]
        service.get_motor_health_summary.return_value = {
            "base_yaw": {
                "latest": {
                    "status": "healthy",
                    "temperature": 45.0,
                    "voltage": 12.0,
                    "load": 0.3,
                    "position": 0.0
                },
                "warning_count": 0,
                "critical_count": 0,
                "stall_count": 0
            }
        }
        service.reset_health_statistics = MagicMock()
        return service

    @pytest.fixture
    def mock_robot(self):
        """模拟机器人"""
        robot = MagicMock()
        bus = MagicMock()
        bus.write = MagicMock()
        robot.bus = bus
        return robot

    @pytest.fixture
    def system_tools(self, motors_service):
        """创建 SystemTools 实例"""
        state_manager = MagicMock()
        state_manager.set_light_override = MagicMock()

        tools = SystemTools(
            motors_service=motors_service,
            rgb_service=MagicMock(),
            ota_manager=MagicMock(),
            ota_url="https://example.com/ota",
            state_manager=state_manager,
            get_rate_limit_stats_func=MagicMock(return_value={}),
        )
        return tools

    @pytest.mark.asyncio
    async def test_get_available_recordings(self, system_tools):
        """测试获取可用录制列表"""
        result = await system_tools.get_available_recordings()
        assert "Available recordings:" in result
        assert "nod" in result
        assert "shake" in result

    @pytest.mark.asyncio
    async def test_get_available_recordings_empty(self, system_tools, motors_service):
        """测试没有录制时的情况"""
        motors_service.get_available_recordings.return_value = []
        result = await system_tools.get_available_recordings()
        assert "No recordings found" in result

    @pytest.mark.asyncio
    async def test_tune_motor_pid_success(self, system_tools, motors_service, mock_robot):
        """测试成功调整电机 PID"""
        motors_service.robot = mock_robot

        result = await system_tools.tune_motor_pid("base_yaw", 20, 5, 30)

        assert "成功更新" in result
        assert "base_yaw" in result
        assert "P: 20" in result
        assert "I: 5" in result
        assert "D: 30" in result

        # 验证 bus.write 被调用
        mock_robot.bus.write.assert_any_call("P_Coefficient", "base_yaw", 20)
        mock_robot.bus.write.assert_any_call("I_Coefficient", "base_yaw", 5)
        mock_robot.bus.write.assert_any_call("D_Coefficient", "base_yaw", 30)

    @pytest.mark.asyncio
    async def test_tune_motor_pid_invalid_motor(self, system_tools):
        """测试无效的电机名称"""
        result = await system_tools.tune_motor_pid("invalid_motor", 20, 5, 30)
        assert "无效的舵机名称" in result

    @pytest.mark.asyncio
    async def test_tune_motor_pid_invalid_p_coefficient(self, system_tools):
        """测试无效的 P 系数"""
        result = await system_tools.tune_motor_pid("base_yaw", 50, 5, 30)
        assert "P 系数必须在" in result

    @pytest.mark.asyncio
    async def test_tune_motor_pid_invalid_i_coefficient(self, system_tools):
        """测试无效的 I 系数"""
        result = await system_tools.tune_motor_pid("base_yaw", 20, -1, 30)
        assert "I 系数必须在" in result

    @pytest.mark.asyncio
    async def test_tune_motor_pid_invalid_d_coefficient(self, system_tools):
        """测试无效的 D 系数"""
        result = await system_tools.tune_motor_pid("base_yaw", 20, 5, 50)
        assert "D 系数必须在" in result

    @pytest.mark.asyncio
    async def test_tune_motor_pid_no_robot(self, system_tools, motors_service):
        """测试机器人未连接"""
        motors_service.robot = None
        result = await system_tools.tune_motor_pid("base_yaw", 20, 5, 30)
        assert "舵机服务未连接" in result

    @pytest.mark.asyncio
    async def test_reset_motor_health_stats_single(self, system_tools, motors_service):
        """测试重置单个电机的健康统计"""
        result = await system_tools.reset_motor_health_stats("base_yaw")
        assert "已重置舵机 base_yaw" in result
        motors_service.reset_health_statistics.assert_called_once_with("base_yaw")

    @pytest.mark.asyncio
    async def test_reset_motor_health_stats_all(self, system_tools, motors_service):
        """测试重置所有电机的健康统计"""
        result = await system_tools.reset_motor_health_stats()
        assert "已重置所有舵机" in result
        motors_service.reset_health_statistics.assert_called_once_with(None)

    @pytest.mark.asyncio
    async def test_reset_motor_health_stats_invalid_motor(self, system_tools, motors_service):
        """测试重置无效电机的健康统计"""
        result = await system_tools.reset_motor_health_stats("invalid_motor")
        assert "无效的舵机名称" in result
        motors_service.reset_health_statistics.assert_not_called()


class TestSystemToolsRGB:
    """RGB 效果扩展测试"""

    @pytest.fixture
    def rgb_service(self):
        """模拟 RGB 服务"""
        service = MagicMock()
        return service

    @pytest.fixture
    def state_manager(self):
        """模拟状态管理器"""
        manager = MagicMock()
        return manager

    @pytest.fixture
    def system_tools(self, rgb_service, state_manager):
        """创建 SystemTools 实例"""
        tools = SystemTools(
            motors_service=MagicMock(),
            rgb_service=rgb_service,
            ota_manager=MagicMock(),
            ota_url="https://example.com/ota",
            state_manager=state_manager,
            get_rate_limit_stats_func=MagicMock(return_value={}),
        )
        return tools

    @pytest.mark.asyncio
    async def test_set_rgb_brightness(self, system_tools, rgb_service, state_manager):
        """测试设置 RGB 亮度"""
        result = await system_tools.set_rgb_brightness(50)

        assert "亮度设置为 50%" in result
        rgb_service.dispatch.assert_called_once_with("brightness", 128, priority=1)

    @pytest.mark.asyncio
    async def test_set_rgb_brightness_invalid(self, system_tools, rgb_service):
        """测试无效的亮度值"""
        result = await system_tools.set_rgb_brightness(150)
        assert "亮度范围是" in result
        rgb_service.dispatch.assert_not_called()

    @pytest.mark.asyncio
    async def test_rgb_effect_wave(self, system_tools, rgb_service, state_manager):
        """测试波纹效果"""
        result = await system_tools.rgb_effect_wave(60, 180, 255, 1.0, 1.2, 30)

        assert "波纹动态灯效" in result
        state_manager.set_light_override.assert_called_once()
        rgb_service.dispatch.assert_called_once()
        call_args = rgb_service.dispatch.call_args
        assert call_args[0][0] == "effect"
        assert call_args[0][1]["name"] == "wave"
        assert call_args[0][1]["color"] == (60, 180, 255)

    @pytest.mark.asyncio
    async def test_rgb_effect_wave_invalid_rgb(self, system_tools, rgb_service):
        """测试波纹效果的无效 RGB 值"""
        result = await system_tools.rgb_effect_wave(300, 180, 255, 1.0, 1.2, 30)
        assert "RGB 值必须在" in result
        rgb_service.dispatch.assert_not_called()

    @pytest.mark.asyncio
    async def test_rgb_effect_fire(self, system_tools, rgb_service, state_manager):
        """测试火焰效果"""
        result = await system_tools.rgb_effect_fire(1.0, 30)

        assert "火焰动态灯效" in result
        state_manager.set_light_override.assert_called_once()
        rgb_service.dispatch.assert_called_once()
        call_args = rgb_service.dispatch.call_args
        assert call_args[0][0] == "effect"
        assert call_args[0][1]["name"] == "fire"

    @pytest.mark.asyncio
    async def test_rgb_effect_emoji(self, system_tools, rgb_service, state_manager):
        """测试表情效果"""
        result = await system_tools.rgb_effect_emoji("smile", 255, 200, 60, 0, 0, 0, True, 2.2, 30)

        assert "表情动画：smile" in result
        state_manager.set_light_override.assert_called_once()
        rgb_service.dispatch.assert_called_once()
        call_args = rgb_service.dispatch.call_args
        assert call_args[0][0] == "effect"
        assert call_args[0][1]["name"] == "emoji"
        assert call_args[0][1]["emoji"] == "smile"

    @pytest.mark.asyncio
    async def test_rgb_effect_emoji_invalid_rgb(self, system_tools, rgb_service):
        """测试表情效果的无效 RGB 值"""
        result = await system_tools.rgb_effect_emoji("smile", 300, 200, 60, 0, 0, 0, True, 2.2, 30)
        assert "RGB 值必须在" in result
        rgb_service.dispatch.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_rgb_effect(self, system_tools, rgb_service, state_manager):
        """测试停止 RGB 效果"""
        result = await system_tools.stop_rgb_effect()

        assert "停止动态灯效" in result
        state_manager.set_light_override.assert_called_once()
        rgb_service.dispatch.assert_called_once_with("effect_stop", None, priority=1)


class TestSystemToolsSystem:
    """系统控制测试"""

    @pytest.fixture
    def system_tools(self):
        """创建 SystemTools 实例"""
        tools = SystemTools(
            motors_service=MagicMock(),
            rgb_service=MagicMock(),
            ota_manager=MagicMock(),
            ota_url="https://example.com/ota",
            state_manager=MagicMock(),
            get_rate_limit_stats_func=MagicMock(return_value={
                "search": {
                    "requests_total": 100,
                    "requests_allowed": 95,
                    "requests_denied": 5,
                    "denial_rate": 0.05,
                    "avg_wait_time": 0.2
                }
            }),
        )
        return tools

    @pytest.mark.asyncio
    @patch("asyncio.create_subprocess_exec")
    async def test_set_volume(self, mock_subprocess, system_tools):
        """测试设置音量"""
        mock_process = MagicMock()
        mock_process.wait = AsyncMock()
        mock_subprocess.return_value = mock_process

        result = await system_tools.set_volume(75)

        assert "音量设置为 75%" in result or "volume" in result.lower()
        mock_subprocess.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_volume_invalid(self, system_tools):
        """测试无效的音量值"""
        result = await system_tools.set_volume(150)
        assert "Volume must be between" in result or "音量" in result

    @pytest.mark.asyncio
    async def test_get_rate_limit_stats(self, system_tools):
        """测试获取速率限制统计"""
        result = await system_tools.get_rate_limit_stats()

        assert "API 速率限制统计" in result or "Rate Limit Stats" in result
        assert "search" in result
        assert "95" in result or "requests_allowed" in result

    @pytest.mark.asyncio
    @patch("lelamp.utils.url_validation.validate_external_url", return_value=True)
    @patch("urllib.request.urlopen")
    async def test_web_search_success(self, mock_urlopen, mock_validate, system_tools):
        """测试成功的网络搜索"""
        # 模拟 API 响应
        json_data = {
            "code": 200,
            "data": {
                "webPages": {
                    "value": [
                        {"name": "测试结果1", "snippet": "这是测试摘要1"},
                        {"name": "测试结果2", "snippet": "这是测试摘要2"},
                    ]
                }
            }
        }

        # 使用真实的 Mock 响应
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(json_data).encode("utf-8")

        # 支持上下文管理器
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_response
        mock_context.__exit__.return_value = False
        mock_urlopen.return_value = mock_context

        result = await system_tools.web_search("测试查询")

        assert "联网搜索" in result or "search" in result.lower()
        assert "测试结果1" in result

    @pytest.mark.asyncio
    async def test_web_search_empty_query(self, system_tools):
        """测试空搜索查询"""
        result = await system_tools.web_search("")
        assert "搜索关键词不能为空" in result or "query cannot be empty" in result.lower()

    @pytest.mark.asyncio
    async def test_web_search_too_long(self, system_tools):
        """测试过长的搜索查询"""
        result = await system_tools.web_search("x" * 600)
        assert "搜索关键词过长" in result or "too long" in result.lower()

    @pytest.mark.asyncio
    async def test_web_search_no_api_key(self, system_tools):
        """测试没有 API Key 的情况"""
        with patch.dict(os.environ, {"BOCHA_API_KEY": ""}, clear=False):
            # 重新创建不带 API key 的实例
            tools = SystemTools(
                motors_service=MagicMock(),
                rgb_service=MagicMock(),
                ota_manager=MagicMock(),
                ota_url="https://example.com/ota",
                state_manager=MagicMock(),
                get_rate_limit_stats_func=MagicMock(return_value={}),
            )
            result = await tools.web_search("test")
            assert "未配置 BOCHA_API_KEY" in result or "API key" in result.lower()

    @pytest.mark.asyncio
    @patch("lelamp.utils.url_validation.validate_external_url")
    async def test_web_search_url_validation_failed(self, mock_validate, system_tools):
        """测试 URL 验证失败"""
        mock_validate.return_value = False

        result = await system_tools.web_search("测试查询")
        assert "搜索服务配置错误" in result or "configuration error" in result.lower()


class TestSystemToolsOTA:
    """OTA 更新测试"""

    @pytest.fixture
    def ota_manager(self):
        """模拟 OTA 管理器"""
        manager = MagicMock()
        manager.check_for_update.return_value = (True, "1.0.1", "修复了一些bug")
        manager.perform_update.return_value = "更新成功 (v1.0.1)。请重启设备或服务以生效。"
        return manager

    @pytest.fixture
    def system_tools(self, ota_manager):
        """创建 SystemTools 实例"""
        tools = SystemTools(
            motors_service=MagicMock(),
            rgb_service=MagicMock(),
            ota_manager=ota_manager,
            ota_url="https://example.com/ota",
            state_manager=MagicMock(),
            get_rate_limit_stats_func=MagicMock(return_value={}),
        )
        return tools

    @pytest.mark.asyncio
    async def test_check_for_updates_has_update(self, system_tools):
        """测试检查更新 - 有新版本"""
        result = await system_tools.check_for_updates()

        assert "发现新版本" in result or "new version" in result.lower()
        assert "1.0.1" in result

    @pytest.mark.asyncio
    async def test_check_for_updates_no_update(self, system_tools, ota_manager):
        """测试检查更新 - 无新版本"""
        ota_manager.check_for_update.return_value = (False, "1.0.0", "当前已是最新版本")

        result = await system_tools.check_for_updates()

        assert "最新版本" in result or "already up to date" in result.lower()
        assert "1.0.0" in result

    @pytest.mark.asyncio
    async def test_check_for_updates_no_url(self):
        """测试没有配置 OTA URL 的情况"""
        tools = SystemTools(
            motors_service=MagicMock(),
            rgb_service=MagicMock(),
            ota_manager=MagicMock(),
            ota_url=None,
            state_manager=MagicMock(),
            get_rate_limit_stats_func=MagicMock(return_value={}),
        )

        result = await tools.check_for_updates()
        assert "OTA 更新服务未配置" in result or "not configured" in result.lower()

    @pytest.mark.asyncio
    @patch("lelamp.agent.tools.system_tools.sys.exit")
    async def test_perform_ota_update_success(self, mock_exit, system_tools, ota_manager):
        """测试执行 OTA 更新成功"""
        result = await system_tools.perform_ota_update()

        assert "更新成功" in result or "successfully" in result.lower()
        assert "重启" in result or "restart" in result.lower()

    @pytest.mark.asyncio
    async def test_perform_ota_update_no_update(self, system_tools, ota_manager):
        """测试执行 OTA 更新 - 无可用更新"""
        ota_manager.check_for_update.return_value = (False, "1.0.0", "")

        result = await system_tools.perform_ota_update()
        assert "没有可用更新" in result or "no update" in result.lower()

    @pytest.mark.asyncio
    async def test_perform_ota_update_no_url(self):
        """测试没有配置 OTA URL 时执行更新"""
        tools = SystemTools(
            motors_service=MagicMock(),
            rgb_service=MagicMock(),
            ota_manager=MagicMock(),
            ota_url=None,
            state_manager=MagicMock(),
            get_rate_limit_stats_func=MagicMock(return_value={}),
        )

        result = await tools.perform_ota_update()
        assert "OTA 更新服务未配置" in result or "not configured" in result.lower()
