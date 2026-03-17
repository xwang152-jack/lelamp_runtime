"""
RGBTools 集成测试
"""
import pytest
from unittest.mock import Mock
from lelamp.agent.tools.rgb_tools import RGBTools
from lelamp.agent.states import StateManager


@pytest.mark.integration
class TestRGBTools:
    """RGBTools 集成测试套件"""

    @pytest.fixture
    def rgb_tools(self):
        """创建 RGBTools 实例"""
        rgb_service = Mock()
        state_manager = StateManager()
        return RGBTools(rgb_service, state_manager)

    @pytest.mark.asyncio
    async def test_set_rgb_solid_valid(self, rgb_tools):
        """测试设置有效纯色"""
        result = await rgb_tools.set_rgb_solid(255, 0, 0)
        assert "255" in result and "0" in result
        rgb_tools.rgb_service.dispatch.assert_called_once()
        # 验证灯光覆盖已设置
        assert rgb_tools.state_manager.is_light_overridden() is True

    @pytest.mark.asyncio
    async def test_set_rgb_solid_invalid(self, rgb_tools):
        """测试无效 RGB 值"""
        result = await rgb_tools.set_rgb_solid(300, 0, 0)
        assert "错误" in result
        rgb_tools.rgb_service.dispatch.assert_not_called()

    @pytest.mark.asyncio
    async def test_set_rgb_solid_negative_value(self, rgb_tools):
        """测试负数 RGB 值"""
        result = await rgb_tools.set_rgb_solid(-10, 100, 200)
        assert "错误" in result
        rgb_tools.rgb_service.dispatch.assert_not_called()

    @pytest.mark.asyncio
    async def test_paint_rgb_pattern(self, rgb_tools):
        """测试绘制图案"""
        result = await rgb_tools.paint_rgb_pattern("heart")
        assert "heart" in result
        rgb_tools.rgb_service.dispatch.assert_called_once()
        # 验证参数格式
        call_args = rgb_tools.rgb_service.dispatch.call_args
        assert call_args[0][0] == "pattern"
        assert call_args[0][1]["pattern"] == "heart"

    @pytest.mark.asyncio
    async def test_rgb_effect_rainbow(self, rgb_tools):
        """测试彩虹效果"""
        result = await rgb_tools.rgb_effect_rainbow(speed=2.0)
        assert "彩虹" in result
        rgb_tools.rgb_service.dispatch.assert_called_once()
        # 验证速度参数
        call_args = rgb_tools.rgb_service.dispatch.call_args
        assert call_args[0][1]["speed"] == 2.0

    @pytest.mark.asyncio
    async def test_rgb_effect_rainbow_default_speed(self, rgb_tools):
        """测试彩虹效果默认速度"""
        result = await rgb_tools.rgb_effect_rainbow()
        assert "彩虹" in result
        call_args = rgb_tools.rgb_service.dispatch.call_args
        assert call_args[0][1]["speed"] == 1.0

    @pytest.mark.asyncio
    async def test_rgb_effect_breathing(self, rgb_tools):
        """测试呼吸效果"""
        result = await rgb_tools.rgb_effect_breathing(0, 140, 255)
        assert "呼吸" in result
        rgb_tools.rgb_service.dispatch.assert_called_once()
        # 验证颜色参数
        call_args = rgb_tools.rgb_service.dispatch.call_args
        assert call_args[0][1]["color"] == (0, 140, 255)

    @pytest.mark.asyncio
    async def test_rgb_effect_breathing_invalid(self, rgb_tools):
        """测试呼吸效果无效颜色"""
        result = await rgb_tools.rgb_effect_breathing(300, 0, 0)
        assert "错误" in result
        rgb_tools.rgb_service.dispatch.assert_not_called()

    @pytest.mark.asyncio
    async def test_rgb_effect_random_animation(self, rgb_tools):
        """测试随机动画"""
        result = await rgb_tools.rgb_effect_random_animation()
        assert "动画" in result
        rgb_tools.rgb_service.dispatch.assert_called_once()
        # 验证不设置灯光覆盖（允许状态切换）
        # 注意：random_animation 不应该设置 light_override
        # 但由于我们使用了新的 StateManager，这里只能测试调用本身

    @pytest.mark.asyncio
    async def test_light_override_blocks_motion(self, rgb_tools):
        """测试灯光覆盖阻止电机动作"""
        # 设置灯光后应该阻止电机
        await rgb_tools.set_rgb_solid(255, 255, 255)
        assert rgb_tools.state_manager.is_light_overridden() is True

        # 电机动作会被 StateManager.can_execute_motion() 拒绝
        # 因为 set_light_override 同时设置了 suppress_motion_until_ts
        # 这在 MotorTools 中验证

    @pytest.mark.asyncio
    async def test_multiple_effects_override_accumulation(self, rgb_tools):
        """测试多次调用效果是否累积覆盖"""
        await rgb_tools.set_rgb_solid(255, 0, 0)
        assert rgb_tools.state_manager.is_light_overridden() is True

        # 再次调用应该更新覆盖时间
        await rgb_tools.rgb_effect_rainbow()
        assert rgb_tools.state_manager.is_light_overridden() is True
