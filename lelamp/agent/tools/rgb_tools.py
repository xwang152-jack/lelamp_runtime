"""
RGB 灯光控制工具模块
"""
import logging
import random
from typing import TYPE_CHECKING

from livekit.agents import function_tool

from .utils import validate_rgb_color

if TYPE_CHECKING:
    from lelamp.service.rgb.rgb_service import RGBService
    from lelamp.agent.states import StateManager


class RGBTools:
    """RGB 灯光控制工具类 - 管理灯光颜色和动画效果"""

    def __init__(
        self,
        rgb_service: "RGBService",
        state_manager: "StateManager",
    ):
        """
        初始化 RGB 工具

        Args:
            rgb_service: RGB 服务实例
            state_manager: 状态管理器实例
        """
        self.rgb_service = rgb_service
        self.state_manager = state_manager
        self.logger = logging.getLogger("lelamp.agent.tools.rgb")

    @function_tool
    async def set_rgb_solid(self, r: int, g: int, b: int) -> str:
        """
        设置纯色灯光

        Args:
            r: 红色值 (0-255)
            g: 绿色值 (0-255)
            b: 蓝色值 (0-255)

        Returns:
            执行结果描述
        """
        self.logger.debug(f"set_rgb_solid called with RGB({r}, {g}, {b})")
        try:
            # 验证颜色范围
            is_valid, error_msg = validate_rgb_color(r, g, b)
            if not is_valid:
                return f"错误：{error_msg}"

            # 导入 Priority（延迟导入避免循环依赖）
            from lelamp.service import Priority

            # 分发纯色事件
            self.rgb_service.dispatch("solid", (r, g, b), priority=Priority.HIGH)

            # 设置灯光覆盖（阻止状态切换改变灯光）
            self.state_manager.set_light_override(duration_s=10.0)

            self.logger.info(f"Set RGB solid color: ({r}, {g}, {b})")
            return f"设置纯色灯光: RGB({r}, {g}, {b})"
        except Exception as e:
            self.logger.error(f"Error setting RGB solid: {e}")
            return f"设置纯色失败：{str(e)}"

    @function_tool
    async def paint_rgb_pattern(self, pattern: str) -> str:
        """
        绘制预定义的 LED 图案

        Args:
            pattern: 图案名称（如 "heart", "smile", "arrow"）

        Returns:
            执行结果描述
        """
        self.logger.debug(f"paint_rgb_pattern called with pattern={pattern}")
        try:
            from lelamp.service import Priority

            # 分发图案事件
            self.rgb_service.dispatch("pattern", {"pattern": pattern}, priority=Priority.NORMAL)

            # 设置灯光覆盖
            self.state_manager.set_light_override(duration_s=10.0)

            self.logger.info(f"Paint RGB pattern: {pattern}")
            return f"绘制图案: {pattern}"
        except Exception as e:
            self.logger.error(f"Error painting RGB pattern: {e}")
            return f"绘制图案失败：{str(e)}"

    @function_tool
    async def rgb_effect_rainbow(self, speed: float = 1.0) -> str:
        """
        启动彩虹效果

        Args:
            speed: 速度倍率（默认 1.0）

        Returns:
            执行结果描述
        """
        self.logger.debug(f"rgb_effect_rainbow called with speed={speed}")
        try:
            from lelamp.service import Priority

            # 分发彩虹效果
            self.rgb_service.dispatch(
                "effect",
                {"name": "rainbow", "speed": float(speed)},
                priority=Priority.NORMAL
            )

            # 设置较长的灯光覆盖时间（动画效果需要更长时间）
            self.state_manager.set_light_override(duration_s=15.0)

            self.logger.info(f"Start rainbow effect with speed {speed}")
            return f"启动彩虹效果（速度: {speed}x）"
        except Exception as e:
            self.logger.error(f"Error starting rainbow effect: {e}")
            return f"启动彩虹效果失败：{str(e)}"

    @function_tool
    async def rgb_effect_breathing(self, r: int, g: int, b: int) -> str:
        """
        启动呼吸效果

        Args:
            r: 红色值 (0-255)
            g: 绿色值 (0-255)
            b: 蓝色值 (0-255)

        Returns:
            执行结果描述
        """
        self.logger.debug(f"rgb_effect_breathing called with RGB({r}, {g}, {b})")
        try:
            # 验证颜色范围
            is_valid, error_msg = validate_rgb_color(r, g, b)
            if not is_valid:
                return f"错误：{error_msg}"

            from lelamp.service import Priority

            # 分发呼吸效果 - 使用 "breath" 事件类型
            self.rgb_service.dispatch(
                "breath",
                {"rgb": (r, g, b)},
                priority=Priority.NORMAL
            )

            # 设置灯光覆盖
            self.state_manager.set_light_override(duration_s=15.0)

            self.logger.info(f"Start breathing effect: RGB({r}, {g}, {b})")
            return f"启动呼吸效果: RGB({r}, {g}, {b})"
        except Exception as e:
            self.logger.error(f"Error starting breathing effect: {e}")
            return f"启动呼吸效果失败：{str(e)}"

    @function_tool
    async def rgb_effect_random_animation(self) -> str:
        """
        启动随机颜色动画（用于 speaking 状态）

        注意：此动画不设置 light_override，允许状态切换时自动更新

        Returns:
            执行结果描述
        """
        self.logger.debug("rgb_effect_random_animation called")
        try:
            # 彩虹色系
            colors = [
                (255, 0, 0),      # 红
                (255, 165, 0),    # 橙
                (255, 255, 0),    # 黄
                (0, 255, 0),      # 绿
                (0, 0, 255),      # 蓝
                (75, 0, 130),     # 靛
                (238, 130, 238)   # 紫
            ]
            color = random.choice(colors)

            from lelamp.service import Priority

            # 分发呼吸效果
            self.rgb_service.dispatch(
                "effect",
                {"name": "breathing", "color": color},
                priority=Priority.HIGH
            )

            # 注意：speaking 动画不设置 light_override，允许状态切换
            self.logger.info(f"Start random speaking animation: {color}")
            return "启动随机说话动画"
        except Exception as e:
            self.logger.error(f"Error starting random animation: {e}")
            return f"启动随机动画失败：{str(e)}"
