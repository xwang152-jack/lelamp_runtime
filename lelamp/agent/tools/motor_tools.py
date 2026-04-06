"""
电机控制工具模块
"""

import logging
from typing import TYPE_CHECKING, Optional

from livekit.agents import function_tool, RunContext

if TYPE_CHECKING:
    from lelamp.service.motors.motors_service import MotorsService
    from lelamp.agent.states import StateManager


# 定义关节安全角度范围
SAFE_JOINT_RANGES = {
    "base_yaw": (-180, 180),
    "base_pitch": (-90, 90),
    "elbow_pitch": (-150, 150),
    "wrist_roll": (-180, 180),
    "wrist_pitch": (-90, 90),
}


class MotorTools:
    """电机控制工具类 - 管理电机动作和健康监控"""

    def __init__(
        self,
        motors_service: "MotorsService",
        state_manager: "StateManager",
    ):
        """
        初始化电机工具

        Args:
            motors_service: 电机服务实例
            state_manager: 状态管理器实例
        """
        self.motors_service = motors_service
        self.state_manager = state_manager
        self.logger = logging.getLogger("lelamp.agent.tools.motor")

    @function_tool()
    async def play_recording(
        self,
        context: RunContext,
        recording_name: str,
    ) -> str:
        """
        播放预录制的电机动作序列

        Args:
            recording_name: 录制动作的名称

        Returns:
            执行结果消息
        """
        self.logger.debug(
            f"play_recording called with recording_name: {recording_name}"
        )
        try:
            # 检查是否允许执行动作
            if not self.state_manager.can_execute_motion():
                return "动作冷却中或已被抑制，暂时无法执行"

            # 记录动作时间
            self.state_manager.record_motion()

            # 分发动作事件到电机服务
            self.motors_service.dispatch("play", recording_name)
            return f"开始执行动作：{recording_name}"
        except Exception as e:
            self.logger.error(f"Error playing recording {recording_name}: {e}")
            return f"播放动作失败：{str(e)}"

    @function_tool()
    async def move_joint(
        self,
        context: RunContext,
        joint_name: str,
        angle: float,
    ) -> str:
        """
        控制指定关节移动到目标角度

        Args:
            joint_name: 关节名称(base_yaw/base_pitch/elbow_pitch/wrist_roll/wrist_pitch)
            angle: 目标角度(度)

        Returns:
            执行结果消息
        """
        self.logger.debug(
            f"move_joint called with joint_name={joint_name}, angle={angle}"
        )
        try:
            # 验证关节名称
            if joint_name not in SAFE_JOINT_RANGES:
                valid_joints = ", ".join(SAFE_JOINT_RANGES.keys())
                return f"无效的关节名称：{joint_name}。可用关节：{valid_joints}"

            # 验证角度范围
            angle_float = float(angle)
            min_angle, max_angle = SAFE_JOINT_RANGES[joint_name]
            if angle_float < min_angle or angle_float > max_angle:
                return (
                    f"角度 {angle_float}° 超出 {joint_name} 的安全范围 "
                    f"({min_angle}° 到 {max_angle}°)。为了安全，我拒绝了这次移动。"
                )

            # 分发移动指令
            self.motors_service.dispatch(
                "move_joint", {"joint_name": joint_name, "angle": angle_float}
            )
            return f"已将 {joint_name} 移动到 {angle_float} 度"
        except ValueError as e:
            self.logger.error(f"Invalid angle value: {e}")
            return f"无效的角度值：{angle}"
        except Exception as e:
            self.logger.error(f"Error moving joint {joint_name}: {e}")
            return f"控制关节失败：{str(e)}"

    @function_tool()
    async def get_joint_positions(
        self,
        context: RunContext,
    ) -> str:
        """
        获取所有关节的当前位置（角度）

        Returns:
            关节位置信息字符串
        """
        self.logger.debug("get_joint_positions called")
        try:
            positions = self.motors_service.get_joint_positions()
            if not positions:
                return "无法获取关节位置，请确保电机已连接"

            lines = [f"{name}: {pos:.1f}度" for name, pos in positions.items()]
            return "当前关节位置：\n" + "\n".join(lines)
        except Exception as e:
            self.logger.error(f"Error getting joint positions: {e}")
            return f"获取关节位置失败：{str(e)}"

    @function_tool()
    async def get_motor_health(
        self,
        context: RunContext,
        motor_name: Optional[str] = None,
    ) -> str:
        """
        获取舵机健康状态(温度、电压、负载等)

        Args:
            motor_name: 舵机名称,留空则返回所有舵机

        Returns:
            健康状态信息字符串
        """
        self.logger.debug(f"get_motor_health called with motor_name={motor_name}")
        try:
            # 验证舵机名称
            if motor_name and motor_name not in SAFE_JOINT_RANGES:
                valid_motors = ", ".join(SAFE_JOINT_RANGES.keys())
                return f"无效的舵机名称: {motor_name}。有效名称: {valid_motors}"

            # 获取健康摘要
            summary = self.motors_service.get_motor_health_summary()

            if "error" in summary:
                return "健康监控未启用。请在配置中设置 LELAMP_MOTOR_HEALTH_CHECK_ENABLED=true"

            # 状态表情符号映射
            status_emoji = {
                "healthy": "✅",
                "warning": "⚠️",
                "critical": "🔴",
                "stalled": "🚫",
            }

            if motor_name:
                # 返回单个舵机的详细信息
                motor_data = summary.get(motor_name)
                if not motor_data:
                    return f"未找到舵机 {motor_name} 的健康数据"

                latest = motor_data.get("latest")
                if not latest:
                    return f"舵机 {motor_name} 暂无健康数据(尚未检测)"

                result = f"舵机 {motor_name} 健康状态 {status_emoji.get(latest['status'], '❓')}:\n"
                result += f"- 状态: {latest['status']}\n"
                if latest.get("temperature"):
                    result += f"- 温度: {latest['temperature']:.1f}°C\n"
                if latest.get("voltage"):
                    result += f"- 电压: {latest['voltage']:.1f}V\n"
                if latest.get("load"):
                    result += f"- 负载: {latest['load'] * 100:.1f}%\n"
                if latest.get("position") is not None:
                    result += f"- 位置: {latest['position']:.1f}°\n"

                result += "\n统计信息:\n"
                result += f"- 警告次数: {motor_data['warning_count']}\n"
                result += f"- 危险次数: {motor_data['critical_count']}\n"
                result += f"- 堵转次数: {motor_data['stall_count']}\n"

                return result
            else:
                # 返回所有舵机的摘要
                result = "所有舵机健康状态:\n\n"
                for motor, data in summary.items():
                    latest = data.get("latest")
                    if latest:
                        result += f"{motor}: {status_emoji.get(latest['status'], '❓')} {latest['status']}"
                        if latest.get("temperature"):
                            result += f" (温度: {latest['temperature']:.1f}°C)"
                        result += "\n"
                    else:
                        result += f"{motor}: ❓ 暂无数据\n"

                return result
        except Exception as e:
            self.logger.error(f"Error getting motor health: {e}")
            return f"获取健康状态失败：{str(e)}"
