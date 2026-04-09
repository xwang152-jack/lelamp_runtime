"""
系统工具模块

包含电机、RGB、系统控制和 OTA 更新相关的工具方法。
"""

import asyncio
import json
import logging
import os
import socket
import sys
import threading
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any, Optional

from livekit.agents import function_tool, RunContext

# 从 motor_tools.py 导入 SAFE_JOINT_RANGES
from lelamp.agent.tools.motor_tools import SAFE_JOINT_RANGES
from .utils import validate_multiple_rgb_colors

if TYPE_CHECKING:
    from lelamp.service.motors.motors_service import MotorsService
    from lelamp.service.rgb.rgb_service import RGBService
    from lelamp.agent.states import StateManager
    from lelamp.utils.ota import OTAManager


class SystemTools:
    """系统工具类 - 管理电机高级功能、RGB 效果扩展、系统控制和 OTA 更新"""

    # 类常量：魔法数字
    RESTART_DELAY_S = 5
    SEARCH_QUERY_MAX_LENGTH = 500
    BRIGHTNESS_OVERRIDE_DURATION_S = 10.0
    LIGHT_OVERRIDE_DURATION_S = 10.0

    def __init__(
        self,
        motors_service: "MotorsService",
        rgb_service: "RGBService",
        ota_manager: "OTAManager | None",
        ota_url: Optional[str],
        state_manager: "StateManager",
        get_rate_limit_stats_func,
    ):
        """
        初始化系统工具

        Args:
            motors_service: 电机服务实例
            rgb_service: RGB 服务实例
            ota_manager: OTA 管理器实例
            ota_url: OTA 更新服务器地址
            state_manager: 状态管理器实例
            get_rate_limit_stats_func: 获取速率限制统计的函数
        """
        self.motors_service = motors_service
        self.rgb_service = rgb_service
        self._ota_manager = ota_manager
        self._ota_url = ota_url
        self.state_manager = state_manager
        self._get_rate_limit_stats = get_rate_limit_stats_func
        self.logger = logging.getLogger("lelamp.agent.tools.system")

        # 后台任务追踪集合
        self._tasks: set[asyncio.Task] = set()

        # 用于运行 amixer 的用户名（可配置）
        self._amixer_user = os.getenv("LELAMP_AMIXER_USER", "pi")

    # ==================== 电机相关工具 ====================

    @function_tool()
    async def get_available_recordings(
        self,
        context: RunContext,
    ) -> str:
        """
        Discover your physical expressions! Get your repertoire of motor movements for body language.
        获取可用的录制动作列表。
        """
        self.logger.debug("get_available_recordings called")
        try:
            recordings = self.motors_service.get_available_recordings()
            if recordings:
                return f"Available recordings: {', '.join(recordings)}"
            return "No recordings found."
        except Exception as e:
            return f"Error getting recordings: {str(e)}"

    @function_tool()
    async def tune_motor_pid(
        self,
        context: RunContext,
        motor_name: str,
        p_coefficient: int,
        i_coefficient: int = 0,
        d_coefficient: int = 32,
    ) -> str:
        """
        远程调整舵机 PID 参数(商用功能,用于优化动作性能)。
        Tune motor PID coefficients remotely (commercial feature for performance optimization).

        Args:
            motor_name: 舵机名称(base_yaw/base_pitch/elbow_pitch/wrist_roll/wrist_pitch)
            p_coefficient: P 系数(比例增益,默认 16,范围 1-32,越大响应越快但可能抖动)
            i_coefficient: I 系数(积分增益,默认 0,范围 0-32)
            d_coefficient: D 系数(微分增益,默认 32,范围 0-32,用于减少超调)

        注意: 不当的 PID 参数可能导致舵机抖动或无法稳定,请谨慎调整!
        """
        if motor_name not in SAFE_JOINT_RANGES:
            return f"无效的舵机名称: {motor_name}。有效名称: {', '.join(SAFE_JOINT_RANGES.keys())}"

        # 参数范围验证
        if not (1 <= p_coefficient <= 32):
            return "P 系数必须在 1-32 之间"
        if not (0 <= i_coefficient <= 32):
            return "I 系数必须在 0-32 之间"
        if not (0 <= d_coefficient <= 32):
            return "D 系数必须在 0-32 之间"

        if not self.motors_service.robot:
            return "舵机服务未连接"

        try:
            bus = self.motors_service.robot.bus

            # 写入 PID 参数
            bus.write("P_Coefficient", motor_name, p_coefficient)
            bus.write("I_Coefficient", motor_name, i_coefficient)
            bus.write("D_Coefficient", motor_name, d_coefficient)

            self.logger.info(
                f"Updated PID for {motor_name}: P={p_coefficient}, I={i_coefficient}, D={d_coefficient}"
            )

            return f"成功更新舵机 {motor_name} 的 PID 参数:\n- P: {p_coefficient}\n- I: {i_coefficient}\n- D: {d_coefficient}\n\n请测试动作是否稳定,如有问题可恢复默认值(P=16, I=0, D=32)"

        except Exception as e:
            self.logger.error(f"Failed to tune PID for {motor_name}: {e}")
            return f"更新 PID 参数失败: {str(e)}"

    @function_tool()
    async def reset_motor_health_stats(
        self,
        context: RunContext,
        motor_name: Optional[str] = None,
    ) -> str:
        """
        重置舵机健康统计数据(警告/危险/堵转计数)。
        Reset motor health statistics (warning/critical/stall counts).

        Args:
            motor_name: 舵机名称,留空则重置所有舵机
        """
        if motor_name and motor_name not in SAFE_JOINT_RANGES:
            return f"无效的舵机名称: {motor_name}"

        self.motors_service.reset_health_statistics(motor_name)

        if motor_name:
            return f"已重置舵机 {motor_name} 的健康统计数据"
        else:
            return "已重置所有舵机的健康统计数据"

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

    # ==================== RGB 效果扩展 ====================

    @function_tool()
    async def set_rgb_brightness(
        self,
        context: RunContext,
        percent: int,
    ) -> str:
        """
        调节灯光亮度（0-100）
        Adjust lamp brightness (0-100).
        """
        self.logger.debug(f"set_rgb_brightness called with percent: {percent}")
        try:
            p = int(percent)
        except Exception:
            return "亮度参数无效，请输入 0-100 的整数"
        if p < 0 or p > 100:
            return "亮度范围是 0-100"

        from lelamp.service import Priority

        v = int(round(p * 255 / 100))
        self.rgb_service.dispatch("brightness", v, priority=Priority.HIGH)
        return f"已将灯光亮度设置为 {p}%"

    @function_tool()
    async def rgb_effect_wave(
        self,
        context: RunContext,
        red: int = 60,
        green: int = 180,
        blue: int = 255,
        speed: float = 1.0,
        freq: float = 1.2,
        fps: int = 30,
    ) -> str:
        """
        波纹/呼吸波动效果（8x8 矩阵）
        Wave/breathing effect on the 8x8 LED matrix.
        """
        self.logger.debug(
            f"rgb_effect_wave called with rgb=({red},{green},{blue}), speed={speed}, freq={freq}, fps={fps}"
        )
        try:
            is_valid, error_msg = validate_multiple_rgb_colors(red, green, blue)
            if not is_valid:
                return error_msg

            # 设置灯光覆盖
            self.state_manager.set_light_override(
                duration_s=self.LIGHT_OVERRIDE_DURATION_S
            )

            from lelamp.service import Priority

            self.rgb_service.dispatch(
                "effect",
                {
                    "name": "wave",
                    "color": (int(red), int(green), int(blue)),
                    "speed": float(speed),
                    "freq": float(freq),
                    "fps": int(fps),
                },
                priority=Priority.HIGH,
            )
            return "已开启波纹动态灯效"
        except Exception as e:
            return f"开启波纹灯效失败：{str(e)}"

    @function_tool()
    async def rgb_effect_fire(
        self,
        context: RunContext,
        intensity: float = 1.0,
        fps: int = 30,
    ) -> str:
        """
        火焰动态效果（8x8 矩阵）
        Fire animation effect on the 8x8 LED matrix.
        """
        self.logger.debug(
            f"rgb_effect_fire called with intensity={intensity}, fps={fps}"
        )
        try:
            # 设置灯光覆盖
            self.state_manager.set_light_override(
                duration_s=self.LIGHT_OVERRIDE_DURATION_S
            )

            from lelamp.service import Priority

            self.rgb_service.dispatch(
                "effect",
                {"name": "fire", "intensity": float(intensity), "fps": int(fps)},
                priority=Priority.HIGH,
            )
            return "已开启火焰动态灯效"
        except Exception as e:
            return f"开启火焰灯效失败：{str(e)}"

    @function_tool()
    async def rgb_effect_emoji(
        self,
        context: RunContext,
        emoji: str = "smile",
        red: int = 255,
        green: int = 200,
        blue: int = 60,
        bg_red: int = 0,
        bg_green: int = 0,
        bg_blue: int = 0,
        blink: bool = True,
        period_s: float = 2.2,
        fps: int = 30,
    ) -> str:
        """
        表情动画（smile/sad/wink/angry/heart）
        Emoji animation on the LED matrix.
        """
        self.logger.debug(
            f"rgb_effect_emoji called with emoji={emoji}, fg=({red},{green},{blue}), bg=({bg_red},{bg_green},{bg_blue}), blink={blink}, period_s={period_s}, fps={fps}"
        )
        try:
            is_valid, error_msg = validate_multiple_rgb_colors(
                red, green, blue, bg_red, bg_green, bg_blue
            )
            if not is_valid:
                return error_msg

            # 设置灯光覆盖
            self.state_manager.set_light_override(
                duration_s=self.LIGHT_OVERRIDE_DURATION_S
            )

            from lelamp.service import Priority

            self.rgb_service.dispatch(
                "effect",
                {
                    "name": "emoji",
                    "emoji": str(emoji or "smile").strip().lower(),
                    "color": (int(red), int(green), int(blue)),
                    "bg": (int(bg_red), int(bg_green), int(bg_blue)),
                    "blink": bool(blink),
                    "period_s": float(period_s),
                    "fps": int(fps),
                },
                priority=Priority.HIGH,
            )
            return f"已开启表情动画：{str(emoji or 'smile').strip().lower()}"
        except Exception as e:
            return f"开启表情动画失败：{str(e)}"

    @function_tool()
    async def stop_rgb_effect(
        self,
        context: RunContext,
    ) -> str:
        """
        停止动态特效/表情动画
        Stop all running RGB effects/animations.
        """
        self.logger.debug("stop_rgb_effect called")
        # 设置灯光覆盖
        self.state_manager.set_light_override(duration_s=self.LIGHT_OVERRIDE_DURATION_S)

        from lelamp.service import Priority

        self.rgb_service.dispatch("effect_stop", None, priority=Priority.HIGH)
        return "已停止动态灯效"

    # ==================== 系统控制 ====================

    @function_tool()
    async def set_volume(
        self,
        context: RunContext,
        volume_percent: int,
    ) -> str:
        """
        Control system audio volume.
        控制系统音量（0-100）。
        """
        self.logger.debug(f"set_volume called with volume: {volume_percent}%")
        try:
            if not 0 <= volume_percent <= 100:
                return "Error: Volume must be between 0 and 100 percent"

            # 使用 amixer 分别设置三个音频输出的音量
            for control in ("Line", "Line DAC", "HP"):
                process = await asyncio.create_subprocess_exec(
                    "sudo",
                    "-u",
                    self._amixer_user,
                    "amixer",
                    "-q",
                    "sset",
                    control,
                    f"{volume_percent}%",
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await process.wait()

            return f"已将音量设置为 {volume_percent}%"
        except Exception as e:
            return f"Error controlling volume: {str(e)}"

    @function_tool()
    async def get_rate_limit_stats(
        self,
        context: RunContext,
    ) -> str:
        """
        获取 API 速率限制统计信息（调试用）
        Get API rate limiting statistics (for debugging).
        """
        stats = self._get_rate_limit_stats()
        lines = ["API 速率限制统计："]
        for name, stat in stats.items():
            lines.append(f"\n{name}:")
            lines.append(f"  请求总数: {stat['requests_total']}")
            lines.append(f"  允许: {stat['requests_allowed']}")
            lines.append(f"  拒绝: {stat['requests_denied']}")
            lines.append(f"  拒绝率: {stat['denial_rate']:.1%}")
            lines.append(f"  平均等待: {stat['avg_wait_time']:.2f}s")
        return "\n".join(lines)

    @function_tool()
    async def web_search(
        self,
        context: RunContext,
        query: str,
    ) -> str:
        """
        当用户问到实时信息、新闻、天气或你不确定的知识时，使用此工具在线搜索。
        Get real-time information from the web.

        Args:
            query: 搜索关键词 (Search query)
        """
        # 输入验证
        if not query or not query.strip():
            return "搜索关键词不能为空"

        if len(query) > self.SEARCH_QUERY_MAX_LENGTH:
            return f"搜索关键词过长，请限制在{self.SEARCH_QUERY_MAX_LENGTH}字以内"

        api_key = os.getenv("BOCHA_API_KEY")
        if not api_key:
            return "未配置 BOCHA_API_KEY，无法进行联网搜索。"

        self.logger.info(f"正在通过博查搜索: {query}")
        url = "https://api.bochaai.com/v1/web-search"

        # 导入 URL 验证函数
        from lelamp.utils.url_validation import (
            validate_external_url,
            ALLOWED_API_DOMAINS,
        )

        # URL 安全验证
        if not validate_external_url(url, ALLOWED_API_DOMAINS):
            self.logger.error(f"搜索 API URL 验证失败: {url}")
            return "搜索服务配置错误"

        payload = json.dumps(
            {"query": query.strip(), "freshness": "oneDay", "summary": True}
        ).encode("utf-8")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        def _call() -> dict:
            req = urllib.request.Request(
                url=url, data=payload, headers=headers, method="POST"
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read()
            return json.loads(raw.decode("utf-8"))

        try:
            data = await asyncio.to_thread(_call)
            if data.get("code") != 200:
                return f"搜索失败: {data.get('msg')}"

            data_content = data.get("data", {})
            web_pages = data_content.get("webPages", {}).get("value", [])

            if not web_pages:
                return "没有找到相关的联网搜索结果。"

            results = []
            for page in web_pages[:3]:  # 取前3个结果
                title = page.get("name")
                snippet = page.get("snippet")
                results.append(f"- {title}: {snippet}")

            return "以下是联网搜索到的信息：\n" + "\n".join(results)

        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            self.logger.error(f"Network error during web search: {e}")
            return "网络连接失败，请稍后重试"
        except (socket.timeout, asyncio.TimeoutError) as e:
            self.logger.error(f"Timeout during web search: {e}")
            return "搜索请求超时，请稍后重试"
        except (json.JSONDecodeError, KeyError) as e:
            self.logger.error(f"Data parsing error during web search: {e}")
            return "搜索结果解析失败"
        except Exception as e:
            self.logger.error(f"Unexpected error during web search: {e}")
            return f"联网搜索发生异常: {str(e)}"

    # ==================== OTA 更新 ====================

    @function_tool()
    async def check_for_updates(
        self,
        context: RunContext,
    ) -> str:
        """
        检查系统是否有新的 OTA 更新。
        Check for system updates.
        """
        if not self._ota_url:
            return "OTA 更新服务未配置 (LELAMP_OTA_URL missing)。"

        has_update, version, notes = self._ota_manager.check_for_update()
        if has_update:
            return f"发现新版本 {version}！\n更新内容：{notes}\n请问是否需要现在更新？(请回复'确认更新')"
        return f"当前已是最新版本 ({version})。"

    @function_tool()
    async def perform_ota_update(
        self,
        context: RunContext,
    ) -> str:
        """
        执行系统更新 (OTA)。注意：更新成功后服务将重启。
        Perform system update. Note: Service will restart upon success.
        """
        if not self._ota_url:
            return "OTA 更新服务未配置。"

        # Double check
        has_update, version, _ = self._ota_manager.check_for_update()
        if not has_update:
            return "当前没有可用更新。"

        result = self._ota_manager.perform_update()
        if "更新成功" in result:
            # Schedule a restart
            task = asyncio.create_task(self._restart_later())
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            return f"{result} 服务将在 {self.RESTART_DELAY_S} 秒后重启。"
        return f"更新失败：{result}"

    async def _restart_later(self):
        """延迟重启服务"""
        try:
            await asyncio.sleep(self.RESTART_DELAY_S)
            self.logger.info("Triggering restart...")
            # Rely on systemd or Docker to restart the container/process
            sys.exit(0)
        except Exception as e:
            self.logger.error(f"Error during restart delay: {e}")
