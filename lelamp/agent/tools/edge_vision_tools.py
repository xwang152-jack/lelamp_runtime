"""
边缘视觉工具类

为 LeLamp Agent 提供本地 AI 推理功能：
- 快速物体识别
- 手势控制
- 用户在场检测
"""

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from lelamp.edge.hybrid_vision import HybridVisionService
    from lelamp.agent.states import StateManager

logger = logging.getLogger("lelamp")


class EdgeVisionTools:
    """
    边缘视觉工具类

    提供本地 AI 推理功能，减少云端 API 调用。
    """

    def __init__(
        self,
        hybrid_vision: "HybridVisionService | None" = None,
        state_manager: "StateManager | None" = None,
    ):
        """
        初始化边缘视觉工具

        Args:
            hybrid_vision: 混合视觉服务实例
            state_manager: 状态管理器实例
        """
        self._hybrid_vision = hybrid_vision
        self._state_manager = state_manager

    async def quick_identify(self, frame=None) -> str:
        """
        快速识别当前画面中的物体（本地推理）

        Args:
            frame: 图像帧（可选，如果不提供则从 vision_service 获取）

        Returns:
            识别结果描述
        """
        if self._hybrid_vision is None:
            return "边缘视觉服务未启用"

        try:
            result = await self._hybrid_vision.answer("这是什么", frame)

            if result.source == "local":
                return f"[本地识别] {result.answer}"
            else:
                return f"[云端识别] {result.answer}"

        except Exception as e:
            logger.error(f"Quick identify error: {e}")
            return f"识别失败: {str(e)}"

    async def detect_gesture(self, frame=None) -> str:
        """
        检测手势（本地推理）- 增强版，带LED和动作反馈

        Args:
            frame: 图像帧

        Returns:
            检测到的手势
        """
        if self._hybrid_vision is None:
            return "手势检测服务未启用"

        try:
            result = self._hybrid_vision.track_hands(frame)

            if result["gestures"]:
                gestures = [g.value for g in result["gestures"]]
                gesture_names = ", ".join(gestures)

                # 触发手势回调（如果有）
                for gesture in result["gestures"]:
                    if (
                        hasattr(self._hybrid_vision, "gesture_callback")
                        and self._hybrid_vision.gesture_callback
                    ):
                        try:
                            self._hybrid_vision.gesture_callback(gesture, {})
                        except Exception as callback_error:
                            logger.error(f"Gesture callback error: {callback_error}")

                return f"检测到手势: {gesture_names}"
            else:
                return "未检测到手势"

        except Exception as e:
            logger.error(f"Gesture detection error: {e}")
            return f"手势检测失败: {str(e)}"

    async def check_presence(self, frame=None) -> str:
        """
        检测用户是否在场（本地推理）

        Args:
            frame: 图像帧

        Returns:
            在场状态描述
        """
        if self._hybrid_vision is None:
            return "在场检测服务未启用"

        try:
            result = self._hybrid_vision.detect_faces(frame)

            if result["presence"]:
                count = result["count"]
                duration = result.get("presence_duration", 0)
                return f"用户在场 (检测到 {count} 人，持续 {duration:.0f} 秒)"
            else:
                return "用户不在场"

        except Exception as e:
            logger.error(f"Presence detection error: {e}")
            return f"在场检测失败: {str(e)}"

    def get_stats(self) -> dict:
        """
        获取边缘视觉统计信息

        Returns:
            统计信息字典
        """
        if self._hybrid_vision is None:
            return {"enabled": False}

        return {"enabled": True, **self._hybrid_vision.get_stats()}

    async def quick_check(self, frame=None) -> str:
        """
        快速检查 - 同时检测用户在场和手势

        Args:
            frame: 图像帧（可选）

        Returns:
            综合检查结果
        """
        if self._hybrid_vision is None:
            return "边缘视觉服务未启用"

        try:
            results = []

            # 1. 检测用户在场
            presence_result = self._hybrid_vision.detect_faces(frame)
            if presence_result.get("presence"):
                count = presence_result.get("count", 0)
                results.append(f"用户在场 (检测到 {count} 人)")
            else:
                results.append("用户不在场")

            # 2. 检测手势
            gesture_result = self._hybrid_vision.track_hands(frame)
            if gesture_result.get("gestures"):
                gestures = [g.value for g in gesture_result["gestures"]]
                results.append(f"检测到手势: {', '.join(gestures)}")

                # 触发手势回调
                for gesture in gesture_result["gestures"]:
                    if (
                        hasattr(self._hybrid_vision, "gesture_callback")
                        and self._hybrid_vision.gesture_callback
                    ):
                        try:
                            self._hybrid_vision.gesture_callback(gesture, {})
                        except Exception as callback_error:
                            logger.error(f"Gesture callback error: {callback_error}")

            return " | ".join(results) if results else "一切正常"

        except Exception as e:
            logger.error(f"Quick check error: {e}")
            return f"检查失败: {str(e)}"
