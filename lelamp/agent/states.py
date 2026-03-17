"""
会话状态管理
"""
from enum import Enum
from dataclasses import dataclass
from typing import Optional
import time
import threading


class ConversationState(str, Enum):
    """会话状态枚举"""
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


@dataclass
class StateColors:
    """状态对应的 RGB 颜色"""
    IDLE = (255, 244, 229)      # 暖白光
    LISTENING = (0, 140, 255)   # 蓝色
    THINKING = (180, 0, 255)    # 紫色
    # SPEAKING 使用随机动画颜色


class StateManager:
    """状态管理器 - 管理会话状态、冷却时间和覆盖逻辑"""

    def __init__(
        self,
        motion_cooldown_s: float = 3.0,
        suppress_motion_after_light_s: float = 5.0,
    ):
        self._current_state = ConversationState.IDLE
        self._motion_cooldown_s = motion_cooldown_s
        self._suppress_motion_after_light_s = suppress_motion_after_light_s

        # 时间戳追踪
        self._last_motion_ts: Optional[float] = None
        self._light_override_until_ts: Optional[float] = None
        self._suppress_motion_until_ts: Optional[float] = None

        # 线程锁
        self._state_lock = threading.Lock()
        self._timestamps_lock = threading.Lock()

    @property
    def current_state(self) -> ConversationState:
        """获取当前状态"""
        with self._state_lock:
            return self._current_state

    def set_state(self, state: ConversationState) -> None:
        """设置当前状态"""
        with self._state_lock:
            self._current_state = state

    def can_execute_motion(self) -> bool:
        """检查是否允许执行电机动作"""
        with self._timestamps_lock:
            now = time.time()

            # 检查是否在抑制期内
            if (self._suppress_motion_until_ts is not None and
                now < self._suppress_motion_until_ts):
                return False

            # 检查冷却时间
            if (self._last_motion_ts is not None and
                (now - self._last_motion_ts) < self._motion_cooldown_s):
                return False

            return True

    def record_motion(self) -> None:
        """记录电机动作时间"""
        with self._timestamps_lock:
            self._last_motion_ts = time.time()

    def is_light_overridden(self) -> bool:
        """检查灯光是否被手动覆盖"""
        with self._timestamps_lock:
            if self._light_override_until_ts is None:
                return False
            return time.time() < self._light_override_until_ts

    def set_light_override(self, duration_s: float) -> None:
        """设置灯光覆盖"""
        with self._timestamps_lock:
            self._light_override_until_ts = time.time() + duration_s
            self._suppress_motion_until_ts = time.time() + self._suppress_motion_after_light_s

    def clear_light_override(self) -> None:
        """清除灯光覆盖"""
        with self._timestamps_lock:
            self._light_override_until_ts = None
