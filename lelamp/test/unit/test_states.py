"""
单元测试: lelamp.agent.states
"""
import pytest
import time
from lelamp.agent.states import ConversationState, StateColors, StateManager


class TestConversationState:
    """测试 ConversationState 枚举"""

    def test_state_values(self):
        """测试状态枚举值"""
        assert ConversationState.IDLE == "idle"
        assert ConversationState.LISTENING == "listening"
        assert ConversationState.THINKING == "thinking"
        assert ConversationState.SPEAKING == "speaking"

    def test_state_string_comparison(self):
        """测试状态枚举可以与字符串比较"""
        assert ConversationState.IDLE == "idle"
        assert "idle" == ConversationState.IDLE


class TestStateColors:
    """测试 StateColors 数据类"""

    def test_color_values(self):
        """测试颜色定义"""
        assert StateColors.IDLE == (255, 244, 229)
        assert StateColors.LISTENING == (0, 140, 255)
        assert StateColors.THINKING == (180, 0, 255)


class TestStateManager:
    """测试 StateManager 状态管理器"""

    def test_initial_state(self):
        """测试初始状态为 IDLE"""
        manager = StateManager()
        assert manager.current_state == ConversationState.IDLE

    def test_set_state(self):
        """测试设置状态"""
        manager = StateManager()

        manager.set_state(ConversationState.LISTENING)
        assert manager.current_state == ConversationState.LISTENING

        manager.set_state(ConversationState.THINKING)
        assert manager.current_state == ConversationState.THINKING

        manager.set_state(ConversationState.SPEAKING)
        assert manager.current_state == ConversationState.SPEAKING

    def test_motion_cooldown(self):
        """测试电机冷却时间"""
        cooldown = 0.2  # 200ms
        manager = StateManager(motion_cooldown_s=cooldown)

        # 初始状态允许执行
        assert manager.can_execute_motion() is True

        # 记录动作后立即检查（应被冷却阻止）
        manager.record_motion()
        assert manager.can_execute_motion() is False

        # 冷却期间仍然被阻止
        time.sleep(cooldown / 2)
        assert manager.can_execute_motion() is False

        # 冷却期结束后允许
        time.sleep(cooldown / 2 + 0.05)
        assert manager.can_execute_motion() is True

    def test_light_override(self):
        """测试灯光覆盖"""
        manager = StateManager()

        # 初始状态无覆盖
        assert manager.is_light_overridden() is False

        # 设置覆盖
        manager.set_light_override(duration_s=0.3)
        assert manager.is_light_overridden() is True

        # 覆盖期间持续有效
        time.sleep(0.1)
        assert manager.is_light_overridden() is True

        # 覆盖期结束后失效
        time.sleep(0.25)
        assert manager.is_light_overridden() is False

    def test_suppress_motion_after_light(self):
        """测试灯光命令后抑制动作"""
        suppress_duration = 0.3
        manager = StateManager(
            motion_cooldown_s=0.1,
            suppress_motion_after_light_s=suppress_duration
        )

        # 初始允许动作
        assert manager.can_execute_motion() is True

        # 设置灯光覆盖后，动作被抑制
        manager.set_light_override(duration_s=0.1)
        assert manager.can_execute_motion() is False

        # 抑制期间持续阻止
        time.sleep(suppress_duration / 2)
        assert manager.can_execute_motion() is False

        # 抑制期结束后恢复
        time.sleep(suppress_duration / 2 + 0.1)
        assert manager.can_execute_motion() is True

    def test_clear_light_override(self):
        """测试清除灯光覆盖"""
        manager = StateManager()

        # 设置覆盖
        manager.set_light_override(duration_s=10.0)
        assert manager.is_light_overridden() is True

        # 清除覆盖
        manager.clear_light_override()
        assert manager.is_light_overridden() is False

    def test_motion_cooldown_independent_of_suppress(self):
        """测试动作冷却与抑制期独立工作"""
        manager = StateManager(
            motion_cooldown_s=0.2,
            suppress_motion_after_light_s=0.1
        )

        # 记录动作
        manager.record_motion()
        assert manager.can_execute_motion() is False

        # 等待冷却结束
        time.sleep(0.25)
        assert manager.can_execute_motion() is True

        # 现在设置抑制（不影响已结束的冷却）
        manager.set_light_override(duration_s=0.05)
        assert manager.can_execute_motion() is False

        # 抑制期结束后恢复
        time.sleep(0.15)
        assert manager.can_execute_motion() is True

    def test_concurrent_state_access(self):
        """测试并发访问状态（基础线程安全验证）"""
        import threading

        manager = StateManager()
        states = [
            ConversationState.IDLE,
            ConversationState.LISTENING,
            ConversationState.THINKING,
            ConversationState.SPEAKING
        ]

        def worker():
            for state in states * 10:
                manager.set_state(state)
                _ = manager.current_state

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 验证状态为有效值
        assert manager.current_state in states

    def test_concurrent_motion_checks(self):
        """测试并发检查电机动作（基础线程安全验证）"""
        import threading

        manager = StateManager(motion_cooldown_s=0.1)
        results = []

        def worker():
            for _ in range(20):
                can_move = manager.can_execute_motion()
                results.append(can_move)
                if can_move:
                    manager.record_motion()
                time.sleep(0.01)

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 验证至少有一些动作被允许，一些被阻止
        assert True in results
        assert False in results
