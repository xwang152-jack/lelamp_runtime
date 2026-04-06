"""
LiveKit Agents Testing Framework 集成测试

使用 livekit.agents.testing 和 livekit.agents.inference
进行行为测试和评估。

参考: https://docs.livekit.io/agents/start/testing/
"""
import pytest
from unittest.mock import Mock, MagicMock

# LiveKit Agents SDK imports
try:
    from livekit.agents import AgentSession, inference, mock_tools, RunContext

    LIVEKIT_AVAILABLE = True
except ImportError:
    LIVEKIT_AVAILABLE = False

from lelamp.agent.lelamp_agent import LeLamp
from lelamp.agent.states import ConversationState, StateManager


pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_motors_service():
    """Mock MotorsService with all required attributes"""
    mock = Mock()
    mock.is_running = True
    mock.has_pending_event.return_value = False
    mock.get_available_recordings.return_value = ["nod", "shake", "wake_up"]
    mock.get_joint_positions.return_value = {
        "base_yaw": 0.0,
        "base_pitch": 0.0,
    }
    mock.get_motor_health_summary.return_value = {}
    return mock


@pytest.fixture
def mock_rgb_service():
    """Mock RGBService with all required attributes"""
    mock = Mock()
    mock.is_running = True
    mock.is_on.return_value = True
    return mock


@pytest.fixture
def mock_vision_service():
    """Mock VisionService"""
    mock = Mock()
    mock.is_running = True
    return mock


@pytest.fixture
def mock_qwen_client():
    """Mock Qwen VL Client"""
    mock = Mock()
    return mock


@pytest.fixture
def mock_rate_limiter():
    """Mock rate limiter that always allows"""
    mock = Mock()
    mock.acquire = Mock(return_value=True)
    return mock


class TestLeLampTools:
    """测试 LeLamp Agent 的工具方法"""

    async def test_play_recording_delegates_correctly(self, mock_motors_service, mock_rgb_service):
        """测试 play_recording 正确委托给 MotorTools"""
        from unittest.mock import AsyncMock, MagicMock

        # Create mock context
        mock_ctx = MagicMock(spec=RunContext)

        # Create mock for MotorTools
        motor_tools = Mock()
        motor_tools.play_recording = AsyncMock(return_value="开始执行动作：nod")

        # Create LeLamp agent with mocked tools
        agent = LeLamp(
            port="/dev/ttyACM0",
            lamp_id="test_lamp",
            motors_service=mock_motors_service,
            rgb_service=mock_rgb_service,
            vision_service=None,
            qwen_client=None,
        )
        agent._motor_tools = motor_tools

        # Call the method
        result = await agent.play_recording(mock_ctx, "nod")

        # Verify
        assert result == "开始执行动作：nod"
        motor_tools.play_recording.assert_called_once()

    async def test_set_rgb_solid_delegates_correctly(self, mock_motors_service, mock_rgb_service):
        """测试 set_rgb_solid 正确委托给 RGBTools"""
        from unittest.mock import AsyncMock, MagicMock

        mock_ctx = MagicMock(spec=RunContext)

        rgb_tools = Mock()
        rgb_tools.set_rgb_solid = AsyncMock(return_value="设置纯色灯光: RGB(255, 0, 0)")

        agent = LeLamp(
            port="/dev/ttyACM0",
            lamp_id="test_lamp",
            motors_service=mock_motors_service,
            rgb_service=mock_rgb_service,
            vision_service=None,
            qwen_client=None,
        )
        agent._rgb_tools = rgb_tools

        result = await agent.set_rgb_solid(mock_ctx, 255, 0, 0)

        assert result == "设置纯色灯光: RGB(255, 0, 0)"
        rgb_tools.set_rgb_solid.assert_called_once()

    async def test_get_joint_positions_returns_positions(self, mock_motors_service, mock_rgb_service):
        """测试 get_joint_positions 返回关节位置"""
        from unittest.mock import AsyncMock, MagicMock

        mock_ctx = MagicMock(spec=RunContext)

        motor_tools = Mock()
        motor_tools.get_joint_positions = AsyncMock(
            return_value="当前关节位置：\nbase_yaw: 0.0度"
        )

        agent = LeLamp(
            port="/dev/ttyACM0",
            lamp_id="test_lamp",
            motors_service=mock_motors_service,
            rgb_service=mock_rgb_service,
            vision_service=None,
            qwen_client=None,
        )
        agent._motor_tools = motor_tools

        result = await agent.get_joint_positions(mock_ctx)

        assert "base_yaw" in result
        motor_tools.get_joint_positions.assert_called_once()


class TestLeLampStateManagement:
    """测试 LeLamp 状态管理"""

    def test_state_manager_initial_state(self):
        """测试 StateManager 初始状态"""
        manager = StateManager()
        assert manager.current_state == ConversationState.IDLE

    def test_state_manager_set_state(self):
        """测试设置状态"""
        manager = StateManager()
        manager.set_state(ConversationState.LISTENING)
        assert manager.current_state == ConversationState.LISTENING

    def test_state_manager_motion_cooldown(self):
        """测试动作冷却机制"""
        import time

        manager = StateManager(motion_cooldown_s=0.5)

        # 初始应该允许
        assert manager.can_execute_motion() is True

        # 记录动作
        manager.record_motion()
        assert manager.can_execute_motion() is False

        # 等待冷却
        time.sleep(0.6)
        assert manager.can_execute_motion() is True


class TestLeLampToolTimeout:
    """测试 LeLamp 工具超时机制"""

    async def test_tool_with_timeout_success(self, mock_motors_service, mock_rgb_service):
        """测试 _tool_with_timeout 成功情况"""
        import asyncio

        agent = LeLamp(
            port="/dev/ttyACM0",
            lamp_id="test_lamp",
            motors_service=mock_motors_service,
            rgb_service=mock_rgb_service,
            vision_service=None,
            qwen_client=None,
        )

        async def quick_task():
            return "success"

        result = await agent._tool_with_timeout(quick_task(), timeout_seconds=5.0)
        assert result == "success"

    async def test_tool_with_timeout_exceeds_limit(self, mock_motors_service, mock_rgb_service):
        """测试 _tool_with_timeout 超时情况"""
        import asyncio

        agent = LeLamp(
            port="/dev/ttyACM0",
            lamp_id="test_lamp",
            motors_service=mock_motors_service,
            rgb_service=mock_rgb_service,
            vision_service=None,
            qwen_client=None,
        )

        async def slow_task():
            await asyncio.sleep(10)  # 10 second task
            return "success"

        result = await agent._tool_with_timeout(
            slow_task(),
            timeout_seconds=0.1,
            error_message="操作超时，请稍后重试",
        )
        assert result == "操作超时，请稍后重试"


class TestLeLampBackgroundTasks:
    """测试 LeLamp 后台任务追踪"""

    async def test_track_task_creates_tracked_task(self, mock_motors_service, mock_rgb_service):
        """测试 _track_task 正确追踪任务"""
        import asyncio

        agent = LeLamp(
            port="/dev/ttyACM0",
            lamp_id="test_lamp",
            motors_service=mock_motors_service,
            rgb_service=mock_rgb_service,
            vision_service=None,
            qwen_client=None,
        )

        initial_task_count = len(agent._background_tasks)

        async def background_job():
            await asyncio.sleep(0.1)
            return "done"

        task = agent._track_task(background_job())

        # Task should be tracked
        assert len(agent._background_tasks) == initial_task_count + 1
        assert task in agent._background_tasks

        # Wait for task to complete
        await asyncio.sleep(0.2)

        # Task should be removed from tracking after completion
        # Note: Due to the done_callback, it may be removed automatically

    async def test_shutdown_cancels_background_tasks(
        self, mock_motors_service, mock_rgb_service
    ):
        """测试 shutdown 取消后台任务"""
        import asyncio

        agent = LeLamp(
            port="/dev/ttyACM0",
            lamp_id="test_lamp",
            motors_service=mock_motors_service,
            rgb_service=mock_rgb_service,
            vision_service=None,
            qwen_client=None,
        )

        async def long_running_job():
            await asyncio.sleep(100)
            return "done"

        # Start a long-running background task
        agent._track_task(long_running_job())

        assert len(agent._background_tasks) > 0

        # Shutdown should cancel tasks
        agent.shutdown()

        # Give the event loop a chance to process the cancellation
        await asyncio.sleep(0)

        # Verify tasks were cancelled - done() returns True for cancelled tasks
        for task in agent._background_tasks:
            assert task.done()


# Skip tests if LiveKit Agents SDK is not available
if not LIVEKIT_AVAILABLE:

    @pytest.fixture
    def skip_if_no_livekit():
        pytest.skip("LiveKit Agents SDK not available")
