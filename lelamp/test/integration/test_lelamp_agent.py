"""
LeLamp Agent 集成测试

测试 LeLamp 类的核心功能：
- 状态管理
- 用户输入记录
- Data Channel 消息处理
- 服务集成
"""
import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from lelamp.agent.lelamp_agent import LeLamp
from lelamp.agent.states import ConversationState, StateColors


# Fixtures
@pytest.fixture
def mock_motors_service():
    """Mock MotorsService"""
    service = MagicMock()
    service.start = MagicMock()
    service.dispatch = MagicMock()
    service.get_available_recordings = MagicMock(return_value=["wake_up", "nod", "shake"])
    service.get_joint_positions = MagicMock(return_value={"base_yaw": 0.0, "base_pitch": 0.0})
    service.get_motor_health_summary = MagicMock(return_value={})
    service.reset_health_statistics = MagicMock()
    return service


@pytest.fixture
def mock_rgb_service():
    """Mock RGBService"""
    service = MagicMock()
    service.start = MagicMock()
    service.dispatch = MagicMock()
    return service


@pytest.fixture
def mock_vision_service():
    """Mock VisionService"""
    service = MagicMock()
    service.get_latest_jpeg_b64 = AsyncMock(return_value=(b"fake_jpeg_data", time.time()))
    service.get_fresh_jpeg_b64 = AsyncMock(return_value=(b"fake_jpeg_data", time.time()))
    service.get_latest_frame = MagicMock(return_value=b"fake_frame_data")
    return service


@pytest.fixture
def mock_qwen_client():
    """Mock Qwen3VLClient"""
    client = MagicMock()
    client.describe = AsyncMock(return_value="这是测试图片描述")
    return client


@pytest.fixture
def mock_ota_manager():
    """Mock OTAManager"""
    manager = MagicMock()
    manager.check_for_update = MagicMock(return_value=(False, "1.0.0", ""))
    manager.perform_update = MagicMock(return_value="更新成功")
    return manager


@pytest.fixture
def mock_rate_limiter():
    """Mock rate limiter"""
    limiter = MagicMock()
    limiter.acquire = AsyncMock(return_value=True)
    return limiter


@pytest.fixture
def lelamp_agent(
    mock_motors_service,
    mock_rgb_service,
    mock_vision_service,
    mock_qwen_client,
    mock_ota_manager,
    mock_rate_limiter,
):
    """创建 LeLamp 实例用于测试"""
    # Patch get_rate_limiter (从 utils 模块导入)
    with patch("lelamp.utils.rate_limiter.get_rate_limiter", return_value=mock_rate_limiter):
        # Patch get_ota_manager (从 utils.ota 模块导入)
        with patch("lelamp.utils.ota.get_ota_manager", return_value=mock_ota_manager):
            # Patch get_all_rate_limiter_stats
            with patch("lelamp.utils.rate_limiter.get_all_rate_limiter_stats", return_value={}):
                agent = LeLamp(
                    port="/dev/ttyACM0",
                    lamp_id="test_lamp",
                    vision_service=mock_vision_service,
                    qwen_client=mock_qwen_client,
                    ota_url="http://test.com/ota",
                    motors_service=mock_motors_service,
                    rgb_service=mock_rgb_service,
                )
                yield agent


class TestLeLampAgentInit:
    """测试 LeLamp 初始化"""

    def test_services_initialized(self, lelamp_agent):
        """测试服务正确初始化"""
        assert lelamp_agent.motors_service is not None
        assert lelamp_agent.rgb_service is not None
        assert lelamp_agent._vision_service is not None
        assert lelamp_agent._qwen_client is not None

    def test_services_started(self, lelamp_agent, mock_motors_service, mock_rgb_service):
        """测试服务已启动"""
        mock_motors_service.start.assert_called_once()
        mock_rgb_service.start.assert_called_once()

    def test_boot_animation(self, lelamp_agent, mock_motors_service, mock_rgb_service):
        """测试启动动画"""
        # wake_up 动画应该被播放
        mock_motors_service.dispatch.assert_any_call("play", "wake_up")
        # 白灯应该被设置
        mock_rgb_service.dispatch.assert_any_call("solid", (255, 255, 255))

    def test_state_manager_initialized(self, lelamp_agent):
        """测试状态管理器初始化"""
        assert lelamp_agent._state_manager is not None
        assert lelamp_agent._state_manager.current_state == ConversationState.IDLE


class TestNoteUserText:
    """测试用户输入记录"""

    @pytest.mark.asyncio
    async def test_note_user_text(self, lelamp_agent):
        """测试记录用户输入"""
        test_text = "测试用户输入"
        await lelamp_agent.note_user_text(test_text)

        assert lelamp_agent._last_user_text == test_text
        assert lelamp_agent._last_user_text_ts > 0


class TestSetConversationState:
    """测试会话状态设置"""

    @pytest.mark.asyncio
    async def test_set_state_to_listening(self, lelamp_agent, mock_rgb_service):
        """测试切换到 listening 状态"""
        await lelamp_agent.set_conversation_state("listening")

        assert lelamp_agent._state_manager.current_state == ConversationState.LISTENING
        # 检查蓝色灯光被调用（使用 assert_any_call 因为启动时也会调用 dispatch）
        mock_rgb_service.dispatch.assert_any_call("solid", StateColors.LISTENING, priority=1)

    @pytest.mark.asyncio
    async def test_set_state_to_thinking(self, lelamp_agent, mock_rgb_service):
        """测试切换到 thinking 状态"""
        await lelamp_agent.set_conversation_state("thinking")

        assert lelamp_agent._state_manager.current_state == ConversationState.THINKING
        # 检查紫色灯光被调用
        mock_rgb_service.dispatch.assert_any_call("solid", StateColors.THINKING, priority=1)

    @pytest.mark.asyncio
    async def test_set_state_to_idle(self, lelamp_agent, mock_rgb_service):
        """测试切换到 idle 状态"""
        # 先切换到另一个状态
        await lelamp_agent.set_conversation_state("listening")
        # 再切换回 idle
        await lelamp_agent.set_conversation_state("idle")

        assert lelamp_agent._state_manager.current_state == ConversationState.IDLE
        # 检查暖白灯光被调用
        mock_rgb_service.dispatch.assert_any_call("solid", StateColors.IDLE, priority=1)

    @pytest.mark.asyncio
    async def test_set_state_to_speaking(self, lelamp_agent, mock_rgb_service):
        """测试切换到 speaking 状态"""
        await lelamp_agent.set_conversation_state("speaking")

        assert lelamp_agent._state_manager.current_state == ConversationState.SPEAKING
        # speaking 状态使用呼吸效果
        calls = mock_rgb_service.dispatch.call_args_list
        # 找到 breath 调用
        breath_call = None
        for call in calls:
            if call[0][0] == "breath":
                breath_call = call
                break
        assert breath_call is not None

    @pytest.mark.asyncio
    async def test_set_same_state_no_change(self, lelamp_agent, mock_rgb_service):
        """测试设置相同状态不触发更新"""
        await lelamp_agent.set_conversation_state("idle")
        mock_rgb_service.dispatch.reset_mock()

        await lelamp_agent.set_conversation_state("idle")

        # 不应该再次调用 dispatch
        assert not mock_rgb_service.dispatch.called

    @pytest.mark.asyncio
    async def test_light_override_prevents_state_change(self, lelamp_agent, mock_rgb_service):
        """测试灯光覆盖阻止状态变化"""
        # 设置灯光覆盖
        lelamp_agent._state_manager.set_light_override(duration_s=10.0)

        initial_calls = mock_rgb_service.dispatch.call_count
        await lelamp_agent.set_conversation_state("listening")

        # 由于灯光覆盖，不应该改变灯光
        assert mock_rgb_service.dispatch.call_count == initial_calls


class TestDataChannelMessageHandling:
    """测试 Data Channel 消息处理"""

    @pytest.mark.asyncio
    async def test_handle_chat_message(self, lelamp_agent):
        """测试处理聊天消息"""
        message = json.dumps({"type": "chat", "content": "你好"}).encode("utf-8")
        participant = MagicMock()

        await lelamp_agent.handle_data_message(message, participant)

        assert lelamp_agent._last_user_text == "你好"

    @pytest.mark.asyncio
    async def test_handle_command_play_recording(
        self, lelamp_agent, mock_motors_service, mock_rgb_service
    ):
        """测试处理播放录制动画指令"""
        # Mock send_chat_message
        lelamp_agent._send_chat_message = AsyncMock()
        lelamp_agent.room = MagicMock()
        lelamp_agent.room.local_participant.publish_data = AsyncMock()

        message = json.dumps({"type": "command", "action": "play_recording", "params": {"recording_name": "nod"}}).encode("utf-8")
        participant = MagicMock()

        await lelamp_agent.handle_data_message(message, participant)

        # 验证动作被播放
        mock_motors_service.dispatch.assert_called_with("play", "nod")

    @pytest.mark.asyncio
    async def test_handle_command_set_rgb(
        self, lelamp_agent, mock_rgb_service
    ):
        """测试处理设置 RGB 灯光指令"""
        # Mock send_chat_message
        lelamp_agent._send_chat_message = AsyncMock()
        lelamp_agent.room = MagicMock()
        lelamp_agent.room.local_participant.publish_data = AsyncMock()

        message = json.dumps({
            "type": "command",
            "action": "set_rgb_solid",
            "params": {"r": 255, "g": 0, "b": 0}
        }).encode("utf-8")
        participant = MagicMock()

        await lelamp_agent.handle_data_message(message, participant)

        # 验证 RGB 被设置
        mock_rgb_service.dispatch.assert_called()

    @pytest.mark.asyncio
    async def test_send_chat_message(self, lelamp_agent):
        """测试发送聊天消息"""
        lelamp_agent.room = MagicMock()
        lelamp_agent.room.local_participant.publish_data = AsyncMock()

        await lelamp_agent._send_chat_message("测试消息")

        lelamp_agent.room.local_participant.publish_data.assert_called_once()
        call_args = lelamp_agent.room.local_participant.publish_data.call_args[0][0]
        message = json.loads(call_args.decode("utf-8"))
        assert message["type"] == "chat"
        assert message["content"] == "测试消息"

    @pytest.mark.asyncio
    async def test_update_camera_status(self, lelamp_agent):
        """测试更新摄像头状态"""
        lelamp_agent.room = MagicMock()
        lelamp_agent.room.local_participant.publish_data = AsyncMock()

        await lelamp_agent._update_camera_status(True)

        lelamp_agent.room.local_participant.publish_data.assert_called_once()
        call_args = lelamp_agent.room.local_participant.publish_data.call_args[0][0]
        message = json.loads(call_args.decode("utf-8"))
        assert message["type"] == "camera_status"
        assert message["active"] is True


class TestExecuteCommand:
    """测试指令执行"""

    @pytest.mark.asyncio
    async def test_execute_command_play_recording(self, lelamp_agent, mock_motors_service):
        """测试执行播放录制动画"""
        result = await lelamp_agent._execute_command("play_recording", {"recording_name": "wake_up"})
        assert "开始执行动作" in result
        mock_motors_service.dispatch.assert_called_with("play", "wake_up")

    @pytest.mark.asyncio
    async def test_execute_command_move_joint(self, lelamp_agent, mock_motors_service):
        """测试执行移动关节"""
        result = await lelamp_agent._execute_command("move_joint", {"joint_name": "base_yaw", "angle": 45})
        assert "已将 base_yaw 移动到 45" in result
        mock_motors_service.dispatch.assert_called()

    @pytest.mark.asyncio
    async def test_execute_command_unknown(self, lelamp_agent):
        """测试执行未知指令"""
        result = await lelamp_agent._execute_command("unknown_command", {})
        assert "未知指令" in result


class TestSetSystemVolume:
    """测试系统音量设置"""

    @pytest.mark.asyncio
    async def test_set_system_volume(self, lelamp_agent):
        """测试设置系统音量"""
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.wait = AsyncMock()
            mock_process.communicate = AsyncMock()
            mock_subprocess.return_value = mock_process

            await lelamp_agent._set_system_volume(50)

            # 验证调用了 amixer 命令
            assert mock_subprocess.call_count == 3


class TestVisionTools:
    """测试视觉工具"""

    @pytest.mark.asyncio
    async def test_vision_answer(self, lelamp_agent, mock_qwen_client):
        """测试视觉问答"""
        result = await lelamp_agent.vision_answer("这是什么？")
        assert result == "这是测试图片描述"
        mock_qwen_client.describe.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_homework(self, lelamp_agent, mock_qwen_client):
        """测试作业检查"""
        result = await lelamp_agent.check_homework()
        assert result == "这是测试图片描述"
        mock_qwen_client.describe.assert_called_once()


class TestRGBEffectExecution:
    """测试 RGB 效果执行"""

    @pytest.mark.asyncio
    async def test_execute_rgb_effect_rainbow(self, lelamp_agent, mock_rgb_service):
        """测试执行彩虹效果"""
        result = await lelamp_agent._execute_rgb_effect("rainbow")
        # 应该调用 rainbow 效果
        mock_rgb_service.dispatch.assert_called()

    @pytest.mark.asyncio
    async def test_execute_rgb_effect_unknown(self, lelamp_agent):
        """测试执行未知效果"""
        result = await lelamp_agent._execute_rgb_effect("unknown_effect")
        assert "未知灯效" in result


class TestSendVisionResult:
    """测试发送视觉结果"""

    @pytest.mark.asyncio
    async def test_send_vision_result_with_image(self, lelamp_agent):
        """测试发送带图片的视觉结果"""
        lelamp_agent.room = MagicMock()
        lelamp_agent.room.local_participant.publish_data = AsyncMock()

        await lelamp_agent._send_vision_result("测试结果", b"fake_image_data")

        call_args = lelamp_agent.room.local_participant.publish_data.call_args[0][0]
        message = json.loads(call_args.decode("utf-8"))
        assert message["type"] == "vision_result"
        assert message["content"] == "测试结果"
        assert message["image_base64"] == "ZmFrZV9pbWFnZV9kYXRh"  # base64 encoded

    @pytest.mark.asyncio
    async def test_send_vision_result_without_image(self, lelamp_agent, mock_vision_service):
        """测试发送不带图片的视觉结果（从服务获取）"""
        lelamp_agent.room = MagicMock()
        lelamp_agent.room.local_participant.publish_data = AsyncMock()

        await lelamp_agent._send_vision_result("测试结果", None)

        call_args = lelamp_agent.room.local_participant.publish_data.call_args[0][0]
        message = json.loads(call_args.decode("utf-8"))
        assert message["type"] == "vision_result"
        assert message["image_base64"] == "ZmFrZV9mcmFtZV9kYXRh"  # base64 of fake_frame_data
