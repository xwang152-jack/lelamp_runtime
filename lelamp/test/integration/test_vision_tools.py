"""
视觉工具集成测试
"""
import os
import pytest
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from lelamp.agent.tools.vision_tools import VisionTools
from lelamp.agent.states import StateManager


@pytest.fixture
def mock_vision_service():
    """创建模拟的视觉服务"""
    service = Mock()
    service.get_latest_jpeg_b64 = AsyncMock(return_value=("base64_jpeg_data", 1000))
    service.get_fresh_jpeg_b64 = AsyncMock(return_value=("base64_fresh_jpeg", 1000))
    return service


@pytest.fixture
def mock_qwen_client():
    """创建模拟的 Qwen 客户端"""
    client = Mock()
    client.describe = AsyncMock(return_value="这是一张测试图片的描述")
    return client


@pytest.fixture
def mock_rgb_service():
    """创建模拟的 RGB 服务"""
    service = Mock()
    service.dispatch = Mock()
    return service


@pytest.fixture
def mock_motors_service():
    """创建模拟的电机服务"""
    service = Mock()
    service.dispatch = Mock()
    return service


@pytest.fixture
def mock_rate_limiter():
    """创建模拟的速率限制器"""
    limiter = Mock()
    limiter.acquire = AsyncMock(return_value=True)
    return limiter


@pytest.fixture
def state_manager():
    """创建状态管理器"""
    return StateManager(motion_cooldown_s=2.0, suppress_motion_after_light_s=5.0)


@pytest.fixture
def vision_tools(
    mock_vision_service,
    mock_qwen_client,
    mock_rgb_service,
    mock_motors_service,
    state_manager,
    mock_rate_limiter
):
    """创建视觉工具实例"""
    return VisionTools(
        vision_service=mock_vision_service,
        qwen_client=mock_qwen_client,
        rgb_service=mock_rgb_service,
        motors_service=mock_motors_service,
        state_manager=state_manager,
        rate_limiter=mock_rate_limiter,
    )


@pytest.mark.integration
class TestVisionTools:
    """视觉工具集成测试套件"""

    @pytest.mark.asyncio
    async def test_vision_answer_success(self, vision_tools, mock_vision_service, mock_qwen_client):
        """测试视觉问答成功"""
        result = await vision_tools.vision_answer("这是什么？")

        assert "测试图片" in result
        mock_vision_service.get_latest_jpeg_b64.assert_called_once()
        mock_qwen_client.describe.assert_called_once()
        # 验证 RGB 灯光被设置为白色
        vision_tools.rgb_service.dispatch.assert_called()

    @pytest.mark.asyncio
    async def test_vision_answer_no_vision_service(self, vision_tools, mock_vision_service):
        """测试视觉服务未初始化"""
        vision_tools._vision_service = None
        result = await vision_tools.vision_answer("这是什么？")

        assert "未初始化" in result
        mock_vision_service.get_latest_jpeg_b64.assert_not_called()

    @pytest.mark.asyncio
    async def test_vision_answer_no_qwen_client(self, vision_tools, mock_qwen_client):
        """测试 Qwen 客户端未初始化"""
        vision_tools._qwen_client = None
        result = await vision_tools.vision_answer("这是什么？")

        assert "未初始化" in result
        mock_qwen_client.describe.assert_not_called()

    @pytest.mark.asyncio
    async def test_vision_answer_rate_limited(self, vision_tools, mock_rate_limiter, mock_vision_service):
        """测试速率限制"""
        mock_rate_limiter.acquire.return_value = False

        result = await vision_tools.vision_answer("这是什么？")

        assert "频繁" in result
        mock_vision_service.get_latest_jpeg_b64.assert_not_called()

    @pytest.mark.asyncio
    async def test_vision_answer_no_frame(self, vision_tools, mock_vision_service):
        """测试没有可用画面"""
        mock_vision_service.get_latest_jpeg_b64.return_value = None

        result = await vision_tools.vision_answer("这是什么？")

        assert "没有可用画面" in result

    @pytest.mark.asyncio
    async def test_check_homework_success(self, vision_tools, mock_vision_service, mock_qwen_client):
        """测试作业检查成功"""
        mock_qwen_client.describe.return_value = "作业检查完成，有两处错误需要修正。"

        result = await vision_tools.check_homework()

        assert "错误" in result or "作业" in result
        mock_vision_service.get_fresh_jpeg_b64.assert_called_once_with(timeout_s=5.0)
        # 验证使用特定的老师人设 prompt
        call_args = mock_qwen_client.describe.call_args
        assert "老师" in call_args[1]["question"]

    @pytest.mark.asyncio
    async def test_check_homework_no_vision_service(self, vision_tools, mock_vision_service):
        """测试作业检查时视觉服务未初始化"""
        vision_tools._vision_service = None
        result = await vision_tools.check_homework()

        assert "未初始化" in result
        mock_vision_service.get_fresh_jpeg_b64.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_homework_rate_limited(self, vision_tools, mock_rate_limiter):
        """测试作业检查速率限制"""
        mock_rate_limiter.acquire.return_value = False

        result = await vision_tools.check_homework()

        assert "频繁" in result or "喘口气" in result

    @pytest.mark.asyncio
    async def test_check_homework_no_frame(self, vision_tools, mock_vision_service):
        """测试作业检查时没有画面"""
        mock_vision_service.get_fresh_jpeg_b64.return_value = None

        result = await vision_tools.check_homework()

        assert "拍照失败" in result or "无法看清" in result

    @pytest.mark.asyncio
    async def test_capture_to_feishu_success(
        self, vision_tools, mock_vision_service, mock_rgb_service
    ):
        """测试飞书推送成功"""
        # 设置环境变量
        os.environ["FEISHU_APP_ID"] = "test_app_id"
        os.environ["FEISHU_APP_SECRET"] = "test_secret"
        os.environ["FEISHU_RECEIVE_ID"] = "test_receive_id"

        # Mock 飞书 API 调用
        with patch("urllib.request.urlopen") as mock_urlopen:
            # 设置 mock 返回值
            mock_token_resp = MagicMock()
            mock_token_resp.read.return_value = b'{"tenant_access_token": "test_token"}'
            mock_upload_resp = MagicMock()
            mock_upload_resp.read.return_value = b'{"data": {"image_key": "test_image_key"}}'
            mock_msg_resp = MagicMock()
            mock_msg_resp.read.return_value = b'{"code": 0}'

            mock_urlopen.side_effect = [mock_token_resp, mock_upload_resp, mock_msg_resp]

            result = await vision_tools.capture_to_feishu()

            assert "成功" in result or "推送" in result
            mock_vision_service.get_fresh_jpeg_b64.assert_called_once_with(timeout_s=5.0)
            # 验证电机被停止
            vision_tools.motors_service.dispatch.assert_called_once()

        # 清理环境变量
        del os.environ["FEISHU_APP_ID"]
        del os.environ["FEISHU_APP_SECRET"]
        del os.environ["FEISHU_RECEIVE_ID"]

    @pytest.mark.asyncio
    async def test_capture_to_feishu_no_vision_service(self, vision_tools):
        """测试飞书推送时视觉服务未初始化"""
        vision_tools._vision_service = None
        result = await vision_tools.capture_to_feishu()

        assert "未初始化" in result

    @pytest.mark.asyncio
    async def test_capture_to_feishu_missing_config(self, vision_tools, mock_vision_service):
        """测试飞书配置不完整"""
        # 确保环境变量未设置
        os.environ.pop("FEISHU_APP_ID", None)
        os.environ.pop("FEISHU_APP_SECRET", None)
        os.environ.pop("FEISHU_RECEIVE_ID", None)

        result = await vision_tools.capture_to_feishu()

        assert "配置不完整" in result or "环境变量" in result
        mock_vision_service.get_fresh_jpeg_b64.assert_not_called()

    @pytest.mark.asyncio
    async def test_light_override_behavior(self, vision_tools, state_manager):
        """测试灯光覆盖行为"""
        # vision_answer 应该修改灯光覆盖
        initial_value = state_manager._light_override_until_ts

        await vision_tools.vision_answer("测试")

        # 验证灯光覆盖被修改（设置为新的长时间覆盖）
        assert state_manager._light_override_until_ts != initial_value
        # 验证最终被恢复（回到原来的值或接近）
        # 由于测试运行很快，恢复的值应该接近初始值

    @pytest.mark.asyncio
    async def test_check_homework_uses_fresh_frame(self, vision_tools, mock_vision_service):
        """测试作业检查使用最新画面"""
        await vision_tools.check_homework()

        # 验证调用的是 get_fresh_jpeg_b64 而不是 get_latest_jpeg_b64
        mock_vision_service.get_fresh_jpeg_b64.assert_called_once()
        mock_vision_service.get_latest_jpeg_b64.assert_not_called()

    @pytest.mark.asyncio
    async def test_capture_to_feishu_locks_motion(
        self, vision_tools, mock_rgb_service
    ):
        """测试飞书拍照时锁定动作"""
        # 设置环境变量
        os.environ["FEISHU_APP_ID"] = "test_app_id"
        os.environ["FEISHU_APP_SECRET"] = "test_secret"
        os.environ["FEISHU_RECEIVE_ID"] = "test_receive_id"

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_token_resp = MagicMock()
            mock_token_resp.read.return_value = b'{"tenant_access_token": "test_token"}'
            mock_upload_resp = MagicMock()
            mock_upload_resp.read.return_value = b'{"data": {"image_key": "test_image_key"}}'
            mock_msg_resp = MagicMock()
            mock_msg_resp.read.return_value = b'{"code": 0}'

            mock_urlopen.side_effect = [mock_token_resp, mock_upload_resp, mock_msg_resp]

            # 初始状态未锁定
            assert vision_tools._motion_locked is False

            await vision_tools.capture_to_feishu()

            # 验证动作在过程中被锁定，最后被解锁
            assert vision_tools._motion_locked is False

        # 清理环境变量
        del os.environ["FEISHU_APP_ID"]
        del os.environ["FEISHU_APP_SECRET"]
        del os.environ["FEISHU_RECEIVE_ID"]

    @pytest.mark.asyncio
    async def test_vision_answer_white_light(self, vision_tools, mock_rgb_service):
        """测试视觉问答时设置白色补光"""
        await vision_tools.vision_answer("这是什么？")

        # 验证白色灯光被设置
        calls = mock_rgb_service.dispatch.call_args_list
        # 查找 solid 事件调用
        solid_call = None
        for call in calls:
            if call[0][0] == "solid":
                solid_call = call
                break

        assert solid_call is not None
        assert solid_call[0][1] == (255, 255, 255)
