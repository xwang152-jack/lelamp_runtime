"""
电机工具集成测试
"""
import pytest
from unittest.mock import Mock, MagicMock
from lelamp.agent.tools.motor_tools import MotorTools, SAFE_JOINT_RANGES
from lelamp.agent.states import StateManager


@pytest.fixture
def mock_motors_service():
    """创建模拟的电机服务"""
    service = Mock()
    service.dispatch = Mock()
    service.get_joint_positions = Mock(return_value={
        "base_yaw": 0.0,
        "base_pitch": 0.0,
        "elbow_pitch": 0.0,
        "wrist_roll": 0.0,
        "wrist_pitch": 0.0,
    })
    service.get_motor_health_summary = Mock(return_value={
        "base_yaw": {
            "latest": {
                "status": "healthy",
                "temperature": 45.0,
                "voltage": 12.0,
                "load": 0.3,
                "position": 0.0,
            },
            "warning_count": 0,
            "critical_count": 0,
            "stall_count": 0,
        },
    })
    return service


@pytest.fixture
def state_manager():
    """创建状态管理器"""
    return StateManager(motion_cooldown_s=2.0, suppress_motion_after_light_s=5.0)


@pytest.fixture
def motor_tools(mock_motors_service, state_manager):
    """创建电机工具实例"""
    return MotorTools(
        motors_service=mock_motors_service,
        state_manager=state_manager,
    )


@pytest.mark.integration
class TestMotorTools:
    """电机工具集成测试"""

    @pytest.mark.asyncio
    async def test_play_recording_success(self, motor_tools, mock_motors_service):
        """测试播放录制动作成功"""
        result = await motor_tools.play_recording("test_recording")

        assert "开始执行动作：test_recording" in result
        mock_motors_service.dispatch.assert_called_once_with("play", "test_recording")

    @pytest.mark.asyncio
    async def test_play_recording_cooldown(self, motor_tools, state_manager, mock_motors_service):
        """测试动作冷却期阻止执行"""
        # 第一次执行成功
        result1 = await motor_tools.play_recording("test_recording")
        assert "开始执行动作" in result1

        # 第二次执行立即调用应该被冷却期阻止
        result2 = await motor_tools.play_recording("test_recording")
        assert "冷却中" in result2 or "抑制" in result2

        # dispatch 只被调用一次
        assert mock_motors_service.dispatch.call_count == 1

    @pytest.mark.asyncio
    async def test_move_joint_valid(self, motor_tools, mock_motors_service):
        """测试移动关节到有效角度"""
        result = await motor_tools.move_joint("base_yaw", 90.0)

        assert "已将 base_yaw 移动到 90.0 度" in result
        mock_motors_service.dispatch.assert_called_once_with(
            "move_joint",
            {"joint_name": "base_yaw", "angle": 90.0}
        )

    @pytest.mark.asyncio
    async def test_move_joint_invalid_name(self, motor_tools, mock_motors_service):
        """测试无效的关节名称"""
        result = await motor_tools.move_joint("invalid_joint", 90.0)

        assert "无效的关节名称" in result
        mock_motors_service.dispatch.assert_not_called()

    @pytest.mark.asyncio
    async def test_move_joint_out_of_range(self, motor_tools, mock_motors_service):
        """测试角度超出安全范围"""
        # 测试超出最大范围
        result1 = await motor_tools.move_joint("base_pitch", 120.0)
        assert "超出" in result1
        assert "安全范围" in result1

        # 测试超出最小范围
        result2 = await motor_tools.move_joint("base_pitch", -120.0)
        assert "超出" in result2
        assert "安全范围" in result2

        # dispatch 不应该被调用
        mock_motors_service.dispatch.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_joint_positions(self, motor_tools, mock_motors_service):
        """测试获取关节位置"""
        result = await motor_tools.get_joint_positions()

        assert "当前关节位置" in result
        assert "base_yaw: 0.0度" in result
        assert "base_pitch: 0.0度" in result
        mock_motors_service.get_joint_positions.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_joint_positions_no_connection(self, motor_tools, mock_motors_service):
        """测试电机未连接时获取关节位置"""
        mock_motors_service.get_joint_positions.return_value = {}

        result = await motor_tools.get_joint_positions()
        assert "无法获取关节位置" in result

    @pytest.mark.asyncio
    async def test_get_motor_health_single(self, motor_tools, mock_motors_service):
        """测试获取单个舵机健康状态"""
        result = await motor_tools.get_motor_health("base_yaw")

        assert "舵机 base_yaw 健康状态" in result
        assert "温度: 45.0°C" in result
        assert "电压: 12.0V" in result
        assert "负载: 30.0%" in result
        mock_motors_service.get_motor_health_summary.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_motor_health_all(self, motor_tools, mock_motors_service):
        """测试获取所有舵机健康状态"""
        result = await motor_tools.get_motor_health()

        assert "所有舵机健康状态" in result
        assert "base_yaw:" in result
        mock_motors_service.get_motor_health_summary.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_motor_health_invalid_name(self, motor_tools, mock_motors_service):
        """测试无效的舵机名称"""
        result = await motor_tools.get_motor_health("invalid_motor")

        assert "无效的舵机名称" in result
        mock_motors_service.get_motor_health_summary.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_motor_health_monitoring_disabled(self, motor_tools, mock_motors_service):
        """测试健康监控未启用"""
        mock_motors_service.get_motor_health_summary.return_value = {"error": "disabled"}

        result = await motor_tools.get_motor_health()
        assert "健康监控未启用" in result

    def test_safe_joint_ranges_coverage(self):
        """测试所有关节都有安全范围定义"""
        expected_joints = ["base_yaw", "base_pitch", "elbow_pitch", "wrist_roll", "wrist_pitch"]

        for joint in expected_joints:
            assert joint in SAFE_JOINT_RANGES, f"Missing SAFE_JOINT_RANGES for {joint}"

            min_angle, max_angle = SAFE_JOINT_RANGES[joint]
            assert isinstance(min_angle, (int, float)), f"Invalid min_angle type for {joint}"
            assert isinstance(max_angle, (int, float)), f"Invalid max_angle type for {joint}"
            assert min_angle < max_angle, f"Invalid range for {joint}"

    @pytest.mark.asyncio
    async def test_move_joint_angle_conversion(self, motor_tools, mock_motors_service):
        """测试角度类型转换"""
        # 测试整数输入
        result1 = await motor_tools.move_joint("base_yaw", 90)
        assert "已将 base_yaw 移动到 90.0 度" in result1

        # 测试字符串输入（应该被转换）
        result2 = await motor_tools.move_joint("base_yaw", "45")
        assert "已将 base_yaw 移动到 45.0 度" in result2 or "无效的角度值" in result2
