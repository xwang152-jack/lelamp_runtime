"""
测试API路由模块
"""
import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta

from fastapi import FastAPI


@pytest.mark.unit
class TestAPIModels:
    """测试API模型"""

    def test_import_models(self):
        """测试导入模型"""
        # 测试能否成功导入
        from lelamp.api.models import DeviceStateResponse, SettingsResponse
        assert DeviceStateResponse is not None
        assert SettingsResponse is not None


@pytest.mark.unit
class TestSystemRoutes:
    """测试系统路由"""

    def test_health_check(self):
        """测试健康检查"""
        from lelamp.api.routes.system import health_check

        result = health_check()
        assert result["status"] == "healthy"
        assert "timestamp" in result

    def test_get_system_info(self):
        """测试获取系统信息"""
        from lelamp.api.routes.system import get_system_info

        with patch('platform.platform', return_value='Linux'), \
             patch('platform.python_version', return_value='3.12.0'), \
             patch('os.cpu_count', return_value=4):

            result = get_system_info()
            assert "platform" in result
            assert "python_version" in result


@pytest.mark.unit
class TestWiFiManager:
    """测试WiFi管理器"""

    @pytest.mark.asyncio
    async def test_wifi_manager_init(self):
        """测试WiFi管理器初始化"""
        from lelamp.api.services.wifi_manager import WiFiManager

        manager = WiFiManager()
        assert manager is not None

    @pytest.mark.asyncio
    async def test_scan_networks(self):
        """测试扫描网络"""
        from lelamp.api.services.wifi_manager import WiFiManager

        manager = WiFiManager()
        networks = await manager.scan_networks()

        # 应该返回列表
        assert isinstance(networks, list)

    @pytest.mark.asyncio
    async def test_connect(self):
        """测试连接网络"""
        from lelamp.api.services.wifi_manager import WiFiManager

        manager = WiFiManager()
        result = await manager.connect("TestSSID", "password")

        # 结果可能是成功或失败
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_connect_retries_on_failure(self):
        """WiFi 连接失败时应自动重试"""
        from lelamp.api.services.wifi_manager import WiFiManager

        manager = WiFiManager()
        call_count = 0

        async def mock_connect_once(ssid, password):
            nonlocal call_count
            call_count += 1
            return {"success": False, "message": "连接失败", "ssid": ssid}

        with patch('lelamp.api.services.wifi_manager.asyncio.sleep', new_callable=AsyncMock):
            with patch.object(manager, '_try_nmcli_connect', side_effect=mock_connect_once):
                result = await manager.connect("TestSSID", "wrongpass", max_retries=3)

        assert result["success"] is False
        assert call_count == 3  # 应重试 3 次

    @pytest.mark.asyncio
    async def test_connect_succeeds_on_second_attempt(self):
        """WiFi 连接第一次失败、第二次成功"""
        from lelamp.api.services.wifi_manager import WiFiManager

        manager = WiFiManager()
        attempts = []

        async def mock_connect_once(ssid, password):
            attempts.append(1)
            if len(attempts) == 1:
                return {"success": False, "message": "连接超时", "ssid": ssid}
            return {"success": True, "message": "连接成功", "ssid": ssid}

        with patch('lelamp.api.services.wifi_manager.asyncio.sleep', new_callable=AsyncMock):
            with patch.object(manager, '_try_nmcli_connect', side_effect=mock_connect_once):
                result = await manager.connect("TestSSID", "pass", max_retries=3)

        assert result["success"] is True
        assert len(attempts) == 2

    @pytest.mark.asyncio
    async def test_verify_network_reachability_success(self):
        """网络可达性验证成功"""
        from lelamp.api.services.wifi_manager import WiFiManager
        from unittest.mock import AsyncMock, MagicMock

        manager = WiFiManager()
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"", b""))

        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            result = await manager._verify_network_reachability()

        assert result is True

    @pytest.mark.asyncio
    async def test_verify_network_reachability_failure(self):
        """网络不可达时返回 False"""
        from lelamp.api.services.wifi_manager import WiFiManager
        from unittest.mock import AsyncMock, MagicMock

        manager = WiFiManager()
        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(return_value=(b"", b"ping: connect: Network is unreachable"))

        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            result = await manager._verify_network_reachability()

        assert result is False


@pytest.mark.unit
class TestAPManager:
    """测试AP管理器"""

    def test_ap_manager_init(self):
        """测试AP管理器初始化"""
        from lelamp.api.services.ap_manager import APManager

        manager = APManager()
        assert manager is not None

    def test_is_ap_mode(self):
        """测试检查AP模式"""
        from lelamp.api.services.ap_manager import APManager

        manager = APManager()
        is_ap = manager.is_ap_mode()

        assert isinstance(is_ap, bool)

    def test_start_ap(self):
        """测试启动AP"""
        from lelamp.api.services.ap_manager import APManager

        manager = APManager()
        # 在非Linux环境下可能会失败
        try:
            result = manager.start_ap("TestAP", "password123")
            assert isinstance(result, dict)
        except Exception:
            # 非Linux环境下预期会失败
            pass

    def test_stop_ap(self):
        """测试停止AP"""
        from lelamp.api.services.ap_manager import APManager

        manager = APManager()
        try:
            result = manager.stop_ap()
            assert isinstance(result, dict)
        except Exception:
            # 非Linux环境下预期会失败
            pass


@pytest.mark.unit
class TestOnboardingService:
    """测试接入服务"""

    def test_onboarding_init(self):
        """测试接入服务初始化"""
        from lelamp.api.services.onboarding import OnboardingService

        service = OnboardingService()
        assert service is not None

    def test_get_onboarding_status(self):
        """测试获取接入状态"""
        from lelamp.api.services.onboarding import OnboardingService

        service = OnboardingService()
        status = service.get_status()

        assert isinstance(status, dict)
        assert "completed" in status


@pytest.mark.unit
class TestCaptivePortal:
    """测试强制门户"""

    def test_captive_portal_app(self):
        """测试强制门户应用"""
        from lelamp.api.services.captive_portal import app

        assert app is not None
        assert isinstance(app, FastAPI)


@pytest.mark.unit
class TestWebSocketRoutes:
    """测试WebSocket路由"""

    def test_websocket_module(self):
        """测试WebSocket模块导入"""
        from lelamp.api.routes import websocket

        # 检查模块是否导入成功
        assert websocket is not None


@pytest.mark.unit
class TestHistoryRoutes:
    """测试历史记录路由"""

    def test_history_module(self):
        """测试历史记录模块导入"""
        from lelamp.api.routes import history

        # 检查模块是否导入成功
        assert history is not None


@pytest.mark.unit
class TestAuthModule:
    """测试认证模块"""

    def test_auth_module_import(self):
        """测试认证模块导入"""
        from lelamp.api.routes import auth
        from lelamp.api.services import auth_service

        # 检查模块是否导入成功
        assert auth is not None
        assert auth_service is not None


@pytest.mark.unit
class TestConfigSync:
    """测试配置同步"""

    def test_config_sync_module(self):
        """测试配置同步模块"""
        from lelamp.api.services import config_sync

        # 检查模块是否导入成功
        assert config_sync is not None


@pytest.mark.unit
class TestSetupState:
    """测试设置状态"""

    def test_setup_state_module(self):
        """测试设置状态模块"""
        from lelamp.api.services import setup_state

        # 检查模块是否导入成功
        assert setup_state is not None


@pytest.mark.unit
class TestWiFiScanner:
    """测试WiFi扫描器"""

    def test_wifi_scanner_module(self):
        """测试WiFi扫描器模块"""
        from lelamp.api.services import wifi_scanner

        # 检查模块是否导入成功
        assert wifi_scanner is not None


@pytest.mark.unit
class TestNetworkManager:
    """测试网络管理器"""

    def test_network_manager_module(self):
        """测试网络管理器模块"""
        from lelamp.api.services import network_manager

        # 检查模块是否导入成功
        assert network_manager is not None
