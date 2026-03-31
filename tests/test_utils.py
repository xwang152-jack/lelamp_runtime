"""
测试工具类模块
"""
import os
import socket
import pytest
import asyncio
from unittest.mock import patch, MagicMock, Mock
from pathlib import Path

from lelamp.utils.rate_limiter import (
    RateLimitConfig,
    RateLimiter,
    RateLimiterManager,
    get_rate_limiter,
    get_all_rate_limiter_stats,
)
from lelamp.utils.security import (
    get_device_id,
    generate_license_key,
    verify_license,
)
from lelamp.utils.url_validation import (
    validate_external_url,
    ALLOWED_API_DOMAINS,
)


@pytest.mark.unit
class TestRateLimitConfig:
    """测试速率限制配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = RateLimitConfig()
        assert config.rate == 2.0
        assert config.capacity == 10

    def test_custom_config(self):
        """测试自定义配置"""
        config = RateLimitConfig(rate=5.0, capacity=20)
        assert config.rate == 5.0
        assert config.capacity == 20

    def test_invalid_rate(self):
        """测试无效速率"""
        with pytest.raises(ValueError, match="Rate must be positive"):
            RateLimitConfig(rate=0, capacity=10)

        with pytest.raises(ValueError, match="Rate must be positive"):
            RateLimitConfig(rate=-1, capacity=10)

    def test_invalid_capacity(self):
        """测试无效容量"""
        with pytest.raises(ValueError, match="Capacity must be positive"):
            RateLimitConfig(rate=2.0, capacity=0)

        with pytest.raises(ValueError, match="Capacity must be positive"):
            RateLimitConfig(rate=2.0, capacity=-1)


@pytest.mark.unit
class TestRateLimiter:
    """测试速率限制器"""

    @pytest.mark.asyncio
    async def test_rate_limiter_init(self):
        """测试速率限制器初始化"""
        config = RateLimitConfig(rate=10, capacity=20)
        limiter = RateLimiter(config, name="test")

        assert limiter.name == "test"
        assert limiter.config.rate == 10
        assert limiter.config.capacity == 20

    @pytest.mark.asyncio
    async def test_acquire_immediate(self):
        """测试立即获取令牌"""
        config = RateLimitConfig(rate=10, capacity=20)
        limiter = RateLimiter(config, name="test")

        result = await limiter.acquire(1)
        assert result is True

    @pytest.mark.asyncio
    async def test_acquire_multiple(self):
        """测试获取多个令牌"""
        config = RateLimitConfig(rate=100, capacity=20)
        limiter = RateLimiter(config, name="test")

        result = await limiter.acquire(10)
        assert result is True

    @pytest.mark.asyncio
    async def test_acquire_exceed_capacity(self):
        """测试请求超过容量"""
        config = RateLimitConfig(rate=10, capacity=20)
        limiter = RateLimiter(config, name="test")

        with pytest.raises(ValueError, match="exceeds capacity"):
            await limiter.acquire(25)

    @pytest.mark.asyncio
    async def test_acquire_invalid_tokens(self):
        """测试无效令牌数"""
        config = RateLimitConfig(rate=10, capacity=20)
        limiter = RateLimiter(config, name="test")

        with pytest.raises(ValueError, match="Tokens must be positive"):
            await limiter.acquire(0)

        with pytest.raises(ValueError, match="Tokens must be positive"):
            await limiter.acquire(-1)

    @pytest.mark.asyncio
    async def test_acquire_with_timeout(self):
        """测试带超时的获取"""
        config = RateLimitConfig(rate=1, capacity=2)
        limiter = RateLimiter(config, name="test")

        # 消耗所有令牌
        await limiter.acquire(2)

        # 请求更多令牌，但设置短超时
        result = await limiter.acquire(1, timeout=0.1)
        assert result is False

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """测试获取统计信息"""
        config = RateLimitConfig(rate=10, capacity=20)
        limiter = RateLimiter(config, name="test")

        stats = limiter.get_stats()
        assert stats["name"] == "test"
        assert stats["rate"] == 10
        assert stats["capacity"] == 20
        assert stats["requests_total"] == 0

    @pytest.mark.asyncio
    async def test_reset_stats(self):
        """测试重置统计"""
        config = RateLimitConfig(rate=10, capacity=20)
        limiter = RateLimiter(config, name="test")

        await limiter.acquire(1)
        limiter.reset_stats()

        stats = limiter.get_stats()
        assert stats["requests_total"] == 0
        assert stats["requests_allowed"] == 0

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """测试异步上下文管理器"""
        config = RateLimitConfig(rate=10, capacity=20)
        limiter = RateLimiter(config, name="test")

        async with limiter:
            pass

        stats = limiter.get_stats()
        assert stats["requests_allowed"] == 1


@pytest.mark.unit
class TestRateLimiterManager:
    """测试速率限制器管理器"""

    def test_create_limiter(self):
        """测试创建限制器"""
        manager = RateLimiterManager()
        limiter = manager.create_limiter("test", rate=10, capacity=20)

        assert limiter.name == "test"
        assert limiter.config.rate == 10

    def test_get_existing_limiter(self):
        """测试获取已有限制器"""
        manager = RateLimiterManager()
        limiter1 = manager.create_limiter("test", rate=10, capacity=20)
        limiter2 = manager.create_limiter("test", rate=5, capacity=10)

        # 应该返回同一个实例
        assert limiter1 is limiter2
        assert limiter2.config.rate == 10  # 保持原始配置

    def test_get_limiter(self):
        """测试获取限制器"""
        manager = RateLimiterManager()
        manager.create_limiter("test", rate=10, capacity=20)

        limiter = manager.get_limiter("test")
        assert limiter is not None

        limiter = manager.get_limiter("nonexistent")
        assert limiter is None

    def test_get_all_stats(self):
        """测试获取所有统计"""
        manager = RateLimiterManager()
        manager.create_limiter("limiter1", rate=10, capacity=20)
        manager.create_limiter("limiter2", rate=5, capacity=10)

        stats = manager.get_all_stats()
        assert "limiter1" in stats
        assert "limiter2" in stats
        assert stats["limiter1"]["rate"] == 10
        assert stats["limiter2"]["rate"] == 5

    def test_reset_all_stats(self):
        """测试重置所有统计"""
        manager = RateLimiterManager()
        manager.create_limiter("test", rate=10, capacity=20)

        # 这里无法直接测试因为需要asyncio来acquire令牌
        # 但可以确保方法存在
        manager.reset_all_stats()


@pytest.mark.unit
class TestGlobalRateLimiter:
    """测试全局速率限制器"""

    def test_get_rate_limiter(self):
        """测试获取全局限制器"""
        limiter = get_rate_limiter("test", rate=10, capacity=20)

        assert limiter.name == "test"
        assert limiter.config.rate == 10

    def test_get_singleton(self):
        """测试单例模式"""
        limiter1 = get_rate_limiter("singleton_test", rate=10, capacity=20)
        limiter2 = get_rate_limiter("singleton_test", rate=5, capacity=10)

        assert limiter1 is limiter2

    def test_get_all_stats(self):
        """测试获取所有统计"""
        get_rate_limiter("stats_test", rate=10, capacity=20)
        stats = get_all_rate_limiter_stats()

        assert "stats_test" in stats


@pytest.mark.unit
class TestSecurity:
    """测试安全工具"""

    @patch('os.path.exists', return_value=True)
    @patch('builtins.open', create=True)
    def test_get_device_id_linux(self, mock_open, mock_exists):
        """测试Linux设备ID获取"""
        mock_file = MagicMock()
        mock_file.__enter__ = Mock(return_value=mock_file)
        mock_file.__exit__ = Mock(return_value=False)
        mock_file.__iter__ = Mock(return_value=iter(["Serial\t: 00000000abcdef\n"]))
        mock_open.return_value = mock_file
        device_id = get_device_id()
        assert device_id == "00000000abcdef"

    @patch('os.path.exists', return_value=False)
    @patch('uuid.getnode', return_value=12345)
    def test_get_device_id_mac(self, mock_getnode, mock_exists):
        """测试Mac设备ID获取"""
        device_id = get_device_id()
        assert device_id == "12345"

    def test_generate_license_key(self):
        """测试生成许可证密钥"""
        device_id = "test_device_123"
        secret = "test_secret"
        key = generate_license_key(device_id, secret)
        assert isinstance(key, str)
        assert len(key) == 16

    @patch.dict(os.environ, {"LELAMP_LICENSE_SECRET": "test_secret"})
    def test_generate_and_verify_key(self):
        """测试生成和验证密钥"""
        device_id = "test_device_123"
        key = generate_license_key(device_id)
        assert len(key) == 16

    @patch.dict(os.environ, {"LELAMP_DEV_MODE": "1"})
    def test_verify_license_dev_mode(self):
        """测试开发模式验证"""
        assert verify_license() is True

    @patch('lelamp.utils.security.get_device_id', return_value='test_device')
    @patch.dict(os.environ, {}, clear=True)
    def test_verify_license_no_key(self, mock_dev):
        """测试无密钥验证"""
        assert verify_license() is False

    @patch('lelamp.utils.security.get_device_id', return_value='test_device')
    @patch.dict(os.environ, {"LELAMP_LICENSE_SECRET": "test_secret"}, clear=False)
    def test_verify_license_no_env_key(self, mock_dev):
        """测试有secret但无license key"""
        os.environ.pop("LELAMP_LICENSE_KEY", None)
        os.environ.pop("LELAMP_DEV_MODE", None)
        assert verify_license() is False

    @patch.dict(os.environ, {"LELAMP_LICENSE_SECRET": "test_secret", "LELAMP_DEV_MODE": "1"})
    def test_dev_mode_bypass(self):
        """测试开发模式绕过验证"""
        assert verify_license() is True


@pytest.mark.unit
class TestURLValidation:
    """测试URL验证"""

    def test_allowed_domains(self):
        """测试允许的域名列表"""
        assert isinstance(ALLOWED_API_DOMAINS, list)
        assert len(ALLOWED_API_DOMAINS) > 0
        assert "api.deepseek.com" in ALLOWED_API_DOMAINS

    def test_validate_external_url_valid(self):
        """测试验证有效外部URL"""
        # 使用白名单中的域名
        result = validate_external_url(
            "https://api.deepseek.com/v1/chat",
            ALLOWED_API_DOMAINS
        )
        assert result is True

    def test_validate_external_url_invalid_scheme(self):
        """测试无效协议"""
        result = validate_external_url(
            "http://api.deepseek.com/v1/chat",
            ALLOWED_API_DOMAINS
        )
        assert result is False

    def test_validate_external_url_not_whitelisted(self):
        """测试非白名单域名"""
        result = validate_external_url(
            "https://evil.com/api",
            ALLOWED_API_DOMAINS
        )
        assert result is False

    @patch('socket.gethostbyname', return_value='93.184.216.34')
    def test_validate_external_url_subdomain(self, mock_dns):
        """测试子域名匹配"""
        custom_whitelist = ["example.com", "api.example.com"]
        result = validate_external_url(
            "https://api.example.com/v1",
            custom_whitelist
        )
        assert result is True

    @patch('socket.gethostbyname', return_value='10.0.0.1')
    def test_validate_external_url_private_ip(self, mock_dns):
        """测试私有IP拒绝"""
        custom_whitelist = ["example.com"]
        result = validate_external_url(
            "https://example.com/api",
            custom_whitelist
        )
        assert result is False

    @patch('socket.gethostbyname', side_effect=socket.gaierror('DNS failed'))
    def test_validate_external_url_dns_failure(self, mock_dns):
        """测试DNS解析失败"""
        result = validate_external_url(
            "https://api.deepseek.com/v1/chat",
            ALLOWED_API_DOMAINS
        )
        assert result is False
