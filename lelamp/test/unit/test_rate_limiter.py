"""
Rate Limiter 单元测试
"""
import pytest
import asyncio
import time
from lelamp.utils.rate_limiter import (
    RateLimiter,
    RateLimitConfig,
    RateLimiterManager,
    get_rate_limiter,
)


@pytest.mark.unit
class TestRateLimitConfig:
    """RateLimitConfig 测试套件"""

    def test_valid_config(self):
        """测试有效配置"""
        config = RateLimitConfig(rate=2.0, capacity=10)
        assert config.rate == 2.0
        assert config.capacity == 10

    def test_invalid_rate(self):
        """测试无效速率"""
        with pytest.raises(ValueError, match="Rate must be positive"):
            RateLimitConfig(rate=0, capacity=10)

        with pytest.raises(ValueError, match="Rate must be positive"):
            RateLimitConfig(rate=-1.0, capacity=10)

    def test_invalid_capacity(self):
        """测试无效容量"""
        with pytest.raises(ValueError, match="Capacity must be positive"):
            RateLimitConfig(rate=2.0, capacity=0)

        with pytest.raises(ValueError, match="Capacity must be positive"):
            RateLimitConfig(rate=2.0, capacity=-1)


@pytest.mark.unit
class TestRateLimiter:
    """RateLimiter 测试套件"""

    @pytest.mark.asyncio
    async def test_basic_acquisition(self):
        """测试基本令牌获取"""
        config = RateLimitConfig(rate=10.0, capacity=2)
        limiter = RateLimiter(config, name="test_basic")

        # 应该能立即获取 2 个令牌
        assert await limiter.acquire(tokens=1) is True
        assert await limiter.acquire(tokens=1) is True

        # 第三次需要等待
        start = time.time()
        assert await limiter.acquire(tokens=1, timeout=0.01) is False  # 超时
        elapsed = time.time() - start
        assert elapsed < 0.1  # 应该快速返回

    @pytest.mark.asyncio
    async def test_token_refill(self):
        """测试令牌补充"""
        config = RateLimitConfig(rate=10.0, capacity=1)
        limiter = RateLimiter(config, name="test_refill")

        # 消耗一个令牌
        assert await limiter.acquire(tokens=1) is True

        # 等待令牌补充（10 tokens/s = 0.1s per token）
        await asyncio.sleep(0.15)

        # 现在应该有令牌了
        assert await limiter.acquire(tokens=1, timeout=0.01) is True

    @pytest.mark.asyncio
    async def test_invalid_tokens(self):
        """测试无效令牌数量"""
        config = RateLimitConfig(rate=1.0, capacity=5)
        limiter = RateLimiter(config, name="test_invalid")

        # 零令牌
        with pytest.raises(ValueError, match="Tokens must be positive"):
            await limiter.acquire(tokens=0)

        # 负令牌
        with pytest.raises(ValueError, match="Tokens must be positive"):
            await limiter.acquire(tokens=-1)

        # 超出容量
        with pytest.raises(ValueError, match="exceeds capacity"):
            await limiter.acquire(tokens=10)

    @pytest.mark.asyncio
    async def test_timeout(self):
        """测试超时机制"""
        config = RateLimitConfig(rate=1.0, capacity=1)
        limiter = RateLimiter(config, name="test_timeout")

        # 消耗令牌
        assert await limiter.acquire(tokens=1) is True

        # 立即请求应该超时
        start = time.time()
        result = await limiter.acquire(tokens=1, timeout=0.1)
        elapsed = time.time() - start

        assert result is False
        assert elapsed < 0.2  # 应该在超时时间内返回

    @pytest.mark.asyncio
    async def test_wait_and_acquire(self):
        """测试等待并获取令牌"""
        config = RateLimitConfig(rate=10.0, capacity=1)
        limiter = RateLimiter(config, name="test_wait")

        # 消耗令牌
        assert await limiter.acquire(tokens=1) is True

        # 等待获取（10 tokens/s = 0.1s per token）
        start = time.time()
        result = await limiter.acquire(tokens=1, timeout=0.5)
        elapsed = time.time() - start

        assert result is True
        assert 0.05 < elapsed < 0.25  # 应该等待约 0.1 秒

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """测试异步上下文管理器"""
        config = RateLimitConfig(rate=10.0, capacity=2)
        limiter = RateLimiter(config, name="test_context")

        async with limiter:
            pass  # 应该自动获取令牌

        # 检查统计信息
        stats = limiter.get_stats()
        assert stats["requests_allowed"] == 1

    @pytest.mark.asyncio
    async def test_statistics(self):
        """测试统计信息"""
        config = RateLimitConfig(rate=10.0, capacity=2)
        limiter = RateLimiter(config, name="test_stats")

        # 成功获取
        await limiter.acquire(tokens=1)
        await limiter.acquire(tokens=1)

        # 超时拒绝
        await limiter.acquire(tokens=1, timeout=0.01)

        stats = limiter.get_stats()
        assert stats["requests_total"] == 3
        assert stats["requests_allowed"] == 2
        assert stats["requests_denied"] == 1
        assert stats["denial_rate"] == pytest.approx(1 / 3)
        assert stats["name"] == "test_stats"
        assert stats["rate"] == 10.0
        assert stats["capacity"] == 2

    @pytest.mark.asyncio
    async def test_reset_stats(self):
        """测试重置统计信息"""
        config = RateLimitConfig(rate=10.0, capacity=1)
        limiter = RateLimiter(config, name="test_reset")

        # 生成一些统计数据
        await limiter.acquire(tokens=1)
        await limiter.acquire(tokens=1, timeout=0.01)

        # 重置
        limiter.reset_stats()

        stats = limiter.get_stats()
        assert stats["requests_total"] == 0
        assert stats["requests_allowed"] == 0
        assert stats["requests_denied"] == 0


@pytest.mark.unit
class TestRateLimiterManager:
    """RateLimiterManager 测试套件"""

    def test_create_limiter(self):
        """测试创建限制器"""
        manager = RateLimiterManager()
        limiter = manager.create_limiter("test_api", rate=2.0, capacity=5)

        assert limiter is not None
        assert limiter.name == "test_api"
        assert limiter.config.rate == 2.0
        assert limiter.config.capacity == 5

    def test_singleton_limiter(self):
        """测试限制器单例"""
        manager = RateLimiterManager()
        limiter1 = manager.create_limiter("test_api", rate=2.0, capacity=5)
        limiter2 = manager.create_limiter("test_api", rate=10.0, capacity=10)

        # 应该返回同一实例（忽略新参数）
        assert limiter1 is limiter2

    def test_different_limiters(self):
        """测试不同限制器"""
        manager = RateLimiterManager()
        limiter1 = manager.create_limiter("api1", rate=1.0, capacity=1)
        limiter2 = manager.create_limiter("api2", rate=2.0, capacity=2)

        assert limiter1 is not limiter2
        assert limiter1.name == "api1"
        assert limiter2.name == "api2"

    def test_get_limiter(self):
        """测试获取限制器"""
        manager = RateLimiterManager()
        manager.create_limiter("test_api", rate=1.0, capacity=1)

        limiter = manager.get_limiter("test_api")
        assert limiter is not None
        assert limiter.name == "test_api"

        # 不存在的限制器
        assert manager.get_limiter("nonexistent") is None

    @pytest.mark.asyncio
    async def test_get_all_stats(self):
        """测试获取所有统计信息"""
        manager = RateLimiterManager()
        limiter1 = manager.create_limiter("api1", rate=10.0, capacity=1)
        limiter2 = manager.create_limiter("api2", rate=10.0, capacity=1)

        # 生成一些统计数据
        await limiter1.acquire(tokens=1)
        await limiter2.acquire(tokens=1)

        all_stats = manager.get_all_stats()
        assert "api1" in all_stats
        assert "api2" in all_stats
        assert all_stats["api1"]["requests_allowed"] == 1
        assert all_stats["api2"]["requests_allowed"] == 1

    @pytest.mark.asyncio
    async def test_reset_all_stats(self):
        """测试重置所有统计信息"""
        manager = RateLimiterManager()
        limiter1 = manager.create_limiter("api1", rate=10.0, capacity=1)
        limiter2 = manager.create_limiter("api2", rate=10.0, capacity=1)

        # 生成一些统计数据
        await limiter1.acquire(tokens=1)
        await limiter2.acquire(tokens=1)

        # 重置所有
        manager.reset_all_stats()

        all_stats = manager.get_all_stats()
        assert all_stats["api1"]["requests_total"] == 0
        assert all_stats["api2"]["requests_total"] == 0


@pytest.mark.unit
class TestGlobalRateLimiter:
    """全局 rate limiter 函数测试套件"""

    @pytest.mark.asyncio
    async def test_get_rate_limiter(self):
        """测试全局 get_rate_limiter 函数"""
        limiter = get_rate_limiter("global_test", rate=5.0, capacity=3)

        assert limiter is not None
        assert limiter.name == "global_test"
        assert limiter.config.rate == 5.0
        assert limiter.config.capacity == 3

        # 可以正常使用
        assert await limiter.acquire(tokens=1) is True

    def test_get_rate_limiter_singleton(self):
        """测试全局函数返回单例"""
        limiter1 = get_rate_limiter("global_singleton", rate=1.0, capacity=1)
        limiter2 = get_rate_limiter("global_singleton", rate=2.0, capacity=2)

        assert limiter1 is limiter2
