"""
Integration tests for rate limiting middleware.
"""
import pytest
import asyncio
from lelamp.api.middleware.rate_limit import rate_limiter, RATE_LIMITS


@pytest.fixture
def reset_rate_limiter():
    """每个测试后重置速率限制器"""
    yield
    rate_limiter._requests.clear()


@pytest.mark.usefixtures("reset_rate_limiter")
@pytest.mark.asyncio
async def test_rate_limiter_direct():
    """直接测试速率限制器逻辑"""
    # 测试允许请求
    allowed, info = await rate_limiter.is_allowed("test_client", 10, 60)
    assert allowed is True
    assert info["remaining"] == 9

    # 发送10个请求
    for _ in range(10):
        await rate_limiter.is_allowed("test_client", 10, 60)

    # 第11个请求应该被拒绝
    allowed, info = await rate_limiter.is_allowed("test_client", 10, 60)
    assert allowed is False
    assert info["remaining"] == 0


@pytest.mark.usefixtures("reset_rate_limiter")
def test_rate_limit_reset():
    """测试速率限制重置"""
    # 发送一些请求
    async def send_requests():
        for _ in range(50):
            await rate_limiter.is_allowed("test_client2", 100, 60)

    asyncio.run(send_requests())

    # 重置
    rate_limiter.reset("test_client2")

    # 现在应该可以再次发送请求
    async def check_reset():
        allowed, _ = await rate_limiter.is_allowed("test_client2", 100, 60)
        return allowed

    assert asyncio.run(check_reset()) is True


def test_rate_limit_configuration():
    """测试速率限制配置"""
    assert "default" in RATE_LIMITS
    assert "strict" in RATE_LIMITS
    assert "loose" in RATE_LIMITS

    assert RATE_LIMITS["default"] == (100, 60)
    assert RATE_LIMITS["strict"] == (20, 60)
    assert RATE_LIMITS["loose"] == (1000, 60)


@pytest.mark.usefixtures("reset_rate_limiter")
@pytest.mark.asyncio
async def test_rate_limit_window_cleanup():
    """测试时间窗口清理"""
    import time

    # 发送5个请求
    for _ in range(5):
        await rate_limiter.is_allowed("test_client3", 10, 1)

    # 等待2秒 (超过1秒窗口)
    time.sleep(2)

    # 现在应该可以再次发送请求
    allowed, info = await rate_limiter.is_allowed("test_client3", 10, 1)
    assert allowed is True
    assert info["remaining"] >= 9  # 因为旧请求被清理了


