"""
测试 lelamp.integrations.exceptions 模块
"""
import asyncio
import pytest
from unittest.mock import patch, MagicMock, Mock

from lelamp.integrations.exceptions import (
    IntegrationError,
    AuthenticationError,
    RateLimitError,
    NetworkError,
    ValidationError,
    ServiceUnavailableError,
    TimeoutError,
    RetryConfig,
    retry_on_error,
    SilentFallback,
    MessageFallback,
    CachedFallback,
    FallbackStrategy,
    with_fallback,
    convert_api_error,
    convert_httpx_error,
)


@pytest.mark.unit
class TestIntegrationError:
    """测试基础异常类"""

    def test_basic_attributes(self):
        err = IntegrationError("test error", provider="baidu", code="TEST", retryable=True)
        assert err.message == "test error"
        assert err.provider == "baidu"
        assert err.code == "TEST"
        assert err.retryable is True
        assert err.original is None

    def test_str_representation(self):
        err = IntegrationError("fail", provider="deepseek", code="E001")
        s = str(err)
        assert "fail" in s
        assert "[deepseek]" in s
        assert "(code: E001)" in s

    def test_str_without_optional(self):
        err = IntegrationError("simple error")
        assert str(err) == "simple error"

    def test_to_dict(self):
        original = ValueError("inner")
        err = IntegrationError("outer", provider="qwen", code="C1", original=original)
        d = err.to_dict()
        assert d["error_type"] == "IntegrationError"
        assert d["message"] == "outer"
        assert d["provider"] == "qwen"
        assert d["code"] == "C1"
        assert d["original_type"] == "ValueError"

    def test_to_dict_no_original(self):
        err = IntegrationError("no original")
        d = err.to_dict()
        assert d["original_type"] is None

    def test_catch_as_exception(self):
        with pytest.raises(IntegrationError):
            raise IntegrationError("caught")


@pytest.mark.unit
class TestExceptionSubclasses:
    """测试异常子类"""

    def test_authentication_error(self):
        err = AuthenticationError("unauthorized", provider="baidu")
        assert err.code == "AUTH_FAILED"
        assert err.retryable is False
        assert err.provider == "baidu"

    def test_rate_limit_error(self):
        err = RateLimitError("too many", provider="deepseek", retry_after=5.0)
        assert err.code == "RATE_LIMITED"
        assert err.retryable is True
        assert err.retry_after == 5.0

    def test_network_error(self):
        err = NetworkError("connection lost")
        assert err.code == "NETWORK_ERROR"
        assert err.retryable is True

    def test_validation_error(self):
        err = ValidationError("bad param", field="image")
        assert err.code == "VALIDATION_ERROR"
        assert err.retryable is False
        assert err.field == "image"

    def test_service_unavailable_error(self):
        err = ServiceUnavailableError("503")
        assert err.code == "SERVICE_UNAVAILABLE"
        assert err.retryable is True

    def test_timeout_error(self):
        err = TimeoutError("timed out", timeout_seconds=30.0)
        assert err.code == "TIMEOUT"
        assert err.retryable is True
        assert err.timeout_seconds == 30.0


@pytest.mark.unit
class TestRetryConfig:
    """测试重试配置"""

    def test_defaults(self):
        cfg = RetryConfig()
        assert cfg.max_attempts == 3
        assert cfg.base_delay == 1.0
        assert cfg.max_delay == 10.0
        assert cfg.exponential_base == 2.0
        assert cfg.jitter is True

    def test_custom(self):
        cfg = RetryConfig(max_attempts=5, base_delay=0.5, jitter=False)
        assert cfg.max_attempts == 5
        assert cfg.base_delay == 0.5
        assert cfg.jitter is False


@pytest.mark.unit
class TestRetryOnError:
    """测试重试装饰器"""

    @pytest.mark.asyncio
    async def test_success_no_retry(self):
        @retry_on_error(config=RetryConfig(max_attempts=3))
        async def succeed():
            return "ok"

        result = await succeed()
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_retry_then_succeed(self):
        call_count = 0

        @retry_on_error(config=RetryConfig(max_attempts=3, base_delay=0.01, jitter=False))
        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise NetworkError("transient", provider="test")
            return "recovered"

        result = await flaky()
        assert result == "recovered"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self):
        @retry_on_error(config=RetryConfig(max_attempts=2, base_delay=0.01, jitter=False))
        async def always_fail():
            raise NetworkError("permanent", provider="test")

        with pytest.raises(NetworkError, match="permanent"):
            await always_fail()

    @pytest.mark.asyncio
    async def test_non_retryable_error_propagates(self):
        @retry_on_error(config=RetryConfig(max_attempts=3))
        async def bad_input():
            raise ValidationError("invalid")

        with pytest.raises(ValidationError):
            await bad_input()


@pytest.mark.unit
class TestFallbackStrategies:
    """测试降级策略"""

    @pytest.mark.asyncio
    async def test_silent_fallback(self):
        fb = SilentFallback(default_value=None)
        result = await fb.get_fallback(IntegrationError("err"))
        assert result is None

    @pytest.mark.asyncio
    async def test_silent_fallback_custom(self):
        fb = SilentFallback(default_value="fallback_val")
        result = await fb.get_fallback(IntegrationError("err"))
        assert result == "fallback_val"

    @pytest.mark.asyncio
    async def test_message_fallback(self):
        fb = MessageFallback(message="服务不可用")
        result = await fb.get_fallback(IntegrationError("err"))
        assert result == "服务不可用"

    @pytest.mark.asyncio
    async def test_cached_fallback_hit(self):
        fb = CachedFallback()
        fb.update_cache("key1", "cached_value")

        result = await fb.get_fallback(IntegrationError("err"), key="key1")
        assert result == "cached_value"

    @pytest.mark.asyncio
    async def test_cached_fallback_miss(self):
        fb = CachedFallback()

        with pytest.raises(IntegrationError):
            await fb.get_fallback(IntegrationError("err"), key="missing")

    @pytest.mark.asyncio
    async def test_base_fallback_not_implemented(self):
        fb = FallbackStrategy()
        with pytest.raises(NotImplementedError):
            await fb.get_fallback(IntegrationError("err"))


@pytest.mark.unit
class TestWithFallback:
    """测试降级装饰器"""

    @pytest.mark.asyncio
    async def test_success_no_fallback(self):
        fb = SilentFallback()

        @with_fallback(fb)
        async def succeed():
            return "ok"

        assert await succeed() == "ok"

    @pytest.mark.asyncio
    async def test_error_triggers_fallback(self):
        fb = MessageFallback(message="降级结果")

        @with_fallback(fb)
        async def fail():
            raise NetworkError("down")

        result = await fail()
        assert result == "降级结果"

    @pytest.mark.asyncio
    async def test_cached_fallback_updates_on_success(self):
        fb = CachedFallback()

        @with_fallback(fb)
        async def succeed_with_key(key="k1"):
            return "value1"

        await succeed_with_key(key="k1")
        # 缓存应该被更新
        result = await fb.get_fallback(IntegrationError("err"), key="k1")
        assert result == "value1"


@pytest.mark.unit
class TestConvertApiError:
    """测试 APIError 转换"""

    def test_convert_401(self):
        mock_err = MagicMock()
        mock_err.message = "Unauthorized"
        mock_err.status_code = 401
        result = convert_api_error(mock_err, provider="livekit")
        assert isinstance(result, AuthenticationError)
        assert result.provider == "livekit"

    def test_convert_429(self):
        mock_err = MagicMock()
        mock_err.message = "Too Many"
        mock_err.status_code = 429
        result = convert_api_error(mock_err)
        assert isinstance(result, RateLimitError)

    def test_convert_500(self):
        mock_err = MagicMock()
        mock_err.message = "Server Error"
        mock_err.status_code = 500
        result = convert_api_error(mock_err)
        assert isinstance(result, ServiceUnavailableError)

    def test_convert_400(self):
        mock_err = MagicMock()
        mock_err.message = "Bad Request"
        mock_err.status_code = 400
        result = convert_api_error(mock_err)
        assert isinstance(result, ValidationError)

    def test_convert_no_status(self):
        mock_err = MagicMock()
        mock_err.message = "Unknown"
        mock_err.status_code = None
        result = convert_api_error(mock_err)
        assert isinstance(result, NetworkError)


@pytest.mark.unit
class TestConvertHttpxError:
    """测试 httpx 错误转换"""

    def test_convert_timeout(self):
        import httpx
        err = httpx.TimeoutException("timeout")
        result = convert_httpx_error(err, provider="baidu")
        assert isinstance(result, TimeoutError)
        assert result.provider == "baidu"

    def test_convert_network(self):
        import httpx
        err = httpx.NetworkError("conn refused")
        result = convert_httpx_error(err)
        assert isinstance(result, NetworkError)

    def test_convert_http_401(self):
        import httpx
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        err = httpx.HTTPStatusError("401", request=MagicMock(), response=mock_resp)
        result = convert_httpx_error(err)
        assert isinstance(result, AuthenticationError)

    def test_convert_http_429(self):
        import httpx
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        err = httpx.HTTPStatusError("429", request=MagicMock(), response=mock_resp)
        result = convert_httpx_error(err)
        assert isinstance(result, RateLimitError)

    def test_convert_http_500(self):
        import httpx
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        err = httpx.HTTPStatusError("500", request=MagicMock(), response=mock_resp)
        result = convert_httpx_error(err)
        assert isinstance(result, ServiceUnavailableError)

    def test_convert_http_400(self):
        import httpx
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        err = httpx.HTTPStatusError("400", request=MagicMock(), response=mock_resp)
        result = convert_httpx_error(err)
        assert isinstance(result, ValidationError)

    def test_convert_unknown(self):
        err = RuntimeError("weird error")
        result = convert_httpx_error(err)
        assert isinstance(result, IntegrationError)
        assert result.code == "UNKNOWN_ERROR"
        assert result.retryable is True
