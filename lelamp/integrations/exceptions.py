"""
统一异常处理模块
定义集成层的异常类型和错误处理工具
"""

import asyncio
import logging
from functools import wraps
from typing import Any, Callable, TypeVar, ParamSpec, Optional, Awaitable
from dataclasses import dataclass

from livekit.agents._exceptions import APIError

logger = logging.getLogger("lelamp.integrations")


# ============================================================================
# 异常类定义
# ============================================================================

class IntegrationError(Exception):
    """
    集成模块基础异常类

    所有集成相关的异常都应该继承此类，提供统一的错误处理接口。

    Attributes:
        message: 错误消息
        provider: 服务提供商 (e.g., "baidu", "modelscope", "qwen")
        code: 错误码
        retryable: 是否可重试
        original: 原始异常
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        code: str | None = None,
        retryable: bool = False,
        original: Exception | None = None,
    ):
        self.message = message
        self.provider = provider
        self.code = code
        self.retryable = retryable
        self.original = original
        super().__init__(message)

    def __str__(self) -> str:
        parts = [self.message]
        if self.provider:
            parts.append(f"[{self.provider}]")
        if self.code:
            parts.append(f"(code: {self.code})")
        return " ".join(parts)

    def to_dict(self) -> dict:
        """转换为字典格式，便于日志记录"""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "provider": self.provider,
            "code": self.code,
            "retryable": self.retryable,
            "original_type": type(self.original).__name__ if self.original else None,
        }


class AuthenticationError(IntegrationError):
    """认证失败错误 (e.g., API key 无效、token 过期)"""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        original: Exception | None = None,
    ):
        super().__init__(
            message,
            provider=provider,
            code="AUTH_FAILED",
            retryable=False,  # 认证错误通常不可重试
            original=original,
        )


class RateLimitError(IntegrationError):
    """API 速率限制错误"""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        retry_after: float | None = None,
        original: Exception | None = None,
    ):
        super().__init__(
            message,
            provider=provider,
            code="RATE_LIMITED",
            retryable=True,  # 速率限制可以重试
            original=original,
        )
        self.retry_after = retry_after


class NetworkError(IntegrationError):
    """网络连接错误"""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        original: Exception | None = None,
    ):
        super().__init__(
            message,
            provider=provider,
            code="NETWORK_ERROR",
            retryable=True,  # 网络错误可以重试
            original=original,
        )


class ValidationError(IntegrationError):
    """数据验证错误 (e.g., 参数无效、响应格式错误)"""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        field: str | None = None,
        original: Exception | None = None,
    ):
        super().__init__(
            message,
            provider=provider,
            code="VALIDATION_ERROR",
            retryable=False,  # 验证错误不可重试
            original=original,
        )
        self.field = field


class ServiceUnavailableError(IntegrationError):
    """服务不可用错误 (e.g., 5xx 错误、服务维护)"""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        original: Exception | None = None,
    ):
        super().__init__(
            message,
            provider=provider,
            code="SERVICE_UNAVAILABLE",
            retryable=True,  # 服务不可用可以重试
            original=original,
        )


class TimeoutError(IntegrationError):
    """请求超时错误"""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        timeout_seconds: float | None = None,
        original: Exception | None = None,
    ):
        super().__init__(
            message,
            provider=provider,
            code="TIMEOUT",
            retryable=True,  # 超时可以重试
            original=original,
        )
        self.timeout_seconds = timeout_seconds


# ============================================================================
# 重试装饰器
# ============================================================================

P = ParamSpec('P')
T = TypeVar('T')


@dataclass
class RetryConfig:
    """重试配置"""
    max_attempts: int = 3  # 最大尝试次数
    base_delay: float = 1.0  # 基础延迟（秒）
    max_delay: float = 10.0  # 最大延迟（秒）
    exponential_base: float = 2.0  # 指数退避基数
    jitter: bool = True  # 添加随机抖动


def retry_on_error(
    *,
    config: RetryConfig | None = None,
    error_types: tuple[type[IntegrationError], ...] = (
        NetworkError,
        ServiceUnavailableError,
        TimeoutError,
        RateLimitError,
    ),
):
    """
    重试装饰器，对可重试的错误自动重试

    Args:
        config: 重试配置
        error_types: 可重试的错误类型

    Example:
        @retry_on_error(config=RetryConfig(max_attempts=3))
        async def fetch_data():
            ...
    """
    if config is None:
        config = RetryConfig()

    import random

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_error: Exception | None = None

            for attempt in range(1, config.max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except error_types as e:
                    last_error = e

                    # 最后一次尝试不再等待
                    if attempt == config.max_attempts:
                        break

                    # 计算延迟时间（指数退避 + 抖动）
                    delay = min(
                        config.base_delay * (config.exponential_base ** (attempt - 1)),
                        config.max_delay
                    )
                    if config.jitter:
                        delay *= (0.5 + random.random() * 0.5)

                    logger.warning(
                        f"{func.__name__} 失败 (尝试 {attempt}/{config.max_attempts}): "
                        f"{e}, {delay:.2f}s 后重试..."
                    )

                    await asyncio.sleep(delay)

                # 不可重试的错误直接抛出
                except IntegrationError as e:
                    if not e.retryable:
                        logger.error(f"{func.__name__} 遇到不可重试错误: {e}")
                        raise
                    # 其他可重试错误（但不在指定类型中）
                    last_error = e
                    if attempt == config.max_attempts:
                        break
                    await asyncio.sleep(config.base_delay)

            # 所有重试都失败
            assert last_error is not None
            logger.error(
                f"{func.__name__} 在 {config.max_attempts} 次尝试后仍然失败"
            )
            raise last_error

        return wrapper
    return decorator


# ============================================================================
# 降级策略
# ============================================================================

class FallbackStrategy:
    """
    降级策略基类

    当主服务失败时，提供备用方案
    """

    async def get_fallback(self, error: IntegrationError, **context: Any) -> Any:
        """
        获取降级结果

        Args:
            error: 原始错误
            **context: 额外上下文信息

        Returns:
            降级结果
        """
        raise NotImplementedError("子类必须实现 get_fallback 方法")


class SilentFallback(FallbackStrategy):
    """静默降级 - 返回 None 或默认值"""

    def __init__(self, default_value: Any = None):
        self.default_value = default_value

    async def get_fallback(self, error: IntegrationError, **context: Any) -> Any:
        logger.info(f"使用静默降级: {error}")
        return self.default_value


class MessageFallback(FallbackStrategy):
    """消息降级 - 返回友好的错误消息"""

    def __init__(self, message: str = "服务暂时不可用，请稍后再试"):
        self.message = message

    async def get_fallback(self, error: IntegrationError, **context: Any) -> str:
        logger.info(f"使用消息降级: {error}")
        return self.message


class CachedFallback(FallbackStrategy):
    """缓存降级 - 返回最近成功的缓存结果"""

    def __init__(self):
        self._cache: dict[str, tuple[Any, float]] = {}

    async def get_fallback(self, error: IntegrationError, key: str, ttl: float = 300.0, **context: Any) -> Any:
        import time

        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < ttl:
                logger.info(f"使用缓存降级: {key}")
                return value
            else:
                # 缓存过期，删除
                del self._cache[key]

        logger.warning(f"缓存降级失败，无可用缓存: {key}")
        raise error

    def update_cache(self, key: str, value: Any):
        """更新缓存"""
        import time
        self._cache[key] = (value, time.time())


def with_fallback(
    fallback_strategy: FallbackStrategy,
    *,
    log_error: bool = True,
):
    """
    降级装饰器

    Args:
        fallback_strategy: 降级策略
        log_error: 是否记录错误

    Example:
        @with_fallback(MessageFallback("视觉服务暂时不可用"))
        async def describe_image(image):
            ...
    """
    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                result = await func(*args, **kwargs)
                # 如果降级策略支持缓存，更新缓存
                if isinstance(fallback_strategy, CachedFallback) and 'key' in kwargs:
                    fallback_strategy.update_cache(kwargs['key'], result)
                return result
            except IntegrationError as e:
                if log_error:
                    logger.error(f"{func.__name__} 失败，尝试降级: {e}")
                return await fallback_strategy.get_fallback(e, **kwargs)
            except Exception as e:
                # 非集成错误也尝试降级
                if log_error:
                    logger.error(f"{func.__name__} 遇到未预期错误: {e}")
                integration_error = IntegrationError(
                    str(e),
                    code="UNEXPECTED_ERROR",
                    retryable=False,
                    original=e,
                )
                return await fallback_strategy.get_fallback(integration_error, **kwargs)

        return wrapper
    return decorator


# ============================================================================
# 错误转换工具
# ============================================================================

def convert_api_error(error: APIError, provider: str | None = None) -> IntegrationError:
    """
    将 LiveKit APIError 转换为 IntegrationError

    Args:
        error: LiveKit APIError
        provider: 服务提供商名称

    Returns:
        对应的 IntegrationError 子类
    """
    status_code = getattr(error, 'status_code', None)

    # 根据状态码判断错误类型
    if status_code == 401 or status_code == 403:
        return AuthenticationError(
            error.message or "认证失败",
            provider=provider,
            original=error,
        )
    elif status_code == 429:
        return RateLimitError(
            error.message or "API 速率限制",
            provider=provider,
            original=error,
        )
    elif status_code and status_code >= 500:
        return ServiceUnavailableError(
            error.message or "服务不可用",
            provider=provider,
            original=error,
        )
    elif status_code and status_code >= 400:
        return ValidationError(
            error.message or "请求参数无效",
            provider=provider,
            original=error,
        )
    else:
        # 默认为网络错误
        return NetworkError(
            error.message or "网络请求失败",
            provider=provider,
            original=error,
        )


def convert_httpx_error(error: Exception, provider: str | None = None) -> IntegrationError:
    """
    将 httpx 错误转换为 IntegrationError

    Args:
        error: httpx 异常
        provider: 服务提供商名称

    Returns:
        对应的 IntegrationError 子类
    """
    import httpx

    if isinstance(error, httpx.TimeoutException):
        return TimeoutError(
            "请求超时",
            provider=provider,
            original=error,
        )
    elif isinstance(error, httpx.NetworkError):
        return NetworkError(
            "网络连接失败",
            provider=provider,
            original=error,
        )
    elif isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        if status == 401 or status == 403:
            return AuthenticationError(
                "认证失败",
                provider=provider,
                original=error,
            )
        elif status == 429:
            return RateLimitError(
                "API 速率限制",
                provider=provider,
                original=error,
            )
        elif status >= 500:
            return ServiceUnavailableError(
                f"服务不可用 (HTTP {status})",
                provider=provider,
                original=error,
            )
        else:
            return ValidationError(
                f"请求失败 (HTTP {status})",
                provider=provider,
                original=error,
            )
    else:
        return IntegrationError(
            f"未知错误: {str(error)}",
            provider=provider,
            code="UNKNOWN_ERROR",
            retryable=True,
            original=error,
        )
