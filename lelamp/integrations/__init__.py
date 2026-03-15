# LeLamp 集成模块

from .baidu_auth import BaiduAuth
from .baidu_speech import BaiduShortSpeechSTT, BaiduTTS
from .qwen_vl import Qwen3VLClient

# 统一异常处理
from .exceptions import (
    IntegrationError,
    AuthenticationError,
    RateLimitError,
    NetworkError,
    ValidationError,
    ServiceUnavailableError,
    TimeoutError,
    retry_on_error,
    RetryConfig,
    with_fallback,
    FallbackStrategy,
    SilentFallback,
    MessageFallback,
    CachedFallback,
    convert_api_error,
    convert_httpx_error,
)

__all__ = [
    # 集成类
    "BaiduAuth",
    "BaiduShortSpeechSTT",
    "BaiduTTS",
    "Qwen3VLClient",
    # 异常类
    "IntegrationError",
    "AuthenticationError",
    "RateLimitError",
    "NetworkError",
    "ValidationError",
    "ServiceUnavailableError",
    "TimeoutError",
    # 重试和降级
    "retry_on_error",
    "RetryConfig",
    "with_fallback",
    "FallbackStrategy",
    "SilentFallback",
    "MessageFallback",
    "CachedFallback",
    # 错误转换
    "convert_api_error",
    "convert_httpx_error",
]
