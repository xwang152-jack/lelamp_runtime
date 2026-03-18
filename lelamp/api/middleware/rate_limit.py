"""
Rate limiting middleware for API endpoints.

Prevents abuse and ensures fair usage of API resources.
"""
from fastapi import Request, HTTPException, status
from typing import Dict, Tuple
import time
import logging
from collections import defaultdict

logger = logging.getLogger("lelamp.api.middleware.rate_limit")


class RateLimiter:
    """
    内存中的速率限制器

    使用滑动窗口算法限制请求频率
    """

    def __init__(self):
        # Dict: identifier -> [(timestamp, count)]
        self._requests: Dict[str, list] = defaultdict(list)
        self._lock = None

    async def is_allowed(
        self,
        identifier: str,
        max_requests: int,
        window_seconds: int
    ) -> Tuple[bool, Dict[str, int]]:
        """
        检查是否允许请求

        Args:
            identifier: 唯一标识符 (IP地址或用户ID)
            max_requests: 时间窗口内最大请求数
            window_seconds: 时间窗口(秒)

        Returns:
            (是否允许, 剩余信息)
        """
        current_time = time.time()

        # 获取该标识符的请求历史
        request_history = self._requests[identifier]

        # 移除时间窗口外的旧请求
        cutoff_time = current_time - window_seconds
        self._requests[identifier] = [
            (ts, count) for ts, count in request_history if ts > cutoff_time
        ]

        # 计算当前窗口内的请求数
        current_count = sum(count for _, count in self._requests[identifier])

        # 计算剩余请求数
        remaining = max(0, max_requests - current_count)
        reset_time = int(current_time + window_seconds)

        if current_count >= max_requests:
            logger.warning(f"速率限制触发: {identifier}, 请求数={current_count}")
            return False, {
                "remaining": 0,
                "reset": reset_time,
                "limit": max_requests
            }

        # 记录本次请求
        self._requests[identifier].append((current_time, 1))

        return True, {
            "remaining": remaining - 1,
            "reset": reset_time,
            "limit": max_requests
        }

    def reset(self, identifier: str) -> None:
        """重置指定标识符的限制计数"""
        if identifier in self._requests:
            del self._requests[identifier]


# 全局速率限制器实例
rate_limiter = RateLimiter()


# 速率限制配置
RATE_LIMITS = {
    "default": (100, 60),  # 100 请求/分钟
    "strict": (20, 60),    # 20 请求/分钟 (敏感操作)
    "loose": (1000, 60),   # 1000 请求/分钟 (公开端点)
}


async def check_rate_limit(
    request: Request,
    limit_type: str = "default"
) -> None:
    """
    速率限制依赖 - 检查请求是否超过限制

    Args:
        request: FastAPI 请求对象
        limit_type: 限制类型 (default/strict/loose)

    Raises:
        HTTPException: 超过速率限制时
    """
    # 获取客户端标识符
    # 优先使用用户ID,否则使用IP地址
    user_id = getattr(request.state, "user_id", None)
    identifier = str(user_id) if user_id else request.client.host

    max_requests, window_seconds = RATE_LIMITS.get(
        limit_type,
        RATE_LIMITS["default"]
    )

    # 检查是否允许请求
    allowed, info = await rate_limiter.is_allowed(
        identifier,
        max_requests,
        window_seconds
    )

    # 将限制信息存储到请求状态
    request.state.rate_limit = info

    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "Rate limit exceeded",
                "limit": info["limit"],
                "reset": info["reset"]
            },
            headers={
                "X-RateLimit-Limit": str(info["limit"]),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(info["reset"]),
                "Retry-After": str(window_seconds)
            }
        )
