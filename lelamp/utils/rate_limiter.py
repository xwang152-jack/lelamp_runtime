"""
API 速率限制器
防止 API 调用过于频繁导致费用失控或服务被封禁
"""

import asyncio
import time
import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger("lelamp.rate_limiter")


@dataclass
class RateLimitConfig:
    """速率限制配置"""
    rate: float = 2.0  # 令牌每秒填充速率
    capacity: int = 10  # 桶容量（最大令牌数）

    def __post_init__(self):
        if self.rate <= 0:
            raise ValueError("Rate must be positive")
        if self.capacity <= 0:
            raise ValueError("Capacity must be positive")


class RateLimiter:
    """
    令牌桶算法速率限制器

    使用令牌桶算法限制 API 调用频率：
    - 桶中有固定数量的令牌（capacity）
    - 令牌以固定速率（rate）填充
    - 每次调用消耗一个或多个令牌
    - 令牌不足时等待或拒绝请求

    Args:
        config: 速率限制配置
        name: 限制器名称（用于日志）
    """

    def __init__(self, config: RateLimitConfig, name: str = "default"):
        self.config = config
        self.name = name

        # 当前令牌数
        self._tokens = float(config.capacity)

        # 上次填充时间
        self._last_time = time.time()

        # 线程锁（用于跨线程安全）
        self._lock = asyncio.Lock()

        # 统计信息
        self._requests_total = 0
        self._requests_allowed = 0
        self._requests_denied = 0
        self._wait_time_total = 0.0

        logger.info(f"RateLimiter '{name}' initialized: rate={config.rate}/s, capacity={config.capacity}")

    async def acquire(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """
        获取令牌

        Args:
            tokens: 需要的令牌数量
            timeout: 最大等待时间（秒），None 表示无限等待

        Returns:
            True 如果成功获取令牌，False 如果超时

        Raises:
            ValueError: 如果 tokens 数量无效
        """
        if tokens <= 0:
            raise ValueError(f"Tokens must be positive, got {tokens}")

        if tokens > self.config.capacity:
            raise ValueError(
                f"Tokens request ({tokens}) exceeds capacity ({self.config.capacity})"
            )

        start_time = time.time()
        self._requests_total += 1

        async with self._lock:
            while True:
                now = time.time()

                # 计算需要填充的时间
                elapsed = now - self._last_time

                # 填充令牌
                if elapsed > 0:
                    new_tokens = elapsed * self.config.rate
                    self._tokens = min(
                        self.config.capacity,
                        self._tokens + new_tokens
                    )
                    self._last_time = now

                # 检查是否有足够的令牌
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    self._requests_allowed += 1
                    wait_time = time.time() - start_time
                    if wait_time > 0:
                        self._wait_time_total += wait_time
                        logger.debug(
                            f"RateLimiter '{self.name}': acquired {tokens} tokens "
                            f"after {wait_time:.2f}s wait"
                        )
                    else:
                        logger.debug(f"RateLimiter '{self.name}': acquired {tokens} tokens immediately")
                    return True

                # 令牌不足，计算需要等待的时间
                tokens_needed = tokens - self._tokens
                wait_time_needed = tokens_needed / self.config.rate

                # 检查超时
                if timeout is not None:
                    elapsed_time = time.time() - start_time
                    if elapsed_time + wait_time_needed > timeout:
                        self._requests_denied += 1
                        logger.warning(
                            f"RateLimiter '{self.name}': timeout after {elapsed_time:.2f}s, "
                            f"needed {wait_time_needed:.2f}s more"
                        )
                        return False

                # 等待后重试
                logger.debug(
                    f"RateLimiter '{self.name}': waiting {wait_time_needed:.2f}s for tokens"
                )
                await asyncio.sleep(wait_time_needed)

    async def __aenter__(self):
        """异步上下文管理器支持"""
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器支持"""
        pass

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "name": self.name,
            "rate": self.config.rate,
            "capacity": self.config.capacity,
            "current_tokens": self._tokens,
            "requests_total": self._requests_total,
            "requests_allowed": self._requests_allowed,
            "requests_denied": self._requests_denied,
            "denial_rate": (
                self._requests_denied / self._requests_total
                if self._requests_total > 0 else 0
            ),
            "avg_wait_time": (
                self._wait_time_total / self._requests_allowed
                if self._requests_allowed > 0 else 0
            ),
        }

    def reset_stats(self):
        """重置统计信息"""
        self._requests_total = 0
        self._requests_allowed = 0
        self._requests_denied = 0
        self._wait_time_total = 0.0
        logger.info(f"RateLimiter '{self.name}': stats reset")


class RateLimiterManager:
    """速率限制器管理器，管理多个 API 的速率限制"""

    def __init__(self):
        self._limiters: dict[str, RateLimiter] = {}

    def create_limiter(
        self,
        name: str,
        rate: float = 2.0,
        capacity: int = 10
    ) -> RateLimiter:
        """
        创建或获取速率限制器

        Args:
            name: 限制器名称
            rate: 令牌每秒填充速率
            capacity: 桶容量

        Returns:
            RateLimiter 实例
        """
        if name not in self._limiters:
            config = RateLimitConfig(rate=rate, capacity=capacity)
            self._limiters[name] = RateLimiter(config, name)
        return self._limiters[name]

    def get_limiter(self, name: str) -> Optional[RateLimiter]:
        """获取已存在的速率限制器"""
        return self._limiters.get(name)

    def get_all_stats(self) -> dict:
        """获取所有限制器的统计信息"""
        return {
            name: limiter.get_stats()
            for name, limiter in self._limiters.items()
        }

    def reset_all_stats(self):
        """重置所有限制器的统计信息"""
        for limiter in self._limiters.values():
            limiter.reset_stats()


# 全局限制器管理器实例
_global_manager = RateLimiterManager()


def get_rate_limiter(
    name: str,
    rate: float = 2.0,
    capacity: int = 10
) -> RateLimiter:
    """
    获取或创建速率限制器（便捷函数）

    Args:
        name: 限制器名称
        rate: 令牌每秒填充速率
        capacity: 桶容量

    Returns:
        RateLimiter 实例
    """
    return _global_manager.create_limiter(name, rate, capacity)


def get_all_rate_limiter_stats() -> dict:
    """获取所有速率限制器的统计信息（便捷函数）"""
    return _global_manager.get_all_stats()
