"""
LLM 响应缓存模块
使用 TTL 缓存避免重复调用相同的视觉问题和搜索查询
"""

import asyncio
import hashlib
import json
import logging
import time
from typing import Any, Optional, Dict
from dataclasses import dataclass

logger = logging.getLogger("lelamp.cache")


@dataclass
class CacheConfig:
    """缓存配置"""
    max_size: int = 100  # 最大缓存条目数
    ttl_seconds: int = 300  # 默认 5 分钟过期
    enable_stats: bool = True  # 启用统计


class TTLCache:
    """
    带 TTL 的异步缓存

    Features:
    - 自动过期清理
    - 大小限制（LRU 淘汰）
    - 缓存统计
    - 线程安全
    """

    def __init__(self, config: CacheConfig):
        self._config = config
        self._cache: Dict[str, tuple[Any, float]] = {}  # key: (value, expires_at)
        self._lock = asyncio.Lock()

        # 统计信息
        self._hits = 0
        self._misses = 0
        self._evictions = 0

        logger.info(
            f"TTLCache initialized: max_size={config.max_size}, "
            f"ttl={config.ttl_seconds}s"
        )

    def _make_key(self, *args, **kwargs) -> str:
        """生成缓存键"""
        key_parts = []

        # 处理位置参数
        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
            elif isinstance(arg, (list, tuple)):
                key_parts.append(str(list(arg)))
            elif isinstance(arg, dict):
                key_parts.append(str(sorted(arg.items())))
            else:
                # 对于复杂对象，使用字符串表示
                key_parts.append(str(arg))

        # 处理关键字参数
        if kwargs:
            key_parts.append(str(sorted(kwargs.items())))

        key_string = ":".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    async def get(self, *args, **kwargs) -> Optional[Any]:
        """
        从缓存获取值

        Returns:
            缓存的值，如果不存在或已过期则返回 None
        """
        key = self._make_key(*args, **kwargs)

        async with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            value, expires_at = self._cache[key]

            # 检查是否过期
            if time.time() > expires_at:
                del self._cache[key]
                self._misses += 1
                logger.debug(f"Cache expired for key: {key[:8]}...")
                return None

            self._hits += 1
            logger.debug(f"Cache hit for key: {key[:8]}...")
            return value

    async def set(self, value: Any, ttl_seconds: Optional[int] = None, *args, **kwargs):
        """
        设置缓存值

        Args:
            value: 要缓存的值
            ttl_seconds: 过期时间（秒），None 表示使用默认 TTL
            *args, **kwargs: 用于生成缓存键的参数
        """
        if ttl_seconds is None:
            ttl_seconds = self._config.ttl_seconds

        key = self._make_key(*args, **kwargs)

        async with self._lock:
            # 检查缓存大小，必要时淘汰
            if len(self._cache) >= self._config.max_size:
                # 淘汰最旧的条目（简单的 LRU）
                oldest_key = min(
                    self._cache.keys(),
                    key=lambda k: self._cache[k][1]  # 按过期时间排序
                )
                del self._cache[oldest_key]
                self._evictions += 1
                logger.debug(f"Evicted key: {oldest_key[:8]}...")

            expires_at = time.time() + ttl_seconds
            self._cache[key] = (value, expires_at)

    async def clear(self):
        """清空缓存"""
        async with self._lock:
            cleared_count = len(self._cache)
            self._cache.clear()
            logger.info(f"Cleared {cleared_count} entries from cache")

    def get_stats(self) -> dict:
        """获取统计信息"""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0

        return {
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": hit_rate,
            "current_size": len(self._cache),
            "max_size": self._config.max_size,
        }


class VisionCache:
    """
    视觉 API 响应专用缓存

    针对视觉模型的特点进行优化：
    - 图像 + 问题作为缓存键
    - 较长的默认 TTL（视觉内容通常不变）
    - 更大的缓存容量
    """

    def __init__(self, max_size: int = 50, ttl_seconds: int = 600):
        """
        初始化视觉缓存

        Args:
            max_size: 最大缓存数量（视觉响应较大）
            ttl_seconds: 默认 10 分钟过期
        """
        config = CacheConfig(
            max_size=max_size,
            ttl_seconds=ttl_seconds,
            enable_stats=True
        )
        self._cache = TTLCache(config)
        logger.info("VisionCache initialized")

    def _make_vision_key(self, image_jpeg_b64: str, question: str) -> str:
        """生成视觉缓存键"""
        # 使用图像哈希（避免存储大图像）
        image_hash = hashlib.md5(image_jpeg_b64.encode()).hexdigest()
        question_normalized = question.strip().lower()
        return f"vision:{image_hash}:{question_normalized}"

    async def get(self, image_jpeg_b64: str, question: str) -> Optional[str]:
        """获取缓存的视觉响应"""
        key = self._make_vision_key(image_jpeg_b64, question)
        return await self._cache.get(key)

    async def set(self, image_jpeg_b64: str, question: str, response: str, ttl_seconds: Optional[int] = None):
        """缓存视觉响应"""
        key = self._make_vision_key(image_jpeg_b64, question)
        await self._cache.set(response, ttl_seconds, key)

    async def clear(self):
        """清空视觉缓存"""
        await self._cache.clear()

    def get_stats(self) -> dict:
        """获取统计信息"""
        base_stats = self._cache.get_stats()
        return {
            **base_stats,
            "cache_type": "vision"
        }


class SearchCache:
    """
    搜索 API 响应专用缓存

    针对搜索特点进行优化：
    - 查询作为缓存键
    - 较短的默认 TTL（新闻内容变化快）
    - 中等缓存容量
    """

    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        """
        初始化搜索缓存

        Args:
            max_size: 最大缓存数量
            ttl_seconds: 默认 5 分钟过期
        """
        config = CacheConfig(
            max_size=max_size,
            ttl_seconds=ttl_seconds,
            enable_stats=True
        )
        self._cache = TTLCache(config)
        logger.info("SearchCache initialized")

    def _make_search_key(self, query: str, freshness: str = "oneDay") -> str:
        """生成搜索缓存键"""
        query_normalized = query.strip().lower()
        return f"search:{freshness}:{query_normalized}"

    async def get(self, query: str, freshness: str = "oneDay") -> Optional[dict]:
        """获取缓存的搜索响应"""
        key = self._make_search_key(query, freshness)
        return await self._cache.get(key)

    async def set(self, query: str, response: dict, freshness: str = "oneDay", ttl_seconds: Optional[int] = None):
        """缓存搜索响应"""
        key = self._make_search_key(query, freshness)
        await self._cache.set(response, ttl_seconds, key)

    async def clear(self):
        """清空搜索缓存"""
        await self._cache.clear()

    def get_stats(self) -> dict:
        """获取统计信息"""
        base_stats = self._cache.get_stats()
        return {
            **base_stats,
            "cache_type": "search"
        }
