"""
测试 lelamp.cache.cache_manager 模块
"""
import asyncio
import time
import pytest

from lelamp.cache.cache_manager import (
    CacheConfig,
    TTLCache,
    VisionCache,
    SearchCache,
)


@pytest.mark.unit
class TestCacheConfig:
    """测试缓存配置"""

    def test_defaults(self):
        cfg = CacheConfig()
        assert cfg.max_size == 100
        assert cfg.ttl_seconds == 300
        assert cfg.enable_stats is True

    def test_custom(self):
        cfg = CacheConfig(max_size=50, ttl_seconds=60)
        assert cfg.max_size == 50
        assert cfg.ttl_seconds == 60


@pytest.mark.unit
class TestTTLCache:
    """测试 TTL 缓存"""

    @pytest.mark.asyncio
    async def test_miss(self):
        cache = TTLCache(CacheConfig(max_size=10, ttl_seconds=60))
        result = await cache.get("nonexistent_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_and_get(self):
        cache = TTLCache(CacheConfig(max_size=10, ttl_seconds=60))
        await cache.set("value1", 60, "key_part1")
        result = await cache.get("key_part1")
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_expiry(self):
        cache = TTLCache(CacheConfig(max_size=10, ttl_seconds=1))
        await cache.set("value1", 0, "expiring_key")
        # 等待过期
        time.sleep(0.05)
        result = await cache.get("expiring_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_custom_ttl(self):
        cache = TTLCache(CacheConfig(max_size=10, ttl_seconds=300))
        await cache.set("short_lived", 0, "key1")
        result = await cache.get("key1")
        assert result is None  # 已过期

    @pytest.mark.asyncio
    async def test_eviction(self):
        cache = TTLCache(CacheConfig(max_size=2, ttl_seconds=60))
        await cache.set("v1", 60, "k1")
        await cache.set("v2", 60, "k2")
        await cache.set("v3", 60, "k3")

        # 第一个条目应该被淘汰
        result = await cache.get("k1")
        assert result is None

        # 新条目应该存在
        assert await cache.get("k3") == "v3"

    @pytest.mark.asyncio
    async def test_clear(self):
        cache = TTLCache(CacheConfig(max_size=10, ttl_seconds=60))
        await cache.set("v1", 60, "k1")
        await cache.set("v2", 60, "k2")
        await cache.clear()
        assert await cache.get("k1") is None
        assert await cache.get("k2") is None

    @pytest.mark.asyncio
    async def test_stats(self):
        cache = TTLCache(CacheConfig(max_size=10, ttl_seconds=60))
        await cache.set("v1", 60, "k1")
        await cache.get("k1")  # hit
        await cache.get("k2")  # miss

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["current_size"] == 1
        assert stats["max_size"] == 10
        assert 0 < stats["hit_rate"] <= 1.0

    @pytest.mark.asyncio
    async def test_stats_no_requests(self):
        cache = TTLCache(CacheConfig(max_size=10, ttl_seconds=60))
        stats = cache.get_stats()
        assert stats["hit_rate"] == 0

    @pytest.mark.asyncio
    async def test_key_with_kwargs(self):
        cache = TTLCache(CacheConfig(max_size=10, ttl_seconds=60))
        await cache.set("v1", 60, "k1", option="a")
        result = await cache.get("k1", option="a")
        assert result == "v1"

        # 不同 kwargs = 不同 key
        result2 = await cache.get("k1", option="b")
        assert result2 is None


@pytest.mark.unit
class TestVisionCache:
    """测试视觉缓存"""

    @pytest.mark.asyncio
    async def test_set_and_get(self):
        cache = VisionCache(max_size=10, ttl_seconds=60)
        await cache.set("image_data", "what is this?", "A cat")
        result = await cache.get("image_data", "what is this?")
        assert result == "A cat"

    @pytest.mark.asyncio
    async def test_miss(self):
        cache = VisionCache(max_size=10, ttl_seconds=60)
        result = await cache.get("no_image", "question?")
        assert result is None

    @pytest.mark.asyncio
    async def test_question_normalization(self):
        cache = VisionCache(max_size=10, ttl_seconds=60)
        await cache.set("img", "What Is This?", "answer")
        result = await cache.get("img", "what is this?")  # lowercased
        assert result == "answer"

    @pytest.mark.asyncio
    async def test_clear(self):
        cache = VisionCache(max_size=10, ttl_seconds=60)
        await cache.set("img", "q", "a")
        await cache.clear()
        assert await cache.get("img", "q") is None

    def test_stats_type(self):
        cache = VisionCache(max_size=10, ttl_seconds=60)
        stats = cache.get_stats()
        assert stats["cache_type"] == "vision"


@pytest.mark.unit
class TestSearchCache:
    """测试搜索缓存"""

    @pytest.mark.asyncio
    async def test_set_and_get(self):
        cache = SearchCache(max_size=10, ttl_seconds=60)
        await cache.set("lelamp robot", {"result": "found"})
        result = await cache.get("lelamp robot")
        assert result == {"result": "found"}

    @pytest.mark.asyncio
    async def test_freshness_parameter(self):
        cache = SearchCache(max_size=10, ttl_seconds=60)
        await cache.set("query", {"r": 1}, freshness="oneDay")
        result = await cache.get("query", freshness="oneDay")
        assert result == {"r": 1}

        # 不同 freshness = 不同 key
        result2 = await cache.get("query", freshness="oneWeek")
        assert result2 is None

    @pytest.mark.asyncio
    async def test_clear(self):
        cache = SearchCache(max_size=10, ttl_seconds=60)
        await cache.set("q", {"r": 1})
        await cache.clear()
        assert await cache.get("q") is None

    def test_stats_type(self):
        cache = SearchCache(max_size=10, ttl_seconds=60)
        stats = cache.get_stats()
        assert stats["cache_type"] == "search"
