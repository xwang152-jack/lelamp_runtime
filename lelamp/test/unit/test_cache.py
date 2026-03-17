"""
Cache Manager 单元测试
"""
import pytest
import asyncio
import time
from lelamp.cache.cache_manager import (
    VisionCache,
    SearchCache,
    TTLCache,
    CacheConfig,
)


@pytest.mark.unit
class TestTTLCache:
    """TTLCache 基础测试套件"""

    @pytest.mark.asyncio
    async def test_basic_set_get(self):
        """测试基本的存取操作"""
        config = CacheConfig(max_size=10, ttl_seconds=60)
        cache = TTLCache(config)

        # 使用位置参数生成键
        await cache.set("value1", None, "key1")
        result = await cache.get("key1")
        assert result == "value1"

        # 不存在的键
        result = await cache.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_ttl_expiration(self):
        """测试 TTL 过期"""
        config = CacheConfig(max_size=10, ttl_seconds=1)
        cache = TTLCache(config)

        await cache.set("value1", None, "key1")
        result = await cache.get("key1")
        assert result == "value1"

        # 等待超过 TTL
        await asyncio.sleep(1.2)
        result = await cache.get("key1")
        assert result is None  # 应该已过期

    @pytest.mark.asyncio
    async def test_custom_ttl(self):
        """测试自定义 TTL"""
        config = CacheConfig(max_size=10, ttl_seconds=60)
        cache = TTLCache(config)

        # 使用自定义 TTL (1秒)
        await cache.set("value1", ttl_seconds=1, key1="key1")
        result = await cache.get(key1="key1")
        assert result == "value1"

        # 等待超过自定义 TTL
        await asyncio.sleep(1.2)
        result = await cache.get(key1="key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_overwrite_existing_key(self):
        """测试覆盖已存在的键"""
        config = CacheConfig(max_size=10, ttl_seconds=60)
        cache = TTLCache(config)

        await cache.set("value1", None, "key1")
        await cache.set("value2", None, "key1")
        result = await cache.get("key1")
        assert result == "value2"

    @pytest.mark.asyncio
    async def test_clear(self):
        """测试清空缓存"""
        config = CacheConfig(max_size=10, ttl_seconds=60)
        cache = TTLCache(config)

        await cache.set("value1", None, "key1")
        await cache.set("value2", None, "key2")
        await cache.clear()

        assert await cache.get("key1") is None
        assert await cache.get("key2") is None

        stats = cache.get_stats()
        assert stats["current_size"] == 0

    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        """测试 LRU 淘汰策略"""
        config = CacheConfig(max_size=2, ttl_seconds=60)
        cache = TTLCache(config)

        await cache.set("value1", None, "key1")
        await asyncio.sleep(0.1)  # 确保时间戳不同
        await cache.set("value2", None, "key2")
        await asyncio.sleep(0.1)
        await cache.set("value3", None, "key3")  # 应该淘汰 key1 (最早过期)

        assert await cache.get("key1") is None  # 已被淘汰
        assert await cache.get("key2") == "value2"
        assert await cache.get("key3") == "value3"

        stats = cache.get_stats()
        assert stats["evictions"] == 1

    @pytest.mark.asyncio
    async def test_stats_tracking(self):
        """测试统计信息跟踪"""
        config = CacheConfig(max_size=10, ttl_seconds=60)
        cache = TTLCache(config)

        # 初始统计
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["evictions"] == 0

        # 添加并获取（命中）
        await cache.set("value1", None, "key1")
        await cache.get("key1")

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 0

        # 获取不存在的键（未命中）
        await cache.get("nonexistent")

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    @pytest.mark.asyncio
    async def test_make_key_with_various_types(self):
        """测试使用不同类型参数生成缓存键"""
        config = CacheConfig(max_size=10, ttl_seconds=60)
        cache = TTLCache(config)

        # 字符串（位置参数）
        await cache.set("value1", None, "string_key")
        assert await cache.get("string_key") == "value1"

        # 整数（位置参数）
        await cache.set("value2", None, 123)
        assert await cache.get(123) == "value2"

        # 列表（位置参数）
        await cache.set("value3", None, [1, 2, 3])
        assert await cache.get([1, 2, 3]) == "value3"

        # 字典（关键字参数）
        await cache.set("value4", None, key="a", param="b")
        assert await cache.get(key="a", param="b") == "value4"


@pytest.mark.unit
class TestVisionCache:
    """VisionCache 测试套件"""

    @pytest.mark.asyncio
    async def test_basic_operations(self):
        """测试基本操作"""
        cache = VisionCache(max_size=10, ttl_seconds=60)

        image_b64 = "fake_base64_image_data"
        question = "What's in this image?"
        response = "This is a cat."

        await cache.set(image_b64, question, response)
        result = await cache.get(image_b64, question)
        assert result == response

    @pytest.mark.asyncio
    async def test_ttl_expiration(self):
        """测试 TTL 过期"""
        cache = VisionCache(max_size=10, ttl_seconds=1)

        image_b64 = "fake_image"
        question = "test question"
        response = "test response"

        await cache.set(image_b64, question, response)
        assert await cache.get(image_b64, question) == response

        await asyncio.sleep(1.2)
        assert await cache.get(image_b64, question) is None

    @pytest.mark.asyncio
    async def test_question_normalization(self):
        """测试问题归一化（大小写、空格）"""
        cache = VisionCache(max_size=10, ttl_seconds=60)

        image_b64 = "fake_image"
        question1 = "  What's This?  "
        question2 = "what's this?"
        response = "A cat."

        await cache.set(image_b64, question1, response)

        # 应该命中缓存（问题归一化后相同）
        result = await cache.get(image_b64, question2)
        assert result == response

    @pytest.mark.asyncio
    async def test_different_images_same_question(self):
        """测试不同图像相同问题"""
        cache = VisionCache(max_size=10, ttl_seconds=60)

        image1 = "image_data_1"
        image2 = "image_data_2"
        question = "What is this?"

        await cache.set(image1, question, "Response 1")
        await cache.set(image2, question, "Response 2")

        assert await cache.get(image1, question) == "Response 1"
        assert await cache.get(image2, question) == "Response 2"

    @pytest.mark.asyncio
    async def test_clear(self):
        """测试清空缓存"""
        cache = VisionCache(max_size=10, ttl_seconds=60)

        await cache.set("image1", "q1", "r1")
        await cache.set("image2", "q2", "r2")
        await cache.clear()

        assert await cache.get("image1", "q1") is None
        assert await cache.get("image2", "q2") is None

    @pytest.mark.asyncio
    async def test_stats(self):
        """测试统计信息"""
        cache = VisionCache(max_size=10, ttl_seconds=60)

        await cache.set("image1", "q1", "r1")
        await cache.get("image1", "q1")  # hit
        await cache.get("image2", "q2")  # miss

        stats = cache.get_stats()
        assert stats["cache_type"] == "vision"
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        """测试 LRU 淘汰"""
        cache = VisionCache(max_size=2, ttl_seconds=60)

        await cache.set("img1", "q1", "r1")
        await asyncio.sleep(0.1)
        await cache.set("img2", "q2", "r2")
        await asyncio.sleep(0.1)
        await cache.set("img3", "q3", "r3")  # 淘汰 img1

        assert await cache.get("img1", "q1") is None
        assert await cache.get("img2", "q2") == "r2"
        assert await cache.get("img3", "q3") == "r3"


@pytest.mark.unit
class TestSearchCache:
    """SearchCache 测试套件"""

    @pytest.mark.asyncio
    async def test_basic_operations(self):
        """测试基本操作"""
        cache = SearchCache(max_size=10, ttl_seconds=60)

        query = "Python programming"
        response = {"results": ["result1", "result2"]}

        await cache.set(query, response)
        result = await cache.get(query)
        assert result == response

    @pytest.mark.asyncio
    async def test_ttl_expiration(self):
        """测试 TTL 过期"""
        cache = SearchCache(max_size=10, ttl_seconds=1)

        query = "test query"
        response = {"results": ["result1"]}

        await cache.set(query, response)
        assert await cache.get(query) == response

        await asyncio.sleep(1.2)
        assert await cache.get(query) is None

    @pytest.mark.asyncio
    async def test_query_normalization(self):
        """测试查询归一化"""
        cache = SearchCache(max_size=10, ttl_seconds=60)

        query1 = "  Python Programming  "
        query2 = "python programming"
        response = {"results": ["result1"]}

        await cache.set(query1, response)
        result = await cache.get(query2)
        assert result == response

    @pytest.mark.asyncio
    async def test_freshness_parameter(self):
        """测试 freshness 参数"""
        cache = SearchCache(max_size=10, ttl_seconds=60)

        query = "news"
        response_day = {"results": ["day result"]}
        response_week = {"results": ["week result"]}

        await cache.set(query, response_day, freshness="oneDay")
        await cache.set(query, response_week, freshness="oneWeek")

        assert await cache.get(query, freshness="oneDay") == response_day
        assert await cache.get(query, freshness="oneWeek") == response_week

    @pytest.mark.asyncio
    async def test_clear(self):
        """测试清空缓存"""
        cache = SearchCache(max_size=10, ttl_seconds=60)

        await cache.set("query1", {"results": ["r1"]})
        await cache.set("query2", {"results": ["r2"]})
        await cache.clear()

        assert await cache.get("query1") is None
        assert await cache.get("query2") is None

    @pytest.mark.asyncio
    async def test_stats(self):
        """测试统计信息"""
        cache = SearchCache(max_size=10, ttl_seconds=60)

        await cache.set("query1", {"results": ["r1"]})
        await cache.get("query1")  # hit
        await cache.get("query2")  # miss

        stats = cache.get_stats()
        assert stats["cache_type"] == "search"
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    @pytest.mark.asyncio
    async def test_different_cache_instances(self):
        """测试 VisionCache 和 SearchCache 独立"""
        vision = VisionCache(max_size=10, ttl_seconds=60)
        search = SearchCache(max_size=10, ttl_seconds=60)

        # 使用完全不同的数据
        await vision.set("image1", "question1", "vision_response")
        await search.set("query1", {"results": ["search_response"]})

        vision_result = await vision.get("image1", "question1")
        search_result = await search.get("query1")

        assert vision_result == "vision_response"
        assert search_result == {"results": ["search_response"]}

        # 验证统计独立
        vision_stats = vision.get_stats()
        search_stats = search.get_stats()

        assert vision_stats["cache_type"] == "vision"
        assert search_stats["cache_type"] == "search"
