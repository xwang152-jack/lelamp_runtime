"""
Integration tests for API response caching.
"""
import pytest
import time
from fastapi.testclient import TestClient
from fastapi import FastAPI, APIRouter, Request
from pydantic import BaseModel
from lelamp.api.middleware.cache import cache_response, clear_cache, get_cache_stats, cleanup_expired_cache


# 创建测试应用
test_app = FastAPI()
test_router = APIRouter()


# 路由定义(不是测试)
class _TestData(BaseModel):
    value: str


@test_router.get("/test-cached")
@cache_response(ttl_seconds=30)
async def _test_cached_endpoint(request: Request):
    """测试缓存端点"""
    return {"data": "cached_response", "timestamp": time.time()}


@test_router.get("/test-no-cache")
async def _test_no_cache_endpoint(request: Request):
    """无缓存端点"""
    return {"data": "no_cache", "timestamp": time.time()}


@test_router.post("/test-post")
@cache_response(ttl_seconds=30)
async def _test_post_endpoint(request: Request, data: _TestData):
    """POST 端点(不会被缓存)"""
    return {"received": data.value}


@test_router.get("/test-expired")
@cache_response(ttl_seconds=1)
async def _test_expired_cache(request: Request):
    """测试过期缓存"""
    return {"data": "expired", "timestamp": time.time()}


test_app.include_router(test_router, prefix="/api")


# 实际测试函数
def test_cache_hit():
    """测试缓存命中"""
    client = TestClient(test_app)

    # 第一次请求 - 缓存未命中
    response1 = client.get("/api/test-cached")
    assert response1.status_code == 200

    data1 = response1.json()

    # 第二次请求 - 应该返回缓存的数据
    response2 = client.get("/api/test-cached")
    assert response2.status_code == 200

    data2 = response2.json()

    # 验证返回的是相同的数据 (包括时间戳)
    assert data1["data"] == data2["data"]
    assert data1["timestamp"] == data2["timestamp"]  # 时间戳应该相同(来自缓存)


def test_cache_expiration():
    """测试缓存过期"""
    client = TestClient(test_app)

    # 第一次请求
    response1 = client.get("/api/test-expired")
    assert response1.status_code == 200
    data1 = response1.json()

    # 立即再次请求 - 应该命中缓存
    response2 = client.get("/api/test-expired")
    assert response2.status_code == 200
    data2 = response2.json()

    # 验证时间戳相同(来自缓存)
    assert data1["timestamp"] == data2["timestamp"]

    # 等待缓存过期
    time.sleep(1.5)

    # 缓存过期后的请求
    response3 = client.get("/api/test-expired")
    assert response3.status_code == 200
    data3 = response3.json()

    # 时间戳应该不同(新生成的)
    assert data1["timestamp"] != data3["timestamp"]


def test_cache_clear():
    """测试清除缓存"""
    client = TestClient(test_app)

    # 创建缓存
    response1 = client.get("/api/test-cached")
    assert response1.status_code == 200
    data1 = response1.json()

    # 清除所有缓存
    cleared_count = clear_cache()
    assert cleared_count >= 1

    # 再次请求应该生成新数据
    response2 = client.get("/api/test-cached")
    assert response2.status_code == 200
    data2 = response2.json()

    # 时间戳应该不同(缓存被清除)
    assert data1["timestamp"] != data2["timestamp"]


def test_cache_stats():
    """测试缓存统计"""
    # 清空缓存
    clear_cache()

    client = TestClient(test_app)

    # 初始统计
    stats1 = get_cache_stats()
    initial_count = stats1["total_entries"]

    # 添加缓存
    client.get("/api/test-cached")

    # 新统计
    stats2 = get_cache_stats()
    assert stats2["total_entries"] == initial_count + 1
    assert stats2["active_entries"] >= 1


def test_cleanup_expired_cache():
    """测试清理过期缓存"""
    # 清空缓存
    clear_cache()

    # 创建一些会过期的缓存
    client = TestClient(test_app)

    # 添加短期缓存
    client.get("/api/test-expired")

    # 验证缓存存在
    stats1 = get_cache_stats()
    assert stats1["total_entries"] >= 1

    # 等待过期
    time.sleep(1.5)

    # 清理过期缓存
    cleaned = cleanup_expired_cache()
    assert cleaned >= 1

    # 验证统计
    stats2 = get_cache_stats()
    # 过期的缓存应该被清理
    assert stats2["active_entries"] == 0
    assert stats2["expired_entries"] == 0

