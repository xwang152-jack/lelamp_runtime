"""
API 响应缓存中间件

提供基于内存的 API 响应缓存功能,减少重复查询和数据库负载。
"""
from fastapi import Request, Response
from functools import wraps
import hashlib
import json
import logging
import time
from typing import Optional, Dict, Tuple, Any

logger = logging.getLogger("lelamp.api.cache")

# 简单的内存缓存 - 存储响应数据而不是 Response 对象
_cache: Dict[str, Tuple[Dict[str, Any], float]] = {}  # key: (response_data, expiry_timestamp)


def get_cache_key(request: Request) -> Optional[str]:
    """
    生成缓存键

    基于请求方法和 URL 生成唯一键

    Args:
        request: FastAPI 请求对象

    Returns:
        缓存键,仅对 GET 请求返回
    """
    if request.method != "GET":
        return None

    # 基于 URL 和查询参数生成键
    url = str(request.url)
    key = hashlib.md5(url.encode()).hexdigest()
    logger.debug(f"Generated cache key: {key} for URL: {url}")
    return key


def get_cache_stats() -> Dict[str, Any]:
    """
    获取缓存统计信息

    Returns:
        包含缓存大小和统计信息的字典
    """
    current_time = time.time()
    active_entries = sum(1 for _, expiry in _cache.values() if expiry > current_time)
    expired_entries = len(_cache) - active_entries

    return {
        "total_entries": len(_cache),
        "active_entries": active_entries,
        "expired_entries": expired_entries,
        "cache_keys": list(_cache.keys())
    }


def cache_response(ttl_seconds: int = 60):
    """
    缓存响应装饰器

    Args:
        ttl_seconds: 缓存生存时间(秒),默认60秒

    Note:
        被装饰的函数必须接受 Request 作为第一个参数或关键字参数
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 尝试从 args 或 kwargs 中获取 Request 对象
            request = None

            # 检查 args
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            # 检查 kwargs
            if not request:
                for key, value in kwargs.items():
                    if isinstance(value, Request):
                        request = value
                        break

            if not request:
                # 没有 Request 对象,无法使用缓存,直接执行函数
                logger.debug(f"No Request object found for {func.__name__}, skipping cache")
                return await func(*args, **kwargs)

            cache_key = get_cache_key(request)

            # 尝试从缓存获取
            if cache_key:
                logger.debug(f"Checking cache for key: {cache_key}, cache size: {len(_cache)}")
                if cache_key in _cache:
                    cached_data, expiry = _cache[cache_key]
                    if expiry > time.time():
                        logger.debug(f"Cache hit: {cache_key}")
                        # 返回缓存的数据
                        return cached_data
                    else:
                        # 缓存过期,删除
                        del _cache[cache_key]
                        logger.debug(f"Cache expired: {cache_key}")
                else:
                    logger.debug(f"Cache miss: {cache_key}")

            # 执行原函数
            response = await func(*args, **kwargs)

            # 缓存响应数据 (只缓存成功的响应)
            if cache_key:
                # 尝试获取响应数据
                response_data = None

                if hasattr(response, 'model_dump'):
                    # Pydantic 模型
                    response_data = response.model_dump()
                elif hasattr(response, 'dict'):
                    # 旧版 Pydantic
                    response_data = response.dict()
                elif isinstance(response, dict):
                    response_data = response
                elif hasattr(response, 'body'):
                    # FastAPI Response
                    try:
                        import asyncio
                        body = response.body
                        if hasattr(body, '__aiter__'):
                            # 异步迭代器
                            body_parts = []
                            async def read_body():
                                async for chunk in body:
                                    body_parts.append(chunk)
                                return b''.join(body_parts)
                            body_bytes = asyncio.run(read_body())
                        else:
                            body_bytes = body

                        response_data = json.loads(body_bytes)
                    except Exception as e:
                        logger.warning(f"Failed to extract response data: {e}")
                        response_data = None

                if response_data is not None:
                    _cache[cache_key] = (response_data, time.time() + ttl_seconds)
                    logger.debug(f"Cache set: {cache_key} (TTL: {ttl_seconds}s)")
                    logger.debug(f"Cache size after set: {len(_cache)}")

            return response
        return wrapper
    return decorator


def clear_cache(pattern: Optional[str] = None) -> int:
    """
    清除缓存

    Args:
        pattern: 可选的模式匹配,如果提供则只删除匹配的缓存

    Returns:
        删除的缓存条目数
    """
    if pattern:
        keys_to_delete = [k for k in _cache.keys() if pattern in k]
        for key in keys_to_delete:
            del _cache[key]
        logger.info(f"Cleared {len(keys_to_delete)} cache entries matching '{pattern}'")
        return len(keys_to_delete)
    else:
        count = len(_cache)
        _cache.clear()
        logger.info(f"Cleared all cache ({count} entries)")
        return count


def cleanup_expired_cache() -> int:
    """
    清理过期的缓存条目

    Returns:
        清理的条目数
    """
    current_time = time.time()
    expired_keys = [
        key for key, (_, expiry) in _cache.items()
        if expiry <= current_time
    ]

    for key in expired_keys:
        del _cache[key]

    if expired_keys:
        logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

    return len(expired_keys)
