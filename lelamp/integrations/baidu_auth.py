"""
百度 API 共享认证类
消除 STT 和 TTS 中重复的 OAuth Token 获取代码
"""

import time
import asyncio
import logging
from typing import Optional

import httpx

from livekit.agents._exceptions import APIError
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS

logger = logging.getLogger("lelamp.integrations.baidu")


class BaiduAuth:
    """
    百度 API 认证管理器

    处理 OAuth Token 的获取、缓存和自动刷新，
    避免在 STT 和 TTS 中重复实现相同逻辑。

    Features:
    - Token 自动缓存和刷新（提前 60 秒）
    - 线程安全的并发访问保护
    - 统一的错误处理
    - 详细的日志记录
    """

    def __init__(
        self,
        api_key: str | None,
        secret_key: str | None,
        oauth_endpoint: str = "https://aip.baidubce.com/oauth/2.0/token",
    ):
        """
        初始化百度认证管理器

        Args:
            api_key: 百度 API Key
            secret_key: 百度 Secret Key
            oauth_endpoint: OAuth 端点 URL
        """
        self._api_key = api_key
        self._secret_key = secret_key
        self._oauth_endpoint = oauth_endpoint

        # Token 缓存
        self._access_token: str | None = None
        self._access_token_expires_at: float = 0.0

        # 并发控制
        self._lock = asyncio.Lock()

        # 统计信息
        self._token_requests = 0
        self._token_refreshes = 0
        self._cache_hits = 0

        logger.info("BaiduAuth initialized")

    async def get_access_token(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> str:
        """
        获取访问令牌（带缓存和自动刷新）

        Args:
            conn_options: API 连接选项

        Returns:
            访问令牌字符串

        Raises:
            APIError: 如果获取令牌失败
        """
        self._token_requests += 1

        # 检查缓存（提前 60 秒刷新）
        if self._access_token and time.time() < self._access_token_expires_at - 60:
            self._cache_hits += 1
            logger.debug("Using cached access token")
            return self._access_token

        # 需要刷新 token
        async with self._lock:
            # 双重检查（可能在等待锁的过程中已被其他协程刷新）
            if self._access_token and time.time() < self._access_token_expires_at - 60:
                self._cache_hits += 1
                return self._access_token

            # 验证凭证
            if not self._api_key or not self._secret_key:
                raise APIError(
                    "百度 API 需要 (api_key, secret_key)",
                    body={"provider": "baidu"},
                    retryable=False,
                )

            # 获取新的 token
            return await self._fetch_token(conn_options)

    async def _fetch_token(
        self,
        conn_options: APIConnectOptions
    ) -> str:
        """
        获取新的访问令牌

        Args:
            conn_options: API 连接选项

        Returns:
            访问令牌字符串

        Raises:
            APIError: 如果获取令牌失败
        """
        self._token_refreshes += 1
        logger.info("Fetching new Baidu access token")

        params = {
            "grant_type": "client_credentials",
            "client_id": self._api_key,
            "client_secret": self._secret_key,
        }

        try:
            async with httpx.AsyncClient(timeout=conn_options.timeout) as client:
                resp = await client.get(self._oauth_endpoint, params=params)
                resp.raise_for_status()
                data = resp.json()

        except httpx.HTTPStatusError as e:
            body = None
            try:
                body = e.response.json()
            except Exception:
                pass

            error_msg = f"百度 OAuth HTTP {e.response.status_code}"
            logger.error(f"{error_msg}: {body}")
            raise APIError(
                error_msg,
                body=body,
                retryable=e.response.status_code >= 500,
            ) from e

        except Exception as e:
            logger.error(f"百度 OAuth 请求失败: {e}")
            raise APIError(
                "百度 OAuth 请求失败",
                body={"error": str(e)},
                retryable=True,
            ) from e

        # 解析响应
        access_token = data.get("access_token")
        expires_in = data.get("expires_in", 0)

        if not access_token:
            logger.error(f"百度 OAuth 响应缺少 access_token: {data}")
            raise APIError(
                "百度 OAuth 响应缺少 access_token",
                body=data,
                retryable=False,
            )

        # 缓存 token
        self._access_token = access_token
        self._access_token_expires_at = time.time() + float(expires_in or 0)

        logger.info(
            f"Successfully fetched access token, expires in {expires_in}s, "
            f"will refresh at {self._access_token_expires_at - 60:.0f}"
        )

        return access_token

    def invalidate_cache(self):
        """使缓存失效（强制下次重新获取）"""
        self._access_token = None
        self._access_token_expires_at = 0.0
        logger.info("Token cache invalidated")

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "token_requests": self._token_requests,
            "token_refreshes": self._token_refreshes,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": (
                self._cache_hits / self._token_requests
                if self._token_requests > 0 else 0
            ),
            "token_valid": (
                self._access_token is not None and
                time.time() < self._access_token_expires_at
            ),
            "token_expires_at": self._access_token_expires_at,
        }

    def reset_stats(self):
        """重置统计信息"""
        self._token_requests = 0
        self._token_refreshes = 0
        self._cache_hits = 0
        logger.info("BaiduAuth stats reset")
