"""
Captive Portal 中间件

AP 模式下拦截手机连通性检测请求，重定向到配网页面。
支持 iOS/Android/Windows/Mac 的自动检测机制。
"""
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import RedirectResponse
import re

logger = logging.getLogger(__name__)

# 各平台的 Captive Portal 检测 URL
CAPTIVE_PORTAL_HOSTS = {
    # Apple
    "captive.apple.com",
    "www.apple.com",
    "apple.com",
    # Android
    "connectivitycheck.gstatic.com",
    "clients3.google.com",
    "connectivitycheck.android.com",
    # Windows
    "www.msftconnecttest.com",
    "msftconnecttest.com",
    "www.msftncsi.com",
    "msftncsi.com",
    # Firefox / 其他
    "detectportal.firefox.com",
}

# 匹配 generate_204 或 hotspot-detect 等路径的模式
_CAPTIVE_PATHS = re.compile(
    r"/(generate_204|hotspot-detect\.html|success\.txt|canonical.html|connecttest\.txt|ncsi\.txt)",
    re.IGNORECASE,
)


class CaptivePortalMiddleware(BaseHTTPMiddleware):
    """
    AP 模式下的 Captive Portal 中间件。

    工作原理：
    1. 手机连上 WiFi 热点后自动发送 HTTP 请求到特定 URL 检测网络
    2. dnsmasq 将所有 DNS 解析到 192.168.4.1
    3. 本中间件拦截这些请求，返回 302 重定向到配网页面
    """

    def __init__(self, app, setup_url: str = "/setup"):
        super().__init__(app)
        self._setup_url = setup_url

    async def dispatch(self, request: Request, call_next):
        host = request.headers.get("host", "").split(":")[0].lower()
        path = request.url.path.lower()

        # 只拦截非本地请求（排除 localhost、192.168.4.1 本机 IP）
        if host in ("localhost", "127.0.0.1", "192.168.4.1"):
            return await call_next(request)

        # 匹配 Captive Portal 检测请求
        is_captive_host = host in CAPTIVE_PORTAL_HOSTS
        is_captive_path = bool(_CAPTIVE_PATHS.match(path))

        if is_captive_host or is_captive_path:
            logger.info(f"Captive Portal redirect: {host}{path} -> {self._setup_url}")
            return RedirectResponse(url=self._setup_url, status_code=302)

        # 对其他非本机 Host 的请求也重定向到配网页面
        if host not in ("", "lelamp.local") and not path.startswith("/api/"):
            logger.info(f"Non-API redirect: {host}{path} -> {self._setup_url}")
            return RedirectResponse(url=self._setup_url, status_code=302)

        return await call_next(request)
