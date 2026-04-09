"""
URL 验证工具，用于防止 SSRF (Server-Side Request Forgery) 攻击。
"""
import logging
from urllib.parse import urlparse
import socket
import ipaddress

logger = logging.getLogger("lelamp.url_validation")


def validate_external_url(url: str, allowed_domains: list[str]) -> bool:
    """
    验证 URL 是否安全（防止 SSRF 攻击）

    Args:
        url: 要验证的 URL
        allowed_domains: 允许的域名白名单列表

    Returns:
        True 如果 URL 安全，False 否则
    """
    try:
        parsed = urlparse(url)

        # 1. 只允许 HTTPS
        if parsed.scheme != 'https':
            logger.error(f"URL 必须使用 HTTPS: {url}")
            return False

        # 2. 验证域名白名单
        hostname = parsed.netloc.split(':')[0]  # 移除端口号
        if not any(hostname.endswith(domain) or hostname == domain for domain in allowed_domains):
            logger.error(f"域名不在白名单中: {hostname}")
            return False

        # 3. 防止 SSRF 到内网/保留地址
        # 注意：本检查存在 DNS rebinding 时间窗口（检查与请求之间 DNS 记录可能改变）。
        # 对于安全性要求极高的场景，应使用 HTTP 客户端的 DNS 解析钩子
        # 在实际发起请求时再次校验 IP。
        try:
            # 解析域名到 IP
            ip = socket.gethostbyname(hostname)
            ip_obj = ipaddress.ip_address(ip)

            # 检查是否为危险地址类型
            if (ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local
                    or ip_obj.is_reserved or ip_obj.is_multicast or ip_obj.is_unspecified):
                logger.error(f"禁止访问内网/保留地址: {ip} (from {hostname})")
                return False

        except socket.gaierror as e:
            logger.error(f"DNS 解析失败: {hostname} - {e}")
            return False
        except ValueError as e:
            logger.error(f"无效的 IP 地址: {e}")
            return False

        return True

    except Exception as e:
        logger.error(f"URL 验证失败: {e}")
        return False


# 预定义的域名白名单
ALLOWED_API_DOMAINS = [
    'api.bochaai.com',
    'open.feishu.cn',
    'api-inference.modelscope.cn',
    'api.deepseek.com',
    'aip.baidubce.com',
    'vop.baidu.com',
    'tsn.baidu.com',
]
