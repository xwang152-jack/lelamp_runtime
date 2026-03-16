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

        # 3. 防止 SSRF 到内网地址
        try:
            # 解析域名到 IP
            ip = socket.gethostbyname(hostname)
            ip_obj = ipaddress.ip_address(ip)

            # 检查是否为私有 IP 或回环地址
            if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local:
                logger.error(f"禁止访问内网地址: {ip} (from {hostname})")
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
