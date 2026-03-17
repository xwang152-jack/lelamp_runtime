"""
URL 验证模块单元测试
"""
import pytest
from unittest.mock import patch
from lelamp.utils.url_validation import validate_external_url, ALLOWED_API_DOMAINS


class TestURLValidation:
    """URL 验证测试套件"""

    def test_valid_https_url(self):
        """测试有效的 HTTPS URL"""
        url = "https://api.bochaai.com/v1/search"
        assert validate_external_url(url, ALLOWED_API_DOMAINS) is True

    def test_reject_http_url(self):
        """测试拒绝 HTTP URL"""
        url = "http://api.bochaai.com/v1/search"
        assert validate_external_url(url, ALLOWED_API_DOMAINS) is False

    def test_reject_unlisted_domain(self):
        """测试拒绝不在白名单的域名"""
        url = "https://evil.example.com/attack"
        assert validate_external_url(url, ALLOWED_API_DOMAINS) is False

    def test_allowed_domain_exact_match(self):
        """测试精确匹配的域名"""
        url = "https://api.deepseek.com/v1/chat"
        assert validate_external_url(url, ALLOWED_API_DOMAINS) is True

    def test_allowed_domain_subdomain(self):
        """测试子域名匹配"""
        url = "https://sub.api.bochaai.com/endpoint"
        assert validate_external_url(url, ALLOWED_API_DOMAINS) is True

    def test_reject_private_ip_localhost(self):
        """测试拒绝 localhost"""
        url = "https://localhost:8080/api"

        with patch('socket.gethostbyname', return_value='127.0.0.1'):
            # 需要先通过域名检查，所以添加到白名单
            assert validate_external_url(url, ['localhost']) is False

    def test_reject_private_ip_10_network(self):
        """测试拒绝 10.x.x.x 内网 IP"""
        url = "https://internal.example.com/api"

        with patch('socket.gethostbyname', return_value='10.0.0.1'):
            assert validate_external_url(url, ['internal.example.com']) is False

    def test_reject_private_ip_192_network(self):
        """测试拒绝 192.168.x.x 内网 IP"""
        url = "https://router.local/admin"

        with patch('socket.gethostbyname', return_value='192.168.1.1'):
            assert validate_external_url(url, ['router.local']) is False

    def test_reject_private_ip_172_network(self):
        """测试拒绝 172.16-31.x.x 内网 IP"""
        url = "https://docker.internal/api"

        with patch('socket.gethostbyname', return_value='172.17.0.1'):
            assert validate_external_url(url, ['docker.internal']) is False

    def test_reject_link_local_ip(self):
        """测试拒绝链路本地地址 169.254.x.x"""
        url = "https://device.local/config"

        with patch('socket.gethostbyname', return_value='169.254.1.1'):
            assert validate_external_url(url, ['device.local']) is False

    def test_reject_dns_failure(self):
        """测试 DNS 解析失败时拒绝"""
        import socket
        url = "https://nonexistent.example.com/api"

        with patch('socket.gethostbyname', side_effect=socket.gaierror("DNS lookup failed")):
            assert validate_external_url(url, ['nonexistent.example.com']) is False

    def test_valid_public_ip(self):
        """测试允许公网 IP"""
        url = "https://api.example.com/endpoint"

        with patch('socket.gethostbyname', return_value='8.8.8.8'):  # Google DNS
            assert validate_external_url(url, ['api.example.com']) is True

    def test_url_with_port(self):
        """测试带端口号的 URL"""
        url = "https://api.bochaai.com:443/v1/search"
        assert validate_external_url(url, ALLOWED_API_DOMAINS) is True

    def test_url_with_path_and_query(self):
        """测试带路径和查询参数的 URL"""
        url = "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id"
        assert validate_external_url(url, ALLOWED_API_DOMAINS) is True

    def test_malformed_url(self):
        """测试格式错误的 URL"""
        url = "not-a-valid-url"
        assert validate_external_url(url, ALLOWED_API_DOMAINS) is False

    def test_empty_url(self):
        """测试空 URL"""
        url = ""
        assert validate_external_url(url, ALLOWED_API_DOMAINS) is False

    def test_allowed_domains_list(self):
        """测试预定义的域名白名单"""
        expected_domains = [
            'api.bochaai.com',
            'open.feishu.cn',
            'api-inference.modelscope.cn',
            'api.deepseek.com',
            'aip.baidubce.com',
            'vop.baidu.com',
            'tsn.baidu.com',
        ]

        for domain in expected_domains:
            assert domain in ALLOWED_API_DOMAINS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
