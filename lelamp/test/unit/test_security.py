"""
安全模块单元测试
"""
import os
import pytest
from unittest.mock import patch
from lelamp.utils.security import (
    get_device_id,
    generate_license_key,
    verify_license,
)


class TestSecurityModule:
    """安全模块测试套件"""

    def test_get_device_id_from_cpuinfo(self):
        """测试从 /proc/cpuinfo 读取设备 ID"""
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', create=True) as mock_open:
                mock_open.return_value.__enter__.return_value = [
                    "processor: 0\n",
                    "Serial: 1000000012345678\n"
                ]
                device_id = get_device_id()
                assert device_id == "1000000012345678"

    def test_get_device_id_fallback_to_mac(self):
        """测试回退到 MAC 地址"""
        with patch('os.path.exists', return_value=False):
            with patch('uuid.getnode', return_value=123456789):
                device_id = get_device_id()
                assert device_id == "123456789"

    def test_generate_license_key_with_secret(self):
        """测试使用自定义密钥生成授权码"""
        device_id = "test_device_123"
        secret = "test_secret_key"

        key = generate_license_key(device_id, secret=secret)

        # 验证返回值为 16 字符
        assert len(key) == 16
        assert isinstance(key, str)

        # 验证一致性（相同输入应产生相同输出）
        key2 = generate_license_key(device_id, secret=secret)
        assert key == key2

    def test_generate_license_key_different_device_ids(self):
        """测试不同设备 ID 产生不同授权码"""
        secret = "test_secret"

        key1 = generate_license_key("device_1", secret=secret)
        key2 = generate_license_key("device_2", secret=secret)

        assert key1 != key2

    def test_generate_license_key_requires_secret(self):
        """测试缺少密钥时抛出异常"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="LELAMP_LICENSE_SECRET 环境变量未设置"):
                generate_license_key("test_device")

    def test_verify_license_success(self):
        """测试授权验证成功"""
        device_id = "test_device_success"
        secret = "test_secret_verify"
        expected_key = generate_license_key(device_id, secret=secret)

        with patch('lelamp.utils.security.get_device_id', return_value=device_id):
            with patch.dict(os.environ, {
                'LELAMP_LICENSE_KEY': expected_key,
                'LELAMP_LICENSE_SECRET': secret
            }):
                assert verify_license() is True

    def test_verify_license_failure_wrong_key(self):
        """测试授权验证失败（错误的密钥）"""
        device_id = "test_device_fail"
        secret = "test_secret"

        with patch('lelamp.utils.security.get_device_id', return_value=device_id):
            with patch.dict(os.environ, {
                'LELAMP_LICENSE_KEY': 'wrong_key_12345',
                'LELAMP_LICENSE_SECRET': secret
            }):
                assert verify_license() is False

    def test_verify_license_missing_key(self):
        """测试授权验证失败（缺少密钥）"""
        with patch('lelamp.utils.security.get_device_id', return_value="test_device"):
            with patch.dict(os.environ, {}, clear=True):
                assert verify_license() is False

    def test_verify_license_dev_mode_bypass(self):
        """测试开发模式跳过验证"""
        with patch.dict(os.environ, {'LELAMP_DEV_MODE': '1'}):
            assert verify_license() is True

        with patch.dict(os.environ, {'LELAMP_DEV_MODE': 'true'}):
            assert verify_license() is True

        with patch.dict(os.environ, {'LELAMP_DEV_MODE': 'yes'}):
            assert verify_license() is True

    def test_license_key_uses_hmac(self):
        """测试授权码使用 HMAC 而非简单哈希"""
        import hmac
        import hashlib

        device_id = "test_device"
        secret = "test_secret"

        # 生成授权码
        key = generate_license_key(device_id, secret=secret)

        # 验证使用 HMAC 算法
        expected = hmac.new(
            secret.encode(),
            device_id.encode(),
            hashlib.sha256
        ).hexdigest()[:16]

        assert key == expected

        # 验证不等于简单 SHA256
        simple_hash = hashlib.sha256(f"{device_id}:{secret}".encode()).hexdigest()[:16]
        assert key != simple_hash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
