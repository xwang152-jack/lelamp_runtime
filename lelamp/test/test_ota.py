"""
OTA 更新模块单元测试
"""
import pytest
from unittest.mock import patch, MagicMock
from lelamp.utils.ota import OTAManager


class TestOTAManager:
    """OTA 管理器测试套件"""

    def test_initialization(self):
        """测试 OTA 管理器初始化"""
        manager = OTAManager("1.0.0", "https://api.lelamp.com/ota/check")

        assert manager.current_version == "1.0.0"
        assert manager.update_url == "https://api.lelamp.com/ota/check"
        assert manager.download_path == "/tmp/lelamp_update_package"

    def test_check_for_update_no_url(self):
        """测试未配置 OTA URL 的情况"""
        manager = OTAManager("1.0.0", "")

        has_update, version, notes = manager.check_for_update()

        assert has_update is False
        assert "无法连接更新服务器" in notes

    def test_check_for_update_requires_https(self):
        """测试强制 HTTPS"""
        manager = OTAManager("1.0.0", "http://api.lelamp.com/ota/check")

        has_update, version, notes = manager.check_for_update()

        assert has_update is False

    @patch('lelamp.utils.ota.PACKAGING_AVAILABLE', True)
    def test_check_for_update_newer_version_available(self):
        """测试检测到新版本"""
        manager = OTAManager("1.0.0", "https://api.lelamp.com/ota/check")

        mock_response = {
            "version": "1.1.0",
            "release_notes": "Bug fixes and improvements",
            "download_url": "https://api.lelamp.com/download/v1.1.0",
            "sha256": "abc123"
        }

        with patch.object(manager, '_get_remote_version_info', return_value=mock_response):
            has_update, version, notes = manager.check_for_update()

        assert has_update is True
        assert version == "1.1.0"
        assert "Bug fixes" in notes

    @patch('lelamp.utils.ota.PACKAGING_AVAILABLE', True)
    def test_check_for_update_already_latest(self):
        """测试已是最新版本"""
        manager = OTAManager("1.1.0", "https://api.lelamp.com/ota/check")

        mock_response = {
            "version": "1.1.0",
            "release_notes": "Current version",
        }

        with patch.object(manager, '_get_remote_version_info', return_value=mock_response):
            has_update, version, notes = manager.check_for_update()

        assert has_update is False
        assert "当前已是最新版本" in notes

    @patch('lelamp.utils.ota.PACKAGING_AVAILABLE', False)
    def test_check_for_update_fallback_string_comparison(self):
        """测试回退到字符串比较（无 packaging 库）"""
        manager = OTAManager("1.0.0", "https://api.lelamp.com/ota/check")

        mock_response = {
            "version": "1.0.1",
            "release_notes": "Patch release",
        }

        with patch.object(manager, '_get_remote_version_info', return_value=mock_response):
            has_update, version, notes = manager.check_for_update()

        assert has_update is True

    def test_perform_update_requires_sha256(self):
        """测试更新需要 SHA256 校验值"""
        manager = OTAManager("1.0.0", "https://api.lelamp.com/ota/check")

        mock_response = {
            "version": "1.1.0",
            "download_url": "https://api.lelamp.com/download/v1.1.0",
            # 缺少 sha256
        }

        with patch.object(manager, '_get_remote_version_info', return_value=mock_response):
            result = manager.perform_update()

        assert "缺少校验值" in result

    def test_perform_update_requires_https_download(self):
        """测试下载链接必须使用 HTTPS"""
        manager = OTAManager("1.0.0", "https://api.lelamp.com/ota/check")

        mock_response = {
            "version": "1.1.0",
            "download_url": "http://api.lelamp.com/download/v1.1.0",  # HTTP
            "sha256": "abc123"
        }

        with patch.object(manager, '_get_remote_version_info', return_value=mock_response):
            result = manager.perform_update()

        assert "不安全" in result or "无效" in result

    def test_perform_update_hash_mismatch(self):
        """测试 Hash 不匹配时拒绝更新"""
        import tempfile
        import os

        manager = OTAManager("1.0.0", "https://api.lelamp.com/ota/check")

        # 创建临时文件模拟下载
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"fake update content")
            temp_path = f.name

        manager.download_path = temp_path

        mock_response = {
            "version": "1.1.0",
            "download_url": "https://api.lelamp.com/download/v1.1.0",
            "sha256": "wrong_hash_value_12345678901234567890123456789012"
        }

        try:
            with patch.object(manager, '_get_remote_version_info', return_value=mock_response):
                with patch.object(manager, '_download_with_progress'):
                    result = manager.perform_update()

            assert "Hash Mismatch" in result
            # 验证文件已被删除
            assert not os.path.exists(temp_path)
        finally:
            # 清理
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_perform_update_windows_not_supported(self):
        """测试 Windows 系统不支持热更新"""
        manager = OTAManager("1.0.0", "https://api.lelamp.com/ota/check")

        mock_response = {
            "version": "1.1.0",
            "download_url": "https://api.lelamp.com/download/v1.1.0",
            "sha256": "a" * 64
        }

        with patch('sys.platform', 'win32'):
            with patch.object(manager, '_get_remote_version_info', return_value=mock_response):
                with patch.object(manager, '_download_with_progress'):
                    with patch('hashlib.sha256'):
                        result = manager.perform_update()

        assert "Windows" in result and "不支持" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
