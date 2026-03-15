import os
import sys
import json
import time
import shutil
import logging
import hashlib
import urllib.request
from typing import Optional, Dict, Any, Tuple
import threading
import subprocess

logger = logging.getLogger("lelamp.ota")

class OTAManager:
    def __init__(self, current_version: str, update_url: str):
        self.current_version = current_version
        self.update_url = update_url
        self.download_path = "/tmp/lelamp_update_package"
        self._lock = threading.Lock()
        
    def _get_remote_version_info(self) -> Optional[Dict[str, Any]]:
        """
        Fetch version info from remote server.
        Expected JSON format:
        {
            "version": "1.0.1",
            "release_notes": "Fixed bugs...",
            "download_url": "https://api.lelamp.com/download/v1.0.1/lelamp-runtime",
            "sha256": "abcdef..."
        }
        """
        if not self.update_url:
            logger.warning("未配置 OTA 更新服务器地址 (LELAMP_OTA_URL)")
            return None
            
        try:
            req = urllib.request.Request(
                self.update_url, 
                headers={'User-Agent': f'LeLamp-Runtime/{self.current_version}'}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode('utf-8'))
                    return data
        except Exception as e:
            logger.error(f"检查更新失败: {e}")
            return None
        return None

    def check_for_update(self) -> Tuple[bool, str, str]:
        """
        Check if a newer version is available.
        Returns: (has_update, new_version, release_notes)
        """
        info = self._get_remote_version_info()
        if not info:
            return False, "", "无法连接更新服务器"
            
        remote_version = info.get("version")
        if not remote_version:
            return False, "", "无效的版本信息"
            
        # Simple string comparison (for now). Ideally use semver or packaging.version
        if remote_version != self.current_version:
            # Assuming strictly increasing version numbers
            # In production, use packaging.version.parse
            if remote_version > self.current_version:
                return True, remote_version, info.get("release_notes", "")
        
        return False, self.current_version, "当前已是最新版本"

    def perform_update(self) -> str:
        """
        Download and apply update.
        Returns status message.
        """
        with self._lock:
            info = self._get_remote_version_info()
            if not info:
                return "获取更新信息失败"
                
            remote_version = info.get("version")
            if remote_version <= self.current_version:
                return "没有可用的更新"
                
            download_url = info.get("download_url")
            expected_hash = info.get("sha256")
            
            if not download_url:
                return "下载链接无效"
                
            logger.info(f"开始下载更新: {remote_version} from {download_url}")
            
            try:
                # 1. Download
                urllib.request.urlretrieve(download_url, self.download_path)
                
                # 2. Verify Hash
                if expected_hash:
                    sha256_hash = hashlib.sha256()
                    with open(self.download_path, "rb") as f:
                        for byte_block in iter(lambda: f.read(4096), b""):
                            sha256_hash.update(byte_block)
                    if sha256_hash.hexdigest() != expected_hash:
                        return "更新包校验失败 (Hash Mismatch)"
                
                # 3. Apply Update (Replace Binary)
                # This assumes we are running as a standalone binary or have write access
                current_executable = sys.executable if getattr(sys, 'frozen', False) else sys.argv[0]
                
                # Backup
                backup_path = current_executable + ".bak"
                if os.path.exists(current_executable):
                    shutil.copy2(current_executable, backup_path)
                
                # Replace
                # Note: On Linux, you can replace a running binary. On Windows, you cannot.
                # Since target is likely Linux (Raspberry Pi), we proceed.
                if sys.platform == "win32":
                    return "Windows 系统暂不支持在线热更新，请手动下载安装包。"
                
                shutil.move(self.download_path, current_executable)
                os.chmod(current_executable, 0o755)
                
                logger.info("更新文件替换成功，准备重启服务...")
                
                # 4. Restart Service
                # We can exit with a specific code that systemd recognizes to restart
                # Or trigger a reboot
                # For now, we just return success and let the main loop decide when to exit
                return f"更新成功 (v{remote_version})。请重启设备或服务以生效。"

            except Exception as e:
                logger.error(f"更新过程中发生错误: {e}")
                # Restore backup if possible
                return f"更新失败: {str(e)}"

# Singleton instance
_ota_manager = None

def get_ota_manager(current_version: str, update_url: str) -> OTAManager:
    global _ota_manager
    if _ota_manager is None:
        _ota_manager = OTAManager(current_version, update_url)
    return _ota_manager
