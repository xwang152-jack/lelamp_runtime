import os
import sys
import json
import time
import shutil
import logging
import hashlib
from typing import Optional, Dict, Any, Tuple
import threading
import subprocess

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    import urllib.request
    import ssl
    import certifi

try:
    from packaging import version as pkg_version
    PACKAGING_AVAILABLE = True
except ImportError:
    PACKAGING_AVAILABLE = False

logger = logging.getLogger("lelamp.ota")

class OTAManager:
    def __init__(self, current_version: str, update_url: str):
        self.current_version = current_version
        self.update_url = update_url
        self.download_path = "/tmp/lelamp_update_package"
        self._lock = threading.Lock()

    async def _get_remote_version_info_async(self) -> Optional[Dict[str, Any]]:
        """
        使用 httpx 异步获取远程版本信息（推荐方式）
        """
        if not self.update_url:
            logger.warning("未配置 OTA 更新服务器地址 (LELAMP_OTA_URL)")
            return None

        # 强制 HTTPS
        if not self.update_url.startswith('https://'):
            logger.error(f"OTA URL 必须使用 HTTPS: {self.update_url}")
            return None

        try:
            async with httpx.AsyncClient(verify=True, timeout=10.0) as client:
                response = await client.get(
                    self.update_url,
                    headers={'User-Agent': f'LeLamp-Runtime/{self.current_version}'}
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            logger.error(f"检查更新失败: {e}")
            return None

    def _get_remote_version_info(self) -> Optional[Dict[str, Any]]:
        """
        Fetch version info from remote server (同步方式，使用 urllib + SSL 验证)
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

        # 强制 HTTPS
        if not self.update_url.startswith('https://'):
            logger.error(f"OTA URL 必须使用 HTTPS: {self.update_url}")
            return None

        try:
            # 创建安全的 SSL 上下文
            if HTTPX_AVAILABLE:
                # 优先使用 httpx (但这是同步方法，所以还是用 urllib)
                pass

            ssl_context = ssl.create_default_context(cafile=certifi.where())
            req = urllib.request.Request(
                self.update_url,
                headers={'User-Agent': f'LeLamp-Runtime/{self.current_version}'}
            )

            with urllib.request.urlopen(req, timeout=10, context=ssl_context) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode('utf-8'))
                    return data
        except ssl.SSLError as e:
            logger.error(f"SSL 证书验证失败: {e}")
            return None
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

        # 使用 packaging.version 进行语义化版本比较
        if PACKAGING_AVAILABLE:
            try:
                if pkg_version.parse(remote_version) > pkg_version.parse(self.current_version):
                    return True, remote_version, info.get("release_notes", "")
            except Exception as e:
                logger.warning(f"版本比较失败，回退到字符串比较: {e}")
                # 回退到简单字符串比较
                if remote_version > self.current_version:
                    return True, remote_version, info.get("release_notes", "")
        else:
            # 回退方案：简单字符串比较
            if remote_version > self.current_version:
                logger.warning("未安装 packaging 库，使用字符串比较版本（不精确）")
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

            # 使用语义化版本比较
            if PACKAGING_AVAILABLE:
                try:
                    if pkg_version.parse(remote_version) <= pkg_version.parse(self.current_version):
                        return "没有可用的更新"
                except Exception:
                    if remote_version <= self.current_version:
                        return "没有可用的更新"
            else:
                if remote_version <= self.current_version:
                    return "没有可用的更新"

            download_url = info.get("download_url")
            expected_hash = info.get("sha256")

            # 强制要求 hash 验证
            if not expected_hash:
                logger.error("更新包缺少 SHA256 校验值")
                return "更新包校验失败：缺少校验值"

            # 强制 HTTPS
            if not download_url or not download_url.startswith('https://'):
                logger.error(f"下载链接必须使用 HTTPS: {download_url}")
                return "下载链接无效或不安全"

            logger.info(f"开始下载更新: {remote_version} from {download_url}")

            try:
                # 1. Download with progress
                self._download_with_progress(download_url)

                # 2. Verify Hash (REQUIRED)
                sha256_hash = hashlib.sha256()
                with open(self.download_path, "rb") as f:
                    for byte_block in iter(lambda: f.read(4096), b""):
                        sha256_hash.update(byte_block)

                actual_hash = sha256_hash.hexdigest()
                if actual_hash != expected_hash:
                    logger.error(f"Hash mismatch: expected {expected_hash}, got {actual_hash}")
                    os.remove(self.download_path)  # Delete corrupted file
                    return "更新包校验失败 (Hash Mismatch)"

                logger.info("更新包校验通过")

                # 3. Apply Update (Replace Binary)
                current_executable = sys.executable if getattr(sys, 'frozen', False) else sys.argv[0]

                # Backup
                backup_path = current_executable + ".bak"
                if os.path.exists(current_executable):
                    shutil.copy2(current_executable, backup_path)
                    logger.info(f"已备份当前版本到: {backup_path}")

                # Replace (Windows check)
                if sys.platform == "win32":
                    return "Windows 系统暂不支持在线热更新，请手动下载安装包。"

                # 移动并设置权限
                shutil.move(self.download_path, current_executable)
                os.chmod(current_executable, 0o755)

                logger.info("更新文件替换成功，准备重启服务...")

                return f"更新成功 (v{remote_version})。请重启设备或服务以生效。"

            except Exception as e:
                logger.error(f"更新过程中发生错误: {e}")

                # Rollback
                backup_path = (sys.executable if getattr(sys, 'frozen', False) else sys.argv[0]) + ".bak"
                current_executable = sys.executable if getattr(sys, 'frozen', False) else sys.argv[0]

                if os.path.exists(backup_path):
                    try:
                        shutil.move(backup_path, current_executable)
                        os.chmod(current_executable, 0o755)
                        logger.info("回滚成功，已恢复原版本")
                    except Exception as rollback_error:
                        logger.error(f"回滚失败: {rollback_error}")
                        return f"更新失败且回滚失败: {str(e)}"

                return f"更新失败（已回滚）: {str(e)}"

    def _download_with_progress(self, url: str) -> None:
        """下载文件并显示进度"""
        if HTTPX_AVAILABLE:
            # 使用 httpx 的同步客户端（或者你可以改成异步版本）
            import httpx
            with httpx.Client(verify=True, timeout=300.0) as client:
                with client.stream("GET", url) as resp:
                    resp.raise_for_status()
                    total = int(resp.headers.get("content-length", 0))
                    with open(self.download_path, "wb") as f:
                        downloaded = 0
                        for chunk in resp.iter_bytes(chunk_size=8192):
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total > 0:
                                progress = (downloaded / total * 100)
                                if downloaded % (1024 * 1024) < 8192:  # 每 MB 输出一次
                                    logger.info(f"下载进度: {progress:.1f}% ({downloaded}/{total} bytes)")
                    logger.info("下载完成")
        else:
            # 回退到 urllib（不支持进度显示）
            logger.warning("建议安装 httpx 以获得更好的下载体验")
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            urllib.request.urlretrieve(url, self.download_path, context=ssl_context)
            logger.info("下载完成")

# Singleton instance
_ota_manager = None

def get_ota_manager(current_version: str, update_url: str) -> OTAManager:
    global _ota_manager
    if _ota_manager is None:
        _ota_manager = OTAManager(current_version, update_url)
    return _ota_manager
