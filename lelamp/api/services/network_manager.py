"""
网络连接管理器
处理 WiFi 连接、断开和状态查询
"""
import asyncio
import logging
import subprocess
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class NetworkConnectionManager:
    """网络连接管理器"""

    def __init__(self, interface: str = "wlan0"):
        self.interface = interface

    def connect_wifi(self, ssid: str, password: str, timeout: int = 30) -> Dict[str, any]:
        """连接到指定 WiFi 网络"""
        try:
            logger.info(f"正在连接到 {ssid}...")

            # 使用 nmcli 连接
            cmd = [
                'nmcli',
                'device',
                'wifi',
                'connect',
                ssid,
                'password',
                password
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if result.returncode == 0:
                logger.info(f"成功连接到 {ssid}")

                # 等待获取 IP 地址
                ip_info = self._wait_for_ip(timeout=10)

                return {
                    "success": True,
                    "ssid": ssid,
                    "ip_address": ip_info.get("ip_address"),
                    "message": "连接成功"
                }
            else:
                error_msg = self._parse_error(result.stderr)
                logger.error(f"连接失败: {error_msg}")

                return {
                    "success": False,
                    "error": error_msg,
                    "message": f"连接失败: {error_msg}"
                }

        except subprocess.TimeoutExpired:
            logger.error("连接超时")
            return {
                "success": False,
                "error": "timeout",
                "message": "连接超时"
            }
        except Exception as e:
            logger.error(f"连接异常: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"连接异常: {str(e)}"
            }

    def disconnect_wifi(self) -> bool:
        """断开当前 WiFi 连接"""
        try:
            result = subprocess.run(
                ['nmcli', 'device', 'disconnect', self.interface],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"断开连接失败: {e}")
            return False

    def get_connection_status(self) -> Dict[str, any]:
        """获取当前连接状态"""
        try:
            # 获取连接状态
            result = subprocess.run(
                ['nmcli', '-t', '-f', 'ACTIVE,SSID,IP', 'connection', 'show'],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                return {"connected": False}

            # 解析输出
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue

                parts = line.split(':')
                if len(parts) >= 3 and parts[0] == 'yes':
                    return {
                        "connected": True,
                        "ssid": parts[1],
                        "ip_address": parts[2] if parts[2] else None
                    }

            return {"connected": False}

        except Exception as e:
            logger.error(f"获取连接状态失败: {e}")
            return {"connected": False, "error": str(e)}

    def is_connected(self) -> bool:
        """检查是否已连接到 WiFi"""
        status = self.get_connection_status()
        return status.get("connected", False)

    def get_current_ssid(self) -> Optional[str]:
        """获取当前连接的 SSID"""
        status = self.get_connection_status()
        return status.get("ssid")

    def get_current_ip(self) -> Optional[str]:
        """获取当前 IP 地址"""
        status = self.get_connection_status()
        return status.get("ip_address")

    def test_connection(self) -> Dict[str, any]:
        """测试网络连接"""
        try:
            # 测试 DNS 解析
            import socket
            socket.setdefaulttimeout(5)
            socket.gethostbyname('www.baidu.com')

            return {
                "connected": True,
                "internet_available": True
            }
        except Exception as e:
            return {
                "connected": True,
                "internet_available": False,
                "error": str(e)
            }

    def test_internet_connection(self, host: str = "8.8.8.8", timeout: int = 3) -> bool:
        """测试互联网连接"""
        try:
            result = subprocess.run(
                ['ping', '-c', '1', '-W', str(timeout), host],
                capture_output=True,
                timeout=timeout + 1
            )
            return result.returncode == 0
        except Exception:
            return False

    def _wait_for_ip(self, timeout: int = 10) -> Dict[str, any]:
        """等待获取 IP 地址"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            status = self.get_connection_status()
            if status.get("connected") and status.get("ip_address"):
                return {
                    "ip_address": status["ip_address"]
                }
            time.sleep(1)

        return {"ip_address": None}

    def _parse_error(self, error_output: str) -> str:
        """解析错误信息"""
        output = error_output.lower()

        if "secrets were required" in output or "password" in output:
            return "password_error"
        elif "no network" in output or "not found" in output:
            return "network_not_found"
        elif "timeout" in output:
            return "timeout"
        elif "invalid" in output or "incorrect" in output:
            return "invalid_password"
        else:
            return "connection_failed"

    def get_connection_info(self) -> Dict[str, any]:
        """获取完整的连接信息"""
        status = self.get_connection_status()

        if not status.get("connected"):
            return {
                "connected": False,
                "ssid": None,
                "ip_address": None,
                "internet_available": None
            }

        # 测试互联网连接
        internet_test = self.test_connection()

        return {
            "connected": True,
            "ssid": status.get("ssid"),
            "ip_address": status.get("ip_address"),
            "internet_available": internet_test.get("internet_available", False),
            "interface": self.interface
        }

    async def async_connect_wifi(self, ssid: str, password: str) -> Dict[str, any]:
        """异步连接 WiFi"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.connect_wifi,
            ssid,
            password
        )

    async def async_disconnect_wifi(self) -> bool:
        """异步断开 WiFi"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.disconnect_wifi)

    async def async_get_status(self) -> Dict[str, any]:
        """异步获取连接状态"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_connection_status)
