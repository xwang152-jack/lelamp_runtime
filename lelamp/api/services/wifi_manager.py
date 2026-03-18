"""
WiFi 管理服务

使用 nmcli (NetworkManager) 进行 WiFi 配置
需要在树莓派上安装 networkmanager: sudo apt-get install networkmanager
"""
import asyncio
import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class WiFiNetwork:
    """WiFi 网络信息"""
    ssid: str
    bssid: str
    signal_strength: int  # 0-100
    security: str  # WPA2, WPA3, WEP, open
    frequency: str  # 2.4GHz / 5GHz
    is_hidden: bool = False


class WiFiManager:
    """
    WiFi 管理器

    使用 nmcli 命令进行 WiFi 网络管理
    需要 sudo 权限执行 nmcli 命令
    """

    def __init__(self, nmcli_path: str = "nmcli"):
        """
        初始化 WiFi 管理器

        Args:
            nmcli_path: nmcli 命令路径，默认为系统 PATH 中的 nmcli
        """
        self._nmcli_path = nmcli_path
        self._scan_lock = asyncio.Lock()
        self._connect_lock = asyncio.Lock()

    async def scan_networks(self, timeout: int = 10) -> List[WiFiNetwork]:
        """
        扫描可用 WiFi 网络

        Args:
            timeout: 扫描超时时间（秒）

        Returns:
            可用网络列表，按信号强度降序排序
        """
        async with self._scan_lock:
            try:
                # 使用 nmcli 扫描网络
                # -t: terse 格式, -f: 指定字段
                cmd = [
                    "sudo", self._nmcli_path, "-t", "-f",
                    "SSID,BSSID,SIGNAL,SECURITY,FREQ",
                    "device", "wifi", "list"
                ]

                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )

                if process.returncode != 0:
                    error_msg = stderr.decode().strip()
                    logger.error(f"WiFi scan failed: {error_msg}")
                    return []

                # 解析输出
                networks = []
                for line in stdout.decode().strip().split('\n'):
                    if not line:
                        continue

                    parts = line.split(':')
                    if len(parts) >= 5:
                        ssid = parts[0] or "-- Hidden Network --"
                        bssid = parts[1]
                        signal = int(float(parts[2])) if parts[2] else 0  # 信号质量
                        security = parts[3] if parts[3] else "open"

                        # 解析频率
                        try:
                            freq_mhz = int(parts[4].replace(' MHz', '')) if parts[4] else 2400
                            frequency = "5GHz" if freq_mhz > 4000 else "2.4GHz"
                        except (ValueError, IndexError):
                            frequency = "2.4GHz"

                        # 规范化安全类型
                        if security.lower() in ("", "none"):
                            security = "open"

                        networks.append(WiFiNetwork(
                            ssid=ssid,
                            bssid=bssid,
                            signal_strength=signal,
                            security=security,
                            frequency=frequency,
                            is_hidden=ssid == "-- Hidden Network --"
                        ))

                # 按信号强度排序
                networks.sort(key=lambda n: n.signal_strength, reverse=True)
                return networks

            except asyncio.TimeoutError:
                logger.error("WiFi scan timeout")
                return []
            except FileNotFoundError:
                logger.error(f"nmcli not found at {self._nmcli_path}")
                return []
            except Exception as e:
                logger.error(f"WiFi scan error: {e}", exc_info=True)
                return []

    async def get_status(self) -> dict:
        """
        获取当前 WiFi 连接状态

        Returns:
            状态信息字典，包含:
            - connected: 是否已连接
            - ssid: 连接的网络名称
            - signal_strength: 信号强度 (0-100)
            - ip_address: IP 地址
            - gateway: 网关地址
            - dns_servers: DNS 服务器列表
        """
        try:
            # 获取连接状态
            cmd = [self._nmcli_path, "-t", "-f", "ACTIVE,SSID,SIGNAL", "connection", "show"]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, _ = await process.communicate()

            connected = False
            ssid = None
            signal = None

            for line in stdout.decode().strip().split('\n'):
                if not line:
                    continue
                parts = line.split(':')
                if len(parts) >= 3 and parts[0] == 'yes':
                    connected = True
                    ssid = parts[1]
                    try:
                        signal = int(float(parts[2]))
                    except (ValueError, IndexError):
                        signal = None
                    break

            # 获取 IP 地址
            ip_address = None
            if connected:
                cmd = [self._nmcli_path, "-t", "-f", "IP4.ADDRESS", "device", "show", "wlan0"]
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, _ = await process.communicate()
                for line in stdout.decode().strip().split('\n'):
                    if 'IP4.ADDRESS' in line:
                        parts = line.split(':')
                        if len(parts) > 1:
                            ip_address = parts[1].split('/')[0]
                            break

            return {
                "connected": connected,
                "ssid": ssid,
                "signal_strength": signal,
                "ip_address": ip_address,
                "gateway": None,  # 需要额外命令获取
                "dns_servers": []  # 需要额外命令获取
            }

        except Exception as e:
            logger.error(f"Get WiFi status error: {e}", exc_info=True)
            return {
                "connected": False,
                "ssid": None,
                "signal_strength": None,
                "ip_address": None,
                "gateway": None,
                "dns_servers": []
            }

    async def connect(self, ssid: str, password: Optional[str] = None) -> dict:
        """
        连接到 WiFi 网络

        Args:
            ssid: 网络名称
            password: WiFi 密码（开放网络可为 None）

        Returns:
            连接结果字典，包含 success, message, ssid
        """
        async with self._connect_lock:
            try:
                # 先删除旧连接（如果存在）
                await self._delete_connection(ssid)

                # 创建新连接
                cmd = ["sudo", self._nmcli_path, "device", "wifi", "connect", ssid]

                if password:
                    cmd.extend(["password", password])

                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=30  # 连接可能需要较长时间
                )

                if process.returncode == 0:
                    logger.info(f"Successfully connected to {ssid}")
                    return {"success": True, "message": "连接成功", "ssid": ssid}
                else:
                    error_msg = stderr.decode().strip()
                    logger.error(f"Failed to connect to {ssid}: {error_msg}")
                    return {"success": False, "message": f"连接失败: {error_msg}", "ssid": ssid}

            except asyncio.TimeoutError:
                logger.error(f"WiFi connect timeout for {ssid}")
                return {"success": False, "message": "连接超时", "ssid": ssid}
            except Exception as e:
                logger.error(f"WiFi connect error: {e}", exc_info=True)
                return {"success": False, "message": str(e), "ssid": ssid}

    async def disconnect(self) -> bool:
        """
        断开当前 WiFi 连接

        Returns:
            是否成功断开
        """
        try:
            # 获取当前连接的 SSID
            status = await self.get_status()
            if status["connected"] and status["ssid"]:
                # 断开指定连接
                cmd = ["sudo", self._nmcli_path, "connection", "down", status["ssid"]]
            else:
                # 断开 wlan0
                cmd = ["sudo", self._nmcli_path, "connection", "down", "wlan0"]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            return process.returncode == 0
        except Exception as e:
            logger.error(f"WiFi disconnect error: {e}", exc_info=True)
            return False

    async def _delete_connection(self, ssid: str) -> None:
        """
        删除指定名称的连接

        Args:
            ssid: 要删除的连接名称（SSID）
        """
        try:
            cmd = ["sudo", self._nmcli_path, "connection", "delete", ssid]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
        except Exception:
            pass  # 忽略删除错误

    async def check_available(self) -> bool:
        """
        检查 nmcli 是否可用

        Returns:
            nmcli 是否可用
        """
        try:
            process = await asyncio.create_subprocess_exec(
                self._nmcli_path, "-version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            return process.returncode == 0
        except Exception:
            return False


# 全局单例实例
wifi_manager = WiFiManager()
