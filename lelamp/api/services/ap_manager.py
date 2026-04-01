"""
AP 模式管理器

负责创建和管理 WiFi 热点（Access Point 模式）
使用 hostapd 和 dnsmasq 实现热点功能
"""
import asyncio
import logging
import os
import secrets
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class APConfig:
    """AP 热点配置"""
    ssid: str = "LeLamp-Setup"
    password: str | None = None  # None 时自动生成随机密码
    ip_address: str = "192.168.4.1"
    netmask: str = "255.255.255.0"
    channel: int = 6
    hw_mode: str = "g"  # a, b, g, n, ac
    interface: str = "wlan0"
    # Captive Portal 相关配置
    captive_portal_enabled: bool = False
    portal_port: int = 8080
    dhcp_start: str = "192.168.4.100"
    dhcp_end: str = "192.168.4.200"

    def __post_init__(self):
        """验证配置参数"""
        import re
        import ipaddress

        # 验证 SSID
        if not self.ssid or len(self.ssid) > 32:
            raise ValueError("SSID 必须在 1-32 个字符之间")

        # 验证密码 (WPA2 要求 8-63 个 ASCII 字符，None 时跳过)
        if self.password is not None:
            if not (8 <= len(self.password) <= 63):
                raise ValueError("密码长度必须在 8-63 个字符之间")
            if not self.password.encode('utf-8').isascii():
                raise ValueError("密码只能包含 ASCII 字符")

        # 验证频道
        if not (1 <= self.channel <= 14):
            raise ValueError("频道必须在 1-14 之间")

        # 验证硬件模式
        if self.hw_mode not in ("a", "b", "g", "n", "ac"):
            raise ValueError(f"无效的硬件模式: {self.hw_mode}")

        # 验证 IP 地址格式
        try:
            ipaddress.IPv4Address(self.ip_address)
        except ValueError:
            raise ValueError(f"无效的 IP 地址: {self.ip_address}")


@dataclass
class ClientInfo:
    """连接的客户端信息"""
    mac: str
    ip: str
    connected_at: float


class APManager:
    """
    AP 模式管理器

    功能:
    - 启动/停止 AP 模式
    - 使用 hostapd 创建 WiFi 热点
    - 使用 dnsmasq 提供 DHCP 和 DNS
    - 管理 iptables NAT 规则
    - 获取已连接的客户端列表
    """

    def __init__(
        self,
        config: Optional[APConfig] = None,
        hostapd_conf_path: str = "/etc/hostapd/hostapd.conf",
        dnsmasq_conf_path: str = "/etc/dnsmasq.conf",
        pid_dir: str = "/var/run"
    ):
        """
        初始化 AP 管理器

        Args:
            config: AP 配置，默认使用 LeLamp-Setup
            hostapd_conf_path: hostapd 配置文件路径
            dnsmasq_conf_path: dnsmasq 配置文件路径
            pid_dir: PID 文件目录
        """
        self._config = config or APConfig()
        self._hostapd_conf_path = hostapd_conf_path
        self._dnsmasq_conf_path = dnsmasq_conf_path
        self._pid_dir = Path(pid_dir)

        # PID 文件路径
        self._hostapd_pid = self._pid_dir / "hostapd_lelamp.pid"
        self._dnsmasq_pid = self._pid_dir / "dnsmasq_lelamp.pid"

        self._is_running = False
        self._lock = asyncio.Lock()

    async def start_ap_mode(self) -> dict:
        """
        启动 AP 模式

        流程:
        1. 停止现有的 WiFi 连接
        2. 配置静态 IP
        3. 生成 hostapd 配置
        4. 生成 dnsmasq 配置
        5. 启动 hostapd
        6. 启动 dnsmasq
        7. 配置 iptables NAT

        Returns:
            操作结果字典，包含 success, message, ip_address
        """
        async with self._lock:
            if self._is_running:
                return {"success": True, "message": "AP 模式已在运行", "ip_address": self._config.ip_address}

            try:
                # 自动生成随机密码
                if self._config.password is None:
                    self._config.password = secrets.token_urlsafe(6)
                    logger.info(f"Generated random AP password: {self._config.password}")

                logger.info(f"Starting AP mode: {self._config.ssid}")

                # 1. 停止现有的 WiFi 连接
                await self._stop_wifi_client()

                # 2. 配置静态 IP
                if not await self._configure_static_ip():
                    return {"success": False, "message": "配置静态 IP 失败"}

                # 3. 生成并写入 hostapd 配置
                if not await self._write_hostapd_config():
                    return {"success": False, "message": "生成 hostapd 配置失败"}

                # 4. 生成并写入 dnsmasq 配置
                if not await self._write_dnsmasq_config():
                    return {"success": False, "message": "生成 dnsmasq 配置失败"}

                # 5. 启动 hostapd
                if not await self._start_hostapd():
                    return {"success": False, "message": "启动 hostapd 失败"}

                # 6. 启动 dnsmasq
                if not await self._start_dnsmasq():
                    await self._stop_hostapd()
                    return {"success": False, "message": "启动 dnsmasq 失败"}

                # 7. 配置 iptables NAT
                await self._configure_nat()

                self._is_running = True

                # 持久化 AP 密码到 setup_status.json（供 Captive Portal 读取）
                try:
                    from pathlib import Path
                    import json
                    status_file = Path("/var/lib/lelamp/setup_status.json")
                    status_file.parent.mkdir(parents=True, exist_ok=True)
                    data = {}
                    if status_file.exists():
                        data = json.loads(status_file.read_text())
                    data["ap_password"] = self._config.password
                    status_file.write_text(json.dumps(data, indent=2, ensure_ascii=False))
                except Exception as e:
                    logger.warning(f"Failed to persist AP password: {e}")

                # LED 提示：蓝色呼吸灯表示配置模式
                try:
                    from lelamp.service.rgb.rgb_service import RGBService
                    rgb = RGBService()
                    rgb.start()
                    rgb.dispatch_event("breath", {"color": (0, 100, 255), "period": 2.0})
                    logger.info("LED breathing blue to indicate setup mode")
                except Exception:
                    pass  # NoOp on macOS

                logger.info(f"AP mode started successfully: {self._config.ssid} at {self._config.ip_address}")

                return {
                    "success": True,
                    "message": "AP 模式已启动",
                    "ssid": self._config.ssid,
                    "password": self._config.password,
                    "ip_address": self._config.ip_address
                }

            except Exception as e:
                logger.error(f"Failed to start AP mode: {e}", exc_info=True)
                # 清理
                await self.stop_ap_mode()
                return {"success": False, "message": f"启动 AP 模式失败: {str(e)}"}

    async def stop_ap_mode(self) -> bool:
        """
        停止 AP 模式

        Returns:
            是否成功停止
        """
        async with self._lock:
            if not self._is_running:
                return True

            try:
                logger.info("Stopping AP mode")

                await self._stop_dnsmasq()
                await self._stop_hostapd()
                await self._clear_nat()

                self._is_running = False

                logger.info("AP mode stopped")
                return True

            except Exception as e:
                logger.error(f"Failed to stop AP mode: {e}", exc_info=True)
                return False

    @property
    def current_password(self) -> str | None:
        """获取当前 AP 密码（可能为 None）"""
        return self._config.password

    @property
    def current_ssid(self) -> str:
        """获取当前 AP SSID"""
        return self._config.ssid

    # ==================== Captive Portal Support ====================

    async def check_captive_portal_enabled(self) -> bool:
        """
        检查是否启用了 Captive Portal 模式

        Returns:
            是否启用 Captive Portal
        """
        return self._config.captive_portal_enabled

    def get_portal_config(self) -> dict:
        """
        获取 Captive Portal 配置信息

        Returns:
            Portal 配置字典
        """
        return {
            "enabled": self._config.captive_portal_enabled,
            "ssid": self._config.ssid,
            "password": self._config.password,
            "ip_address": self._config.ip_address,
            "port": self._config.portal_port,
            "dhcp_start": self._config.dhcp_start,
            "dhcp_end": self._config.dhcp_end,
        }

    async def is_captive_portal_running(self) -> bool:
        """
        检查 Captive Portal 服务是否正在运行

        Returns:
            Portal 是否运行
        """
        if not self._config.captive_portal_enabled:
            return False

        # 检查端口是否在监听
        try:
            process = await asyncio.create_subprocess_exec(
                "sudo", "netstat", "-tlnp",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await process.communicate()
            port_str = f":{self._config.portal_port}"
            return port_str in stdout.decode()
        except Exception as e:
            logger.error(f"检查 Portal 状态失败: {e}")
            return False

    async def is_in_ap_mode(self) -> bool:
        """
        检查当前是否在 AP 模式

        Returns:
            是否在 AP 模式
        """
        # 检查 hostapd 进程是否运行
        try:
            if self._hostapd_pid.exists():
                pid = int(self._hostapd_pid.read_text().strip())
                # 检查进程是否存在
                os.kill(pid, 0)
                self._is_running = True
                return True
        except (FileNotFoundError, ValueError, OSError):
            pass

        self._is_running = False
        return False

    def is_ap_mode(self) -> bool:
        """
        同步包装：检查当前是否在 AP 模式（供单元测试使用）
        """
        try:
            return asyncio.run(self.is_in_ap_mode())
        except Exception:
            return False

    async def get_connected_clients(self) -> List[ClientInfo]:
        """
        获取已连接的客户端列表

        Returns:
            客户端信息列表
        """
        clients = []

        try:
            # 从 dnsmasq 的 leases 文件读取
            leases_path = Path("/var/lib/misc/dnsmasq.leases")
            if not leases_path.exists():
                return clients

            async with asyncio.to_thread(leases_path.read_text) as content:
                for line in content.strip().split('\n'):
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) >= 4:
                        # 格式: timestamp mac ip hostname
                        clients.append(ClientInfo(
                            mac=parts[1],
                            ip=parts[2],
                            connected_at=float(parts[0]) if parts[0].isdigit() else 0
                        ))

        except Exception as e:
            logger.error(f"Failed to get connected clients: {e}")

        return clients

    # ========================================================================
    # 私有方法
    # ========================================================================

    async def _stop_wifi_client(self) -> None:
        """停止现有的 WiFi 客户端连接"""
        try:
            # 使用 nmcli 断开 WiFi
            process = await asyncio.create_subprocess_exec(
                "sudo", "nmcli", "connection", "down", self._config.interface,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
        except Exception as e:
            logger.warning(f"Failed to stop WiFi client: {e}")

    async def _configure_static_ip(self) -> bool:
        """配置 wlan0 的静态 IP 地址"""
        try:
            # 使用 ifconfig 或 ip 命令配置静态 IP
            process = await asyncio.create_subprocess_exec(
                "sudo", "ip", "addr", "add",
                f"{self._config.ip_address}/{self._config.netmask}",
                "dev", self._config.interface,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                # 可能需要先删除旧地址
                await asyncio.create_subprocess_exec(
                    "sudo", "ip", "addr", "flush", "dev", self._config.interface,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                # 重新添加
                process = await asyncio.create_subprocess_exec(
                    "sudo", "ip", "addr", "add",
                    f"{self._config.ip_address}/{self._config.netmask}",
                    "dev", self._config.interface,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await process.communicate()

            # 启用接口
            process = await asyncio.create_subprocess_exec(
                "sudo", "ip", "link", "set", "dev", self._config.interface, "up",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()

            return True

        except Exception as e:
            logger.error(f"Failed to configure static IP: {e}")
            return False

    async def _write_hostapd_config(self) -> bool:
        """生成并写入 hostapd 配置文件"""
        try:
            config_content = f"""# LeLamp AP Mode Configuration
interface={self._config.interface}
driver=nl80211
ssid={self._config.ssid}
hw_mode={self._config.hw_mode}
channel={self._config.channel}
auth_algs=1
wpa=2
wpa_passphrase={self._config.password}
wpa_key_mgmt=WPA-PSK
wpa_pairwise=CCMP
rsn_pairwise=CCMP
"""

            # 需要管理员权限写入
            process = await asyncio.create_subprocess_exec(
                "sudo", "tee", self._hostapd_conf_path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate(input=config_content.encode())

            return process.returncode == 0

        except Exception as e:
            logger.error(f"Failed to write hostapd config: {e}")
            return False

    async def _write_dnsmasq_config(self) -> bool:
        """生成并写入 dnsmasq 配置文件"""
        try:
            # 备份原配置
            if os.path.exists(self._dnsmasq_conf_path):
                await asyncio.create_subprocess_exec(
                    "sudo", "cp", self._dnsmasq_conf_path, f"{self._dnsmasq_conf_path}.bak",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

            config_content = f"""# LeLamp AP Mode DHCP/DNS Configuration
interface={self._config.interface}
dhcp-range={self._config.ip_address[:-1]}20,{self._config.ip_address[:-1]}100,12h
dhcp-option=3,{self._config.ip_address}
dhcp-option=6,{self._config.ip_address}
address=/#/{self._config.ip_address}
no-resolv
server=8.8.8.8
server=8.8.4.4
"""

            process = await asyncio.create_subprocess_exec(
                "sudo", "tee", self._dnsmasq_conf_path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate(input=config_content.encode())

            return process.returncode == 0

        except Exception as e:
            logger.error(f"Failed to write dnsmasq config: {e}")
            return False

    async def _start_hostapd(self) -> bool:
        """启动 hostapd 服务"""
        try:
            # 使用自定义 PID 文件启动
            process = await asyncio.create_subprocess_exec(
                "sudo", "hostapd",
                "-B",  # 后台运行
                "-P", str(self._hostapd_pid),
                self._hostapd_conf_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                logger.error(f"hostapd failed to start: {stderr.decode()}")
                return False

            # 等待一小段时间确保服务启动
            await asyncio.sleep(2)

            return True

        except Exception as e:
            logger.error(f"Failed to start hostapd: {e}")
            return False

    async def _stop_hostapd(self) -> None:
        """停止 hostapd 服务"""
        try:
            if self._hostapd_pid.exists():
                pid = int(self._hostapd_pid.read_text().strip())
                os.kill(pid, 15)  # SIGTERM

                # 等待进程结束
                for _ in range(10):
                    try:
                        os.kill(pid, 0)
                        await asyncio.sleep(0.5)
                    except OSError:
                        break
                else:
                    # 强制杀死
                    os.kill(pid, 9)

                self._hostapd_pid.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to stop hostapd: {e}")

    async def _start_dnsmasq(self) -> bool:
        """启动 dnsmasq 服务"""
        try:
            process = await asyncio.create_subprocess_exec(
                "sudo", "dnsmasq",
                "-C", self._dnsmasq_conf_path,
                "-x", str(self._dnsmasq_pid),  # PID 文件
                "-k",  # 保持运行
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                logger.error(f"dnsmasq failed to start: {stderr.decode()}")
                return False

            await asyncio.sleep(1)
            return True

        except Exception as e:
            logger.error(f"Failed to start dnsmasq: {e}")
            return False

    async def _stop_dnsmasq(self) -> None:
        """停止 dnsmasq 服务"""
        try:
            if self._dnsmasq_pid.exists():
                pid = int(self._dnsmasq_pid.read_text().strip())
                os.kill(pid, 15)

                for _ in range(10):
                    try:
                        os.kill(pid, 0)
                        await asyncio.sleep(0.5)
                    except OSError:
                        break
                else:
                    os.kill(pid, 9)

                self._dnsmasq_pid.unlink(missing_ok=True)

            # 同时尝试通过 service 停止
            await asyncio.create_subprocess_exec(
                "sudo", "systemctl", "stop", "dnsmasq",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
        except Exception as e:
            logger.warning(f"Failed to stop dnsmasq: {e}")

    async def _configure_nat(self) -> None:
        """配置 iptables NAT 规则"""
        try:
            # 启用 IP 转发
            await asyncio.create_subprocess_exec(
                "sudo", "sysctl", "-w", "net.ipv4.ip_forward=1",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            # 配置 NAT 规则
            # 获取默认网关接口
            process = await asyncio.create_subprocess_exec(
                "ip", "route", "show", "default",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await process.communicate()
            default_route = stdout.decode().strip()
            gateway_interface = None
            if default_route:
                parts = default_route.split()
                if len(parts) >= 5:
                    gateway_interface = parts[4]

            if gateway_interface:
                # MASQUERADE 规则
                await asyncio.create_subprocess_exec(
                    "sudo", "iptables", "-t", "nat", "-A", "POSTROUTING",
                    "-o", gateway_interface, "-j", "MASQUERADE",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                # 转发规则
                await asyncio.create_subprocess_exec(
                    "sudo", "iptables", "-A", "FORWARD",
                    "-i", self._config.interface, "-o", gateway_interface,
                    "-j", "ACCEPT",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await asyncio.create_subprocess_exec(
                    "sudo", "iptables", "-A", "FORWARD",
                    "-i", gateway_interface, "-o", self._config.interface,
                    "-m", "state", "--state", "RELATED,ESTABLISHED",
                    "-j", "ACCEPT",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

        except Exception as e:
            logger.warning(f"Failed to configure NAT: {e}")

    async def _clear_nat(self) -> None:
        """清除 iptables NAT 规则"""
        try:
            # 清除 NAT 规则
            await asyncio.create_subprocess_exec(
                "sudo", "iptables", "-t", "nat", "-F", "POSTROUTING",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await asyncio.create_subprocess_exec(
                "sudo", "iptables", "-F", "FORWARD",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
        except Exception as e:
            logger.warning(f"Failed to clear NAT: {e}")


# 全局单例实例
ap_manager = APManager()
