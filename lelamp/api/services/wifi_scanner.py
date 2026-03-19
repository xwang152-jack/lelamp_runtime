"""
WiFi 网络扫描器
使用 NetworkManager 和 iw 工具扫描附近的 WiFi 网络
"""
import asyncio
import logging
import re
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class WiFiNetwork:
    """WiFi 网络信息"""
    ssid: str
    signal_strength: int  # 0-100
    encryption: str  # "WPA2", "WPA", "WEP", "Open"
    channel: int
    mac_address: str
    frequency: float  # GHz


class WiFiScanner:
    """WiFi 网络扫描器"""

    def __init__(self, interface: str = "wlan0"):
        self.interface = interface

    def scan_networks(self) -> List[Dict[str, any]]:
        """扫描可用的 WiFi 网络"""
        try:
            # 使用 nmcli 扫描网络
            result = subprocess.run(
                ['nmcli', '-t', '-f', 'SSID,SIGNAL,SECURITY,CHAN,FREQ', 'device', 'wifi', 'list'],
                capture_output=True,
                text=True,
                timeout=15
            )

            if result.returncode != 0:
                logger.error(f"WiFi 扫描失败: {result.stderr}")
                return []

            networks = []
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue

                parts = line.split(':')
                if len(parts) >= 5:
                    ssid = parts[0] or "Hidden Network"
                    signal = int(parts[1]) if parts[1].isdigit() else 0
                    security = parts[2] or "Open"
                    channel = int(parts[3]) if parts[3].isdigit() else 0
                    frequency = float(parts[4]) / 1000000 if parts[4].isdigit() else 0  # MHz to GHz

                    networks.append({
                        "ssid": ssid,
                        "signal_strength": signal,
                        "encryption": self._parse_encryption(security),
                        "channel": channel,
                        "frequency": frequency
                    })

            # 按信号强度排序
            networks.sort(key=lambda x: x["signal_strength"], reverse=True)
            return networks

        except subprocess.TimeoutExpired:
            logger.error("WiFi 扫描超时")
            return []
        except Exception as e:
            logger.error(f"WiFi 扫描异常: {e}")
            return []

    def get_network_info(self, ssid: str) -> Optional[Dict[str, any]]:
        """获取特定网络的详细信息"""
        networks = self.scan_networks()
        for network in networks:
            if network["ssid"] == ssid:
                return network
        return None

    def _parse_encryption(self, security: str) -> str:
        """解析加密类型"""
        if 'WPA3' in security:
            return 'WPA3'
        elif 'WPA2' in security:
            return 'WPA2'
        elif 'WPA' in security:
            return 'WPA'
        elif 'WEP' in security:
            return 'WEP'
        else:
            return 'Open'

    async def async_scan_networks(self) -> List[Dict[str, any]]:
        """异步扫描 WiFi 网络"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.scan_networks)

    def get_signal_strength_label(self, signal: int) -> str:
        """获取信号强度标签"""
        if signal >= 80:
            return "极强"
        elif signal >= 60:
            return "强"
        elif signal >= 40:
            return "中等"
        elif signal >= 20:
            return "弱"
        else:
            return "极弱"

    def format_network_for_display(self, network: Dict[str, any]) -> str:
        """格式化网络信息用于显示"""
        ssid = network.get("ssid", "Unknown")
        signal = network.get("signal_strength", 0)
        encryption = network.get("encryption", "Open")
        signal_label = self.get_signal_strength_label(signal)

        signal_icons = {
            "极强": "📶📶📶",
            "强": "📶📶",
            "中等": "📶",
            "弱": "📡",
            "极弱": "📡"
        }

        icon = signal_icons.get(signal_label, "📡")
        lock_icon = "🔒" if encryption != "Open" else "🔓"

        return f"{icon} {ssid} {lock_icon} 信号:{signal_label} {encryption}"
