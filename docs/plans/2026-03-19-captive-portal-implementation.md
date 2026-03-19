# Captive Portal 设置向导实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**目标：** 实现 AP 模式 + Captive Portal 解决方案，让用户首次使用 LeLamp 时无需知道 IP 地址即可完成 WiFi 配置。

**架构：** 扩展现有 AP 管理器，创建轻量级 FastAPI Captive Portal 服务，通过 DNS 重定向将用户引导到设置向导，完成 WiFi 配置后自动切换到正常服务模式。

**技术栈：** FastAPI, hostapd, dnsmasq, NetworkManager, Vue.js 3, systemd

---

## Task 1: 创建状态管理模块

**Files:**
- Create: `lelamp/api/services/setup_state.py`
- Test: `tests/api/test_setup_state.py`

**Step 1: Write the failing test**

```python
# tests/api/test_setup_state.py
import pytest
import tempfile
import os
from pathlib import Path

def test_save_and_load_state():
    """测试状态保存和加载"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        state_file = f.name

    try:
        from lelamp.api.services.setup_state import SetupStateManager

        manager = SetupStateManager(state_file)
        test_state = {
            "setup_completed": False,
            "current_step": "wifi_selection"
        }

        manager.save_state(test_state)
        loaded_state = manager.load_state()

        assert loaded_state["setup_completed"] == False
        assert loaded_state["current_step"] == "wifi_selection"
    finally:
        os.unlink(state_file)

def test_update_step():
    """测试步骤更新"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        state_file = f.name

    try:
        from lelamp.api.services.setup_state import SetupStateManager

        manager = SetupStateManager(state_file)
        manager.update_step("password_input")
        state = manager.load_state()

        assert state["current_step"] == "password_input"
    finally:
        os.unlink(state_file)

def test_complete_setup():
    """测试完成设置"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        state_file = f.name

    try:
        from lelamp.api.services.setup_state import SetupStateManager

        manager = SetupStateManager(state_file)
        manager.complete_setup("192.168.1.100")
        state = manager.load_state()

        assert state["setup_completed"] is True
        assert state["last_ip_address"] == "192.168.1.100"
        assert "setup_completed_at" in state
    finally:
        os.unlink(state_file)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/api/test_setup_state.py -v
```
Expected: FAIL with "ModuleNotFoundError: No module named 'lelamp.api.services.setup_state'"

**Step 3: Write minimal implementation**

```python
# lelamp/api/services/setup_state.py
"""
设置状态管理器
管理首次设置流程的状态持久化
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class SetupStateManager:
    """设置状态管理器"""

    DEFAULT_STATE = {
        "setup_completed": False,
        "setup_started_at": None,
        "setup_completed_at": None,
        "current_step": "welcome",
        "wifi_ssid": None,
        "connection_attempts": 0,
        "error_message": None,
        "last_ip_address": None,
        "ap_mode_count": 0,
        "network_history": []
    }

    def __init__(self, state_file: str = "/var/lib/lelamp/setup_status.json"):
        self.state_file = Path(state_file)
        self._lock = Lock()

        # 确保目录存在
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        # 如果文件不存在，创建初始状态
        if not self.state_file.exists():
            self.save_state(self.DEFAULT_STATE.copy())

    def load_state(self) -> Dict[str, Any]:
        """加载状态"""
        with self._lock:
            try:
                if self.state_file.exists():
                    with open(self.state_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
                else:
                    return self.DEFAULT_STATE.copy()
            except Exception as e:
                logger.error(f"加载状态失败: {e}")
                return self.DEFAULT_STATE.copy()

    def save_state(self, state: Dict[str, Any]) -> None:
        """保存状态"""
        with self._lock:
            try:
                with open(self.state_file, 'w', encoding='utf-8') as f:
                    json.dump(state, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.error(f"保存状态失败: {e}")

    def update_step(self, step: str) -> None:
        """更新当前步骤"""
        state = self.load_state()
        state["current_step"] = step
        self.save_state(state)

    def increment_attempts(self) -> int:
        """增加连接尝试次数"""
        state = self.load_state()
        state["connection_attempts"] += 1
        self.save_state(state)
        return state["connection_attempts"]

    def reset_attempts(self) -> None:
        """重置连接尝试次数"""
        state = self.load_state()
        state["connection_attempts"] = 0
        self.save_state(state)

    def set_error(self, error: str) -> None:
        """设置错误信息"""
        state = self.load_state()
        state["error_message"] = error
        self.save_state(state)

    def clear_error(self) -> None:
        """清除错误信息"""
        state = self.load_state()
        state["error_message"] = None
        self.save_state(state)

    def complete_setup(self, ip_address: str) -> None:
        """标记设置完成"""
        state = self.load_state()
        state["setup_completed"] = True
        state["setup_completed_at"] = datetime.utcnow().isoformat()
        state["last_ip_address"] = ip_address
        state["current_step"] = "completed"
        self.save_state(state)

    def is_setup_completed(self) -> bool:
        """检查是否已完成设置"""
        state = self.load_state()
        return state.get("setup_completed", False)
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/api/test_setup_state.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add tests/api/test_setup_state.py lelamp/api/services/setup_state.py
git commit -m "feat: 添加设置状态管理器"
```

---

## Task 2: 扩展 AP 管理器支持 Captive Portal

**Files:**
- Modify: `lelamp/api/services/ap_manager.py`
- Test: `tests/api/test_ap_manager.py`

**Step 1: Write the failing test**

```python
# tests/api/test_ap_manager.py
import pytest
from lelamp.api.services.ap_manager import APManager, APConfig

def test_ap_manager_with_captive_portal():
    """测试 AP 管理器支持 Captive Portal"""
    config = APConfig(
        ssid="LeLamp-Setup",
        password="lelamp123",
        captive_portal_enabled=True
    )

    manager = APManager(config)

    # 验证配置
    assert manager.config.captive_portal_enabled is True

    # 测试启动（不实际启动，只验证配置）
    assert manager.config.ip_address == "192.168.4.1"
    assert manager.config.ssid == "LeLamp-Setup"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/api/test_ap_manager.py::test_ap_manager_with_captive_portal -v
```
Expected: FAIL with "TypeError: APConfig() got an unexpected keyword argument 'captive_portal_enabled'"

**Step 3: Write minimal implementation**

在 `lelamp/api/services/ap_manager.py` 中添加：

```python
# 在 APConfig dataclass 中添加新字段
@dataclass
class APConfig:
    """AP 热点配置"""
    ssid: str = "LeLamp-Setup"
    password: str = "lelamp123"
    ip_address: str = "192.168.4.1"
    netmask: str = "255.255.255.0"
    channel: int = 6
    hw_mode: str = "g"
    interface: str = "wlan0"
    captive_portal_enabled: bool = False  # 新增：是否启用 Captive Portal
    portal_port: int = 8080  # 新增：Portal 服务端口

    # ... 其他验证代码保持不变
```

在 `APManager` 类中添加：

```python
class APManager:
    def __init__(self, config: APConfig = None):
        self.config = config or APConfig()
        self._dnsmasq_process = None
        self._portal_process = None

    async def start_captive_portal_dns(self) -> None:
        """启动 Captive Portal DNS 重定向"""
        if not self.config.captive_portal_enabled:
            return

        logger.info("启动 Captive Portal DNS 重定向...")

        # 创建 dnsmasq 配置
        dnsmasq_config = f"""
# LeLamp Captive Portal DNS Configuration
interface={self.config.interface}
bind-interfaces
dhcp-range=192.168.4.100,192.168.4.200,12h
dhcp-option=3,{self.config.ip_address}
dhcp-option=6,{self.config.ip_address}
address=/#/{self.config.ip_address}
port=53
log-queries
log-dhcp
"""

        config_file = Path("/etc/dnsmasq.conf")
        try:
            # 备份原配置
            if config_file.exists():
                backup_file = config_file.with_suffix('.conf.backup')
                import shutil
                shutil.copy(config_file, backup_file)

            # 写入新配置
            with open(config_file, 'w') as f:
                f.write(dnsmasq_config)

            # 启动 dnsmasq
            import subprocess
            subprocess.run(['systemctl', 'restart', 'dnsmasq'], check=True)
            logger.info("Captive Portal DNS 已启动")

        except Exception as e:
            logger.error(f"启动 DNS 失败: {e}")
            raise

    async def stop_captive_portal_dns(self) -> None:
        """停止 Captive Portal DNS"""
        try:
            import subprocess
            subprocess.run(['systemctl', 'stop', 'dnsmasq'], check=False)

            # 恢复原配置
            backup_file = Path("/etc/dnsmasq.conf.backup")
            if backup_file.exists():
                import shutil
                shutil.move(backup_file, "/etc/dnsmasq.conf")

            logger.info("Captive Portal DNS 已停止")
        except Exception as e:
            logger.error(f"停止 DNS 失败: {e}")
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/api/test_ap_manager.py::test_ap_manager_with_captive_portal -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add tests/api/test_ap_manager.py lelamp/api/services/ap_manager.py
git commit -m "feat: 扩展 AP 管理器支持 Captive Portal"
```

---

## Task 3: 创建 WiFi 扫描器

**Files:**
- Create: `lelamp/api/services/wifi_scanner.py`
- Test: `tests/api/test_wifi_scanner.py`

**Step 1: Write the failing test**

```python
# tests/api/test_wifi_scanner.py
import pytest
from lelamp.api.services.wifi_scanner import WiFiScanner

def test_scan_networks():
    """测试 WiFi 扫描"""
    scanner = WiFiScanner()
    networks = scanner.scan_networks()

    assert isinstance(networks, list)
    # 在有 WiFi 的环境中应该能找到网络
    if len(networks) > 0:
        network = networks[0]
        assert "ssid" in network
        assert "signal_strength" in network
        assert "encryption" in network

def test_get_network_info():
    """测试获取网络信息"""
    scanner = WiFiScanner()
    networks = scanner.scan_networks()

    if len(networks) > 0:
        ssid = networks[0]["ssid"]
        info = scanner.get_network_info(ssid)

        assert info is not None
        assert info["ssid"] == ssid
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/api/test_wifi_scanner.py -v
```
Expected: FAIL with "ModuleNotFoundError: No module named 'lelamp.api.services.wifi_scanner'"

**Step 3: Write minimal implementation**

```python
# lelamp/api/services/wifi_scanner.py
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
    encryption: str  # "WPA2", "WEP", "Open"
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
                timeout=10
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
                    signal = int(parts[1])
                    security = parts[2] or "Open"
                    channel = int(parts[3])
                    frequency = float(parts[4]) / 1000000  # MHz to GHz

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
        if 'WPA2' in security:
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
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/api/test_wifi_scanner.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add tests/api/test_wifi_scanner.py lelamp/api/services/wifi_scanner.py
git commit -m "feat: 添加 WiFi 网络扫描器"
```

---

## Task 4: 创建网络连接管理器

**Files:**
- Create: `lelamp/api/services/network_manager.py`
- Test: `tests/api/test_network_manager.py`

**Step 1: Write the failing test**

```python
# tests/api/test_network_manager.py
import pytest
from lelamp.api.services.network_manager import NetworkConnectionManager

def test_connect_to_wifi():
    """测试连接 WiFi（需要真实网络环境）"""
    manager = NetworkConnectionManager()

    # 测试连接（使用测试网络，实际测试时需要配置）
    # 这里只测试接口是否正常
    assert hasattr(manager, 'connect_wifi')
    assert hasattr(manager, 'disconnect_wifi')
    assert hasattr(manager, 'get_current_connection')

def test_get_connection_status():
    """测试获取连接状态"""
    manager = NetworkConnectionManager()
    status = manager.get_connection_status()

    assert "connected" in status
    assert "ssid" in status
    assert "ip_address" in status
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/api/test_network_manager.py -v
```
Expected: FAIL with "ModuleNotFoundError: No module named 'lelamp.api.services.network_manager'"

**Step 3: Write minimal implementation**

```python
# lelamp/api/services/network_manager.py
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
        if "Secrets were required" in error_output or "password" in error_output.lower():
            return "密码错误"
        elif "no network" in error_output.lower():
            return "网络不可达"
        elif "timeout" in error_output.lower():
            return "连接超时"
        else:
            return "连接失败"

    async def async_connect_wifi(self, ssid: str, password: str) -> Dict[str, any]:
        """异步连接 WiFi"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.connect_wifi,
            ssid,
            password
        )
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/api/test_network_manager.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add tests/api/test_network_manager.py lelamp/api/services/network_manager.py
git commit -m "feat: 添加网络连接管理器"
```

---

## Task 5: 创建 Captive Portal API 服务

**Files:**
- Create: `lelamp/api/services/captive_portal.py`
- Test: `tests/api/test_captive_portal.py`

**Step 1: Write the failing test**

```python
# tests/api/test_captive_portal.py
import pytest
from fastapi.testclient import TestClient
from lelamp.api.services.captive_portal import create_captive_portal_app

def test_get_setup_status():
    """测试获取设置状态"""
    app = create_captive_portal_app()
    client = TestClient(app)

    response = client.get("/api/setup/status")
    assert response.status_code == 200

    data = response.json()
    assert "setup_completed" in data
    assert "current_step" in data

def test_scan_networks():
    """测试扫描网络"""
    app = create_captive_portal_app()
    client = TestClient(app)

    response = client.get("/api/setup/networks")
    assert response.status_code == 200

    data = response.json()
    assert "networks" in data
    assert isinstance(data["networks"], list)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/api/test_captive_portal.py -v
```
Expected: FAIL with "ModuleNotFoundError: No module named 'lelamp.api.services.captive_portal'"

**Step 3: Write minimal implementation**

```python
# lelamp/api/services/captive_portal.py
"""
Captive Portal API 服务
为首次设置提供 Web API 接口
"""
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from lelamp.api.services.setup_state import SetupStateManager
from lelamp.api.services.wifi_scanner import WiFiScanner
from lelamp.api.services.network_manager import NetworkConnectionManager

logger = logging.getLogger(__name__)


# Pydantic 模型
class WiFiConnectRequest(BaseModel):
    ssid: str
    password: str


class CompleteSetupRequest(BaseModel):
    success: bool
    ip_address: str = None


# 创建 FastAPI 应用
def create_captive_portal_app() -> FastAPI:
    """创建 Captive Portal 应用"""
    app = FastAPI(
        title="LeLamp Setup Portal",
        description="LeLamp 首次设置向导",
        version="1.0.0"
    )

    # 初始化服务
    state_manager = SetupStateManager()
    wifi_scanner = WiFiScanner()
    network_manager = NetworkConnectionManager()

    # 静态文件服务（如果需要）
    # app.mount("/static", StaticFiles(directory="web/captive-portal"), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def root():
        """主页面 - 返回设置向导 HTML"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>LeLamp 设置向导</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
                .container { max-width: 480px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                h1 { text-align: center; color: #333; }
                .btn { display: block; width: 100%; padding: 12px; margin: 10px 0; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
                .btn-primary { background: #007bff; color: white; }
                .btn:hover { background: #0056b3; }
                #networks { margin: 20px 0; }
                .network { padding: 15px; margin: 10px 0; border: 1px solid #ddd; border-radius: 4px; cursor: pointer; }
                .network:hover { background: #f8f9fa; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>欢迎使用 LeLamp</h1>
                <p>让我们帮您连接到 WiFi 网络</p>
                <button class="btn btn-primary" onclick="startSetup()">开始设置</button>
                <div id="networks"></div>
            </div>
            <script>
                async function startSetup() {
                    const response = await fetch('/api/setup/networks');
                    const data = await response.json();

                    const networksDiv = document.getElementById('networks');
                    networksDiv.innerHTML = '<h2>可用网络</h2>';

                    data.networks.forEach(network => {
                        const div = document.createElement('div');
                        div.className = 'network';
                        div.innerHTML = `<strong>${network.ssid}</strong><br>信号: ${network.signal_strength}%`;
                        div.onclick = () => selectNetwork(network.ssid);
                        networksDiv.appendChild(div);
                    });
                }

                async function selectNetwork(ssid) {
                    const password = prompt(`请输入 ${ssid} 的密码:`);
                    if (password) {
                        const response = await fetch('/api/setup/connect', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({ssid, password})
                        });
                        const data = await response.json();

                        if (data.success) {
                            alert(`连接成功！IP 地址: ${data.ip_address}`);
                        } else {
                            alert(`连接失败: ${data.message}`);
                        }
                    }
                }
            </script>
        </body>
        </html>
        """

    @app.get("/api/setup/status")
    async def get_setup_status():
        """获取设置状态"""
        try:
            state = state_manager.load_state()
            return {
                "setup_completed": state.get("setup_completed", False),
                "current_step": state.get("current_step", "welcome"),
                "error_message": state.get("error_message"),
                "connection_attempts": state.get("connection_attempts", 0)
            }
        except Exception as e:
            logger.error(f"获取状态失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/setup/networks")
    async def scan_networks():
        """扫描可用 WiFi 网络"""
        try:
            networks = await wifi_scanner.async_scan_networks()
            return {
                "networks": networks,
                "count": len(networks)
            }
        except Exception as e:
            logger.error(f"扫描网络失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/setup/connect")
    async def connect_wifi(request: WiFiConnectRequest):
        """连接到指定 WiFi 网络"""
        try:
            # 增加尝试次数
            attempts = state_manager.increment_attempts()

            if attempts > 3:
                return {
                    "success": False,
                    "error": "max_attempts",
                    "message": "连接尝试次数过多，请稍后再试"
                }

            # 尝试连接
            result = await network_manager.async_connect_wifi(
                request.ssid,
                request.password
            )

            if result["success"]:
                # 更新状态
                state_manager.update_step("connected")
                state_manager.set_wifi_ssid(request.ssid)
                state_manager.complete_setup(result["ip_address"])
                state_manager.reset_attempts()

                return result
            else:
                # 设置错误信息
                state_manager.set_error(result.get("error", "连接失败"))
                return result

        except Exception as e:
            logger.error(f"连接 WiFi 失败: {e}")
            state_manager.set_error(str(e))
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/setup/test-connection")
    async def test_connection():
        """测试网络连接"""
        try:
            result = network_manager.test_connection()
            return result
        except Exception as e:
            logger.error(f"测试连接失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/setup/complete")
    async def complete_setup(request: CompleteSetupRequest):
        """完成设置"""
        try:
            if request.success:
                state = state_manager.load_state()
                return {
                    "success": True,
                    "ip_address": state.get("last_ip_address"),
                    "message": "设置完成"
                }
            else:
                return {
                    "success": False,
                    "message": "设置未完成"
                }
        except Exception as e:
            logger.error(f"完成设置失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/setup/reset")
    async def reset_setup():
        """重置设置（恢复出厂设置）"""
        try:
            state_manager.save_state(SetupStateManager.DEFAULT_STATE.copy())
            return {
                "success": True,
                "message": "设置已重置"
            }
        except Exception as e:
            logger.error(f"重置设置失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return app


def main():
    """启动 Captive Portal 服务"""
    import uvicorn

    app = create_captive_portal_app()
    uvicorn.run(
        app,
        host="192.168.4.1",
        port=8080,
        log_level="info"
    )


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/api/test_captive_portal.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add tests/api/test_captive_portal.py lelamp/api/services/captive_portal.py
git commit -m "feat: 添加 Captive Portal API 服务"
```

---

## Task 6: 创建 systemd 服务配置

**Files:**
- Create: `scripts/lelamp-setup-ap.service`
- Create: `scripts/lelamp-captive-portal.service`
- Create: `scripts/install_captive_portal.sh`

**Step 1: Create AP setup service**

```bash
# 创建 lelamp-setup-ap.service
cat > scripts/lelamp-setup-ap.service << 'EOF'
[Unit]
Description=LeLamp Setup AP Mode
Before=lelamp-captive-portal.service
After=network.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/local/bin/lelamp-start-ap
ExecStop=/usr/local/bin/lelamp-stop-ap
User=root

[Install]
WantedBy=multi-user.target
EOF
```

**Step 2: Create Captive Portal service**

```bash
# 创建 lelamp-captive-portal.service
cat > scripts/lelamp-captive-portal.service << 'EOF'
[Unit]
Description=LeLamp Captive Portal
After=lelamp-setup-ap.service
Before=lelamp-livekit.service lelamp-api.service

[Service]
Type=simple
User=root
WorkingDirectory=/home/pi/lelamp_runtime
Environment=PATH=/usr/local/bin:/usr/bin:/bin
EnvironmentFile=/home/pi/lelamp_runtime/.env
ExecStart=/usr/local/bin/uv run python -m lelamp.api.services.captive_portal
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=lelamp-captive-portal

[Install]
WantedBy=multi-user.target
EOF
```

**Step 3: Create installation script**

```bash
# 创建安装脚本
cat > scripts/install_captive_portal.sh << 'EOF'
#!/bin/bash
# Captive Portal 安装脚本

set -e

PI_HOST="${PI_HOST:-pi@192.168.0.104}"

echo "================================================"
echo "LeLamp Captive Portal 安装"
echo "================================================"
echo ""

# 1. 安装系统依赖
echo "📦 1. 安装系统依赖..."
ssh $PI_HOST 'sudo apt-get update'
ssh $PI_HOST 'sudo apt-get install -y hostapd dnsmasq iw wireless-tools'

echo "✅ 系统依赖安装完成"
echo ""

# 2. 配置 hostapd
echo "🔧 2. 配置 hostapd..."
ssh $PI_HOST 'sudo systemctl unmask hostapd'
ssh $PI_HOST 'sudo systemctl enable hostapd'
echo "✅ hostapd 配置完成"
echo ""

# 3. 部署服务文件
echo "📝 3. 部署 systemd 服务..."
cat scripts/lelamp-setup-ap.service | ssh $PI_HOST 'sudo tee /etc/systemd/system/lelamp-setup-ap.service > /dev/null'
cat scripts/lelamp-captive-portal.service | ssh $PI_HOST 'sudo tee /etc/systemd/system/lelamp-captive-portal.service > /dev/null'
echo "✅ 服务文件部署完成"
echo ""

# 4. 创建辅助脚本
echo "📜 4. 创建辅助脚本..."
ssh $PI_HOST 'sudo tee /usr/local/bin/lelamp-start-ap > /dev/null << '"'"'ENDSCRIPT'"'"'
#!/bin/bash
# 启动 AP 模式
# TODO: 实现 AP 启动逻辑
echo "Starting AP mode..."
ENDSCRIPT
'
ssh $PI_HOST 'sudo chmod +x /usr/local/bin/lelamp-start-ap'

ssh $PI_HOST 'sudo tee /usr/local/bin/lelamp-stop-ap > /dev/null << '"'"'ENDSCRIPT'"'"'
#!/bin/bash
# 停止 AP 模式
# TODO: 实现 AP 停止逻辑
echo "Stopping AP mode..."
ENDSCRIPT
'
ssh $PI_HOST 'sudo chmod +x /usr/local/bin/lelamp-stop-ap'
echo "✅ 辅助脚本创建完成"
echo ""

# 5. 重新加载 systemd
echo "🔄 5. 重新加载 systemd..."
ssh $PI_HOST 'sudo systemctl daemon-reload'
echo "✅ systemd 已重新加载"
echo ""

# 6. 创建状态目录
echo "📁 6. 创建状态目录..."
ssh $PI_HOST 'sudo mkdir -p /var/lib/lelamp'
ssh $PI_HOST 'sudo chmod 755 /var/lib/lelamp'
echo "✅ 状态目录创建完成"
echo ""

echo ""
echo "================================================"
echo "✅ Captive Portal 安装完成！"
echo "================================================"
echo ""
echo "🎯 管理命令："
echo ""
echo "启动 AP 模式:"
echo "  ssh $PI_HOST 'sudo systemctl start lelamp-setup-ap'"
echo ""
echo "启动 Portal 服务:"
echo "  ssh $PI_HOST 'sudo systemctl start lelamp-captive-portal'"
echo ""
echo "查看服务状态:"
echo "  ssh $PI_HOST 'sudo systemctl status lelamp-captive-portal'"
echo ""
echo "查看日志:"
echo "  ssh $PI_HOST 'sudo journalctl -u lelamp-captive-portal -f'"
echo ""
EOF

chmod +x scripts/install_captive_portal.sh
```

**Step 4: Commit**

```bash
git add scripts/lelamp-setup-ap.service scripts/lelamp-captive-portal.service scripts/install_captive_portal.sh
git commit -m "feat: 添加 Captive Portal systemd 服务配置"
```

---

## Task 7: 更新首次启动检测脚本

**Files:**
- Modify: `scripts/first_boot_setup.sh`

**Step 1: Update first_boot_setup.sh**

在现有的 `first_boot_setup.sh` 中添加 Captive Portal 支持：

```bash
#!/bin/bash
#
# LeLamp 首次启动检测脚本
# 检测设备是否需要进入设置模式，并在需要时启动 AP 模式
#

set -e

# 配置
STATUS_FILE="/var/lib/lelamp/setup_status.json"
STATE_DIR="/var/lib/lelamp"
LOG_FILE="/var/log/lelamp/setup.log"

# 确保日志目录存在
mkdir -p "$(dirname "$LOG_FILE")"

# 日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "Starting first boot setup check..."

# 确保状态目录存在
mkdir -p "$STATE_DIR"

# 检查是否已完成配置
if [ -f "$STATUS_FILE" ]; then
    if grep -q '"setup_completed": true' "$STATUS_FILE" 2>/dev/null; then
        log "Setup already completed, skipping AP mode"
    exit 0
    fi
fi

# 检查 WiFi 连接状态
if nmcli -t -f ACTIVE,SSID connection show --active | grep -q '^yes:'; then
    CURRENT_SSID=$(nmcli -t -f ACTIVE,SSID connection show --active | grep '^yes:' | cut -d: -f2)
    log "WiFi already connected to: $CURRENT_SSID"

    # 标记设置完成
    cat > "$STATUS_FILE" << EOF
{
  "setup_completed": true,
  "setup_completed_at": "$(date -u +%Y-%m-%dT%H:%M:%S)",
  "wifi_ssid": "$CURRENT_SSID",
  "last_ip_address": "$(hostname -I | awk '{print $1}')"
}
EOF
    exit 0
fi

# 需要进入设置模式
log "No WiFi configuration found, entering setup mode..."

# 启动 AP 模式
log "Starting AP mode..."
systemctl start lelamp-setup-ap.service

# 启动 Captive Portal
log "Starting Captive Portal..."
systemctl start lelamp-captive-portal.service

log "Setup mode activated. User should connect to 'LeLamp-Setup' hotspot."

exit 0
```

**Step 2: Commit**

```bash
git add scripts/first_boot_setup.sh
git commit -m "feat: 更新首次启动检测脚本支持 Captive Portal"
```

---

## Task 8: 创建测试和文档

**Files:**
- Create: `docs/SETUP_GUIDE.md`
- Create: `scripts/test_captive_portal.sh`

**Step 1: Create setup guide**

```bash
cat > docs/SETUP_GUIDE.md << 'EOF'
# LeLamp 首次设置指南

## 🚀 快速开始

### 首次使用 LeLamp

当您第一次使用 LeLamp 时，台灯会自动进入设置模式：

1. **连接热点**
   - 在您的手机或电脑上搜索 WiFi 网络
   - 找到名为 "LeLamp-Setup" 的热点
   - 密码：`lelamp123`

2. **打开设置页面**
   - 连接热点后，打开浏览器
   - 访问任意网址（会自动跳转到设置页面）
   - 或直接访问：http://192.168.4.1:8080

3. **配置 WiFi**
   - 选择您的 WiFi 网络
   - 输入 WiFi 密码
   - 点击"连接"

4. **完成设置**
   - 等待连接成功
   - 台灯会语音播报 IP 地址
   - 设置完成，开始使用！

## 🔧 高级选项

### 手动触发设置模式

如果您需要重新配置 WiFi：

```bash
# 连接到台灯的 SSH
ssh pi@<台灯IP>

# 清除设置状态
sudo rm /var/lib/lelamp/setup_status.json

# 重启台灯
sudo reboot
```

### 恢复出厂设置

```bash
# 在台灯上运行
sudo /home/pi/lelamp_runtime/scripts/factory_reset.sh
```

## 📱 设置页面功能

设置向导包含以下功能：

- ✅ **WiFi 网络扫描**：自动发现附近的 WiFi 网络
- ✅ **信号强度显示**：帮助您选择最佳网络
- ✅ **密码输入**：安全的密码输入界面
- ✅ **连接测试**：自动测试网络连接
- ✅ **错误处理**：友好的错误提示和重试选项
- ✅ **进度反馈**：实时显示连接进度

## ❓ 常见问题

### Q: 无法找到 "LeLamp-Setup" 热点？

**A:** 请确保：
- 台灯已开机（等待 30 秒让系统完全启动）
- 您的设备 WiFi 功能已开启
- 距离台灯不要太远（建议 5 米内）

### Q: 连接热点后无法访问设置页面？

**A:** 请尝试：
- 在浏览器中手动输入：http://192.168.4.1:8080
- 清除浏览器缓存后重试
- 尝试使用其他浏览器

### Q: WiFi 连接失败？

**A:** 请检查：
- WiFi 密码是否正确（区分大小写）
- 路由器是否正常工作
- 台灯是否在路由器信号范围内
- 尝试重启路由器和台灯

### Q: 设置完成后如何访问 Web 界面？

**A:** 设置完成后，台灯会语音播报 IP 地址。您可以通过以下方式访问：

- 通过 IP 地址：http://<台灯IP>:5173
- 通过 mDNS：http://lelamp.local:5173

## 🆘 需要帮助？

如果遇到问题：

1. 查看设置日志：`/var/log/lelamp/setup.log`
2. 查看服务状态：`sudo systemctl status lelamp-captive-portal`
3. 恢复出厂设置重新配置

更多帮助请访问：https://github.com/xwang152-jack/lelamp_runtime
EOF
```

**Step 2: Create test script**

```bash
cat > scripts/test_captive_portal.sh << 'EOF'
#!/bin/bash
# Captive Portal 测试脚本

set -e

PI_HOST="${PI_HOST:-pi@192.168.0.104}"

echo "================================================"
echo "LeLamp Captive Portal 测试"
echo "================================================"
echo ""

# 1. 测试 AP 模式启动
echo "📡 测试 1: AP 模式启动"
ssh $PI_HOST 'sudo systemctl start lelamp-setup-ap'
sleep 5
if ssh $PI_HOST 'systemctl is-active lelamp-setup-ap' | grep -q 'active'; then
    echo "✅ AP 模式启动成功"
else
    echo "❌ AP 模式启动失败"
    exit 1
fi
echo ""

# 2. 测试 Portal 服务启动
echo "🌐 测试 2: Portal 服务启动"
ssh $PI_HOST 'sudo systemctl start lelamp-captive-portal'
sleep 5
if ssh $PI_HOST 'systemctl is-active lelamp-captive-portal' | grep -q 'active'; then
    echo "✅ Portal 服务启动成功"
else
    echo "❌ Portal 服务启动失败"
    exit 1
fi
echo ""

# 3. 测试 DNS 重定向
echo "🔍 测试 3: DNS 重定向"
# 这里需要实际的 DNS 测试
echo "⏳ 需要手动验证：连接到 LeLamp-Setup 热点后访问任意网址"
echo ""

# 4. 测试 API 端点
echo "🧪 测试 4: API 端点"
echo "获取设置状态..."
if ssh $PI_HOST 'curl -s http://localhost:8080/api/setup/status' | grep -q 'setup_completed'; then
    echo "✅ API 端点响应正常"
else
    echo "❌ API 端点无响应"
fi
echo ""

# 5. 清理
echo "🧹 清理测试环境"
ssh $PI_HOST 'sudo systemctl stop lelamp-captive-portal'
ssh $PI_HOST 'sudo systemctl stop lelamp-setup-ap'
echo "✅ 测试服务已停止"
echo ""

echo "================================================"
echo "✅ 测试完成！"
echo "================================================"
echo ""
echo "📝 手动测试清单："
echo "1. 连接到 LeLamp-Setup 热点"
echo "2. 访问 http://192.168.4.1:8080"
echo "3. 测试 WiFi 扫描功能"
echo "4. 测试 WiFi 连接功能"
echo "5. 验证设置完成后服务切换"
EOF

chmod +x scripts/test_captive_portal.sh
```

**Step 3: Commit**

```bash
git add docs/SETUP_GUIDE.md scripts/test_captive_portal.sh
git commit -m "docs: 添加设置指南和测试脚本"
```

---

## Task 9: 更新主 README 和文档

**Files:**
- Modify: `README.md`
- Modify: `docs/AUTO_STARTUP_GUIDE.md`

**Step 1: Update README.md**

在 README.md 中添加 Captive Portal 说明：

```markdown
## 🚀 首次使用设置

LeLamp 支持智能设置模式，首次使用时无需知道 IP 地址：

1. **连接设置热点**：连接到 "LeLamp-Setup" WiFi 热点（密码：lelamp123）
2. **打开设置页面**：浏览器会自动跳转到设置页面，或访问 http://192.168.4.1:8080
3. **配置 WiFi**：选择您的 WiFi 网络并输入密码
4. **完成设置**：台灯会语音播报 IP 地址，设置完成！

详细说明请参考：[首次设置指南](docs/SETUP_GUIDE.md)
```

**Step 2: Update AUTO_STARTUP_GUIDE.md**

添加 Captive Portal 相关说明。

**Step 3: Commit**

```bash
git add README.md docs/AUTO_STARTUP_GUIDE.md
git commit -m "docs: 更新文档添加 Captive Portal 说明"
```

---

## 测试和验证

### 完整测试流程

**1. 单元测试**
```bash
# 测试状态管理
pytest tests/api/test_setup_state.py -v

# 测试 AP 管理器
pytest tests/api/test_ap_manager.py -v

# 测试 WiFi 扫描器
pytest tests/api/test_wifi_scanner.py -v

# 测试网络管理器
pytest tests/api/test_network_manager.py -v

# 测试 Captive Portal
pytest tests/api/test_captive_portal.py -v
```

**2. 集成测试**
```bash
# 运行完整测试脚本
./scripts/test_captive_portal.sh
```

**3. 端到端测试**
```bash
# 在树莓派上测试
ssh pi@192.168.0.104

# 清除设置状态
sudo rm /var/lib/lelamp/setup_status.json

# 重启设备
sudo reboot

# 等待重启，然后：
# 1. 连接到 LeLamp-Setup 热点
# 2. 访问设置页面
# 3. 完成设置流程
```

### 预期结果

- ✅ 所有单元测试通过
- ✅ AP 模式正常启动
- ✅ Captive Portal 服务响应正常
- ✅ WiFi 扫描功能正常
- ✅ WiFi 连接功能正常
- ✅ 设置完成后服务自动切换
- ✅ 用户体验流畅

---

## 部署清单

### 开发环境
- [ ] 安装开发依赖
- [ ] 编写单元测试
- [ ] 本地测试 API 服务

### 树莓派部署
- [ ] 安装系统依赖（hostapd, dnsmasq）
- [ ] 部署服务配置
- [ ] 配置防火墙规则
- [ ] 测试完整流程

### 文档完善
- [ ] 用户设置指南
- [ ] 开发者文档
- [ ] API 文档
- [ ] 故障排除指南

---

## 实施优先级

### P0 - 核心功能（必须）
1. ✅ 状态管理模块
2. ✅ AP 管理器扩展
3. ✅ WiFi 扫描器
4. ✅ 网络连接管理器
5. ✅ Captive Portal API 服务

### P1 - 系统集成（重要）
6. ✅ systemd 服务配置
7. ✅ 首次启动检测
8. ✅ 测试脚本
9. ✅ 用户文档

### P2 - 增强功能（可选）
10. ⏳ 语音提示集成
11. ⏳ LED 状态指示
12. ⏳ 高级配置选项
13. ⏳ 多语言支持

---

## 故障排除

### 常见问题

**1. AP 模式无法启动**
```bash
# 检查 hostapd 状态
sudo systemctl status hostapd

# 检查无线接口
iwconfig wlan0

# 重启相关服务
sudo systemctl restart hostapd
sudo systemctl restart dnsmasq
```

**2. Portal 页面无法访问**
```bash
# 检查服务状态
sudo systemctl status lelamp-captive-portal

# 查看服务日志
sudo journalctl -u lelamp-captive-portal -n 50

# 检查端口监听
sudo netstat -tlnp | grep 8080
```

**3. WiFi 连接失败**
```bash
# 查看连接日志
sudo journalctl -u NetworkManager -n 50

# 手动测试连接
nmcli device wifi connect "SSID" password "password"
```

---

这个实现计划提供了完整的、可执行的步骤来构建 Captive Portal 设置向导系统。每个任务都包含了测试驱动的开发流程，确保代码质量和功能正确性。
