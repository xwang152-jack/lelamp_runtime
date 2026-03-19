# Captive Portal 设置向导设计文档

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**目标：** 解决用户首次使用 LeLamp 时无法通过 IP 地址访问的问题，通过 AP 模式 + Captive Portal 提供友好的首次设置体验。

**架构：** 扩展现有 AP 管理器，添加轻量级 Captive Portal 服务，实现自动检测、友好设置、状态管理的完整流程。

**技术栈：** FastAPI, hostapd, dnsmasq, NetworkManager, Vue.js 3

---

## 问题背景

用户拿到新的 LeLamp 台灯时，面临以下问题：
- 无法知道台灯的 IP 地址
- 没有网络连接无法访问 Web 界面
- 无法进行初始 WiFi 配置

**影响范围：**
- 首次使用用户
- WiFi 配置丢失的情况
- 网络环境变化需要重新配置

## 解决方案

### 方案选择

**方案一：智能 AP 模式 + Web 设置向导（已选中）**

基于现有的 `ap_manager.py` 和 `first_boot_setup.sh`，创建完整的 Captive Portal 解决方案。

**核心优势：**
- 复用现有代码，开发快速
- 用户体验友好，界面现代化
- 支持多种触发条件
- 便于未来扩展功能

---

## 系统架构

### 组件架构

```
┌─────────────────────────────────────────────────────────────┐
│                    LeLamp 首次设置系统                        │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  AP 模式管理 (扩展 ap_manager.py)                     │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │  │
│  │  │ hostapd      │  │ dnsmasq      │  │ AP Config   │ │  │
│  │  │ (热点创建)   │  │ (DNS/DHCP)   │  │ (配置管理)  │ │  │
│  │  └──────────────┘  └──────────────┘  └─────────────┘ │  │
│  └───────────────────────────────────────────────────────┘  │
│                          ↓                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Captive Portal 服务 (新增)                           │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │  │
│  │  │ FastAPI      │  │ WiFiScanner  │  │ SetupWizard │ │  │
│  │  │ (Web 服务)   │  │ (网络扫描)   │  │ (设置逻辑)  │ │  │
│  │  └──────────────┘  └──────────────┘  └─────────────┘ │  │
│  └───────────────────────────────────────────────────────┘  │
│                          ↓                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  状态管理 (新增)                                       │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │  │
│  │  │ SetupState   │  │ NetworkState │  │ ErrorRecovery│ │  │
│  │  │ (设置状态)   │  │ (网络状态)   │  │ (错误恢复)  │ │  │
│  │  └──────────────┘  └──────────────┘  └─────────────┘ │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Web 用户界面 (新增)                                   │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │  │
│  │  │ 欢迎页面     │  │ WiFi 选择    │  │ 连接进度    │ │  │
│  │  │ 密码输入     │  │ 完成页面     │  │ 错误处理    │ │  │
│  │  └──────────────┘  └──────────────┘  └─────────────┘ │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 服务依赖关系

```
启动顺序：
1. lelamp-setup-ap.service       → 创建 AP 热点
2. lelamp-captive-portal.service → 启动 Web 设置向导
3. (设置完成) → 停止上述服务，启动正常服务
   - lelamp-livekit.service
   - lelamp-api.service
   - lelamp-frontend.service
```

---

## 数据流程

### 1. 启动检测流程

```
开机启动
    ↓
first_boot_setup.sh 检查状态
    ↓
检查 /var/lib/lelamp/setup_status.json
    ↓
┌─────────────────────────────┐
│ setup_completed == true?    │
└─────────────────────────────┘
    ↓ 是              ↓ 否
正常启动          检查 WiFi 连接状态
                      ↓
                  ┌─────────────────────────────┐
                  │ WiFi 已连接?                │
                  └─────────────────────────────┘
                      ↓ 是              ↓ 否
                  正常启动          启动 AP 模式
                                        ↓
                                  启动 Captive Portal
```

### 2. 用户设置流程

```
用户设备连接 "LeLamp-Setup" 热点
    ↓
获取 IP 地址 (192.168.4.x)
    ↓
用户访问任意网址
    ↓
DNS 重定向到 192.168.4.1:8080
    ↓
显示设置向导欢迎页面
    ↓
扫描可用 WiFi 网络
    ↓
用户选择 WiFi + 输入密码
    ↓
尝试连接 WiFi
    ↓
┌─────────────────────────────┐
│ 连接成功?                   │
└─────────────────────────────┘
    ↓ 是              ↓ 否
测试网络连接      显示错误信息
    ↓                  ↓
保存配置         提供重试选项
    ↓                  ↓
重启网络服务    ┌─────────────────┐
    ↓            │ 3次失败?       │
语音播报 IP    └─────────────────┘
    ↓                ↓ 是        ↓ 否
完成          跳过 WiFi    返回重试
              进入离线模式
```

### 3. 错误恢复流程

```
WiFi 连接失败
    ↓
显示错误信息（密码错误/信号弱/超时）
    ↓
┌─────────────────────────────┐
│ 用户选择重试?               │
└─────────────────────────────┘
    ↓ 是              ↓ 否
重新尝试连接    ┌─────────────────────────────┐
    ↓            │ connection_attempts >= 3?  │
    └────────────┴─────────────────────────────┘
                     ↓ 是                ↓ 否
              提供高级选项         返回 WiFi 选择
              - 跳过 WiFi 设置
              - 手动输入 SSID
              - 恢复出厂设置
```

### 4. 状态持久化

**状态文件：** `/var/lib/lelamp/setup_status.json`

```json
{
  "setup_completed": false,
  "setup_started_at": "2026-03-19T12:00:00Z",
  "setup_completed_at": null,
  "current_step": "wifi_selection",
  "wifi_ssid": null,
  "connection_attempts": 0,
  "error_message": null,
  "last_ip_address": null,
  "ap_mode_count": 1,
  "network_history": []
}
```

---

## 用户界面设计

### 页面结构

**1. 欢迎页面 (`/`)**
```
┌─────────────────────────────────┐
│                                 │
│     [LeLamp Logo]               │
│                                 │
│   欢迎使用 LeLamp 智能台灯       │
│                                 │
│   让我们帮您连接到 WiFi 网络     │
│                                 │
│   [开始设置] 按钮               │
│                                 │
│   简单说明：                    │
│   1. 选择您的 WiFi 网络         │
│   2. 输入密码                   │
│   3. 完成设置                   │
│                                 │
└─────────────────────────────────┘
```

**2. WiFi 选择页面 (`/wifi-select`)**
```
┌─────────────────────────────────┐
│   选择 WiFi 网络               │
│   [🔍 搜索中...] [刷新按钮]     │
│                                 │
│ ┌─────────────────────────────┐│
│ │ 📶 HomeNetwork         🔒  ││
│ │ 信号强度: 强                ││
│ └─────────────────────────────┘│
│                                 │
│ ┌─────────────────────────────┐│
│ │ 📶 OfficeWiFi         🔒  ││
│ │ 信号强度: 中                ││
│ └─────────────────────────────┘│
│                                 │
│ ┌─────────────────────────────┐│
│ │ 📶 GuestWiFi          🔒  ││
│ │ 信号强度: 弱                ││
│ └─────────────────────────────┘│
│                                 │
│   [手动输入网络名称]            │
│   [返回]                       │
└─────────────────────────────────┘
```

**3. 密码输入页面 (`/password-input`)**
```
┌─────────────────────────────────┐
│   连接到 "HomeNetwork"         │
│                                 │
│   WiFi 密码:                   │
│   [••••••••••]  [👁️ 显示]     │
│                                 │
│   [连接] 按钮                  │
│   [返回] 按钮                  │
│                                 │
│   💡 密码提示：                 │
│   - 请确保输入正确              │
│   - 区分大小写                  │
│   - 检查特殊字符                │
└─────────────────────────────────┘
```

**4. 连接进度页面 (`/connecting`)**
```
┌─────────────────────────────────┐
│   正在连接...                  │
│                                 │
│   [████████░░] 60%             │
│                                 │
│   当前步骤:                    │
│   ✅ 验证密码                  │
│   🔄 获取 IP 地址              │
│   ⏳ 测试网络连接              │
│                                 │
│   请稍候...                    │
└─────────────────────────────────┘
```

**5. 完成页面 (`/complete`)**
```
┌─────────────────────────────────┐
│                                 │
│      [✅ 成功图标]              │
│                                 │
│   设置完成！                   │
│                                 │
│   您的台灯 IP 地址:            │
│   🌐 192.168.1.100             │
│                                 │
│   [完成] 按钮                  │
│                                 │
│   下次访问:                    │
│   http://192.168.1.100:5173    │
│   或 http://lelamp.local:5173   │
│                                 │
└─────────────────────────────────┘
```

**6. 错误页面 (`/error`)**
```
┌─────────────────────────────────┐
│      [❌ 错误图标]              │
│                                 │
│   连接失败                     │
│                                 │
│   错误信息:                    │
│   "密码错误或网络不可达"       │
│                                 │
│   [重试] 按钮                  │
│   [选择其他网络] 按钮          │
│   [高级选项] 按钮              │
│                                 │
│   高级选项:                    │
│   - 跳过 WiFi 设置             │
│   - 手动输入网络名称           │
│   - 恢复出厂设置               │
└─────────────────────────────────┘
```

### 界面特性

- **响应式设计**：支持手机、平板、桌面设备
- **动画效果**：平滑的页面切换和加载动画
- **实时反馈**：连接状态实时更新
- **错误处理**：友好的错误提示和解决方案
- **无障碍支持**：清晰的文字和高对比度
- **中文界面**：完整的本地化支持

---

## 技术实现

### 后端 API 设计

**新增文件：** `lelamp/api/services/captive_portal.py`

**核心类：**
```python
class CaptivePortalService:
    """Captive Portal 服务管理"""

    async def start_portal(self) -> None
    async def stop_portal(self) -> None
    async def scan_networks(self) -> List[WiFiNetwork]
    async def connect_wifi(self, ssid: str, password: str) -> bool
    async def test_connection(self) -> Dict[str, Any]
    async def complete_setup(self) -> None
```

**API 端点：**
```python
# 获取设置状态
GET /api/setup/status
Response: {
  "setup_completed": false,
  "current_step": "wifi_selection",
  "error_message": null
}

# 扫描可用网络
GET /api/setup/networks
Response: {
  "networks": [
    {
      "ssid": "HomeNetwork",
      "signal_strength": 85,
      "encryption": "WPA2",
      "channel": 6
    }
  ]
}

# 连接到 WiFi
POST /api/setup/connect
Request: {
  "ssid": "HomeNetwork",
  "password": "password123"
}
Response: {
  "success": true,
  "ip_address": "192.168.1.100",
  "message": "连接成功"
}

# 测试连接
GET /api/setup/test-connection
Response: {
  "connected": true,
  "internet_available": true,
  "latency_ms": 25
}

# 完成设置
POST /api/setup/complete
Response: {
  "success": true,
  "ip_address": "192.168.1.100"
}

# 获取配置
GET /api/setup/config
Response: {
  "ap_ssid": "LeLamp-Setup",
  "ap_password": "lelamp123",
  "portal_port": 8080
}
```

### 前端实现

**新增目录：** `web/captive-portal/`

**文件结构：**
```
web/captive-portal/
├── index.html              # 主页面
├── css/
│   └── styles.css          # 样式文件
├── js/
│   ├── app.js              # 主应用逻辑
│   ├── api.js              # API 调用封装
│   └── utils.js            # 工具函数
└── assets/
    └── images/             # 图片资源
        ├── logo.png
        ├── success.png
        └── error.png
```

**核心 JavaScript 模块：**
```javascript
// 应用状态管理
class AppState {
  constructor() {
    this.currentStep = 'welcome';
    this.selectedNetwork = null;
    this.connectionAttempts = 0;
    this.error = null;
  }

  setState(step, data) { /* ... */ }
  getState() { /* ... */ }
}

// API 调用封装
class SetupAPI {
  async getStatus() { /* ... */ }
  async scanNetworks() { /* ... */ }
  async connectWifi(ssid, password) { /* ... */ }
  async testConnection() { /* ... */ }
  async completeSetup() { /* ... */ }
}

// 页面导航
class Router {
  navigateTo(page) { /* ... */ }
  getCurrentPage() { /* ... */ }
}
```

### DNS 重定向配置

**dnsmasq 配置文件：** `/etc/dnsmasq.conf`

```bash
# 监听接口
interface=wlan0
bind-interfaces

# DHCP 配置
dhcp-range=192.168.4.100,192.168.4.200,12h
dhcp-option=3,192.168.4.1
dhcp-option=6,192.168.4.1

# DNS 重定向（核心配置）
address=/#/192.168.4.1
server=8.8.8.8
server=8.8.4.4

# 日志
log-queries
log-dhcp
```

### AP 配置

**hostapd 配置文件：** `/etc/hostapd/hostapd.conf`

```bash
# 接口配置
interface=wlan0
driver=nl80211

# 热点配置
ssid=LeLamp-Setup
hw_mode=g
channel=6

# 认证配置
auth_algs=1
wpa=2
wpa_passphrase=lelamp123
wpa_key_mgmt=WPA-PSK
wpa_pairwise=CCMP

# 其他配置
rsn_pairwise=CCMP
beacon_int=100
```

### 状态管理实现

**新增文件：** `lelamp/api/services/setup_state.py`

```python
class SetupStateManager:
    """设置状态管理器"""

    def __init__(self, state_file: str = "/var/lib/lelamp/setup_status.json"):
        self.state_file = state_file
        self._lock = threading.Lock()

    def load_state(self) -> Dict[str, Any]:
        """加载状态"""

    def save_state(self, state: Dict[str, Any]) -> None:
        """保存状态"""

    def update_step(self, step: str) -> None:
        """更新当前步骤"""

    def increment_attempts(self) -> int:
        """增加连接尝试次数"""

    def set_error(self, error: str) -> None:
        """设置错误信息"""

    def complete_setup(self, ip_address: str) -> None:
        """标记设置完成"""
```

### 错误处理机制

**错误类型定义：**
```python
class SetupError(Exception):
    """设置错误基类"""

class WiFiScanError(SetupError):
    """WiFi 扫描失败"""

class WiFiConnectionError(SetupError):
    """WiFi 连接失败"""

class PasswordError(SetupError):
    """密码错误"""

class TimeoutError(SetupError):
    """连接超时"""

class IPAddressError(SetupError):
    """IP 地址获取失败"""
```

**错误处理策略：**
```python
async def handle_wifi_connection(ssid: str, password: str):
    """处理 WiFi 连接，包含错误恢复"""
    max_attempts = 3
    attempts = 0

    while attempts < max_attempts:
        try:
            result = await connect_wifi(ssid, password)
            return result
        except PasswordError:
            # 密码错误，不重试
            raise
        except (ConnectionError, TimeoutError):
            attempts += 1
            if attempts >= max_attempts:
                raise MaxRetriesExceededError()
            await asyncio.sleep(2)
```

---

## 系统集成

### systemd 服务配置

**1. AP 设置服务**
```ini
# /etc/systemd/system/lelamp-setup-ap.service
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
```

**2. Captive Portal 服务**
```ini
# /etc/systemd/system/lelamp-captive-portal.service
[Unit]
Description=LeLamp Captive Portal
After=lelamp-setup-ap.service
Before=lelamp-livekit.service lelamp-api.service

[Service]
Type=simple
User=root
WorkingDirectory=/home/pi/lelamp_runtime
Environment=PATH=/usr/local/bin:/usr/bin:/bin
ExecStart=/usr/local/bin/uv run python -m lelamp.api.services.captive_portal
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### 服务启动逻辑

**修改 `first_boot_setup.sh`：**
```bash
#!/bin/bash
# 首次启动检测脚本

STATUS_FILE="/var/lib/lelamp/setup_status.json"

# 检查是否已完成设置
if [ -f "$STATUS_FILE" ] && grep -q '"setup_completed": true' "$STATUS_FILE"; then
    # 设置已完成，正常启动
    exit 0
fi

# 检查 WiFi 连接状态
if nmcli -t -f ACTIVE,SSID connection show --active | grep -q '^yes:'; then
    # 已连接 WiFi，标记为已完成
    update_setup_status "completed"
    exit 0
fi

# 需要进入设置模式
echo "Entering setup mode..."
systemctl start lelamp-setup-ap.service
systemctl start lelamp-captive-portal.service
```

### 依赖管理

**新增系统依赖：**
```bash
# AP 模式依赖
hostapd      # WiFi 热点创建
dnsmasq      # DNS 和 DHCP 服务器
iw           # 无线网络工具
wireless-tools# 无线工具集

# 已有依赖（项目中已有）
python3-fastapi>=0.104.0
python3-requests>=2.31.0
python3-websockets>=11.0
```

**安装脚本更新：**
```bash
# scripts/install_ap_dependencies.sh
#!/bin/bash
echo "安装 AP 模式依赖..."

sudo apt-get update
sudo apt-get install -y hostapd dnsmasq iw wireless-tools

# 配置 hostapd
sudo systemctl unmask hostapd
sudo systemctl enable hostapd

echo "AP 依赖安装完成"
```

---

## 部署策略

### 开发环境

**1. 本地开发**
```bash
# 安装依赖
uv sync --extra api

# 本地运行 Portal（模拟模式）
uv run python -m lelamp.api.services.captive_portal --dev

# 访问本地开发服务器
open http://localhost:8080
```

**2. 硬件测试**
```bash
# 部署到树莓派
./scripts/push_to_pi.sh

# 手动启动 AP 模式测试
ssh pi@192.168.0.104 'sudo systemctl start lelamp-setup-ap'

# 连接到 "LeLamp-Setup" 热点
# 访问 http://192.168.4.1:8080
```

### 生产环境部署

**1. 自动部署脚本**
```bash
# scripts/deploy_captive_portal.sh
#!/bin/bash
echo "部署 Captive Portal 系统..."

# 1. 安装系统依赖
./scripts/install_ap_dependencies.sh

# 2. 部署代码
./scripts/push_to_pi.sh

# 3. 配置 systemd 服务
ssh pi@192.168.0.104 'sudo systemctl daemon-reload'
ssh pi@192.168.0.104 'sudo systemctl enable lelamp-setup-ap.service'
ssh pi@192.168.0.104 'sudo systemctl enable lelamp-captive-portal.service'

# 4. 测试服务
echo "测试 AP 模式..."
ssh pi@192.168.0.104 'sudo systemctl start lelamp-setup-ap'
ssh pi@192.168.0.104 'sudo systemctl status lelamp-setup-ap'

echo "部署完成！"
```

**2. 配置管理**
```bash
# 创建配置目录
ssh pi@192.168.0.104 'sudo mkdir -p /etc/lelamp'
ssh pi@192.168.0.104 'sudo mkdir -p /var/lib/lelamp'

# 设置权限
ssh pi@192.168.0.104 'sudo chown root:root /etc/lelamp'
ssh pi@192.168.0.104 'sudo chmod 755 /etc/lelamp'
```

---

## 测试策略

### 单元测试

**测试文件：** `tests/api/test_captive_portal.py`

```python
import pytest
from lelamp.api.services.captive_portal import CaptivePortalService
from lelamp.api.services.setup_state import SetupStateManager

class TestCaptivePortalService:
    """Captive Portal 服务测试"""

    def test_scan_networks(self):
        """测试 WiFi 扫描"""
        service = CaptivePortalService()
        networks = await service.scan_networks()
        assert isinstance(networks, list)
        assert len(networks) > 0

    def test_connect_wifi_success(self):
        """测试成功连接 WiFi"""
        service = CaptivePortalService()
        result = await service.connect_wifi("TestNetwork", "password")
        assert result["success"] is True
        assert "ip_address" in result

    def test_connect_wifi_wrong_password(self):
        """测试密码错误"""
        service = CaptivePortalService()
        with pytest.raises(PasswordError):
            await service.connect_wifi("TestNetwork", "wrong_password")

class TestSetupStateManager:
    """状态管理器测试"""

    def test_save_and_load_state(self):
        """测试状态保存和加载"""
        manager = SetupStateManager()
        state = {
            "setup_completed": False,
            "current_step": "wifi_selection"
        }
        manager.save_state(state)
        loaded = manager.load_state()
        assert loaded == state

    def test_update_step(self):
        """测试步骤更新"""
        manager = SetupStateManager()
        manager.update_step("password_input")
        state = manager.load_state()
        assert state["current_step"] == "password_input"
```

### 集成测试

**测试场景：**
```bash
# tests/integration/test_setup_flow.sh
#!/bin/bash
echo "集成测试：完整设置流程"

# 1. 清除设置状态
ssh pi@192.168.0.104 'sudo rm -f /var/lib/lelamp/setup_status.json'

# 2. 重启设备
ssh pi@192.168.0.104 'sudo reboot'

# 3. 等待 AP 模式启动
sleep 30

# 4. 检查 AP 服务状态
assert_ssh 'sudo systemctl is-active lelamp-setup-ap' == 'active'

# 5. 检查 Portal 服务状态
assert_ssh 'sudo systemctl is-active lelamp-captive-portal' == 'active'

# 6. 测试 DNS 重定向
assert_dns_redirect "192.168.4.1"

# 7. 测试 WiFi 连接流程
# (需要自动化测试脚本)

echo "集成测试通过"
```

### 端到端测试

**测试场景：**
1. **首次开机设置**
   - 清除所有配置
   - 开机验证自动进入 AP 模式
   - 连接热点并完成设置

2. **WiFi 连接失败**
   - 模拟密码错误场景
   - 验证错误提示和重试功能

3. **网络环境变化**
   - 设置完成后更换 WiFi
   - 验证手动触发 AP 模式

4. **设置完成后重启**
   - 验证正常启动流程
   - 验证不再进入 AP 模式

---

## 故障排除

### 常见问题诊断

**1. AP 模式无法启动**
```bash
# 检查 hostapd 状态
sudo systemctl status hostapd

# 检查无线接口
iwconfig wlan0

# 检查 AP 配置
sudo hostapd -d /etc/hostapd/hostapd.conf

# 常见解决方法
sudo rfkill unblock wifi
sudo iw dev wlan0 interface add wlan0.ap type __ap
```

**2. DNS 重定向不工作**
```bash
# 检查 dnsmasq 状态
sudo systemctl status dnsmasq

# 测试 DNS 解析
nslookup google.com 192.168.4.1

# 检查防火墙规则
sudo iptables -t nat -L -n -v

# 重启 dnsmasq
sudo systemctl restart dnsmasq
```

**3. Portal 页面无法访问**
```bash
# 检查 Portal 服务状态
sudo systemctl status lelamp-captive-portal

# 查看服务日志
sudo journalctl -u lelamp-captive-portal -n 50

# 测试端口监听
sudo netstat -tlnp | grep 8080

# 手动启动服务
/usr/local/bin/uv run python -m lelamp.api.services.captive_portal
```

**4. WiFi 连接失败**
```bash
# 检查 NetworkManager 状态
sudo systemctl status NetworkManager

# 查看可用网络
nmcli device wifi list

# 查看连接日志
sudo journalctl -u NetworkManager -n 50

# 手动测试连接
nmcli device wifi connect "SSID" password "password"
```

### 诊断工具

**综合诊断脚本：**
```bash
#!/bin/bash
# scripts/diagnose_setup.sh

echo "=== LeLamp 设置诊断 ==="

echo "1. 检查 AP 服务状态"
sudo systemctl status lelamp-setup-ap --no-pager

echo "2. 检查 Portal 服务状态"
sudo systemctl status lelamp-captive-portal --no-pager

echo "3. 检查无线接口"
iwconfig wlan0

echo "4. 检查 hostapd"
sudo systemctl status hostapd --no-pager

echo "5. 检查 dnsmasq"
sudo systemctl status dnsmasq --no-pager

echo "6. 检查端口监听"
sudo netstat -tlnp | grep -E "(8080|53|67)"

echo "7. 检查设置状态"
cat /var/lib/lelamp/setup_status.json

echo "=== 诊断完成 ==="
```

### 恢复出厂设置

**恢复脚本：**
```bash
#!/bin/bash
# scripts/factory_reset.sh

echo "恢复出厂设置..."

# 停止所有服务
sudo systemctl stop lelamp-{livekit,api,frontend,setup-ap,captive-portal}.service

# 清除配置文件
sudo rm -f /var/lib/lelamp/setup_status.json
sudo rm -f /etc/lelamp/*.conf

# 清除 WiFi 配置
nmcli connection delete "LeLamp-Setup" 2>/dev/null || true

# 重启设备
sudo reboot
```

---

## 安全考虑

### AP 模式安全

**1. 热点认证**
- 使用 WPA2-PSK 加密
- 强密码策略（lelamp123 → 随机生成）
- 定期更换密码

**2. 网络隔离**
- AP 模式下禁止外网访问
- 仅允许必要的本地通信
- 防火墙规则限制

**3. 会话管理**
- 设置会话超时（30分钟）
- 单次会话限制
- 完成后自动关闭 AP 模式

### 数据安全

**1. 密码处理**
- 密码不在日志中显示
- 内存中加密存储
- 传输过程 HTTPS 加密

**2. 配置文件权限**
```bash
# 设置状态文件权限
sudo chmod 600 /var/lib/lelamp/setup_status.json
sudo chown root:root /var/lib/lelamp/setup_status.json

# 设置配置目录权限
sudo chmod 755 /etc/lelamp
sudo chown root:root /etc/lelamp
```

---

## 性能考虑

### 资源占用

**AP 模式资源消耗：**
- CPU: ~5-10%
- 内存: ~50MB
- 网络: 管理 wlan0 接口

**Portal 服务资源消耗：**
- CPU: ~2-5%
- 内存: ~30MB
- 磁盘: ~10MB（代码+静态文件）

### 优化策略

**1. 延迟加载**
- WiFi 扫描按需执行
- 页面资源懒加载
- API 响应缓存

**2. 后台任务**
- 网络状态定期检查
- 后台 WiFi 扫描（缓存结果）
- 自动清理过期会话

---

## 用户体验优化

### 语音提示集成

**设置过程中的语音反馈：**
```python
# 在关键步骤添加语音提示
async def voice_prompts():
    await speak("欢迎使用 LeLamp 智能台灯")
    await speak("请连接到 LeLamp-Setup 热点进行设置")
    # ... 完成设置后
    await speak(f"设置完成，我的 IP 地址是 {ip_address}")
```

### LED 状态指示

**设置模式 LED 状态：**
- **AP 模式启动**：蓝色呼吸灯
- **等待连接**：绿色慢闪
- **正在连接**：黄色快闪
- **连接成功**：白色常亮
- **连接失败**：红色快闪

### 移动端优化

**响应式设计断点：**
```css
/* 手机 */
@media (max-width: 480px) {
  .container { padding: 16px; }
  .button { width: 100%; }
}

/* 平板 */
@media (min-width: 481px) and (max-width: 768px) {
  .container { padding: 24px; }
  .button { width: 80%; }
}

/* 桌面 */
@media (min-width: 769px) {
  .container { max-width: 480px; margin: 0 auto; }
}
```

---

## 文档和培训

### 用户文档

**新增文档文件：**
1. `docs/SETUP_GUIDE.md` - 首次设置指南
2. `docs/TROUBLESHOOTING_SETUP.md` - 设置问题排除
3. `docs/FAQ.md` - 常见问题解答

### 开发者文档

**技术文档：**
1. `docs/ARCHITECTURE_CAPTIVE_PORTAL.md` - 架构设计
2. `docs/API_REFERENCE_SETUP.md` - API 参考
3. `docs/TESTING_GUIDE.md` - 测试指南

---

## 未来扩展

### 计划功能

**短期（1-2个月）：**
1. 账户注册和登录集成
2. 设备绑定功能
3. 高级网络配置（静态 IP、代理等）

**中期（3-6个月）：**
1. 多语言支持
2. 无障碍功能增强
3. 设置模板和预设

**长期（6-12个月）：**
1. 远程协助功能
2. 批量配置工具
3. 云端备份和恢复

### 扩展接口

**插件化架构：**
```python
class SetupPlugin:
    """设置插件基类"""

    def on_setup_start(self) -> None:
        """设置开始时调用"""

    def on_wifi_connected(self, ssid: str) -> None:
        """WiFi 连接成功时调用"""

    def on_setup_complete(self) -> None:
        """设置完成时调用"""

# 示例插件
class AccountSetupPlugin(SetupPlugin):
    def on_setup_complete(self):
        # 引导用户创建账户
        pass
```

---

## 总结

本设计文档描述了完整的 AP 模式 + Captive Portal 解决方案，解决了用户首次使用 LeLamp 时无法通过 IP 地址访问的核心问题。

**关键特性：**
- ✅ 自动检测和触发 AP 模式
- ✅ 友好的 Web 设置向导
- ✅ 完整的错误处理和恢复机制
- ✅ 基于现有代码扩展，开发成本可控
- ✅ 良好的用户体验和移动端支持

**实施优先级：**
1. **P0（核心功能）：** AP 模式、WiFi 扫描、连接功能
2. **P1（用户体验）：** Web 界面、状态管理、错误处理
3. **P2（增强功能）：** 语音提示、LED 指示、高级选项

**预期效果：**
- 用户首次设置成功率 > 95%
- 平均设置时间 < 3 分钟
- 用户满意度显著提升
