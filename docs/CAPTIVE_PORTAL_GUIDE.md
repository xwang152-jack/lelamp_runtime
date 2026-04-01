# LeLamp Captive Portal 设置指南

## 🚀 快速开始

### 首次使用 LeLamp

当您第一次使用 LeLamp 台灯时，台灯会自动进入设置模式：

### 1. 连接热点

- 在您的手机或电脑上搜索 WiFi 网络
- 找到名为 **"LeLamp-Setup"** 的热点
- 密码：每次 AP 会话随机生成（显示在 Captive Portal 欢迎页面）
- LED 呈蓝色呼吸效果表示 AP 模式已激活

### 2. 打开设置页面

连接热点后，打开浏览器：
- 访问任意网址（会自动跳转到设置页面）
- 或直接访问：http://192.168.4.1:8080

### 3. 配置 WiFi

- 选择您的 WiFi 网络
- 输入 WiFi 密码
- 点击"连接"

### 4. 完成设置

- 等待连接成功
- 系统自动生成 `device_secret`（16 位十六进制字符），用于设备绑定
- 台灯会显示/播报 IP 地址
- 设置完成，开始使用！

> **提示**: 配置完成后可通过 mDNS 访问设备：`http://lelamp.local:8000`（macOS 自带 Bonjour，Linux 需安装 Avahi）

---

## 🔧 高级选项

### 手动触发设置模式

如果您需要重新配置 WiFi：

```bash
# 连接到台灯的 SSH
ssh pi@<台灯IP>

# 清除设置状态
sudo rm /var/lib/lelamp/setup_status.json

# 启动设置服务
sudo systemctl start lelamp-setup-ap
sudo systemctl start lelamp-captive-portal

# 或者直接重启台灯让它自动进入设置模式
sudo reboot
```

### 恢复出厂设置

```bash
# 在台灯上运行
sudo rm /var/lib/lelamp/setup_status.json
sudo reboot
```

---

## 📱 设置页面功能

设置向导包含以下功能：

| 功能 | 说明 |
|------|------|
| WiFi 扫描 | 自动发现附近的 WiFi 网络 |
| 信号强度 | 显示每个网络的信号强度 |
| 密码输入 | 安全的密码输入界面 |
| 连接测试 | 自动测试网络连接 |
| 错误处理 | 友好的错误提示和重试选项 |
| 进度反馈 | 实时显示连接进度 |

---

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
- 尝试使用其他浏览器（Chrome、Safari、Firefox）

### Q: WiFi 连接失败？

**A:** 请检查：
- WiFi 密码是否正确（区分大小写）
- 路由器是否正常工作
- 台灯是否在路由器信号范围内
- 尝试重启路由器和台灯

### Q: 设置完成后如何访问 Web 界面？

**A:** 设置完成后，台灯会显示 IP 地址。您可以通过以下方式访问：

- 通过 mDNS（推荐）：http://lelamp.local:8000（macOS 自带 / Linux 需 Avahi）
- 通过 IP 地址：http://`<台灯IP>`:8000
- 开发模式前端：http://`<台灯IP>`:5173

> **说明**: FastAPI 服务在同一端口 (8000) 同时托管 API 和 Vue 前端构建产物。需先执行 `bash scripts/build_web.sh` 构建前端。

### Q: 忘记台灯的 IP 地址？

**A:**
- 重新启动设置模式，查看新分配的 IP
- 登录路由器查看已连接设备列表
- 使用网络扫描工具（如 Fing）发现设备

---

## 🆘 需要帮助？

### 查看日志

```bash
# 设置日志
ssh pi@<台灯IP> 'sudo cat /var/log/lelamp/setup.log'

# 服务日志
ssh pi@<台灯IP> 'sudo journalctl -u lelamp-captive-portal -n 50'
```

### 诊断工具

```bash
# 检查服务状态
ssh pi@<台灯IP> 'sudo systemctl status lelamp-captive-portal'

# 检查端口监听
ssh pi@<台灯IP> 'sudo netstat -tlnp | grep 8080'
```

### 常见错误解决方案

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| AP 模式无法启动 | hostapd 配置错误 | 检查 /etc/hostapd.conf |
| DNS 不工作 | dnsmasq 未运行 | 重启 dnsmasq 服务 |
| Portal 无法访问 | 端口被占用 | 检查 8080 端口占用情况 |
| WiFi 连接失败 | 密码错误 | 重新输入正确密码 |

---

## 📖 技术详情

### 工作原理

```
开机
  ↓
first_boot_setup.sh 检查状态
  ↓
┌─────────────────────────────┐
│ setup_completed == true?      │
│ WiFi 已连接?                 │
└─────────────────────────────┘
  ↓ 是                    ↓ 否
正常启动              启动 AP 模式
                      ↓
                    启动 Portal 服务
                      ↓
                    用户连接热点
                      ↓
                    配置 WiFi
                      ↓
                    保存状态，重启服务
```

### 服务架构

| 服务 | 说明 | 端口 |
|------|------|------|
| lelamp-setup-ap | AP 模式管理 | - |
| lelamp-captive-portal | 设置向导服务 | 8080 |
| lelamp-livekit | 语音交互服务 | - |
| lelamp-api | REST API + Vue 前端托管 | 8000 |
| lelamp-frontend | Web 界面（开发模式） | 5173 |

### 状态文件

位置：`/var/lib/lelamp/setup_status.json`

```json
{
  "setup_completed": true,
  "setup_completed_at": "2026-03-19T12:00:00Z",
  "wifi_ssid": "MyHomeWiFi",
  "last_ip_address": "192.168.1.100",
  "device_secret": "a1b2c3d4e5f67890",
  "ap_password": "xK9mP2nQ"
}
```

**字段说明**：
- `device_secret`: 设备绑定密钥，首次 WiFi 设置时自动生成（16 位十六进制字符），可通过 `LELAMP_DEVICE_SECRET` 环境变量覆盖
- `ap_password`: 上次 AP 热点的随机密码（每次 AP 会话重新生成）

---

## 📚 更多资源

- [用户指南](USER_GUIDE.md)
- [自动启动配置指南](AUTO_STARTUP_GUIDE.md)
- [GitHub 项目](https://github.com/xwang152-jack/lelamp_runtime)
- [问题反馈](https://github.com/xwang152-jack/lelamp_runtime/issues)

---

**提示**：首次设置建议在 WiFi 信号良好的环境下进行。
