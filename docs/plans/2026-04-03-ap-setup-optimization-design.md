# AP 热点配网优化设计

日期：2026-04-03

## 目标

优化 LeLamp 首次配网体验，使其接近涂鸦 APP 的「添加设备」流程：
- 手机连上台灯热点后自动弹出配网页面（Captive Portal）
- 简化配网步骤从 6 步减到 4 步
- 连接成功后语音 + LED 反馈

## 现有问题

1. 手机连上 `LeLamp-Setup` 热点后不会自动弹出配网页面
2. 用户不知道要手动访问 `http://192.168.4.1:8000`
3. 6 步流程过于冗长（首次配网不需要注册/登录）
4. 连接成功后无语音/LED 反馈

## 设计方案

### 1. Captive Portal 自动跳转

**原理**：手机连上 WiFi 后会自动访问特定 URL 检测网络连通性。劫持这些请求，重定向到配网页面。

**实现**：
- dnsmasq 配置中将所有 DNS 请求解析到 `192.168.4.1`
- FastAPI 中间件拦截 Captive Portal 检测请求并返回 302 重定向：
  - Apple: `captive.apple.com` → 302 → `http://192.168.4.1:8000/setup`
  - Android: `connectivitycheck.gstatic.com/generate_204` → 302 → `http://192.168.4.1:8000/setup`
  - Windows: `www.msftconnecttest.com/redirect` → 302 → `http://192.168.4.1:8000/setup`
- AP 模式下所有非 API 的 HTTP 请求重定向到配网页面

**涉及文件**：
- `lelamp/api/services/ap_manager.py` — dnsmasq 配置添加 `address=/#/192.168.4.1`（已有）
- `lelamp/api/middleware/captive_portal.py` — 新建 Captive Portal 中间件
- `lelamp/api/app.py` — 注册中间件

### 2. 配网步骤简化（6 → 4 步）

| 步骤 | 当前 | 优化后 |
|------|------|--------|
| 1 | 欢迎 | 欢迎 + 设备发现确认 |
| 2 | 选择 WiFi | 选择 WiFi + 输入密码 |
| 3 | 输入密码 | （合并到步骤 2） |
| 4 | 连接中 | 连接中（实时进度） |
| 5 | 注册/登录 | （移除，后续在设置页绑定） |
| 6 | 完成 | 完成 + 语音/LED 反馈 |

**涉及文件**：
- `web/src/views/SetupWizardView.vue` — 重构为 4 步
- `lelamp/api/routes/system.py` — 调整 `/setup/complete` 逻辑

### 3. 连接成功反馈

- LED：蓝色呼吸 → 绿色常亮 3 秒 → 恢复正常
- 语音播报：「WiFi 连接成功」
- 自动完成配置（无需重启，直接切换到 WiFi 客户端模式）

**涉及文件**：
- `lelamp/api/services/wifi_manager.py` — 连接成功后触发回调
- `lelamp/api/routes/system.py` — `/setup/complete` 中添加反馈

### 4. 首次启动自动 AP 模式

- 确保 `first_boot_setup.sh` 正确检测 `setup_status.json`
- systemd 服务 `lelamp-setup-ap.service` 在首次启动时自动运行
- API 服务同时在 AP 模式下可用

**涉及文件**：
- `scripts/setup/first_boot_setup.sh`
- `scripts/services/lelamp-setup-ap.service`

## 不做的事情

- 不引入 BLE 配网（保持 AP 方案简单）
- 不修改设备绑定流程（保持现有 auto-bind 机制）
- 不修改 WiFi 连接核心逻辑（wifi_manager 保持不变）

## 验收标准

1. 手机连上 `LeLamp-Setup` 热点后自动弹出配网页面（iOS + Android）
2. 配网页面 4 步完成，无需注册/登录
3. WiFi 连接成功后 LED 变绿 + 语音播报
4. 配网完成后自动切换到 WiFi 客户端模式
5. 后续启动不再进入 AP 模式（除非重置）
