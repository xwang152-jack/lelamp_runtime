# LeLamp 商业化部署设计

> 日期：2026-04-01
> 状态：已批准

## 目标

B2C 硬件售卖场景，用户开箱即用，AP 热点 + Captive Portal 配置，出厂预装 License，本地注册/登录账号。

## 用户旅程

```
1. 开箱 → 插电开机
2. 手机 WiFi 列表出现 "LeLamp-XXXX"（AP 热点，随机密码）
3. 连接热点 → 浏览器自动弹出 / 访问 http://192.168.4.1:8000
4. 看到 LeLamp 欢迎页 + 设备信息卡片（设备 ID、状态）
5. WiFi 配置 → 连接家庭 WiFi
6. 注册/登录（本地账号）
7. 登录即自动绑定设备 → 显示"配置完成"
8. 设备重启 → 进入正常模式
9. 用户通过 http://lelamp.local:8000 访问完整控制面板
10. 前端自动获取 LiveKit Token（已认证）→ 开始语音对话
```

## 设计方案：自动化部署脚本 + 前端引导增强

### 1. 出厂预配置脚本

**新增 `scripts/factory/prepare_factory_env.sh`**

一键完成出厂预配置：

1. 生成 `device_id`（调用 `lelamp.utils.security.get_device_id()`）
2. 生成 `device_secret`（`secrets.token_hex(8)`）
3. 生成 License Key（调用 `generate_license_key(device_id, secret)`）
4. 生成 `JWT_SECRET`（`secrets.token_hex(32)`）
5. 生成 AP 热点随机密码（`secrets.token_urlsafe(6)`）
6. 将以上全部写入 `/opt/lelamp/.env`
7. 初始化 `setup_status.json`（`setup_completed: false`，写入 `device_secret`）
8. 禁用 `LELAMP_DEV_MODE=0`
9. 确保所有 systemd 服务已安装并启用（AP、Captive Portal、LeLamp API）
10. 打印设备信息摘要（device_id + device_secret），用于出厂记录

**关键**：API 密钥（DEEPSEEK、BAIDU 等）出厂时统一写入，所有设备共用同一套 API Key。

### 2. 开机流程（改造 `first_boot_setup.sh`）

```
开机 → first_boot_setup.sh 检测 setup_status.json
  ├─ setup_completed=false → 启动 AP 热点 + Captive Portal
  │   用户手机连接 "LeLamp-XXXX" 热点
  │   → 浏览器弹出配置页面（或访问 http://192.168.4.1:8000）
  │   → Setup Wizard: WiFi 配置 → 注册/登录 → 设备自动绑定
  │   → 写入 WiFi 配置，重启进入客户端模式
  │
  └─ setup_completed=true → 正常启动 API 服务 + LiveKit Agent
      用户通过 http://lelamp.local:8000 访问
```

### 3. 前端引导增强（改造 `SetupWizardView.vue`）

当前 5 步（欢迎→WiFi→密码→连接→完成）扩展为 6 步：

```
步骤 1: 欢迎
步骤 2: WiFi 扫描 + 选择
步骤 3: WiFi 密码输入
步骤 4: WiFi 连接验证
步骤 5: 注册/登录（新增）
  - 已有账号：输入用户名+密码登录
  - 新用户：注册（用户名+邮箱+密码）
  - 登录后自动调用 /bind-device
步骤 6: 完成 → 显示设备信息 + 访问地址
```

**设备绑定自动化**：步骤 5 登录成功后，前端自动调用 `POST /api/auth/bind-device`，`device_id` 和 `device_secret` 由后端自动提供（无需用户手动输入），实现"登录即绑定"。

### 4. 设备信息页面（无登录可访问）

改造 `GET /api/system/device` 端点：
- 当前需要 JWT 认证 → 改为**无需认证即可访问设备基本信息**
- 返回：`device_id`、`hostname`、`ip`、`mac`、`is_bound`、`setup_completed`
- **不返回** `device_secret`、`jwt_secret` 等敏感信息

前端新增"设备信息卡片"组件，未登录状态下显示设备名称、状态、IP 地址。

### 5. 自动绑定后端支持

新增 `POST /api/auth/auto-bind` 端点：
- 需要有效 JWT（已登录）
- 后端自动从 `setup_status.json` 读取 `device_id` 和 `device_secret`
- 自动创建设备绑定（permission_level=admin）
- 首次绑定成功后标记 `setup_status.json` 中 `user_bound: true`
- 已绑定时返回友好提示而非错误

## 涉及的文件变更

### 新增文件
- `scripts/factory/prepare_factory_env.sh` — 出厂预配置脚本

### 修改文件
- `scripts/setup/first_boot_setup.sh` — 改造开机流程
- `web/src/views/SetupWizardView.vue` — 扩展为 6 步（+注册/登录步骤）
- `web/src/views/DeviceManageView.vue` — 设备信息卡片（无登录可见）
- `lelamp/api/routes/system.py` — 设备信息端点无需认证
- `lelamp/api/routes/auth.py` — 新增 auto-bind 端点
- `lelamp/api/services/auth_service.py` — auto-bind 服务逻辑

## 安全考虑

- `device_secret` 仅在出厂脚本输出中显示，不返回给前端
- auto-bind 端点需要有效 JWT，防止未认证绑定
- AP 热点密码每次随机生成，不再使用固定密码 `lelamp123`
- 设备信息端点不暴露任何敏感信息
