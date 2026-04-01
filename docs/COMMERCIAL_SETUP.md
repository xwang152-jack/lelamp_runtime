# LeLamp 商业化配置指南

本文档说明如何配置 LeLamp 商业化版本。

## 两种商业模式

| 模式 | 配置要求 | 适用场景 | 启动方式 |
|------|----------|----------|----------|
| **本地版本** | Console + 许可证验证 | 个人/家庭使用，台灯本机操作 | `sudo uv run main.py console` |
| **远程版本** | LiveKit Room + 许可证验证 | 手机 App 远程控制 | 通过 LiveKit Room 连接 |

## 快速开始

### 1. 交互式配置向导（推荐）

```bash
# 启动配置向导
uv run python scripts/generate_license.py --wizard
```

向导会引导您完成：
- 选择商业模式（本地/远程）
- 生成签名密钥
- 获取设备 ID
- 生成授权码
- 自动配置 .env 文件

### 2. 手动配置步骤

#### 步骤 1：生成签名密钥（只需执行一次）

```bash
# 生成强随机密钥
uv run python scripts/generate_license.py --generate-secret
```

输出示例：
```
LELAMP_LICENSE_SECRET=0f518a13d181c75f3258bbd3b1bed1c6c6cebcb62e5ebf8e250a2c4826dfdd5a
```

**重要**：将此密钥添加到 .env 文件，并妥善保管，不要泄露！

#### 步骤 2：获取设备 ID

```bash
# 在目标设备上运行
uv run python scripts/generate_license.py --device-id
```

输出示例：
```
当前设备 ID: 208773663939684
```

#### 步骤 3：生成授权码

```bash
# 为指定设备生成授权码
LELAMP_LICENSE_SECRET=<你的密钥> uv run python scripts/generate_license.py --license-for <设备ID>
```

输出示例：
```
授权码: ce17c88b64885b45
```

#### 步骤 4：配置 .env 文件

将以下内容添加到 .env 文件：

```bash
# 商业化配置（必需）
LELAMP_LICENSE_KEY=ce17c88b64885b45
LELAMP_LICENSE_SECRET=0f518a13d181c75f3258bbd3b1bed1c6c6cebcb62e5ebf8e250a2c4826dfdd5a
LELAMP_DEV_MODE=0

# 必需 API
DEEPSEEK_API_KEY=sk-xxxxx
BAIDU_SPEECH_API_KEY=xxxxx
BAIDU_SPEECH_SECRET_KEY=xxxxx

# 远程版本还需要
LIVEKIT_URL=wss://your-server.livekit.cloud
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret

# JWT 认证密钥（生产环境必须设置）
LELAMP_JWT_SECRET=your-strong-random-secret-key

# 设备绑定密钥（首次 WiFi 设置时自动生成，也可手动设置）
LELAMP_DEVICE_SECRET=

# Web 前端构建路径（默认: web/dist）
LELAMP_WEB_DIST=web/dist
```

#### 步骤 5：验证配置

```bash
# 验证配置是否正确
uv run python scripts/generate_license.py --verify
```

## 命令参考

### 查看设备 ID

```bash
uv run python scripts/generate_license.py --device-id
```

### 生成签名密钥

```bash
uv run python scripts/generate_license.py --generate-secret
```

### 为指定设备生成授权码

```bash
# 方式 1：通过环境变量传递密钥
LELAMP_LICENSE_SECRET=your_secret uv run python scripts/generate_license.py --license-for device_id

# 方式 2：通过命令行参数传递密钥
uv run python scripts/generate_license.py --license-for device_id --secret your_secret
```

### 设置开发模式

```bash
# 开启开发模式（跳过授权检查）
uv run python scripts/generate_license.py --dev-mode 1

# 关闭开发模式（需要授权验证）
uv run python scripts/generate_license.py --dev-mode 0
```

## 设备绑定机制

设备绑定通过 `device_secret` 实现，用于设备与云端服务的安全通信。

### 工作流程

1. **自动生成**：设备首次完成 WiFi 设置时，自动生成 16 位 hex 字符串的 `device_secret`
2. **持久化存储**：`device_secret` 存储在 `/var/lib/lelamp/setup_status.json`
3. **查询接口**：`GET /api/system/device` 返回设备信息及 `device_secret`
4. **安全验证**：密钥比对使用 `hmac.compare_digest()`（常量时间比较），防止时序攻击

### 手动配置

如果需要手动设置设备绑定密钥（例如回退机制）：

```bash
# 在 .env 文件中设置
LELAMP_DEVICE_SECRET=your-16-char-hex-secret
```

> **注意**：`LELAMP_DEVICE_SECRET` 是回退机制，优先使用 `setup_status.json` 中的自动生成值。

## JWT 认证

API 服务使用 JWT 进行身份认证，保护设备绑定和 LiveKit Token 等敏感接口。

### 配置

```bash
# 生产环境必须设置强随机密钥
LELAMP_JWT_SECRET=your-strong-random-secret-key
```

- **密钥未设置**：启动时自动生成随机密钥并输出警告（每次重启密钥变化，会导致已签发的 Token 失效）
- **生产环境**：务必设置固定密钥

### LiveKit Token 获取

商业化部署中，LiveKit Token 通过 API 端点获取，不再使用 CLI 脚本：

```
POST /api/livekit/token
Authorization: Bearer <jwt_token>
```

- 需要有效的 JWT 认证
- 强制使用已认证用户身份（不可伪造其他用户）
- 详见 [LiveKit Token 管理指南](COMMERCIAL_LIVEKIT_TOKEN_GUIDE.md)

## 安全注意事项

1. **LELAMP_LICENSE_SECRET 必须保密**
   - 此密钥用于生成所有设备的授权码
   - 只在生成授权码的服务器上使用
   - 不要分发给最终用户

2. **LELAMP_LICENSE_KEY 可以分发**
   - 这是设备授权码，可以安全地提供给用户
   - 每个设备有唯一的授权码

3. **生产环境**
   - 设置 `LELAMP_DEV_MODE=0` 启用授权验证
   - 确保 .env 文件不被提交到版本控制系统

## 故障排查

### 授权验证失败

```bash
# 检查设备 ID
uv run python scripts/generate_license.py --device-id

# 检查配置
uv run python scripts/generate_license.py --verify

# 查看详细日志
LOG_LEVEL=DEBUG sudo uv run main.py console
```

### 开发环境测试

在开发环境中，可以设置 `LELAMP_DEV_MODE=1` 跳过授权检查：

```bash
uv run python scripts/generate_license.py --dev-mode 1
```

## API 配置要求

### 本地版本（Console 模式）

| 配置项 | 是否必需 | 说明 |
|--------|----------|------|
| `DEEPSEEK_API_KEY` | ✅ 必需 | LLM 服务 |
| `BAIDU_SPEECH_API_KEY` | ✅ 必需 | 语音识别 |
| `BAIDU_SPEECH_SECRET_KEY` | ✅ 必需 | 语音合成 |
| `LIVEKIT_URL` | ❌ 可选 | 不需要 |
| `LELAMP_LICENSE_KEY` | ✅ 必需 | 设备授权 |

### 远程版本（Room 模式）

| 配置项 | 是否必需 | 说明 |
|--------|----------|------|
| `DEEPSEEK_API_KEY` | ✅ 必需 | LLM 服务 |
| `BAIDU_SPEECH_API_KEY` | ✅ 必需 | 语音识别 |
| `BAIDU_SPEECH_SECRET_KEY` | ✅ 必需 | 语音合成 |
| `LIVEKIT_URL` | ✅ 必需 | 远程连接 |
| `LIVEKIT_API_KEY` | ✅ 必需 | API 访问 |
| `LIVEKIT_API_SECRET` | ✅ 必需 | API 签名 |
| `LELAMP_LICENSE_KEY` | ✅ 必需 | 设备授权 |
| `LELAMP_JWT_SECRET` | ✅ 必需 | JWT 签名密钥（生产环境） |
| `LELAMP_DEVICE_SECRET` | ❌ 自动 | 设备绑定密钥（首次 WiFi 设置自动生成） |
| `LELAMP_WEB_DIST` | ❌ 可选 | Vue 前端路径（默认 web/dist） |
