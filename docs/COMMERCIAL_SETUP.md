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
