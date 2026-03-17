# 🔐 LeLamp 授权配置指南

## 🎯 问题解决

### 错误信息
```
设备授权校验失败。请检查 LELAMP_LICENSE_KEY 配置。
```

### 解决方案

## 📝 方案 1: 开发模式（推荐，最简单）⭐

在 `.env` 文件中添加：

```bash
LELAMP_DEV_MODE=1
```

**即可跳过授权检查！**

---

## 🔐 方案 2: 生成正式授权码

### 步骤 1: 设置签名密钥

在 `.env` 文件中添加：

```bash
# 生成一个随机密钥（至少 32 字符）
LELAMP_LICENSE_SECRET=your-random-secret-key-at-least-32-chars-long
```

生成随机密钥：
```bash
# macOS/Linux
openssl rand -hex 32

# 或使用 Python
python3 -c "import secrets; print(secrets.token_hex(32))"
```

### 步骤 2: 获取设备 ID

运行以下命令获取设备 ID：

```bash
cd /Users/jackwang/lelamp_runtime
uv run python -c "from lelamp.utils.security import get_device_id; print(f'Device ID: {get_device_id()}')"
```

输出类似：
```
Device ID: 33988068396959
```

### 步骤 3: 生成授权码

使用设备 ID 和密钥生成授权码：

```bash
cd /Users/jackwang/lelamp_runtime
uv run python -c "
from lelamp.utils.security import generate_license_key, get_device_id
device_id = get_device_id()
license_key = generate_license_key(device_id)
print(f'Device ID: {device_id}')
print(f'License Key: {license_key}')
"
```

### 步骤 4: 添加到 .env

将生成的授权码添加到 `.env`：

```bash
LELAMP_LICENSE_KEY=生成的16位授权码
```

---

## ✅ 验证配置

配置完成后，验证是否生效：

```bash
cd /Users/jackwang/lelamp_runtime
uv run python -c "
from lelamp.utils.security import verify_license
result = verify_license()
print(f'授权校验结果: {result}')
"
```

如果输出 `授权校验结果: True`，说明配置成功！

---

## 🎯 快速修复（推荐）

**最简单的方法**：在 `.env` 文件中添加一行：

```bash
LELAMP_DEV_MODE=1
```

然后重新启动后端：

```bash
sudo uv run main.py console
```

**问题解决！** ✅

---

## 📋 配置示例

完整的 `.env` 配置示例：

```bash
# 开发模式（推荐用于测试）
LELAMP_DEV_MODE=1

# 或者使用正式授权（生产环境）
# LELAMP_LICENSE_SECRET=your-random-secret-key-at-least-32-chars
# LELAMP_LICENSE_KEY=your-generated-16-char-key
```

---

## 🔍 授权说明

- **开发模式** (`LELAMP_DEV_MODE=1`)：跳过所有授权检查，适合开发和测试
- **正式授权**：使用设备 ID 和密钥生成授权码，适合生产环境

**推荐**：
- 开发/测试：使用 `LELAMP_DEV_MODE=1`
- 生产环境：使用正式授权码

---

## 🚀 立即解决

**最快速的解决方案**：

```bash
# 1. 添加开发模式
echo "LELAMP_DEV_MODE=1" >> .env

# 2. 重新启动后端
sudo uv run main.py console
```

**完成！** ✅
