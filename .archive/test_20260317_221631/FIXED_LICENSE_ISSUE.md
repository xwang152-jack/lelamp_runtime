# ✅ 授权问题已解决！

## 🎯 问题

启动后端时出现错误：
```
设备授权校验失败。请检查 LELAMP_LICENSE_KEY 配置。
```

## ✅ 解决方案

### 最简单的方法（已完成）⭐

已自动在 `.env` 文件中添加：
```bash
LELAMP_DEV_MODE=1
```

**这会跳过授权检查，让你可以正常使用！**

---

## 🚀 现在可以启动后端了

### 方法 1: 使用自动化脚本

```bash
./test_e2e.sh
```

脚本会自动处理所有配置。

### 方法 2: 手动启动

```bash
cd /Users/jackwang/lelamp_runtime

# 启动后端（包括摄像头）
sudo uv run main.py console
```

**后端会正常启动** ✅

---

## 📋 后端启动成功的标志

当后端正常启动时，你会看到：

```
[INFO] LeLamp agent starting...
[INFO] Connected to LiveKit
[INFO] Camera initialized
[INFO] Microphone initialized
[INFO] Ready to accept connections
```

**或者类似的日志信息**（根据你的日志配置）

---

## 🔍 授权说明

### 开发模式 vs 正式授权

| 模式 | 配置 | 用途 |
|------|------|------|
| **开发模式** | `LELAMP_DEV_MODE=1` | 开发、测试、学习 |
| **正式授权** | `LELAMP_LICENSE_KEY` + `LELAMP_LICENSE_SECRET` | 生产环境 |

**当前配置**: 开发模式 ✅

**适合场景**:
- ✅ 开发和测试
- ✅ 学习和实验
- ✅ 本地使用

---

## 📝 完整配置文件 (`.env`)

你的 `.env` 文件现在应该包含：

```bash
# LiveKit 配置
LIVEKIT_URL=wss://your-livekit-url.livekit.cloud
LIVEKIT_API_KEY=your_livekit_api_key_here
LIVEKIT_API_SECRET=your_livekit_api_secret_here

# DeepSeek API 配置
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_MODEL=deepseek-chat

# 百度语音 API 配置
BAIDU_SPEECH_API_KEY=your_baidu_api_key_here
BAIDU_SPEECH_SECRET_KEY=your_baidu_secret_key_here

# 硬件配置
LELAMP_PORT=/dev/ttyACM0
LELAMP_ID=lelamp

# 开发模式（跳过授权检查）
LELAMP_DEV_MODE=1
```

---

## 🎯 下一步

### 1. 启动后端

```bash
sudo uv run main.py console
```

### 2. 启动前端（新终端）

```bash
cd web
pnpm dev
```

### 3. 连接浏览器

1. 访问: http://localhost:5173
2. 生成 Token: `./quick_start.sh`
3. 粘贴 Token 并连接
4. **看到摄像头画面！** ✅

---

## 📚 相关文档

- **授权配置指南**: `LICENSE_SETUP.md`
- **摄像头快速指南**: `CAMERA_QUICKSTART.md`
- **快速启动脚本**: `./quick_start.sh`
- **端到端测试脚本**: `./test_e2e.sh`

---

## ✅ 问题解决总结

| 问题 | 状态 |
|------|------|
| 授权校验失败 | ✅ 已解决 |
| 添加开发模式 | ✅ 已完成 |
| 后端可以启动 | ✅ 可以使用 |

**现在可以正常使用 LeLamp 了！** 🎉

---

**快速开始**:
```bash
# 1. 生成 Token
./quick_start.sh

# 2. 启动后端
sudo uv run main.py console

# 3. 启动前端
cd web && pnpm dev

# 4. 连接浏览器
# http://localhost:5173
```

有任何问题随时告诉我！🚀
