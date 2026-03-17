# 前端 URL 配置指南

## 🎯 优化说明

**问题**: 之前用户需要在网页端手动输入 LiveKit URL 和 Token

**解决方案**: 预配置 URL，用户只需粘贴 Token

## 📝 配置步骤

### 1. 更新前端环境变量

#### 开发环境 (`web/.env.development`)

```bash
VITE_APP_TITLE=LeLamp Dev
# LiveKit URL (预配置)
# 替换为你的实际 LiveKit URL
VITE_LIVEKIT_URL=wss://your-livekit-url.livekit.cloud
```

#### 生产环境 (`web/.env.production`)

```bash
VITE_APP_TITLE=LeLamp
# LiveKit URL (预配置)
# 替换为你的实际 LiveKit URL
VITE_LIVEKIT_URL=wss://your-livekit-url.livekit.cloud
```

### 2. 获取 LiveKit URL

#### 方法 1: 从 LiveKit Cloud 获取

1. 访问 https://cloud.livekit.io
2. 登录你的账户
3. 选择项目
4. 进入 "Settings" → "URLs"
5. 复制 WebSocket URL (格式: `wss://xxx.livekit.cloud`)

#### 方法 2: 从自托管服务器获取

如果你使用自托管 LiveKit Server:
```bash
# 默认 WebSocket URL
wss://your-server.com:7880

# 或使用 HTTP (不推荐生产环境)
ws://your-server.com:7880
```

### 3. 更新配置文件

将获取到的 URL 填入环境变量文件：

```bash
# 示例
VITE_LIVEKIT_URL=wss://lelamp-test-abc123.livekit.cloud
```

### 4. 重启开发服务器

配置修改后需要重启：

```bash
# 停止当前服务器 (Ctrl+C)
# 重新启动
cd web
pnpm dev
```

## 🎯 用户体验改进

### 优化前
- ❌ 用户需要手动输入 LiveKit URL
- ❌ 用户需要手动输入 Token
- ❌ 容易输入错误

### 优化后
- ✅ URL 自动填充（已预配置）
- ✅ 用户只需粘贴 Token
- ✅ 显示配置状态提示
- ✅ 用户仍可修改 URL（灵活性）

## 🔐 安全说明

### ✅ 安全的做法

**前端环境变量** (仅 URL):
```bash
VITE_LIVEKIT_URL=wss://xxx.livekit.cloud  # ✅ 可以暴露
```

**后端环境变量** (包含敏感信息):
```bash
LIVEKIT_URL=wss://xxx.livekit.cloud
LIVEKIT_API_KEY=dev_xxxxx           # 🔐 敏感信息
LIVEKIT_API_SECRET=xxxxx            # 🔐 绝对不能暴露给前端
```

### ⚠️ 重要提醒

- ✅ 前端只配置 URL（非敏感）
- ✅ API Key 和 Secret 保留在后端
- ✅ Token 由后端生成，前端只使用
- ❌ 永远不要把 API Key/Secret 放在前端代码中

## 📊 配置状态检测

前端会自动检测配置状态：

### 状态 1: ✅ 已预配置
```
LiveKit Server URL
┌─────────────────────────────────┐
│ 🌐 wss://lelamp-test.livekit.cloud │
└─────────────────────────────────┘
✅ URL 已预配置，可直接使用或修改
```

### 状态 2: ⚠️ 未配置
```
LiveKit Server URL
┌─────────────────────────────────┐
│ 🌐                              │
└─────────────────────────────────┘
⚠️ 请在 web/.env.development 中配置 VITE_LIVEKIT_URL
```

## 🚀 使用流程

### 开发环境

1. **配置 URL** (首次)
   ```bash
   cd web
   nano .env.development
   # 填入 VITE_LIVEKIT_URL
   ```

2. **启动前端**
   ```bash
   pnpm dev
   ```

3. **生成 Token** (每次连接)
   ```bash
   ./quick_start.sh
   ```

4. **连接设备**
   - 访问 http://localhost:5173
   - URL 已自动填充 ✅
   - 粘贴 Token
   - 点击"连接设备"

### 生产环境

1. **构建前端**
   ```bash
   cd web
   # 更新 .env.production
   pnpm build
   ```

2. **部署**
   ```bash
   # 部署 dist/ 目录到你的服务器
   ```

3. **用户访问**
   - URL 已预配置
   - 只需输入 Token

## 🔧 故障排查

### 问题 1: URL 未自动填充

**症状**: URL 输入框为空，显示警告提示

**解决方案**:
```bash
# 1. 检查环境变量文件
cat web/.env.development

# 2. 确认格式正确（没有多余的引号或空格）
VITE_LIVEKIT_URL=wss://xxx.livekit.cloud

# 3. 重启开发服务器
cd web
pnpm dev
```

### 问题 2: 连接失败

**症状**: 点击"连接设备"后提示连接失败

**检查清单**:
- ✅ LiveKit URL 是否正确？
- ✅ Token 是否有效（未过期）？
- ✅ 后端服务是否正在运行？
- ✅ 网络连接是否正常？

### 问题 3: Token 生成失败

**症状**: 运行 `./quick_start.sh` 无输出或报错

**解决方案**:
```bash
# 1. 检查后端配置
cat .env | grep LIVEKIT

# 2. 确认 API Key 和 Secret 正确

# 3. 手动测试
cd /Users/jackwang/lelamp_runtime
uv run python scripts/generate_client_token.py --room test --user test
```

## 📚 相关文档

- **优化方案**: `URL_CONFIG_OPTIMIZATION.md` - 详细的技术方案
- **Token 生成**: `QUICK_REFERENCE.md` - 快速参考指南
- **用户指南**: `docs/USER_GUIDE.md` - 完整使用文档

## 💡 最佳实践

1. **开发环境**
   - 在 `web/.env.development` 中配置开发用 LiveKit URL
   - 使用测试 Token

2. **生产环境**
   - 在 `web/.env.production` 中配置生产用 LiveKit URL
   - 定期更新 Token
   - 使用 HTTPS

3. **团队协作**
   - 将 `.env.example` 提交到 Git（模板）
   - 将 `.env.local` 加入 `.gitignore`（个人配置）
   - 在团队文档中记录 LiveKit URL

---

**配置完成后，用户体验将显著提升！** 🎉
