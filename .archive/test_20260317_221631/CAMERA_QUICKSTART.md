# 📹 摄像头画面开启 - 快速指南

## 🎯 最简单的方法

### 步骤 1: 生成 Token

```bash
./quick_start.sh
```

这会自动生成 Token 并复制到剪贴板。

### 步骤 2: 启动后端（包括摄像头）

```bash
sudo uv run main.py console
```

**后端会自动**:
- ✅ 启动摄像头 📹
- ✅ 连接到 LiveKit
- ✅ 推送视频流

### 步骤 3: 启动前端（新终端）

```bash
cd web
pnpm dev
```

### 步骤 4: 连接浏览器

1. 访问: http://localhost:5173
2. 粘贴 Token（Cmd+V 或从 `/tmp/lelamp_quick_connect.txt` 复制）
3. 点击"连接设备"
4. **看到摄像头画面！** ✅

---

## 🔍 详细说明

### 摄像头如何工作

```
后端服务 (main.py)
    ↓
启动摄像头 📹
    ↓
推送到 LiveKit
    ↓
前端接收
    ↓
显示在浏览器 🖥️
```

**关键点**: 摄像头由**后端**控制，前端只是**显示**。

---

## ⚠️ 常见问题

### Q: 看不到摄像头画面？

**A: 检查后端是否启动**

```bash
# 查看后端进程
ps aux | grep "python.*main.py"

# 如果没有运行，启动它
sudo uv run main.py console
```

### Q: 后端启动失败？

**A: 检查权限和硬件**

```bash
# 需要 sudo 访问摄像头
sudo uv run main.py console

# 检查摄像头是否存在
ls /dev/video*
```

### Q: Token 无效？

**A: 重新生成 Token**

```bash
./quick_start.sh
```

---

## 🚀 快速命令参考

```bash
# 1. 生成 Token（一次性）
./quick_start.sh

# 2. 启动后端（终端 1）
sudo uv run main.py console

# 3. 启动前端（终端 2）
cd web && pnpm dev

# 4. 访问浏览器
# http://localhost:5173
# 粘贴 Token → 连接
```

---

## 📝 Token 位置

生成后的 Token 保存在：

- **剪贴板**: 直接粘贴（Cmd+V）
- **文件**: `/tmp/lelamp_quick_connect.txt`
- **屏幕**: 运行 `./quick_start.sh` 时会显示

---

## ✅ 成功标志

### 后端终端显示：
```
[INFO] LeLamp agent starting...
[INFO] Connected to LiveKit
[INFO] Camera initialized
[INFO] Ready to accept connections
```

### 浏览器显示：
- 📹 视频区域显示摄像头画面
- 🔴 隐私指示器：摄像头已开启
- 💡 LED 指示灯：红色

---

**记住**: 摄像头由**后端**管理，前端只是**接收端**！

**快速开始**: `./quick_start.sh` → 启动后端 → 启动前端 → 连接浏览器 ✅
