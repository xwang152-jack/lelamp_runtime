# ⚠️ LeLamp 后端无法在 macOS 上运行

## 🎯 **问题**

```
ModuleNotFoundError: No module named 'rpi_ws281x'
```

### 原因

**LeLamp 后端是为 Raspberry Pi（Linux）设计的硬件控制软件**，依赖于：
- `rpi-ws281x` - RGB LED 控制（仅 Linux）
- 硬件端口 `/dev/ttyACM0` - 串口通信
- 摄像头设备 `/dev/video0` - 视频采集

**这些硬件依赖只能在 Raspberry Pi 上使用，不能在 macOS 上运行。**

---

## 💡 **解决方案**

### 方案 1: 使用 Raspberry Pi（推荐）⭐

**硬件需求**：
- Raspberry Pi 3B+ 或 4B
- Raspberry Pi OS (Linux)
- 摄像头（USB 或 CSI）
- LeLamp 硬件套件

**步骤**：
1. 准备 Raspberry Pi 硬件
2. 安装 Raspberry Pi OS
3. 克隆代码到 Pi
4. 安装依赖：`uv sync --all-extras`
5. 配置 `.env` 文件
6. 运行：`sudo uv run main.py console`

### 方案 2: 仅测试前端（推荐用于开发测试）✅

**如果你只想测试前端 UI**，可以：

#### 1. 启动前端开发服务器

```bash
cd /Users/jackwang/lelamp_runtime/web
pnpm dev
```

#### 2. 访问前端页面

- 连接页面: http://localhost:5173
- 控制台页面: 需要 Token，但无法连接真实后端

#### 3. 测试前端功能（不需要后端）

✅ 可以测试的 UI 功能：
- 表单验证
- 按钮交互
- 页面布局
- 响应式设计
- 颜色选择器
- 聊天界面

❌ 无法测试的功能（需要后端）：
- 实时视频流
- 语音对话
- 设备控制
- 灯光控制
- 电机动作

### 方案 3: 使用 LiveKit Cloud 测试（高级）

如果你想测试完整的实时通信，但不依赖硬件：

#### 1. 修改后端代码（跳过硬件初始化）

创建一个不依赖硬件的测试版本。

#### 2. 使用 LiveKit Cloud

- 连接到 LiveKit Cloud
- 测试音视频通话
- 测试数据通道
- 但无法控制物理设备

---

## 🎯 **推荐方案**

### 开发/测试前端

**在 macOS 上**：
```bash
cd web
pnpm dev
```

访问 http://localhost:5173，可以测试：
- ✅ 所有 UI 组件
- ✅ 页面布局
- ✅ 交互逻辑
- ❌ 无法连接真实后端

### 部署到 Raspberry Pi

**在 Raspberry Pi 上**：
```bash
# 1. 准备 Raspberry Pi 硬件
# 2. 安装系统
# 3. 克隆代码
git clone <repo>
cd <repo>

# 4. 安装依赖
uv sync --all-extras

# 5. 配置环境
cp .env.example .env
nano .env  # 填写配置

# 6. 运行后端
sudo uv run main.py console
```

---

## 📋 **平台兼容性**

| 功能 | macOS | Raspberry Pi (Linux) |
|------|-------|---------------------|
| 前端开发 | ✅ 完全支持 | ✅ 支持 |
| 前端测试 | ✅ UI 测试 | ✅ 完整测试 |
| 后端运行 | ❌ 硬件依赖 | ✅ 完整支持 |
| 摄像头 | ❌ 不支持 | ✅ 支持 |
| RGB LED | ❌ 不支持 | ✅ 支持 |
| 电机控制 | ❌ 不支持 | ✅ 支持 |
| 语音识别 | ✅ 支持 | ✅ 支持 |
| AI 对话 | ✅ 支持 | ✅ 支持 |

---

## 🎯 **当前建议**

### 立即可做

**在 macOS 上测试前端**：

```bash
cd /Users/jackwang/lelamp_runtime/web

# 运行前端测试
pnpm type-check
pnpm lint
pnpm build

# 启动开发服务器
pnpm dev
```

访问 http://localhost:5173 测试 UI。

### 完整测试

**需要 Raspberry Pi 硬件**：
1. 准备 Raspberry Pi 4B（推荐 8GB RAM）
2. 安装 Raspberry Pi OS
3. 设置硬件（摄像头、电机、LED）
4. 部署代码到 Pi
5. 运行完整系统

---

## 📝 **临时解决方案（仅用于演示）**

如果你想演示前端 UI 但没有硬件，可以：

1. **创建模拟后端**
   - 使用 LiveKit Cloud
   - 模拟设备响应
   - 演示 UI 交互

2. **使用截图/视频**
   - 录制演示视频
   - 展示 UI 功能
   - 说明需要硬件才能运行

3. **部署到测试 Pi**
   - 使用测试 Raspberry Pi
   - 远程访问进行演示

---

## 🎊 **总结**

### macOS (当前环境)
- ✅ 可以开发前端
- ✅ 可以测试前端 UI
- ❌ 无法运行后端（硬件依赖）

### Raspberry Pi
- ✅ 完整支持
- ✅ 所有功能可用
- ✅ 推荐用于生产环境

---

## 🚀 **立即行动**

### 选项 1: 测试前端（推荐当前环境）

```bash
cd web
pnpm dev
```

访问 http://localhost:5173

### 选项 2: 准备 Raspberry Pi（完整功能）

1. 获取 Raspberry Pi 4B
2. 安装 Raspberry Pi OS
3. 部署到 Pi
4. 完整测试

---

## 📞 **需要帮助？**

**当前推荐**：
- 在 macOS 上开发和测试前端 ✅
- 准备 Raspberry Pi 进行完整测试 🎯

**目标**：
- 前端 UI 开发和测试 ✅
- Raspberry Pi 部署完整系统 🎯

---

**记住**: LeLamp 是**硬件项目**，需要 Raspberry Pi 才能完整运行！

前端可以在 macOS 上开发，但完整功能需要硬件支持。

**立即开始**: `cd web && pnpm dev` 🚀
