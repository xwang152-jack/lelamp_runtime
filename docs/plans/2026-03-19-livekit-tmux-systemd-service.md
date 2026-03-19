# LiveKit Tmux Systemd Service Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 创建通过 tmux 后台会话运行 LiveKit Console 模式的 systemd 服务，实现开机自启和自动重启。

**Architecture:** 使用 systemd 服务管理 tmux 会话，tmux 会话中运行 `python3 main.py Console`，既满足 LiveKit 的 Console 模式要求，又实现后台持久化和自动重启。

**Tech Stack:** systemd, tmux, Python 3.12+, LiveKit Agents

---

## Task 1: 创建 Systemd 服务文件

**Files:**
- Create: `scripts/lelamp-livekit.service`

**Step 1: 创建 systemd 服务配置文件**

创建 `scripts/lelamp-livekit.service` 文件：

```ini
[Unit]
Description=LeLamp LiveKit Agent (via tmux)
Documentation=https://github.com/xwang152-jack/lelamp_runtime
After=network.target

[Service]
Type=forking
User=root
WorkingDirectory=/home/pi/lelamp_runtime
Environment=PATH=/usr/local/bin:/usr/bin:/bin
EnvironmentFile=/home/pi/lelamp_runtime/.env

# 启动 tmux 会话运行 LiveKit Console 模式
ExecStart=/usr/bin/tmux new-session -d -s livekit -n livekit 'cd /home/pi/lelamp_runtime && /usr/local/bin/uv run python main.py Console'

# 优雅停止：发送 Ctrl+C 然后关闭会话
ExecStop=/usr/bin/tmux send-keys -t livekit C-c
ExecStop=/bin/sleep 2
ExecStop=/usr/bin/tmux kill-session -t livekit

# 自动重启配置
Restart=always
RestartSec=10
StartLimitInterval=60
StartLimitBurst=3

# 日志配置
StandardOutput=journal
StandardError=journal
SyslogIdentifier=lelamp-livekit

# 安全设置
NoNewPrivileges=false
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

**Step 2: 创建自动化部署脚本**

创建 `scripts/setup_livekit_tmux_service.sh` 脚本：

```bash
#!/bin/bash
# LeLamp LiveKit Tmux Service 设置脚本

set -e

PI_HOST="${PI_HOST:-pi@192.168.0.104}"
PROJECT_DIR="/home/pi/lelamp_runtime"
SERVICE_FILE="scripts/lelamp-livekit.service"

echo "================================================"
echo "LeLamp LiveKit Tmux Service 设置"
echo "================================================"
echo ""

# 检查是否可以连接到树莓派
echo "🔍 检查树莓派连接..."
if ! ssh $PI_HOST "echo '连接成功'" 2>/dev/null; then
    echo "❌ 无法连接到树莓派 $PI_HOST"
    echo "请检查："
    echo "  1. 树莓派是否开机"
    echo "  2. 网络连接是否正常"
    echo "  3. SSH 密钥是否配置"
    exit 1
fi

echo "✅ 树莓派连接正常"
echo ""

# 检查 tmux 是否安装
echo "🔍 检查 tmux 是否安装..."
if ! ssh $PI_HOST "which tmux" 2>/dev/null; then
    echo "📦 正在安装 tmux..."
    ssh $PI_HOST "sudo apt-get update && sudo apt-get install -y tmux"
else
    echo "✅ tmux 已安装"
fi
echo ""

# 停止并禁用旧的 LiveKit 服务（如果存在）
echo "🔧 1. 清理旧服务..."
ssh $PI_HOST "sudo systemctl stop lelamp-livekit.service 2>/dev/null || true"
ssh $PI_HOST "sudo systemctl disable lelamp-livekit.service 2>/dev/null || true"
# 同时停止可能存在的旧服务
ssh $PI_HOST "sudo systemctl stop lelamp-agent.service 2>/dev/null || true"
ssh $PI_HOST "sudo systemctl disable lelamp-agent.service 2>/dev/null || true"
echo "✅ 旧服务已清理"
echo ""

# 创建新的 systemd 服务文件
echo "📝 2. 创建 systemd 服务文件..."
cat $SERVICE_FILE | ssh $PI_HOST "sudo tee /etc/systemd/system/lelamp-livekit.service > /dev/null"
echo "✅ 服务文件已创建"
echo ""

# 重新加载 systemd
echo "🔄 3. 重新加载 systemd..."
ssh $PI_HOST "sudo systemctl daemon-reload"
echo "✅ systemd 已重新加载"
echo ""

# 启用服务
echo "🚀 4. 启用开机自启..."
ssh $PI_HOST "sudo systemctl enable lelamp-livekit.service"
echo "✅ 服务已设置为开机自启"
echo ""

# 启动服务
echo "▶️  5. 启动服务..."
ssh $PI_HOST "sudo systemctl start lelamp-livekit.service"
echo "✅ 服务已启动"
echo ""

# 等待服务启动
sleep 3

# 检查服务状态
echo ""
echo "================================================"
echo "🔍 服务状态检查"
echo "================================================"
echo ""
ssh $PI_HOST "sudo systemctl status lelamp-livekit.service --no-pager -l" | head -20
echo ""

# 检查 tmux 会话
echo "🔍 Tmux 会话状态:"
ssh $PI_HOST "tmux list-sessions 2>/dev/null || echo 'tmux 会话不存在或正在启动'"
echo ""

echo ""
echo "================================================"
echo "✅ LiveKit Tmux Service 设置完成！"
echo "================================================"
echo ""
echo "🎯 管理命令："
echo ""
echo "查看服务状态:"
echo "  ssh $PI_HOST 'sudo systemctl status lelamp-livekit.service'"
echo ""
echo "查看实时日志:"
echo "  ssh $PI_HOST 'sudo journalctl -u lelamp-livekit.service -f'"
echo ""
echo "连接到 tmux 会话（查看实时输出）:"
echo "  ssh $PI_HOST 'tmux attach -t livekit'"
echo "  退出 tmux 会话：按 Ctrl+B 然后 D"
echo ""
echo "重启服务:"
echo "  ssh $PI_HOST 'sudo systemctl restart lelamp-livekit.service'"
echo ""
echo "停止服务:"
echo "  ssh $PI_HOST 'sudo systemctl stop lelamp-livekit.service'"
echo ""
echo "禁用开机自启:"
echo "  ssh $PI_HOST 'sudo systemctl disable lelamp-livekit.service'"
echo ""
```

**Step 3: 添加执行权限**

```bash
chmod +x scripts/setup_livekit_tmux_service.sh
```

**Step 4: 提交文件**

```bash
git add scripts/lelamp-livekit.service scripts/setup_livekit_tmux_service.sh
git commit -m "feat: 添加 LiveKit Tmux systemd 服务配置"
```

---

## Task 2: 更新文档

**Files:**
- Modify: `docs/AUTO_STARTUP_GUIDE.md`

**Step 1: 在 AUTO_STARTUP_GUIDE.md 中添加 LiveKit Tmux 服务说明**

在 `docs/AUTO_STARTUP_GUIDE.md` 文件的"启动模式说明"部分添加新的模式：

在"### 3. 主程序模式"之后添加：

```markdown
### 4. LiveKit Tmux 模式（推荐用于语音交互）⭐
- **适用场景**：日常语音交互、后台持久化运行
- **特点**：
  - 使用 tmux 后台会话运行 LiveKit Console 模式
  - 满足 LiveKit Console 要求，无需显示窗口
  - systemd 自动管理开机自启和重启
  - 支持手动连接 tmux 会话查看实时日志
  - 优雅停止（Ctrl+C）避免强制 kill

- **访问方式**：
  - 通过 LiveKit 客户端连接
  - SSH 连接 tmux 会话：`tmux attach -t livekit`
```

**Step 2: 添加设置说明**

在文档中添加新的设置章节：

```markdown
## 🚀 LiveKit Tmux 模式设置

### 快速设置

```bash
./scripts/setup_livekit_tmux_service.sh
```

### 管理命令

```bash
# 查看服务状态
ssh pi@192.168.0.104 'sudo systemctl status lelamp-livekit.service'

# 查看实时日志
ssh pi@192.168.0.104 'sudo journalctl -u lelamp-livekit.service -f'

# 连接到 tmux 会话（查看实时输出）
ssh pi@192.168.0.104 'tmux attach -t livekit'
# 退出 tmux 会话：按 Ctrl+B 然后 D

# 重启服务
ssh pi@192.168.0.104 'sudo systemctl restart lelamp-livekit.service'

# 停止服务
ssh pi@192.168.0.104 'sudo systemctl stop lelamp-livekit.service'
```

### 故障排除

```bash
# 检查 tmux 会话状态
ssh pi@192.168.0.104 'tmux list-sessions'

# 手动启动测试
ssh pi@192.168.0.104 'tmux new-session -s test -d "cd /home/pi/lelamp_runtime && uv run python main.py Console"'

# 查看详细日志
ssh pi@192.168.0.104 'sudo journalctl -u lelamp-livekit.service -n 100 --no-pager'
```
```

**Step 3: 提交文档更新**

```bash
git add docs/AUTO_STARTUP_GUIDE.md
git commit -m "docs: 添加 LiveKit Tmux 服务设置文档"
```

---

## Task 3: 创建测试脚本

**Files:**
- Create: `scripts/test_livekit_tmux_service.sh`

**Step 1: 创建测试脚本**

```bash
#!/bin/bash
# LiveKit Tmux Service 测试脚本

set -e

PI_HOST="${PI_HOST:-pi@192.168.0.104}"

echo "================================================"
echo "LeLamp LiveKit Tmux Service 测试"
echo "================================================"
echo ""

echo "🔍 测试 1: 检查服务状态"
if ssh $PI_HOST "sudo systemctl is-active lelamp-livekit.service" | grep -q "active"; then
    echo "✅ 服务运行中"
else
    echo "❌ 服务未运行"
    exit 1
fi
echo ""

echo "🔍 测试 2: 检查 tmux 会话"
if ssh $PI_HOST "tmux list-sessions | grep -q livekit"; then
    echo "✅ tmux 会话存在"
    ssh $PI_HOST "tmux list-sessions"
else
    echo "❌ tmux 会话不存在"
    exit 1
fi
echo ""

echo "🔍 测试 3: 检查服务日志"
echo "最近的日志："
ssh $PI_HOST "sudo journalctl -u lelamp-livekit.service -n 10 --no-pager"
echo ""

echo "🔍 测试 4: 测试服务重启"
echo "重启服务..."
ssh $PI_HOST "sudo systemctl restart lelamp-livekit.service"
sleep 5

if ssh $PI_HOST "sudo systemctl is-active lelamp-livekit.service" | grep -q "active"; then
    echo "✅ 服务重启成功"
else
    echo "❌ 服务重启失败"
    exit 1
fi
echo ""

echo "🔍 测试 5: 检查自动重启功能"
echo "杀死 tmux 会话..."
ssh $PI_HOST "tmux kill-session -t livekit" 2>/dev/null || true
sleep 10

if ssh $PI_HOST "sudo systemctl is-active lelamp-livekit.service" | grep -q "active"; then
    echo "✅ 自动重启功能正常"
else
    echo "❌ 自动重启功能失败"
    exit 1
fi
echo ""

echo "================================================"
echo "✅ 所有测试通过！"
echo "================================================"
```

**Step 2: 添加执行权限**

```bash
chmod +x scripts/test_livekit_tmux_service.sh
```

**Step 3: 提交测试脚本**

```bash
git add scripts/test_livekit_tmux_service.sh
git commit -m "test: 添加 LiveKit Tmux 服务测试脚本"
```

---

## Task 4: 创建辅助脚本

**Files:**
- Create: `scripts/livekit_service_manager.sh`

**Step 1: 创建服务管理脚本**

```bash
#!/bin/bash
# LeLamp LiveKit Service 管理脚本

PI_HOST="${PI_HOST:-pi@192.168.0.104}"

case "$1" in
    status)
        echo "📊 服务状态："
        ssh $PI_HOST 'sudo systemctl status lelamp-livekit.service --no-pager -l' | head -15
        echo ""
        echo "🖥️  Tmux 会话："
        ssh $PI_HOST 'tmux list-sessions 2>/dev/null || echo "无活跃会话"'
        ;;

    logs)
        echo "📋 实时日志（Ctrl+C 退出）："
        ssh $PI_HOST 'sudo journalctl -u lelamp-livekit.service -f'
        ;;

    attach)
        echo "🔌 连接到 tmux 会话（Ctrl+B 然后 D 退出）："
        ssh $PI_HOST 'tmux attach -t livekit'
        ;;

    restart)
        echo "🔄 重启服务..."
        ssh $PI_HOST 'sudo systemctl restart lelamp-livekit.service'
        echo "✅ 服务已重启"
        echo ""
        echo "📊 服务状态："
        ssh $PI_HOST 'sudo systemctl status lelamp-livekit.service --no-pager' | head -10
        ;;

    stop)
        echo "⏹️  停止服务..."
        ssh $PI_HOST 'sudo systemctl stop lelamp-livekit.service'
        echo "✅ 服务已停止"
        ;;

    start)
        echo "▶️  启动服务..."
        ssh $PI_HOST 'sudo systemctl start lelamp-livekit.service'
        echo "✅ 服务已启动"
        echo ""
        echo "📊 服务状态："
        ssh $PI_HOST 'sudo systemctl status lelamp-livekit.service --no-pager' | head -10
        ;;

    enable)
        echo "🚀 启用开机自启..."
        ssh $PI_HOST 'sudo systemctl enable lelamp-livekit.service'
        echo "✅ 已启用开机自启"
        ;;

    disable)
        echo "🔒 禁用开机自启..."
        ssh $PI_HOST 'sudo systemctl disable lelamp-livekit.service'
        echo "✅ 已禁用开机自启"
        ;;

    *)
        echo "LeLamp LiveKit Service 管理脚本"
        echo ""
        echo "用法: $0 {status|logs|attach|restart|stop|start|enable|disable}"
        echo ""
        echo "命令说明："
        echo "  status  - 查看服务状态和 tmux 会话"
        echo "  logs    - 查看实时日志"
        echo "  attach  - 连接到 tmux 会话（查看实时输出）"
        echo "  restart - 重启服务"
        echo "  stop    - 停止服务"
        echo "  start   - 启动服务"
        echo "  enable  - 启用开机自启"
        echo "  disable - 禁用开机自启"
        echo ""
        echo "示例："
        echo "  $0 status     # 查看状态"
        echo "  $0 logs       # 查看日志"
        echo "  $0 attach     # 连接到 tmux"
        exit 1
        ;;
esac
```

**Step 2: 添加执行权限**

```bash
chmod +x scripts/livekit_service_manager.sh
```

**Step 3: 提交管理脚本**

```bash
git add scripts/livekit_service_manager.sh
git commit -m "feat: 添加 LiveKit 服务管理脚本"
```

---

## Task 5: 更新主 README

**Files:**
- Modify: `README.md`

**Step 1: 在 README.md 中添加 LiveKit Tmux 服务说明**

在适当位置添加：

```markdown
## 🚀 快速启动

### LiveKit 语音交互模式（推荐）

使用 systemd + tmux 实现 LiveKit Console 模式的后台持久化运行：

```bash
# 设置 LiveKit Tmux 服务
./scripts/setup_livekit_tmux_service.sh

# 管理服务
./scripts/livekit_service_manager.sh status    # 查看状态
./scripts/livekit_service_manager.sh logs      # 查看日志
./scripts/livekit_service_manager.sh attach    # 连接到 tmux
```

详细说明请参考：[自动启动配置指南](docs/AUTO_STARTUP_GUIDE.md)
```

**Step 2: 提交 README 更新**

```bash
git add README.md
git commit -m "docs: 在 README 中添加 LiveKit Tmux 服务说明"
```

---

## 测试验证

### 完整测试流程

**1. 设置服务**
```bash
./scripts/setup_livekit_tmux_service.sh
```

**2. 验证服务状态**
```bash
./scripts/livekit_service_manager.sh status
```

**3. 查看日志**
```bash
./scripts/livekit_service_manager.sh logs
```

**4. 测试 tmux 连接**
```bash
./scripts/livekit_service_manager.sh attach
# 按 Ctrl+B 然后 D 退出
```

**5. 测试自动重启**
```bash
./scripts/test_livekit_tmux_service.sh
```

**6. 验证开机自启**
```bash
ssh pi@192.168.0.104 'sudo reboot'
# 等待重启后
ssh pi@192.168.0.104 'sudo systemctl status lelamp-livekit.service'
```

### 预期结果

- ✅ 服务启动成功
- ✅ tmux 会话正常运行
- ✅ 日志无错误信息
- ✅ 可以连接到 tmux 会话查看实时输出
- ✅ 服务可以正常重启和停止
- ✅ 开机自动启动
- ✅ 异常退出后自动重启

---

## 架构说明

### 为什么使用 tmux？

1. **满足 LiveKit Console 要求**：LiveKit 需要在 Console 模式下运行，不能使用 Headless 模式
2. **后台持久化**：tmux 会话在后台持续运行，即使 SSH 连接断开
3. **优雅停止**：通过 `tmux send-keys C-c` 发送 Ctrl+C，让程序优雅退出
4. **实时查看**：可以随时连接到 tmux 会话查看程序输出
5. **systemd 集成**：通过 systemd 实现开机自启和自动重启

### 服务配置要点

- `Type=forking`：tmux 创建会话后会立即退出，符合 forking 类型
- `ExecStart`：使用 `-d` 参数创建后台会话
- `ExecStop`：先发送 Ctrl+C，等待 2 秒，然后强制关闭会话
- `Restart=always`：任何异常退出都会自动重启
- `RestartSec=10`：重启间隔 10 秒，避免频繁重启

### 与其他服务的关系

- **lelamp-api.service**：API 服务器，提供 REST API 和 WebSocket
- **lelamp-livekit.service**：LiveKit 语音代理，提供语音交互能力
- **lelamp-setup.service**：首次启动配置服务

两个服务可以同时运行，分别处理不同的功能需求。
