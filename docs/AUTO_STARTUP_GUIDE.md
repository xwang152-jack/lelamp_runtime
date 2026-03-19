# LeLamp 自动启动配置指南

## 🚀 快速开始

### 方法零：首次设置（Captive Portal）⭐

新设备首次使用？无需知道 IP 地址！

```bash
# 在树莓派上安装 Captive Portal
./scripts/install_captive_portal.sh

# 清除设置状态（重新进入设置模式）
ssh pi@192.168.0.104 'sudo rm /var/lib/lelamp/setup_status.json'
```

然后：
1. 连接到 "LeLamp-Setup" WiFi 热点（密码：lelamp123）
2. 浏览器访问 http://192.168.4.1:8080
3. 按向导完成 WiFi 配置

详细说明请参考：[Captive Portal 设置指南](CAPTIVE_PORTAL_GUIDE.md)

### 方法一：完整服务系统（推荐）⭐

```bash
# 一键设置完整三服务架构
./scripts/setup_all_services.sh
```

**服务包含：**
- 📱 **LiveKit 服务**（语音交互）
- 🔌 **API 服务**（后端 REST API）
- 🌐 **Frontend 服务**（Web 用户界面）

**访问地址：**
- Web 界面：http://192.168.0.104:5173
- API 文档：http://192.168.0.104:8000/docs
- 健康检查：http://192.168.0.104:8000/health

### 方法二：仅 LiveKit 服务（纯语音交互）

```bash
# 仅设置 LiveKit Tmux 服务
./scripts/setup_livekit_tmux_service.sh
```

### 方法三：使用部署脚本（完整流程）

```bash
# 完整部署：构建前端 → 推送代码 → 配置自动启动
./scripts/deploy_to_pi.sh
```

### 方法四：分步配置

1. **构建并推送代码**
   ```bash
   ./scripts/push_to_pi.sh
   ```

2. **配置自动启动**
   ```bash
   ./scripts/setup_lelamp_startup.sh
   ```

## 📋 启动模式说明

### 1. 完整服务模式（推荐）⭐
- **适用场景**：日常使用、完整功能体验
- **特点**：
  - LiveKit 服务（语音交互）+ API 服务（后端）+ Frontend 服务（Web 界面）
  - 三服务协同工作，提供完整功能
  - systemd 统一管理，开机自动启动
  - 支持语音交互和 Web 控制两种方式

- **访问地址**：
  - Web 界面：http://192.168.0.104:5173
  - API 文档：http://192.168.0.104:8000/docs
  - LiveKit 语音交互：通过 LiveKit 客户端

### 2. 简化模式
- **适用场景**：资源受限环境、仅需 API 功能
- **特点**：
  - 仅启动后端 API 服务
  - 前端静态文件由 API 服务
  - 单一端口（8000）访问前后端
  - 占用资源少，启动快速

- **访问地址**：http://192.168.0.104:8000

### 3. 完整模式
- **适用场景**：开发调试、需要实时预览
- **特点**：
  - 后端 API + 前端开发服务器
  - 支持前端热更新
  - 占用资源较多，启动较慢

- **访问地址**：
  - 前端：http://192.168.0.104:5173
  - API：http://192.168.0.104:8000

### 4. 主程序模式
- **适用场景**：无头使用、纯语音交互
- **特点**：
  - 仅启动 main.py console
  - 纯命令行交互
  - 最低资源占用

- **访问方式**：SSH 到树莓派直接交互

### 5. LiveKit Tmux 模式（推荐用于语音交互）⭐
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

## 🛠️ 管理命令

### 完整服务系统管理

```bash
# 查看所有服务状态
ssh pi@192.168.0.104 'sudo systemctl status lelamp-{livekit,api,frontend}.service'

# 查看特定服务状态
ssh pi@192.168.0.104 'sudo systemctl status lelamp-livekit.service'
ssh pi@192.168.0.104 'sudo systemctl status lelamp-api.service'
ssh pi@192.168.0.104 'sudo systemctl status lelamp-frontend.service'
```

### 服务控制

```bash
# 重启所有服务
ssh pi@192.168.0.104 'sudo systemctl restart lelamp-{livekit,api,frontend}.service'

# 重启特定服务
ssh pi@192.168.0.104 'sudo systemctl restart lelamp-livekit.service'
ssh pi@192.168.0.104 'sudo systemctl restart lelamp-api.service'
ssh pi@192.168.0.104 'sudo systemctl restart lelamp-frontend.service'

# 停止所有服务
ssh pi@192.168.0.104 'sudo systemctl stop lelamp-{livekit,api,frontend}.service'

# 启动所有服务
ssh pi@192.168.0.104 'sudo systemctl start lelamp-{livekit,api,frontend}.service'
```

### 日志查看

```bash
# 实时查看所有服务日志
ssh pi@192.168.0.104 'sudo journalctl -u lelamp-{livekit,api,frontend}.service -f'

# 查看特定服务日志
ssh pi@192.168.0.104 'sudo journalctl -u lelamp-livekit.service -f'
ssh pi@192.168.0.104 'sudo journalctl -u lelamp-api.service -f'
ssh pi@192.168.0.104 'sudo journalctl -u lelamp-frontend.service -f'
```

### 自动启动控制

```bash
# 禁用所有服务开机自启
ssh pi@192.168.0.104 'sudo systemctl disable lelamp-{livekit,api,frontend}.service'

# 重新启用所有服务开机自启
ssh pi@192.168.0.104 'sudo systemctl enable lelamp-{livekit,api,frontend}.service'
```

### 单个服务管理

**服务状态查看**
```bash
# 查看所有服务状态
ssh pi@192.168.0.104 'sudo systemctl status lelamp-*.service'

# 查看特定服务状态
ssh pi@192.168.0.104 'sudo systemctl status lelamp-api.service'
ssh pi@192.168.0.104 'sudo systemctl status lelamp-frontend.service'
```

### 服务控制
```bash
# 重启服务
ssh pi@192.168.0.104 'sudo systemctl restart lelamp-api.service'
ssh pi@192.168.0.104 'sudo systemctl restart lelamp-frontend.service'

# 停止服务
ssh pi@192.168.0.104 'sudo systemctl stop lelamp-api.service'
ssh pi@192.168.0.104 'sudo systemctl stop lelamp-frontend.service'

# 启动服务
ssh pi@192.168.0.104 'sudo systemctl start lelamp-api.service'
ssh pi@192.168.0.104 'sudo systemctl start lelamp-frontend.service'
```

### 日志查看
```bash
# 实时查看日志
ssh pi@192.168.0.104 'sudo journalctl -u lelamp-api.service -f'
ssh pi@192.168.0.104 'sudo journalctl -u lelamp-frontend.service -f'

# 查看最近 50 行日志
ssh pi@192.168.0.104 'sudo journalctl -u lelamp-api.service -n 50'
```

### 自动启动控制
```bash
# 禁用开机自启
ssh pi@192.168.0.104 'sudo systemctl disable lelamp-api.service'
ssh pi@192.168.0.104 'sudo systemctl disable lelamp-frontend.service'

# 重新启用开机自启
ssh pi@192.168.0.104 'sudo systemctl enable lelamp-api.service'
ssh pi@192.168.0.104 'sudo systemctl enable lelamp-frontend.service'
```

## 🔧 高级配置

### 修改服务配置
服务配置文件位于：`/etc/systemd/system/lelamp-*.service`

```bash
# 编辑服务文件
ssh pi@192.168.0.104 'sudo nano /etc/systemd/system/lelamp-api.service'

# 修改后重新加载
ssh pi@192.168.0.104 'sudo systemctl daemon-reload'
ssh pi@192.168.0.104 'sudo systemctl restart lelamp-api.service'
```

### 环境变量配置
编辑 `/home/pi/lelamp_runtime/.env` 文件来配置环境变量。

```bash
ssh pi@192.168.0.104 'nano ~/lelamp_runtime/.env'
# 修改后重启服务
ssh pi@192.168.0.104 'sudo systemctl restart lelamp-api.service'
```

## 📊 故障排除

### 服务无法启动
```bash
# 查看详细状态
ssh pi@192.168.0.104 'sudo systemctl status lelamp-api.service -l'

# 查看启动日志
ssh pi@192.168.0.104 'sudo journalctl -u lelamp-api.service -n 100'

# 检查端口占用
ssh pi@192.168.0.104 'sudo netstat -tlnp | grep -E "(8000|5173)"'
```

### 前端无法访问
```bash
# 检查静态文件是否存在
ssh pi@192.168.0.104 'ls -la ~/lelamp_runtime/web/dist/'

# 重新部署前端
./scripts/deploy_to_pi.sh
```

### 依赖问题
```bash
# 在树莓派上重新安装依赖
ssh pi@192.168.0.104 'cd ~/lelamp_runtime && sudo uv sync --extra api --extra hardware'
```

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

## 🔧 高级配置

### 完整服务系统故障排除

```bash
# 检查所有服务状态
ssh pi@192.168.0.104 'sudo systemctl status lelamp-{livekit,api,frontend}.service'

# 检查端口占用
ssh pi@192.168.0.104 'sudo netstat -tlnp | grep -E "(8000|5173)"'

# 检查服务依赖关系
ssh pi@192.168.0.104 'systemctl list-dependencies lelamp-frontend.service'

# 重新设置完整服务系统
./scripts/setup_all_services.sh
```

### API 服务故障排除

```bash
# 检查 API 服务日志
ssh pi@192.168.0.104 'sudo journalctl -u lelamp-api.service -n 50 --no-pager'

# 测试 API 端点
ssh pi@192.168.0.104 'curl -s http://localhost:8000/health'

# 检查数据库文件
ssh pi@192.168.0.104 'ls -la ~/lelamp_runtime/lelamp.db'
```

### Frontend 服务故障排除

```bash
# 检查 Frontend 服务日志
ssh pi@192.168.0.104 'sudo journalctl -u lelamp-frontend.service -n 50 --no-pager'

# 检查前端构建文件
ssh pi@192.168.0.104 'ls -la ~/lelamp_runtime/web/dist/'

# 手动启动前端服务测试
ssh pi@192.168.0.104 'cd ~/lelamp_runtime/web && npm run dev'
```

### 服务依赖问题

```bash
# 查看服务启动顺序
ssh pi@192.168.0.104 'systemd-analyze critical-chain lelamp-frontend.service'

# 手动按顺序启动服务
ssh pi@192.168.0.104 'sudo systemctl start lelamp-livekit.service'
ssh pi@192.168.0.104 'sudo systemctl start lelamp-api.service'
ssh pi@192.168.0.104 'sudo systemctl start lelamp-frontend.service'
```

## 🎯 推荐工作流

### 日常开发
1. 本地开发测试
2. `./scripts/deploy_to_pi.sh` 部署到树莓派
3. 访问 http://192.168.0.104:5173 测试完整功能
4. 访问 http://192.168.0.104:8000/docs 查看 API 文档

### 快速更新
1. 修改代码后 `git commit`
2. `./scripts/push_to_pi.sh` 推送代码
3. 树莓派自动重启服务

### 生产环境
1. 使用简化模式（低资源占用）
2. 定期检查服务状态
3. 监控日志文件大小

## 📞 获取帮助

- **查看服务状态**：`./scripts/sync_status.sh`
- **API 测试**：`./scripts/test_api_on_pi.sh`
- **依赖检查**：`./scripts/ensure_dependencies.sh`

---

**提示**：首次设置建议使用简化模式，如需开发调试可切换到完整模式。