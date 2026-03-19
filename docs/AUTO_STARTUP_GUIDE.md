# LeLamp 自动启动配置指南

## 🚀 快速开始

### 方法一：使用部署脚本（推荐）

```bash
# 完整部署：构建前端 → 推送代码 → 配置自动启动
./scripts/deploy_to_pi.sh
```

### 方法二：分步配置

1. **构建并推送代码**
   ```bash
   ./scripts/push_to_pi.sh
   ```

2. **配置自动启动**
   ```bash
   ./scripts/setup_lelamp_startup.sh
   ```

## 📋 启动模式说明

### 1. 简化模式（推荐）⭐
- **适用场景**：日常使用、资源受限环境
- **特点**：
  - 仅启动后端 API 服务
  - 前端静态文件由 API 服务
  - 单一端口（8000）访问前后端
  - 占用资源少，启动快速

- **访问地址**：http://192.168.0.104:8000

### 2. 完整模式
- **适用场景**：开发调试、需要实时预览
- **特点**：
  - 后端 API + 前端开发服务器
  - 支持前端热更新
  - 占用资源较多，启动较慢

- **访问地址**：
  - 前端：http://192.168.0.104:5173
  - API：http://192.168.0.104:8000

### 3. 主程序模式
- **适用场景**：无头使用、纯语音交互
- **特点**：
  - 仅启动 main.py console
  - 纯命令行交互
  - 最低资源占用

- **访问方式**：SSH 到树莓派直接交互

## 🛠️ 管理命令

### 服务状态查看
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

## 🎯 推荐工作流

### 日常开发
1. 本地开发测试
2. `./scripts/deploy_to_pi.sh` 部署到树莓派
3. 访问 http://192.168.0.104:8000 测试

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