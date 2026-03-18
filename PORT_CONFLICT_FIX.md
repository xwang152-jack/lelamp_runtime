# 🔧 端口冲突问题解决方案

## ❌ 问题描述
```
ERROR: [Errno 98] error while attempting to bind on address ('0.0.0.0', 8000): address already in use
```

## 🔍 问题原因
端口8000已被现有服务占用，通常是因为：
1. 之前的API服务没有正确停止
2. 多个启动命令同时执行
3. 服务崩溃但端口仍被占用

## ✅ 快速解决方案

### 方案1: 使用重启脚本 (推荐)
```bash
# 在树莓派上执行
cd ~/lelamp_runtime
./restart_service.sh
```

### 方案2: 手动重启
```bash
# 1. 停止现有服务
pkill -f "uvicorn lelamp.api.app:app"

# 2. 等待进程停止
sleep 2

# 3. 启动新服务
cd ~/lelamp_runtime
.venv/bin/python -m uvicorn lelamp.api.app:app --host 0.0.0.0 --port 8000
```

### 方案3: 强制清理端口
```bash
# 1. 查找占用端口的进程
sudo lsof -i :8000

# 2. 强制终止进程
sudo kill -9 $(sudo lsof -t -i:8000)

# 3. 启动新服务
cd ~/lelamp_runtime
.venv/bin/python -m uvicorn lelamp.api.app:app --host 0.0.0.0 --port 8000
```

## 🚀 验证服务状态

### 检查服务是否启动成功
```bash
# 健康检查
curl http://localhost:8000/health

# 应该返回: {"status":"healthy","service":"lelamp-api","active_connections":{}}
```

### 查看服务日志
```bash
# 实时日志
tail -f /tmp/uvicorn.log

# 或API服务日志
tail -f /tmp/lelamp-api.log
```

### 查看进程状态
```bash
# 检查uvicorn进程
ps aux | grep uvicorn

# 检查端口占用
sudo netstat -tulpn | grep :8000
```

## 🛡️ 预防措施

### 使用systemd服务管理 (推荐)
创建系统服务避免端口冲突：

```bash
# 创建服务文件
sudo nano /etc/systemd/system/lelamp-api.service
```

服务配置：
```ini
[Unit]
Description=LeLamp API Server
After=network.target

[Service]
Type=notify
User=pi
WorkingDirectory=/home/pi/lelamp_runtime
Environment="PATH=/home/pi/lelamp_runtime/.venv/bin"
ExecStart=/home/pi/lelamp_runtime/.venv/bin/python -m uvicorn lelamp.api.app:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

启动服务：
```bash
# 重载配置
sudo systemctl daemon-reload

# 启用服务
sudo systemctl enable lelamp-api.service

# 启动服务
sudo systemctl start lelamp-api.service

# 查看状态
sudo systemctl status lelamp-api.service
```

### 管理服务
```bash
# 启动服务
sudo systemctl start lelamp-api

# 停止服务
sudo systemctl stop lelamp-api

# 重启服务
sudo systemctl restart lelamp-api

# 查看日志
sudo journalctl -u lelamp-api.service -f
```

## 🔧 故障排除

### 检查端口占用详情
```bash
# 详细端口信息
sudo netstat -tulpn | grep :8000

# 或使用ss命令
sudo ss -tulpn | grep :8000
```

### 查找所有Python进程
```bash
# 所有LeLamp相关进程
ps aux | grep -E "(lelamp|uvicorn)" | grep -v grep
```

### 清理僵尸进程
```bash
# 清理所有相关进程
pkill -9 -f lelamp
pkill -9 -f uvicorn

# 等待清理完成
sleep 2

# 重新启动服务
cd ~/lelamp_runtime
./restart_service.sh
```

## 📝 日常操作建议

### 正确的启动流程
1. 检查现有服务：`ps aux | grep uvicorn`
2. 如有运行，先停止：`pkill -f uvicorn`
3. 等待端口释放：`sleep 2`
4. 启动新服务：`./restart_service.sh`

### 推荐的启动方式
- **开发测试**: 直接运行uvicorn
- **生产环境**: 使用systemd服务
- **后台运行**: 使用restart_service.sh

## 🎯 当前状态确认

如果您看到了端口冲突错误，当前系统状态：

✅ **实际上有一个服务正在运行**
- 进程ID: 2086
- 端口: 8000
- 状态: 正常响应

📍 **API服务地址**
- 本地: `http://localhost:8000`
- 网络: `http://192.168.0.104:8000`

🧪 **验证功能**
```bash
# 测试API
curl http://192.168.0.104:8000/health

# 测试WebSocket
python3 -c "
import websockets, asyncio, json
async def test():
    async with websockets.connect('ws://192.168.0.104:8000/api/ws/lelamp') as ws:
        print(await ws.recv())
asyncio.run(test())
"
```

## 💡 总结

端口8000冲突通常不是严重问题，因为：
- ✅ 服务实际上正在运行
- ✅ 可以正常访问和使用
- ✅ 只需要知道正确的重启方法

**建议**: 使用systemd服务管理，避免手动端口冲突。

---

**最后更新**: 2025-03-18 23:45
**问题状态**: ✅ 已解决
**服务状态**: 🟢 正常运行
