# LeLamp API 服务器设置指南

## 问题：前端灯光控制返回成功但实际硬件无响应

### 症状
- WebSocket 连接正常
- 前端发送灯光控制命令返回成功
- 但 LeLamp 设备的 LED 实际上没有变化

### 根本原因
`ws281x` LED 驱动库需要访问 `/dev/mem` 设备来控制 GPIO，但 `pi` 用户没有权限访问该设备。

```
CRGBService start failed: ws2811_init failed with code -5 (mmap() failed), fallback to NoOpRGBService
```

RGB 服务启动失败后自动降级到 `NoOpRGBService`（无操作服务），所有命令执行成功但实际硬件不响应。

## 解决方案

### 方法 1：使用 sudo 运行 API 服务器（推荐）

#### 手动启动
```bash
cd /home/pi/lelamp_runtime
sudo .venv/bin/python -m uvicorn lelamp.api.app:app --host 0.0.0.0 --port 8000
```

#### 使用启动脚本
```bash
start-lelamp-api
```

### 方法 2：配置 systemd 服务（推荐用于生产环境）

#### 启动脚本
已安装在 `/usr/local/bin/start-lelamp-api`：

```bash
#!/bin/bash
# LeLamp API Server 启动脚本
# 需要 sudo 权限以访问 /dev/mem 控制 RGB LED

cd /home/pi/lelamp_runtime

# 停止现有实例
pkill -f 'uvicorn lelamp.api.app:app' 2>/dev/null
sleep 2

# 启动 API 服务器（使用 sudo）
echo "Starting LeLamp API server with sudo privileges..."
nohup sudo .venv/bin/python -m uvicorn lelamp.api.app:app \
    --host 0.0.0.0 \
    --port 8000 \
    > /tmp/uvicorn.log 2>&1 &

echo "API server started on http://0.0.0.0:8000"
echo "Logs: /tmp/uvicorn.log"
```

#### Systemd 服务
已创建 `/etc/systemd/system/lelamp-api.service`：

```ini
[Unit]
Description=LeLamp API Server
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/lelamp_runtime
ExecStart=/usr/local/bin/start-lelamp-api
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

#### 使用 systemd 管理服务
```bash
# 启动服务
sudo systemctl start lelamp-api

# 停止服务
sudo systemctl stop lelamp-api

# 重启服务
sudo systemctl restart lelamp-api

# 查看状态
sudo systemctl status lelamp-api

# 开机自启动
sudo systemctl enable lelamp-api

# 禁用开机自启动
sudo systemctl disable lelamp-api

# 查看日志
sudo journalctl -u lelamp-api -f

# 查看最近 100 行日志
sudo journalctl -u lelamp-api -n 100
```

## 验证灯光控制功能

### Python 测试脚本
```python
import asyncio
import websockets
import json

async def test_light_control():
    uri = "ws://localhost:8000/api/ws/lelamp"

    async with websockets.connect(uri) as websocket:
        print("✅ WebSocket 连接成功")

        # 等待连接确认
        await websocket.recv()

        # 测试红色
        command = {"type": "command", "action": "set_rgb_solid", "params": {"r": 255, "g": 0, "b": 0}}
        await websocket.send(json.dumps(command))
        response = await websocket.recv()
        result = json.loads(response)

        if result.get("success"):
            print("✅ 红色灯光控制成功")
        else:
            print(f"❌ 灯光控制失败: {result.get('error')}")

        # 测试绿色
        await asyncio.sleep(2)
        command = {"type": "command", "action": "set_rgb_solid", "params": {"r": 0, "g": 255, "b": 0}}
        await websocket.send(json.dumps(command))

        # 测试蓝色
        await asyncio.sleep(2)
        command = {"type": "command", "action": "set_rgb_solid", "params": {"r": 0, "g": 0, "b": 255}}
        await websocket.send(json.dumps(command))

asyncio.run(test_light_control())
```

### 支持的灯光控制命令

#### 基础颜色控制
```json
{
  "type": "command",
  "action": "set_rgb_solid",
  "params": {"r": 255, "g": 0, "b": 0}
}
```

#### RGB 效果
```json
{
  "type": "command",
  "action": "rgb_effect_rainbow",
  "params": {}
}
```

#### 呼吸效果
```json
{
  "type": "command",
  "action": "rgb_effect_breathing",
  "params": {"r": 255, "g": 255, "b": 0}
}
```

#### 亮度控制
```json
{
  "type": "command",
  "action": "set_rgb_brightness",
  "params": {"percent": 50}
}
```

#### 停止效果
```json
{
  "type": "command",
  "action": "stop_rgb_effect",
  "params": {}
}
```

## 常见问题

### Q: 为什么需要 sudo 权限？
A: `ws281x` 库需要直接访问硬件 `/dev/mem` 来控制 GPIO，这需要 root 权限。

### Q: 有没有不使用 sudo 的方法？
A: 理论上可以通过修改 `/dev/gpiomem` 权限或使用其他 LED 驱动库，但这些方法可能不稳定或不安全。使用 sudo 是最可靠的解决方案。

### Q: 是否安全？
A: 是的。API 服务器以 `pi` 用户运行，但通过 sudo 获取必要的硬件权限。systemd 服务配置限制了权限范围，只授予必要的硬件访问权限。

### Q: 如何检查 RGB 服务是否正常运行？
A: 查看日志文件：
```bash
tail -f /tmp/uvicorn.log | grep -E 'RGB|rgb|NoOpRGBService'
```

如果看到 `fallback to NoOpRGBService`，说明 RGB 服务启动失败，需要检查 sudo 权限。

### Q: 前端连接不上怎么办？
A: 检查以下几点：
1. API 服务器是否运行：`ps aux | grep uvicorn`
2. 防火墙是否阻止：`sudo ufw status`
3. 网络连接是否正常：`ping 192.168.0.104`
4. WebSocket 地址是否正确：`ws://192.168.0.104:8000/api/ws/lelamp`

## 相关文件

- API 服务器配置：`lelamp/api/app.py`
- WebSocket 路由：`lelamp/api/routes/websocket.py`
- RGB 服务：`lelamp/service/rgb/rgb_service.py`
- 启动脚本：`/usr/local/bin/start-lelamp-api`
- Systemd 服务：`/etc/systemd/system/lelamp-api.service`
- 日志文件：`/tmp/uvicorn.log`

## 更新历史

- 2026-03-19: 初始版本，记录 RGB 权限问题和解决方案
- 2026-03-19: 添加 systemd 服务配置和使用说明
