# 🚀 LeLamp Runtime 部署指南

## 📋 目录
- [系统要求](#系统要求)
- [快速部署](#快速部署)
- [详细配置](#详细配置)
- [数据库设置](#数据库设置)
- [服务启动](#服务启动)
- [故障排除](#故障排除)
- [生产部署](#生产部署)

---

## 系统要求

### 硬件要求
- **设备**: Raspberry Pi 4B (推荐) 或 Raspberry Pi 5
- **内存**: 最少 2GB RAM，推荐 4GB+
- **存储**: 最少 16GB SD卡
- **电源**: 5V/3A USB-C 电源适配器

### 软件要求
- **操作系统**: Raspberry Pi OS Bookworm (64位)
- **Python**: 3.12+ (系统自带或手动安装)
- **网络**: WiFi 或以太网连接

### 外设要求
- **LeLamp 机器人台灯**: 包含5轴舵机、64颗LED、摄像头模块
- **串口连接**: USB转串口模块 (/dev/ttyACM0)

---

## 快速部署

### 1. 系统更新
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-dev python3-venv python3-pip git
```

### 2. 克隆项目
```bash
cd ~
git clone https://github.com/humancomputerlab/LeLamp.git
cd LeLamp
```

### 3. 安装依赖
```bash
# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装项目依赖
pip install --upgrade pip
pip install -e .

# 如果包含硬件依赖
pip install -e ".[hardware]"
```

### 4. 配置环境
```bash
# 复制配置模板
cp .env.example .env

# 编辑配置文件
nano .env
```

### 5. 最小配置
```bash
# LiveKit (必需)
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secretsecret

# DeepSeek (必需)
DEEPSEEK_API_KEY=sk-your-deepseek-key

# Baidu Speech (必需)
BAIDU_SPEECH_API_KEY=your-baidu-api-key
BAIDU_SPEECH_SECRET_KEY=your-baidu-secret
```

### 6. 初始化数据库
```bash
python3 << 'EOF'
from lelamp.database.session import engine
from lelamp.database.models import Base
Base.metadata.create_all(bind=engine)
print('✅ 数据库初始化完成')
EOF
```

### 7. 启动服务
```bash
# 启动 API 服务器
.venv/bin/python -m uvicorn lelamp.api.app:app --host 0.0.0.0 --port 8000
```

### 8. 验证部署
```bash
# 测试 API
curl http://localhost:8000/health

# 应该返回: {"status":"healthy","service":"lelamp-api","active_connections":{}}
```

---

## 详细配置

### 环境变量说明

#### 核心服务配置
```bash
# 设备标识
LELAMP_ID=lelamp                # 设备唯一ID
LELAMP_PORT=/dev/ttyACM0        # 串口设备路径

# LiveKit 实时通信
LIVEKIT_URL=wss://xxx.livekit.cloud
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secretsecret
```

#### AI 服务配置
```bash
# DeepSeek LLM
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com

# ModelScope 视觉 (可选)
MODELSCOPE_API_KEY=ms-xxx
MODELSCOPE_MODEL=Qwen/Qwen3-VL-235B-A22B-Instruct
MODELSCOPE_BASE_URL=https://api-inference.modelscope.cn/v1
MODELSCOPE_TIMEOUT_S=60.0

# Baidu 语音
BAIDU_SPEECH_API_KEY=xxx
BAIDU_SPEECH_SECRET_KEY=xxx
BAIDU_SPEECH_CUID=lelamp
```

#### 硬件配置
```bash
# LED 矩阵
LELAMP_LED_BRIGHTNESS=25
LELAMP_MATRIX_W=8
LELAMP_MATRIX_H=8

# 摄像头
LELAMP_CAMERA_INDEX_OR_PATH=0
LELAMP_CAMERA_WIDTH=1024
LELAMP_CAMERA_HEIGHT=768
LELAMP_CAMERA_ROTATE_DEG=0
```

### 硬件校准

#### 1. 查找串口设备
```bash
# 查找连接的舵机控制器
ls /dev/ttyACM*

# 或使用工具
uv run lerobot-find-port
```

#### 2. 设置电机 ID
```bash
uv run -m lelamp.setup_motors --id lelamp --port /dev/ttyACM0
```

#### 3. 校准电机
```bash
sudo uv run -m lelamp.calibrate --id lelamp --port /dev/ttyACM0
```

---

## 数据库设置

### 数据库类型
项目支持两种数据库：
- **SQLite** (默认): 无需额外安装，适合单机部署
- **PostgreSQL** (可选): 适合多设备或高并发场景

### SQLite 初始化
```bash
# 方式1: 使用 Python 脚本
python3 << 'EOF'
from lelamp.database.session import engine
from lelamp.database.models import Base
Base.metadata.create_all(bind=engine)
print('✅ 数据库初始化完成')
EOF

# 方式2: 直接创建表
sqlite3 lelamp.db << 'EOF'
CREATE TABLE IF NOT EXISTS user_settings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lamp_id VARCHAR(100) UNIQUE NOT NULL,
    theme VARCHAR(20) DEFAULT 'light',
    deepseek_model VARCHAR(100) DEFAULT 'deepseek-chat',
    -- ... 其他字段
);
EOF
```

### PostgreSQL 配置 (可选)
```bash
# 安装 PostgreSQL
sudo apt install -y postgresql postgresql-contrib

# 创建数据库
sudo -u postgres psql
CREATE DATABASE lelamp_runtime;
CREATE USER lelamp WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE lelamp_runtime TO lelamp;
\q

# 更新 .env 文件
DATABASE_URL=postgresql://lelamp:your_password@localhost/lelamp_runtime
```

---

## 服务启动

### 开发模式启动

#### 启动 API 服务器
```bash
# 使用 uv (推荐)
uv run uvicorn lelamp.api.app:app --host 0.0.0.0 --port 8000 --reload

# 或直接使用虚拟环境
.venv/bin/python -m uvicorn lelamp.api.app:app --host 0.0.0.0 --port 8000 --reload
```

#### 启动前端开发服务器
```bash
cd web
npm run dev
```

### 生产模式启动

#### 使用 systemd 服务
```bash
# 创建服务文件
sudo nano /etc/systemd/system/lelamp-api.service
```

**服务配置**:
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

#### 启用和启动服务
```bash
# 重载 systemd 配置
sudo systemctl daemon-reload

# 启用开机自启
sudo systemctl enable lelamp-api.service

# 启动服务
sudo systemctl start lelamp-api.service

# 查看状态
sudo systemctl status lelamp-api.service

# 查看日志
sudo journalctl -u lelamp-api.service -f
```

#### 后台运行 (无 systemd)
```bash
# 使用 nohup 后台运行
nohup .venv/bin/python -m uvicorn lelamp.api.app:app --host 0.0.0.0 --port 8000 > /tmp/lelamp-api.log 2>&1 &

# 查看进程
ps aux | grep uvicorn

# 停止服务
pkill -f uvicorn
```

---

## 故障排除

### 常见问题

#### 1. API Key 未加载
**症状**: 提示"请在配置中设置 DEEPSEEK_API_KEY"
**原因**: .env 文件未正确加载
**解决**:
```bash
# 检查 .env 文件是否存在
cat .env | grep DEEPSEEK_API_KEY

# 检查文件权限
chmod 600 .env

# 重启服务
sudo systemctl restart lelamp-api.service
```

#### 2. 数据库表不存在
**症状**: 设置界面无法工作，返回500错误
**原因**: 数据库未初始化
**解决**:
```bash
# 重新初始化数据库
python3 << 'EOF'
from lelamp.database.session import engine
from lelamp.database.models import Base
Base.metadata.create_all(bind=engine)
EOF
```

#### 3. 串口连接失败
**症状**: 电机无法控制，提示"Could not connect on port"
**原因**: 串口设备权限或路径错误
**解决**:
```bash
# 检查串口设备
ls -la /dev/ttyACM*

# 添加用户到 dialout 组
sudo usermod -a -G dialout pi

# 或使用 sudo 运行
sudo python -m uvicorn lelamp.api.app:app
```

#### 4. 虚拟环境问题
**症状**: 模块导入错误，Python版本不兼容
**解决**:
```bash
# 删除旧环境
rm -rf .venv

# 创建新环境
python3 -m venv .venv

# 重新安装依赖
.venv/bin/pip install -e .
```

### 日志查看

#### API 服务日志
```bash
# systemd 服务日志
sudo journalctl -u lelamp-api.service -n 50 -f

# 后台进程日志
tail -f /tmp/lelamp-api.log

# 实时日志
sudo journalctl -f _PID=$(pgrep -f "uvicorn lelamp.api.app:app")
```

#### 应用日志
```bash
# 查看应用级别日志
LOG_LEVEL=DEBUG .venv/bin/python -m uvicorn lelamp.api.app:app
```

---

## 生产部署

### 性能优化

#### 1. 多进程部署
```bash
# 启动多worker进程
.venv/bin/python -m uvicorn lelamp.api.app:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker
```

#### 2. 反向代理配置 (Nginx)
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /ws/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

#### 3. 自动启动配置
```bash
# 添加到 crontab
crontab -e

# 添加开机自启
@reboot cd /home/pi/lelamp_runtime && .venv/bin/python -m uvicorn lelamp.api.app:app --host 0.0.0.0 --port 8000 > /tmp/lelamp.log 2>&1 &
```

### 安全加固

#### 1. 文件权限
```bash
# 设置合适的文件权限
chmod 600 .env
chmod 700 .venv
chown -R pi:pi /home/pi/lelamp_runtime
```

#### 2. 防火墙配置
```bash
# 仅允许必要端口
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw allow 8000/tcp  # API 服务
sudo ufw enable
```

#### 3. 定期备份
```bash
# 备份脚本
cat > /home/pi/backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/home/pi/backups/$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"

# 备份数据库
cp lelamp.db "$BACKUP_DIR/"

# 备份配置
cp .env "$BACKUP_DIR/"

# 备份代码
tar -czf "$BACKUP_DIR/code.tar.gz" lelamp/ web/
EOF

chmod +x /home/pi/backup.sh

# 添加到定时任务 (每天凌晨3点)
crontab -e
# 添加: 0 3 * * * /home/pi/backup.sh
```

---

## 更新升级

### 代码更新
```bash
# 拉取最新代码
git pull origin main

# 更新依赖
.venv/bin/pip install -e .

# 重启服务
sudo systemctl restart lelamp-api.service
```

### 数据库迁移
```bash
# 备份数据库
cp lelamp.db lelamp.db.backup

# 执行迁移
python3 << 'EOF'
from lelamp.database.session import engine
from lelamp.database.models import Base
Base.metadata.create_all(bind=engine)  # 自动处理表结构更新
EOF
```

---

## 监控维护

### 健康检查
```bash
# API 健康检查
curl http://localhost:8000/health

# WebSocket 连接测试
python3 << 'EOF'
import asyncio
import websockets
async def test():
    async with websockets.connect('ws://localhost:8000/api/ws/lelamp') as ws:
        print(await ws.recv())
asyncio.run(test())
EOF
```

### 性能监控
```bash
# CPU 和内存使用
htop

# 磁盘使用
df -h

# 网络连接
netstat -tuln | grep 8000
```

---

## 📞 获取帮助

### 文档资源
- [项目 README](README.md)
- [开发指南](CLAUDE.md)
- [快速参考](QUICK_REFERENCE.md)
- [设置界面指南](SETTINGS_QUICK_REFERENCE.md)

### 问题报告
- [GitHub Issues](https://github.com/humancomputerlab/LeLamp/issues)
- [技术支持](mailto:support@humancomputerlab.com)

---

**部署指南版本**: v2.0
**最后更新**: 2025-03-18
**维护者**: Human Computer Lab
