# LeLamp API 部署指南

本文档提供 LeLamp API 的部署指南，包括开发环境、生产环境和 Docker 部署。

---

## 目录

- [开发环境设置](#开发环境设置)
- [生产环境部署](#生产环境部署)
- [Docker 部署](#docker-部署)
- [系统配置](#系统配置)
- [监控和日志](#监控和日志)
- [故障排除](#故障排除)

---

## 开发环境设置

### 前置要求

- Python 3.12+
- UV 包管理器
- SQLite (默认) 或 PostgreSQL

### 安装步骤

#### 1. 克隆代码库

```bash
git clone https://github.com/your-org/lelamp_runtime.git
cd lelamp_runtime
```

#### 2. 安装依赖

```bash
# 安装 API 相关依赖
uv sync --extra api

# 或安装所有依赖
uv sync --all-extras
```

#### 3. 配置环境变量

创建 `.env` 文件：

```bash
cp .env.example .env
```

编辑 `.env` 文件，配置以下变量：

```bash
# 数据库配置
LELAMP_DATABASE_URL=sqlite:///./lelamp.db

# API 配置
LELAMP_API_HOST=127.0.0.1
LELAMP_API_PORT=8000
LELAMP_API_RELOAD=true

# JWT 认证密钥（生产环境必须设置强随机密钥）
LELAMP_JWT_SECRET=your-strong-random-secret-key

# 设备绑定密钥（可选，首次 WiFi 设置时自动生成）
LELAMP_DEVICE_SECRET=

# Web 前端构建输出路径（默认: web/dist）
LELAMP_WEB_DIST=web/dist

# 日志配置
LOG_LEVEL=INFO
```

#### 4. 初始化数据库

```bash
# 创建数据库表
uv run -m lelamp.database.init_db
```

#### 5. 运行开发服务器

```bash
# 使用 UV 运行
uv run uvicorn lelamp.api.app:app --reload --host 127.0.0.1 --port 8000

# 或使用 Python
uv run python -m uvicorn lelamp.api.app:app --reload --host 127.0.0.1 --port 8000
```

#### 6. 验证安装

访问以下 URL 验证 API 运行：

- API 文档: http://127.0.0.1:8000/docs
- 健康检查: http://127.0.0.1:8000/health
- 设备列表: http://127.0.0.1:8000/api/devices
- Web 前端: http://127.0.0.1:8000/ （构建后可用）
- mDNS 自动发现: http://<device_id>.local:8000 （局域网内）

---

## 生产环境部署

### 系统要求

- **操作系统**: Linux (Ubuntu 20.04+ 推荐)
- **Python**: 3.12+
- **内存**: 最小 512MB，推荐 1GB+
- **CPU**: 最小 1 核，推荐 2 核+
- **存储**: 最小 1GB，推荐 10GB+

### 安装步骤

#### 1. 系统准备

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装系统依赖
sudo apt install -y python3.12 python3.12-venv python3-pip nginx postgresql postgresql-contrib

# 安装 UV 包管理器
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 2. 创建应用用户

```bash
# 创建专用用户
sudo useradd -r -s /bin/false lelamp
sudo mkdir -p /opt/lelamp
sudo chown lelamp:lelamp /opt/lelamp
```

#### 3. 部署应用

```bash
# 复制代码到服务器
sudo rsync -avz /path/to/lelamp_runtime/ lelamp@server:/opt/lelamp/

# 设置权限
sudo chown -R lelamp:lelamp /opt/lelamp
```

#### 4. 配置 PostgreSQL (可选)

如果使用 PostgreSQL：

```bash
# 创建数据库和用户
sudo -u postgres psql <<EOF
CREATE DATABASE lelamp;
CREATE USER lelamp WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE lelamp TO lelamp;
EOF
```

更新 `.env` 文件：

```bash
LELAMP_DATABASE_URL=postgresql://lelamp:your_secure_password@localhost/lelamp
```

#### 5. 安装依赖

```bash
cd /opt/lelamp
sudo -u lelamp uv sync --extra api
```

#### 6. 初始化数据库

```bash
sudo -u lelamp uv run -m lelamp.database.init_db
```

#### 7. 创建 Systemd 服务

创建 `/etc/systemd/system/lelamp-api.service`：

```ini
[Unit]
Description=LeLamp API Server
After=network.target postgresql.service

[Service]
Type=notify
User=lelamp
Group=lelamp
WorkingDirectory=/opt/lelamp
Environment="PATH=/opt/lelamp/.venv/bin"
EnvironmentFile=/opt/lelamp/.env
ExecStart=/opt/lelamp/.venv/bin/uvicorn lelamp.api.app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --log-config /opt/lelamp/lelamp/api/logging.conf

# Restart policy
Restart=always
RestartSec=10

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/lelamp

[Install]
WantedBy=multi-user.target
```

启动服务：

```bash
# 重载 systemd
sudo systemctl daemon-reload

# 启动服务
sudo systemctl start lelamp-api

# 设置开机自启
sudo systemctl enable lelamp-api

# 检查状态
sudo systemctl status lelamp-api
```

#### 8. 配置 Nginx 反向代理

创建 `/etc/nginx/sites-available/lelamp-api`：

```nginx
upstream lelamp_api {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name api.lelamp.example.com;

    # 日志
    access_log /var/log/nginx/lelamp-api-access.log;
    error_log /var/log/nginx/lelamp-api-error.log;

    # 请求大小限制
    client_max_body_size 10M;

    # WebSocket 支持
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";

    # 代理头
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;

    # 超时设置
    proxy_connect_timeout 60s;
    proxy_send_timeout 60s;
    proxy_read_timeout 60s;

    # API 路由
    location /api/ {
        proxy_pass http://lelamp_api;
    }

    # WebSocket 路由
    location /api/ws/ {
        proxy_pass http://lelamp_api;
        proxy_buffering off;
    }

    # 健康检查
    location /health {
        proxy_pass http://lelamp_api;
    }
}
```

启用站点：

```bash
# 创建符号链接
sudo ln -s /etc/nginx/sites-available/lelamp-api /etc/nginx/sites-enabled/

# 测试配置
sudo nginx -t

# 重启 Nginx
sudo systemctl restart nginx
```

#### 9. 配置 SSL/TLS (推荐)

使用 Let's Encrypt 免费证书：

```bash
# 安装 Certbot
sudo apt install certbot python3-certbot-nginx -y

# 获取证书
sudo certbot --nginx -d api.lelamp.example.com

# 自动续期
sudo certbot renew --dry-run
```

Nginx 配置会自动更新为 HTTPS。

#### 10. 配置防火墙

```bash
# 允许 HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# 启用防火墙
sudo ufw enable
```

---

## Docker 部署

### 创建 Dockerfile

`Dockerfile`:

```dockerfile
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# 安装 UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# 复制依赖文件
COPY pyproject.toml uv.lock ./

# 安装依赖
RUN uv sync --frozen --extra api

# 复制应用代码
COPY . .

# 创建非 root 用户
RUN useradd -m -u 1000 lelamp && chown -R lelamp:lelamp /app
USER lelamp

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uv", "run", "uvicorn", "lelamp.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 创建 docker-compose.yml

`docker-compose.yml`:

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - LELAMP_DATABASE_URL=postgresql://lelamp:password@db:5432/lelamp
      - LOG_LEVEL=INFO
    depends_on:
      - db
    volumes:
      - ./data:/app/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=lelamp
      - POSTGRES_USER=lelamp
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - api
    restart: unless-stopped

volumes:
  postgres_data:
```

### 构建和运行

```bash
# 构建镜像
docker-compose build

# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f api

# 停止服务
docker-compose down

# 重启服务
docker-compose restart api
```

---

## 系统配置

### 环境变量

#### 必需变量

```bash
# 数据库
LELAMP_DATABASE_URL=postgresql://user:password@localhost/dbname

# API
LELAMP_API_HOST=0.0.0.0
LELAMP_API_PORT=8000
```

#### 可选变量

```bash
# 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL=INFO

# 工作进程数（生产环境建议 CPU 核心数 * 2 + 1）
LELAMP_API_WORKERS=4

# 数据库连接池
LELAMP_DB_POOL_SIZE=20
LELAMP_DB_MAX_OVERFLOW=10

# WebSocket 配置
LELAMP_WS_HEARTBEAT_INTERVAL=30
LELAMP_WS_CONNECTION_TIMEOUT=60

# JWT 签名密钥（生产环境必须设置，否则启动时使用随机密钥并输出警告）
LELAMP_JWT_SECRET=your-strong-random-secret-key

# 设备绑定密钥（首次 WiFi 设置时自动生成 16 位 hex 字符串，存储在 /var/lib/lelamp/setup_status.json）
LELAMP_DEVICE_SECRET=

# Vue 前端构建产物路径（默认: web/dist）
LELAMP_WEB_DIST=web/dist
```

### 性能优化

#### 1. Uvicorn 配置

```bash
uvicorn lelamp.api.app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --max-requests 1000 \
    --max-requests-jitter 100 \
    --preload \
    --access-log \
    --log-level info
```

**参数说明**:
- `--workers`: 工作进程数
- `--worker-class`: 使用 UvicornWorker
- `--max-requests`: 最大请求数后重启 worker（防止内存泄漏）
- `--max-requests-jitter`: 随机抖动，避免所有 worker 同时重启
- `--preload`: 预加载应用（减少内存占用）
- `--access-log`: 启用访问日志

#### 2. 数据库优化

**PostgreSQL 配置** (`postgresql.conf`):

```ini
# 连接设置
max_connections = 100

# 内存设置
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 16MB

# 查询优化
random_page_cost = 1.1

# 日志设置
logging_collector = on
log_directory = 'pg_log'
log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log'
log_statement = 'mod'
```

#### 3. Nginx 优化

```nginx
# 启用 gzip 压缩
gzip on;
gzip_vary on;
gzip_min_length 1024;
gzip_types text/plain application/json application/javascript;

# 启用缓存
proxy_cache_path /var/cache/nginx levels=1:2 keys_zone=api_cache:10m max_size=100m inactive=60m;

location /api/ {
    proxy_cache api_cache;
    proxy_cache_valid 200 5m;
    proxy_cache_use_stale error timeout updating;
}
```

---

## mDNS 自动发现

API 服务启动时会自动注册 mDNS 服务，局域网内设备可通过以下地址访问：

```
http://<device_id>.local:8000
```

- **依赖**：`zeroconf`（已包含在 `api` extra 中）
- **降级**：如果 `zeroconf` 不可用，mDNS 注册会静默失败，不影响 API 正常运行
- **适用场景**：无需知道设备 IP 地址，适用于局域网内便捷访问

---

## Vue 前端托管

API 服务内置前端静态文件托管，无需额外部署 Nginx 或其他静态服务器。

### 构建前端

```bash
# 使用构建脚本
bash scripts/build_web.sh
```

构建产物默认输出到 `web/dist` 目录，可通过 `LELAMP_WEB_DIST` 环境变量自定义路径。

### 单端口部署

构建完成后，FastAPI 服务在端口 8000 上同时提供：
- API 服务：`http://host:8000/api/...`
- WebSocket：`ws://host:8000/api/ws/...`
- 前端页面：`http://host:8000/`

前端路由使用 SPA fallback，未匹配的路径会返回 `index.html`。

---

## 监控和日志

### 日志配置

创建 `lelamp/api/logging.conf`:

```ini
[loggers]
keys=root,uvicorn,lelamp

[handlers]
keys=console,file

[formatters]
keys=default,detailed

[logger_root]
level=INFO
handlers=console,file

[logger_uvicorn]
level=INFO
handlers=console,file
propagate=0
qualname=uvicorn

[logger_lelamp]
level=INFO
handlers=console,file
propagate=0
qualname=lelamp

[handler_console]
class=StreamHandler
formatter=default
args=(sys.stdout,)

[handler_file]
class=handlers.RotatingFileHandler
formatter=detailed
args=('/var/log/lelamp/api.log', 'a', 10485760, 5)

[formatter_default]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
datefmt=%Y-%m-%d %H:%M:%S

[formatter_detailed]
format=%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s
datefmt=%Y-%m-%d %H:%M:%S
```

### 日志轮转

创建 `/etc/logrotate.d/lelamp`:

```
/var/log/lelamp/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 0640 lelamp lelamp
    sharedscripts
    postrotate
        systemctl reload lelamp-api > /dev/null 2>&1 || true
    endscript
}
```

### 监控设置

#### 1. Prometheus + Grafana

安装 Prometheus：

```bash
sudo apt install prometheus -y
```

配置 Prometheus (`/etc/prometheus/prometheus.yml`):

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'lelamp-api'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

#### 2. 健康检查脚本

创建 `/usr/local/bin/lelamp-healthcheck.sh`:

```bash
#!/bin/bash

HEALTH_URL="http://localhost:8000/health"
TIMEOUT=10
MAX_RETRIES=3

for i in $(seq 1 $MAX_RETRIES); do
    if curl -f -s --max-time $TIMEOUT $HEALTH_URL > /dev/null; then
        echo "Health check passed"
        exit 0
    fi
    sleep 5
done

echo "Health check failed"
systemctl restart lelamp-api
exit 1
```

添加到 crontab：

```bash
# 每 5 分钟检查一次
*/5 * * * * /usr/local/bin/lelamp-healthcheck.sh
```

---

## 故障排除

### 常见问题

#### 1. 服务无法启动

**检查日志**:
```bash
sudo journalctl -u lelamp-api -n 50
```

**常见原因**:
- 端口被占用: `sudo lsof -i :8000`
- 数据库连接失败: 检查 `LELAMP_DATABASE_URL`
- 权限问题: 确保文件所有者正确

#### 2. 数据库连接失败

**PostgreSQL**:
```bash
# 检查服务状态
sudo systemctl status postgresql

# 测试连接
psql -U lelamp -d lelamp -h localhost

# 检查连接数
sudo -u postgres psql -c "SELECT count(*) FROM pg_stat_activity;"
```

**SQLite**:
```bash
# 检查文件权限
ls -la /opt/lelamp/lelamp.db

# 检查文件完整性
sqlite3 /opt/lelamp/lelamp.db "PRAGMA integrity_check;"
```

#### 3. WebSocket 连接失败

**检查 Nginx 配置**:
```bash
# 验证配置
sudo nginx -t

# 检查代理设置
curl -i -N \
  -H "Connection: Upgrade" \
  -H "Upgrade: websocket" \
  http://localhost/api/ws/test_lamp
```

#### 4. 性能问题

**检查资源使用**:
```bash
# CPU 和内存
top -p $(pgrep -f lelamp-api)

# 数据库性能
sudo -u postgres psql -c "SELECT * FROM pg_stat_statements ORDER BY total_time DESC LIMIT 10;"

# 慢查询日志
sudo tail -f /var/log/postgresql/postgresql-slow.log
```

**优化建议**:
- 增加 worker 数量
- 启用数据库连接池
- 使用 Redis 缓存
- 优化数据库查询

#### 5. 内存泄漏

**监控内存使用**:
```bash
# 监控进程内存
watch -n 5 'ps aux | grep lelamp-api'

# 检查内存泄漏
valgrind --leak-check=full --show-leak-kinds=all \
    uvicorn lelamp.api.app:app
```

**解决方案**:
- 配置 `--max-requests` 自动重启 worker
- 定期重启服务
- 分析内存使用模式

---

## 安全建议

### 1. 认证和授权

API 已内置 JWT 认证机制：

- **JWT 签名密钥**：通过 `LELAMP_JWT_SECRET` 环境变量配置
- **密钥未设置时**：启动时自动生成随机密钥，并输出警告日志（生产环境务必手动设置固定密钥）
- **设备绑定**：首次 WiFi 设置时自动生成 `device_secret`（16 位 hex 字符串），存储在 `/var/lib/lelamp/setup_status.json`
- **设备信息接口**：`GET /api/system/device` 返回设备信息及 `device_secret`
- **密钥验证**：使用 `hmac.compare_digest()` 进行常量时间比较，防止时序攻击
- **LiveKit Token**：通过 `POST /api/livekit/token` 端点获取（需 JWT 认证），强制使用已认证用户身份

### 2. CORS 配置

生产环境限制 CORS：

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://lelamp.example.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

### 3. 速率限制

使用 slowapi:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.get("/api/devices/{lamp_id}/state")
@limiter.limit("10/minute")
async def get_device_state(request: Request, lamp_id: str):
    ...
```

### 4. 输入验证

严格验证所有输入：

```python
from pydantic import BaseModel, validator

class CommandRequest(BaseModel):
    type: str
    action: str
    params: dict

    @validator('type')
    def validate_type(cls, v):
        allowed_types = ['motor_move', 'rgb_set', 'vision_capture']
        if v not in allowed_types:
            raise ValueError(f'Invalid command type: {v}')
        return v
```

---

## 备份和恢复

### 数据库备份

**PostgreSQL**:
```bash
# 备份
sudo -u postgres pg_dump lelamp > /backup/lelamp_$(date +%Y%m%d).sql

# 恢复
sudo -u postgres psql lelamp < /backup/lelamp_20260317.sql
```

**SQLite**:
```bash
# 备份
cp /opt/lelamp/lelamp.db /backup/lelamp_$(date +%Y%m%d).db

# 恢复
cp /backup/lelamp_20260317.db /opt/lelamp/lelamp.db
```

### 自动化备份

创建 `/usr/local/bin/lelamp-backup.sh`:

```bash
#!/bin/bash

BACKUP_DIR="/backup"
DATE=$(date +%Y%m%d_%H%M%S)
DB_NAME="lelamp"

# 创建备份目录
mkdir -p $BACKUP_DIR

# 备份数据库
sudo -u postgres pg_dump $DB_NAME | gzip > $BACKUP_DIR/${DB_NAME}_${DATE}.sql.gz

# 保留最近 30 天的备份
find $BACKUP_DIR -name "${DB_NAME}_*.sql.gz" -mtime +30 -delete

echo "Backup completed: ${DB_NAME}_${DATE}.sql.gz"
```

添加到 crontab：

```bash
# 每天凌晨 2 点备份
0 2 * * * /usr/local/bin/lelamp-backup.sh
```

---

## 升级和维护

### 应用升级

```bash
# 停止服务
sudo systemctl stop lelamp-api

# 备份当前版本
sudo cp -r /opt/lelamp /opt/lelamp.backup

# 拉取新代码
cd /opt/lelamp
git pull origin main

# 更新依赖
sudo -u lelamp uv sync --extra api

# 运行数据库迁移（如有）
sudo -u lelamp uv run -m lelamp.database.migrate

# 启动服务
sudo systemctl start lelamp-api

# 验证
curl http://localhost:8000/health
```

### 定期维护

```bash
# 每周任务
- 清理旧日志
- 检查磁盘空间
- 更新系统补丁

# 每月任务
- 审查访问日志
- 性能分析
- 备份验证
```

---

## 支持

如有问题，请参考：

- API 文档: `/docs`
- 日志文件: `/var/log/lelamp/`
- 系统日志: `journalctl -u lelamp-api`
- GitHub Issues: https://github.com/your-org/lelamp_runtime/issues
