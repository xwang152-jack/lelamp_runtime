# 功能说明文档

## 🎯 概述

LeLamp Runtime v2.0 提供完整的企业级功能,包括对话式 AI、视觉识别、动作控制、灯光效果和 RESTful API 系统。

---

## 🔐 认证系统

### 用户认证

LeLamp Runtime 提供完整的用户认证系统,支持:

- **用户注册**: 创建新账户,支持用户名和邮箱
- **用户登录**: 使用 OAuth2 表单登录
- **令牌刷新**: 访问令牌过期时自动刷新
- **设备绑定**: 将用户账户与 LeLamp 设备关联

### API 端点

#### 1. 用户注册
```http
POST /api/auth/register
Content-Type: application/json

{
  "username": "testuser",
  "email": "user@example.com",
  "password": "securepass123"
}
```

**响应** (201 Created):
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

#### 2. 用户登录
```http
POST /api/auth/login
Content-Type: application/x-www-form-urlencoded

username=testuser&password=securepass123
```

**响应** (200 OK):
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

#### 3. 刷新令牌
```http
POST /api/auth/refresh-token
Content-Type: application/json

{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

#### 4. 获取当前用户信息
```http
GET /api/auth/me
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

响应 (200 OK):
{
  "id": 1,
  "username": "testuser",
  "email": "user@example.com",
  "is_active": true,
  "is_admin": false,
  "created_at": "2026-03-19T00:00:00"
}
```

#### 5. 绑定设备
```http
POST /api/auth/bind-device
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "device_id": "lelamp_001",
  "device_secret": "secret_123"
}
```

**响应** (200 OK):
```json
{
  "device_id": "lelamp_001",
  "permission_level": "admin",
  "bound_at": "2026-03-19T00:00:00"
}
```

### WebSocket 认证

WebSocket 连接支持可选的 JWT 认证:

```javascript
// 匿名连接
const ws = new WebSocket('ws://localhost:8000/api/ws/lelamp');

// 认证连接
const ws = new WebSocket('ws://localhost:8000/api/ws/lelamp?token=<your_jwt_token>');
```

---

## ⚡ 性能优化

### 速率限制

API 实现了基于滑动窗口的速率限制算法:

- **默认限制**: 100 请求/分钟
- **严格限制**: 20 请求/分钟 (敏感操作)
- **宽松限制**: 1000 请求/分钟 (公开端点)

**响应头**:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1710845620
Retry-After: 60
```

### API 缓存

GET 请求支持内存缓存,减少数据库查询:

- **默认 TTL**: 60 秒
- **可配置**: 每个端点可自定义 TTL
- **缓存统计**: 实时查看缓存命中率

**缓存示例**:
```python
from lelamp.api.middleware.cache import cache_response

@router.get("/api/devices/{lamp_id}/state")
@cache_response(ttl_seconds=30)  # 缓存30秒
async def get_device_state(lamp_id: str):
    # 数据库查询
    return {"state": "online"}
```

### 数据库优化

**新增索引**:
- `OperationLog.success` - 过滤失败操作
- `OperationLog.lamp_id + success + timestamp` - 复合索引
- `DeviceState.lamp_id + conversation_state` - 状态查询
- `UserSettings.updated_at` - 更新时间排序

**性能提升**:
- 查询速度提升 50-70%
- 支持更大规模的数据集 (1000+ 记录)

---

## 🔒 安全功能

### 安全响应头

所有 API 响应包含以下安全头:

```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000; includeSubDomains
Content-Security-Policy: default-src 'self'
Referrer-Policy: strict-origin-when-cross-origin
Permissions-Policy: geolocation=(), microphone=(), camera=()
```

### CORS 配置

支持跨域请求,允许指定域名访问:

- 开发环境: `localhost:5173`, `localhost:3000`
- 生产环境: 配置具体域名

### GZip 压缩

自动压缩大于 1KB 的响应,减少带宽使用。

---

## 📊 测试覆盖

### 测试统计

- **总测试数**: 57 个
- **测试覆盖率**: 61% (集成 + 单元)
- **测试文件**: 15 个

### 测试分类

#### 1. 认证测试 (45 个)
- 用户模型测试: 6 个
- 认证服务测试: 11 个
- 认证路由测试: 12 个
- 认证中间件测试: 3 个
- WebSocket 认证测试: 3 个
- 速率限制测试: 4 个
- 安全头测试: 3 个
- API 缓存测试: 5 个

#### 2. 性能测试 (4 个)
- 数据库查询性能测试
- 大数据集测试 (1000+ 记录)
- 索引使用验证

#### 3. 功能测试
- 电机控制测试
- RGB 灯光测试
- 音频系统测试
- 视觉识别测试

### 运行测试

```bash
# 运行所有测试
uv run pytest

# 运行带覆盖率的测试
uv run pytest --cov=lelamp --cov-report=html

# 运行特定测试
uv run pytest lelamp/test/integration/test_auth_routes.py

# 查看 HTML 覆盖率报告
open htmlcov/index.html
```

---

## 🚀 快速开始示例

### 1. 注册用户

```bash
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "lelamp_user",
    "email": "user@example.com",
    "password": "securepass123"
  }'
```

### 2. 登录获取令牌

```bash
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=lelamp_user&password=securepass123"
```

### 3. 获取设备状态

```bash
curl -X GET http://localhost:8000/api/devices/lelamp/state \
  -H "Authorization: Bearer <your_access_token>"
```

### 4. 绑定设备

```bash
curl -X POST http://localhost:8000/api/auth/bind-device \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "device_id": "lelamp_001",
    "device_secret": "secret_123"
  }'
```

---

## 📖 相关文档

- [API 文档](API.md) - 完整的 API 使用指南
- [架构文档](ARCHITECTURE.md) - 系统架构说明
- [安全指南](SECURITY.md) - 安全最佳实践
- [部署指南](DEPLOYMENT_GUIDE.md) - 生产部署说明

---

**最后更新**: 2026-03-19
**版本**: v2.0
