# API 使用指南

## 📚 概述

LeLamp Runtime 提供完整的 RESTful API 和 WebSocket 接口,用于设备控制、状态监控和配置管理。

---

## 🔌 基础信息

### API 基础 URL

```
开发环境: http://localhost:8000
生产环境: https://api.yourdomain.com
mDNS 发现: http://<device_id>.local:8000
```

> **mDNS 设备发现**: API 启动时通过 zeroconf 注册 `_http._tcp` 服务，局域网内可通过 `http://<device_id>.local:8000` 直接访问设备，无需知道 IP 地址。zeroconf 不可用时静默降级。

### 认证方式

API 使用 JWT (JSON Web Token) 进行认证:

```http
Authorization: Bearer <your_access_token>
```

**获取令牌**: 参见 [认证 API](#认证-api)

### 文档界面

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

---

## 🔐 认证 API

### 1. 用户注册

创建新用户账户并返回访问令牌。

```http
POST /api/auth/register
Content-Type: application/json
```

**请求体**:
```json
{
  "username": "lelamp_user",
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

**字段验证**:
- `username`: 3-50 字符,唯一
- `email`: 有效邮箱格式,唯一
- `password`: 6-100 字符

### 2. 用户登录

使用用户名和密码登录。

```http
POST /api/auth/login
Content-Type: application/x-www-form-urlencoded
```

**请求体**:
```
username=lelamp_user&password=securepass123
```

**响应** (200 OK):
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

### 3. 刷新令牌

使用刷新令牌获取新的访问令牌。

```http
POST /api/auth/refresh-token
Content-Type: application/json
```

**请求体**:
```json
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**响应** (200 OK):
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

**注意**: 旧刷新令牌使用后自动撤销。

### 4. 获取当前用户信息

获取当前登录用户的信息。

```http
GET /api/auth/me
Authorization: Bearer <access_token>
```

**响应** (200 OK):
```json
{
  "id": 1,
  "username": "lelamp_user",
  "email": "user@example.com",
  "is_active": true,
  "is_admin": false,
  "created_at": "2026-03-19T00:00:00",
  "last_login": "2026-03-19T12:00:00"
}
```

### 5. 绑定设备

将用户账户与 LeLamp 设备关联。`device_secret` 在设备首次 WiFi 配网时自动生成（`secrets.token_hex(8)`，16 字符），存储在 `/var/lib/lelamp/setup_status.json`。服务端使用 `hmac.compare_digest()` 验证，不存储明文。

```http
POST /api/auth/bind-device
Authorization: Bearer <access_token>
Content-Type: application/json
```

**请求体**:
```json
{
  "device_id": "lelamp_001",
  "device_secret": "a1b2c3d4e5f6g7h8"
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

**安全说明**:
- `device_secret` 使用 `hmac.compare_digest()` 比较，防止时序攻击
- 环境变量 `LELAMP_DEVICE_SECRET` 可作为 fallback 密钥
- 前端显示设备信息卡片，包含可复制的 secret 和一键绑定按钮

---

## 📱 设备管理 API

### 6. 获取设备状态

获取设备的实时状态。

```http
GET /api/devices/{lamp_id}/state
Authorization: Bearer <access_token>
```

**路径参数**:
- `lamp_id`: 设备 ID (例如: "lelamp")

**响应** (200 OK):
```json
{
  "lamp_id": "lelamp",
  "status": "online",
  "conversation_state": "idle",
  "motor_positions": {
    "base_yaw": 0,
    "base_pitch": 0,
    "elbow_pitch": 0,
    "wrist_roll": 0,
    "wrist_pitch": 0
  },
  "rgb_colors": {
    "r": 255,
    "g": 244,
    "b": 229
  },
  "timestamp": "2026-03-19T12:00:00"
}
```

### 7. 获取设备历史记录

获取设备的操作历史。

```http
GET /api/devices/{lamp_id}/history?limit=10&offset=0
Authorization: Bearer <access_token>
```

**查询参数**:
- `limit`: 返回记录数 (默认: 10)
- `offset`: 偏移量 (默认: 0)

**响应** (200 OK):
```json
{
  "total": 100,
  "items": [
    {
      "id": 1,
      "operation_type": "motor_move",
      "parameters": {"joint": "base_yaw", "position": 45},
      "success": true,
      "timestamp": "2026-03-19T12:00:00"
    }
  ]
}
```

---

## 📋 设备信息 API

### 获取设备信息

获取当前设备的基本信息和绑定密钥（用于设备绑定流程）。

```http
GET /api/system/device
```

**响应** (200 OK):
```json
{
  "device_id": "lelamp",
  "hostname": "lelamp",
  "model": "LeLamp v1",
  "version": "3.3.0",
  "device_secret": "a1b2c3d4e5f6g7h8"
}
```

**说明**:
- `device_secret` 在设备首次 WiFi 配网时自动生成（16 字符 hex）
- 前端使用此端点获取设备信息并展示绑定卡片
- `device_secret` 也可通过环境变量 `LELAMP_DEVICE_SECRET` 覆盖

---

## 🎙️ LiveKit Token API

### 生成 LiveKit 客户端 Token

替代旧的 CLI 脚本 `scripts/tools/generate_client_token.py`，通过 API 端点生成 LiveKit 客户端连接 Token。

```http
POST /api/livekit/token
Authorization: Bearer <access_token>
Content-Type: application/json
```

**请求体**:
```json
{
  "room": "lelamp-room"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `room` | string | 是 | 房间名（1-128 字符，Pydantic Field 验证） |
| `identity` | string | 否 | 用户身份（未提供时使用 JWT 认证身份） |

**响应** (200 OK):
```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "room": "lelamp-room",
  "identity": "user_123"
}
```

**安全特性**:
- 需要 JWT 认证
- 强制使用已认证用户的身份，防止身份伪造
- 房间名经过 Pydantic Field 验证

---

## ⚙️ 设置 API

### 8. 获取设备设置

获取设备的配置设置。

```http
GET /api/devices/{lamp_id}/settings
Authorization: Bearer <access_token>
```

**响应** (200 OK):
```json
{
  "lamp_id": "lelamp",
  "settings": {
    "led_brightness": 25,
    "volume": 80,
    "motion_cooldown": 1.0,
    "vision_enabled": true
  },
  "updated_at": "2026-03-19T12:00:00"
}
```

### 9. 更新设备设置

更新设备的配置设置。

```http
PUT /api/devices/{lamp_id}/settings
Authorization: Bearer <access_token>
Content-Type: application/json
```

**请求体**:
```json
{
  "led_brightness": 50,
  "volume": 90
}
```

**响应** (200 OK):
```json
{
  "lamp_id": "lelamp",
  "updated_fields": ["led_brightness", "volume"],
  "updated_at": "2026-03-19T12:00:00"
}
```

---

## 🔌 WebSocket API

### 连接端点

建立 WebSocket 连接以接收实时状态更新。

```
ws://localhost:8000/api/ws/{lamp_id}
```

**带认证的连接**:
```
ws://localhost:8000/api/ws/{lamp_id}?token=<access_token>
```

### 消息类型

WebSocket 服务器推送以下类型的事件:

| 消息类型 | 描述 | 示例数据 |
|---------|------|---------|
| `state_changed` | 对话状态变化 | `{"state": "listening"}` |
| `motor_moved` | 电机运动 | `{"joint": "base_yaw", "position": 45}` |
| `rgb_changed` | RGB 灯光变化 | `{"r": 0, "g": 140, "b": 255}` |
| `operation_log` | 操作日志 | `{"operation": "move_joint", "success": true}` |
| `conversation_started` | 对话开始 | `{"conversation_id": 123}` |
| `conversation_ended` | 对话结束 | `{"conversation_id": 123, "duration": 60}` |
| `error` | 错误事件 | `{"message": "Motor stalled"}` |

### JavaScript 客户端示例

```javascript
// 建立 WebSocket 连接
const ws = new WebSocket('ws://localhost:8000/api/ws/lelamp?token=<your_token>');

// 监听消息
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  switch (data.type) {
    case 'state_changed':
      console.log('Conversation state:', data.state);
      break;
    case 'motor_moved':
      console.log(`Motor ${data.joint} moved to ${data.position}`);
      break;
    case 'rgb_changed':
      console.log(`RGB color: ${data.r}, ${data.g}, ${data.b}`);
      break;
    case 'error':
      console.error('Error:', data.message);
      break;
  }
};

// 发送命令 (如果支持)
ws.send(JSON.stringify({
  type: 'command',
  action: 'move_joint',
  parameters: { joint: 'base_yaw', position: 45 }
}));

// 关闭连接
ws.close();
```

### Python 客户端示例

```python
import asyncio
import websockets
import json

async def connect_to_websocket(lamp_id: str, token: str):
    uri = f"ws://localhost:8000/api/ws/{lamp_id}?token={token}"

    async with websockets.connect(uri) as websocket:
        while True:
            message = await websocket.recv()
            data = json.loads(message)

            if data['type'] == 'state_changed':
                print(f"State: {data['state']}")
            elif data['type'] == 'motor_moved':
                print(f"Motor: {data['joint']} -> {data['position']}")

# 运行客户端
asyncio.run(connect_to_websocket('lelamp', '<your_token>'))
```

---

## 🛡️ 安全功能

### 速率限制

API 实现了基于滑动窗口的速率限制:

| 端点类型 | 限制 | 时间窗口 |
|---------|------|---------|
| 普通 API | 100 请求 | 60 秒 |
| 敏感操作 (注册/登录) | 20 请求 | 60 秒 |
| 公开端点 | 1000 请求 | 60 秒 |

**超限响应** (429 Too Many Requests):
```json
{
  "error": "Rate limit exceeded",
  "limit": 100,
  "reset": 1710845620
}
```

**速率限制头**:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1710845620
Retry-After: 60
```

### CORS 配置

支持跨域请求,允许指定域名:

**开发环境**:
- `http://localhost:5173`
- `http://localhost:3000`
- `http://127.0.0.1:5173`
- `http://127.0.0.1:3000`

**生产环境**: 需要在 `lelamp/api/app.py` 中配置具体域名。

### JWT 签名密钥

API 使用 `LELAMP_JWT_SECRET` 环境变量作为 JWT 签名密钥：

| 环境变量 | 说明 |
|---------|------|
| `LELAMP_JWT_SECRET` | JWT 签名密钥（生产环境必须设置） |
| `LELAMP_DEVICE_SECRET` | 设备绑定密钥（fallback） |

**行为**:
- `LELAMP_JWT_SECRET` 未设置时，自动生成随机密钥并输出警告
- 随机密钥在进程重启后变化，所有现有 token 失效
- 生产环境务必设置固定密钥

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

---

## 📊 错误处理

### 标准错误响应

**400 Bad Request**:
```json
{
  "detail": "Validation error",
  "errors": [
    {
      "field": "username",
      "message": "Username must be at least 3 characters"
    }
  ]
}
```

**401 Unauthorized**:
```json
{
  "detail": "Invalid authentication credentials"
}
```

**403 Forbidden**:
```json
{
  "detail": "You don't have permission to access this resource"
}
```

**404 Not Found**:
```json
{
  "detail": "Resource not found"
}
```

**429 Too Many Requests**:
```json
{
  "error": "Rate limit exceeded",
  "limit": 100,
  "reset": 1710845620
}
```

**500 Internal Server Error**:
```json
{
  "detail": "Internal server error",
  "error_id": "abc123"
}
```

---

## 🚀 快速开始示例

### cURL 示例

```bash
# 1. 注册用户
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "lelamp_user",
    "email": "user@example.com",
    "password": "securepass123"
  }'

# 2. 登录获取令牌
RESPONSE=$(curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=lelamp_user&password=securepass123")

TOKEN=$(echo $RESPONSE | jq -r '.access_token')

# 3. 获取设备状态
curl -X GET http://localhost:8000/api/devices/lelamp/state \
  -H "Authorization: Bearer $TOKEN"

# 4. 更新设置
curl -X PUT http://localhost:8000/api/devices/lelamp/settings \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"led_brightness": 50}'
```

### Python 示例

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. 注册用户
response = requests.post(f"{BASE_URL}/api/auth/register", json={
    "username": "lelamp_user",
    "email": "user@example.com",
    "password": "securepass123"
})
data = response.json()
token = data["access_token"]

# 2. 设置认证头
headers = {"Authorization": f"Bearer {token}"}

# 3. 获取设备状态
response = requests.get(
    f"{BASE_URL}/api/devices/lelamp/state",
    headers=headers
)
state = response.json()
print(f"Device status: {state['status']}")

# 4. 更新设置
response = requests.put(
    f"{BASE_URL}/api/devices/lelamp/settings",
    headers=headers,
    json={"led_brightness": 50}
)
print(f"Settings updated: {response.json()}")
```

### JavaScript/TypeScript 示例

```typescript
const BASE_URL = 'http://localhost:8000';

// 1. 注册用户
async function register(username: string, email: string, password: string) {
  const response = await fetch(`${BASE_URL}/api/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, email, password })
  });
  const data = await response.json();
  return data.access_token;
}

// 2. 获取设备状态
async function getDeviceState(token: string, lampId: string) {
  const response = await fetch(
    `${BASE_URL}/api/devices/${lampId}/state`,
    {
      headers: { 'Authorization': `Bearer ${token}` }
    }
  );
  return await response.json();
}

// 使用
const token = await register('lelamp_user', 'user@example.com', 'securepass123');
const state = await getDeviceState(token, 'lelamp');
console.log('Device status:', state.status);
```

---

## 📚 相关文档

- [功能说明](FEATURES.md) - 完整功能介绍
- [架构文档](ARCHITECTURE.md) - 系统架构说明
- [安全指南](SECURITY.md) - 安全最佳实践
- [部署指南](DEPLOYMENT_GUIDE.md) - 生产部署说明

---

**最后更新**: 2026-04-01
**版本**: v2.1
