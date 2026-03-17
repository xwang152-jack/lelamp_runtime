# LeLamp API 文档

## 概述

LeLamp API 是一个 RESTful API，用于管理和监控 LeLamp 智能台灯设备。提供设备状态查询、命令发送、对话记录、操作日志等功能。

### 基础信息

- **Base URL**: `http://localhost:8000/api`
- **API 版本**: v1.0.0
- **Content-Type**: `application/json`
- **认证**: 当前版本无需认证（生产环境建议添加）

### 响应格式

所有 API 响应遵循统一格式：

**成功响应** (2xx):
```json
{
  "field1": "value1",
  "field2": "value2"
}
```

**错误响应** (4xx/5xx):
```json
{
  "detail": "错误描述信息"
}
```

### 通用参数

#### 分页参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `skip` | int | 0 | 跳过记录数 |
| `limit` | int | 50 | 返回记录数（最大100）|

#### 时间窗口参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `hours` | int | 24 | 时间窗口（小时），1-168 |
| `days` | int | 7 | 统计时间窗口（天），1-30 |

---

## REST API 端点

### 1. 设备列表

#### 获取所有设备

```http
GET /api/devices
```

**响应示例**:
```json
{
  "devices": [
    {
      "lamp_id": "lelamp_001",
      "last_seen": "2026-03-17T12:00:00Z",
      "state": "idle"
    },
    {
      "lamp_id": "lelamp_002",
      "last_seen": "2026-03-17T11:30:00Z",
      "state": "listening"
    }
  ]
}
```

---

### 2. 设备状态

#### 获取设备状态

```http
GET /api/devices/{lamp_id}/state
```

**路径参数**:
- `lamp_id` (string): 设备 ID，格式：字母、数字、连字符、下划线

**响应示例**:
```json
{
  "lamp_id": "lelamp_001",
  "status": "online",
  "conversation_state": "idle",
  "timestamp": "2026-03-17T12:00:00Z",
  "motor_positions": {
    "base_yaw": 0.0,
    "base_pitch": 0.0,
    "elbow_pitch": 45.0,
    "wrist_roll": 0.0,
    "wrist_pitch": -30.0
  },
  "light_color": {
    "r": 255,
    "g": 244,
    "b": 229
  },
  "camera_active": false,
  "uptime_seconds": 3600
}
```

**错误响应**:
- `400`: Invalid lamp_id format

---

### 3. 设备命令

#### 发送设备命令

```http
POST /api/devices/{lamp_id}/command
```

**路径参数**:
- `lamp_id` (string): 设备 ID

**请求体**:
```json
{
  "type": "motor_move",
  "action": "move_joint",
  "params": {
    "joint_name": "base_yaw",
    "position": 45.0,
    "speed": 50
  }
}
```

**命令类型**:
- `motor_move`: 电机控制
- `rgb_set`: RGB 颜色设置
- `rgb_effect`: RGB 灯效
- `vision_capture`: 视觉捕获
- `play_recording`: 播放动作
- `set_volume`: 设置音量
- `system_command`: 系统命令

**响应示例**:
```json
{
  "success": true,
  "command_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Command received",
  "timestamp": "2026-03-17T12:00:00Z"
}
```

**错误响应**:
- `400`: Invalid lamp_id format / Missing required fields
- `404`: Device not found

---

### 4. 对话记录

#### 获取设备对话历史

```http
GET /api/devices/{lamp_id}/conversations?skip=0&limit=50
```

**路径参数**:
- `lamp_id` (string): 设备 ID

**查询参数**:
- `skip` (int): 跳过记录数，默认 0
- `limit` (int): 返回记录数，默认 50，最大 100

**响应示例**:
```json
{
  "total": 150,
  "conversations": [
    {
      "id": 1,
      "timestamp": "2026-03-17T12:00:00Z",
      "lamp_id": "lelamp_001",
      "user_input": "你好",
      "ai_response": "你好！我是 LeLamp，有什么可以帮你的吗？",
      "duration": 2500,
      "messages": [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！我是 LeLamp，有什么可以帮你的吗？"}
      ]
    }
  ]
}
```

**错误响应**:
- `400`: Invalid lamp_id format
- `422`: Invalid query parameters

---

### 5. 操作日志

#### 获取设备操作日志

```http
GET /api/devices/{lamp_id}/operations?skip=0&limit=50&hours=24
```

**路径参数**:
- `lamp_id` (string): 设备 ID

**查询参数**:
- `skip` (int): 跳过记录数，默认 0
- `limit` (int): 返回记录数，默认 50，最大 100
- `hours` (int): 时间窗口（小时），默认 24，范围 1-168

**响应示例**:
```json
{
  "total": 50,
  "operations": [
    {
      "id": 1,
      "timestamp": "2026-03-17T12:00:00Z",
      "lamp_id": "lelamp_001",
      "operation_type": "motor_move",
      "action": "move_joint",
      "params": {
        "joint_name": "base_yaw",
        "position": 45.0
      },
      "success": true,
      "error_message": null,
      "duration_ms": 250
    }
  ]
}
```

**错误响应**:
- `400`: Invalid lamp_id format
- `422`: Invalid query parameters

---

### 6. 设备健康

#### 获取设备健康状态

```http
GET /api/devices/{lamp_id}/health
```

**路径参数**:
- `lamp_id` (string): 设备 ID

**响应示例**:
```json
{
  "lamp_id": "lelamp_001",
  "overall_status": "healthy",
  "motors": [
    {
      "name": "base_yaw",
      "temperature": 45.0,
      "voltage": 12.0,
      "load": 0.3,
      "position_error": 0.5,
      "status": "healthy"
    }
  ],
  "last_check": "2026-03-17T12:00:00Z"
}
```

**健康状态值**:
- `healthy`: 健康
- `warning`: 警告
- `critical`: 严重
- `stalled`: 卡死
- `unknown`: 未知

**错误响应**:
- `400`: Invalid lamp_id format

---

### 7. 设备统计

#### 获取设备统计数据

```http
GET /api/devices/{lamp_id}/statistics?days=7
```

**路径参数**:
- `lamp_id` (string): 设备 ID

**查询参数**:
- `days` (int): 统计天数，默认 7，范围 1-30

**响应示例**:
```json
{
  "lamp_id": "lelamp_001",
  "period_days": 7,
  "total_operations": 500,
  "success_rate": 0.98,
  "operation_counts": {
    "motor_move": 300,
    "rgb_set": 150,
    "vision_capture": 50
  },
  "avg_duration_ms": 200.5,
  "most_common_operation": "motor_move"
}
```

**错误响应**:
- `400`: Invalid lamp_id format
- `422`: Invalid query parameters

---

### 8. 历史记录查询

#### 查询单条对话记录

```http
GET /api/history/conversations/{id}
```

**路径参数**:
- `id` (int): 对话记录 ID

**响应示例**:
```json
{
  "id": 1,
  "timestamp": "2026-03-17T12:00:00Z",
  "lamp_id": "lelamp_001",
  "user_input": "你好",
  "ai_response": "你好！我是 LeLamp，有什么可以帮你的吗？",
  "duration": 2500,
  "messages": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！我是 LeLamp，有什么可以帮你的吗？"}
  ]
}
```

**错误响应**:
- `404`: Conversation not found

---

#### 查询单条操作日志

```http
GET /api/history/operations/{id}
```

**路径参数**:
- `id` (int): 操作日志 ID

**响应示例**:
```json
{
  "id": 1,
  "timestamp": "2026-03-17T12:00:00Z",
  "lamp_id": "lelamp_001",
  "operation_type": "motor_move",
  "action": "move_joint",
  "params": {
    "joint_name": "base_yaw",
    "position": 45.0
  },
  "success": true,
  "error_message": null,
  "duration_ms": 250
}
```

**错误响应**:
- `404`: Operation log not found

---

### 9. 健康检查

#### API 健康检查

```http
GET /health
```

**响应示例**:
```json
{
  "status": "healthy",
  "service": "lelamp-api",
  "active_connections": {
    "lelamp_001": 2,
    "lelamp_002": 1
  }
}
```

---

## WebSocket API

### 连接端点

```text
ws://localhost:8000/api/ws/{lamp_id}
```

**路径参数**:
- `lamp_id` (string): 设备 ID

### 消息类型

#### 客户端 → 服务端

**1. 心跳 (Ping)**
```json
{
  "type": "ping"
}
```

**响应 (Pong)**:
```json
{
  "type": "pong",
  "timestamp": "2026-03-17T12:00:00Z"
}
```

**2. 订阅频道**
```json
{
  "type": "subscribe",
  "channels": ["state", "events", "logs", "notifications"]
}
```

**可用频道**:
- `state`: 状态更新
- `events`: 事件通知
- `logs`: 操作日志
- `notifications`: 系统通知
- `conversations`: 对话更新
- `health`: 健康状态

**响应**:
```json
{
  "type": "subscription_confirmed",
  "channels": ["state", "events"],
  "timestamp": "2026-03-17T12:00:00Z"
}
```

**3. 取消订阅**
```json
{
  "type": "unsubscribe",
  "channels": ["logs"]
}
```

**4. 发送命令**
```json
{
  "type": "command",
  "action": "move_joint",
  "params": {
    "joint_name": "base_yaw",
    "position": 45.0
  }
}
```

#### 服务端 → 客户端

**1. 连接确认**
```json
{
  "type": "connected",
  "lamp_id": "lelamp_001",
  "server_time": "2026-03-17T12:00:00Z",
  "message": "WebSocket connection established"
}
```

**2. 状态更新**
```json
{
  "type": "state_update",
  "data": {
    "lamp_id": "lelamp_001",
    "conversation_state": "listening",
    "motor_positions": {...},
    "light_color": {...},
    "timestamp": "2026-03-17T12:00:00Z"
  },
  "timestamp": "2026-03-17T12:00:01Z"
}
```

**3. 事件通知**
```json
{
  "type": "event",
  "event_type": "motor_move",
  "data": {
    "joint_name": "base_yaw",
    "position": 45.0
  },
  "timestamp": "2026-03-17T12:00:00Z"
}
```

**4. 操作日志**
```json
{
  "type": "log",
  "log_entry": {
    "id": 1,
    "operation_type": "motor_move",
    "action": "move_joint",
    "params": {...},
    "success": true,
    "timestamp": "2026-03-17T12:00:00Z",
    "duration_ms": 250
  },
  "timestamp": "2026-03-17T12:00:00Z"
}
```

**5. 系统通知**
```json
{
  "type": "notification",
  "message": "Device temperature warning",
  "level": "warning",
  "timestamp": "2026-03-17T12:00:00Z",
  "metadata": {
    "temperature": 65.0
  }
}
```

**6. 错误消息**
```json
{
  "type": "error",
  "message": "Invalid message type",
  "code": "INVALID_MESSAGE_TYPE",
  "timestamp": "2026-03-17T12:00:00Z"
}
```

### 连接管理

- **心跳间隔**: 建议 30 秒发送一次 ping
- **重连策略**: 指数退避，初始 1 秒，最大 30 秒
- **连接超时**: 60 秒无活动自动断开

---

## 数据模型

### 设备状态 (DeviceState)

```typescript
{
  lamp_id: string;
  status: "online" | "offline";
  conversation_state: "idle" | "listening" | "thinking" | "speaking";
  timestamp: string;  // ISO 8601
  motor_positions: {
    [joint_name: string]: number;
  };
  light_color: {
    r: number;  // 0-255
    g: number;  // 0-255
    b: number;  // 0-255
  };
  camera_active: boolean;
  uptime_seconds?: number;
}
```

### 对话记录 (Conversation)

```typescript
{
  id: number;
  timestamp: string;  // ISO 8601
  lamp_id: string;
  user_input?: string;
  ai_response?: string;
  duration?: number;  // 毫秒
  messages: Array<{
    role: "user" | "assistant" | "system";
    content: string;
  }>;
}
```

### 操作日志 (OperationLog)

```typescript
{
  id: number;
  timestamp: string;  // ISO 8601
  lamp_id: string;
  operation_type: string;
  action: string;
  params: object;
  success: boolean;
  error_message?: string;
  duration_ms?: number;
}
```

### 健康状态 (HealthStatus)

```typescript
{
  lamp_id: string;
  overall_status: "healthy" | "warning" | "critical" | "stalled" | "unknown";
  motors: Array<{
    name: string;
    temperature: number;
    voltage: number;
    load: number;
    position_error?: number;
    status: string;
  }>;
  last_check: string;  // ISO 8601
}
```

---

## 错误码

### HTTP 状态码

| 状态码 | 说明 |
|--------|------|
| 200 | 成功 |
| 400 | 请求参数错误 |
| 404 | 资源不存在 |
| 422 | 请求参数验证失败 |
| 500 | 服务器内部错误 |

### 常见错误

**Invalid lamp_id format**
```json
{
  "detail": "Invalid lamp_id format"
}
```
**原因**: lamp_id 包含非法字符（只允许字母、数字、连字符、下划线）

**Missing required fields**
```json
{
  "detail": "Command must include 'type' and 'action'"
}
```
**原因**: 命令请求缺少必需字段

**Invalid query parameters**
```json
{
  "detail": [
    {
      "type": "less_than_equal",
      "loc": ["query", "limit"],
      "msg": "Input should be less than or equal to 100",
      "input": "200",
      "ctx": {"le": 100}
    }
  ]
}
```
**原因**: 查询参数超出范围

**Device not found**
```json
{
  "detail": "Device not found"
}
```
**原因**: 设备不存在

---

## 速率限制

当前版本未实施速率限制。生产环境建议：

- 认证用户: 100 req/min
- 匿名用户: 20 req/min
- WebSocket 连接: 每用户最多 5 个并发连接

---

## 最佳实践

### 1. 错误处理

始终检查 HTTP 状态码并处理错误：

```python
import requests

response = requests.get("http://localhost:8000/api/devices/lelamp_001/state")

if response.status_code == 200:
    data = response.json()
    # 处理数据
else:
    error = response.json()
    print(f"Error {response.status_code}: {error['detail']}")
```

### 2. 分页处理

使用分页避免一次性加载大量数据：

```python
def get_all_conversations(lamp_id):
    skip = 0
    limit = 50
    all_conversations = []

    while True:
        response = requests.get(
            f"http://localhost:8000/api/devices/{lamp_id}/conversations",
            params={"skip": skip, "limit": limit}
        )
        data = response.json()
        all_conversations.extend(data["conversations"])

        if len(data["conversations"]) < limit:
            break

        skip += limit

    return all_conversations
```

### 3. WebSocket 重连

实现自动重连机制：

```python
import asyncio
import websockets
import json

async def connect_websocket(lamp_id):
    url = f"ws://localhost:8000/api/ws/{lamp_id}"
    retry_delay = 1

    while True:
        try:
            async with websockets.connect(url) as ws:
                retry_delay = 1  # 重置重连延迟
                await subscribe(ws, ["state", "events"])

                async for message in ws:
                    data = json.loads(message)
                    handle_message(data)

        except Exception as e:
            print(f"Connection error: {e}")
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 30)  # 指数退避，最大 30 秒

async def subscribe(ws, channels):
    await ws.send(json.dumps({
        "type": "subscribe",
        "channels": channels
    }))
```

### 4. 缓存策略

对不常变化的数据使用缓存：

```python
import time
from functools import lru_cache

class DeviceStateCache:
    def __init__(self, ttl=60):
        self.ttl = ttl
        self.cache = {}

    def get(self, lamp_id):
        if lamp_id in self.cache:
            data, timestamp = self.cache[lamp_id]
            if time.time() - timestamp < self.ttl:
                return data
        return None

    def set(self, lamp_id, data):
        self.cache[lamp_id] = (data, time.time())
```

---

## 版本历史

### v1.0.0 (2026-03-17)
- 初始版本发布
- 设备状态 API
- 对话记录 API
- 操作日志 API
- 健康监控 API
- WebSocket 实时推送
