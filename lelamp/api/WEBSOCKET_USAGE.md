# WebSocket 实时推送使用指南

## 概述

LeLamp API 提供了 WebSocket 实时推送功能，用于实时接收设备状态更新、事件通知、日志消息等。

## 连接端点

```
ws://<host>/api/ws/{lamp_id}
```

例如：
```javascript
const ws = new WebSocket('ws://localhost:8000/api/ws/lelamp');
```

## 消息格式

所有消息都是 JSON 格式：

```typescript
{
  type: string,
  [key: string]: any
}
```

## 客户端消息

### 1. Ping（心跳）

发送心跳保持连接：

```json
{
  "type": "ping"
}
```

服务端响应：
```json
{
  "type": "pong",
  "timestamp": "2025-03-17T12:34:56.789Z"
}
```

### 2. Subscribe（订阅频道）

订阅一个或多个频道：

```json
{
  "type": "subscribe",
  "channels": ["state", "events", "logs", "notifications"]
}
```

可用频道：
- `state`: 设备状态更新
- `events`: 事件通知（如电机移动、RGB变化等）
- `logs`: 操作日志
- `notifications`: 通知消息
- `conversations`: 对话更新
- `health`: 健康状态

服务端确认：
```json
{
  "type": "subscription_confirmed",
  "channels": ["state", "events", "logs"],
  "timestamp": "2025-03-17T12:34:56.789Z"
}
```

### 3. Unsubscribe（取消订阅）

取消订阅频道：

```json
{
  "type": "unsubscribe",
  "channels": ["logs"]
}
```

### 4. Command（发送命令）

向设备发送命令：

```json
{
  "type": "command",
  "action": "move_joint",
  "params": {
    "joint": "base_yaw",
    "position": 90
  }
}
```

## 服务端消息

### 1. Connected（连接确认）

连接建立后自动发送：

```json
{
  "type": "connected",
  "lamp_id": "lelamp",
  "server_time": "2025-03-17T12:34:56.789Z",
  "message": "WebSocket connection established"
}
```

### 2. State Update（状态更新）

设备状态变化时推送：

```json
{
  "type": "state_update",
  "data": {
    "lamp_id": "lelamp",
    "conversation_state": "idle",
    "motor_positions": {
      "base_yaw": 0,
      "base_pitch": 0,
      "elbow_pitch": 0,
      "wrist_roll": 0,
      "wrist_pitch": 0
    },
    "light_color": {"r": 255, "g": 244, "b": 229},
    "health_status": {
      "overall": "healthy",
      "motors": []
    },
    "uptime_seconds": 3600,
    "timestamp": "2025-03-17T12:34:56.789Z"
  },
  "timestamp": "2025-03-17T12:34:56.789Z"
}
```

### 3. Event（事件通知）

设备事件发生时推送：

```json
{
  "type": "event",
  "event_type": "motor_move",
  "data": {
    "joint": "base_yaw",
    "from": 0,
    "to": 90,
    "duration_ms": 1000
  },
  "timestamp": "2025-03-17T12:34:56.789Z"
}
```

事件类型：
- `motor_move`: 电机移动
- `rgb_set`: RGB灯光设置
- `vision_capture`: 视觉捕获
- `command_sent`: 命令发送
- 等等

### 4. Log（日志消息）

操作日志创建时推送：

```json
{
  "type": "log",
  "log_entry": {
    "id": 123,
    "operation_type": "motor",
    "action": "move_joint",
    "params": {"joint": "base_yaw", "position": 90},
    "success": true,
    "timestamp": "2025-03-17T12:34:56.789Z",
    "duration_ms": 1000
  },
  "timestamp": "2025-03-17T12:34:56.789Z"
}
```

### 5. Notification（通知消息）

重要通知推送：

```json
{
  "type": "notification",
  "message": "Motor temperature warning",
  "level": "warning",
  "timestamp": "2025-03-17T12:34:56.789Z",
  "metadata": {
    "motor": "base_yaw",
    "temperature": 70
  }
}
```

通知级别：
- `info`: 信息
- `warning`: 警告
- `error`: 错误

### 6. Error（错误消息）

客户端错误时推送：

```json
{
  "type": "error",
  "message": "没有有效的频道",
  "code": "INVALID_CHANNELS",
  "timestamp": "2025-03-17T12:34:56.789Z"
}
```

## JavaScript 客户端示例

### 基础连接

```javascript
const lampId = 'lelamp';
const ws = new WebSocket(`ws://localhost:8000/api/ws/${lampId}`);

// 连接打开
ws.addEventListener('open', () => {
  console.log('WebSocket connected');

  // 订阅频道
  ws.send(JSON.stringify({
    type: 'subscribe',
    channels: ['state', 'events', 'logs']
  }));
});

// 接收消息
ws.addEventListener('message', (event) => {
  const message = JSON.parse(event.data);

  switch (message.type) {
    case 'connected':
      console.log('Connected:', message.lamp_id);
      break;

    case 'state_update':
      console.log('State updated:', message.data);
      updateUI(message.data);
      break;

    case 'event':
      console.log('Event:', message.event_type, message.data);
      handleEvent(message.event_type, message.data);
      break;

    case 'log':
      console.log('Log entry:', message.log_entry);
      addLogToUI(message.log_entry);
      break;

    case 'notification':
      console.log('Notification:', message.level, message.message);
      showNotification(message.level, message.message);
      break;

    case 'error':
      console.error('Error:', message.message);
      break;

    default:
      console.log('Unknown message type:', message.type);
  }
});

// 连接关闭
ws.addEventListener('close', () => {
  console.log('WebSocket disconnected');
});

// 连接错误
ws.addEventListener('error', (error) => {
  console.error('WebSocket error:', error);
});

// 心跳保持
setInterval(() => {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'ping' }));
  }
}, 30000); // 每30秒
```

### 带重连的客户端

```javascript
class LeLampWebSocket {
  constructor(lampId, options = {}) {
    this.lampId = lampId;
    this.reconnectInterval = options.reconnectInterval || 5000;
    this.maxReconnectAttempts = options.maxReconnectAttempts || 10;
    this.reconnectAttempts = 0;
    this.subscriptions = new Set(['state']);
    this.messageHandlers = new Map();

    this.connect();
  }

  connect() {
    const url = `ws://localhost:8000/api/ws/${this.lampId}`;
    this.ws = new WebSocket(url);

    this.ws.addEventListener('open', () => {
      console.log('WebSocket connected');
      this.reconnectAttempts = 0;

      // 恢复订阅
      if (this.subscriptions.size > 0) {
        this.subscribe(Array.from(this.subscriptions));
      }
    });

    this.ws.addEventListener('message', (event) => {
      const message = JSON.parse(event.data);
      this.handleMessage(message);
    });

    this.ws.addEventListener('close', () => {
      console.log('WebSocket disconnected');
      this.scheduleReconnect();
    });

    this.ws.addEventListener('error', (error) => {
      console.error('WebSocket error:', error);
    });
  }

  scheduleReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnect attempts reached');
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectInterval * Math.pow(1.5, this.reconnectAttempts - 1);

    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
    setTimeout(() => this.connect(), delay);
  }

  subscribe(channels) {
    channels.forEach(ch => this.subscriptions.add(ch));
    this.send({
      type: 'subscribe',
      channels: Array.from(this.subscriptions)
    });
  }

  unsubscribe(channels) {
    channels.forEach(ch => this.subscriptions.delete(ch));
    this.send({
      type: 'unsubscribe',
      channels
    });
  }

  send(message) {
    if (this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    }
  }

  on(messageType, handler) {
    if (!this.messageHandlers.has(messageType)) {
      this.messageHandlers.set(messageType, []);
    }
    this.messageHandlers.get(messageType).push(handler);
  }

  handleMessage(message) {
    const handlers = this.messageHandlers.get(message.type) || [];
    handlers.forEach(handler => handler(message));
  }

  close() {
    if (this.ws) {
      this.ws.close();
    }
  }
}

// 使用示例
const client = new LeLampWebSocket('lelamp', {
  reconnectInterval: 3000,
  maxReconnectAttempts: 20
});

// 注册消息处理器
client.on('state_update', (message) => {
  console.log('State updated:', message.data);
});

client.on('event', (message) => {
  console.log('Event:', message.event_type);
});

client.on('notification', (message) => {
  alert(`${message.level}: ${message.message}`);
});

// 订阅频道
client.subscribe(['state', 'events', 'logs', 'notifications']);
```

## Python 客户端示例

```python
import asyncio
import json
import websockets
from typing import Set

async def lelamp_websocket_client(lamp_id: str):
    uri = f"ws://localhost:8000/api/ws/{lamp_id}"

    async with websockets.connect(uri) as websocket:
        print("Connected to LeLamp WebSocket")

        # 接收连接确认
        message = await websocket.recv()
        data = json.loads(message)
        print(f"Connected: {data['lamp_id']}")

        # 订阅频道
        subscribe_msg = {
            "type": "subscribe",
            "channels": ["state", "events", "logs"]
        }
        await websocket.send(json.dumps(subscribe_msg))

        # 接收消息
        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)

                # 处理不同类型的消息
                if data["type"] == "state_update":
                    print(f"State updated: {data['data']}")
                elif data["type"] == "event":
                    print(f"Event: {data['event_type']} - {data['data']}")
                elif data["type"] == "log":
                    print(f"Log: {data['log_entry']}")
                elif data["type"] == "notification":
                    print(f"Notification: {data['level']} - {data['message']}")
                elif data["type"] == "pong":
                    print("Pong received")

            except websockets.exceptions.ConnectionClosed:
                print("Connection closed")
                break

# 运行客户端
asyncio.run(lelamp_websocket_client("lelamp"))
```

## 最佳实践

### 1. 心跳保持

定期发送 ping 保持连接活跃：

```javascript
setInterval(() => {
  ws.send(JSON.stringify({ type: 'ping' }));
}, 30000); // 每30秒
```

### 2. 订阅管理

只订阅需要的频道，减少不必要的消息：

```javascript
// 只订阅状态和事件
ws.send(JSON.stringify({
  type: 'subscribe',
  channels: ['state', 'events']
}));
```

### 3. 错误处理

实现完整的错误处理和重连机制：

```javascript
ws.addEventListener('error', (error) => {
  console.error('WebSocket error:', error);
  // 显示错误提示
});
```

### 4. 消息队列

在离线时缓存消息，重连后处理：

```javascript
const messageQueue = [];

function queueMessage(message) {
  messageQueue.push(message);
}

function processQueue() {
  while (messageQueue.length > 0) {
    const message = messageQueue.shift();
    ws.send(JSON.stringify(message));
  }
}
```

### 5. 资源清理

在页面关闭时清理连接：

```javascript
window.addEventListener('beforeunload', () => {
  ws.close();
});
```

## 测试

可以使用 WebSocket 客户端工具测试：

### wscat

```bash
# 安装 wscat
npm install -g wscat

# 连接
wscat -c "ws://localhost:8000/api/ws/lelamp"

# 发送消息
> {"type": "ping"}
< {"type": "pong", "timestamp": "2025-03-17T12:34:56.789Z"}
```

### Postman

Postman 也支持 WebSocket 测试：
1. 创建新的 WebSocket 请求
2. 输入 URL：`ws://localhost:8000/api/ws/lelamp`
3. 连接后发送消息

## 故障排除

### 连接失败

1. 检查服务器是否运行
2. 确认 URL 格式正确
3. 检查防火墙设置

### 消息未接收

1. 确认已订阅正确的频道
2. 检查消息类型是否正确
3. 查看浏览器控制台错误

### 连接频繁断开

1. 增加心跳间隔
2. 检查网络稳定性
3. 实现自动重连机制

## API 文档

更多详细信息请参考：
- [FastAPI 文档](http://localhost:8000/docs)
- [ReDoc 文档](http://localhost:8000/redoc)
