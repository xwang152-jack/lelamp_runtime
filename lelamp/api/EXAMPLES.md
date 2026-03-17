# LeLamp API 使用示例

本文档提供 LeLamp API 的详细使用示例，包括 Python、JavaScript 和 cURL 示例。

---

## 目录

- [Python 客户端示例](#python-客户端示例)
- [JavaScript/TypeScript 客户端示例](#javascripttypescript-客户端示例)
- [cURL 示例](#curl-示例)
- [WebSocket 示例](#websocket-示例)

---

## Python 客户端示例

### 基础设置

```python
import requests
import json
from typing import Dict, Any, Optional

# API 基础 URL
BASE_URL = "http://localhost:8000/api"

class LeLampClient:
    """LeLamp API 客户端"""

    def __init__(self, base_url: str = BASE_URL, timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        json_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """发送 HTTP 请求"""
        url = f"{self.base_url}{endpoint}"

        try:
            response = self.session.request(
                method,
                url,
                params=params,
                json=json_data,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            if hasattr(e.response, 'json'):
                error = e.response.json()
                print(f"Error details: {error.get('detail', 'Unknown error')}")
            raise

    def close(self):
        """关闭会话"""
        self.session.close()

# 使用示例
client = LeLampClient()
```

### 获取设备列表

```python
def get_devices() -> list:
    """获取所有设备"""
    try:
        data = client._request("GET", "/devices")
        devices = data.get("devices", [])
        print(f"找到 {len(devices)} 个设备:")
        for device in devices:
            print(f"  - {device['lamp_id']}: {device.get('state', 'unknown')}")
        return devices
    except Exception as e:
        print(f"获取设备列表失败: {e}")
        return []

# 使用
devices = get_devices()
```

### 获取设备状态

```python
def get_device_state(lamp_id: str) -> Dict[str, Any]:
    """获取设备状态"""
    try:
        data = client._request("GET", f"/devices/{lamp_id}/state")
        print(f"设备 {lamp_id} 状态:")
        print(f"  状态: {data['status']}")
        print(f"  对话状态: {data['conversation_state']}")
        print(f"  电机位置: {data['motor_positions']}")
        print(f"  灯光颜色: {data['light_color']}")
        print(f"  运行时间: {data['uptime_seconds']} 秒")
        return data
    except Exception as e:
        print(f"获取设备状态失败: {e}")
        return {}

# 使用
state = get_device_state("lelamp_001")
```

### 发送设备命令

```python
def send_command(
    lamp_id: str,
    command_type: str,
    action: str,
    params: Optional[Dict] = None
) -> bool:
    """发送设备命令"""
    try:
        command_data = {
            "type": command_type,
            "action": action,
            "params": params or {}
        }
        data = client._request("POST", f"/devices/{lamp_id}/command", json_data=command_data)
        print(f"命令已发送: {data['command_id']}")
        return data['success']
    except Exception as e:
        print(f"发送命令失败: {e}")
        return False

# 使用示例

# 1. 移动电机
send_command(
    "lelamp_001",
    "motor_move",
    "move_joint",
    {
        "joint_name": "base_yaw",
        "position": 45.0,
        "speed": 50
    }
)

# 2. 设置颜色
send_command(
    "lelamp_001",
    "rgb_set",
    "set_color",
    {"r": 255, "g": 0, "b": 0}
)

# 3. 播放动作
send_command(
    "lelamp_001",
    "play_recording",
    "play",
    {"name": "wave_hello"}
)

# 4. 设置音量
send_command(
    "lelamp_001",
    "set_volume",
    "set",
    {"volume": 80}
)
```

### 获取对话记录

```python
def get_conversations(
    lamp_id: str,
    skip: int = 0,
    limit: int = 50
) -> list:
    """获取对话记录"""
    try:
        params = {"skip": skip, "limit": limit}
        data = client._request("GET", f"/devices/{lamp_id}/conversations", params=params)
        conversations = data.get("conversations", [])
        print(f"找到 {data['total']} 条对话记录:")
        for conv in conversations[:5]:  # 只显示前 5 条
            print(f"  [{conv['timestamp']}] {conv['user_input']}")
        return conversations
    except Exception as e:
        print(f"获取对话记录失败: {e}")
        return []

# 使用
conversations = get_conversations("lelamp_001", skip=0, limit=10)
```

### 获取操作日志

```python
def get_operations(
    lamp_id: str,
    skip: int = 0,
    limit: int = 50,
    hours: int = 24
) -> list:
    """获取操作日志"""
    try:
        params = {"skip": skip, "limit": limit, "hours": hours}
        data = client._request("GET", f"/devices/{lamp_id}/operations", params=params)
        operations = data.get("operations", [])
        print(f"最近 {hours} 小时内的操作: {data['total']} 条")
        for op in operations[:10]:  # 只显示前 10 条
            status = "✓" if op['success'] else "✗"
            print(f"  {status} [{op['timestamp']}] {op['operation_type']}/{op['action']}")
        return operations
    except Exception as e:
        print(f"获取操作日志失败: {e}")
        return []

# 使用
operations = get_operations("lelamp_001", hours=24)
```

### 获取健康状态

```python
def get_health(lamp_id: str) -> Dict[str, Any]:
    """获取设备健康状态"""
    try:
        data = client._request("GET", f"/devices/{lamp_id}/health")
        print(f"设备 {lamp_id} 健康状态:")
        print(f"  整体状态: {data['overall_status']}")
        for motor in data['motors']:
            print(f"  电机 {motor['name']}:")
            print(f"    温度: {motor['temperature']}°C")
            print(f"    电压: {motor['voltage']}V")
            print(f"    负载: {motor['load']*100:.1f}%")
            print(f"    状态: {motor['status']}")
        return data
    except Exception as e:
        print(f"获取健康状态失败: {e}")
        return {}

# 使用
health = get_health("lelamp_001")
```

### 获取统计数据

```python
def get_statistics(lamp_id: str, days: int = 7) -> Dict[str, Any]:
    """获取设备统计"""
    try:
        params = {"days": days}
        data = client._request("GET", f"/devices/{lamp_id}/statistics", params=params)
        print(f"设备 {lamp_id} 最近 {days} 天统计:")
        print(f"  总操作数: {data['total_operations']}")
        print(f"  成功率: {data['success_rate']*100:.1f}%")
        print(f"  平均耗时: {data['avg_duration_ms']:.1f}ms")
        print(f"  最常见操作: {data['most_common_operation']}")
        print(f"  操作分布:")
        for op_type, count in data['operation_counts'].items():
            print(f"    {op_type}: {count}")
        return data
    except Exception as e:
        print(f"获取统计数据失败: {e}")
        return {}

# 使用
stats = get_statistics("lelamp_001", days=7)
```

### 完整示例

```python
def monitor_device(lamp_id: str):
    """监控设备状态"""
    print(f"开始监控设备 {lamp_id}...")

    try:
        # 获取设备状态
        state = get_device_state(lamp_id)
        if not state:
            print("设备离线或不存在")
            return

        # 获取健康状态
        health = get_health(lamp_id)
        if health['overall_status'] != 'healthy':
            print(f"警告: 设备健康状态为 {health['overall_status']}")

        # 获取最近操作
        operations = get_operations(lamp_id, hours=1)
        failed_ops = [op for op in operations if not op['success']]
        if failed_ops:
            print(f"警告: 发现 {len(failed_ops)} 个失败操作")

        # 获取统计数据
        stats = get_statistics(lamp_id, days=1)
        if stats['success_rate'] < 0.9:
            print(f"警告: 成功率低于 90% ({stats['success_rate']*100:.1f}%)")

        print("设备监控完成")

    finally:
        client.close()

# 使用
monitor_device("lelamp_001")
```

---

## JavaScript/TypeScript 客户端示例

### 基础设置

```typescript
// API 基础 URL
const BASE_URL = 'http://localhost:8000/api';

class LeLampClient {
    private baseUrl: string;
    private timeout: number;

    constructor(baseUrl: string = BASE_URL, timeout: number = 30000) {
        this.baseUrl = baseUrl;
        this.timeout = timeout;
    }

    private async request(
        method: string,
        endpoint: string,
        params?: Record<string, any>,
        json?: Record<string, any>
    ): Promise<any> {
        const url = new URL(endpoint, this.baseUrl);

        if (params) {
            Object.keys(params).forEach(key =>
                url.searchParams.append(key, params[key].toString())
            );
        }

        const options: RequestInit = {
            method,
            headers: {
                'Content-Type': 'application/json',
            },
        };

        if (json) {
            options.body = JSON.stringify(json);
        }

        try {
            const response = await fetch(url.toString(), options);

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Request failed');
            }

            return await response.json();
        } catch (error) {
            console.error('Request error:', error);
            throw error;
        }
    }
}

// 使用示例
const client = new LeLampClient();
```

### Fetch API 示例

```typescript
// 获取设备列表
async function getDevices(): Promise<void> {
    try {
        const response = await fetch(`${BASE_URL}/devices`);
        const data = await response.json();
        console.log('设备列表:', data.devices);
    } catch (error) {
        console.error('获取设备列表失败:', error);
    }
}

// 获取设备状态
async function getDeviceState(lampId: string): Promise<void> {
    try {
        const response = await fetch(`${BASE_URL}/devices/${lampId}/state`);
        const data = await response.json();
        console.log('设备状态:', data);
    } catch (error) {
        console.error('获取设备状态失败:', error);
    }
}

// 发送命令
async function sendCommand(
    lampId: string,
    commandType: string,
    action: string,
    params?: Record<string, any>
): Promise<void> {
    try {
        const response = await fetch(`${BASE_URL}/devices/${lampId}/command`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                type: commandType,
                action,
                params: params || {},
            }),
        });

        const data = await response.json();
        console.log('命令已发送:', data.command_id);
    } catch (error) {
        console.error('发送命令失败:', error);
    }
}

// 使用示例
await getDevices();
await getDeviceState('lelamp_001');
await sendCommand('lelamp_001', 'motor_move', 'move_joint', {
    joint_name: 'base_yaw',
    position: 45.0,
    speed: 50
});
```

### React 集成

```typescript
// hooks/useLeLampAPI.ts
import { useState, useEffect } from 'react';

interface DeviceState {
    lamp_id: string;
    status: string;
    conversation_state: string;
    motor_positions: Record<string, number>;
    light_color: { r: number; g: number; b: number };
    camera_active: boolean;
    uptime_seconds: number;
}

export function useDeviceState(lampId: string) {
    const [state, setState] = useState<DeviceState | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<Error | null>(null);

    useEffect(() => {
        const fetchState = async () => {
            try {
                const response = await fetch(`${BASE_URL}/devices/${lampId}/state`);
                const data = await response.json();
                setState(data);
            } catch (err) {
                setError(err as Error);
            } finally {
                setLoading(false);
            }
        };

        fetchState();

        // 每 5 秒刷新一次
        const interval = setInterval(fetchState, 5000);
        return () => clearInterval(interval);
    }, [lampId]);

    return { state, loading, error };
}

// components/DeviceDashboard.tsx
import React from 'react';
import { useDeviceState } from '../hooks/useLeLampAPI';

export function DeviceDashboard({ lampId }: { lampId: string }) {
    const { state, loading, error } = useDeviceState(lampId);

    if (loading) return <div>加载中...</div>;
    if (error) return <div>错误: {error.message}</div>;
    if (!state) return <div>设备离线</div>;

    return (
        <div className="device-dashboard">
            <h2>设备 {lampId}</h2>
            <p>状态: {state.status}</p>
            <p>对话状态: {state.conversation_state}</p>
            <p>运行时间: {Math.floor(state.uptime_seconds / 60)} 分钟</p>

            <div className="motor-positions">
                <h3>电机位置</h3>
                {Object.entries(state.motor_positions).map(([name, pos]) => (
                    <div key={name}>
                        {name}: {pos}°
                    </div>
                ))}
            </div>

            <div className="light-color">
                <h3>灯光颜色</h3>
                <div
                    style={{
                        backgroundColor: `rgb(${state.light_color.r}, ${state.light_color.g}, ${state.light_color.b})`,
                        width: 100,
                        height: 100
                    }}
                />
            </div>
        </div>
    );
}
```

### Vue.js 集成

```typescript
// composables/useLeLampAPI.ts
import { ref, onMounted } from 'vue';

export function useDeviceState(lampId: string) {
    const state = ref<any>(null);
    const loading = ref(true);
    const error = ref<Error | null>(null);

    const fetchState = async () => {
        try {
            const response = await fetch(`${BASE_URL}/devices/${lampId}/state`);
            const data = await response.json();
            state.value = data;
        } catch (err) {
            error.value = err as Error;
        } finally {
            loading.value = false;
        }
    };

    onMounted(() => {
        fetchState();
        setInterval(fetchState, 5000);
    });

    return { state, loading, error };
}

// components/DeviceCard.vue
<template>
    <div class="device-card">
        <div v-if="loading">加载中...</div>
        <div v-else-if="error">错误: {{ error.message }}</div>
        <div v-else-if="state">
            <h3>设备 {{ lampId }}</h3>
            <p>状态: {{ state.status }}</p>
            <p>对话状态: {{ state.conversation_state }}</p>
            <div class="color-preview" :style="colorStyle"></div>
        </div>
    </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useDeviceState } from '../composables/useLeLampAPI';

const props = defineProps<{
    lampId: string;
}>();

const { state, loading, error } = useDeviceState(props.lampId);

const colorStyle = computed(() => ({
    backgroundColor: `rgb(${state.value?.light_color.r}, ${state.value?.light_color.g}, ${state.value?.light_color.b})`,
    width: '100px',
    height: '100px'
}));
</script>
```

---

## cURL 示例

### 基础请求

```bash
# 设置基础 URL
BASE_URL="http://localhost:8000/api"

# 获取设备列表
curl -X GET "$BASE_URL/devices"

# 获取设备状态
curl -X GET "$BASE_URL/devices/lelamp_001/state"

# 健康检查
curl -X GET "http://localhost:8000/health"
```

### 发送命令

```bash
# 移动电机
curl -X POST "$BASE_URL/devices/lelamp_001/command" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "motor_move",
    "action": "move_joint",
    "params": {
      "joint_name": "base_yaw",
      "position": 45.0,
      "speed": 50
    }
  }'

# 设置颜色
curl -X POST "$BASE_URL/devices/lelamp_001/command" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "rgb_set",
    "action": "set_color",
    "params": {
      "r": 255,
      "g": 0,
      "b": 0
    }
  }'

# 播放动作
curl -X POST "$BASE_URL/devices/lelamp_001/command" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "play_recording",
    "action": "play",
    "params": {
      "name": "wave_hello"
    }
  }'

# 设置音量
curl -X POST "$BASE_URL/devices/lelamp_001/command" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "set_volume",
    "action": "set",
    "params": {
      "volume": 80
    }
  }'
```

### 查询数据

```bash
# 获取对话记录（分页）
curl -X GET "$BASE_URL/devices/lelamp_001/conversations?skip=0&limit=10"

# 获取操作日志（时间窗口）
curl -X GET "$BASE_URL/devices/lelamp_001/operations?skip=0&limit=50&hours=24"

# 获取健康状态
curl -X GET "$BASE_URL/devices/lelamp_001/health"

# 获取统计数据
curl -X GET "$BASE_URL/devices/lelamp_001/statistics?days=7"
```

### 格式化输出

```bash
# 使用 jq 格式化 JSON 输出
curl -s -X GET "$BASE_URL/devices/lelamp_001/state" | jq '.'

# 提取特定字段
curl -s -X GET "$BASE_URL/devices/lelamp_001/state" | jq '.conversation_state'

# 过滤数组
curl -s -X GET "$BASE_URL/devices/lelamp_001/operations" | jq '.operations[] | select(.success == false)'
```

### 高级用法

```bash
# 认证（如果需要）
curl -X GET "$BASE_URL/devices/lelamp_001/state" \
  -H "Authorization: Bearer YOUR_TOKEN"

# 超时设置
curl -X GET "$BASE_URL/devices/lelamp_001/state" \
  --max-time 10

# 详细输出（调试）
curl -X GET "$BASE_URL/devices/lelamp_001/state" \
  --verbose

# 保存响应到文件
curl -X GET "$BASE_URL/devices/lelamp_001/state" \
  -o response.json

# 从文件读取请求体
curl -X POST "$BASE_URL/devices/lelamp_001/command" \
  -H "Content-Type: application/json" \
  -d @command.json
```

---

## WebSocket 示例

### Python WebSocket 客户端

```python
import asyncio
import json
import websockets

async def connect_websocket(lamp_id: str):
    """连接到 WebSocket"""
    uri = f"ws://localhost:8000/api/ws/{lamp_id}"

    try:
        async with websockets.connect(uri) as ws:
            print(f"已连接到 {uri}")

            # 接收连接确认
            message = await ws.recv()
            data = json.loads(message)
            print(f"服务端消息: {data['message']}")

            # 订阅频道
            await ws.send(json.dumps({
                "type": "subscribe",
                "channels": ["state", "events", "logs"]
            }))

            # 接收订阅确认
            message = await ws.recv()
            data = json.loads(message)
            print(f"已订阅: {data['channels']}")

            # 持续接收消息
            async for message in ws:
                data = json.loads(message)
                handle_message(data)

    except websockets.exceptions.WebSocketException as e:
        print(f"WebSocket 错误: {e}")

def handle_message(data: dict):
    """处理接收到的消息"""
    msg_type = data.get('type')

    if msg_type == 'state_update':
        print(f"状态更新: {data['data']['conversation_state']}")
    elif msg_type == 'event':
        print(f"事件: {data['event_type']}")
    elif msg_type == 'log':
        print(f"日志: {data['log_entry']['operation_type']}")
    elif msg_type == 'notification':
        print(f"通知: {data['message']}")
    elif msg_type == 'pong':
        print("心跳响应")

# 使用
asyncio.run(connect_websocket("lelamp_001"))
```

### JavaScript WebSocket 客户端

```typescript
class LeLampWebSocket {
    private ws: WebSocket | null = null;
    private lampId: string;
    private url: string;
    private reconnectAttempts = 0;
    private maxReconnectAttempts = 5;

    constructor(lampId: string, url: string = 'ws://localhost:8000/api/ws') {
        this.lampId = lampId;
        this.url = `${url}/${lampId}`;
    }

    connect(): void {
        this.ws = new WebSocket(this.url);

        this.ws.onopen = () => {
            console.log('WebSocket 已连接');
            this.reconnectAttempts = 0;

            // 订阅频道
            this.send({
                type: 'subscribe',
                channels: ['state', 'events', 'logs']
            });
        };

        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket 错误:', error);
        };

        this.ws.onclose = () => {
            console.log('WebSocket 已断开');
            this.reconnect();
        };

        // 心跳
        setInterval(() => {
            this.send({ type: 'ping' });
        }, 30000);
    }

    private reconnect(): void {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
            console.log(`${delay}ms 后重连...`);
            setTimeout(() => this.connect(), delay);
        }
    }

    send(data: any): void {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(data));
        }
    }

    private handleMessage(data: any): void {
        switch (data.type) {
            case 'connected':
                console.log('连接确认:', data.message);
                break;
            case 'subscription_confirmed':
                console.log('已订阅:', data.channels);
                break;
            case 'state_update':
                console.log('状态更新:', data.data);
                break;
            case 'event':
                console.log('事件:', data.event_type, data.data);
                break;
            case 'log':
                console.log('日志:', data.log_entry);
                break;
            case 'notification':
                console.log('通知:', data.message);
                break;
            case 'pong':
                console.log('心跳响应');
                break;
            default:
                console.log('未知消息类型:', data.type);
        }
    }

    close(): void {
        if (this.ws) {
            this.ws.close();
        }
    }
}

// 使用示例
const ws = new LeLampWebSocket('lelamp_001');
ws.connect();
```

### React WebSocket Hook

```typescript
// hooks/useWebSocket.ts
import { useEffect, useRef, useState } from 'react';

interface WebSocketMessage {
    type: string;
    data?: any;
    [key: string]: any;
}

export function useWebSocket(lampId: string) {
    const [messages, setMessages] = useState<WebSocketMessage[]>([]);
    const [connected, setConnected] = useState(false);
    const wsRef = useRef<WebSocket | null>(null);

    useEffect(() => {
        const url = `ws://localhost:8000/api/ws/${lampId}`;
        wsRef.current = new WebSocket(url);

        wsRef.current.onopen = () => {
            console.log('WebSocket 已连接');
            setConnected(true);

            // 订阅频道
            wsRef.current?.send(JSON.stringify({
                type: 'subscribe',
                channels: ['state', 'events', 'logs']
            }));
        };

        wsRef.current.onmessage = (event) => {
            const data = JSON.parse(event.data);
            setMessages(prev => [...prev, data]);
        };

        wsRef.current.onclose = () => {
            console.log('WebSocket 已断开');
            setConnected(false);
        };

        return () => {
            wsRef.current?.close();
        };
    }, [lampId]);

    const sendMessage = (message: any) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify(message));
        }
    };

    return { messages, connected, sendMessage };
}

// components/RealtimeMonitor.tsx
import React from 'react';
import { useWebSocket } from '../hooks/useWebSocket';

export function RealtimeMonitor({ lampId }: { lampId: string }) {
    const { messages, connected } = useWebSocket(lampId);

    return (
        <div className="realtime-monitor">
            <h3>实时监控</h3>
            <p>状态: {connected ? '已连接' : '未连接'}</p>

            <div className="messages">
                {messages.slice(-10).map((msg, i) => (
                    <div key={i} className="message">
                        <strong>{msg.type}</strong>: {JSON.stringify(msg.data)}
                    </div>
                ))}
            </div>
        </div>
    );
}
```

---

## 错误处理

### Python 错误处理

```python
def safe_api_call(func):
    """API 调用装饰器，添加错误处理"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.Timeout:
            print("请求超时")
        except requests.exceptions.ConnectionError:
            print("连接失败")
        except requests.exceptions.HTTPError as e:
            print(f"HTTP 错误: {e.response.status_code}")
            print(f"响应: {e.response.json()}")
        except Exception as e:
            print(f"未知错误: {e}")
    return wrapper

@safe_api_call
def get_device_state_safe(lamp_id: str):
    return get_device_state(lamp_id)
```

### JavaScript 错误处理

```typescript
async function safeApiCall<T>(
    fn: () => Promise<T>
): Promise<T | null> {
    try {
        return await fn();
    } catch (error) {
        if (error instanceof TypeError) {
            console.error('网络错误:', error.message);
        } else {
            console.error('API 错误:', error);
        }
        return null;
    }
}

// 使用
const state = await safeApiCall(() => getDeviceState('lelamp_001'));
```

---

## 最佳实践

### 1. 连接池

```python
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

session = requests.Session()

# 配置重试
retry = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[500, 502, 503, 504]
)

adapter = HTTPAdapter(
    max_retries=retry,
    pool_connections=10,
    pool_maxsize=20
)

session.mount('http://', adapter)
session.mount('https://', adapter)
```

### 2. 缓存

```python
from functools import lru_cache
import time

@lru_cache(maxsize=100)
def get_device_state_cached(lamp_id: str, cache_time: int):
    """带缓存的设备状态查询"""
    return get_device_state(lamp_id)

def get_device_state_with_ttl(lamp_id: str, ttl: int = 60):
    """带 TTL 的缓存"""
    cache_key = f"state:{lamp_id}"
    current_time = time.time()

    if cache_key in cache and current_time - cache[cache_key]['time'] < ttl:
        return cache[cache_key]['data']

    data = get_device_state(lamp_id)
    cache[cache_key] = {'data': data, 'time': current_time}
    return data
```

### 3. 批量操作

```python
def batch_send_commands(lamp_id: str, commands: list) -> list:
    """批量发送命令"""
    results = []
    for cmd in commands:
        result = send_command(
            lamp_id,
            cmd['type'],
            cmd['action'],
            cmd.get('params')
        )
        results.append(result)
    return results

# 使用
commands = [
    {'type': 'motor_move', 'action': 'move_joint', 'params': {...}},
    {'type': 'rgb_set', 'action': 'set_color', 'params': {...}},
    {'type': 'play_recording', 'action': 'play', 'params': {...}}
]
results = batch_send_commands('lelamp_001', commands)
```

---

## 更多资源

- [完整 API 文档](./API_DOCUMENTATION.md)
- [部署指南](./DEPLOYMENT_GUIDE.md)
- [测试文档](./TESTING.md)
- GitHub: https://github.com/your-org/lelamp_runtime
