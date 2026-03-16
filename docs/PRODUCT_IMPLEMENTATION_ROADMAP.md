# LeLamp App 技术实现路线图

## 🏗️ 系统架构演进

### 当前架构 (v1.0 - Web Client MVP)

```
┌─────────────────────────────────────────────────────┐
│                 Web Browser                         │
│  (LiveKit JS SDK + WebRTC)                          │
└─────────────────┬───────────────────────────────────┘
                  │ WebSocket + DTLS/SRTP
                  ↓
┌─────────────────────────────────────────────────────┐
│          LiveKit Cloud / Self-hosted                │
│  (SFU - Selective Forwarding Unit)                  │
└─────────────────┬───────────────────────────────────┘
                  │
                  ↓
┌─────────────────────────────────────────────────────┐
│      Raspberry Pi (LeLamp Runtime)                  │
│  - Python 3.12                                      │
│  - LiveKit Agents SDK                               │
│  - DeepSeek LLM + Qwen VL + Baidu Speech            │
│  - Motors / RGB / Camera Services                   │
└─────────────────────────────────────────────────────┘
```

**问题**:
- ❌ 没有中心化后端 (用户账号、设备管理、支付)
- ❌ Token 生成依赖本地 Python 脚本
- ❌ 无法支持多用户、多设备

---

### 目标架构 (v2.0 - 商业化版本)

```
┌──────────────┐          ┌──────────────┐
│  iOS App     │          │ Android App  │
│  (Flutter)   │          │  (Flutter)   │
└───────┬──────┘          └──────┬───────┘
        │                        │
        │  HTTPS (REST + WS)     │
        ↓                        ↓
┌─────────────────────────────────────────────┐
│        业务后端 (Backend API)                │
│  - Node.js / Go / Python FastAPI            │
│  - Supabase (PostgreSQL + Auth + Storage)   │
│  - Redis (Session / Cache)                  │
│                                             │
│  功能模块:                                   │
│  1. 用户认证 (User Auth)                     │
│  2. 设备管理 (Device Management)             │
│  3. Token 签发 (LiveKit Token Generation)    │
│  4. 支付系统 (Payment Gateway)               │
│  5. OTA 管理 (Firmware Management)           │
└─────────────┬───────────────────────────────┘
              │
              ↓
┌─────────────────────────────────────────────┐
│          LiveKit Cloud                      │
└─────────────┬───────────────────────────────┘
              │
              ↓
┌─────────────────────────────────────────────┐
│      Raspberry Pi (LeLamp Runtime)          │
└─────────────────────────────────────────────┘
```

---

## 📝 API 设计规范

### 1. 用户认证 API

#### POST /api/auth/register
**功能**: 用户注册 (手机号 + 验证码)

**请求体**:
```json
{
  "phone": "+86 13800138000",
  "sms_code": "123456",
  "password": "Lelamp@2026",
  "device_info": {
    "platform": "iOS",
    "version": "1.0.0",
    "model": "iPhone 14 Pro"
  }
}
```

**响应** (200 OK):
```json
{
  "success": true,
  "data": {
    "user_id": "usr_abc123",
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "expires_in": 86400
  }
}
```

**错误响应** (400 Bad Request):
```json
{
  "success": false,
  "error": {
    "code": "INVALID_SMS_CODE",
    "message": "验证码错误或已过期"
  }
}
```

---

#### POST /api/auth/login
**功能**: 用户登录

**请求体**:
```json
{
  "phone": "+86 13800138000",
  "password": "Lelamp@2026"
}
```

---

### 2. 设备管理 API

#### POST /api/devices/bind
**功能**: 绑定新设备

**请求头**:
```
Authorization: Bearer <access_token>
```

**请求体**:
```json
{
  "device_id": "lelamp_rpi4_001",
  "secret": "abc123def456",  // 从二维码扫描得到
  "device_name": "客厅台灯",
  "location": "living_room"
}
```

**响应** (200 OK):
```json
{
  "success": true,
  "data": {
    "device_id": "lelamp_rpi4_001",
    "device_name": "客厅台灯",
    "status": "online",
    "firmware_version": "1.2.0",
    "bound_at": "2026-03-16T10:30:00Z"
  }
}
```

**安全验证逻辑** (后端伪代码):
```python
def bind_device(user_id, device_id, secret):
    # 1. 验证 Secret 是否合法 (HMAC 验证)
    expected_secret = hmac.sha256(device_id, MASTER_SECRET).hexdigest()[:16]
    if secret != expected_secret:
        raise InvalidSecretError()

    # 2. 检查设备是否已被其他用户绑定
    existing_binding = db.query("SELECT * FROM device_bindings WHERE device_id = ?", device_id)
    if existing_binding:
        raise DeviceAlreadyBoundError()

    # 3. 创建绑定关系
    db.execute("""
        INSERT INTO device_bindings (user_id, device_id, role, bound_at)
        VALUES (?, ?, 'owner', NOW())
    """, user_id, device_id)

    # 4. 生成设备长期 Token (存储在设备端)
    device_token = generate_jwt({"device_id": device_id, "user_id": user_id}, expires_in=365*24*3600)
    return device_token
```

---

#### GET /api/devices
**功能**: 获取用户绑定的所有设备

**响应**:
```json
{
  "success": true,
  "data": {
    "devices": [
      {
        "device_id": "lelamp_rpi4_001",
        "device_name": "客厅台灯",
        "status": "online",
        "last_seen": "2026-03-16T10:35:00Z",
        "firmware_version": "1.2.0",
        "battery_level": 85
      },
      {
        "device_id": "lelamp_rpi4_002",
        "device_name": "书房台灯",
        "status": "offline",
        "last_seen": "2026-03-15T22:00:00Z",
        "firmware_version": "1.1.0",
        "battery_level": null
      }
    ]
  }
}
```

---

### 3. LiveKit Token 生成 API

#### POST /api/livekit/token
**功能**: 为用户生成临时 LiveKit AccessToken (用于连接实时通话)

**请求头**:
```
Authorization: Bearer <access_token>
```

**请求体**:
```json
{
  "device_id": "lelamp_rpi4_001",
  "room_name": "lelamp-room-001"  // 可选,默认自动生成
}
```

**响应**:
```json
{
  "success": true,
  "data": {
    "livekit_url": "wss://your-project.livekit.cloud",
    "livekit_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "room_name": "lelamp-room-001",
    "participant_identity": "user-12345",
    "expires_in": 3600
  }
}
```

**后端实现** (Python FastAPI):
```python
from livekit import api
import os

@app.post("/api/livekit/token")
async def generate_livekit_token(
    device_id: str,
    room_name: str | None = None,
    current_user: User = Depends(get_current_user)
):
    # 1. 验证用户是否有权访问该设备
    binding = db.query(
        "SELECT * FROM device_bindings WHERE user_id = ? AND device_id = ?",
        current_user.id, device_id
    )
    if not binding:
        raise HTTPException(403, "You don't have access to this device")

    # 2. 生成房间名 (设备专属房间)
    if not room_name:
        room_name = f"lelamp-{device_id}"

    # 3. 创建 LiveKit Token
    livekit_api_key = os.getenv("LIVEKIT_API_KEY")
    livekit_api_secret = os.getenv("LIVEKIT_API_SECRET")

    token = api.AccessToken(livekit_api_key, livekit_api_secret) \
        .with_identity(f"user-{current_user.id}") \
        .with_name(current_user.name) \
        .with_grants(api.VideoGrants(
            room_join=True,
            room=room_name,
            can_publish=True,
            can_subscribe=True,
        )) \
        .with_ttl(3600)  # 1 小时有效期

    return {
        "livekit_url": os.getenv("LIVEKIT_URL"),
        "livekit_token": token.to_jwt(),
        "room_name": room_name,
        "participant_identity": f"user-{current_user.id}",
        "expires_in": 3600
    }
```

---

### 4. 控制指令 API (可选,用于非实时控制)

#### POST /api/devices/{device_id}/command
**功能**: 向设备发送控制指令 (例如开关灯、播放动画)

**请求体**:
```json
{
  "command": "set_rgb_solid",
  "params": {
    "r": 255,
    "g": 182,
    "b": 193
  }
}
```

**响应**:
```json
{
  "success": true,
  "data": {
    "command_id": "cmd_xyz789",
    "status": "pending",
    "created_at": "2026-03-16T10:40:00Z"
  }
}
```

**实现方式**:
- 后端通过 WebSocket / MQTT 推送指令到设备
- 或设备定期轮询 `GET /api/devices/{device_id}/commands/pending`

---

### 5. OTA 更新 API

#### GET /api/ota/check
**功能**: 设备检查是否有新固件 (设备端调用)

**请求头**:
```
Authorization: Bearer <device_token>
X-Device-ID: lelamp_rpi4_001
X-Current-Version: 1.1.0
```

**响应** (有更新):
```json
{
  "has_update": true,
  "version": "1.2.0",
  "release_notes": "- 修复作业检查 Bug\n- 新增呼吸灯效果\n- 性能优化",
  "download_url": "https://cdn.lelamp.com/firmware/v1.2.0/lelamp-runtime.tar.gz",
  "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "size_bytes": 52428800,
  "mandatory": false,
  "released_at": "2026-03-15T00:00:00Z"
}
```

**响应** (无更新):
```json
{
  "has_update": false,
  "current_version": "1.2.0"
}
```

---

## 🎨 前端功能实现示例

### 1. 新增视觉功能面板 (Web Client)

**HTML 结构**:
```html
<!-- 在 index.html 的 #room-panel 中添加 -->
<div class="vision-panel">
    <h3>📸 视觉助手</h3>
    <div class="vision-controls">
        <button onclick="captureAndAsk('这是什么')">
            <span class="icon">🔍</span>
            拍照识别
        </button>
        <button onclick="checkHomework()">
            <span class="icon">📚</span>
            检查作业
        </button>
        <button onclick="sendToFeishu()">
            <span class="icon">✈️</span>
            推送飞书
        </button>
    </div>

    <!-- 结果展示区 -->
    <div id="vision-result" class="hidden">
        <img id="captured-image" alt="拍摄画面">
        <div class="result-text">
            <p id="vision-response"></p>
        </div>
    </div>
</div>
```

**JavaScript 实现**:
```javascript
async function checkHomework() {
    const resultDiv = document.getElementById('vision-result');
    const responseText = document.getElementById('vision-response');

    // 1. 显示加载状态
    responseText.textContent = '🤔 AI 正在批改中,请稍候...';
    resultDiv.classList.remove('hidden');

    // 2. 通过 Data Channel 发送指令
    await sendChat('检查作业');

    // 3. 等待 Agent 的响应 (通过 DataReceived 事件接收)
    // 响应会在 handleDataReceived() 中处理
}

// 修改 handleDataReceived 以支持视觉结果
function handleDataReceived(payload, participant) {
    const str = new TextDecoder().decode(payload);
    console.log('Received data:', str);

    try {
        const msg = JSON.parse(str);

        if (msg.type === 'chat') {
            appendMessage('agent', msg.content);
        } else if (msg.type === 'vision_result') {
            // 展示视觉结果
            document.getElementById('vision-response').textContent = msg.content;

            // 如果包含图片 Base64
            if (msg.image_base64) {
                const img = document.getElementById('captured-image');
                img.src = `data:image/jpeg;base64,${msg.image_base64}`;
            }
        }
    } catch (e) {
        appendMessage('agent', str);
    }
}
```

---

### 2. 动画按钮面板

**HTML**:
```html
<div class="animation-panel">
    <h3>🎭 动作表情</h3>
    <div class="animation-grid">
        <button class="anim-btn" onclick="playAnimation('nod')">
            <div class="anim-icon">👍</div>
            <span>点头</span>
        </button>
        <button class="anim-btn" onclick="playAnimation('shake')">
            <div class="anim-icon">👎</div>
            <span>摇头</span>
        </button>
        <button class="anim-btn" onclick="playAnimation('excited')">
            <div class="anim-icon">🎉</div>
            <span>兴奋</span>
        </button>
        <button class="anim-btn" onclick="playAnimation('sleep')">
            <div class="anim-icon">😴</div>
            <span>睡觉</span>
        </button>
        <button class="anim-btn" onclick="playAnimation('dance')">
            <div class="anim-icon">💃</div>
            <span>跳舞</span>
        </button>
        <button class="anim-btn" onclick="playAnimation('think')">
            <div class="anim-icon">🤔</div>
            <span>思考</span>
        </button>
    </div>
</div>
```

**CSS** (添加到 style.css):
```css
.animation-panel {
    margin: 2rem 0;
    padding: 1.5rem;
    background: #f9fafb;
    border-radius: 0.75rem;
}

.animation-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-top: 1rem;
}

.anim-btn {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.5rem;
    padding: 1rem;
    background: white;
    border: 2px solid #e5e7eb;
    border-radius: 0.5rem;
    transition: all 0.2s;
}

.anim-btn:hover {
    border-color: var(--primary);
    transform: scale(1.05);
}

.anim-icon {
    font-size: 2rem;
}

.anim-btn span {
    font-size: 0.875rem;
    font-weight: 500;
}
```

**JavaScript**:
```javascript
async function playAnimation(animName) {
    if (!room) {
        showToast('❌ 请先连接设备', 'error');
        return;
    }

    // 发送指令到 Agent
    const command = {
        type: 'command',
        action: 'play_recording',
        params: { recording_name: animName }
    };

    const encoder = new TextEncoder();
    const data = encoder.encode(JSON.stringify(command));
    await room.localParticipant.publishData(data, LivekitClient.DataPacket_Kind.RELIABLE);

    showToast(`🎭 正在播放动画: ${animName}`, 'info');
}
```

---

### 3. RGB 灯光调色板

**HTML**:
```html
<div class="rgb-panel">
    <h3>💡 灯光魔法</h3>

    <!-- 颜色选择器 -->
    <div class="color-picker">
        <label for="light-color">选择颜色:</label>
        <input type="color" id="light-color" value="#FFB6C1">
        <button onclick="setCustomColor()">设置</button>
    </div>

    <!-- 预设灯效 -->
    <div class="effect-grid">
        <button onclick="setRgbEffect('breathing')">💗 呼吸灯</button>
        <button onclick="setRgbEffect('rainbow')">🌈 彩虹</button>
        <button onclick="setRgbEffect('wave')">🌊 波浪</button>
        <button onclick="setRgbEffect('fire')">🔥 火焰</button>
        <button onclick="setRgbEffect('fireworks')">🎆 烟花</button>
        <button onclick="setRgbEffect('starry')">⭐ 星空</button>
    </div>
</div>
```

**JavaScript**:
```javascript
function setCustomColor() {
    const colorInput = document.getElementById('light-color');
    const hex = colorInput.value;

    // 转换 HEX 到 RGB
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);

    sendRgbCommand('set_rgb_solid', { r, g, b });
}

function setRgbEffect(effectName) {
    sendRgbCommand(`rgb_effect_${effectName}`, {});
}

async function sendRgbCommand(action, params) {
    if (!room) {
        showToast('❌ 请先连接设备', 'error');
        return;
    }

    const command = {
        type: 'command',
        action: action,
        params: params
    };

    const encoder = new TextEncoder();
    const data = encoder.encode(JSON.stringify(command));
    await room.localParticipant.publishData(data, LivekitClient.DataPacket_Kind.RELIABLE);

    showToast('💡 灯光已设置', 'success');
}
```

---

## 🔧 后端功能实现 (Agent 端)

### 修改 main.py 以支持 Data Channel 指令

**新增指令处理逻辑**:
```python
# 在 LeLamp 类中新增方法

async def handle_data_message(self, data: bytes, participant):
    """处理来自 App 的 Data Channel 消息"""
    try:
        message_str = data.decode('utf-8')
        message = json.loads(message_str)

        if message.get('type') == 'command':
            action = message.get('action')
            params = message.get('params', {})

            # 路由到对应的功能
            if action == 'play_recording':
                result = await self.play_recording(params.get('recording_name'))
            elif action.startswith('rgb_effect_'):
                effect = action.replace('rgb_effect_', '')
                result = await self._execute_rgb_effect(effect)
            elif action == 'set_rgb_solid':
                result = await self.set_rgb_solid(
                    params.get('r'), params.get('g'), params.get('b')
                )
            elif action == 'check_homework':
                result = await self.check_homework()
                # 返回结果时包含图片
                await self._send_vision_result(result)
                return
            else:
                result = f"未知指令: {action}"

            # 返回执行结果
            await self._send_chat_message(result)

    except Exception as e:
        self.logger.error(f"处理 Data Channel 消息失败: {e}")
        await self._send_chat_message(f"指令执行失败: {str(e)}")

async def _send_chat_message(self, content: str):
    """向 App 发送聊天消息"""
    message = {
        "type": "chat",
        "content": content,
        "timestamp": time.time()
    }
    data = json.dumps(message).encode('utf-8')
    await self.room.local_participant.publish_data(data)

async def _send_vision_result(self, result: str, image_base64: str = None):
    """向 App 发送视觉结果 (包含图片)"""
    message = {
        "type": "vision_result",
        "content": result,
        "image_base64": image_base64,
        "timestamp": time.time()
    }
    data = json.dumps(message).encode('utf-8')
    await self.room.local_participant.publish_data(data)
```

**在 entrypoint 中注册事件**:
```python
async def entrypoint(ctx: agents.JobContext):
    agent = LeLamp(lamp_id="lelamp")

    # 注册 Data Channel 事件
    @ctx.room.on("data_received")
    async def on_data(data: bytes, participant):
        await agent.handle_data_message(data, participant)

    # 启动 Agent
    await agent.start()
```

---

## 📱 移动端实现 (Flutter)

### 项目结构

```
lelamp_app/
├── lib/
│   ├── main.dart
│   ├── screens/
│   │   ├── login_screen.dart
│   │   ├── device_list_screen.dart
│   │   ├── device_binding_screen.dart
│   │   ├── livekit_room_screen.dart
│   │   └── settings_screen.dart
│   ├── services/
│   │   ├── auth_service.dart
│   │   ├── device_service.dart
│   │   └── livekit_service.dart
│   ├── models/
│   │   ├── user.dart
│   │   ├── device.dart
│   │   └── livekit_token.dart
│   └── widgets/
│       ├── animation_button.dart
│       ├── rgb_picker.dart
│       └── vision_panel.dart
└── pubspec.yaml
```

### 核心依赖 (pubspec.yaml)

```yaml
dependencies:
  flutter:
    sdk: flutter

  # LiveKit SDK
  livekit_client: ^2.0.0

  # 网络请求
  http: ^1.1.0
  dio: ^5.0.0

  # 状态管理
  provider: ^6.0.0

  # 本地存储
  shared_preferences: ^2.0.0

  # 二维码扫描
  mobile_scanner: ^3.0.0

  # UI 组件
  flutter_colorpicker: ^1.0.0
```

---

### LiveKit 连接实现 (Dart)

```dart
import 'package:livekit_client/livekit_client.dart';

class LiveKitService {
  Room? _room;

  Future<void> connect(String url, String token) async {
    _room = Room();

    // 监听事件
    _room!.addListener(_onRoomChanged);

    // 连接房间
    await _room!.connect(url, token);

    // 启用麦克风和摄像头
    await _room!.localParticipant.setMicrophoneEnabled(true);
    await _room!.localParticipant.setCameraEnabled(true);
  }

  void _onRoomChanged() {
    // 处理房间状态变化
  }

  Future<void> sendCommand(String action, Map<String, dynamic> params) async {
    final message = {
      'type': 'command',
      'action': action,
      'params': params,
    };

    final data = utf8.encode(jsonEncode(message));
    await _room!.localParticipant.publishData(data);
  }

  void dispose() {
    _room?.disconnect();
    _room?.dispose();
  }
}
```

---

## 📊 数据库设计 (Supabase PostgreSQL)

### 表结构

```sql
-- 1. 用户表
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    phone VARCHAR(20) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    name VARCHAR(100),
    avatar_url TEXT,
    subscription_plan VARCHAR(20) DEFAULT 'free',  -- 'free' | 'premium'
    subscription_expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 2. 设备表
CREATE TABLE devices (
    device_id VARCHAR(50) PRIMARY KEY,
    secret_hash VARCHAR(64) NOT NULL,  -- HMAC SHA256 of (device_id + master_secret)
    firmware_version VARCHAR(20),
    status VARCHAR(20) DEFAULT 'offline',  -- 'online' | 'offline'
    last_seen_at TIMESTAMP,
    battery_level INT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 3. 设备绑定表 (支持多用户共享设备)
CREATE TABLE device_bindings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    device_id VARCHAR(50) REFERENCES devices(device_id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL,  -- 'owner' | 'member' | 'guest'
    device_name VARCHAR(100),  -- 用户自定义设备名称
    location VARCHAR(50),  -- 'living_room' | 'bedroom' | 'study'
    bound_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, device_id)
);

-- 4. 使用统计表 (用于生成学习报告)
CREATE TABLE usage_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    device_id VARCHAR(50) REFERENCES devices(device_id),
    event_type VARCHAR(50) NOT NULL,  -- 'homework_check' | 'video_call' | 'animation_play'
    event_data JSONB,  -- 额外数据,如作业正确率、通话时长
    created_at TIMESTAMP DEFAULT NOW()
);

-- 5. OTA 固件版本表
CREATE TABLE firmware_versions (
    version VARCHAR(20) PRIMARY KEY,
    release_notes TEXT,
    download_url TEXT NOT NULL,
    sha256 VARCHAR(64) NOT NULL,
    size_bytes BIGINT,
    mandatory BOOLEAN DEFAULT FALSE,
    released_at TIMESTAMP DEFAULT NOW()
);

-- 创建索引
CREATE INDEX idx_bindings_user ON device_bindings(user_id);
CREATE INDEX idx_bindings_device ON device_bindings(device_id);
CREATE INDEX idx_usage_logs_user ON usage_logs(user_id);
CREATE INDEX idx_usage_logs_created ON usage_logs(created_at);
```

---

## 🚀 部署方案

### 后端部署 (Supabase + Vercel)

**选项 1: Supabase Edge Functions** (推荐)
- 无需管理服务器
- 自动扩展
- 与 Supabase Auth / Database 原生集成

**项目结构**:
```
supabase/
├── functions/
│   ├── auth-register/
│   │   └── index.ts
│   ├── device-bind/
│   │   └── index.ts
│   ├── livekit-token/
│   │   └── index.ts
│   └── ota-check/
│       └── index.ts
└── migrations/
    └── 001_initial_schema.sql
```

**部署命令**:
```bash
supabase functions deploy auth-register
supabase functions deploy device-bind
supabase functions deploy livekit-token
```

---

### 移动端发布

**iOS**:
1. 配置 App ID / Provisioning Profile
2. 在 Xcode 中 Archive 打包
3. 上传到 App Store Connect
4. 提交审核 (准备隐私政策、演示视频)

**Android**:
1. 配置签名证书
2. 构建 Release APK/AAB
3. 上传到 Google Play Console
4. 提交审核

---

## 📈 监控与分析

### 核心指标埋点

**客户端埋点**:
```dart
// 使用 Firebase Analytics
FirebaseAnalytics.instance.logEvent(
  name: 'homework_check_completed',
  parameters: {
    'device_id': deviceId,
    'accuracy': 0.85,
    'duration_seconds': 8,
  },
);
```

**后端日志**:
```python
logger.info("作业检查完成", extra={
    "user_id": user.id,
    "device_id": device.id,
    "accuracy": 0.85,
    "duration_ms": 8000,
})
```

---

> **文档版本**: v1.0
> **作者**: Claude (Tech Lead Mode)
> **最后更新**: 2026-03-16
