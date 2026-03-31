# 功能说明文档

## 🎯 概述

LeLamp Runtime v3.2 提供完整的企业级功能,包括对话式 AI、视觉识别、动作控制、灯光效果、**边缘推理**、**记忆系统**和 RESTful API 系统。

---

## 🧠 记忆系统 🆕

让 LeLamp 具备长期记忆能力,能够跨会话记住用户偏好、重要事实和对话上下文。灵感来自 [nanobot](https://github.com/HKUDS/nanobot) 的双层记忆架构。

### 功能特性

| 功能 | 说明 |
|------|------|
| **长期记忆** | 自动注入 system prompt,每次对话都能回忆 |
| **对话摘要** | LLM 自动提取关键信息并生成摘要 |
| **Agent 工具** | `save_memory` / `recall_memory` / `forget_memory` |
| **Token 预算** | 按重要性排序,自动控制注入量 |
| **自动整合** | 对话达 10 轮后后台自动整合 |
| **优雅降级** | 记忆系统失败不影响正常对话 |

### 启用方式

记忆系统默认启用:

```bash
# 默认已启用,设置环境变量禁用
export LELAMP_MEMORY_ENABLED=0
```

### Agent 工具

LeLamp 在对话中可以主动使用以下工具:

```
用户: 我叫小明,今年上三年级

LeLamp: (内部调用 save_memory) 已记住: 我叫小明,今年上三年级

# 后续对话
用户: 你还记得我叫什么吗?
LeLamp: (内部调用 recall_memory) 当然记得! 你叫小明,今年上三年级~
```

### 记忆分类

| 分类 | 说明 | 示例 |
|------|------|------|
| `preference` | 用户偏好 | "喜欢暖色调灯光"、"喜欢蓝色" |
| `fact` | 事实信息 | "今年上三年级"、"养了一只猫" |
| `relationship` | 关系信息 | "是妈妈的儿子"、"最好的朋友是小红" |
| `context` | 上下文 | "最近在学乘法"、"作业经常做错" |
| `general` | 通用 | 其他信息 |

### 记忆整合流程

```
用户对话持续进行...
    ↓
对话轮次达到 10 轮
    ↓
后台触发整合 (asyncio.create_task, 不阻塞对话)
    ↓
DeepSeek LLM 分析对话:
  1. 提取值得记住的信息 → 写入长期记忆
  2. 生成对话摘要 → 写入对话摘要
  3. 提取话题标签 → 用于后续搜索
    ↓
新记忆注入 system prompt → 下次对话自动生效
```

### 配置选项

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `LELAMP_MEMORY_ENABLED` | `1` | 启用/禁用记忆系统 |
| `LELAMP_MEMORY_TOKEN_BUDGET` | `400` | system prompt 中记忆最大 token 数 |
| `LELAMP_MEMORY_CONSOLIDATION_MIN_TURNS` | `10` | 触发整合的最少对话轮数 |
| `LELAMP_MEMORY_CONSOLIDATION_COOLDOWN_S` | `300` | 整合冷却时间 (秒) |
| `LELAMP_MEMORY_MAX_CONTENT_LENGTH` | `500` | 单条记忆最大字符数 |

### 优雅降级

记忆系统设计了多层降级保障:

1. **初始化失败** → 记录警告,Agent 正常运行无记忆
2. **数据库操作失败** → 返回空结果,对话不受影响
3. **整合 API 不可用** → 静默失败,不阻塞对话
4. **工具调用失败** → 返回友好提示 ("记忆功能未启用")
5. **Prompt 构建失败** → 回退到静态 system prompt

---

## 🆕 边缘推理 (Edge Vision) 🆕

基于 MediaPipe 的本地 AI 推理功能，实现低延迟、隐私保护的视觉交互。

### 功能特性

| 功能 | 说明 | 延迟 |
|------|------|------|
| **人脸检测** | 用户在场检测、自动唤醒/休眠 | < 50ms |
| **手势追踪** | 8种手势识别、手势控制 | < 100ms |
| **物体检测** | 80类COCO物体本地识别 | < 300ms |
| **混合推理** | 智能路由本地/云端 | 自动选择 |

### 启用方式

```bash
# 设置环境变量启用边缘视觉
export LELAMP_EDGE_VISION_ENABLED=1

# 可选：安装 MediaPipe（完整功能）
# 先装小依赖
uv pip install numpy opencv-contrib-python --index-url https://pypi.tuna.tsinghua.edu.cn/simple/
# 再装大依赖
uv pip install scipy jaxlib --index-url https://pypi.tuna.tsinghua.edu.cn/simple/
# 最后装 mediapipe
uv pip install mediapipe --index-url https://pypi.tuna.tsinghua.edu.cn/simple/
```

### 支持的手势

| 手势 | 触发动作 |
|------|----------|
| 👍 点赞 | 点头 (nod) |
| 👎 踩 | 摇头 (shake) |
| ✌️ 耶 | 兴奋 (excited) |
| 👋 挥手 | 开关灯 |
| ✊ 握拳 | 静音/取消静音 |
| 👆 指向 | 台灯看向指定方向 |
| 👌 OK | 确认 |
| 👐 张开 | 恢复默认 |

### 混合推理策略

```
用户问："这是什么？"
    ↓
HybridVisionService 分析查询复杂度
    ↓
┌─────────────────────────────────────┐
│ QueryComplexity.SIMPLE              │
│ → 本地 MediaPipe 物体检测 (< 200ms) │
│ → 能识别？直接回答                   │
└─────────────────────────────────────┘
    ↓ 无法识别
┌─────────────────────────────────────┐
│ QueryComplexity.COMPLEX             │
│ → 云端 Qwen VL (3-8s)               │
│ → 详细回答                          │
└─────────────────────────────────────┘
```

### Agent 工具

边缘推理提供以下 Agent 工具：

```python
# 快速识别物体（本地推理）
await quick_identify()

# 检测手势
await detect_gesture()

# 检测用户在场
await check_presence()

# 获取边缘视觉统计
await get_edge_vision_stats()
```

### 配置选项

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `LELAMP_EDGE_VISION_ENABLED` | `0` | 启用边缘视觉 |
| `LELAMP_EDGE_VISION_PREFER_LOCAL` | `1` | 优先本地推理 |
| `LELAMP_EDGE_VISION_LOCAL_THRESHOLD` | `0.7` | 本地置信度阈值 |

### 优雅降级

- **MediaPipe 不可用**: 自动降级到 NoOp 模式，不影响其他功能
- **物体检测模型缺失**: 降级到云端 Qwen VL
- **云端服务不可用**: 使用本地结果（如有）

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
uv run pytest tests/integration/test_auth_routes.py

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


**最后更新**: 2026-03-31
**版本**: v3.2 (新增记忆系统模块)
