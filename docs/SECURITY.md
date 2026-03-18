# 安全指南

## 🔒 概述

LeLamp Runtime 实现了多层次的安全防护,包括认证、授权、速率限制、安全响应头等功能。

---

## 🔐 认证安全

### JWT 令牌机制

#### 访问令牌 (Access Token)
- **有效期**: 30 分钟
- **用途**: API 认证
- **存储**: 客户端内存
- **刷新**: 过期后使用刷新令牌获取新令牌

#### 刷新令牌 (Refresh Token)
- **有效期**: 7 天
- **用途**: 获取新的访问令牌
- **存储**: 数据库持久化
- **撤销**: 使用后自动撤销旧令牌

### 密码安全

#### 密码哈希
- **算法**: bcrypt
- **工作因子**: 自动调整
- **盐值**: 每个密码唯一盐值

#### 密码要求
- **最小长度**: 6 个字符
- **最大长度**: 100 个字符
- **验证**: Pydantic EmailStr 验证邮箱格式

### 认证流程

```
1. 用户注册
   ↓
   创建 User 记录 (bcrypt 哈希密码)
   ↓
   生成 Access Token + Refresh Token
   ↓
   返回令牌给客户端

2. 访问 API
   ↓
   携带 Access Token (Authorization: Bearer <token>)
   ↓
   验证令牌有效性和权限
   ↓
   返回 API 响应

3. 令牌刷新
   ↓
   使用 Refresh Token 调用 /api/auth/refresh-token
   ↓
   验证 Refresh Token (数据库查询)
   ↓
   生成新的 Access Token + Refresh Token
   ↓
   撤销旧的 Refresh Token
   ↓
   返回新令牌
```

---

## 🛡️ 速率限制

### 滑动窗口算法

速率限制使用滑动窗口算法,确保:

- **时间窗口**: 固定时长 (60 秒)
- **最大请求数**: 窗口内允许的最大请求数
- **实时跟踪**: 记录每个请求的时间戳

### 限制级别

| 级别 | 限制 | 用途 |
|------|------|------|
| default | 100 req/min | 普通 API 端点 |
| strict | 20 req/min | 敏感操作 (注册、登录) |
| loose | 1000 req/min | 公开端点 (健康检查) |

### 识别机制

- **已认证用户**: 使用 `user_id` 作为标识符
- **匿名用户**: 使用 IP 地址作为标识符

### 响应头

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1710845620
Retry-After: 60
```

### 超限响应

**429 Too Many Requests**:
```json
{
  "error": "Rate limit exceeded",
  "limit": 100,
  "reset": 1710845620
}
```

---

## 🌐 CORS 配置

### 允许的域名

开发环境:
```
http://localhost:5173
http://localhost:3000
http://127.0.0.1:5173
http://127.0.0.1:3000
```

生产环境:
```python
# 在 lelamp/api/app.py 中配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # 生产环境应限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### CORS 头

```
Access-Control-Allow-Origin: *
Access-Control-Allow-Credentials: true
Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS
Access-Control-Allow-Headers: *
Access-Control-Expose-Headers: X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset
```

---

## 🔒 安全响应头

### 自动添加的安全头

所有 API 响应包含以下安全头:

#### X-Content-Type-Options
```
X-Content-Type-Options: nosniff
```
**作用**: 防止浏览器 MIME 类型嗅探

#### X-Frame-Options
```
X-Frame-Options: DENY
```
**作用**: 防止点击劫持攻击

#### X-XSS-Protection
```
X-XSS-Protection: 1; mode=block
```
**作用**: 启用浏览器 XSS 保护

#### Strict-Transport-Security
```
Strict-Transport-Security: max-age=31536000; includeSubDomains
```
**作用**: 强制 HTTPS 连接

#### Content-Security-Policy
```
Content-Security-Policy: default-src 'self'
```
**作用**: 防止 XSS 和数据注入

#### Referrer-Policy
```
Referrer-Policy: strict-origin-when-cross-origin
```
**作用**: 控制引用信息泄露

#### Permissions-Policy
```
Permissions-Policy: geolocation=(), microphone=(), camera=()
```
**作用**: 限制浏览器功能访问

---

## 🛡️ 输入验证

### Pydantic 模型验证

所有 API 请求使用 Pydantic 模型验证:

```python
class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr  # 自动验证邮箱格式
    password: str = Field(..., min_length=6, max_length=100)
```

### SQL 注入防护

使用 SQLAlchemy ORM 防止 SQL 注入:

- ✅ **安全**: `db.query(User).filter(User.username == username)`
- ❌ **不安全**: `db.execute(f"SELECT * FROM users WHERE username = '{username}'")`

### XSS 防护

- 所有输出经过 JSON 序列化
- HTML 内容自动转义
- CSP 策略限制脚本来源

---

## 🔐 WebSocket 安全

### 连接认证

WebSocket 支持可选 JWT 认证:

```javascript
// 匿名连接 (允许)
const ws = new WebSocket('ws://localhost:8000/api/ws/lelamp');

// 认证连接 (推荐)
const ws = new WebSocket('ws://localhost:8000/api/ws/lelamp?token=<jwt_token>');
```

### 连接验证

- 令牌有效: 允许连接,记录用户信息
- 令牌无效: 允许匿名连接,记录警告日志
- 令牌过期: 拒绝连接,返回 1008 错误码

### 连接日志

```python
logger.info(f"WebSocket 连接建立: {lamp_id}, user: {username}")
logger.warning(f"WebSocket token 无效,但允许匿名连接")
```

---

## 📊 安全最佳实践

### 1. 环境变量管理

**✅ 正确做法**:
```bash
# .env 文件 (不提交到 Git)
LIVEKIT_API_KEY=secret_key_here
DEEPSEEK_API_KEY=secret_key_here
```

**❌ 错误做法**:
```bash
# 硬编码 API Key (不安全)
API_KEY = "sk-xxxxxxxx"
```

### 2. 令牌存储

**客户端存储**:
- ✅ 内存 (推荐)
- ✅ SessionStorage (可接受)
- ❌ localStorage (不推荐,易受 XSS 攻击)

**示例**:
```javascript
// 推荐: 内存存储
let accessToken = null;

// 可接受: SessionStorage
sessionStorage.setItem('token', accessToken);
```

### 3. HTTPS 强制

生产环境必须启用 HTTPS:

```nginx
server {
    listen 443 ssl;
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    # 强制 HTTPS
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
}
```

### 4. 数据库连接

使用环境变量配置数据库:

```bash
# .env
DATABASE_URL=postgresql://user:pass@localhost/lelamp
```

避免硬编码连接字符串。

### 5. 日志安全

**不要记录敏感信息**:
- ❌ 密码
- ❌ API Key
- ❌ 令牌完整内容
- ✅ 用户 ID
- ✅ 操作类型
- ✅ 时间戳

---

## 🔍 安全审计

### 审计日志

LeLamp Runtime 记录以下安全事件:

- 用户登录成功/失败
- 令牌刷新
- 速率限制触发
- 无效令牌尝试
- 设备绑定事件

### 监控指标

关注以下安全指标:

- 登录失败率
- 速率限制触发频率
- 异常 IP 请求
- WebSocket 连接异常

---

## 🚨 事件响应

### 安全事件响应流程

1. **检测**: 监控日志发现异常
2. **分析**: 确认是否为攻击
3. **响应**:
   - 轻微: 记录日志
   - 中度: 临时封禁 IP
   - 严重: 永久封禁,通知管理员
4. **恢复**: 解除封锁,恢复正常服务

### 紧急响应

发现安全漏洞时:

1. **立即**: 部署热修复
2. **通知**: 通知用户和管理员
3. **调查**: 分析攻击向量
4. **修复**: 部署永久修复
5. **总结**: 编写事件报告

---

## 📚 相关资源

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [OWASP JWT 最佳实践](https://cheatsheetseries.owasp.org/cheatsheets/JSON_Web_Token_for_Java_Cheat_Sheet.html)
- [FastAPI 安全文档](https://fastapi.tiangolo.com/tutorial/security/)
- [SQLAlchemy 安全](https://docs.sqlalchemy.org/en/20/core/core_connections.html)

---

**最后更新**: 2026-03-19
**版本**: v2.0
