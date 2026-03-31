# LiveKit Token 商业化管理机制

在 LeLamp 产品商业化后，给用户分发和管理 LiveKit Token 是一个核心的安全与架构问题。以下是关于**不同用户的 Token 隔离**和 **Token 的有效期**的详细说明：

## 1. 怎么确保每个用户获得不同的 Token（隔离机制）

在商业化部署中，**绝不能把 `LIVEKIT_API_KEY` 和 `LIVEKIT_API_SECRET` 写在前端 App 里**，也不能通过手动脚本给所有人生成一样的 Token。

**标准的商业化流程如下**：

1. **云端业务服务器 (你的后端)**：你需要部署一个自己的云端服务器（比如基于 Node.js/Python FastAPI）。这个服务器持有 `LIVEKIT_API_SECRET`。
2. **设备绑定 (Device/Room)**：每个售出的 LeLamp 硬件在出厂或激活时，都有一个唯一的 `lamp_id`（比如 MAC 地址或序列号），这个 `lamp_id` 就是 LiveKit 的 **Room Name**（房间号）。
3. **用户登录认证**：用户打开手机 App，通过账号密码或微信/手机号登录你的云端服务器。云端服务器知道该用户绑定了哪台设备（哪个 `lamp_id`）。
4. **动态请求 Token**：
   - 用户在 App 里点击“连接台灯”。
   - App 向你的云端服务器发送 API 请求（例如 `GET /api/get_livekit_token?device_id=xxx`）。
   - 云端服务器校验用户的身份和设备绑定权限。
   - **如果验证通过，云端服务器使用该设备的 `lamp_id` 作为 Room，用户的 `user_id` 作为 Identity，动态生成一个 JWT Token，并返回给 App。**
   - App 拿着这个 Token 去连接 LiveKit Server。

**通过这种方式，实现了严格的隔离：**
- A 用户的 Token 里限制了 `room=lamp_A`，他绝对无法连接到 B 用户的 `lamp_B` 房间。
- 只有你的云端服务器掌握 Secret，别人无法伪造 Token。

---

## 2. 这个 Token 有有效期吗？

**有的，LiveKit 的 Token 默认和设计上都是带有有效期的。**

- **默认有效期**：如果不特殊指定，LiveKit 的 AccessToken 默认有效期通常是 **6 小时**。
- **自定义有效期**：在生成 Token 的代码中，你可以通过 `.with_ttl()` 明确设置有效期。

### 代码示例（Python）

```python
# 例如在你的云端服务器生成 Token 的代码中
from datetime import timedelta
from livekit import api

token = api.AccessToken(api_key, api_secret) \
    .with_identity(user_id) \
    .with_name(user_name) \
    .with_ttl(timedelta(hours=2)) \  # <--- 明确设置 2 小时过期
    .with_grants(api.VideoGrants(
        room_join=True,
        room=lamp_id, # 限制只能加入他自己的台灯房间
    ))
```

### Token 过期了怎么办？

由于 LiveKit 是基于 WebSocket 的长连接：
1. **连接建立前**：如果 Token 已过期，LiveKit Server 会拒绝建立连接。
2. **连接保持中**：只要连接成功建立了，哪怕中途 Token 过期，**正在进行的音视频连接不会被强行断开**（除非你的服务器主动调用 API 踢人）。
3. **重连机制**：如果网络波动导致断开，App 尝试重连时，如果原 Token 已过期，App 需要重新调用你的云端接口（静默刷新）获取一个新的 Token，然后再去连接 LiveKit。

## 总结

商业化后，你需要一个简单的**业务中台**来负责用户鉴权和设备绑定关系的映射，并由这个中台为合法的请求动态、实时地颁发具有短时效性（如 2 小时）和严格房间限制的 LiveKit Token。这就构成了商业化落地的安全基石。
