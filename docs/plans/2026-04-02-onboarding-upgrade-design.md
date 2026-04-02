# 配网首次体验升级设计文档

**日期**: 2026-04-02
**状态**: 已批准，待实施
**范围**: 前端 + 后端

---

## 背景

当前配网流程（SetupWizardView 6 步向导）存在以下商业化风险：

- WiFi 连接失败后无自动重试，用户需手动点击重试
- 连接成功后不验证网络可达性（可能连上了但没有互联网）
- 中途关闭浏览器重新进入会从头开始，无法恢复
- 错误消息英文原始异常，不友好
- auto-bind 失败会卡住整个流程
- 重启后无检测机制，用户不知道设备是否成功启动

这些问题直接影响开箱成功率，是 IoT 产品最高频投诉点。

---

## 改善目标

1. 减少用户因网络问题需要手动干预的次数
2. 让用户在每个关键步骤都知道"现在在发生什么"
3. 流程中断后可以恢复，不需重头再来
4. 任何错误都有明确的中文引导，而非技术异常信息

---

## 后端改动

### 1. WiFi 连接重试 + 网络验证

**文件**: `lelamp/api/services/wifi_manager.py`

`connect()` 方法改为支持重试：

```python
async def connect(ssid: str, password: str, max_retries: int = 3) -> dict:
    for attempt in range(1, max_retries + 1):
        # 推送进度事件 wifi_connecting(attempt, max_retries)
        result = await _try_connect(ssid, password)
        if result.success:
            # 验证网络可达性
            reachable = await _verify_network_reachability()
            return { "success": True, "network_ok": reachable }
        if attempt < max_retries:
            await asyncio.sleep(2 ** attempt)  # 2s, 4s, 8s
    return { "success": False, "message": "..." }
```

网络验证使用 `ping 223.5.5.5`（阿里 DNS，国内可达），超时 5 秒。

### 2. 配网进度 WebSocket

**文件**: `lelamp/api/routes/setup_ws.py`（新文件）
**端点**: `GET /ws/setup`

推送事件格式：
```json
{ "event": "wifi_connecting", "attempt": 1, "max_attempts": 3 }
{ "event": "wifi_connected", "ssid": "HomeWiFi" }
{ "event": "wifi_failed", "attempt": 2, "retry_in": 4 }
{ "event": "network_checking" }
{ "event": "network_ok" }
{ "event": "network_failed", "reason": "no_internet" }
{ "event": "setup_complete" }
{ "event": "rebooting", "countdown": 5 }
```

WiFi 连接流程中通过 `setup_event_bus`（asyncio.Queue）广播事件到所有连接的 WebSocket 客户端。

### 3. 恢复状态 API

**文件**: `lelamp/api/routes/system.py`（新端点）
**端点**: `GET /api/setup/recovery`

逻辑：
1. 若 `setup_completed == true` → `{ can_recover: false, reason: "already_done" }`
2. 检查当前 WiFi 连接状态（`nmcli -t -f GENERAL.STATE device show`）
3. 若已连接 WiFi → `{ can_recover: true, skip_to_step: 4, current_ssid: "..." }`
4. 否则 → `{ can_recover: false }`

### 4. auto-bind 失败降级

**文件**: `lelamp/api/routes/auth.py`

`POST /api/auth/auto-bind` 失败时：
- 不返回 HTTP 错误
- 返回 `{ "success": false, "skipped": true, "message": "绑定暂时跳过，可在设置中完成" }`
- 配网流程继续正常完成

---

## 前端改动

### 1. 进入配网时的恢复检测

**文件**: `web/src/views/SetupWizardView.vue`

`onMounted` 增加：
```typescript
const recovery = await api.get('/api/setup/recovery')
if (recovery.can_recover) {
  showRecoveryDialog(recovery.current_ssid, recovery.skip_to_step)
}
```

用户看到确认对话框："检测到设备已连接到 WiFi（HomeWiFi），是否从账号认证步骤继续？"
- "是，继续" → 跳到步骤 4
- "否，重新开始" → 从步骤 1 开始

### 2. 步骤 3：WebSocket 驱动的实时进度

**文件**: `web/src/views/SetupWizardView.vue`（步骤 3 组件）

替换静态 spinner，改为：
```
[✓] 正在发送连接请求...
[●] 正在连接 WiFi（第 1/3 次）...
[ ] 验证网络连通性...
[ ] 完成
```

自动重试时显示倒计时："连接失败，4 秒后重试（第 2/3 次）..."

3 次全部失败后才展示错误 + 重新选择网络的按钮。

### 3. 错误消息中文化

所有错误消息通过统一的 `formatApiError(err)` 函数处理：

| 原始错误 | 用户友好消息 |
|---------|------------|
| `Incorrect username or password` | 用户名或密码错误，请重新输入 |
| `Username already exists` | 该用户名已被注册，请换一个或直接登录 |
| `设备密钥未配置，无法自动绑定` | 设备绑定已跳过，后续可在设置页面完成 |
| WiFi 连接失败（3次后）| 密码可能有误，或信号较弱，建议靠近路由器重试 |
| 网络不可达 | WiFi 已连接，但无法访问互联网，请检查路由器设置 |
| 500 Internal Server Error | 设备遇到内部错误，请重启后再试 |

**文件**: `web/src/utils/errorMessages.ts`（新文件）

### 4. 步骤 5：重启后验证

**文件**: `web/src/views/SetupWizardView.vue`（步骤 5 组件）

发出重启请求后：
- 每 3 秒轮询 `GET /api/setup/status`（通过新 IP/mDNS）
- 最多等待 60 秒
- 检测到设备在线 → 自动跳转主界面
- 超时后显示："设备启动可能需要更长时间，请手动刷新页面或访问 `http://lelamp.local:8000`"

---

## 不在本次范围内

- AP 启动失败的回滚机制（方案 C）
- OnboardingManager / SetupStateManager 合并（方案 C）
- HTTPS 强制
- 配网审计日志

---

## 验收标准

1. 在 WiFi 密码错误的情况下，用户不需要手动点击重试，系统自动尝试 3 次后再报错
2. 步骤 3 有实时的文字进度，用户知道当前在做什么
3. 断开浏览器重新访问，检测到已连接 WiFi 时自动跳过到认证步骤
4. 全部错误消息为中文且包含操作建议
5. 设备重启后自动检测上线并跳转主界面，无需用户手动刷新
