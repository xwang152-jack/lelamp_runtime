# AP 热点配网优化实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 优化 LeLamp 首次配网流程，实现 Captive Portal 自动跳转、4 步简化流程和连接成功反馈。

**Architecture:** 在现有 AP 热点模式基础上，添加 Captive Portal 中间件拦截手机连通性检测请求并重定向到配网页面。简化前端 SetupWizardView 从 6 步改为 4 步，合并 WiFi 选择和密码输入步骤，移除首次配网的注册/登录步骤。后端新增 `/setup/complete-and-connect` 端点，完成配置后直接切换到 WiFi 客户端模式并触发 LED/语音反馈。

**Tech Stack:** Python/FastAPI (Captive Portal 中间件), Vue 3 + Element Plus (前端), dnsmasq (DNS 劫持)

---

### Task 1: Captive Portal 中间件

**Files:**
- Create: `lelamp/api/middleware/captive_portal.py`
- Modify: `lelamp/api/app.py:1-20` (import + 注册中间件)

**Step 1: 创建 Captive Portal 中间件**

```python
# lelamp/api/middleware/captive_portal.py
"""
Captive Portal 中间件

AP 模式下拦截手机连通性检测请求，重定向到配网页面。
支持 iOS/Android/Windows/Mac 的自动检测机制。
"""
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response
import re

logger = logging.getLogger(__name__)

# 各平台的 Captive Portal 检测 URL
CAPTIVE_PORTAL_HOSTS = {
    # Apple
    "captive.apple.com",
    "www.apple.com",
    "apple.com",
    # Android
    "connectivitycheck.gstatic.com",
    "clients3.google.com",
    "connectivitycheck.android.com",
    # Windows
    "www.msftconnecttest.com",
    "msftconnecttest.com",
    "www.msftncsi.com",
    "msftncsi.com",
    # Firefox / 其他
    "detectportal.firefox.com",
    "nomands.vn",
}

# 匹配 generate_204 或 hotspot-detect 等路径的模式
_CAPTIVE_PATHS = re.compile(
    r"/(generate_204|hotspot-detect\.html|success\.txt|canonical.html|connecttest\.txt|ncsi\.txt)",
    re.IGNORECASE,
)


class CaptivePortalMiddleware(BaseHTTPMiddleware):
    """
    AP 模式下的 Captive Portal 中间件。

    工作原理：
    1. 手机连上 WiFi 热点后自动发送 HTTP 请求到特定 URL 检测网络
    2. dnsmasq 将所有 DNS 解析到 192.168.4.1
    3. 本中间件拦截这些请求，返回 302 重定向到配网页面
    """

    def __init__(self, app, setup_url: str = "/setup"):
        super().__init__(app)
        self._setup_url = setup_url

    async def dispatch(self, request: Request, call_next):
        host = request.headers.get("host", "").split(":")[0].lower()
        path = request.url.path.lower()

        # 只拦截非本地请求（排除 localhost、192.168.4.1 本机 IP）
        if host in ("localhost", "127.0.0.1", "192.168.4.1"):
            return await call_next(request)

        # 匹配 Captive Portal 检测请求
        is_captive_host = host in CAPTIVE_PORTAL_HOSTS
        is_captive_path = bool(_CAPTIVE_PATHS.match(path))

        if is_captive_host or is_captive_path:
            logger.info(f"Captive Portal redirect: {host}{path} -> {self._setup_url}")

            # Android 的 generate_204 期望 302 重定向
            # Apple 的 hotspot-detect 期望 302 重定向
            return RedirectResponse(url=self._setup_url, status_code=302)

        # 对其他非本机 Host 的请求也重定向到配网页面
        # （dnsmasq 已将所有 DNS 指向本机，非 API 请求都是用户在浏览器输入的 URL）
        if host not in ("", "lelamp.local") and not path.startswith("/api/"):
            logger.info(f"Non-API redirect: {host}{path} -> {self._setup_url}")
            return RedirectResponse(url=self._setup_url, status_code=302)

        return await call_next(request)
```

**Step 2: 在 app.py 中注册中间件**

在 `app.py` 的 import 区域添加，在 `SecurityHeadersMiddleware` 之后注册：

```python
# lelamp/api/app.py 在 SecurityHeadersMiddleware 注册之后添加：
from lelamp.api.middleware.captive_portal import CaptivePortalMiddleware
app.add_middleware(CaptivePortalMiddleware, setup_url="/setup")
```

**Step 3: 验证中间件不干扰正常 API**

检查中间件逻辑：仅拦截非本机 Host 的请求和 Captive Portal 检测路径，`/api/` 路径和本机 IP 不受影响。

**Step 4: 提交**

```bash
git add lelamp/api/middleware/captive_portal.py lelamp/api/app.py
git commit -m "feat(setup): 添加 Captive Portal 中间件实现自动跳转配网页面"
```

---

### Task 2: 简化前端 SetupWizardView（6步 → 4步）

**Files:**
- Modify: `web/src/views/SetupWizardView.vue` (全面重构)

**Step 1: 重构步骤定义和状态**

将 `steps` 从 6 步改为 4 步：

```typescript
const steps = ['欢迎', '选择 WiFi', '连接中', '完成']
```

合并状态：删除 `authTab`, `authLoading`, `loginForm`, `registerForm` 等注册/登录相关状态。

**Step 2: 重构模板**

- **步骤 0（欢迎）**：保持不变，去掉多余内容
- **步骤 1（选择 WiFi + 输入密码）**：合并原步骤 2 和 3，WiFi 列表 + 选中后展开密码输入框
- **步骤 2（连接中）**：保持不变，WebSocket 实时进度
- **步骤 3（完成）**：简化为成功提示 + LED 绿色反馈描述，去掉重启倒计时和注册流程

**Step 3: 重构连接逻辑**

- 步骤 1 点击"连接"后自动进入步骤 2
- 连接成功后调用 `/api/system/setup/complete` 并自动进入步骤 3
- 步骤 3 显示"WiFi 连接成功"提示，不需要重启（直接切换模式）

**Step 4: 重构底部导航**

简化底部按钮逻辑：只保留"上一步"和"连接"按钮。

**Step 5: 更新样式**

步骤 1 的 WiFi 列表和密码输入框需要合并到一个视图中。选中网络后展开密码输入区域。

**Step 6: 提交**

```bash
git add web/src/views/SetupWizardView.vue
git commit -m "feat(setup): 简化配网向导从6步到4步，合并WiFi选择和密码输入"
```

---

### Task 3: 后端 setup/complete 优化 — 免重启直接切换

**Files:**
- Modify: `lelamp/api/routes/system.py:211-251` (`complete_setup` 端点)

**Step 1: 修改 complete_setup 端点**

当前逻辑：标记完成 → 停止 AP → 安排重启。

优化为：标记完成 → 停止 AP → 保持 WiFi 客户端连接 → 触发 LED/语音反馈 → **不重启**。

```python
@router.post("/setup/complete")
async def complete_setup(request: SetupCompleteRequest) -> dict:
    try:
        # 1. 标记配置完成
        await onboarding_manager.mark_setup_complete(request.wifi_ssid)

        # 2. 停止 AP 模式（WiFi 客户端连接已在 connect 步骤建立）
        await ap_manager.stop_ap_mode()

        # 3. 触发 LED 反馈（绿色常亮 3 秒）
        try:
            from lelamp.service.rgb.rgb_service import RGBService
            rgb = RGBService()
            rgb.dispatch_event("solid", {"color": (0, 255, 0)})
            # 3 秒后恢复正常
            import asyncio
            async def _restore_led():
                await asyncio.sleep(3)
                rgb.dispatch_event("off", {})
            asyncio.create_task(_restore_led())
        except Exception:
            pass

        # 4. 语音播报「WiFi 连接成功」
        try:
            from lelamp.api.services.setup_event_bus import setup_event_bus
            await setup_event_bus.publish({"event": "setup_complete", "ssid": request.wifi_ssid})
        except Exception:
            pass

        logger.info(f"Setup completed for WiFi: {request.wifi_ssid}")

        return {
            "success": True,
            "message": "配置完成",
        }
    except Exception as e:
        logger.error(f"Complete setup error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"完成配置失败: {str(e)}"
        )
```

**Step 2: 提交**

```bash
git add lelamp/api/routes/system.py
git commit -m "feat(setup): 配网完成后免重启直接切换模式，添加LED绿色反馈"
```

---

### Task 4: dnsmasq 配置确认 — DNS 全劫持

**Files:**
- Review: `lelamp/api/services/ap_manager.py:464-498` (`_write_dnsmasq_config`)

**Step 1: 确认 dnsmasq 配置已有 DNS 全劫持**

当前 `_write_dnsmasq_config` 已包含 `address=/#/192.168.4.1`，这意味着所有 DNS 请求都会解析到 `192.168.4.1`。这已经满足 Captive Portal 的需求。

验证关键行：
```
address=/#/{self._config.ip_address}
```

确认此行存在，无需修改。如果缺失则添加。

**Step 2: 确认无需修改后提交（或跳过）**

此任务可能不需要代码修改，仅确认。

---

### Task 5: 前端路由守卫更新

**Files:**
- Modify: `web/src/router/index.ts:59-114`

**Step 1: 更新路由守卫逻辑**

当设备在 AP 模式时，所有路径都应重定向到 `/setup`，不再只检查首页：

```typescript
// 简化守卫逻辑：AP 模式下所有非 setup 路径都重定向
if (data.is_ap_mode && to.path !== '/setup') {
  next({ path: '/setup', replace: true })
  return
}
```

当前代码只在首页做检查，需要将检查移到更早的位置，让所有路径都被拦截。

**Step 2: 提交**

```bash
git add web/src/router/index.ts
git commit -m "fix(setup): AP模式下所有路由都重定向到配网页面"
```

---

### Task 6: 集成测试

**Files:**
- Create: `tests/test_captive_portal.py`
- Create: `tests/test_setup_wizard.py`

**Step 1: 编写 Captive Portal 中间件测试**

```python
# tests/test_captive_portal.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from starlette.testclient import TestClient
from fastapi import FastAPI

from lelamp.api.middleware.captive_portal import CaptivePortalMiddleware


@pytest.fixture
def app():
    app = FastAPI()
    app.add_middleware(CaptivePortalMiddleware, setup_url="/setup")

    @app.get("/setup")
    async def setup_page():
        return {"page": "setup"}

    @app.get("/api/test")
    async def api_test():
        return {"ok": True}

    return app


@pytest.fixture
def client(app):
    return TestClient(app, follow_redirects=False)


class TestCaptivePortal:
    def test_apple_captive_redirect(self, client):
        """Apple 设备的 captive.apple.com 请求应被重定向"""
        resp = client.get("/", headers={"Host": "captive.apple.com"})
        assert resp.status_code == 302
        assert "/setup" in resp.headers["location"]

    def test_android_generate_204_redirect(self, client):
        """Android 的 generate_204 应被重定向"""
        resp = client.get("/generate_204", headers={"Host": "connectivitycheck.gstatic.com"})
        assert resp.status_code == 302

    def test_windows_redirect(self, client):
        """Windows 的连接测试应被重定向"""
        resp = client.get("/connecttest.txt", headers={"Host": "www.msftconnecttest.com"})
        assert resp.status_code == 302

    def test_local_requests_pass_through(self, client):
        """本机请求不应被拦截"""
        resp = client.get("/api/test", headers={"Host": "192.168.4.1"})
        assert resp.status_code == 200

    def test_api_requests_not_redirected(self, client):
        """API 路径不应被重定向"""
        resp = client.get("/api/test", headers={"Host": "some-random-host.com"})
        assert resp.status_code == 200

    def test_unknown_host_non_api_redirected(self, client):
        """非 API 请求的未知 host 应被重定向"""
        resp = client.get("/some/page", headers={"Host": "example.com"})
        assert resp.status_code == 302
```

**Step 2: 运行测试**

```bash
uv run pytest tests/test_captive_portal.py -v
```

**Step 3: 提交**

```bash
git add tests/test_captive_portal.py
git commit -m "test(setup): 添加 Captive Portal 中间件单元测试"
```

---

### Task 7: 前端构建验证

**Step 1: 构建前端**

```bash
cd web && pnpm install && pnpm build
```

**Step 2: 验证构建产物**

确认 `web/dist/index.html` 存在且包含正确的路由。

**Step 3: 本地启动 API 验证**

```bash
cd /Users/jackwang/lelamp_runtime
uv run uvicorn lelamp.api.app:app --host 0.0.0.0 --port 8000
```

验证：
1. 访问 `http://localhost:8000/setup` 能正常显示配网页面
2. 访问 `http://localhost:8000/api/system/setup/status` 返回正常 JSON
3. Captive Portal 检测请求被正确拦截（用 curl 模拟）

```bash
# 模拟 Apple Captive Portal 检测
curl -v -H "Host: captive.apple.com" http://localhost:8000/
# 应返回 302 重定向到 /setup

# 模拟 Android 检测
curl -v -H "Host: connectivitycheck.gstatic.com" http://localhost:8000/generate_204
# 应返回 302 重定向到 /setup
```

---

## 执行顺序

1. Task 1 (Captive Portal 中间件) → Task 6 (测试) — 可一起做
2. Task 2 (前端简化) → Task 5 (路由守卫) — 一起做
3. Task 3 (后端 complete 优化)
4. Task 4 (dnsmasq 确认) — 可能不需要修改
5. Task 7 (集成验证)

## 风险点

1. **iOS Safari 的 Captive Portal 行为**：iOS 在检测到 Captive Portal 后可能不会自动弹出登录页面，而是显示"无法加入网络"。需要在真实设备上测试。
2. **Android 版本差异**：不同 Android 版本的连通性检测 URL 可能不同。中间件已覆盖主流 URL。
3. **免重启切换**：停止 AP 后 WiFi 客户端连接可能断开。需要确保 nmcli 连接在 AP 停止后仍然活跃。
4. **dnsmasq 配置**：`address=/#/` 会将所有域名解析到本机 IP，确保 API 请求通过 IP 直接访问不受影响。
