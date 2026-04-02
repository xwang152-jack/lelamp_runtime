# 配网首次体验升级 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 提升 LeLamp 开箱配网体验，减少手动干预，提供实时反馈，支持中断恢复。

**Architecture:** 后端新增 WiFi 连接重试逻辑、事件总线、WebSocket 进度推送、恢复状态 API、auto-bind 降级；前端新增错误消息工具函数、恢复检测、实时进度展示、重启后验证。

**Tech Stack:** Python/FastAPI (asyncio)、Vue 3 Composition API、TypeScript、WebSocket

---

## Task 1: WiFi 连接重试 + 网络验证（后端）

**Files:**
- Modify: `lelamp/api/services/wifi_manager.py:270-316`
- Test: `tests/test_api_routes.py`（在现有 `TestWiFiManager` 类中追加）

### Step 1: 在 `tests/test_api_routes.py` 追加失败测试

打开 `tests/test_api_routes.py`，在现有的 `TestWiFiManager` 类末尾追加：

```python
    @pytest.mark.asyncio
    async def test_connect_retries_on_failure(self):
        """WiFi 连接失败时应自动重试"""
        from lelamp.api.services.wifi_manager import WiFiManager

        manager = WiFiManager()
        call_count = 0

        async def mock_connect_once(ssid, password):
            nonlocal call_count
            call_count += 1
            return {"success": False, "message": "连接失败", "ssid": ssid}

        with patch.object(manager, '_try_nmcli_connect', side_effect=mock_connect_once):
            result = await manager.connect("TestSSID", "wrongpass", max_retries=3)

        assert result["success"] is False
        assert call_count == 3  # 应重试 3 次

    @pytest.mark.asyncio
    async def test_connect_succeeds_on_second_attempt(self):
        """WiFi 连接第一次失败、第二次成功"""
        from lelamp.api.services.wifi_manager import WiFiManager

        manager = WiFiManager()
        attempts = []

        async def mock_connect_once(ssid, password):
            attempts.append(1)
            if len(attempts) == 1:
                return {"success": False, "message": "连接超时", "ssid": ssid}
            return {"success": True, "message": "连接成功", "ssid": ssid}

        with patch.object(manager, '_try_nmcli_connect', side_effect=mock_connect_once):
            with patch('asyncio.sleep', return_value=None):  # 不真正等待
                result = await manager.connect("TestSSID", "pass", max_retries=3)

        assert result["success"] is True
        assert len(attempts) == 2

    @pytest.mark.asyncio
    async def test_verify_network_reachability_success(self):
        """网络可达性验证成功"""
        from lelamp.api.services.wifi_manager import WiFiManager

        manager = WiFiManager()
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"", b""))

        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            result = await manager._verify_network_reachability()

        assert result is True

    @pytest.mark.asyncio
    async def test_verify_network_reachability_failure(self):
        """网络不可达时返回 False"""
        from lelamp.api.services.wifi_manager import WiFiManager

        manager = WiFiManager()
        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(return_value=(b"", b"ping: connect: Network is unreachable"))

        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            result = await manager._verify_network_reachability()

        assert result is False
```

在文件顶部 import 中追加（如果没有）：
```python
from unittest.mock import AsyncMock
```

### Step 2: 运行测试，确认失败

```bash
cd /Users/jackwang/lelamp_runtime
uv run pytest tests/test_api_routes.py::TestWiFiManager -v
```

Expected: FAIL，提示 `_try_nmcli_connect` 不存在

### Step 3: 重构 `wifi_manager.py` 的 `connect()` 方法

将 `lelamp/api/services/wifi_manager.py` 第 270-316 行的 `connect()` 方法替换为以下内容：

```python
async def connect(
    self,
    ssid: str,
    password: Optional[str] = None,
    max_retries: int = 3,
    event_callback=None,
) -> dict:
    """
    连接到 WiFi 网络，支持自动重试和网络验证

    Args:
        ssid: 网络名称
        password: WiFi 密码（开放网络可为 None）
        max_retries: 最大重试次数（默认 3）
        event_callback: 可选的异步回调，接收进度事件 dict

    Returns:
        连接结果字典，包含 success, message, ssid, network_ok
    """
    async with self._connect_lock:
        for attempt in range(1, max_retries + 1):
            if event_callback:
                await event_callback({
                    "event": "wifi_connecting",
                    "attempt": attempt,
                    "max_attempts": max_retries,
                    "ssid": ssid,
                })
            result = await self._try_nmcli_connect(ssid, password)
            if result["success"]:
                if event_callback:
                    await event_callback({"event": "wifi_connected", "ssid": ssid})
                    await event_callback({"event": "network_checking"})
                network_ok = await self._verify_network_reachability()
                if event_callback:
                    if network_ok:
                        await event_callback({"event": "network_ok"})
                    else:
                        await event_callback({
                            "event": "network_failed",
                            "reason": "no_internet",
                        })
                result["network_ok"] = network_ok
                return result
            if attempt < max_retries:
                wait_sec = 2 ** attempt  # 2, 4, 8 秒
                if event_callback:
                    await event_callback({
                        "event": "wifi_failed",
                        "attempt": attempt,
                        "retry_in": wait_sec,
                    })
                await asyncio.sleep(wait_sec)

        return {"success": False, "message": f"连接失败（已重试 {max_retries} 次）", "ssid": ssid, "network_ok": False}

async def _try_nmcli_connect(self, ssid: str, password: Optional[str] = None) -> dict:
    """单次 nmcli 连接尝试（原 connect 方法中的核心逻辑）"""
    try:
        await self._delete_connection(ssid)
        cmd = ["sudo", self._nmcli_path, "device", "wifi", "connect", ssid]
        if password:
            cmd.extend(["password", password])
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=30
        )
        if process.returncode == 0:
            logger.info(f"nmcli connect succeeded: {ssid}")
            return {"success": True, "message": "连接成功", "ssid": ssid}
        else:
            error_msg = stderr.decode().strip()
            logger.warning(f"nmcli connect failed: {ssid}: {error_msg}")
            return {"success": False, "message": f"连接失败: {error_msg}", "ssid": ssid}
    except asyncio.TimeoutError:
        logger.warning(f"nmcli connect timeout: {ssid}")
        return {"success": False, "message": "连接超时", "ssid": ssid}
    except Exception as e:
        logger.error(f"nmcli connect error: {e}", exc_info=True)
        return {"success": False, "message": str(e), "ssid": ssid}

async def _verify_network_reachability(self, host: str = "223.5.5.5", timeout: int = 5) -> bool:
    """通过 ping 验证网络是否可达（使用阿里 DNS，国内可达）"""
    try:
        process = await asyncio.create_subprocess_exec(
            "ping", "-c", "1", "-W", str(timeout), host,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(process.communicate(), timeout=timeout + 2)
        return process.returncode == 0
    except Exception:
        return False
```

### Step 4: 运行测试，确认通过

```bash
uv run pytest tests/test_api_routes.py::TestWiFiManager -v
```

Expected: 所有 WiFiManager 测试 PASS

### Step 5: 提交

```bash
git add lelamp/api/services/wifi_manager.py tests/test_api_routes.py
git commit -m "feat(wifi): 支持自动重试和网络可达性验证"
```

---

## Task 2: 配网进度事件总线（后端）

**Files:**
- Create: `lelamp/api/services/setup_event_bus.py`
- Test: `tests/api/test_setup_event_bus.py`

### Step 1: 创建测试文件

创建 `tests/api/test_setup_event_bus.py`：

```python
"""测试配网进度事件总线"""
import pytest
import asyncio


@pytest.mark.asyncio
async def test_publish_and_receive():
    """发布事件后订阅者应收到"""
    from lelamp.api.services.setup_event_bus import SetupEventBus

    bus = SetupEventBus()
    queue = asyncio.Queue()
    bus.subscribe(queue)

    event = {"event": "wifi_connecting", "attempt": 1}
    await bus.publish(event)

    received = queue.get_nowait()
    assert received == event


@pytest.mark.asyncio
async def test_multiple_subscribers():
    """所有订阅者都应收到同一事件"""
    from lelamp.api.services.setup_event_bus import SetupEventBus

    bus = SetupEventBus()
    q1, q2 = asyncio.Queue(), asyncio.Queue()
    bus.subscribe(q1)
    bus.subscribe(q2)

    await bus.publish({"event": "network_ok"})

    assert q1.get_nowait()["event"] == "network_ok"
    assert q2.get_nowait()["event"] == "network_ok"


@pytest.mark.asyncio
async def test_unsubscribe():
    """取消订阅后不再收到事件"""
    from lelamp.api.services.setup_event_bus import SetupEventBus

    bus = SetupEventBus()
    queue = asyncio.Queue()
    bus.subscribe(queue)
    bus.unsubscribe(queue)

    await bus.publish({"event": "wifi_connected"})

    assert queue.empty()
```

### Step 2: 运行测试，确认失败

```bash
uv run pytest tests/api/test_setup_event_bus.py -v
```

Expected: FAIL，`setup_event_bus` 模块不存在

### Step 3: 创建事件总线

创建 `lelamp/api/services/setup_event_bus.py`：

```python
"""
配网进度事件总线

在 WiFi 连接过程中广播进度事件到所有 WebSocket 订阅者
"""
import asyncio
import logging
from typing import List

logger = logging.getLogger(__name__)


class SetupEventBus:
    """简单的 asyncio 广播事件总线"""

    def __init__(self):
        self._subscribers: List[asyncio.Queue] = []

    def subscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.append(queue)

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        try:
            self._subscribers.remove(queue)
        except ValueError:
            pass

    async def publish(self, event: dict) -> None:
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("Setup event queue full, dropping event")


# 全局单例
setup_event_bus = SetupEventBus()
```

### Step 4: 运行测试，确认通过

```bash
uv run pytest tests/api/test_setup_event_bus.py -v
```

Expected: 3 tests PASS

### Step 5: 提交

```bash
git add lelamp/api/services/setup_event_bus.py tests/api/test_setup_event_bus.py
git commit -m "feat(setup): 添加配网进度事件总线"
```

---

## Task 3: 配网进度 WebSocket 端点（后端）

**Files:**
- Create: `lelamp/api/routes/setup_ws.py`
- Modify: `lelamp/api/routes/__init__.py`
- Modify: `lelamp/api/routes/system.py`（wifi/connect 端点接入事件总线）

### Step 1: 创建 `lelamp/api/routes/setup_ws.py`

```python
"""
配网进度 WebSocket 端点

GET /ws/setup - 实时推送 WiFi 连接和配网进度事件
"""
import asyncio
import json
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from lelamp.api.services.setup_event_bus import setup_event_bus

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/setup")
async def setup_progress_ws(websocket: WebSocket):
    """
    配网进度 WebSocket

    连接后实时推送以下事件（JSON）：
    - wifi_connecting: { attempt, max_attempts, ssid }
    - wifi_connected: { ssid }
    - wifi_failed: { attempt, retry_in }
    - network_checking
    - network_ok
    - network_failed: { reason }
    - setup_complete
    - rebooting: { countdown }
    """
    await websocket.accept()
    queue: asyncio.Queue = asyncio.Queue(maxsize=50)
    setup_event_bus.subscribe(queue)
    logger.info("Setup WebSocket client connected")

    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30.0)
                await websocket.send_text(json.dumps(event, ensure_ascii=False))
            except asyncio.TimeoutError:
                # 心跳
                await websocket.send_text(json.dumps({"event": "ping"}))
    except WebSocketDisconnect:
        logger.info("Setup WebSocket client disconnected")
    except Exception as e:
        logger.error(f"Setup WebSocket error: {e}")
    finally:
        setup_event_bus.unsubscribe(queue)
```

### Step 2: 在 `lelamp/api/routes/__init__.py` 注册路由

在文件末尾的 `api_router.include_router(livekit.router, ...)` 之后追加：

```python
from lelamp.api.routes import setup_ws
api_router.include_router(setup_ws.router, prefix="/ws", tags=["setup-ws"])
```

注意：`prefix="/ws"` 与现有 websocket 路由共用相同前缀，最终路径为 `/api/ws/setup`。

### Step 3: 修改 `system.py` 的 wifi/connect 端点以触发事件

找到 `system.py` 中的 `POST /wifi/connect` 处理函数（搜索 `wifi/connect`），在调用 `wifi_manager.connect()` 时传入 event_callback：

```python
from lelamp.api.services.setup_event_bus import setup_event_bus

# ... 在 connect 路由处理函数中:
result = await wifi_manager.connect(
    ssid=request.ssid,
    password=request.password,
    max_retries=3,
    event_callback=setup_event_bus.publish,
)
```

找到现有代码：
```python
result = await wifi_manager.connect(request.ssid, request.password)
```
替换为上面的版本。

### Step 4: 手动验证（无自动测试）

启动 API 服务：
```bash
uv run uvicorn lelamp.api.app:app --host 0.0.0.0 --port 8000 --reload
```

用 wscat 或浏览器 DevTools 连接 `ws://localhost:8000/api/ws/setup`，然后 POST `/api/system/wifi/connect`，观察 WebSocket 是否收到进度事件。

### Step 5: 提交

```bash
git add lelamp/api/routes/setup_ws.py lelamp/api/routes/__init__.py lelamp/api/routes/system.py
git commit -m "feat(setup): 添加配网 WebSocket 进度推送端点"
```

---

## Task 4: 恢复状态 API（后端）

**Files:**
- Modify: `lelamp/api/routes/system.py`（追加 `GET /setup/recovery` 端点）
- Test: `tests/api/test_setup_state.py`（在现有文件追加）

### Step 1: 在 `tests/api/test_setup_state.py` 追加测试

```python
@pytest.mark.asyncio
async def test_setup_recovery_already_configured():
    """已配置时恢复状态应返回 can_recover=False"""
    from lelamp.api.routes.system import get_setup_recovery
    from unittest.mock import AsyncMock, patch

    with patch('lelamp.api.routes.system.onboarding_manager') as mock_mgr:
        mock_mgr.get_configuration_summary = AsyncMock(
            return_value={"is_configured": True}
        )
        result = await get_setup_recovery()

    assert result["can_recover"] is False
    assert result["reason"] == "already_done"


@pytest.mark.asyncio
async def test_setup_recovery_wifi_connected():
    """WiFi 已连接时应返回可恢复到步骤 4"""
    from lelamp.api.routes.system import get_setup_recovery
    from unittest.mock import AsyncMock, patch

    with patch('lelamp.api.routes.system.onboarding_manager') as mock_mgr, \
         patch('lelamp.api.routes.system.wifi_manager') as mock_wifi:
        mock_mgr.get_configuration_summary = AsyncMock(
            return_value={"is_configured": False}
        )
        mock_wifi.get_status = AsyncMock(
            return_value={"connected": True, "ssid": "HomeWiFi"}
        )
        result = await get_setup_recovery()

    assert result["can_recover"] is True
    assert result["skip_to_step"] == 4
    assert result["current_ssid"] == "HomeWiFi"
```

### Step 2: 运行测试，确认失败

```bash
uv run pytest tests/api/test_setup_state.py -v
```

Expected: FAIL，`get_setup_recovery` 不存在

### Step 3: 在 `system.py` 添加 `GET /setup/recovery` 端点

在 `system.py` 的 `@router.get("/setup/status")` 之后追加：

```python
@router.get("/setup/recovery")
async def get_setup_recovery() -> dict:
    """
    检测配网流程是否可以从中断处恢复

    Returns:
        {
          "can_recover": bool,
          "skip_to_step": int | None,   # 4 = 直接跳到认证步骤
          "current_ssid": str | None,
          "reason": str | None,          # "already_done" / "already_connected"
        }
    """
    try:
        summary = await onboarding_manager.get_configuration_summary()
        if summary.get("is_configured"):
            return {"can_recover": False, "reason": "already_done"}

        wifi_status = await wifi_manager.get_status()
        if wifi_status.get("connected"):
            return {
                "can_recover": True,
                "skip_to_step": 4,
                "current_ssid": wifi_status.get("ssid"),
                "reason": "already_connected",
            }

        return {"can_recover": False}
    except Exception as e:
        logger.error(f"get_setup_recovery error: {e}", exc_info=True)
        return {"can_recover": False}
```

### Step 4: 运行测试，确认通过

```bash
uv run pytest tests/api/test_setup_state.py -v
```

Expected: 全部 PASS

### Step 5: 提交

```bash
git add lelamp/api/routes/system.py tests/api/test_setup_state.py
git commit -m "feat(setup): 添加配网中断恢复状态 API"
```

---

## Task 5: auto-bind 失败降级（后端）

**Files:**
- Modify: `lelamp/api/routes/auth.py:230-241`

### Step 1: 直接修改（无需测试，纯行为变更）

将 `auth.py` 第 230-241 行的 `auto_bind_device` 异常处理替换为：

```python
    except ValueError as e:
        logger.warning(f"自动绑定失败（已跳过）: {str(e)}")
        # 不阻断配网流程，返回跳过标记
        return DeviceBindResponse(
            device_id="unknown",
            permission_level="none",
            bound_at="",
            skipped=True,
            skip_reason=str(e),
        )
    except Exception as e:
        logger.error(f"自动绑定错误（已跳过）: {str(e)}")
        return DeviceBindResponse(
            device_id="unknown",
            permission_level="none",
            bound_at="",
            skipped=True,
            skip_reason="internal_error",
        )
```

同时需要在 `DeviceBindResponse` 模型中添加可选字段。找到 `lelamp/api/models/auth_models.py`，在 `DeviceBindResponse` 类中追加：

```python
skipped: bool = False
skip_reason: Optional[str] = None
```

### Step 2: 提交

```bash
git add lelamp/api/routes/auth.py lelamp/api/models/auth_models.py
git commit -m "fix(auth): auto-bind 失败时降级跳过而非阻断配网流程"
```

---

## Task 6: 前端错误消息工具（前端）

**Files:**
- Create: `web/src/utils/errorMessages.ts`

### Step 1: 创建 `web/src/utils/errorMessages.ts`

```typescript
/**
 * 将 API 错误转换为用户友好的中文消息
 */

const ERROR_MAP: Record<string, string> = {
  'Incorrect username or password': '用户名或密码错误，请重新输入',
  'Username already exists': '该用户名已被注册，请换一个或直接登录',
  'Email already exists': '该邮箱已被注册，请直接登录',
  '设备密钥未配置，无法自动绑定': '设备绑定已跳过，后续可在设置页面完成',
  'Invalid device secret': '设备验证失败，请重新开始配置',
  'Internal server error': '设备遇到内部错误，请重启后再试',
}

const WIFI_FAILURE_HINT = '密码可能有误，或信号较弱，建议靠近路由器后重试'
const NETWORK_FAILURE_HINT = 'WiFi 已连接，但无法访问互联网，请检查路由器设置'
const GENERIC_ERROR = '操作失败，请重试'

export function formatApiError(error: unknown, context?: 'wifi' | 'network' | 'auth'): string {
  // axios error with response
  if (error && typeof error === 'object' && 'response' in error) {
    const axiosErr = error as { response?: { data?: { detail?: string } } }
    const detail = axiosErr.response?.data?.detail
    if (detail) {
      return ERROR_MAP[detail] ?? detail
    }
  }

  // Error with message
  if (error instanceof Error) {
    if (context === 'wifi') return WIFI_FAILURE_HINT
    if (context === 'network') return NETWORK_FAILURE_HINT
    return ERROR_MAP[error.message] ?? error.message ?? GENERIC_ERROR
  }

  if (typeof error === 'string') {
    return ERROR_MAP[error] ?? error
  }

  return GENERIC_ERROR
}

export { WIFI_FAILURE_HINT, NETWORK_FAILURE_HINT }
```

### Step 2: 提交

```bash
git add web/src/utils/errorMessages.ts
git commit -m "feat(web): 添加 API 错误消息中文化工具函数"
```

---

## Task 7: 前端恢复检测 + 实时进度展示

**Files:**
- Modify: `web/src/views/SetupWizardView.vue`

这是本次改动最大的前端任务，分为三个子步骤。

### Step 1: 添加 import 和新状态变量

在 `SetupWizardView.vue` 的 `<script setup>` 块中，在现有 import 之后追加：

```typescript
import { formatApiError } from '@/utils/errorMessages'

// 恢复检测
const recoveryChecked = ref(false)
const showRecoveryDialog = ref(false)
const recoveryInfo = ref<{ skip_to_step: number; current_ssid: string } | null>(null)

// 实时进度（步骤 3）
interface ProgressItem {
  text: string
  status: 'pending' | 'running' | 'done' | 'error'
}
const progressItems = ref<ProgressItem[]>([])
const retryCountdown = ref(0)
let setupWs: WebSocket | null = null
```

### Step 2: 替换 `onMounted` 函数

将现有的 `onMounted` 函数替换为：

```typescript
onMounted(async () => {
  // 获取设备信息
  try {
    const response = await axios.get(`${API_BASE}/api/system/device`)
    deviceInfo.value = response.data
  } catch {
    // ignore
  }

  // 检查是否可以从中断处恢复
  try {
    const recovery = await axios.get(`${API_BASE}/api/system/setup/recovery`)
    recoveryChecked.value = true
    if (recovery.data.can_recover) {
      recoveryInfo.value = recovery.data
      showRecoveryDialog.value = true
    } else {
      // 不可恢复，自动开始扫描
      handleScan()
    }
  } catch {
    recoveryChecked.value = true
    handleScan()
  }
})
```

### Step 3: 添加恢复对话框处理函数和进度相关函数

在 `handleRetry` 函数之后追加：

```typescript
// 恢复确认
function acceptRecovery() {
  showRecoveryDialog.value = false
  if (recoveryInfo.value) {
    // 设置已选网络信息（仅用于步骤 5 完成时读取 SSID）
    selectedNetwork.value = {
      ssid: recoveryInfo.value.current_ssid,
      bssid: '',
      signal_strength: 100,
      security: 'wpa2',
      frequency: '2.4GHz',
    }
    currentStep.value = recoveryInfo.value.skip_to_step
  }
}

function rejectRecovery() {
  showRecoveryDialog.value = false
  handleScan()
}

// WebSocket 驱动的进度：连接前初始化进度列表
function initProgress() {
  progressItems.value = [
    { text: '正在发送连接请求...', status: 'running' },
    { text: '正在连接 WiFi...', status: 'pending' },
    { text: '验证网络连通性...', status: 'pending' },
  ]
}

// 处理来自 WebSocket 的进度事件
function handleProgressEvent(event: { event: string; attempt?: number; retry_in?: number }) {
  switch (event.event) {
    case 'wifi_connecting':
      progressItems.value[0].status = 'done'
      progressItems.value[1].status = 'running'
      progressItems.value[1].text = `正在连接 WiFi（第 ${event.attempt}/${3} 次）...`
      break
    case 'wifi_connected':
      progressItems.value[1].status = 'done'
      progressItems.value[2].status = 'running'
      break
    case 'wifi_failed':
      if (event.retry_in) {
        progressItems.value[1].text = `连接失败，${event.retry_in} 秒后重试...`
        retryCountdown.value = event.retry_in
        const timer = setInterval(() => {
          retryCountdown.value--
          if (retryCountdown.value <= 0) clearInterval(timer)
        }, 1000)
      }
      break
    case 'network_ok':
      progressItems.value[2].status = 'done'
      break
    case 'network_failed':
      progressItems.value[2].status = 'error'
      progressItems.value[2].text = 'WiFi 已连接，但无法访问互联网'
      connectionError.value = 'WiFi 已连接，但无法访问互联网，请检查路由器设置'
      break
  }
}

// 建立 WebSocket 连接（在进入步骤 3 前调用）
function connectSetupWs() {
  const wsUrl = `${API_BASE.replace('http', 'ws')}/api/ws/setup`
  setupWs = new WebSocket(wsUrl)
  setupWs.onmessage = (e) => {
    try {
      const event = JSON.parse(e.data)
      handleProgressEvent(event)
    } catch { /* ignore */ }
  }
  setupWs.onerror = () => { /* ignore, 非关键 */ }
}

function disconnectSetupWs() {
  if (setupWs) {
    setupWs.close()
    setupWs = null
  }
}
```

### Step 4: 修改 `handleConnect` 函数

将现有的 `handleConnect` 函数替换为：

```typescript
async function handleConnect() {
  if (!selectedNetwork.value) return

  connecting.value = true
  connectionError.value = ''
  currentStep.value = 3
  initProgress()
  connectSetupWs()

  try {
    const connectResponse = await apiClient.post(`${API_BASE}/api/system/wifi/connect`, {
      ssid: selectedNetwork.value.ssid,
      password: selectedNetwork.value.security === 'open' ? undefined : wifiPassword.value
    })

    if (!connectResponse.data.success) {
      throw new Error(connectResponse.data.message || '连接失败')
    }

    // 检查网络可达性结果
    if (connectResponse.data.network_ok === false) {
      connectionError.value = 'WiFi 已连接，但无法访问互联网，请检查路由器设置'
      connecting.value = false
      return
    }

    connectingStatus.value = { title: '连接成功！', message: '正在跳转到账号设置...' }
    await new Promise(resolve => setTimeout(resolve, 800))
    currentStep.value = 4
  } catch (error: any) {
    connectionError.value = formatApiError(error, 'wifi')
    connecting.value = false
  } finally {
    disconnectSetupWs()
  }
}
```

### Step 5: 修改步骤 3 的模板部分

将 `v-if="currentStep === 3"` 对应的 div 替换为：

```html
<!-- 步骤 4: 连接验证 -->
<div v-if="currentStep === 3" class="connecting-step">
  <h2>{{ connectionError ? '连接失败' : '正在连接...' }}</h2>

  <!-- 实时进度列表 -->
  <div class="progress-list" v-if="progressItems.length > 0 && !connectionError">
    <div
      v-for="(item, i) in progressItems"
      :key="i"
      class="progress-item"
      :class="item.status"
    >
      <span class="progress-icon">
        <span v-if="item.status === 'done'">✓</span>
        <span v-else-if="item.status === 'running'" class="mini-spinner"></span>
        <span v-else-if="item.status === 'error'">✗</span>
        <span v-else>○</span>
      </span>
      <span class="progress-text">{{ item.text }}</span>
    </div>
  </div>

  <!-- 静态 spinner（初始状态） -->
  <div v-else-if="!connectionError" class="connecting-animation">
    <div class="spinner"></div>
    <p>{{ connectingStatus.message }}</p>
  </div>

  <!-- 错误状态 -->
  <div v-if="connectionError" class="error-message">
    <el-alert type="error" :title="connectionError" show-icon :closable="false" />
    <el-button @click="handleRetry" class="retry-button" type="primary">重新选择网络</el-button>
  </div>
</div>
```

在 `</template>` 之前（即 `wizard-footer` div 之后）追加恢复对话框：

```html
<!-- 恢复确认对话框 -->
<el-dialog
  v-model="showRecoveryDialog"
  title="检测到已连接的 WiFi"
  width="90%"
  :close-on-click-modal="false"
  :show-close="false"
>
  <p>设备已连接到 WiFi：<strong>{{ recoveryInfo?.current_ssid }}</strong></p>
  <p>是否跳过 WiFi 配置，直接进入账号绑定步骤？</p>
  <template #footer>
    <el-button @click="rejectRecovery">重新配置</el-button>
    <el-button type="primary" @click="acceptRecovery">是，继续</el-button>
  </template>
</el-dialog>
```

同时在 progress-list 和 mini-spinner 样式中追加（在 `<style>` 块）：

```scss
.progress-list {
  margin: 20px 0;
  text-align: left;

  .progress-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 0;
    opacity: 0.5;

    &.running, &.done, &.error { opacity: 1; }
    &.done .progress-icon { color: #67c23a; }
    &.error .progress-icon { color: #f56c6c; }
  }

  .mini-spinner {
    display: inline-block;
    width: 14px;
    height: 14px;
    border: 2px solid rgba(255,255,255,0.3);
    border-top-color: white;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
}
```

### Step 6: 修改错误消息，使用 `formatApiError`

将 `handleAuthLogin` 和 `handleAuthRegister` 中的 `ElMessage.error(e.message || '...')` 替换为使用 `formatApiError`：

```typescript
// 在 handleAuthLogin catch 块：
ElMessage.error(formatApiError(e, 'auth'))

// 在 handleAuthRegister catch 块：
ElMessage.error(formatApiError(e, 'auth'))

// auto-bind 失败处理（允许 skipped: true 继续流程）：
const result = await authStore.autoBindDevice()
if (result.success || result.skipped) {   // 新增 || result.skipped
  currentStep.value = 5
  await completeSetup()
} else {
  ElMessage.error(formatApiError(result.error, 'auth'))
}
```

### Step 7: 构建确认无语法错误

```bash
cd /Users/jackwang/lelamp_runtime/web
pnpm build 2>&1 | head -30
```

Expected: 无 TypeScript 错误

### Step 8: 提交

```bash
cd /Users/jackwang/lelamp_runtime
git add web/src/views/SetupWizardView.vue
git commit -m "feat(web): 配网向导添加恢复检测、实时进度、错误消息中文化"
```

---

## Task 8: 重启后自动检测上线（前端）

**Files:**
- Modify: `web/src/views/SetupWizardView.vue`（修改 `completeSetup` 函数和步骤 5 模板）

### Step 1: 修改 `completeSetup` 函数

将现有的 `completeSetup` 函数替换为：

```typescript
const postRebootChecking = ref(false)
const postRebootError = ref(false)

async function completeSetup() {
  try {
    const response = await axios.post(`${API_BASE}/api/setup/complete`, {
      wifi_ssid: selectedNetwork.value?.ssid || 'unknown',
      restart_delay: countdown.value
    })

    if (response.data.success) {
      // 开始倒计时
      const timer = setInterval(() => {
        countdown.value--
        if (countdown.value <= 0) {
          clearInterval(timer)
          // 倒计时结束后开始轮询设备是否上线
          pollDeviceReboot()
        }
      }, 1000)
    }
  } catch (error) {
    ElMessage.error('保存配置失败，请重试')
  }
}

async function pollDeviceReboot() {
  postRebootChecking.value = true
  const maxAttempts = 20  // 最多等 60 秒（每 3 秒一次）
  for (let i = 0; i < maxAttempts; i++) {
    await new Promise(resolve => setTimeout(resolve, 3000))
    try {
      const res = await axios.get(`${API_BASE}/api/setup/status`, { timeout: 5000 })
      if (res.data.is_configured) {
        // 设备已重启并上线
        postRebootChecking.value = false
        // 已有 JWT token（注册/登录时存的），直接跳主界面
        window.location.href = '/'
        return
      }
    } catch {
      // 设备重启中，连接失败是正常的，继续等待
    }
  }
  // 超时
  postRebootChecking.value = false
  postRebootError.value = true
}
```

### Step 2: 更新步骤 5（完成）的模板

将 `v-if="currentStep === 5"` 对应的 div 替换为：

```html
<!-- 步骤 6: 完成 -->
<div v-if="currentStep === 5" class="complete-step">
  <div class="success-icon">✓</div>
  <h2>配置完成！</h2>
  <p>LeLamp 即将重启并连接到您的 WiFi</p>

  <!-- 倒计时阶段 -->
  <div v-if="countdown > 0" class="countdown">
    <span>{{ countdown }}</span> 秒后重启...
  </div>

  <!-- 等待重启上线 -->
  <div v-else-if="postRebootChecking" class="reboot-checking">
    <div class="spinner"></div>
    <p>正在等待设备重启...</p>
    <p class="hint">通常需要 15-30 秒</p>
  </div>

  <!-- 超时提示 -->
  <div v-else-if="postRebootError" class="reboot-timeout">
    <p>设备启动可能需要更长时间</p>
    <p>请手动访问：</p>
    <p class="access-url">http://lelamp.local:8000</p>
    <el-button type="primary" @click="() => window.location.href = '/'">
      刷新页面
    </el-button>
  </div>

  <!-- 上线后自动跳转（一般看不到，跳转太快） -->
  <div v-else class="access-info">
    <p>设备访问地址：</p>
    <p class="access-url">http://lelamp.local:8000</p>
  </div>
</div>
```

### Step 3: 在 `<script setup>` 顶部添加新 ref 的声明

确认 `postRebootChecking` 和 `postRebootError` 已在 Step 1 中声明（在 `completeSetup` 前面，放在其他 `ref` 声明旁边）。

### Step 4: 构建验证

```bash
cd /Users/jackwang/lelamp_runtime/web
pnpm build 2>&1 | head -30
```

Expected: 无错误

### Step 5: 提交

```bash
cd /Users/jackwang/lelamp_runtime
git add web/src/views/SetupWizardView.vue
git commit -m "feat(web): 配网完成后自动检测设备重启上线"
```

---

## 验收检查清单

完成所有 Task 后运行：

```bash
# 后端测试全量
uv run pytest tests/ -v -k "not hardware"

# 前端构建
cd web && pnpm build
```

预期：所有测试通过，前端无 TypeScript 错误。

**端到端验收场景**（在真实设备或模拟器上）：

1. 全新设备：完整走一遍 6 步配网，WiFi 第一次连接失败，确认自动重试
2. 中途中断：步骤 2-3 时关闭浏览器，重新打开，确认弹出恢复提示
3. 错误消息：输入错误密码，确认显示的是中文友好消息
4. 重启验证：步骤 5 完成后，确认自动检测设备上线并跳转
