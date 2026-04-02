# 手势置信度分层 + 舵机故障情感化降级 实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 手势识别按置信度三档处理（直接执行/语音确认/忽略），舵机故障时 LED 橙色呼吸 + 语音告知用户。

**Architecture:**
- `ProactiveVisionMonitor._detect_gestures()` 注入 confidence 到 context
- `LeLamp.on_gesture()` 按阈值分层；`_speak_proactively()` 用 event loop 跨线程发声
- `MotorsService._health_check_loop()` 检测状态转换，调用 `_motor_fault_callback`

**Tech Stack:** Python asyncio, threading, livekit-agents, pytest

---

## Task 1: ProactiveVisionMonitor 注入 confidence

**Files:**
- Modify: `lelamp/service/vision/proactive_vision_monitor.py:295-328`
- Test: `tests/test_edge_vision.py`

**Step 1: 写失败测试**

在 `tests/test_edge_vision.py` 末尾追加：

```python
class TestGestureConfidence:
    def test_gesture_context_includes_confidence(self):
        """ProactiveVisionMonitor 传给 gesture_callback 的 context 必须包含 confidence"""
        from unittest.mock import MagicMock, patch
        from lelamp.service.vision.proactive_vision_monitor import ProactiveVisionMonitor
        from lelamp.edge.hand_tracker import Gesture, HandInfo

        received_contexts = []

        def capture_callback(gesture, context):
            received_contexts.append(context)

        monitor = ProactiveVisionMonitor(
            gesture_callback=capture_callback,
            enable_auto_gesture=False,
            enable_auto_presence=False,
        )

        # 模拟 track_hands 返回带置信度的 HandInfo
        fake_hand = HandInfo(
            landmarks=[(0.5, 0.5, 0.0)] * 21,
            handedness="Right",
            gesture=Gesture.THUMBS_UP,
            confidence=0.92,
        )
        mock_hybrid = MagicMock()
        mock_hybrid.track_hands.return_value = {
            "gestures": [Gesture.THUMBS_UP],
            "hands": [fake_hand],
            "count": 1,
        }
        monitor._hybrid_vision = mock_hybrid
        monitor._user_present = True

        monitor._detect_gestures(None, current_time=0.0)

        assert len(received_contexts) == 1
        assert "confidence" in received_contexts[0]
        assert abs(received_contexts[0]["confidence"] - 0.92) < 0.01
```

**Step 2: 运行确认失败**
```bash
uv run pytest tests/test_edge_vision.py::TestGestureConfidence -v
```
期望：FAIL — `'confidence' not in context`

**Step 3: 修改 `_detect_gestures()`**

将 `proactive_vision_monitor.py` 中 `_detect_gestures` 方法里的 context 构建段（约第 314-323 行）改为：

```python
if gestures:
    self._gesture_count += 1
    self._last_gesture_time = current_time

    logger.info(f"检测到手势: {[g.value for g in gestures]}")

    # 构建 gesture → confidence 映射（从 hands 数据中提取）
    hands = result.get("hands", [])
    gesture_confidence: dict = {}
    for hand in hands:
        g = hand.gesture
        gesture_confidence[g] = max(gesture_confidence.get(g, 0.0), hand.confidence)

    if self._gesture_callback:
        try:
            for gesture in gestures:
                context = {
                    "timestamp": current_time,
                    "user_present": self._user_present,
                    "detection_count": self._detection_count,
                    "confidence": gesture_confidence.get(gesture, 0.5),
                }
                self._gesture_callback(gesture, context)
        except Exception as e:
            logger.error(f"Gesture callback error: {e}")
```

**Step 4: 运行确认通过**
```bash
uv run pytest tests/test_edge_vision.py::TestGestureConfidence -v
```
期望：PASS

**Step 5: 提交**
```bash
git add lelamp/service/vision/proactive_vision_monitor.py tests/test_edge_vision.py
git commit -m "feat(gesture): 注入 confidence 到 gesture_callback context"
```

---

## Task 2: Agent 手势置信度分层逻辑

**Files:**
- Modify: `lelamp/agent/lelamp_agent.py`

**Step 1: 在 `LeLamp.__init__` 初始化 event loop 引用和 motor 故障字典**

在 `__init__` 中 `self._pending_volume_set = 100` 那行（约第 337 行）**之前**插入：

```python
self._event_loop: Optional[asyncio.AbstractEventLoop] = None
self._motor_fault_notified: dict = {}  # motor_name -> HealthStatus
```

**Step 2: 在 `_initialize_async` 捕获 event loop**

将方法改为：

```python
async def _initialize_async(self) -> None:
    """异步初始化任务（在有事件循环时调用）"""
    if self._event_loop is None:
        self._event_loop = asyncio.get_running_loop()
    if hasattr(self, '_pending_volume_set'):
        await self._set_system_volume(self._pending_volume_set)
        delattr(self, '_pending_volume_set')
```

**Step 3: 在 `_initialize_async` 下面添加 `_speak_proactively` 方法**

```python
async def _speak_proactively(self, text: str) -> None:
    """从异步上下文主动发声（手势确认、故障提示等）"""
    try:
        if hasattr(self, "session") and self.session is not None:
            await self.session.say(text, allow_interruptions=True)
        else:
            logger.info(f"[speak_proactively] session not ready: {text}")
    except Exception as e:
        logger.warning(f"Proactive speech failed: {e}")
```

**Step 4: 在 `on_gesture` 定义前（约第 250 行）添加常量**

在 `if edge_vision_enabled and EDGE_VISION_AVAILABLE:` 块内，`def on_gesture` 之前插入：

```python
_GESTURE_HIGH_CONF = 0.80
_GESTURE_MID_CONF  = 0.60
_GESTURE_NAMES = {
    "thumbs_up": "点赞", "thumbs_down": "踩", "peace": "耶",
    "wave": "挥手", "fist": "握拳", "point": "指向",
    "ok": "OK", "open": "张开手掌",
}
```

**Step 5: 替换 `on_gesture` 闭包**

将现有 `on_gesture` 函数（约第 251-265 行）替换为：

```python
def on_gesture(gesture, context):
    confidence = context.get("confidence", 1.0)

    if confidence < _GESTURE_MID_CONF:
        logger.debug(f"手势 {gesture.value} 置信度过低 ({confidence:.2f})，忽略")
        return

    if confidence < _GESTURE_HIGH_CONF:
        gesture_name = _GESTURE_NAMES.get(gesture.value, gesture.value)
        logger.info(f"手势 {gesture.value} 置信度中等 ({confidence:.2f})，请求语音确认")
        if self._event_loop and self._event_loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self._speak_proactively(f"你是在比{gesture_name}吗？"),
                self._event_loop,
            )
        return

    logger.info(f"检测到手势: {gesture.value} (confidence={confidence:.2f})")
    if gesture.value == "thumbs_up":
        self.motors_service.dispatch("play", "nod")
    elif gesture.value == "thumbs_down":
        self.motors_service.dispatch("play", "shake")
    elif gesture.value == "peace":
        self.motors_service.dispatch("play", "excited")
    elif gesture.value == "wave":
        if self.rgb_service.is_on():
            self.rgb_service.dispatch("off")
        else:
            self.rgb_service.dispatch("solid", (255, 255, 255))
```

**Step 6: 运行测试**
```bash
uv run pytest tests/ -v -k "not hardware" 2>&1 | tail -20
```
期望：已有测试全部 PASS（不引入 regression）

**Step 7: 提交**
```bash
git add lelamp/agent/lelamp_agent.py
git commit -m "feat(gesture): 手势置信度三档分层，MID 置信请求语音确认，LOW 置信静默忽略"
```

---

## Task 3: MotorsService 舵机故障回调

**Files:**
- Modify: `lelamp/service/motors/motors_service.py`
- Test: `tests/test_motor_service_shutdown.py`

**Step 1: 写失败测试**

在 `tests/test_motor_service_shutdown.py` 末尾追加：

```python
def test_motor_fault_callback_on_critical():
    """舵机进入 CRITICAL 状态时触发 motor_fault_callback"""
    from unittest.mock import MagicMock, patch
    from lelamp.service.motors.motors_service import MotorsService
    from lelamp.service.motors.health_monitor import HealthStatus
    from lelamp.config import load_motor_config

    fault_events = []

    def capture_fault(motor_name, old_status, new_status):
        fault_events.append((motor_name, old_status, new_status))

    config = load_motor_config()
    config.health_check_enabled = False  # 不启动后台线程

    service = MotorsService(port="/dev/ttyACM0", lamp_id="test", motor_config=config)
    service._motor_fault_callback = capture_fault

    # 直接调用 _on_health_status_change（Task 3 新增方法）
    service._prev_motor_status = {"base_yaw": HealthStatus.HEALTHY}
    service._on_health_status_change("base_yaw", HealthStatus.HEALTHY, HealthStatus.CRITICAL)

    assert len(fault_events) == 1
    assert fault_events[0] == ("base_yaw", HealthStatus.HEALTHY, HealthStatus.CRITICAL)
```

**Step 2: 运行确认失败**
```bash
uv run pytest tests/test_motor_service_shutdown.py::test_motor_fault_callback_on_critical -v
```
期望：FAIL

**Step 3: 修改 `MotorsService`**

在 `__init__` 中 `self._bus_lock = threading.Lock()` 那行后添加：
```python
self._motor_fault_callback = None   # 外部注册：fault(motor_name, old, new)
self._prev_motor_status: dict = {}  # motor_name -> HealthStatus
```

在类中添加方法 `_on_health_status_change`（放在 `_health_check_loop` 之后）：
```python
def _on_health_status_change(self, motor_name: str, old_status, new_status) -> None:
    """检测到状态转换时调用（从 _health_check_loop 触发）"""
    if self._motor_fault_callback:
        try:
            self._motor_fault_callback(motor_name, old_status, new_status)
        except Exception as e:
            self.logger.warning(f"motor_fault_callback error: {e}")
```

在 `_health_check_loop` 的 `for motor_name, health_data in health_results.items():` 循环**开头**插入：
```python
old_status = self._prev_motor_status.get(motor_name)
new_status = health_data.status
if old_status != new_status:
    self._prev_motor_status[motor_name] = new_status
    if new_status in (HealthStatus.CRITICAL, HealthStatus.STALLED):
        self._on_health_status_change(motor_name, old_status, new_status)
```

**Step 4: 运行确认通过**
```bash
uv run pytest tests/test_motor_service_shutdown.py::test_motor_fault_callback_on_critical -v
```
期望：PASS

**Step 5: 提交**
```bash
git add lelamp/service/motors/motors_service.py tests/test_motor_service_shutdown.py
git commit -m "feat(motor): 添加 motor_fault_callback，状态转 CRITICAL/STALLED 时触发"
```

---

## Task 4: Agent 注册舵机故障回调

**Files:**
- Modify: `lelamp/agent/lelamp_agent.py`

**Step 1: 添加 `_on_motor_health_change` 方法**

在 `_speak_proactively` 方法之后添加：

```python
def _on_motor_health_change(self, motor_name: str, old_status, new_status) -> None:
    """舵机故障回调（运行在 health_check daemon 线程中）"""
    from lelamp.service.motors.health_monitor import HealthStatus
    # 去重：同一状态不重复通知
    if self._motor_fault_notified.get(motor_name) == new_status:
        return
    self._motor_fault_notified[motor_name] = new_status

    logger.warning(f"舵机故障通知: {motor_name} {old_status} → {new_status}")

    # LED 橙色呼吸（线程安全）
    self.rgb_service.dispatch("breath", {"rgb": (255, 80, 0), "period_s": 2.0})

    # 语音提示（调度到 asyncio 事件循环）
    _FAULT_MSGS = [
        "我今天有点不舒服，动作可能不太灵活",
        "我的关节好像有点问题，先凑合着用吧",
    ]
    import random
    msg = random.choice(_FAULT_MSGS)
    if self._event_loop and self._event_loop.is_running():
        asyncio.run_coroutine_threadsafe(
            self._speak_proactively(msg),
            self._event_loop,
        )
```

**Step 2: 注册回调**

在 `__init__` 中两个服务 `.start()` 之后（约第 204 行 `self.rgb_service.start()` 后）添加：

```python
# 注册舵机故障回调
if hasattr(self.motors_service, '_motor_fault_callback'):
    self.motors_service._motor_fault_callback = self._on_motor_health_change
```

**Step 3: 运行全量测试**
```bash
uv run pytest tests/ -v -k "not hardware" 2>&1 | tail -30
```
期望：全部 PASS

**Step 4: Lint 检查**
```bash
uv run ruff check lelamp/agent/lelamp_agent.py lelamp/service/motors/motors_service.py lelamp/service/vision/proactive_vision_monitor.py
```
期望：无错误

**Step 5: 提交**
```bash
git add lelamp/agent/lelamp_agent.py
git commit -m "feat(motor): 舵机故障时 LED 橙色呼吸 + 语音提示，情感化降级体验"
```
