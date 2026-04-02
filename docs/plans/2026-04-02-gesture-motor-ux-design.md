# 手势置信度分层 + 舵机故障情感化降级 — 设计文档

**日期：** 2026-04-02
**状态：** 已确认，待实施

---

## 背景

本设计源于产品评估后的改进建议（P1 优先级）：
1. 手势识别当前无置信度分层，低质量识别结果与高质量结果等同处理，导致误触发
2. 舵机故障进入 NoOp 后用户无感知，"伙伴不动了"体验很差但系统无任何提示

---

## 功能 1：手势置信度分层

### 设计目标
- HIGH（≥0.80）：直接执行动作，保持现有体验
- MID（0.60–0.80）：语音确认后执行，避免误触发
- LOW（<0.60）：静默忽略

### 数据流

```
HandTracker._classify_gesture()
  → 计算并记录 HandInfo.confidence（已有字段）

ProactiveVisionMonitor._detect_gestures()
  → context 字典注入 "confidence": float

lelamp_agent.on_gesture(gesture, context)
  → confidence >= 0.80 → 直接执行（现有逻辑）
  → 0.60 <= confidence < 0.80 → say("你是在比[手势名]吗？")，等待确认
  → confidence < 0.60 → 静默忽略
```

### 常量
```python
_GESTURE_HIGH_CONFIDENCE = 0.80
_GESTURE_MID_CONFIDENCE  = 0.60
```

### 手势中文名映射（用于语音确认）
```python
_GESTURE_NAMES = {
    "thumbs_up":   "点赞",
    "thumbs_down": "踩",
    "peace":       "耶",
    "wave":        "挥手",
    "fist":        "握拳",
    "point":       "指向",
    "ok":          "OK",
    "open":        "张开手掌",
}
```

### 涉及文件
| 文件 | 修改内容 |
|------|----------|
| `lelamp/edge/hand_tracker.py` | `_classify_gesture()` 返回 confidence；`HandInfo.confidence` 赋值 |
| `lelamp/service/vision/proactive_vision_monitor.py` | `_detect_gestures()` 中 context 注入 confidence |
| `lelamp/agent/lelamp_agent.py` | `on_gesture()` 加置信度分层逻辑 |
| `tests/test_edge_vision.py` | 补充置信度分层测试 |

---

## 功能 2：舵机故障情感化降级

### 设计目标
- 舵机进入 CRITICAL/STALLED 状态时：LED 切换"不舒服"表情 + 语音告知用户
- 恢复 HEALTHY 时：LED 切回正常，静默（不打断对话）

### 数据流

```
MotorHealthMonitor.check_motor_health()
  → 检测 HEALTHY → CRITICAL/STALLED 状态转换
  → 调用 status_change_callback(motor_name, old_status, new_status)

lelamp_agent._on_motor_health_change(motor_name, old_status, new_status)
  → 在 daemon 线程中被调用（线程安全约束）
  → rgb_service.dispatch("expression", "uncomfortable")  # 线程安全
  → asyncio.get_event_loop().call_soon_threadsafe(...)  # 传递语音任务

Agent 事件循环
  → 消费语音队列
  → 调用 say("我今天有点不舒服，动作可能不太灵活") 或类似文案
```

### 状态去重策略
- 每个 motor_name 维护上次通知的状态，相同状态不重复通知
- `_motor_fault_notified: dict[str, HealthStatus]` 追踪

### 线程安全约束
- LED 调用：`rgb_service.dispatch()` 基于 heapq 优先级队列，已线程安全
- 语音调用：必须通过 `call_soon_threadsafe()` 传递到 asyncio 事件循环
- 回调注册：在 Agent `__init__` 主线程中完成，无需加锁

### 涉及文件
| 文件 | 修改内容 |
|------|----------|
| `lelamp/service/motors/health_monitor.py` | 新增 `status_change_callback` 参数；状态转换检测逻辑 |
| `lelamp/agent/lelamp_agent.py` | 注册回调；`_on_motor_health_change()`；语音队列消费 |
| `tests/test_motor_service_shutdown.py` | 补充健康回调触发测试 |

---

## 非目标（本次不实施）

- 儿童 ASR 优化（需要外部服务支持）
- 家长使用时长管控（Phase 1 前移，下一轮实施）
- Pi 性能监控端点（P2，下一轮实施）
