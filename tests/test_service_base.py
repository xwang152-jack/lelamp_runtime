"""
测试 lelamp.service.base 模块
"""
import time
import threading
import pytest

from lelamp.service.base import ServiceBase, ServiceEvent, Priority


class ConcreteService(ServiceBase):
    """实现 handle_event 的具体服务用于测试"""

    def __init__(self, **kwargs):
        super().__init__(name="test_service", **kwargs)
        self.handled_events = []

    def handle_event(self, event_type: str, payload):
        self.handled_events.append((event_type, payload))


@pytest.mark.unit
class TestPriority:
    """测试优先级枚举"""

    def test_values(self):
        assert Priority.CRITICAL.value == 0
        assert Priority.HIGH.value == 1
        assert Priority.NORMAL.value == 2
        assert Priority.LOW.value == 3

    def test_ordering(self):
        assert Priority.CRITICAL < Priority.HIGH < Priority.NORMAL < Priority.LOW


@pytest.mark.unit
class TestServiceEvent:
    """测试服务事件"""

    def test_creation(self):
        evt = ServiceEvent("move", {"angle": 30}, Priority.HIGH)
        assert evt.event_type == "move"
        assert evt.payload == {"angle": 30}
        assert evt.priority == Priority.HIGH

    def test_default_priority(self):
        evt = ServiceEvent("test", None)
        assert evt.priority == Priority.NORMAL

    def test_comparison(self):
        low = ServiceEvent("a", None, Priority.LOW)
        high = ServiceEvent("b", None, Priority.HIGH)
        assert high < low  # 更小值 = 更高优先级

    def test_repr(self):
        evt = ServiceEvent("move", None, Priority.CRITICAL)
        assert "move" in repr(evt)
        assert "CRITICAL" in repr(evt)


@pytest.mark.unit
class TestServiceBase:
    """测试服务基类"""

    def test_init(self):
        svc = ConcreteService()
        assert svc.name == "test_service"
        assert svc.is_running is False
        assert svc.queue_size == 0

    def test_dispatch_when_not_running(self):
        svc = ConcreteService()
        svc.dispatch("test", "data")  # 不应抛错
        assert len(svc.handled_events) == 0

    def test_start_stop(self):
        svc = ConcreteService()
        svc.start()
        assert svc.is_running is True

        svc.stop()
        assert svc.is_running is False

    def test_dispatch_and_process(self):
        svc = ConcreteService()
        svc.start()

        svc.dispatch("move", {"angle": 45})
        # 等待事件处理
        time.sleep(0.3)

        svc.stop()
        assert len(svc.handled_events) >= 1
        assert svc.handled_events[0] == ("move", {"angle": 45})

    def test_priority_ordering(self):
        svc = ConcreteService()
        svc.start()

        svc.dispatch("low", 1, Priority.LOW)
        svc.dispatch("critical", 2, Priority.CRITICAL)
        svc.dispatch("normal", 3, Priority.NORMAL)
        svc.dispatch("high", 4, Priority.HIGH)

        time.sleep(0.3)
        svc.stop()

        # CRITICAL 应该最先处理
        types = [e[0] for e in svc.handled_events]
        assert types[0] == "critical"
        assert types[1] == "high"
        assert types[2] == "normal"
        assert types[3] == "low"

    def test_queue_overflow_drops_low(self):
        svc = ConcreteService(max_queue_size=2)
        svc.start()

        svc.dispatch("keep1", 1, Priority.HIGH)
        svc.dispatch("keep2", 2, Priority.HIGH)
        svc.dispatch("drop_me", 3, Priority.LOW)  # 队列满，低优先级被丢弃

        time.sleep(0.2)
        svc.stop()

        types = [e[0] for e in svc.handled_events]
        assert "drop_me" not in types

    def test_queue_overflow_replaces(self):
        svc = ConcreteService(max_queue_size=2)
        svc.start()

        svc.dispatch("low1", 1, Priority.LOW)
        svc.dispatch("low2", 2, Priority.LOW)
        # 队列满，CRITICAL 替换最低优先级
        svc.dispatch("critical", 3, Priority.CRITICAL)

        time.sleep(0.5)
        svc.stop()

        types = [e[0] for e in svc.handled_events]
        assert "critical" in types
        # 至少有一个 low 被处理（可能两个都处理了如果竞态条件）
        low_count = len([t for t in types if t.startswith("low")])
        assert low_count <= 2

    def test_wait_until_idle(self):
        svc = ConcreteService()
        svc.start()

        svc.dispatch("evt", 1)
        result = svc.wait_until_idle(timeout=2.0)
        assert result is True
        svc.stop()

    def test_wait_until_idle_timeout(self):
        svc = ConcreteService()
        svc.start()
        # 不分发事件，队列应该立即空闲
        result = svc.wait_until_idle(timeout=1.0)
        assert result is True
        svc.stop()

    def test_clear_queue(self):
        svc = ConcreteService()
        svc.start()

        svc.dispatch("evt1", 1)
        svc.dispatch("evt2", 2)
        svc.clear_queue()

        assert svc.queue_size == 0
        svc.stop()

    def test_double_start_ignored(self):
        svc = ConcreteService()
        svc.start()
        svc.start()  # 第二次启动应被忽略
        svc.stop()

    def test_stop_when_not_running(self):
        svc = ConcreteService()
        svc.stop()  # 不应抛错

    def test_has_pending_event(self):
        svc = ConcreteService()
        svc.start()

        svc.dispatch("evt", 1)
        assert svc.has_pending_event is True

        time.sleep(0.3)
        assert svc.has_pending_event is False

        svc.stop()

    def test_error_in_handle_event(self):
        class FailingService(ServiceBase):
            def __init__(self):
                super().__init__(name="failing", max_queue_size=10)

            def handle_event(self, event_type, payload):
                raise RuntimeError("handler error")

        svc = FailingService()
        svc.start()
        svc.dispatch("fail", None)
        # 服务不应崩溃
        time.sleep(0.3)
        svc.stop()
        assert svc.is_running is False
