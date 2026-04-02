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
