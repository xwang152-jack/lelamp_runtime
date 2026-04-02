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
