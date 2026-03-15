import threading
import time
import heapq
from abc import ABC, abstractmethod
from typing import Any, Optional, List
from enum import IntEnum
import logging


class Priority(IntEnum):
    LOW = 3
    NORMAL = 2
    HIGH = 1
    CRITICAL = 0


class ServiceEvent:
    def __init__(self, event_type: str, payload: Any, priority: Priority = Priority.NORMAL):
        self.event_type = event_type
        self.payload = payload
        self.priority = priority

    def __lt__(self, other):
        return self.priority < other.priority

    def __repr__(self):
        return f"ServiceEvent(type={self.event_type}, priority={self.priority.name})"


class ServiceBase(ABC):
    def __init__(self, name: str, max_queue_size: int = 100):
        """
        初始化服务基类

        Args:
            name: 服务名称，用于日志记录
            max_queue_size: 事件队列最大大小，防止内存溢出
        """
        self.name = name
        self.max_queue_size = max_queue_size

        # 使用真正的优先级队列（heapq）而不是单事件缓冲区
        self._event_queue: List[ServiceEvent] = []
        self._queue_lock = threading.Lock()
        self._queue_not_empty = threading.Condition(self._queue_lock)

        self._worker_thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._stop_event = threading.Event()
        self.logger = logging.getLogger(f"service.{name}")

        # 统计信息
        self._events_dispatched = 0
        self._events_processed = 0
        self._events_dropped = 0

    def dispatch(self, event_type: str, payload: Any, priority: Priority = Priority.NORMAL):
        """
        分发事件到服务的事件队列

        Args:
            event_type: 事件类型
            payload: 事件负载数据
            priority: 事件优先级（CRITICAL > HIGH > NORMAL > LOW）
        """
        if not self._running.is_set():
            self.logger.warning(f"Service {self.name} is not running, ignoring event {event_type}")
            return

        event = ServiceEvent(event_type, payload, priority)

        with self._queue_not_empty:
            # 检查队列是否已满
            if len(self._event_queue) >= self.max_queue_size:
                # 队列满时，只有高优先级事件才能替换低优先级事件
                if event.priority < self._event_queue[-1].priority:
                    dropped = heapq.heappushpop(self._event_queue, event)
                    self._events_dropped += 1
                    self.logger.warning(
                        f"Event queue full, dropped lower priority event: {dropped}"
                    )
                else:
                    self._events_dropped += 1
                    self.logger.warning(
                        f"Event queue full, dropping event: {event}"
                    )
                    return

            heapq.heappush(self._event_queue, event)
            self._events_dispatched += 1
            self._queue_not_empty.notify()
            self.logger.debug(f"Dispatched event {event_type} with priority {priority.name}")

    def start(self):
        """启动服务事件处理循环"""
        if self._running.is_set():
            self.logger.warning(f"Service {self.name} is already running")
            return

        self._running.set()
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._event_loop, daemon=True)
        self._worker_thread.start()
        self.logger.info(f"Service {self.name} started")

    def stop(self, timeout: float = 5.0):
        """停止服务"""
        if not self._running.is_set():
            self.logger.warning(f"Service {self.name} is not running")
            return

        self.logger.info(f"Stopping service {self.name}")
        self._stop_event.set()
        self._running.clear()

        # 唤醒等待的线程
        with self._queue_not_empty:
            self._queue_not_empty.notify_all()

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)
            if self._worker_thread.is_alive():
                self.logger.warning(f"Service {self.name} did not stop within timeout")
            else:
                self.logger.info(f"Service {self.name} stopped")

        # 记录统计信息
        self.logger.info(
            f"Service {self.name} statistics: "
            f"dispatched={self._events_dispatched}, "
            f"processed={self._events_processed}, "
            f"dropped={self._events_dropped}, "
            f"remaining={len(self._event_queue)}"
        )

    def _event_loop(self):
        """事件处理循环"""
        while self._running.is_set():
            event = None

            # 从队列中获取事件
            with self._queue_not_empty:
                # 等待队列非空或停止信号
                while len(self._event_queue) == 0 and not self._stop_event.is_set():
                    self._queue_not_empty.wait(timeout=0.1)

                if self._stop_event.is_set():
                    break

                if len(self._event_queue) > 0:
                    # 使用 heapq.heappop 获取最高优先级事件
                    event = heapq.heappop(self._event_queue)

            if event:
                try:
                    self.handle_event(event.event_type, event.payload)
                    self._events_processed += 1
                except Exception as e:
                    self.logger.error(f"Error handling event {event.event_type}: {e}", exc_info=True)

    @abstractmethod
    def handle_event(self, event_type: str, payload: Any):
        """
        处理事件的抽象方法，子类必须实现

        Args:
            event_type: 事件类型
            payload: 事件负载数据
        """
        pass

    @property
    def is_running(self) -> bool:
        """检查服务是否正在运行"""
        return self._running.is_set()

    @property
    def queue_size(self) -> int:
        """获取当前队列大小"""
        with self._queue_lock:
            return len(self._event_queue)

    @property
    def has_pending_event(self) -> bool:
        """检查是否有待处理的事件"""
        return self.queue_size > 0

    def wait_until_idle(self, timeout: Optional[float] = None) -> bool:
        """
        等待直到没有待处理的事件

        Args:
            timeout: 超时时间（秒），None 表示无限等待

        Returns:
            True 如果队列为空，False 如果超时
        """
        start_time = None
        if timeout is not None:
            start_time = time.time()

        while self.has_pending_event:
            if timeout is not None and start_time:
                if time.time() - start_time > timeout:
                    return False
            time.sleep(0.01)

        return True

    def clear_queue(self):
        """清空事件队列"""
        with self._queue_not_empty:
            cleared_count = len(self._event_queue)
            self._event_queue.clear()
            self.logger.info(f"Cleared {cleared_count} events from queue")
