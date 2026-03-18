"""
WebSocket 路由 - 实时设备状态推送

提供 WebSocket 连接管理、消息广播、实时推送等功能。
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Set, Optional
import asyncio
import logging
from datetime import datetime
from openai import AsyncOpenAI
from lelamp.config import load_config

from lelamp.api.models.websocket import (
    WSPing,
    WSSubscribe,
    WSUnsubscribe,
    WSClientCommand,
    WSError,
    validate_client_message,
    is_valid_channel,
)

logger = logging.getLogger("lelamp.api.websocket")

router = APIRouter()


# =============================================================================
# 连接管理器
# =============================================================================


class ConnectionManager:
    """
    WebSocket 连接管理器

    管理所有活跃的 WebSocket 连接，支持：
    - 按设备 ID 分组连接
    - 向特定设备广播消息
    - 向所有客户端广播消息
    - 线程安全的连接管理
    """

    def __init__(self):
        # Dict: lamp_id -> Set[WebSocket]
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # Dict: lamp_id -> Set[channel names]
        self.subscriptions: Dict[str, Set[str]] = {}
        # 连接统计
        self._connection_counts: Dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, lamp_id: str) -> None:
        """
        接受连接并添加到设备连接池

        Args:
            websocket: WebSocket 连接对象
            lamp_id: 设备 ID
        """
        await websocket.accept()

        async with self._lock:
            if lamp_id not in self.active_connections:
                self.active_connections[lamp_id] = set()
                self.subscriptions[lamp_id] = set()
                self._connection_counts[lamp_id] = 0

            self.active_connections[lamp_id].add(websocket)
            self._connection_counts[lamp_id] += 1

        logger.info(f"WebSocket 连接建立: {lamp_id} (总连接数: {self.get_connection_count(lamp_id)})")

        # 发送连接确认
        await websocket.send_json({
            "type": "connected",
            "lamp_id": lamp_id,
            "server_time": datetime.utcnow().isoformat(),
            "message": "WebSocket connection established"
        })

    async def disconnect(self, websocket: WebSocket, lamp_id: str) -> None:
        """
        从设备连接池移除连接

        Args:
            websocket: WebSocket 连接对象
            lamp_id: 设备 ID
        """
        async with self._lock:
            if lamp_id in self.active_connections:
                self.active_connections[lamp_id].discard(websocket)
                self._connection_counts[lamp_id] -= 1

                # 清理空连接
                if not self.active_connections[lamp_id]:
                    del self.active_connections[lamp_id]
                    if lamp_id in self.subscriptions:
                        del self.subscriptions[lamp_id]
                    if lamp_id in self._connection_counts:
                        del self._connection_counts[lamp_id]

        logger.info(f"WebSocket 连接断开: {lamp_id}")

    async def send_personal_message(self, message: dict, websocket: WebSocket) -> bool:
        """
        向特定客户端发送消息

        Args:
            message: 消息内容
            websocket: 目标 WebSocket 连接

        Returns:
            是否发送成功
        """
        try:
            await websocket.send_json(message)
            return True
        except Exception as e:
            logger.error(f"发送个人消息失败: {e}")
            return False

    async def broadcast_to_device(self, lamp_id: str, message: dict, channel: Optional[str] = None) -> int:
        """
        向特定设备的所有连接广播消息

        Args:
            lamp_id: 设备 ID
            message: 消息内容
            channel: 频道过滤（可选）

        Returns:
            成功发送的连接数
        """
        if lamp_id not in self.active_connections:
            return 0

        # 检查频道订阅
        if channel and lamp_id in self.subscriptions:
            if channel not in self.subscriptions[lamp_id]:
                return 0

        sent_count = 0
        failed_connections = []

        for connection in self.active_connections[lamp_id]:
            try:
                await connection.send_json(message)
                sent_count += 1
            except Exception as e:
                logger.error(f"广播到 {lamp_id} 失败: {e}")
                failed_connections.append(connection)

        # 清理失败的连接
        for connection in failed_connections:
            await self.disconnect(connection, lamp_id)

        return sent_count

    async def broadcast_to_all(self, message: dict) -> int:
        """
        向所有连接的客户端广播消息

        Args:
            message: 消息内容

        Returns:
            成功发送的连接数
        """
        sent_count = 0
        all_connections = []

        # 收集所有连接
        async with self._lock:
            for connections in self.active_connections.values():
                all_connections.extend(list(connections))

        # 发送消息
        failed_connections = []
        for connection in all_connections:
            try:
                await connection.send_json(message)
                sent_count += 1
            except Exception as e:
                logger.error(f"广播失败: {e}")
                failed_connections.append(connection)

        # 清理失败的连接
        # （需要找到对应的 lamp_id）
        for connection in failed_connections:
            for lamp_id, connections in self.active_connections.items():
                if connection in connections:
                    await self.disconnect(connection, lamp_id)
                    break

        return sent_count

    def subscribe(self, lamp_id: str, channels: Set[str]) -> None:
        """
        订阅频道

        Args:
            lamp_id: 设备 ID
            channels: 频道集合
        """
        if lamp_id not in self.subscriptions:
            self.subscriptions[lamp_id] = set()

        # 只添加有效频道
        valid_channels = {ch for ch in channels if is_valid_channel(ch)}
        self.subscriptions[lamp_id].update(valid_channels)

        logger.info(f"设备 {lamp_id} 订阅频道: {valid_channels}")

    def unsubscribe(self, lamp_id: str, channels: Set[str]) -> None:
        """
        取消订阅频道

        Args:
            lamp_id: 设备 ID
            channels: 频道集合
        """
        if lamp_id in self.subscriptions:
            self.subscriptions[lamp_id] -= channels
            logger.info(f"设备 {lamp_id} 取消订阅频道: {channels}")

    def get_connection_count(self, lamp_id: str) -> int:
        """
        获取指定设备的活跃连接数

        Args:
            lamp_id: 设备 ID

        Returns:
            连接数
        """
        return self._connection_counts.get(lamp_id, 0)

    def get_all_connection_counts(self) -> Dict[str, int]:
        """
        获取所有设备的连接数

        Returns:
            设备 ID 到连接数的映射
        """
        return dict(self._connection_counts)

    def get_subscriptions(self, lamp_id: str) -> Set[str]:
        """
        获取设备订阅的频道

        Args:
            lamp_id: 设备 ID

        Returns:
            频道集合
        """
        return self.subscriptions.get(lamp_id, set())


# 全局连接管理器实例
manager = ConnectionManager()


# =============================================================================
# WebSocket 端点
# =============================================================================


@router.websocket("/{lamp_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    lamp_id: str
):
    """
    WebSocket 实时推送端点

    支持的功能：
    - 实时接收设备状态更新
    - 接收操作日志和事件通知
    - 发送设备命令
    - 心跳检测

    消息类型：
    - 客户端 -> 服务端:
      * ping: 心跳
      * subscribe: 订阅频道
      * unsubscribe: 取消订阅
      * command: 发送命令

    - 服务端 -> 客户端:
      * pong: 心跳响应
      * connected: 连接确认
      * state_update: 状态更新
      * event: 事件通知
      * log: 日志消息
      * notification: 通知消息

    Args:
        websocket: WebSocket 连接
        lamp_id: 设备 ID
    """
    await manager.connect(websocket, lamp_id)

    try:
        while True:
            # 接收客户端消息
            data = await websocket.receive_json()

            # 验证消息格式
            message = validate_client_message(data)
            if message is None:
                await websocket.send_json(
                    WSError(
                        message=f"无效的消息类型: {data.get('type')}",
                        code="INVALID_MESSAGE_TYPE"
                    ).model_dump()
                )
                continue

            # 处理不同类型的消息
            if isinstance(message, WSPing):
                # 心跳响应
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                })

            elif isinstance(message, WSSubscribe):
                # 订阅频道
                channels = set(message.channels)
                valid_channels = {ch for ch in channels if is_valid_channel(ch)}

                if valid_channels:
                    manager.subscribe(lamp_id, valid_channels)
                    await websocket.send_json({
                        "type": "subscription_confirmed",
                        "channels": list(valid_channels),
                        "timestamp": datetime.utcnow().isoformat()
                    })
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": "没有有效的频道",
                        "code": "INVALID_CHANNELS",
                        "timestamp": datetime.utcnow().isoformat()
                    })

            elif isinstance(message, WSUnsubscribe):
                # 取消订阅
                channels = set(message.channels)
                manager.unsubscribe(lamp_id, channels)

            elif isinstance(message, WSClientCommand):
                # 处理客户端命令
                logger.info(f"收到客户端命令: {message.action}, 参数: {message.params}")
                action = message.action
                params = message.params or {}
                
                rgb_service = getattr(websocket.app.state, "rgb_service", None)
                motors_service = getattr(websocket.app.state, "motors_service", None)
                
                if action == "set_rgb_solid" and rgb_service:
                    r = params.get("r", 0)
                    g = params.get("g", 0)
                    b = params.get("b", 0)
                    rgb_service.dispatch("solid", (r, g, b))
                elif action.startswith("rgb_effect_") and rgb_service:
                    effect_name = action.replace("rgb_effect_", "")
                    rgb_service.dispatch("effect", {"name": effect_name})
                elif action == "stop_rgb_effect" and rgb_service:
                    rgb_service.dispatch("clear", None)
                elif action == "move_joint" and motors_service:
                    joint_name = params.get("joint_name")
                    angle = params.get("angle")
                    if joint_name and angle is not None:
                        motors_service.dispatch("move_joint", {"joint": joint_name, "angle": float(angle)})
                elif action == "play_recording" and motors_service:
                    recording_name = params.get("recording_name")
                    if recording_name:
                        motors_service.dispatch("play", recording_name)
                elif action == "chat":
                    text = params.get("text", "")
                    logger.info(f"收到聊天消息: {text}")
                    
                    # 异步调用 DeepSeek API 进行回复
                    async def process_chat(message_text: str):
                        try:
                            config = load_config()
                            # 简单的测试回复，如果不配置 api key 的话
                            if not config.deepseek_api_key or config.deepseek_api_key == "dummy":
                                await websocket.send_json({
                                    "type": "chat",
                                    "content": f"收到你的消息：{message_text}。请在配置中设置 DEEPSEEK_API_KEY 以启用智能回复。"
                                })
                                return

                            client = AsyncOpenAI(
                                api_key=config.deepseek_api_key,
                                base_url=config.deepseek_base_url
                            )
                            
                            response = await client.chat.completions.create(
                                model=config.deepseek_model,
                                messages=[
                                    {"role": "system", "content": "你是一个名为 LeLamp 的智能台灯机器人，性格带点讽刺但乐于助人，请用简短的中文回复。"},
                                    {"role": "user", "content": message_text}
                                ],
                                max_tokens=150
                            )
                            
                            reply_text = response.choices[0].message.content
                            
                            await websocket.send_json({
                                "type": "chat",
                                "content": reply_text
                            })
                            
                        except Exception as e:
                            logger.error(f"调用大模型失败: {e}")
                            await websocket.send_json({
                                "type": "chat",
                                "content": "抱歉，我的大脑暂时断线了。"
                            })
                            
                    asyncio.create_task(process_chat(text))


    except WebSocketDisconnect:
        logger.info(f"WebSocket 正常断开: {lamp_id}")
    except Exception as e:
        logger.error(f"WebSocket 错误 ({lamp_id}): {e}", exc_info=True)
    finally:
        await manager.disconnect(websocket, lamp_id)


# =============================================================================
# 辅助函数
# =============================================================================


async def push_state_update(lamp_id: str, state_data: dict) -> int:
    """
    推送状态更新到设备订阅者

    Args:
        lamp_id: 设备 ID
        state_data: 状态数据

    Returns:
        推送成功的连接数
    """
    message = {
        "type": "state_update",
        "data": state_data,
        "timestamp": datetime.utcnow().isoformat()
    }
    return await manager.broadcast_to_device(lamp_id, message, channel="state")


async def push_event(lamp_id: str, event_type: str, event_data: dict) -> int:
    """
    推送事件到设备订阅者

    Args:
        lamp_id: 设备 ID
        event_type: 事件类型
        event_data: 事件数据

    Returns:
        推送成功的连接数
    """
    message = {
        "type": "event",
        "event_type": event_type,
        "data": event_data,
        "timestamp": datetime.utcnow().isoformat()
    }
    return await manager.broadcast_to_device(lamp_id, message, channel="events")


async def push_log(lamp_id: str, log_entry: dict) -> int:
    """
    推送日志到设备订阅者

    Args:
        lamp_id: 设备 ID
        log_entry: 日志条目

    Returns:
        推送成功的连接数
    """
    message = {
        "type": "log",
        "log_entry": log_entry,
        "timestamp": datetime.utcnow().isoformat()
    }
    return await manager.broadcast_to_device(lamp_id, message, channel="logs")


async def push_notification(
    lamp_id: str,
    message: str,
    level: str = "info",
    metadata: Optional[dict] = None
) -> int:
    """
    推送通知到设备订阅者

    Args:
        lamp_id: 设备 ID
        message: 通知消息
        level: 通知级别 (info/warning/error)
        metadata: 额外元数据

    Returns:
        推送成功的连接数
    """
    notification = {
        "type": "notification",
        "message": message,
        "level": level,
        "timestamp": datetime.utcnow().isoformat()
    }

    if metadata:
        notification["metadata"] = metadata

    return await manager.broadcast_to_device(lamp_id, notification, channel="notifications")


# 导出管理器和辅助函数
__all__ = [
    "manager",
    "push_state_update",
    "push_event",
    "push_log",
    "push_notification",
]
