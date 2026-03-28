"""
WebSocket 路由 - 实时设备状态推送

提供 WebSocket 连接管理、消息广播、实时推送等功能。
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, status
from typing import Dict, Set, Optional
import asyncio
import logging
from datetime import datetime
from openai import AsyncOpenAI
from lelamp.config import load_config
from lelamp.api.services.auth_service import AuthService

from lelamp.api.models.websocket import (
    WSPing,
    WSSubscribe,
    WSUnsubscribe,
    WSClientCommand,
    WSError,
    validate_client_message,
    is_valid_channel,
)
from lelamp.service.base import Priority

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
        connection_msg = {
            "type": "connected",
            "lamp_id": lamp_id,
            "server_time": datetime.utcnow().isoformat(),
            "message": "WebSocket connection established"
        }

        await websocket.send_json(connection_msg)

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
    lamp_id: str,
    token: Optional[str] = Query(None)
):
    """
    WebSocket 实时推送端点

    支持的功能：
    - 实时接收设备状态更新
    - 接收操作日志和事件通知
    - 发送设备命令
    - 心跳检测
    - JWT 认证 (通过查询参数传递 token)

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
        token: JWT 认证令牌 (可选,用于识别用户)
    """
    # 验证 JWT token (可选)
    user_info = None
    if token:
        payload = AuthService.verify_token(token, "access")
        if payload:
            user_info = {
                "username": payload.get("sub"),
                "user_id": payload.get("user_id")
            }
            logger.info(f"WebSocket 用户认证成功: {user_info['username']}")
        else:
            logger.warning("WebSocket token 无效,但允许匿名连接")

    await manager.connect(websocket, lamp_id)

    try:
        while True:
            # 接收客户端消息
            data = await websocket.receive_json()
            logger.info(f"收到WebSocket消息: {data}")

            # 验证消息格式
            message = validate_client_message(data)
            if message is None:
                logger.warning(f"消息验证失败: type={data.get('type')}, data={data}")
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
                agent = getattr(websocket.app.state, "agent", None)
                
                if action == "chat":
                    text = params.get("text", "")
                    logger.info(f"收到聊天消息: {text}")
                    
                    # 异步调用 DeepSeek API 进行回复
                    async def process_chat(message_text: str):
                        try:
                            if agent:
                                await agent.set_conversation_state("thinking")

                            config = load_config()
                            # 简单的测试回复，如果不配置 api key 的话
                            if not config.deepseek_api_key or config.deepseek_api_key == "dummy":
                                reply_text = f"收到你的消息：{message_text}。请在配置中设置 DEEPSEEK_API_KEY 以启用智能回复。"
                            else:
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
                                
                            if agent:
                                await agent._send_chat_message(reply_text)
                                await agent.set_conversation_state("speaking")
                                await asyncio.sleep(2.0)  # 简单模拟说话时间
                                await agent.set_conversation_state("idle")
                            else:
                                await websocket.send_json({
                                    "type": "chat",
                                    "content": reply_text
                                })
                            
                        except Exception as e:
                            logger.error(f"调用大模型失败: {e}")
                            if agent:
                                await agent._send_chat_message("抱歉，我的大脑暂时断线了。")
                                await agent.set_conversation_state("idle")
                            else:
                                await websocket.send_json({
                                    "type": "chat",
                                    "content": "抱歉，我的大脑暂时断线了。"
                                })
                            
                    asyncio.create_task(process_chat(text))
                else:
                    # 对于非聊天的其他指令，直接使用硬件服务处理
                    try:
                        success = False
                        result = None
                        error_message = None

                        logger.info(f"处理命令: action={action}, params={params}")
                        logger.info(f"服务状态: agent={agent is not None}, rgb_service={rgb_service is not None}, motors_service={motors_service is not None}")

                        # 直接调用硬件服务执行命令
                        logger.info("使用硬件服务执行命令")
                        success = await execute_direct_command(action, params, rgb_service, motors_service, agent)
                        if not success:
                            error_message = "命令执行失败或不支持"
                            logger.warning(f"命令执行失败: {error_message}")
                        else:
                            logger.info("命令执行成功")

                        # 发送执行结果反馈
                        response = {
                            "type": "command_result",
                            "action": action,
                            "success": success,
                            "result": result,
                            "error": error_message,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                        await websocket.send_json(response)
                        logger.info(f"已发送命令执行结果: {success}")

                    except Exception as e:
                        logger.error(f"命令执行异常: {e}", exc_info=True)
                        await websocket.send_json({
                            "type": "command_result",
                            "action": action,
                            "success": False,
                            "error": str(e),
                            "timestamp": datetime.utcnow().isoformat()
                        })


    except WebSocketDisconnect:
        logger.info(f"WebSocket 正常断开: {lamp_id}, user: {user_info['username'] if user_info else 'anonymous'}")
    except Exception as e:
        logger.error(f"WebSocket 错误 ({lamp_id}): {e}", exc_info=True)
    finally:
        await manager.disconnect(websocket, lamp_id)


# =============================================================================
# 辅助函数
# =============================================================================


async def execute_direct_command(action: str, params: dict, rgb_service, motors_service, agent) -> bool:
    """
    直接命令执行 - 直接操作硬件服务

    Args:
        action: 命令动作
        params: 命令参数
        rgb_service: RGB 服务实例
        motors_service: 电机服务实例
        agent: LeLamp agent 实例（可选）

    Returns:
        bool: 命令是否成功执行
    """
    try:
        # RGB 固定颜色命令
        if action == "set_rgb_solid" and rgb_service:
            r = params.get("r", params.get("red", 0))
            g = params.get("g", params.get("green", 0))
            b = params.get("b", params.get("blue", 0))
            rgb_service.dispatch("solid", (r, g, b), priority=Priority.HIGH)
            return True

        # RGB 效果命令
        elif action in ["rgb_effect_rainbow", "rgb_effect_breathing", "rgb_effect_wave", "rgb_effect_fire", "rgb_effect_emoji"] and rgb_service:
            effect_name = action.replace("rgb_effect_", "")

            # 呼吸效果使用 "breath" 事件类型
            if effect_name == "breathing":
                r = params.get("r", params.get("red", 0))
                g = params.get("g", params.get("green", 150))
                b = params.get("b", params.get("blue", 255))
                rgb_service.dispatch("breath", {"rgb": (r, g, b)}, priority=Priority.HIGH)
            else:
                effect_params = {**params, "name": effect_name}
                rgb_service.dispatch("effect", effect_params, priority=Priority.HIGH)
            return True

        # 停止 RGB 效果
        elif action == "stop_rgb_effect" and rgb_service:
            rgb_service.dispatch("clear", None, priority=Priority.HIGH)
            return True

        # RGB 图案绘制
        elif action == "paint_rgb_pattern" and rgb_service:
            pattern = params.get("pattern", "")
            if pattern:
                rgb_service.dispatch("paint", pattern, priority=Priority.HIGH)
                return True

        # RGB 亮度设置
        elif action == "set_rgb_brightness" and rgb_service:
            percent = params.get("percent", 25)
            rgb_service.dispatch("brightness", percent, priority=Priority.HIGH)
            return True

        # 电机移动命令
        elif action == "move_joint" and motors_service:
            joint_name = params.get("joint_name")
            angle = params.get("angle")
            if joint_name and angle is not None:
                motors_service.dispatch("move_joint", {"joint": joint_name, "angle": float(angle)}, priority=Priority.HIGH)
                return True

        # 播放录制动作
        elif action == "play_recording" and motors_service:
            recording_name = params.get("recording_name")
            if recording_name:
                motors_service.dispatch("play", recording_name, priority=Priority.HIGH)
                return True

        # 音量设置
        elif action == "set_volume" and agent:
            volume_percent = params.get("volume_percent", params.get("volume", 50))
            # 异步调用 agent 的方法
            if hasattr(agent, '_set_system_volume'):
                import asyncio
                asyncio.create_task(agent._set_system_volume(volume_percent))
                return True

        # 摄像头激活命令
        elif action == "activate_camera":
            vision_service = getattr(websocket.app.state, "vision_service", None)
            if vision_service:
                try:
                    # 检查是否需要隐私保护
                    if hasattr(vision_service, 'enable_privacy_protection') and vision_service.enable_privacy_protection:
                        # 授予用户同意
                        vision_service.grant_camera_consent()
                    # 激活摄像头
                    result = await vision_service.activate_camera()
                    # 推送状态更新
                    from lelamp.api.routes.websocket import push_camera_status
                    await push_camera_status(lamp_id, True, True)
                    return result
                except Exception as e:
                    logger.error(f"Camera activation failed: {e}")
                    return False
            return False

        # 不支持的命令
        logger.warning(f"不支持的命令: {action}")
        return False

    except Exception as e:
        logger.error(f"命令执行失败: {e}", exc_info=True)
        return False


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


async def push_camera_frame(
    lamp_id: str,
    frame_b64: str,
    width: Optional[int] = None,
    height: Optional[int] = None
) -> int:
    """
    推送摄像头帧到设备订阅者

    Args:
        lamp_id: 设备 ID
        frame_b64: Base64 编码的 JPEG 图像数据
        width: 图像宽度（可选）
        height: 图像高度（可选）

    Returns:
        推送成功的连接数
    """
    message = {
        "type": "camera_frame",
        "frame_b64": frame_b64,
        "width": width,
        "height": height,
        "timestamp": datetime.utcnow().isoformat()
    }
    # 不使用频道过滤，直接推送给所有连接的客户端
    return await manager.broadcast_to_device(lamp_id, message)


async def push_camera_status(
    lamp_id: str,
    active: bool,
    privacy_granted: Optional[bool] = None
) -> int:
    """
    推送摄像头状态到设备订阅者

    Args:
        lamp_id: 设备 ID
        active: 摄像头是否激活
        privacy_granted: 隐私同意状态（可选）

    Returns:
        推送成功的连接数
    """
    message = {
        "type": "camera_status",
        "active": active,
        "timestamp": datetime.utcnow().isoformat()
    }

    if privacy_granted is not None:
        message["privacy_granted"] = privacy_granted

    return await manager.broadcast_to_device(lamp_id, message)


# 导出管理器和辅助函数
__all__ = [
    "manager",
    "push_state_update",
    "push_event",
    "push_log",
    "push_notification",
    "push_camera_frame",
    "push_camera_status",
]
