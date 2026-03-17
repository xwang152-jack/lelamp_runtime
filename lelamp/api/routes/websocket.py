"""
WebSocket 路由 - 实时设备状态推送
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Set
import logging

logger = logging.getLogger("lelamp.api.websocket")

router = APIRouter()

# 连接管理
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, lamp_id: str):
        await websocket.accept()
        if lamp_id not in self.active_connections:
            self.active_connections[lamp_id] = set()
        self.active_connections[lamp_id].add(websocket)
        logger.info(f"WebSocket 连接: {lamp_id}")

    def disconnect(self, websocket: WebSocket, lamp_id: str):
        self.active_connections[lamp_id].discard(websocket)
        if not self.active_connections[lamp_id]:
            del self.active_connections[lamp_id]
        logger.info(f"WebSocket 断开: {lamp_id}")

    async def broadcast_to_device(self, lamp_id: str, message: dict):
        """向特定设备的所有连接广播消息"""
        if lamp_id in self.active_connections:
            for connection in self.active_connections[lamp_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"WebSocket 发送失败: {e}")
                    self.disconnect(connection, lamp_id)

manager = ConnectionManager()

@router.websocket("/state/{lamp_id}")
async def websocket_device_state(websocket: WebSocket, lamp_id: str):
    """设备状态 WebSocket 端点"""
    await manager.connect(websocket, lamp_id)

    try:
        while True:
            # 保持连接，接收客户端消息
            data = await websocket.receive_text()

            # 处理心跳消息
            if data == "ping":
                await websocket.send_text("pong")

    except WebSocketDisconnect:
        manager.disconnect(websocket, lamp_id)
    except Exception as e:
        logger.error(f"WebSocket 错误: {e}")
        manager.disconnect(websocket, lamp_id)
