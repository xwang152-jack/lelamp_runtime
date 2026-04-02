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
