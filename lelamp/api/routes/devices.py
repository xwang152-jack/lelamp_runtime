"""
设备相关 API 路由
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict
import logging

logger = logging.getLogger("lelamp.api.devices")

router = APIRouter()

@router.get("/{lamp_id}/state")
async def get_device_state(lamp_id: str) -> Dict:
    """获取设备状态"""
    # TODO: 从服务获取实际状态
    return {
        "lamp_id": lamp_id,
        "status": "online",
        "motor_positions": {},
        "light_color": {"r": 255, "g": 244, "b": 229},
        "camera_active": False
    }

@router.post("/{lamp_id}/command")
async def send_device_command(
    lamp_id: str,
    command: dict
) -> Dict:
    """发送设备命令"""
    # TODO: 实现命令发送逻辑
    logger.info(f"Sending command to {lamp_id}: {command}")
    return {"status": "success", "command": command}
