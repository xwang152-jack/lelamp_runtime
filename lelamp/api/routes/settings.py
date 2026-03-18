"""
设置管理 API 路由

提供应用配置的获取和更新功能
"""
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from lelamp.database.session import get_db
from lelamp.database.models import UserSettings
from lelamp.api.services.config_sync import config_sync_service
from lelamp.api.models.requests import AppSettingsUpdateRequest
from lelamp.api.models.responses import AppSettingsResponse

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# 设置端点
# ============================================================================

@router.get("/", response_model=AppSettingsResponse)
async def get_settings(
    lamp_id: str = Query(..., description="设备 ID"),
    db: Session = Depends(get_db)
) -> AppSettingsResponse:
    """
    获取应用配置

    优先从数据库读取配置，如果数据库中没有配置则使用环境变量默认值

    Args:
        lamp_id: 设备 ID
        db: 数据库会话

    Returns:
        AppSettingsResponse: 当前配置
    """
    try:
        config = config_sync_service.get_current_config(db, lamp_id)
        return AppSettingsResponse(**config)
    except Exception as e:
        logger.error(f"Get settings error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取配置失败: {str(e)}"
        )


@router.put("/", response_model=AppSettingsResponse)
async def update_settings(
    lamp_id: str = Query(..., description="设备 ID"),
    request: AppSettingsUpdateRequest,
    db: Session = Depends(get_db)
) -> AppSettingsResponse:
    """
    更新应用配置

    配置会被保存到数据库，同时同步到 .env 文件。
    修改后需要重启服务才能生效。

    Args:
        lamp_id: 设备 ID
        request: 配置更新请求（使用请求体传递）
        db: 数据库会话

    Returns:
        AppSettingsResponse: 更新后的配置
    """
    try:
        # 将 Pydantic 模型转换为字典，排除 None 值和未设置的字段
        updates = request.model_dump(exclude_unset=True, exclude_none=True)

        if not updates:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="没有提供任何更新字段"
            )

        # 更新配置
        config, requires_restart = config_sync_service.update_settings(
            db, lamp_id, updates
        )

        logger.info(f"Settings updated for lamp_id={lamp_id}, requires_restart={requires_restart}")

        return AppSettingsResponse(**config)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update settings error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"更新配置失败: {str(e)}"
        )


@router.post("/reset")
async def reset_settings(
    lamp_id: str = Query(..., description="设备 ID"),
    db: Session = Depends(get_db)
) -> dict:
    """
    重置配置为默认值

    删除数据库中的配置，下次重启时将使用环境变量默认值

    Args:
        lamp_id: 设备 ID
        db: 数据库会话

    Returns:
        操作结果
    """
    try:
        settings = db.query(UserSettings).filter(
            UserSettings.lamp_id == lamp_id
        ).first()

        if settings:
            db.delete(settings)
            db.commit()
            logger.info(f"Settings reset for lamp_id={lamp_id}")
            return {"success": True, "message": "配置已重置，重启后生效"}
        else:
            return {"success": True, "message": "配置已是默认值"}

    except Exception as e:
        logger.error(f"Reset settings error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"重置配置失败: {str(e)}"
        )


@router.get("/fields")
async def get_setting_fields() -> dict:
    """
    获取可配置的字段列表

    Returns:
        可配置字段的信息
    """
    return {
        "categories": {
            "llm": {
                "name": "LLM 配置",
                "fields": [
                    {"name": "deepseek_model", "type": "string", "default": "deepseek-chat"},
                    {"name": "deepseek_base_url", "type": "string", "default": "https://api.deepseek.com"},
                    {"name": "deepseek_api_key", "type": "password", "default": ""},
                ]
            },
            "vision": {
                "name": "视觉识别",
                "fields": [
                    {"name": "vision_enabled", "type": "boolean", "default": True},
                    {"name": "modelscope_model", "type": "string", "default": "Qwen/Qwen3-VL-235B-A22B-Instruct"},
                    {"name": "modelscope_api_key", "type": "password", "default": ""},
                    {"name": "modelscope_timeout_s", "type": "float", "default": 60.0},
                ]
            },
            "camera": {
                "name": "摄像头",
                "fields": [
                    {"name": "camera_width", "type": "integer", "default": 1024, "min": 320, "max": 1920},
                    {"name": "camera_height", "type": "integer", "default": 768, "min": 240, "max": 1080},
                    {"name": "camera_rotate_deg", "type": "integer", "default": 0, "min": 0, "max": 360},
                    {"name": "camera_flip", "type": "enum", "default": "none", "options": ["none", "horizontal", "vertical", "both"]},
                ]
            },
            "speech": {
                "name": "语音配置",
                "fields": [
                    {"name": "baidu_tts_per", "type": "integer", "default": 4, "min": 0, "max": 500},
                ]
            },
            "hardware": {
                "name": "硬件配置",
                "fields": [
                    {"name": "led_brightness", "type": "integer", "default": 25, "min": 0, "max": 100},
                    {"name": "lamp_port", "type": "string", "default": "/dev/ttyACM0"},
                    {"name": "lamp_id", "type": "string", "default": "lelamp"},
                ]
            },
            "behavior": {
                "name": "行为配置",
                "fields": [
                    {"name": "greeting_text", "type": "string", "default": "你好！我是 LeLamp，你的智能台灯。"},
                    {"name": "noise_cancellation", "type": "boolean", "default": True},
                    {"name": "motion_cooldown_s", "type": "float", "default": 2.0, "min": 0.5, "max": 30.0},
                ]
            },
            "ui": {
                "name": "界面设置",
                "fields": [
                    {"name": "theme", "type": "enum", "default": "light", "options": ["light", "dark"]},
                    {"name": "language", "type": "enum", "default": "zh", "options": ["zh", "en"]},
                    {"name": "notifications_enabled", "type": "boolean", "default": True},
                    {"name": "brightness_level", "type": "integer", "default": 25, "min": 0, "max": 100},
                    {"name": "volume_level", "type": "integer", "default": 50, "min": 0, "max": 100},
                ]
            }
        }
    }
