"""
首次配置管理器

负责检测和管理设备的首次启动配置状态
"""
import asyncio
import json
import logging
import os
from datetime import datetime, UTC
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class OnboardingManager:
    """
    首次配置管理器

    功能:
    - 检测是否首次启动
    - 检测是否已配置 WiFi
    - 标记配置完成/重置
    - 获取配置状态
    """

    def __init__(
        self,
        status_file: str = "/var/lib/lelamp/setup_status.json",
        state_dir: str = "/var/lib/lelamp"
    ):
        """
        初始化首次配置管理器

        Args:
            status_file: 状态文件路径
            state_dir: 状态目录路径
        """
        self._status_file = Path(status_file)
        self._state_dir = Path(state_dir)
        self._lock = asyncio.Lock()

    def _ensure_state_dir(self) -> None:
        """确保状态目录存在"""
        self._state_dir.mkdir(parents=True, exist_ok=True)

    async def is_first_boot(self) -> bool:
        """
        检测是否首次启动

        Returns:
            是否首次启动（从未完成过配置）
        """
        status = await self.get_setup_status()
        return not status.get("setup_completed", False)

    async def is_setup_mode_needed(self) -> bool:
        """
        检测是否需要进入设置模式

        设置模式条件:
        1. 从未完成过配置
        2. 或者上次配置的 WiFi 不可用且超过重试时间

        Returns:
            是否需要进入设置模式
        """
        # 如果从未配置过
        if await self.is_first_boot():
            return True

        status = await self.get_setup_status()

        # 如果配置的 WiFi 为空
        if not status.get("wifi_ssid"):
            return True

        # 检查上次配置时间
        setup_time = status.get("setup_completed_at")
        if setup_time:
            # 如果配置成功但当前未连接 WiFi，可能需要重新配置
            # 这里可以添加更复杂的逻辑
            pass

        return False

    async def mark_setup_complete(self, wifi_ssid: Optional[str] = None) -> bool:
        """
        标记配置已完成

        Args:
            wifi_ssid: 配置的 WiFi SSID

        Returns:
            是否成功标记
        """
        async with self._lock:
            try:
                self._ensure_state_dir()

                status = await self._get_status_from_file() or {}

                status.update({
                    "setup_completed": True,
                    "setup_completed_at": datetime.now(UTC).isoformat(),
                    "wifi_ssid": wifi_ssid,
                    "last_mode": "client",
                    "last_updated": datetime.now(UTC).isoformat()
                })

                await self._write_status(status)

                logger.info(f"Setup marked as complete for WiFi: {wifi_ssid}")
                return True

            except Exception as e:
                logger.error(f"Failed to mark setup complete: {e}", exc_info=True)
                return False

    async def mark_setup_required(self) -> bool:
        """
        标记需要重新配置（重置配置）

        Returns:
            是否成功标记
        """
        async with self._lock:
            try:
                self._ensure_state_dir()

                status = {
                    "setup_completed": False,
                    "setup_completed_at": None,
                    "wifi_ssid": None,
                    "last_mode": "ap",
                    "reset_at": datetime.now(UTC).isoformat(),
                    "last_updated": datetime.now(UTC).isoformat()
                }

                await self._write_status(status)

                logger.info("Setup reset - configuration required")
                return True

            except Exception as e:
                logger.error(f"Failed to mark setup required: {e}", exc_info=True)
                return False

    async def mark_ap_mode_entered(self) -> bool:
        """
        标记已进入 AP 模式

        Returns:
            是否成功标记
        """
        async with self._lock:
            try:
                status = await self.get_setup_status()
                status["last_mode"] = "ap"
                status["ap_mode_entered_at"] = datetime.now(UTC).isoformat()
                status["last_updated"] = datetime.now(UTC).isoformat()

                await self._write_status(status)

                return True

            except Exception as e:
                logger.error(f"Failed to mark AP mode entered: {e}")
                return False

    async def get_setup_status(self) -> dict:
        """
        获取配置状态（每次从文件读取，确保数据最新）

        Returns:
            状态字典，包含:
            - setup_completed: 是否已完成配置
            - setup_completed_at: 配置完成时间
            - wifi_ssid: 配置的 WiFi SSID
            - last_mode: 上次模式 (ap/client)
            - reset_at: 重置时间
            - ap_mode_entered_at: 进入 AP 模式时间
        """
        # 每次都从文件读取，不使用缓存，确保数据最新
        status = await self._get_status_from_file()

        if not status:
            return {
                "setup_completed": False,
                "setup_completed_at": None,
                "wifi_ssid": None,
                "last_mode": "unknown",
                "reset_at": None,
                "ap_mode_entered_at": None,
                "last_updated": None
            }

        return status

    def get_status(self) -> dict:
        """
        同步包装：获取配置状态（供单元测试/同步场景使用）
        """
        try:
            status = asyncio.run(self.get_setup_status())
            enriched = {"completed": status.get("setup_completed", False)}
            enriched.update(status)
            return enriched
        except Exception:
            return {
                "completed": False,
                "setup_completed": False,
                "setup_completed_at": None,
                "wifi_ssid": None,
                "last_mode": "unknown",
                "reset_at": None,
                "ap_mode_entered_at": None,
                "last_updated": None
            }

    async def update_wifi_status(self, connected: bool, ssid: Optional[str] = None) -> bool:
        """
        更新 WiFi 连接状态

        Args:
            connected: 是否已连接
            ssid: 连接的 SSID

        Returns:
            是否成功更新
        """
        async with self._lock:
            try:
                status = await self.get_setup_status()
                status["wifi_connected"] = connected
                status["current_ssid"] = ssid
                status["wifi_updated_at"] = datetime.now(UTC).isoformat()
                status["last_updated"] = datetime.now(UTC).isoformat()

                await self._write_status(status)

                return True

            except Exception as e:
                logger.error(f"Failed to update WiFi status: {e}")
                return False

    async def get_configuration_summary(self) -> dict:
        """
        获取配置摘要（用于前端显示）

        Returns:
            配置摘要字典
        """
        status = await self.get_setup_status()

        return {
            "is_configured": status.get("setup_completed", False),
            "configured_wifi": status.get("wifi_ssid"),
            "current_mode": status.get("last_mode", "unknown"),
            "needs_setup": await self.is_setup_mode_needed(),
            "can_exit_setup": status.get("setup_completed", False)
        }

    # ========================================================================
    # 私有方法
    # ========================================================================

    async def _get_status_from_file(self) -> Optional[dict]:
        """从文件读取状态"""
        if not self._status_file.exists():
            return None

        try:
            content = await asyncio.to_thread(self._status_file.read_text)
            return json.loads(content)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to read status file: {e}")
            return None

    async def _write_status(self, status: dict) -> None:
        """写入状态到文件"""
        self._ensure_state_dir()

        # 使用原子写入
        temp_file = self._status_file.with_suffix('.tmp')
        await asyncio.to_thread(
            temp_file.write_text,
            json.dumps(status, indent=2, ensure_ascii=False)
        )

        # 重命名（原子操作）
        await asyncio.to_thread(
            temp_file.rename,
            self._status_file
        )


OnboardingService = OnboardingManager

# 全局单例实例
onboarding_manager = OnboardingManager()
