"""
摄像头隐私保护模块
提供 LED 指示和用户同意机制
"""

import asyncio
import logging
import threading
import time
from typing import Callable, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("lelamp.vision.privacy")


class CameraState(Enum):
    """摄像头状态"""
    IDLE = "idle"           # 空闲（未使用）
    ACTIVE = "active"       # 活跃（正在使用）
    PAUSED = "paused"       # 暂停（用户暂停）
    CONSENT_REQUIRED = "consent_required"  # 需要用户同意


@dataclass
class PrivacyConfig:
    """隐私保护配置"""
    # LED 指示配置
    enable_led_indicator: bool = True
    idle_color: tuple = (0, 0, 0)           # 关闭
    active_color: tuple = (255, 0, 0)       # 红色（录制中）
    consent_color: tuple = (255, 255, 0)   # 黄色（等待同意）

    # 用户同意配置
    require_consent: bool = True
    consent_timeout_s: float = 30.0         # 同意超时时间
    consent_ttl_s: float = 3600.0          # 同意有效期（1小时）

    # 提示配置
    show_notification: bool = True
    notification_message: str = "📷 LeLamp 将使用摄像头进行视觉识别"
    data_usage_info: str = "图像数据仅在本地处理，不会上传到云端"


class CameraPrivacyManager:
    """
    摄像头隐私管理器

    功能：
    1. LED 指示 - 显示摄像头当前状态
    2. 用户同意 - 使用摄像头前请求用户许可
    3. 使用记录 - 记录摄像头使用情况
    4. 自动暂停 - 超时自动暂停摄像头
    """

    def __init__(
        self,
        config: PrivacyConfig,
        rgb_setter: Callable[[tuple], None] | None = None,
        notification_shower: Callable[[str], None] | None = None,
    ):
        """
        初始化隐私管理器

        Args:
            config: 隐私配置
            rgb_setter: RGB LED 设置函数 (color) -> None
            notification_shower: 通知显示函数 (message) -> None
        """
        self._config = config
        self._rgb_setter = rgb_setter
        self._notification_shower = notification_shower

        # 状态管理
        self._state = CameraState.IDLE
        self._state_lock = threading.Lock()

        # 用户同意管理
        self._consent_granted = False
        self._consent_granted_at: float = 0.0
        self._consent_lock = threading.Lock()

        # 使用统计
        self._session_count = 0
        self._total_usage_time_s = 0.0
        self._session_start_time: float | None = None

        # LED 线程
        self._led_stop = threading.Event()
        self._led_thread: threading.Thread | None = None

        logger.info("CameraPrivacyManager initialized")

    @property
    def state(self) -> CameraState:
        """获取当前状态"""
        with self._state_lock:
            return self._state

    @property
    def is_active(self) -> bool:
        """摄像头是否活跃"""
        return self.state == CameraState.ACTIVE

    @property
    def has_consent(self) -> bool:
        """是否有有效同意"""
        with self._consent_lock:
            if not self._consent_granted:
                return False

            # 检查同意是否过期
            if time.time() - self._consent_granted_at > self._config.consent_ttl_s:
                self._consent_granted = False
                logger.info("用户同意已过期")
                return False

            return True

    def start(self):
        """启动隐私管理器"""
        if self._config.enable_led_indicator and self._rgb_setter:
            self._led_stop.clear()
            self._led_thread = threading.Thread(
                target=self._led_loop,
                daemon=True,
                name="CameraPrivacyLED"
            )
            self._led_thread.start()
            logger.info("隐私管理器 LED 线程已启动")

    def stop(self):
        """停止隐私管理器"""
        self._led_stop.set()
        if self._led_thread and self._led_thread.is_alive():
            self._led_thread.join(timeout=2.0)

        # 关闭 LED
        if self._rgb_setter:
            self._rgb_setter(self._config.idle_color)

        logger.info("隐私管理器已停止")

    async def request_consent(self) -> bool:
        """
        请求用户同意

        Returns:
            是否获得同意
        """
        with self._consent_lock:
            # 检查是否已有有效同意
            if self.has_consent:
                logger.debug("使用缓存的用户同意")
                return True

            if not self._config.require_consent:
                logger.info("未启用用户同意要求")
                self._consent_granted = True
                self._consent_granted_at = time.time()
                return True

        # 设置状态为需要同意
        self._set_state(CameraState.CONSENT_REQUIRED)

        # 显示通知
        if self._config.show_notification and self._notification_shower:
            message = self._config.notification_message
            if self._config.data_usage_info:
                message += f"\n{self._config.data_usage_info}"
            self._notification_shower(message)

        # 等待用户同意
        logger.info("等待用户同意...")
        start_time = time.time()

        while time.time() - start_time < self._config.consent_timeout_s:
            with self._consent_lock:
                if self._consent_granted:
                    self._consent_granted_at = time.time()
                    logger.info("用户已同意使用摄像头")
                    return True

            await asyncio.sleep(0.1)

        # 超时
        logger.warning(f"用户同意超时（{self._config.consent_timeout_s}秒）")
        return False

    def grant_consent(self):
        """
        授予用户同意（由用户操作触发）

        例如：按下按钮、语音命令等
        """
        with self._consent_lock:
            self._consent_granted = True
            self._consent_granted_at = time.time()
            logger.info("用户授予摄像头使用同意")

    def revoke_consent(self):
        """
        撤销用户同意（由用户操作触发）

        例如：按下按钮、语音命令等
        """
        with self._consent_lock:
            self._consent_granted = False
            self._consent_granted_at = 0.0
            logger.info("用户撤销摄像头使用同意")

        # 停止摄像头
        self._set_state(CameraState.IDLE)

    async def activate_camera(self) -> bool:
        """
        激活摄像头（带隐私检查）

        Returns:
            是否成功激活
        """
        # 检查用户同意
        if not await self.request_consent():
            return False

        # 设置活跃状态
        self._set_state(CameraState.ACTIVE)
        self._session_count += 1
        self._session_start_time = time.time()

        logger.info(f"摄像头已激活（会话 #{self._session_count}）")
        return True

    def deactivate_camera(self):
        """停用摄像头"""
        was_active = self.state == CameraState.ACTIVE

        self._set_state(CameraState.IDLE)

        # 更新使用统计
        if was_active and self._session_start_time:
            session_time = time.time() - self._session_start_time
            self._total_usage_time_s += session_time
            self._session_start_time = None
            logger.info(f"摄像头会话结束，使用时长: {session_time:.1f}秒")

    def pause_camera(self):
        """暂停摄像头"""
        self._set_state(CameraState.PAUSED)
        logger.info("摄像头已暂停")

    def resume_camera(self) -> bool:
        """恢复摄像头"""
        if not self.has_consent:
            logger.warning("无法恢复摄像头：无有效同意")
            return False

        self._set_state(CameraState.ACTIVE)
        logger.info("摄像头已恢复")
        return True

    def get_stats(self) -> dict:
        """获取使用统计"""
        with self._state_lock, self._consent_lock:
            current_session_time = 0.0
            if self._session_start_time and self.state == CameraState.ACTIVE:
                current_session_time = time.time() - self._session_start_time

            return {
                "state": self._state.value,
                "has_consent": self._consent_granted,
                "consent_granted_at": self._consent_granted_at,
                "consent_remaining_s": max(
                    0.0,
                    self._config.consent_ttl_s - (time.time() - self._consent_granted_at)
                    if self._consent_granted else 0.0
                ),
                "session_count": self._session_count,
                "total_usage_time_s": self._total_usage_time_s + current_session_time,
                "current_session_time_s": current_session_time,
            }

    def _set_state(self, new_state: CameraState):
        """设置新状态（线程安全）"""
        with self._state_lock:
            old_state = self._state
            self._state = new_state

            if old_state != new_state:
                logger.info(f"摄像头状态: {old_state.value} -> {new_state.value}")

    def _led_loop(self):
        """LED 指示循环（在单独线程中运行）"""
        if not self._rgb_setter:
            return

        blink_phase = 0

        while not self._led_stop.is_set():
            state = self.state

            if state == CameraState.IDLE:
                # 空闲：关闭 LED
                self._rgb_setter(self._config.idle_color)

            elif state == CameraState.ACTIVE:
                # 活跃：红色呼吸效果
                blink_phase = (blink_phase + 1) % 100
                brightness = 0.3 + 0.7 * (0.5 + 0.5 * __import__('math').sin(
                    blink_phase * 2 * __import__('math').pi / 100
                ))
                color = tuple(int(c * brightness) for c in self._config.active_color)
                self._rgb_setter(color)

            elif state == CameraState.PAUSED:
                # 暂停：慢速闪烁黄色
                blink_phase = (blink_phase + 1) % 50
                if blink_phase < 25:
                    self._rgb_setter(self._config.consent_color)
                else:
                    self._rgb_setter(self._config.idle_color)

            elif state == CameraState.CONSENT_REQUIRED:
                # 等待同意：快速闪烁黄色
                blink_phase = (blink_phase + 1) % 20
                if blink_phase < 10:
                    self._rgb_setter(self._config.consent_color)
                else:
                    self._rgb_setter(self._config.idle_color)

            # 控制刷新率（50Hz）
            time.sleep(0.02)


class PrivacyGuard:
    """
    隐私保护上下文管理器

    用法：
        async with privacy_guard:
            # 使用摄像头
            frame = await camera.get_frame()
    """

    def __init__(
        self,
        manager: CameraPrivacyManager,
        auto_activate: bool = True,
    ):
        """
        初始化隐私保护上下文

        Args:
            manager: 隐私管理器
            auto_activate: 是否自动激活摄像头
        """
        self._manager = manager
        self._auto_activate = auto_activate
        self._was_active = False

    async def __aenter__(self):
        """进入上下文"""
        if self._auto_activate:
            self._was_active = await self._manager.activate_camera()
        return self._was_active

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """退出上下文"""
        if self._was_active:
            self._manager.deactivate_camera()
        return False
