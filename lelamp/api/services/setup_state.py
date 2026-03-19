"""
设置状态管理器
管理首次设置流程的状态持久化
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict

logger = logging.getLogger(__name__)


class SetupStateManager:
    """设置状态管理器"""

    DEFAULT_STATE = {
        "setup_completed": False,
        "setup_started_at": None,
        "setup_completed_at": None,
        "current_step": "welcome",
        "wifi_ssid": None,
        "connection_attempts": 0,
        "error_message": None,
        "last_ip_address": None,
        "ap_mode_count": 0,
        "network_history": []
    }

    def __init__(self, state_file: str = "/var/lib/lelamp/setup_status.json"):
        self.state_file = Path(state_file)
        self._lock = Lock()

        # 确保目录存在
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        # 如果文件不存在，创建初始状态
        if not self.state_file.exists():
            self.save_state(self.DEFAULT_STATE.copy())

    def load_state(self) -> Dict[str, Any]:
        """加载状态"""
        with self._lock:
            try:
                if self.state_file.exists():
                    with open(self.state_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
                else:
                    return self.DEFAULT_STATE.copy()
            except Exception as e:
                logger.error(f"加载状态失败: {e}")
                return self.DEFAULT_STATE.copy()

    def save_state(self, state: Dict[str, Any]) -> None:
        """保存状态"""
        with self._lock:
            try:
                with open(self.state_file, 'w', encoding='utf-8') as f:
                    json.dump(state, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.error(f"保存状态失败: {e}")

    def update_step(self, step: str) -> None:
        """更新当前步骤"""
        state = self.load_state()
        state["current_step"] = step
        self.save_state(state)

    def increment_attempts(self) -> int:
        """增加连接尝试次数"""
        state = self.load_state()
        state["connection_attempts"] += 1
        self.save_state(state)
        return state["connection_attempts"]

    def reset_attempts(self) -> None:
        """重置连接尝试次数"""
        state = self.load_state()
        state["connection_attempts"] = 0
        self.save_state(state)

    def set_error(self, error: str) -> None:
        """设置错误信息"""
        state = self.load_state()
        state["error_message"] = error
        self.save_state(state)

    def clear_error(self) -> None:
        """清除错误信息"""
        state = self.load_state()
        state["error_message"] = None
        self.save_state(state)

    def set_wifi_ssid(self, ssid: str) -> None:
        """设置 WiFi SSID"""
        state = self.load_state()
        state["wifi_ssid"] = ssid
        self.save_state(state)

    def complete_setup(self, ip_address: str) -> None:
        """标记设置完成"""
        state = self.load_state()
        state["setup_completed"] = True
        state["setup_completed_at"] = datetime.utcnow().isoformat()
        state["last_ip_address"] = ip_address
        state["current_step"] = "completed"
        self.save_state(state)

    def is_setup_completed(self) -> bool:
        """检查是否已完成设置"""
        state = self.load_state()
        return state.get("setup_completed", False)

    def reset(self) -> None:
        """重置状态到初始值"""
        self.save_state(self.DEFAULT_STATE.copy())

    def get_current_step(self) -> str:
        """获取当前步骤"""
        state = self.load_state()
        return state.get("current_step", "welcome")

    def get_connection_attempts(self) -> int:
        """获取连接尝试次数"""
        state = self.load_state()
        return state.get("connection_attempts", 0)

    def get_error_message(self) -> str:
        """获取错误信息"""
        state = self.load_state()
        return state.get("error_message")

    def get_last_ip_address(self) -> str:
        """获取上次 IP 地址"""
        state = self.load_state()
        return state.get("last_ip_address")
