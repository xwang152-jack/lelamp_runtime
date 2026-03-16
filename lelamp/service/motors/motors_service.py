import os
import csv
import time
import threading
from typing import Any, List
from ..base import ServiceBase, Priority
from lelamp.follower import LeLampFollowerConfig, LeLampFollower
from .health_monitor import MotorHealthMonitor, HealthThresholds, HealthStatus


class MotorsService(ServiceBase):
    # Valid joint names for the LeLamp robot
    VALID_JOINTS = ["base_yaw", "base_pitch", "elbow_pitch", "wrist_roll", "wrist_pitch"]
    
    def __init__(self, port: str, lamp_id: str, fps: int = 30, motor_config=None):
        super().__init__("motors")
        self.port = port
        self.lamp_id = lamp_id
        self.fps = fps
        self.motor_config = motor_config  # MotorConfig 实例
        self.robot_config = LeLampFollowerConfig(port=port, id=lamp_id)
        self.robot: LeLampFollower = None
        self.recordings_dir = os.path.join(os.path.dirname(__file__), "..", "..", "recordings")
        self._cancel_playback = threading.Event()

        # 添加录制数据缓存，避免重复读取 CSV 文件
        self._recording_cache: dict[str, list] = {}
        self._cache_lock = threading.Lock()

        # 健康监控器
        self.health_monitor: MotorHealthMonitor = None
        self._health_check_thread: threading.Thread = None
        self._health_check_stop = threading.Event()
        self._last_target_positions: dict[str, float] = {}  # 记录最后的目标位置

        self.logger.info(f"Motors service initialized with recording cache enabled")

    def dispatch(self, event_type: str, payload: Any, priority: Priority = Priority.NORMAL):
        if event_type == "stop":
            self._cancel_playback.set()
        return super().dispatch(event_type, payload, priority=priority)
    
    def start(self):
        super().start()
        self.robot = LeLampFollower(self.robot_config)
        self.robot.connect(calibrate=False)
        self.logger.info(f"Motors service connected to {self.port}")

        # 启动健康监控
        if self.motor_config and self.motor_config.health_check_enabled:
            thresholds = HealthThresholds(
                temp_warning_c=self.motor_config.temp_warning_c,
                temp_critical_c=self.motor_config.temp_critical_c,
                voltage_min_v=self.motor_config.voltage_min_v,
                voltage_max_v=self.motor_config.voltage_max_v,
                load_warning=self.motor_config.load_warning,
                load_stall=self.motor_config.load_stall,
                position_error_deg=self.motor_config.position_error_deg,
            )
            self.health_monitor = MotorHealthMonitor(self.robot.bus, thresholds)
            self._start_health_check_thread()
            self.logger.info("Motor health monitoring started")

    def stop(self, timeout: float = 5.0):
        # 停止健康检查线程
        if self._health_check_thread and self._health_check_thread.is_alive():
            self._health_check_stop.set()
            self._health_check_thread.join(timeout=timeout)
            self.logger.info("Motor health check thread stopped")

        if self.robot:
            self.robot.disconnect()
            self.robot = None
        super().stop(timeout)
    
    def handle_event(self, event_type: str, payload: Any):
        if event_type == "play":
            self._handle_play(payload)
        elif event_type == "move_joint":
            self._handle_move_joint(payload)
        elif event_type == "stop":
            self.logger.info("Stopping motors playback")
        else:
            self.logger.warning(f"Unknown event type: {event_type}")
    
    def _handle_move_joint(self, payload: dict):
        """Move a single joint to a specified angle"""
        if not self.robot:
            self.logger.error("Robot not connected")
            return

        joint_name = payload.get("joint_name")
        angle = payload.get("angle")

        if joint_name not in self.VALID_JOINTS:
            self.logger.error(f"Invalid joint name: {joint_name}")
            return

        try:
            # Get current positions of all joints first
            obs = self.robot.get_observation()
            action = {}
            for key, value in obs.items():
                if key.endswith(".pos"):
                    action[key] = value

            # Update only the target joint
            action[f"{joint_name}.pos"] = float(angle)

            # 记录目标位置(用于健康监控)
            self._last_target_positions[joint_name] = float(angle)

            # Send full action with all joints
            self.robot.send_action(action)
            self.logger.info(f"Moved joint {joint_name} to {angle} degrees")
        except Exception as e:
            self.logger.error(f"Error moving joint {joint_name}: {e}")
    
    def get_joint_positions(self) -> dict[str, float]:
        """Get current positions of all joints"""
        if not self.robot:
            self.logger.error("Robot not connected")
            return {}
        
        try:
            obs = self.robot.get_observation()
            # Extract joint positions (remove ".pos" suffix from keys)
            positions = {}
            for key, value in obs.items():
                if key.endswith(".pos"):
                    joint_name = key.removesuffix(".pos")
                    if joint_name in self.VALID_JOINTS:
                        positions[joint_name] = value
            return positions
        except Exception as e:
            self.logger.error(f"Error getting joint positions: {e}")
            return {}
    
    def _load_recording(self, recording_name: str) -> list:
        """加载录制数据（带缓存）"""
        # 检查缓存
        with self._cache_lock:
            if recording_name in self._recording_cache:
                self.logger.debug(f"Using cached recording: {recording_name}")
                return self._recording_cache[recording_name]

        # 从文件加载
        csv_filename = f"{recording_name}.csv"
        csv_path = os.path.join(self.recordings_dir, csv_filename)

        if not os.path.exists(csv_path):
            self.logger.error(f"Recording not found: {csv_path}")
            return []

        try:
            with open(csv_path, 'r') as csvfile:
                csv_reader = csv.DictReader(csvfile)
                actions = list(csv_reader)

            # 缓存数据
            with self._cache_lock:
                self._recording_cache[recording_name] = actions

            self.logger.info(f"Loaded and cached {len(actions)} actions from {recording_name}")
            return actions

        except Exception as e:
            self.logger.error(f"Error loading recording {recording_name}: {e}")
            return []

    def _handle_play(self, recording_name: str):
        """Play a recording by name"""
        if not self.robot:
            self.logger.error("Robot not connected")
            return

        self._cancel_playback.clear()

        # 使用缓存加载录制数据
        actions = self._load_recording(recording_name)
        if not actions:
            return

        try:
            self.logger.info(f"Playing {len(actions)} actions from {recording_name}")
            
            for row in actions:
                if self._cancel_playback.is_set():
                    self.logger.info(f"Playback cancelled: {recording_name}")
                    break
                t0 = time.perf_counter()
                
                # Extract action data (exclude timestamp column)
                action = {key: float(value) for key, value in row.items() if key != 'timestamp'}
                self.robot.send_action(action)
                
                # Use time.sleep instead of busy_wait to avoid blocking other threads
                sleep_time = 1.0 / self.fps - (time.perf_counter() - t0)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            self.logger.info(f"Finished playing recording: {recording_name}")
            
        except Exception as e:
            self.logger.error(f"Error playing recording {recording_name}: {e}")
    
    def clear_cache(self):
        """清空录制数据缓存"""
        with self._cache_lock:
            cleared_count = len(self._recording_cache)
            self._recording_cache.clear()
            self.logger.info(f"Cleared {cleared_count} recordings from cache")

    def get_cache_stats(self) -> dict:
        """获取缓存统计信息"""
        with self._cache_lock:
            return {
                "cached_recordings": len(self._recording_cache),
                "cache_size_mb": sum(
                    len(str(actions)) for actions in self._recording_cache.values()
                ) / (1024 * 1024),
                "cached_names": list(self._recording_cache.keys())
            }

    def get_available_recordings(self) -> List[str]:
        """Get list of recording names available for this lamp ID"""
        if not os.path.exists(self.recordings_dir):
            return []

        recordings = []
        suffix = f".csv"

        for filename in os.listdir(self.recordings_dir):
            if filename.endswith(suffix):
                # Remove the lamp_id suffix to get the recording name
                recording_name = filename[:-len(suffix)]
                recordings.append(recording_name)

        return sorted(recordings)

    # ========== 健康监控方法 ==========

    def _start_health_check_thread(self):
        """启动健康检查后台线程"""
        if not self.health_monitor or not self.motor_config:
            return

        self._health_check_stop.clear()
        self._health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True,
            name="MotorHealthCheck"
        )
        self._health_check_thread.start()

    def _health_check_loop(self):
        """健康检查循环"""
        interval = self.motor_config.health_check_interval_s
        self.logger.info(f"Health check loop started (interval: {interval}s)")

        while not self._health_check_stop.is_set():
            try:
                # 检查所有舵机健康状态
                health_results = self.health_monitor.check_all_motors_health(
                    target_positions=self._last_target_positions
                )

                # 检查是否有异常状态
                for motor_name, health_data in health_results.items():
                    if health_data.status == HealthStatus.STALLED:
                        self.logger.error(f"Motor {motor_name} STALLED! Taking protective action...")
                        # 堵转时停止所有动作
                        self._cancel_playback.set()
                        # 可以在这里添加更多保护措施,如断电等

                    elif health_data.status == HealthStatus.CRITICAL:
                        self.logger.error(f"Motor {motor_name} in CRITICAL state!")
                        # 严重状态时停止动作
                        self._cancel_playback.set()

                    elif health_data.status == HealthStatus.WARNING:
                        self.logger.warning(f"Motor {motor_name} in WARNING state")
                        # 警告状态可以继续运行,但记录日志

            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")

            # 等待下一次检查
            self._health_check_stop.wait(interval)

        self.logger.info("Health check loop stopped")

    def get_motor_health_summary(self) -> dict:
        """获取舵机健康状态汇总"""
        if not self.health_monitor:
            return {"error": "Health monitoring not enabled"}

        return self.health_monitor.get_health_summary()

    def get_motor_health_history(self, motor_name: str, limit: int = 20) -> list[dict]:
        """获取指定舵机的健康历史"""
        if not self.health_monitor:
            return []

        return self.health_monitor.get_motor_history(motor_name, limit)

    def check_motor_stall(self, motor_name: str) -> bool:
        """检测指定舵机是否堵转"""
        if not self.health_monitor:
            return False

        return self.health_monitor.detect_stall(motor_name)

    def reset_health_statistics(self, motor_name: str = None):
        """重置健康统计数据"""
        if self.health_monitor:
            self.health_monitor.reset_statistics(motor_name)

    def clear_health_history(self, motor_name: str = None):
        """清除健康历史记录"""
        if self.health_monitor:
            self.health_monitor.clear_history(motor_name)
