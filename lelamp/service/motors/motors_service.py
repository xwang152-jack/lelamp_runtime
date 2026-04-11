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
        self._bus_lock = threading.Lock()  # 串口访问互斥锁
        self._motor_fault_callback = None   # 外部注册：callback(motor_name, old_status, new_status)
        self._prev_motor_status: dict = {}  # motor_name -> HealthStatus

        self.logger.info("Motors service initialized with recording cache enabled")

    def dispatch(self, event_type: str, payload: Any, priority: Priority = Priority.NORMAL):
        if event_type == "stop":
            self._cancel_playback.set()
        return super().dispatch(event_type, payload, priority=priority)
    
    def start(self):
        super().start()
        self.robot = LeLampFollower(self.robot_config)
        self.robot.connect(calibrate=False)
        self.logger.info(f"Motors service connected to {self.port}")

        # 应用平滑控制参数（覆盖 configure() 中的默认值）
        if self.motor_config:
            for motor in self.robot.bus.motors:
                self.robot.bus.write("P_Coefficient", motor, self.motor_config.motor_p_coefficient)
                self.robot.bus.write("D_Coefficient", motor, self.motor_config.motor_d_coefficient)
                self.robot.bus.write("Acceleration", motor, self.motor_config.motor_acceleration)
            if hasattr(self.robot.bus, "protocol_version") and self.robot.bus.protocol_version == 0:
                for motor in self.robot.bus.motors:
                    self.robot.bus.write("Maximum_Acceleration", motor, self.motor_config.motor_max_acceleration)
            self.logger.info(
                f"Smooth control applied: P={self.motor_config.motor_p_coefficient}, "
                f"D={self.motor_config.motor_d_coefficient}, "
                f"accel={self.motor_config.motor_acceleration}"
            )

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
        # 标记服务正在停止，防止新事件处理
        self._cancel_playback.set()

        # 停止健康检查线程
        if self._health_check_thread and self._health_check_thread.is_alive():
            self._health_check_stop.set()
            self._health_check_thread.join(timeout=timeout)
            self.logger.info("Motor health check thread stopped")

        # 先断开机器人连接，但保留对象引用用于最后的事件处理
        if self.robot:
            self.robot.disconnect()
            # 不立即设置为 None，让剩余事件能正常完成
            # 等待事件队列清空后再设置为 None

        # 停止服务基础类（这会等待事件队列清空）
        super().stop(timeout)

        # 现在可以安全设置为 None
        self.robot = None
    
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
        """Move a single joint to a specified angle with smooth interpolation.

        The interpolation runs in a separate thread to avoid blocking the
        event loop and interfering with audio processing.
        """
        if self._cancel_playback.is_set():
            self.logger.info("Service is stopping, ignoring move_joint request")
            return

        if not self.robot:
            self.logger.error("Robot not connected")
            return

        joint_name = payload.get("joint_name")
        angle = payload.get("angle")

        if joint_name not in self.VALID_JOINTS:
            self.logger.error(f"Invalid joint name: {joint_name}")
            return

        # 在独立线程中执行插值，避免阻塞事件循环
        thread = threading.Thread(
            target=self._interpolated_move,
            args=(joint_name, float(angle)),
            daemon=True,
            name=f"move-{joint_name}",
        )
        thread.start()

    def _send_with_retry(self, fn, joint_name: str, max_retries: int = 2):
        """带重试的串口操作"""
        last_exc = None
        for attempt in range(1 + max_retries):
            try:
                return fn()
            except Exception as e:
                last_exc = e
                if attempt < max_retries:
                    self.logger.warning(f"Serial retry {attempt + 1}/{max_retries} for {joint_name}: {e}")
                    time.sleep(0.05)
        raise last_exc

    def _interpolated_move(self, joint_name: str, target_angle: float):
        """Execute smooth interpolated move in a dedicated thread."""
        try:
            # 只读取一次当前位置
            def read_obs():
                if not self.robot:
                    self.logger.error("Robot disconnected before move operation")
                    return None
                return self.robot.get_observation()

            obs = self._send_with_retry(read_obs, joint_name)
            if obs is None:
                return

            current_pos = 0.0
            for key, value in obs.items():
                if key == f"{joint_name}.pos":
                    current_pos = float(value)
                    break

            diff = target_angle - current_pos

            # 如果差距很小，直接发送
            step_deg = self.motor_config.interpolation_step_deg if self.motor_config else 3.0
            if abs(diff) <= step_deg:
                def send_direct():
                    if not self.robot:
                        return
                    action = {key: float(val) for key, val in obs.items() if key.endswith(".pos")}
                    action[f"{joint_name}.pos"] = target_angle
                    self._last_target_positions[joint_name] = target_angle
                    self.robot.send_action(action)

                with self._bus_lock:
                    self._send_with_retry(send_direct, joint_name)
                self.logger.info(f"Moved joint {joint_name} to {target_angle:.1f}° (direct)")
                return

            # 多帧插值移动
            num_steps = max(2, int(abs(diff) / step_deg))
            for i in range(1, num_steps + 1):
                if self._cancel_playback.is_set():
                    self.logger.info(f"Move cancelled: {joint_name}")
                    break

                progress = i / num_steps
                interpolated = current_pos + diff * progress

                t0 = time.perf_counter()
                def send_step(pos=interpolated):
                    if not self.robot:
                        self.logger.error("Robot disconnected during interpolation")
                        return
                    action = {key: float(val) for key, val in obs.items() if key.endswith(".pos")}
                    action[f"{joint_name}.pos"] = pos
                    self._last_target_positions[joint_name] = pos
                    self.robot.send_action(action)

                with self._bus_lock:
                    self._send_with_retry(send_step, joint_name)

                sleep_time = 1.0 / self.fps - (time.perf_counter() - t0)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            if not self._cancel_playback.is_set():
                self.logger.info(f"Moved joint {joint_name} to {target_angle:.1f}° ({num_steps} steps)")
        except Exception as e:
            self.logger.error(f"Error moving joint {joint_name}: {e}")
    
    def get_joint_positions(self) -> dict[str, float]:
        """Get current positions of all joints"""
        if not self.robot:
            self.logger.error("Robot not connected")
            return {}

        try:
            def read_obs():
                if not self.robot:
                    return None
                return self.robot.get_observation()

            obs = self._send_with_retry(read_obs, "get_positions")
            if obs is None:
                return {}
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
        # 检查服务是否正在停止
        if self._cancel_playback.is_set():
            self.logger.info("Service is stopping, ignoring play request")
            return

        if not self.robot:
            self.logger.error("Robot not connected")
            return

        # 使用缓存加载录制数据
        actions = self._load_recording(recording_name)
        if not actions:
            return

        try:
            self.logger.info(f"Playing {len(actions)} actions from {recording_name}")

            for row in actions:
                # 在每个动作前检查是否需要停止
                if self._cancel_playback.is_set():
                    self.logger.info(f"Playback cancelled: {recording_name}")
                    break

                # 再次检查 robot 连接
                if not self.robot:
                    self.logger.error("Robot disconnected during playback")
                    return

                t0 = time.perf_counter()

                # Extract action data (exclude timestamp column)
                action = {key: float(value) for key, value in row.items() if key != 'timestamp'}

                with self._bus_lock:
                    if self.robot:  # 最后一次检查
                        self.robot.send_action(action)
                    else:
                        self.logger.error("Robot disconnected before sending action")
                        return

                # Use time.sleep instead of busy_wait to avoid blocking other threads
                sleep_time = 1.0 / self.fps - (time.perf_counter() - t0)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            if not self._cancel_playback.is_set():
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
        suffix = ".csv"

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
                # 逐个舵机检查健康状态，每次只短暂持有总线锁，不阻塞动作控制
                health_results = {}
                for motor_name in self.robot.bus.motors:
                    try:
                        with self._bus_lock:
                            health_results[motor_name] = self.health_monitor.check_motor_health(
                                motor_name, self._last_target_positions.get(motor_name)
                            )
                    except Exception as e:
                        self.logger.warning(f"Health check failed for {motor_name}: {e}")

                # 检查是否有异常状态
                has_fault = False
                for motor_name, health_data in health_results.items():
                    new_status = health_data.status
                    old_status = self._prev_motor_status.get(motor_name)
                    if old_status != new_status:
                        self._prev_motor_status[motor_name] = new_status
                        self._on_health_status_change(motor_name, old_status, new_status)

                    if health_data.status == HealthStatus.STALLED:
                        self.logger.error(f"Motor {motor_name} STALLED! Taking protective action...")
                        self._cancel_playback.set()
                        has_fault = True

                    elif health_data.status == HealthStatus.CRITICAL:
                        self.logger.error(f"Motor {motor_name} in CRITICAL state!")
                        self._cancel_playback.set()
                        has_fault = True

                    elif health_data.status == HealthStatus.WARNING:
                        self.logger.warning(f"Motor {motor_name} in WARNING state")

                # 所有舵机恢复正常时，自动重置 _cancel_playback 允许后续动作
                if not has_fault and self._cancel_playback.is_set():
                    all_recovered = all(
                        h.status in (HealthStatus.HEALTHY, HealthStatus.WARNING)
                        for h in health_results.values()
                    )
                    if all_recovered:
                        self._cancel_playback.clear()
                        self.logger.info("All motors recovered, resuming operations")

            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")

            # 等待下一次检查
            self._health_check_stop.wait(interval)

        self.logger.info("Health check loop stopped")

    def _on_health_status_change(self, motor_name: str, old_status, new_status) -> None:
        """舵机状态转为 CRITICAL/STALLED 时触发（在 health_check 线程中调用）"""
        if self._motor_fault_callback:
            try:
                self._motor_fault_callback(motor_name, old_status, new_status)
            except Exception as e:
                self.logger.warning(f"motor_fault_callback error: {e}")

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
        
        with self._bus_lock:
            return self.health_monitor.detect_stall(motor_name)

    def reset_health_statistics(self, motor_name: str = None):
        """重置健康统计数据"""
        if self.health_monitor:
            self.health_monitor.reset_statistics(motor_name)

    def clear_health_history(self, motor_name: str = None):
        """清除健康历史记录"""
        if self.health_monitor:
            self.health_monitor.clear_history(motor_name)
