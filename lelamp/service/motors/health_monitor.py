"""
舵机健康监控模块

提供舵机温度、电压、负载监控,堵转检测等功能,用于商用场景的可靠性保障。
"""

import logging
import time
import threading
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
from enum import Enum


logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """健康状态枚举"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    STALLED = "stalled"


@dataclass
class MotorHealthData:
    """单个舵机的健康数据"""
    motor_name: str
    timestamp: float
    temperature: Optional[float] = None  # 摄氏度
    voltage: Optional[float] = None      # 伏特
    load: Optional[float] = None         # 负载百分比 (0-1)
    position: Optional[float] = None     # 当前位置
    status: HealthStatus = HealthStatus.HEALTHY

    def to_dict(self) -> dict:
        """转换为字典"""
        data = asdict(self)
        data['status'] = self.status.value
        return data


@dataclass
class HealthThresholds:
    """健康阈值配置"""
    temp_warning_c: float = 65.0      # 温度警告阈值
    temp_critical_c: float = 75.0     # 温度危险阈值
    voltage_min_v: float = 4.5        # 最低电压
    voltage_max_v: float = 13.0       # 最高电压
    load_warning: float = 0.8         # 负载警告阈值
    load_stall: float = 0.95          # 堵转阈值
    position_error_deg: float = 5.0   # 位置误差阈值


class MotorHealthMonitor:
    """舵机健康监控器"""

    def __init__(self, bus, thresholds: Optional[HealthThresholds] = None):
        """
        初始化健康监控器

        Args:
            bus: FeetechMotorsBus 实例
            thresholds: 健康阈值配置
        """
        self.bus = bus
        self.thresholds = thresholds or HealthThresholds()
        self.logger = logging.getLogger(__name__)

        # 健康历史记录 (motor_name -> list of HealthData)
        self._health_history: Dict[str, list[MotorHealthData]] = {}
        self._history_lock = threading.Lock()
        self._max_history_size = 100  # 每个舵机保留最近 100 条记录

        # 统计数据
        self._warning_count: Dict[str, int] = {}
        self._critical_count: Dict[str, int] = {}
        self._stall_count: Dict[str, int] = {}

    def check_motor_health(self, motor_name: str, target_position: Optional[float] = None) -> MotorHealthData:
        """
        检查单个舵机的健康状态

        Args:
            motor_name: 舵机名称
            target_position: 目标位置(用于检测位置误差)

        Returns:
            MotorHealthData: 健康数据
        """
        health_data = MotorHealthData(
            motor_name=motor_name,
            timestamp=time.time()
        )

        try:
            # 读取温度 (如果舵机支持)
            try:
                temp_raw = self.bus.read("Present_Temperature", motor_name)
                if temp_raw is not None:
                    health_data.temperature = float(temp_raw)
            except Exception as e:
                self.logger.debug(f"Cannot read temperature for {motor_name}: {e}")

            # 读取电压
            try:
                voltage_raw = self.bus.read("Present_Voltage", motor_name)
                if voltage_raw is not None:
                    # Feetech 舵机电压单位通常是 0.1V
                    health_data.voltage = float(voltage_raw) / 10.0
            except Exception as e:
                self.logger.debug(f"Cannot read voltage for {motor_name}: {e}")

            # 读取负载
            try:
                load_raw = self.bus.read("Present_Load", motor_name)
                if load_raw is not None:
                    # 负载值通常是 0-1023 映射到 0-100%
                    health_data.load = abs(float(load_raw)) / 1023.0
            except Exception as e:
                self.logger.debug(f"Cannot read load for {motor_name}: {e}")

            # 读取位置
            try:
                pos_data = self.bus.sync_read("Present_Position")
                if motor_name in pos_data:
                    health_data.position = pos_data[motor_name]
            except Exception as e:
                self.logger.debug(f"Cannot read position for {motor_name}: {e}")

            # 分析健康状态
            health_data.status = self._analyze_health(health_data, target_position)

            # 记录到历史
            self._record_health(health_data)

            # 更新统计
            self._update_statistics(health_data)

        except Exception as e:
            self.logger.error(f"Error checking health for {motor_name}: {e}")
            health_data.status = HealthStatus.CRITICAL

        return health_data

    def check_all_motors_health(self, target_positions: Optional[Dict[str, float]] = None) -> Dict[str, MotorHealthData]:
        """
        检查所有舵机的健康状态

        Args:
            target_positions: 目标位置字典 {motor_name: position}

        Returns:
            Dict[str, MotorHealthData]: 所有舵机的健康数据
        """
        results = {}
        target_positions = target_positions or {}

        for motor_name in self.bus.motors:
            target_pos = target_positions.get(motor_name)
            results[motor_name] = self.check_motor_health(motor_name, target_pos)

        return results

    def _analyze_health(self, data: MotorHealthData, target_position: Optional[float]) -> HealthStatus:
        """
        分析健康数据,确定状态

        Args:
            data: 健康数据
            target_position: 目标位置

        Returns:
            HealthStatus: 健康状态
        """
        # 检查堵转 (最高优先级)
        if data.load is not None and data.load >= self.thresholds.load_stall:
            self.logger.error(f"Motor {data.motor_name} STALLED! Load: {data.load:.1%}")
            return HealthStatus.STALLED

        # 检查温度危险
        if data.temperature is not None and data.temperature >= self.thresholds.temp_critical_c:
            self.logger.error(f"Motor {data.motor_name} CRITICAL TEMPERATURE! {data.temperature:.1f}°C")
            return HealthStatus.CRITICAL

        # 检查电压异常
        if data.voltage is not None:
            if data.voltage < self.thresholds.voltage_min_v or data.voltage > self.thresholds.voltage_max_v:
                self.logger.error(f"Motor {data.motor_name} CRITICAL VOLTAGE! {data.voltage:.1f}V")
                return HealthStatus.CRITICAL

        # 检查温度警告
        if data.temperature is not None and data.temperature >= self.thresholds.temp_warning_c:
            self.logger.warning(f"Motor {data.motor_name} high temperature: {data.temperature:.1f}°C")
            return HealthStatus.WARNING

        # 检查负载警告
        if data.load is not None and data.load >= self.thresholds.load_warning:
            self.logger.warning(f"Motor {data.motor_name} high load: {data.load:.1%}")
            return HealthStatus.WARNING

        # 检查位置误差 (如果有目标位置)
        if target_position is not None and data.position is not None:
            position_error = abs(data.position - target_position)
            if position_error > self.thresholds.position_error_deg:
                self.logger.warning(f"Motor {data.motor_name} position error: {position_error:.1f}°")
                return HealthStatus.WARNING

        return HealthStatus.HEALTHY

    def _record_health(self, health_data: MotorHealthData):
        """记录健康数据到历史"""
        with self._history_lock:
            motor_name = health_data.motor_name
            if motor_name not in self._health_history:
                self._health_history[motor_name] = []

            self._health_history[motor_name].append(health_data)

            # 限制历史记录大小
            if len(self._health_history[motor_name]) > self._max_history_size:
                self._health_history[motor_name].pop(0)

    def _update_statistics(self, health_data: MotorHealthData):
        """更新统计计数"""
        motor_name = health_data.motor_name

        if motor_name not in self._warning_count:
            self._warning_count[motor_name] = 0
            self._critical_count[motor_name] = 0
            self._stall_count[motor_name] = 0

        if health_data.status == HealthStatus.WARNING:
            self._warning_count[motor_name] += 1
        elif health_data.status == HealthStatus.CRITICAL:
            self._critical_count[motor_name] += 1
        elif health_data.status == HealthStatus.STALLED:
            self._stall_count[motor_name] += 1

    def get_health_summary(self) -> dict:
        """
        获取健康状态汇总

        Returns:
            dict: 包含所有舵机的健康统计信息
        """
        summary = {}

        with self._history_lock:
            for motor_name in self.bus.motors:
                if motor_name in self._health_history and self._health_history[motor_name]:
                    latest = self._health_history[motor_name][-1]
                    history_count = len(self._health_history[motor_name])
                else:
                    latest = None
                    history_count = 0

                summary[motor_name] = {
                    "latest": latest.to_dict() if latest else None,
                    "history_count": history_count,
                    "warning_count": self._warning_count.get(motor_name, 0),
                    "critical_count": self._critical_count.get(motor_name, 0),
                    "stall_count": self._stall_count.get(motor_name, 0),
                }

        return summary

    def get_motor_history(self, motor_name: str, limit: int = 20) -> list[dict]:
        """
        获取指定舵机的健康历史

        Args:
            motor_name: 舵机名称
            limit: 返回最近 N 条记录

        Returns:
            list[dict]: 健康历史数据列表
        """
        with self._history_lock:
            if motor_name not in self._health_history:
                return []

            history = self._health_history[motor_name][-limit:]
            return [h.to_dict() for h in history]

    def detect_stall(self, motor_name: str) -> bool:
        """
        检测舵机是否堵转

        Args:
            motor_name: 舵机名称

        Returns:
            bool: 是否堵转
        """
        health = self.check_motor_health(motor_name)
        return health.status == HealthStatus.STALLED

    def reset_statistics(self, motor_name: Optional[str] = None):
        """
        重置统计数据

        Args:
            motor_name: 舵机名称,如果为 None 则重置所有
        """
        if motor_name:
            self._warning_count[motor_name] = 0
            self._critical_count[motor_name] = 0
            self._stall_count[motor_name] = 0
        else:
            self._warning_count.clear()
            self._critical_count.clear()
            self._stall_count.clear()

        self.logger.info(f"Health statistics reset for {motor_name or 'all motors'}")

    def clear_history(self, motor_name: Optional[str] = None):
        """
        清除健康历史记录

        Args:
            motor_name: 舵机名称,如果为 None 则清除所有
        """
        with self._history_lock:
            if motor_name:
                if motor_name in self._health_history:
                    self._health_history[motor_name].clear()
            else:
                self._health_history.clear()

        self.logger.info(f"Health history cleared for {motor_name or 'all motors'}")
