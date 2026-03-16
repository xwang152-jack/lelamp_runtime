# 舵机健康监控系统使用指南

## 概述

舵机健康监控系统是 LeLamp v2.0 的商业化功能之一,用于实时监控舵机的温度、电压、负载等关键指标,确保设备长期稳定运行。

## 功能特性

### 实时监控指标

- **温度监控**: 检测舵机工作温度,超过阈值时触发警告/危险状态
- **电压监控**: 监控供电电压,防止低电压导致的抖动或损坏
- **负载监控**: 检测舵机负载,识别堵转、过载等异常情况
- **位置精度**: 监控实际位置与目标位置的偏差

### 健康状态分级

| 状态 | 图标 | 说明 | 触发条件 |
|------|------|------|---------|
| HEALTHY | ✅ | 健康 | 所有指标正常 |
| WARNING | ⚠️ | 警告 | 温度/负载接近阈值 |
| CRITICAL | 🔴 | 危险 | 温度过高/电压异常 |
| STALLED | 🚫 | 堵转 | 负载超过 95% |

### 自动保护机制

- **堵转保护**: 检测到堵转时自动停止所有动作
- **过温保护**: 温度达到危险阈值时停止动作
- **历史记录**: 保留最近 100 条健康记录,便于故障诊断

## 配置说明

在 `.env` 文件中添加以下配置:

```bash
# 启用健康监控 (默认: true)
LELAMP_MOTOR_HEALTH_CHECK_ENABLED=true

# 检查间隔 (秒, 默认: 300 = 5分钟)
LELAMP_MOTOR_HEALTH_CHECK_INTERVAL_S=300.0

# 温度阈值 (摄氏度)
LELAMP_MOTOR_TEMP_WARNING_C=65.0    # 警告阈值
LELAMP_MOTOR_TEMP_CRITICAL_C=75.0   # 危险阈值

# 电压范围 (伏特)
LELAMP_MOTOR_VOLTAGE_MIN_V=11.0     # 最低安全电压
LELAMP_MOTOR_VOLTAGE_MAX_V=13.0     # 最高安全电压

# 负载阈值 (0-1)
LELAMP_MOTOR_LOAD_WARNING=0.8       # 负载警告 (80%)
LELAMP_MOTOR_LOAD_STALL=0.95        # 堵转检测 (95%)

# 位置误差容忍度 (度)
LELAMP_MOTOR_POSITION_ERROR_DEG=5.0
```

## 使用方法

### 1. 通过语音交互查询健康状态

用户可以通过语音直接询问舵机健康状态:

```
用户: "帮我检查一下舵机健康状态"
台灯: "所有舵机健康状态:
       base_yaw: ✅ healthy (温度: 52.3°C)
       base_pitch: ✅ healthy (温度: 48.7°C)
       elbow_pitch: ⚠️ warning (温度: 66.2°C)
       wrist_roll: ✅ healthy (温度: 45.1°C)
       wrist_pitch: ✅ healthy (温度: 47.9°C)"
```

查询单个舵机的详细信息:

```
用户: "查看 elbow_pitch 的健康数据"
台灯: "舵机 elbow_pitch 健康状态 ⚠️:
       - 状态: warning
       - 温度: 66.2°C
       - 电压: 12.1V
       - 负载: 75.3%
       - 位置: 45.2°

       统计信息:
       - 警告次数: 3
       - 危险次数: 0
       - 堵转次数: 0"
```

### 2. 通过代码调用 API

```python
# 获取所有舵机的健康摘要
summary = motors_service.get_motor_health_summary()

# 获取单个舵机的健康历史
history = motors_service.get_motor_health_history("base_yaw", limit=20)

# 检测堵转
is_stalled = motors_service.check_motor_stall("base_yaw")

# 重置统计数据
motors_service.reset_health_statistics("base_yaw")

# 清除历史记录
motors_service.clear_health_history("base_yaw")
```

### 3. OTA 远程调参 (PID 优化)

当舵机出现抖动或响应不稳定时,可以远程调整 PID 参数:

```
用户: "帮我调整 base_yaw 的 PID 参数,P 改成 20"
台灯: "✅ 成功更新舵机 base_yaw 的 PID 参数:
       - P: 20
       - I: 0
       - D: 32

       请测试动作是否稳定,如有问题可恢复默认值(P=16, I=0, D=32)"
```

**PID 参数说明**:

- **P (比例增益)**: 控制响应速度
  - 增大 P: 响应更快,但可能抖动
  - 减小 P: 更稳定,但响应慢
  - 推荐范围: 8-24,默认 16

- **I (积分增益)**: 消除稳态误差
  - 通常设为 0,除非有长期偏移
  - 推荐范围: 0-8,默认 0

- **D (微分增益)**: 减少超调
  - 增大 D: 更稳定,减少震荡
  - 减小 D: 可能超调
  - 推荐范围: 16-48,默认 32

### 4. 日志监控

启用 DEBUG 日志查看详细的健康检查信息:

```bash
LOG_LEVEL=DEBUG uv run main.py console
```

健康检查日志示例:

```
2026-03-16 10:30:00 INFO MotorHealthCheck Health check loop started (interval: 300.0s)
2026-03-16 10:30:01 DEBUG MotorHealthMonitor Checking health for base_yaw
2026-03-16 10:30:01 DEBUG MotorHealthMonitor Motor base_yaw: temp=52.3°C, voltage=12.1V, load=45.2%
2026-03-16 10:30:05 WARNING MotorHealthMonitor Motor elbow_pitch high temperature: 66.2°C
2026-03-16 10:35:00 DEBUG MotorHealthCheck Health check completed, next check in 300s
```

## 故障诊断

### 常见问题

#### 1. 温度警告/危险

**症状**: 舵机温度超过 65°C/75°C

**可能原因**:
- 长时间高负载运行
- 环境温度过高
- 散热不良

**解决方案**:
- 减少动作频率,增加冷却时间
- 改善设备散热环境
- 检查是否存在机械卡滞

#### 2. 堵转检测

**症状**: 负载超过 95%,动作停止

**可能原因**:
- 机械结构卡滞
- 碰到障碍物
- 舵机齿轮磨损

**解决方案**:
- 检查机械结构是否有异物
- 手动旋转关节,确认无卡滞
- 必要时更换舵机

#### 3. 电压异常

**症状**: 电压低于 11V 或高于 13V

**可能原因**:
- 电源适配器功率不足
- 线材压降过大
- 电池电量不足

**解决方案**:
- 更换 12V/2A 以上的电源适配器
- 使用更粗的电源线
- 检查电源连接是否松动

#### 4. 位置误差

**症状**: 实际位置与目标位置偏差超过 5°

**可能原因**:
- 负载过大
- PID 参数不合适
- 舵机老化

**解决方案**:
- 减轻负载
- 调整 PID 参数(增大 P 或 D)
- 更换老化舵机

## 商用场景建议

### 1. 量产前测试

在量产前进行疲劳测试:

```bash
# 运行 10 万次动作循环
for i in {1..100000}; do
    uv run -m lelamp.replay --id lelamp --port /dev/ttyACM0 --name nod
    sleep 1
done

# 每小时检查一次健康状态
watch -n 3600 'uv run main.py console <<< "检查舵机健康状态"'
```

记录:
- 温度变化曲线
- 位置精度衰减
- 故障发生频率

### 2. 生产环境监控

设置告警通知:

```python
# 在健康检查回调中添加告警逻辑
def on_motor_health_check(motor_name, health_data):
    if health_data.status == HealthStatus.CRITICAL:
        # 发送告警邮件/短信/飞书通知
        send_alert(f"Motor {motor_name} in CRITICAL state!")
```

### 3. 维护计划

根据健康统计制定维护计划:

- **警告次数 > 10**: 检查环境和负载
- **危险次数 > 3**: 安排维修检查
- **堵转次数 > 1**: 立即检查机械结构

### 4. 升级建议

根据长期监控数据决定硬件升级:

| 场景 | 监控指标 | 升级方案 |
|------|---------|---------|
| 频繁过温 | 温度警告 > 20次/天 | 更换金属齿轮舵机 |
| 精度衰减 | 位置误差 > 10° | 更换老化舵机 |
| 堵转频繁 | 堵转 > 5次/月 | 检查机械设计 |

## API 参考

### Function Tools

#### get_motor_health(motor_name: str = None) -> str

获取舵机健康状态。

**参数**:
- `motor_name`: 舵机名称(可选),留空返回所有舵机

**返回**: 健康状态描述字符串

#### tune_motor_pid(motor_name: str, p_coefficient: int, i_coefficient: int = 0, d_coefficient: int = 32) -> str

远程调整舵机 PID 参数。

**参数**:
- `motor_name`: 舵机名称
- `p_coefficient`: P 系数 (1-32)
- `i_coefficient`: I 系数 (0-32)
- `d_coefficient`: D 系数 (0-32)

**返回**: 调参结果

#### reset_motor_health_stats(motor_name: str = None) -> str

重置健康统计数据。

**参数**:
- `motor_name`: 舵机名称(可选),留空重置所有

**返回**: 重置确认消息

### Python API

```python
from lelamp.service.motors.health_monitor import MotorHealthMonitor, HealthThresholds

# 创建健康监控器
thresholds = HealthThresholds(
    temp_warning_c=65.0,
    temp_critical_c=75.0,
    load_stall=0.95
)
monitor = MotorHealthMonitor(bus, thresholds)

# 检查单个舵机
health = monitor.check_motor_health("base_yaw")
print(f"Status: {health.status.value}")
print(f"Temperature: {health.temperature}°C")

# 检查所有舵机
all_health = monitor.check_all_motors_health()

# 获取健康摘要
summary = monitor.get_health_summary()

# 获取历史记录
history = monitor.get_motor_history("base_yaw", limit=20)
```

## 最佳实践

1. **定期检查**: 每天检查一次健康摘要,关注异常趋势
2. **阈值调整**: 根据实际运行环境调整温度阈值
3. **数据保存**: 定期导出健康历史,用于长期分析
4. **预防性维护**: 警告次数增加时主动检查,避免严重故障
5. **OTA 优化**: 根据监控数据远程优化 PID 参数,减少上门维修

## 常见问题 (FAQ)

**Q: 为什么有的舵机不显示温度数据?**

A: 部分舵机型号不支持温度读取。Feetech STS3215 支持温度监控,其他型号需查阅规格书。

**Q: 健康检查会影响动作性能吗?**

A: 不会。健康检查在后台独立线程运行,默认 5 分钟检查一次,对动作延迟几乎无影响。

**Q: 如何导出健康历史数据?**

A: 暂不支持直接导出。可以通过 API 获取历史并保存为 CSV:

```python
history = motors_service.get_motor_health_history("base_yaw", limit=100)
import csv
with open("health_history.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=history[0].keys())
    writer.writeheader()
    writer.writerows(history)
```

**Q: 调整 PID 参数后需要重启吗?**

A: 不需要。PID 参数立即生效。但重启后会恢复默认值,除非持久化保存(TODO)。

## 技术支持

如遇到舵机健康监控相关问题,请提供:

1. 健康状态日志 (`LOG_LEVEL=DEBUG`)
2. 环境配置 (`.env` 文件,隐藏敏感信息)
3. 健康历史数据 (最近 50 条记录)
4. 硬件型号和使用时长

联系方式: [GitHub Issues](https://github.com/yourusername/lelamp_runtime/issues)
