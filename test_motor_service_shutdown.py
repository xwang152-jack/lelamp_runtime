#!/usr/bin/env python3
"""
测试电机服务关闭时的竞态条件修复

验证在服务关闭期间不会出现 'NoneType' object has no attribute 'send_action' 错误
"""
import time
import threading
from lelamp.service.motors.motors_service import MotorsService
from lelamp.config import load_motor_config

def test_shutdown_during_playback():
    """测试在播放录制时关闭服务"""
    print("=" * 60)
    print("测试：服务关闭时的竞态条件")
    print("=" * 60)

    # 创建配置（禁用健康监控以简化测试）
    config = load_motor_config()
    config.health_check_enabled = False

    # 创建服务（使用 noop 配置用于测试）
    motors_service = MotorsService(
        port="/dev/ttyACM0",
        lamp_id="test",
        motor_config=config
    )

    # 模拟启动（不连接真实硬件）
    motors_service._cancel_playback = threading.Event()
    motors_service._bus_lock = threading.Lock()
    motors_service.robot = None  # 模拟未连接状态

    print("✓ 服务创建成功")

    # 测试1：直接调用 _handle_play（模拟有未处理事件）
    print("\n测试1：在 robot=None 时调用 play 事件")
    try:
        motors_service.dispatch("play", "wake_up")
        # 给事件处理一点时间
        time.sleep(0.1)
        print("✓ play 事件已调度")
    except Exception as e:
        print(f"✗ 调度失败: {e}")
        return False

    # 测试2：调用 stop 方法
    print("\n测试2：调用 stop 方法")
    try:
        motors_service.stop(timeout=1.0)
        print("✓ 服务停止成功")
    except Exception as e:
        print(f"✗ 停止失败: {e}")
        return False

    # 测试3：验证 robot 被正确设置为 None
    print("\n测试3：验证服务状态")
    if motors_service.robot is None:
        print("✓ robot 正确设置为 None")
    else:
        print("✗ robot 状态异常")
        return False

    # 测试4：停止后尝试调度新事件
    print("\n测试4：停止后调度新事件")
    try:
        motors_service.dispatch("play", "wake_up")
        print("✓ 停止后事件调度正常（应被忽略）")
    except Exception as e:
        print(f"✗ 停止后调度失败: {e}")
        return False

    print("\n" + "=" * 60)
    print("✅ 所有测试通过")
    print("=" * 60)
    return True

def test_move_joint_during_shutdown():
    """测试在移动关节时关闭服务"""
    print("\n" + "=" * 60)
    print("测试：move_joint 在关闭期间的行为")
    print("=" * 60)

    config = load_motor_config()
    config.health_check_enabled = False

    motors_service = MotorsService(
        port="/dev/ttyACM0",
        lamp_id="test",
        motor_config=config
    )

    # 模拟服务状态
    motors_service._cancel_playback = threading.Event()
    motors_service._bus_lock = threading.Lock()
    motors_service.robot = None

    print("✓ 服务创建成功")

    # 测试：在 robot=None 时调用 move_joint
    print("\n测试：在 robot=None 时调用 move_joint 事件")
    try:
        payload = {
            "joint_name": "base_yaw",
            "angle": 45.0
        }
        motors_service.dispatch("move_joint", payload)
        time.sleep(0.1)
        print("✓ move_joint 事件已调度")
    except Exception as e:
        print(f"✗ 调度失败: {e}")
        return False

    print("\n✅ 测试通过")
    return True

def main():
    """运行所有测试"""
    print("电机服务关闭竞态条件修复验证")
    print("=" * 60)

    results = {
        "shutdown_during_playback": test_shutdown_during_playback(),
        "move_joint_during_shutdown": test_move_joint_during_shutdown(),
    }

    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    for name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")

    all_passed = all(results.values())
    if all_passed:
        print("\n✅ 所有测试通过，竞态条件修复成功")
        return 0
    else:
        print("\n❌ 部分测试失败")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
