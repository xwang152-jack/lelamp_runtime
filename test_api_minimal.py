#!/usr/bin/env python3
"""
最小化 API 启动测试 - 逐步排查问题
"""
import sys
import asyncio
import logging

# 设置详细日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_imports():
    """测试 1: 模块导入"""
    print("=" * 60)
    print("测试 1: 模块导入")
    print("=" * 60)

    try:
        print("导入 FastAPI...")
        from fastapi import FastAPI
        print("✓ FastAPI 导入成功")

        print("导入 lelamp.api.app...")
        from lelamp.api import app
        print("✓ lelamp.api.app 导入成功")

        return True
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_database():
    """测试 2: 数据库连接"""
    print("\n" + "=" * 60)
    print("测试 2: 数据库连接")
    print("=" * 60)

    try:
        from lelamp.database.base import SessionLocal, engine
        from lelamp.database.models import Base

        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        print("✓ 数据库连接成功")
        return True
    except Exception as e:
        print(f"✗ 数据库连接失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_motors_service():
    """测试 3: 电机服务启动"""
    print("\n" + "=" * 60)
    print("测试 3: 电机服务启动")
    print("=" * 60)

    try:
        from lelamp.service.motors.motors_service import MotorsService
        from lelamp.config import load_motor_config

        motor_config = load_motor_config()
        print(f"✓ 电机配置加载成功: health_check_enabled={motor_config.health_check_enabled}")

        # 尝试启动服务
        motors_service = MotorsService(
            port="/dev/ttyACM0",
            lamp_id="lelamp",
            motor_config=motor_config
        )

        print("正在启动电机服务...")
        motors_service.start()
        print("✓ 电机服务启动成功")

        # 立即停止
        motors_service.stop()
        print("✓ 电机服务停止成功")

        return True
    except Exception as e:
        print(f"✗ 电机服务启动失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rgb_service():
    """测试 4: RGB 服务启动"""
    print("\n" + "=" * 60)
    print("测试 4: RGB 服务启动")
    print("=" * 60)

    try:
        from lelamp.service.rgb.rgb_service import RGBService

        rgb_service = RGBService()
        print("正在启动 RGB 服务...")
        rgb_service.start()
        print("✓ RGB 服务启动成功")

        # 立即停止
        rgb_service.stop()
        print("✓ RGB 服务停止成功")

        return True
    except Exception as e:
        print(f"✗ RGB 服务启动失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_agent_init():
    """测试 5: Agent 初始化"""
    print("\n" + "=" * 60)
    print("测试 5: Agent 初始化")
    print("=" * 60)

    try:
        from lelamp.agent.lelamp_agent import LeLamp
        from lelamp.service.motors.noop_motors_service import NoOpMotorsService
        from lelamp.service.rgb.noop_rgb_service import NoOpRGBService

        # 使用 noop 服务避免硬件依赖
        motors_service = NoOpMotorsService()
        motors_service.start()

        rgb_service = NoOpRGBService()
        rgb_service.start()

        print("正在初始化 LeLamp Agent...")
        agent = LeLamp(
            port="/dev/ttyACM0",
            lamp_id="lelamp",
            motors_service=motors_service,
            rgb_service=rgb_service,
        )
        print("✓ LeLamp Agent 初始化成功")

        # 清理
        motors_service.stop()
        rgb_service.stop()

        return True
    except Exception as e:
        print(f"✗ Agent 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    print("LeLamp API 启动诊断 - 逐步测试")
    print("=" * 60)

    results = {
        "模块导入": test_imports(),
        "数据库连接": test_database(),
        "电机服务": test_motors_service(),
        "RGB服务": test_rgb_service(),
        "Agent初始化": test_agent_init(),
    }

    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    for name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")

    all_passed = all(results.values())
    if all_passed:
        print("\n✅ 所有测试通过，API 应该可以正常启动")
        return 0
    else:
        print("\n❌ 部分测试失败，需要修复对应组件")
        return 1

if __name__ == "__main__":
    sys.exit(main())
