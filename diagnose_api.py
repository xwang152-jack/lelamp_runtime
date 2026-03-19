#!/usr/bin/env python3
"""
API 启动问题诊断脚本
"""
import sys
import os
import serial.tools.list_ports

def check_environment():
    """检查环境变量配置"""
    print("=" * 60)
    print("1. 环境变量检查")
    print("=" * 60)

    required_vars = [
        "DEEPSEEK_API_KEY",
        "BAIDU_SPEECH_API_KEY",
        "BAIDU_SPEECH_SECRET_KEY",
    ]

    optional_vars = [
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "LELAMP_PORT",
        "LELAMP_ID",
        "LELAMP_DEV_MODE",
    ]

    missing_required = []
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✓ {var}: {value[:20]}..." if len(value) > 20 else f"✓ {var}: {value}")
        else:
            print(f"✗ {var}: 缺失")
            missing_required.append(var)

    for var in optional_vars:
        value = os.getenv(var)
        if value:
            print(f"○ {var}: {value}")
        else:
            print(f"○ {var}: 未设置 (使用默认值)")

    if missing_required:
        print(f"\n❌ 缺少必需的环境变量: {', '.join(missing_required)}")
        return False
    else:
        print("\n✅ 所需环境变量已配置")
        return True

def check_hardware_access():
    """检查硬件访问权限"""
    print("\n" + "=" * 60)
    print("2. 硬件访问检查")
    print("=" * 60)

    # 检查串口设备
    port = os.getenv("LELAMP_PORT", "/dev/ttyACM0")
    print(f"配置的串口: {port}")

    available_ports = serial.tools.list_ports.comports()
    print(f"\n可用的串口设备:")
    for p in available_ports:
        print(f"  - {p.device}: {p.description}")

    if available_ports:
        print(f"\n✅ 找到 {len(available_ports)} 个串口设备")
        return True
    else:
        print(f"\n⚠️  未找到串口设备")
        return False

def check_database():
    """检查数据库"""
    print("\n" + "=" * 60)
    print("3. 数据库检查")
    print("=" * 60)

    try:
        from lelamp.database.base import SessionLocal, engine
        from lelamp.database.models import Base

        # 尝试连接数据库
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()

        print("✅ 数据库连接成功")

        # 检查表是否存在
        try:
            from sqlalchemy import inspect
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            print(f"✅ 数据库表: {', '.join(tables)}")
        except Exception as e:
            print(f"⚠️  无法检查表结构: {e}")

        return True
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")
        return False

def check_imports():
    """检查关键模块导入"""
    print("\n" + "=" * 60)
    print("4. 模块导入检查")
    print("=" * 60)

    modules = [
        ("FastAPI", "fastapi"),
        ("SQLAlchemy", "sqlalchemy"),
        ("Pydantic", "pydantic"),
        ("Uvicorn", "uvicorn"),
        ("MotorsService", "lelamp.service.motors.motors_service"),
        ("RGBService", "lelamp.service.rgb.rgb_service"),
        ("LeLamp Agent", "lelamp.agent.lelamp_agent"),
    ]

    all_ok = True
    for name, module_path in modules:
        try:
            __import__(module_path)
            print(f"✓ {name}")
        except ImportError as e:
            print(f"✗ {name}: {e}")
            all_ok = False
        except Exception as e:
            print(f"⚠️  {name}: {e}")

    return all_ok

def check_motor_voltage_config():
    """检查电机电压配置"""
    print("\n" + "=" * 60)
    print("5. 电机电压配置检查")
    print("=" * 60)

    config = {
        "LELAMP_MOTOR_VOLTAGE_MIN_V": os.getenv("LELAMP_MOTOR_VOLTAGE_MIN_V", "11.0"),
        "LELAMP_MOTOR_VOLTAGE_MAX_V": os.getenv("LELAMP_MOTOR_VOLTAGE_MAX_V", "13.0"),
        "LELAMP_MOTOR_TEMP_WARNING_C": os.getenv("LELAMP_MOTOR_TEMP_WARNING_C", "65.0"),
        "LELAMP_MOTOR_TEMP_CRITICAL_C": os.getenv("LELAMP_MOTOR_TEMP_CRITICAL_C", "75.0"),
    }

    print("当前配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    min_v = float(config["LELAMP_MOTOR_VOLTAGE_MIN_V"])
    if min_v > 10.0:
        print(f"\n⚠️  最小电压配置为 {min_v}V，但如果实际电压只有 5V，会导致服务异常")
        return False
    else:
        print(f"\n✅ 电压配置合理")
        return True

def main():
    """主诊断流程"""
    print("LeLamp API 启动诊断工具")
    print("=" * 60)

    results = {
        "环境变量": check_environment(),
        "硬件访问": check_hardware_access(),
        "数据库": check_database(),
        "模块导入": check_imports(),
        "电压配置": check_motor_voltage_config(),
    }

    print("\n" + "=" * 60)
    print("诊断总结")
    print("=" * 60)

    for name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")

    all_passed = all(results.values())
    if all_passed:
        print("\n✅ 所有检查通过，API 应该可以正常启动")
        return 0
    else:
        print("\n❌ 存在问题，API 可能无法正常启动")
        print("\n建议:")
        if not results["环境变量"]:
            print("  1. 检查 .env 文件配置")
        if not results["硬件访问"]:
            print("  2. 检查硬件连接和用户权限")
        if not results["数据库"]:
            print("  3. 检查数据库配置和权限")
        if not results["电压配置"]:
            print("  4. 调整电机电压配置或检查电源")
        return 1

if __name__ == "__main__":
    sys.exit(main())
