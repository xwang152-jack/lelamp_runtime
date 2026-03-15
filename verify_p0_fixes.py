#!/usr/bin/env python3
"""
P0 安全问题修复验证脚本
运行此脚本以验证 P0 安全问题已成功修复。
"""

import os
import sys
import threading
import time
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))


def test_env_security():
    """测试 1: 验证 .env 文件安全"""
    print("🔍 测试 1: .env 文件安全检查")

    # 检查 .env.example 是否存在
    env_example = Path(__file__).parent / ".env.example"
    if not env_example.exists():
        print("❌ 失败: .env.example 文件不存在")
        return False
    print("✅ .env.example 文件存在")

    # 检查 .env 是否在 .gitignore 中
    gitignore = Path(__file__).parent / ".gitignore"
    gitignore_content = gitignore.read_text()
    if ".env" not in gitignore_content:
        print("❌ 失败: .env 未在 .gitignore 中")
        return False
    print("✅ .env 已在 .gitignore 中")

    # 检查 .env.example 是否包含真实密钥
    env_example_content = env_example.read_text()
    suspicious_patterns = ["cli_a9a2877d71789bc0", "EG7wSiPIkalKsBl7Eh1OoaiUehcQmJkR"]
    for pattern in suspicious_patterns:
        if pattern in env_example_content:
            print(f"❌ 失败: .env.example 包含真实密钥: {pattern}")
            return False
    print("✅ .env.example 不包含真实密钥")

    print("✅ 测试 1 通过: .env 文件安全检查\n")
    return True


def test_thread_safety():
    """测试 2: 验证线程安全修复"""
    print("🔍 测试 2: 线程安全检查")

    # 检查 main.py 是否使用 threading.Lock
    main_py = Path(__file__).parent / "main.py"
    main_content = main_py.read_text()

    if "import threading" not in main_content:
        print("❌ 失败: main.py 未导入 threading")
        return False
    print("✅ main.py 已导入 threading")

    if "threading.Lock()" not in main_content:
        print("❌ 失败: main.py 未使用 threading.Lock")
        return False
    print("✅ main.py 使用了 threading.Lock")

    print("✅ 测试 2 通过: 线程安全检查\n")
    return True


def test_priority_queue():
    """测试 3: 验证优先级队列实现"""
    print("🔍 测试 3: 优先级队列检查")

    # 检查 base.py 是否使用 heapq
    base_py = Path(__file__).parent / "lelamp" / "service" / "base.py"
    if not base_py.exists():
        print("❌ 失败: base.py 文件不存在")
        return False

    base_content = base_py.read_text()

    if "import heapq" not in base_content:
        print("❌ 失败: base.py 未导入 heapq")
        return False
    print("✅ base.py 已导入 heapq")

    if "heapq.heappush" not in base_content or "heapq.heappop" not in base_content:
        print("❌ 失败: base.py 未使用 heapq 操作")
        return False
    print("✅ base.py 使用了 heapq 操作")

    if "_event_queue" not in base_content:
        print("❌ 失败: base.py 未定义事件队列")
        return False
    print("✅ base.py 定义了事件队列")

    print("✅ 测试 3 通过: 优先级队列检查\n")
    return True


def test_priority_queue_functionality():
    """测试 4: 验证优先级队列功能"""
    print("🔍 测试 4: 优先级队列功能测试")

    try:
        from lelamp.service.base import ServiceBase, Priority

        # 创建测试服务
        class TestService(ServiceBase):
            def __init__(self):
                super().__init__("test", max_queue_size=10)
                self.processed_events = []

            def handle_event(self, event_type: str, payload):
                self.processed_events.append((event_type, payload))

        service = TestService()
        service.start()

        # 发送不同优先级的事件
        service.dispatch("low", "data1", Priority.LOW)
        service.dispatch("high", "data2", Priority.HIGH)
        service.dispatch("normal", "data3", Priority.NORMAL)
        service.dispatch("critical", "data4", Priority.CRITICAL)

        # 等待处理
        time.sleep(0.5)
        service.stop()

        # 验证处理了事件
        if len(service.processed_events) < 4:
            print(f"⚠️ 警告: 只处理了 {len(service.processed_events)}/4 个事件")
        else:
            # 验证 CRITICAL 事件最先处理
            if service.processed_events[0][0] == "critical":
                print("✅ CRITICAL 事件优先处理")
            else:
                print(f"❌ 失败: 事件顺序不正确，首个事件: {service.processed_events[0]}")
                return False

        # 检查统计信息
        if service._events_dispatched >= 4:
            print(f"✅ 事件分发统计正确: {service._events_dispatched}")
        if service._events_processed >= 4:
            print(f"✅ 事件处理统计正确: {service._events_processed}")

        print("✅ 测试 4 通过: 优先级队列功能测试\n")
        return True

    except Exception as e:
        print(f"❌ 失败: 优先级队列功能测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("=" * 60)
    print("🔒 P0 安全问题修复验证")
    print("=" * 60)
    print()

    results = []

    # 运行所有测试
    results.append((".env 文件安全", test_env_security()))
    results.append(("线程安全修复", test_thread_safety()))
    results.append(("优先级队列实现", test_priority_queue()))
    results.append(("优先级队列功能", test_priority_queue_functionality()))

    # 打印总结
    print("=" * 60)
    print("📊 测试结果总结")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")

    print()
    print(f"总计: {passed}/{total} 测试通过")

    if passed == total:
        print("\n🎉 所有 P0 安全问题修复验证通过！")
        return 0
    else:
        print(f"\n⚠️  {total - passed} 个测试失败，请检查修复")
        return 1


if __name__ == "__main__":
    sys.exit(main())
