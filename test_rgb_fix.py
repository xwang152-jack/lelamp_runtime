#!/usr/bin/env python3
"""
测试RGB控制修复 - 验证命令执行逻辑

此脚本模拟WebSocket命令执行,验证修复是否正确。
"""
import asyncio
import sys
from lelamp.service.motors.noop_motors_service import NoOpMotorsService
from lelamp.service.rgb.noop_rgb_service import NoOpRGBService
from lelamp.agent.lelamp_agent import LeLamp


async def test_command_execution():
    """测试命令执行和错误检测"""

    print("=" * 50)
    print("RGB控制修复验证测试")
    print("=" * 50)
    print()

    # 初始化服务(使用NoOp版本用于开发环境测试)
    motors_service = NoOpMotorsService()
    rgb_service = NoOpRGBService()

    motors_service.start()
    rgb_service.start()

    # 创建Agent
    agent = LeLamp(
        port="/dev/ttyACM0",
        lamp_id="lelamp",
        motors_service=motors_service,
        rgb_service=rgb_service,
    )

    print("✓ Agent 创建成功")
    print()

    # 测试用例: (action, params, description, should_succeed)
    test_cases = [
        ("set_rgb_solid", {"r": 255, "g": 0, "b": 0}, "设置红色", True),
        ("set_rgb_solid", {"r": 0, "g": 255, "b": 0}, "设置绿色", True),
        ("set_rgb_solid", {"r": 0, "g": 0, "b": 255}, "设置蓝色", True),
        ("rgb_effect_rainbow", {}, "彩虹效果", True),
        ("rgb_effect_breathing", {"r": 255, "g": 0, "b": 0}, "红色呼吸", True),
        ("invalid_command", {}, "无效命令", False),
    ]

    passed = 0
    failed = 0

    for action, params, description, should_succeed in test_cases:
        print(f"测试: {description} ({action})")
        print(f"参数: {params}")

        try:
            result = await agent._execute_command(action, params)
            print(f"结果: {result}")

            # 应用WebSocket路由中的错误检测逻辑
            is_error = (
                result and (
                    "失败" in result or
                    "错误" in result or
                    "未知指令" in result
                )
            )

            if should_succeed:
                if is_error:
                    print(f"❌ 失败: 预期成功但检测到错误")
                    failed += 1
                else:
                    print(f"✓ 成功")
                    passed += 1
            else:
                if is_error:
                    print(f"✓ 成功: 正确检测到错误")
                    passed += 1
                else:
                    print(f"❌ 失败: 预期错误但未检测到")
                    failed += 1

        except Exception as e:
            print(f"❌ 异常: {e}")
            failed += 1

        print()

    # 清理
    rgb_service.stop()
    motors_service.stop()

    # 总结
    print("=" * 50)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 50)

    if failed == 0:
        print("✅ 所有测试通过! 修复验证成功")
        return 0
    else:
        print("❌ 部分测试失败")
        return 1


async def test_error_detection_logic():
    """测试错误检测逻辑"""

    print("\n" + "=" * 50)
    print("错误检测逻辑测试")
    print("=" * 50)
    print()

    # 测试字符串
    test_strings = [
        ("设置纯色灯光: RGB(255, 0, 0)", False, "正常成功消息"),
        ("启动彩虹效果（速度: 1.0x）", False, "正常成功消息"),
        ("执行失败: 参数无效", True, "包含'失败'"),
        ("错误：缺少 RGB 参数", True, "包含'错误'"),
        ("未知指令: invalid_command", True, "包含'未知指令'"),
        # 注意: 空字符串和None在WebSocket路由中不会触发错误检测
        # 这是正确的,因为没有实际执行任何命令
    ]

    passed = 0
    failed = 0

    for text, should_error, description in test_strings:
        # 应用WebSocket路由中的检测逻辑
        is_error = (
            text and (
                "失败" in text or
                "错误" in text or
                "未知指令" in text
            )
        )

        if is_error == should_error:
            print(f"✓ {description}: 正确检测")
            passed += 1
        else:
            print(f"❌ {description}: 检测错误")
            print(f"   文本: {text}")
            print(f"   预期: {'错误' if should_error else '成功'}")
            print(f"   实际: {'错误' if is_error else '成功'}")
            failed += 1

    print()
    print(f"结果: {passed}/{passed + failed} 通过")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    print("LeLamp RGB控制修复验证\n")

    # 测试1: 命令执行
    result1 = asyncio.run(test_command_execution())

    # 测试2: 错误检测逻辑
    result2 = asyncio.run(test_error_detection_logic())

    sys.exit(max(result1, result2))
