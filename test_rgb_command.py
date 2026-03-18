#!/usr/bin/env python3
"""
测试 RGB 指令功能
"""
import asyncio
import json
from lelamp.service.rgb.rgb_service import RGBService
from lelamp.service.motors.noop_motors_service import NoOpMotorsService
from lelamp.agent.lelamp_agent import LeLamp


async def test_rgb_command():
    """测试 RGB 指令执行"""

    # 初始化服务
    rgb_service = RGBService()
    motors_service = NoOpMotorsService()

    # 启动服务
    rgb_service.start()
    motors_service.start()

    # 创建 Agent
    agent = LeLamp(
        port="/dev/ttyACM0",
        lamp_id="lelamp",
        motors_service=motors_service,
        rgb_service=rgb_service,
    )

    print("✓ Agent 创建成功")

    # 测试 _execute_command 方法
    test_commands = [
        ("set_rgb_solid", {"r": 255, "g": 0, "b": 0}, "设置红色"),
        ("set_rgb_solid", {"r": 0, "g": 255, "b": 0}, "设置绿色"),
        ("set_rgb_solid", {"r": 0, "g": 0, "b": 255}, "设置蓝色"),
        ("rgb_effect_rainbow", {}, "彩虹效果"),
        ("rgb_effect_breathing", {"r": 255, "g": 0, "b": 0}, "红色呼吸"),
    ]

    for action, params, description in test_commands:
        print(f"\n测试: {description} ({action})")
        print(f"参数: {params}")

        try:
            result = await agent._execute_command(action, params)
            print(f"结果: {result}")

            # 检查结果
            if "失败" in result or "错误" in result or "未知指令" in result:
                print(f"❌ 失败: {result}")
            else:
                print(f"✓ 成功")
        except Exception as e:
            print(f"❌ 异常: {e}")

    # 清理
    rgb_service.stop()
    motors_service.stop()

    print("\n测试完成!")


if __name__ == "__main__":
    asyncio.run(test_rgb_command())
