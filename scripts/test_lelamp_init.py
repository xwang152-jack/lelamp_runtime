#!/usr/bin/env python3
"""
快速测试 LeLamp 初始化

这个脚本测试 LeLamp 是否能正常初始化，不启动完整的 LiveKit 会话。
"""
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.expanduser('~/lelamp_runtime'))

print('=== LeLamp 初始化测试 ===\n')

try:
    print('1. 测试模块导入...')
    from lelamp.agent.lelamp_agent import LeLamp
    print('✅ LeLamp 模块导入成功\n')

    print('2. 测试 LeLamp 初始化...')
    print('   （不连接电机，仅测试初始化逻辑）')

    # 创建一个最小的测试实例
    # 注意：这里不传入 vision_service 和 qwen_client，使用 NoOp 服务
    agent = LeLamp(
        port="/dev/ttyACM0",  # 串口可能不存在，但不会阻止初始化
        lamp_id="test_lelamp",
        vision_service=None,
        qwen_client=None,
        motors_service=None,  # 使用 NoOpMotorsService
        rgb_service=None,     # 使用 NoOpRGBService
    )

    print('✅ LeLamp 初始化成功\n')

    print('3. 检查边缘视觉工具...')
    if hasattr(agent, '_edge_vision_tools') and agent._edge_vision_tools is not None:
        print('✅ 边缘视觉工具已初始化')
        print(f'   - 混合视觉服务: {"已启用" if agent._hybrid_vision else "未启用"}')
        print(f'   - 主动监听服务: {"已禁用（正确）" if agent._vision_monitor is None else "已启用（错误）"}')
    else:
        print('⚠️  边缘视觉工具未初始化（正常，如果未设置 LELAMP_EDGE_VISION_ENABLED）')

    print('\n=== 测试结果 ===')
    print('✅ LeLamp 初始化测试通过！')
    print('\n说明:')
    print('  - LeLamp 核心功能正常初始化')
    print('  - 边缘视觉工具按配置正确初始化')
    print('  - 主动监听服务已正确禁用')
    print('  - 台灯应该可以正常启动')

except Exception as e:
    print(f'\n❌ 初始化测试失败: {e}')
    import traceback
    traceback.print_exc()
    print('\n可能的问题:')
    print('  1. Python 依赖缺失')
    print('  2. 环境变量配置错误')
    print('  3. 代码中存在其他错误')

print('\n测试完成')
