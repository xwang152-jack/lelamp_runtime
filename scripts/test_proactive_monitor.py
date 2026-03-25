#!/usr/bin/env python3
"""
测试主动监听功能
"""
import sys
import os
import time

# 添加项目路径
sys.path.insert(0, '/home/pi/lelamp_runtime')

from lelamp.service.vision.proactive_vision_monitor import ProactiveVisionMonitor
from lelamp.edge.hybrid_vision import HybridVisionService

print('=== 主动监听服务测试 ===\n')

# 创建测试回调
def on_gesture(gesture, context):
    print(f'🎭 检测到手势: {gesture.value}')
    print(f'   上下文: {context}')

def on_presence(present):
    status = '👤 用户在场' if present else '🚫 用户离开'
    print(f'{status} at {time.strftime("%H:%M:%S")}')

print('初始化混合视觉服务...')
try:
    # 注意：这个测试不需要云端客户端，因为我们只测试本地检测
    hybrid_vision = HybridVisionService(
        enable_face=True,
        enable_hand=True,
        enable_object=False,  # 物体检测在树莓派上有问题
    )
    print('✅ 混合视觉服务初始化成功')

    print('\n初始化主动监听服务...')
    monitor = ProactiveVisionMonitor(
        hybrid_vision=hybrid_vision,
        gesture_callback=on_gesture,
        presence_callback=on_presence,
        enable_auto_gesture=True,
        enable_auto_presence=True,
        active_fps=5,    # 降低FPS用于测试
        idle_fps=1,
    )
    print('✅ 主动监听服务初始化成功')

    print('\n开始监听测试（30秒）...')
    print('请在摄像头前展示人脸或手势')
    print('按 Ctrl+C 停止测试\n')

    monitor.start()

    # 运行30秒
    try:
        for i in range(30):
            time.sleep(1)
            stats = monitor.get_stats()
            if i % 5 == 0:  # 每5秒显示一次状态
                print(f'[{i+1}s] 模式: {stats["mode"]}, '
                      f'用户在场: {stats["user_present"]}, '
                      f'检测次数: {stats["detection_count"]}, '
                      f'手势次数: {stats["gesture_count"]}')
    except KeyboardInterrupt:
        print('\n测试被用户中断')

    monitor.stop()

    print('\n=== 测试结果 ===')
    final_stats = monitor.get_stats()
    print(f'总检测次数: {final_stats["detection_count"]}')
    print(f'总手势次数: {final_stats["gesture_count"]}')
    print(f'用户在场: {final_stats["user_present"]}')
    print(f'运行模式: {final_stats["mode"]}')

    if final_stats["gesture_count"] > 0:
        print('\n🎉 主动监听功能正常工作！')
        print('   系统能够自动检测并响应手势')
    elif final_stats["detection_count"] > 0:
        print('\n✅ 监听功能正在运行')
        print('   建议在摄像头前展示手势进行测试')
    else:
        print('\n⚠️  未检测到任何活动')
        print('   请检查:')
        print('   1. 摄像头是否正常')
        print('   2. 是否在摄像头前展示人脸或手势')
        print('   3. 光线条件是否良好')

except Exception as e:
    print(f'\n❌ 测试失败: {e}')
    import traceback
    traceback.print_exc()

print('\n测试完成')