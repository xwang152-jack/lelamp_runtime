#!/usr/bin/env python3
"""
快速测试主动监听功能
"""
import sys
import os
import time

# 添加项目路径
sys.path.insert(0, os.path.expanduser('~/lelamp_runtime'))

print('=== 快速测试主动监听 ===\n')

try:
    # 1. 测试导入
    print('1. 测试模块导入...')
    from lelamp.service.vision.proactive_vision_monitor import ProactiveVisionMonitor
    from lelamp.edge.hybrid_vision import HybridVisionService
    from lelamp.service.vision.vision_service import VisionService
    print('✅ 模块导入成功\n')

    # 2. 初始化视觉服务
    print('2. 初始化视觉服务...')
    vision_service = VisionService(enabled=True)
    vision_service.start()
    print('✅ 视觉服务启动成功\n')

    # 3. 初始化混合视觉
    print('3. 初始化混合视觉...')
    hybrid_vision = HybridVisionService(
        enable_face=True,
        enable_hand=True,
        enable_object=False,  # 树莓派上物体检测有问题
    )
    print('✅ 混合视觉初始化成功\n')

    # 4. 创建测试回调
    def on_gesture(gesture, context):
        print(f'🎭 检测到手势: {gesture.value} at {time.strftime("%H:%M:%S")}')

    def on_presence(present):
        status = '👤 用户在场' if present else '🚫 用户离开'
        print(f'{status} at {time.strftime("%H:%M:%S")}')

    # 5. 初始化主动监听
    print('4. 初始化主动监听服务...')
    monitor = ProactiveVisionMonitor(
        hybrid_vision=hybrid_vision,
        vision_service=vision_service,
        gesture_callback=on_gesture,
        presence_callback=on_presence,
        enable_auto_gesture=True,
        enable_auto_presence=True,
        active_fps=5,    # 降低FPS避免占用摄像头
        idle_fps=1,
    )
    print('✅ 主动监听服务初始化成功\n')

    # 6. 启动监听
    print('5. 启动主动监听...')
    print('   测试时长: 20 秒')
    print('   请在摄像头前展示人脸或手势\n')
    print('开始监听...\n')

    monitor.start()

    # 运行测试
    try:
        start_time = time.time()
        last_stats_time = start_time

        while (time.time() - start_time) < 20:
            time.sleep(1)

            # 每3秒显示一次状态
            if time.time() - last_stats_time >= 3:
                stats = monitor.get_stats()
                print(f'[{int(time.time()-start_time)}s] '
                      f'模式:{stats["mode"]} | '
                      f'用户:{("在场" if stats["user_present"] else "不在")} | '
                      f'检测:{stats["detection_count"]} | '
                      f'手势:{stats["gesture_count"]}')
                last_stats_time = time.time()

    except KeyboardInterrupt:
        print('\n测试被中断')

    # 停止监听
    monitor.stop()
    vision_service.stop()

    # 显示结果
    print('\n=== 测试结果 ===')
    final_stats = monitor.get_stats()

    print(f'运行状态: {"运行中" if final_stats["running"] else "已停止"}')
    print(f'检测次数: {final_stats["detection_count"]}')
    print(f'手势次数: {final_stats["gesture_count"]}')

    if final_stats["gesture_count"] > 0:
        print('\n🎉 主动监听功能正常！')
        print('   系统能够自动检测手势')
    elif final_stats["detection_count"] > 0:
        print('\n✅ 监听服务在运行')
        print('   建议在摄像头前展示手势进行测试')
    else:
        print('\n⚠️  未检测到任何活动')
        print('   请检查:')
        print('   1. 摄像头是否正常工作')
        print('   2. 是否在摄像头前展示人脸或手势')
        print('   3. 光线条件是否良好')

except Exception as e:
    print(f'\n❌ 测试失败: {e}')
    import traceback
    traceback.print_exc()

print('\n测试完成')