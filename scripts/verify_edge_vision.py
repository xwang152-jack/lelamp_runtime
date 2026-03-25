#!/usr/bin/env python3
"""
验证边缘视觉功能是否已启用
"""
import os
import sys

print('=== LeLamp 边缘视觉功能验证 ===\n')

# 检查环境变量
env_file = '/home/pi/lelamp_runtime/.env'
print('📋 检查配置文件:')

edge_vision_config = {
    'LELAMP_EDGE_VISION_ENABLED': 'false',
    'LELAMP_EDGE_VISION_MODEL_DIR': '',
    'LELAMP_EDGE_VISION_CONFIDENCE_THRESHOLD': '0.5',
    'LELAMP_EDGE_VISION_FPS': '15',
}

if os.path.exists(env_file):
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                if key in edge_vision_config:
                    edge_vision_config[key] = value

# 显示配置状态
print('边缘视觉配置状态:')
for key, value in edge_vision_config.items():
    if key == 'LELAMP_EDGE_VISION_ENABLED':
        status = '✅ 启用' if value.lower() == 'true' else '❌ 未启用'
        print(f'  {key}: {value} {status}')
    else:
        print(f'  {key}: {value}')

print('')

# 验证模型文件
print('📁 检查模型文件:')
model_dir = edge_vision_config['LELAMP_EDGE_VISION_MODEL_DIR'] or '/home/pi/lelamp_runtime/models'
models = {
    'efficientdet_lite0.tflite': '物体检测模型',
    'blaze_face_full_range.tflite': '人脸检测模型',
    'gesture_recognizer.task': '手势识别模型'
}

for model_file, description in models.items():
    model_path = os.path.join(model_dir, model_file)
    if os.path.exists(model_path):
        size = os.path.getsize(model_path)
        print(f'  ✅ {description}: {model_file} ({size} bytes)')
    else:
        print(f'  ❌ {description}: {model_file} 未找到')

print('')

# 测试导入
print('🔧 测试边缘视觉模块:')
try:
    from lelamp.edge.hybrid_vision import HybridVisionService
    print('  ✅ HybridVisionService 导入成功')

    # 测试初始化
    service = HybridVisionService(
        enable_face=True,
        enable_hand=True,
        enable_object=True
    )
    print('  ✅ HybridVisionService 初始化成功')

    # 获取统计信息
    stats = service.get_stats()
    print(f'  📊 服务统计: {stats}')

except ImportError as e:
    print(f'  ❌ 导入失败: {e}')
except Exception as e:
    print(f'  ❌ 初始化失败: {e}')

print('')

# 测试具体检测器
print('🎯 测试具体检测器:')
try:
    from lelamp.edge.face_detector import FaceDetector
    detector = FaceDetector()
    stats = detector.get_stats()
    noop = stats.get('noop_mode', True)
    status = '❌ NoOp 模式' if noop else '✅ 正常模式'
    print(f'  FaceDetector: {status}')

except Exception as e:
    print(f'  ❌ FaceDetector 测试失败: {e}')

try:
    from lelamp.edge.hand_tracker import HandTracker
    tracker = HandTracker()
    stats = tracker.get_stats()
    noop = stats.get('noop_mode', True)
    status = '❌ NoOp 模式' if noop else '✅ 正常模式'
    print(f'  HandTracker: {status}')

except Exception as e:
    print(f'  ❌ HandTracker 测试失败: {e}')

try:
    from lelamp.edge.object_detector import ObjectDetector
    detector = ObjectDetector()
    stats = detector.get_stats()
    noop = stats.get('noop_mode', True)
    status = '❌ NoOp 模式' if noop else '✅ 正常模式'
    print(f'  ObjectDetector: {status}')

except Exception as e:
    print(f'  ❌ ObjectDetector 测试失败: {e}')

print('')

# 总结
print('='*50)
if edge_vision_config['LELAMP_EDGE_VISION_ENABLED'].lower() == 'true':
    print('✅ 边缘视觉功能已启用')
    print('   人脸识别和手部追踪将在本地运行')
else:
    print('❌ 边缘视觉功能未启用')
    print('   请在 .env 中设置: LELAMP_EDGE_VISION_ENABLED=true')
print('='*50)