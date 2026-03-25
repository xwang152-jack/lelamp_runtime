#!/usr/bin/env python3
"""
测试树莓派上的人脸识别和手部追踪功能
"""
import cv2
import mediapipe as mp
import numpy as np
import time

print('=== LeLamp 边缘视觉测试 - 人脸和手部追踪 ===\n')

# 初始化 MediaPipe
mp_face = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

print('📋 测试配置:')
print('   人脸检测模型: 短距离 (0)')
print('   手部追踪: 最大 2 只手')
print('   摄像头: /dev/video0')
print('')

# 初始化检测器
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

print('✅ MediaPipe 组件初始化成功')
print('')

# 打开摄像头
print('📹 正在打开摄像头...')
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('❌ 无法打开摄像头，请检查:')
    print('   1. 摄像头是否连接')
    print('   2. 用户是否有 video 组权限')
    print('   运行: sudo usermod -a -G video $USER')
    exit(1)

print('✅ 摄像头打开成功')
print('')

# 设置摄像头参数
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

print('🎮 测试说明:')
print('   - 按空格键: 捕获当前帧并分析')
print('   - 按 q 键: 退出测试')
print('   - 按 s 键: 切换显示模式')
print('')

# 测试参数
show_landmarks = True
test_frames = 0
max_test_frames = 100  # 测试 100 帧后自动退出
face_count = 0
hand_count = 0

start_time = time.time()

print('🚀 开始测试...\n')

try:
    while test_frames < max_test_frames:
        ret, frame = cap.read()
        if not ret:
            print('❌ 无法读取摄像头帧')
            break

        # 转换为 RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 人脸检测
        face_results = face_detection.process(rgb_frame)

        # 手部追踪
        hand_results = hands.process(rgb_frame)

        # 统计
        current_faces = 0
        current_hands = 0

        # 绘制人脸检测结果
        if face_results.detections:
            current_faces = len(face_results.detections)
            for detection in face_results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)

                # 绘制边框
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(frame, 'Face', (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 绘制手部追踪结果
        if hand_results.multi_hand_landmarks:
            current_hands = len(hand_results.multi_hand_landmarks)
            for hand_landmarks in hand_results.multi_hand_landmarks:
                if show_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # 显示统计信息
        fps = test_frames / (time.time() - start_time)
        info_text = f'Frame: {test_frames}/{max_test_frames} | FPS: {fps:.1f} | Faces: {current_faces} | Hands: {current_hands}'
        cv2.putText(frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 显示画面
        cv2.imshow('LeLamp Edge Vision Test', frame)

        # 更新统计
        if current_faces > 0:
            face_count += 1
        if current_hands > 0:
            hand_count += 1
        test_frames += 1

        # 键盘控制
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print('⏹️  用户退出测试')
            break
        elif key == ord('s'):
            show_landmarks = not show_landmarks
            mode = "显示骨架" if show_landmarks else "仅显示边框"
            print(f'🔄 切换显示模式: {mode}')
        elif key == ord(' '):
            print(f'📸 帧捕获 - 人脸: {current_faces}, 手部: {current_hands}')

except KeyboardInterrupt:
    print('\n⏹️  测试被中断')

finally:
    # 清理
    cap.release()
    cv2.destroyAllWindows()

    # 显示测试总结
    elapsed_time = time.time() - start_time
    avg_fps = test_frames / elapsed_time if elapsed_time > 0 else 0

    print('')
    print('=== 测试完成 ===')
    print(f'📊 测试统计:')
    print(f'   测试帧数: {test_frames}')
    print(f'   测试时长: {elapsed_time:.2f} 秒')
    print(f'   平均 FPS: {avg_fps:.1f}')
    print(f'   检测到人脸的帧数: {face_count} ({face_count/test_frames*100:.1f}%)')
    print(f'   检测到手部的帧数: {hand_count} ({hand_count/test_frames*100:.1f}%)')
    print('')

    if face_count > 0 or hand_count > 0:
        print('✅ 人脸识别和手部追踪功能正常工作！')
        print('   边缘视觉已就绪，可以启用本地检测功能。')
    else:
        print('⚠️  未检测到人脸或手部')
        print('   这可能是因为:')
        print('   1. 测试环境中没有人脸或手部')
        print('   2. 光线条件不佳')
        print('   3. 摄像头角度问题')
        print('   ')
        print('   建议: 在有人的环境中重新测试')