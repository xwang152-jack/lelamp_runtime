#!/usr/bin/env python3
"""
简化的人脸和手部追踪测试（无需GUI）
"""
import cv2
import mediapipe as mp
import time

print('=== LeLamp 边缘视觉测试 ===')
print('测试人脸识别和手部追踪功能\n')

# 初始化
mp_face = mp.solutions.face_detection
mp_hands = mp.solutions.hands

face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

print('✅ MediaPipe 初始化成功')

# 打开摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('❌ 无法打开摄像头')
    exit(1)

print('✅ 摄像头打开成功')
print('\n开始 10 秒测试...')
print('请在摄像头前展示人脸或手部\n')

# 测试参数
test_duration = 10  # 10秒测试
start_time = time.time()
frame_count = 0
face_frames = 0
hand_frames = 0
max_faces = 0
max_hands = 0

try:
    while (time.time() - start_time) < test_duration:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 转换为 RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 人脸检测
        face_results = face_detection.process(rgb_frame)
        current_faces = len(face_results.detections) if face_results.detections else 0
        if current_faces > 0:
            face_frames += 1
            max_faces = max(max_faces, current_faces)

        # 手部追踪
        hand_results = hands.process(rgb_frame)
        current_hands = len(hand_results.multi_hand_landmarks) if hand_results.multi_hand_landmarks else 0
        if current_hands > 0:
            hand_frames += 1
            max_hands = max(max_hands, current_hands)

        # 每秒显示一次进度
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            print(f'[{elapsed:.0f}s] 帧: {frame_count} | 当前人脸: {current_faces} | 当前手部: {current_hands}')

except KeyboardInterrupt:
    print('\n测试被中断')

finally:
    cap.release()

    # 显示结果
    elapsed_time = time.time() - start_time
    avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    print('\n=== 测试结果 ===')
    print(f'测试时长: {elapsed_time:.1f} 秒')
    print(f'总帧数: {frame_count}')
    print(f'平均 FPS: {avg_fps:.1f}')
    print(f'检测到人脸的帧数: {face_frames} ({face_frames/frame_count*100:.1f}%)')
    print(f'检测到手部的帧数: {hand_frames} ({hand_frames/frame_count*100:.1f}%)')
    print(f'最大同时检测人脸数: {max_faces}')
    print(f'最大同时检测手部数: {max_hands}')
    print('')

    if face_frames > 0:
        print('✅ 人脸识别功能正常')
    else:
        print('⚠️  未检测到人脸')

    if hand_frames > 0:
        print('✅ 手部追踪功能正常')
    else:
        print('⚠️  未检测到手部')

    if face_frames > 0 or hand_frames > 0:
        print('\n🎉 边缘视觉功能测试通过！')
        print('   人脸识别和手部追踪可以在树莓派上正常运行。')
    else:
        print('\n💡 建议:')
        print('   1. 确保摄像头前有人脸或手部')
        print('   2. 检查光线条件')
        print('   3. 调整摄像头角度')
        print('   4. 重新运行测试')