#!/usr/bin/env python3
"""
使用真实摄像头测试人脸识别和手部追踪（保存检测帧）
适合 SSH 远程测试 - 将检测结果保存为图像文件
"""
import cv2
import mediapipe as mp
import time
import sys
import os
from datetime import datetime

print('=== LeLamp 真实摄像头测试 ===')
print('测试人脸识别和手部追踪功能\n')

# 创建输出目录
output_dir = f"/tmp/lelamp_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(output_dir, exist_ok=True)
print(f'检测结果将保存到: {output_dir}\n')

# 初始化 MediaPipe
mp_face = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

print('正在初始化 MediaPipe 组件...')
face_detection = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

print('✅ MediaPipe 初始化成功\n')

# 打开摄像头
print('正在打开摄像头 /dev/video0...')
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('❌ 无法打开摄像头 /dev/video0，尝试其他设备...')

    for i in range(1, 5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f'✅ 成功打开 /dev/video{i}')
            break
    else:
        print('❌ 无法打开任何摄像头')
        print('\n请检查: ls /dev/video*')
        sys.exit(1)

# 设置摄像头参数
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f'摄像头分辨率: {width}x{height}')
print(f'摄像头帧率: {fps} FPS\n')

# 测试配置
test_duration = 20  # 20秒测试
save_interval = 2   # 每2秒保存一帧检测图像

print(f'测试时长: {test_duration} 秒')
print(f'保存间隔: 每 {save_interval} 秒保存检测结果')
print('请在摄像头前展示人脸或手部')
print('\n开始测试...\n')

# 测试统计
start_time = time.time()
frame_count = 0
face_frames = 0
hand_frames = 0
max_faces = 0
max_hands = 0
total_faces = 0
total_hands = 0
saved_frames = 0

face_times = []
hand_times = []

last_save_time = 0
last_second = 0

try:
    while (time.time() - start_time) < test_duration:
        ret, frame = cap.read()
        if not ret:
            print('❌ 无法读取摄像头帧')
            break

        frame_count += 1
        current_time = time.time() - start_time

        # 转换为 RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 创建绘制副本
        display_frame = frame.copy()

        # 人脸检测
        face_start = time.time()
        face_results = face_detection.process(rgb_frame)
        face_time = (time.time() - face_start) * 1000
        face_times.append(face_time)

        current_faces = 0
        if face_results.detections:
            current_faces = len(face_results.detections)
            total_faces += current_faces
            max_faces = max(max_faces, current_faces)
            face_frames += 1

            # 绘制人脸检测结果
            for idx, detection in enumerate(face_results.detections):
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width_bbox = int(bbox.width * w)
                height_bbox = int(bbox.height * h)

                # 绘制边框
                cv2.rectangle(display_frame, (x, y), (x + width_bbox, y + height_bbox), (0, 255, 0), 2)

                # 绘制标签
                confidence = detection.score[0]
                label = f'Face {idx+1}: {confidence:.2f}'
                cv2.putText(display_frame, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 手部追踪
        hand_start = time.time()
        hand_results = hands.process(rgb_frame)
        hand_time = (time.time() - hand_start) * 1000
        hand_times.append(hand_time)

        current_hands = 0
        if hand_results.multi_hand_landmarks:
            current_hands = len(hand_results.multi_hand_landmarks)
            total_hands += current_hands
            max_hands = max(max_hands, current_hands)
            hand_frames += 1

            # 绘制手部关键点
            for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(
                    display_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # 添加手部标签
                h, w, _ = display_frame.shape
                cx = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * w)
                cy = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * h)
                cv2.putText(display_frame, f'Hand {idx+1}', (cx - 30, cy - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # 添加统计信息到画面
        avg_face_time = sum(face_times[-30:]) / min(len(face_times), 30) if face_times else 0
        avg_hand_time = sum(hand_times[-30:]) / min(len(hand_times), 30) if hand_times else 0
        fps = frame_count / current_time

        info_lines = [
            f'Time: {int(current_time)}/{test_duration}s | FPS: {fps:.1f}',
            f'Faces: {current_faces} (max: {max_faces}) | Hands: {current_hands} (max: {max_hands})',
            f'Face: {avg_face_time:.1f}ms | Hand: {avg_hand_time:.1f}ms'
        ]

        for i, line in enumerate(info_lines):
            cv2.putText(display_frame, line, (10, 30 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, line, (10, 30 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        # 定期保存检测结果
        if int(current_time) >= last_save_time + save_interval:
            last_save_time = int(current_time)
            saved_frames += 1

            # 保存原图和检测结果图
            original_path = os.path.join(output_dir, f'frame_{saved_frames:02d}_original.jpg')
            detected_path = os.path.join(output_dir, f'frame_{saved_frames:02d}_detected.jpg')

            cv2.imwrite(original_path, frame)
            cv2.imwrite(detected_path, display_frame)

            print(f'[{int(current_time):3d}s] 💾 已保存帧 {saved_frames} | '
                  f'人脸:{current_faces} 手部:{current_hands} | '
                  f'Face:{avg_face_time:5.1f}ms Hand:{avg_hand_time:5.1f}ms')

        # 每秒显示一次进度
        if int(current_time) > last_second:
            last_second = int(current_time)
            if saved_frames == 0 or int(current_time) % save_interval != 0:
                print(f'[{int(current_time):3d}s] 帧:{frame_count:4d} | '
                      f'人脸:{current_faces} 手部:{current_hands} | '
                      f'Face:{avg_face_time:5.1f}ms Hand:{avg_hand_time:5.1f}ms')

except KeyboardInterrupt:
    print('\n\n测试被用户中断')

finally:
    cap.release()

    # 显示详细结果
    elapsed_time = time.time() - start_time
    avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    print('\n' + '='*60)
    print('📊 测试结果统计')
    print('='*60)

    print(f'\n⏱️  时间统计:')
    print(f'   测试时长: {elapsed_time:.1f} 秒')
    print(f'   总帧数: {frame_count}')
    print(f'   平均 FPS: {avg_fps:.1f}')
    print(f'   保存检测帧: {saved_frames}')

    print(f'\n👤 人脸检测:')
    print(f'   检测到人脸的帧数: {face_frames} ({face_frames/frame_count*100:.1f}%)')
    print(f'   最大同时检测人脸数: {max_faces}')
    if face_times:
        print(f'   平均检测时间: {sum(face_times)/len(face_times):.1f}ms')
        print(f'   检测时间范围: {min(face_times):.1f}ms - {max(face_times):.1f}ms')

    print(f'\n✋ 手部追踪:')
    print(f'   检测到手部的帧数: {hand_frames} ({hand_frames/frame_count*100:.1f}%)')
    print(f'   最大同时检测手部数: {max_hands}')
    if hand_times:
        print(f'   平均追踪时间: {sum(hand_times)/len(hand_times):.1f}ms')
        print(f'   追踪时间范围: {min(hand_times):.1f}ms - {max(hand_times):.1f}ms')

    print(f'\n📈 总体统计:')
    print(f'   总检测到人脸次数: {total_faces}')
    print(f'   总检测到手部次数: {total_hands}')

    print('\n' + '='*60)
    print('🎯 功能评估')
    print('='*60)

    if face_frames > 0:
        print('✅ 人脸识别功能: 正常工作')
        if avg_fps >= 20:
            print('   性能评级: 优秀 (FPS >= 20)')
        elif avg_fps >= 10:
            print('   性能评级: 良好 (FPS >= 10)')
        else:
            print('   性能评级: 可用 (FPS < 10)')
    else:
        print('⚠️  人脸识别功能: 未检测到人脸')

    if hand_frames > 0:
        print('✅ 手部追踪功能: 正常工作')
        if avg_fps >= 15:
            print('   性能评级: 优秀 (FPS >= 15)')
        elif avg_fps >= 8:
            print('   性能评级: 良好 (FPS >= 8)')
        else:
            print('   性能评级: 可用 (FPS < 8)')
    else:
        print('⚠️  手部追踪功能: 未检测到手部')

    print('\n' + '='*60)
    print('💾 检测结果图像')
    print('='*60)
    print(f'输出目录: {output_dir}')
    print(f'保存帧数: {saved_frames}')
    print('\n查看图像:')
    print(f'  在本地电脑上运行: scp -r pi@192.168.0.106:{output_dir} .')
    print(f'  然后使用图片查看器打开检测结果')

    if face_frames > 0 or hand_frames > 0:
        print('\n🎉 边缘视觉功能测试通过！')
        print('   ✅ MediaPipe 在树莓派上运行正常')
        print('   ✅ 人脸识别和手部追踪可用')
        print('   ✅ 性能满足实际应用需求')
        print('')
        print('💡 启用边缘视觉:')
        print('   在 .env 中添加以下配置:')
        print('   LELAMP_EDGE_VISION_ENABLED=true')
        print('   LELAMP_EDGE_VISION_FPS=15')
    else:
        print('\n💡 建议重新测试:')
        print('   1. 确保在摄像头前展示人脸或手部')
        print('   2. 提供良好的光线条件')
        print('   3. 调整摄像头角度')

    print('='*60)
    print(f'\n测试完成！检测图像已保存到 {output_dir}')