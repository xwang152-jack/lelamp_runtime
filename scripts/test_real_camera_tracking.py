#!/usr/bin/env python3
"""
使用真实摄像头测试人脸识别和手部追踪
"""
import cv2
import mediapipe as mp
import time

print('=== LeLamp 真实摄像头测试 ===')
print('测试人脸识别和手部追踪功能\n')

# 初始化 MediaPipe
mp_face = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

print('📋 测试配置:')
print('   人脸检测: 短距离模型')
print('   手部追踪: 最多 2 只手')
print('   测试时长: 30 秒')
print('')

# 初始化检测器
face_detection = mp_face.FaceDetection(
    model_selection=0,  # 0=短距离, 1=长距离
    min_detection_confidence=0.5
)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

print('✅ MediaPipe 组件初始化成功')

# 尝试打开摄像头
print('\n📹 正在打开摄像头...')
print('   尝试 /dev/video0...')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('   ❌ /dev/video0 失败，尝试 /dev/video1...')
    cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print('   ❌ 无法打开任何摄像头')
    print('\n可能的原因:')
    print('   1. 摄像头未连接')
    print('   2. 摄像头被其他程序占用')
    print('   3. 权限问题（运行: sudo usermod -a -G video $USER）')
    exit(1)

print('✅ 摄像头打开成功')

# 设置摄像头参数
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
actual_fps = cap.get(cv2.CAP_PROP_FPS)

print(f'   分辨率: {int(actual_width)}x{int(actual_height)}')
print(f'   帧率: {actual_fps} FPS')
print('')

print('🎮 测试控制:')
print('   - 程序将自动运行 30 秒')
print('   - 按 q 键: 立即退出')
print('   - 按 s 键: 切换显示模式')
print('   - 在摄像头前展示人脸或手部')
print('')

# 测试参数
test_duration = 30  # 30秒测试
show_landmarks = True
show_info = True

start_time = time.time()
frame_count = 0
face_frames = 0
hand_frames = 0
max_faces = 0
max_hands = 0
total_faces_detected = 0
total_hands_detected = 0

# 性能统计
face_times = []
hand_times = []

print('🚀 开始测试...\n')

try:
    while (time.time() - start_time) < test_duration:
        ret, frame = cap.read()
        if not ret:
            print('❌ 无法读取摄像头帧')
            break

        frame_count += 1

        # 转换为 RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 人脸检测
        face_start = time.time()
        face_results = face_detection.process(rgb_frame)
        face_time = (time.time() - face_start) * 1000
        face_times.append(face_time)

        current_faces = 0
        if face_results.detections:
            current_faces = len(face_results.detections)
            total_faces_detected += current_faces
            max_faces = max(max_faces, current_faces)
            face_frames += 1

            # 绘制人脸检测结果
            for detection in face_results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)

                # 绘制边框
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

                # 绘制标签
                confidence = detection.score[0]
                label = f'Face {confidence:.2f}'
                cv2.putText(frame, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 手部追踪
        hand_start = time.time()
        hand_results = hands.process(rgb_frame)
        hand_time = (time.time() - hand_start) * 1000
        hand_times.append(hand_time)

        current_hands = 0
        if hand_results.multi_hand_landmarks:
            current_hands = len(hand_results.multi_hand_landmarks)
            total_hands_detected += current_hands
            max_hands = max(max_hands, current_hands)
            hand_frames += 1

            # 绘制手部关键点
            for hand_landmarks in hand_results.multi_hand_landmarks:
                if show_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

        # 显示统计信息
        if show_info:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            avg_face_time = sum(face_times[-30:]) / min(len(face_times), 30)
            avg_hand_time = sum(hand_times[-30:]) / min(len(hand_times), 30)

            info_lines = [
                f'Time: {int(elapsed)}/{test_duration}s | FPS: {fps:.1f}',
                f'Faces: {current_faces} (max: {max_faces}) | Hands: {current_hands} (max: {max_hands})',
                f'Face: {avg_face_time:.1f}ms | Hand: {avg_hand_time:.1f}ms'
            ]

            for i, line in enumerate(info_lines):
                cv2.putText(frame, line, (10, 30 + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, line, (10, 30 + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        # 显示画面
        cv2.imshow('LeLamp Camera Test - Face & Hands Tracking', frame)

        # 键盘控制
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print('⏹️  用户退出测试')
            break
        elif key == ord('s'):
            show_landmarks = not show_landmarks
            mode = "显示骨架" if show_landmarks else "仅显示边框"
            print(f'🔄 切换显示模式: {mode}')

        # 每秒显示一次进度
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            print(f'[{elapsed:.0f}s] 帧: {frame_count} | 人脸: {current_faces} | 手部: {current_hands}')

except KeyboardInterrupt:
    print('\n⏹️  测试被中断')

finally:
    cap.release()
    cv2.destroyAllWindows()

    # 显示详细测试结果
    elapsed_time = time.time() - start_time
    avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    print('\n' + '='*50)
    print('📊 测试结果统计')
    print('='*50)

    print(f'\n⏱️  时间统计:')
    print(f'   测试时长: {elapsed_time:.1f} 秒')
    print(f'   总帧数: {frame_count}')
    print(f'   平均 FPS: {avg_fps:.1f}')

    print(f'\n👤 人脸检测:')
    print(f'   检测到人脸的帧数: {face_frames} ({face_frames/frame_count*100:.1f}%)')
    print(f'   最大同时检测人脸数: {max_faces}')
    if face_times:
        print(f'   平均检测时间: {sum(face_times)/len(face_times):.1f}ms')
        print(f'   最快检测时间: {min(face_times):.1f}ms')
        print(f'   最慢检测时间: {max(face_times):.1f}ms')

    print(f'\n✋ 手部追踪:')
    print(f'   检测到手部的帧数: {hand_frames} ({hand_frames/frame_count*100:.1f}%)')
    print(f'   最大同时检测手部数: {max_hands}')
    if hand_times:
        print(f'   平均追踪时间: {sum(hand_times)/len(hand_times):.1f}ms')
        print(f'   最快追踪时间: {min(hand_times):.1f}ms')
        print(f'   最慢追踪时间: {max(hand_times):.1f}ms')

    print(f'\n📈 总体检测:')
    print(f'   总检测到人脸次数: {total_faces_detected}')
    print(f'   总检测到手部次数: {total_hands_detected}')

    print('\n' + '='*50)
    print('🎯 功能评估')
    print('='*50)

    if face_frames > 0:
        print('✅ 人脸识别功能: 正常工作')
        if avg_fps >= 20:
            print('   性能: 优秀 (FPS >= 20)')
        elif avg_fps >= 10:
            print('   性能: 良好 (FPS >= 10)')
        else:
            print('   性能: 一般 (FPS < 10)')
    else:
        print('⚠️  人脸识别功能: 未检测到人脸')
        print('   可能原因: 测试环境中无人脸、光线不足或摄像头角度问题')

    if hand_frames > 0:
        print('✅ 手部追踪功能: 正常工作')
        if avg_fps >= 15:
            print('   性能: 优秀 (FPS >= 15)')
        elif avg_fps >= 8:
            print('   性能: 良好 (FPS >= 8)')
        else:
            print('   性能: 一般 (FPS < 8)')
    else:
        print('⚠️  手部追踪功能: 未检测到手部')
        print('   可能原因: 测试环境中无手部、光线不足或摄像头角度问题')

    print('\n' + '='*50)
    if face_frames > 0 or hand_frames > 0:
        print('🎉 边缘视觉功能测试通过！')
        print('   人脸识别和手部追踪在树莓派上运行正常。')
        print('   可以启用边缘视觉功能用于实际应用。')
        print('='*50)
        print('\n💡 启用建议:')
        print('   在 .env 中设置:')
        print('   LELAMP_EDGE_VISION_ENABLED=true')
        print('   LELAMP_EDGE_VISION_FPS=15')
        print('   LELAMP_EDGE_VISION_CONFIDENCE_THRESHOLD=0.5')
    else:
        print('💡 建议:')
        print('   1. 在有人的环境中重新测试')
        print('   2. 确保良好的光线条件')
        print('   3. 调整摄像头角度，确保人脸或手部在画面中')
        print('   4. 如果问题持续，检查摄像头硬件')
        print('='*50)

    print('\n测试完成！')