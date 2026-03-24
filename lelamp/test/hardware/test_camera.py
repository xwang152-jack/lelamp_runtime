"""
摄像头硬件测试脚本

用于诊断摄像头问题，包括：
- 测试 OpenCV 能否打开摄像头
- 测试不同摄像头索引
- 测试帧捕获
- 显示摄像头配置
"""
import asyncio
import os
import sys
import time
import platform

# Add parent directory to path to import from lelamp package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from lelamp.config import load_config


def test_opencv_import():
    """测试 OpenCV 是否可以导入"""
    print("\n=== 测试 OpenCV 导入 ===")
    try:
        import cv2
        print(f"OpenCV 版本: {cv2.__version__}")
        print(f"OpenCV 构建信息（前500字符）:\n{cv2.getBuildInformation()[:500]}")
        return cv2
    except ImportError as e:
        print(f"OpenCV 导入失败: {e}")
        return None


def test_camera_indices(cv2, max_index=5):
    """测试不同的摄像头索引"""
    print(f"\n=== 测试摄像头索引 (0-{max_index}) ===")

    results = {}
    for index in range(max_index + 1):
        print(f"尝试打开摄像头索引 {index}...", end=" ")
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2] if len(frame.shape) >= 2 else (0, 0)
                print(f"成功! 分辨率: {w}x{h}")
                results[index] = {"status": "ok", "width": w, "height": h}
            else:
                print("可以打开但无法读取帧")
                results[index] = {"status": "read_failed"}
        else:
            print("无法打开")
            results[index] = {"status": "failed"}
        cap.release()
    return results


def test_camera_config(cv2, index=0):
    """测试摄像头配置参数"""
    print(f"\n=== 测试摄像头配置 (索引 {index}) ===")

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"无法打开摄像头索引 {index}")
        return None

    # 获取当前配置
    print("\n当前配置:")
    print(f"  CAP_PROP_FRAME_WIDTH: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
    print(f"  CAP_PROP_FRAME_HEIGHT: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"  CAP_PROP_FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    print(f"  CAP_PROP_FORMAT: {cap.get(cv2.CAP_PROP_FORMAT)}")
    print(f"  CAP_PROP_MODE: {cap.get(cv2.CAP_PROP_MODE)}")

    # 尝试设置配置
    width = 1024
    height = 768
    print(f"\n尝试设置分辨率: {width}x{height}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"实际分辨率: {actual_w}x{actual_h}")

    # 读取几帧
    print("\n读取 3 帧测试...")
    for i in range(3):
        ret, frame = cap.read()
        if ret:
            print(f"  帧 {i+1}: OK, shape={frame.shape}")
        else:
            print(f"  帧 {i+1}: 失败")
        time.sleep(0.5)

    cap.release()
    return {"width": actual_w, "height": actual_h}


def test_vision_service():
    """测试 VisionService"""
    print("\n=== 测试 VisionService ===")

    from lelamp.config import load_vision_config

    config = load_config()
    vision_config = load_vision_config()

    print(f"配置:")
    print(f"  LELAMP_VISION_ENABLED: {vision_config.enabled}")
    print(f"  LELAMP_CAMERA_INDEX_OR_PATH: {vision_config.index_or_path}")
    print(f"  LELAMP_CAMERA_WIDTH: {vision_config.width}")
    print(f"  LELAMP_CAMERA_HEIGHT: {vision_config.height}")
    print(f"  LELAMP_CAMERA_ROTATE_DEG: {vision_config.rotate_deg}")
    print(f"  LELAMP_CAMERA_FLIP: {vision_config.flip}")
    print(f"  LELAMP_CAMERA_CAPTURE_INTERVAL_S: {vision_config.capture_interval_s}")
    print(f"  LELAMP_CAMERA_JPEG_QUALITY: {vision_config.jpeg_quality}")
    print(f"  LELAMP_CAMERA_MAX_AGE_S: {vision_config.max_age_s}")
    print(f"  隐私保护: {vision_config.enable_privacy_protection}")

    # 创建 VisionService（硬件测试时禁用隐私保护以避免等待用户同意）
    from lelamp.service.vision import VisionService

    print("\n创建 VisionService...")
    print("提示: 硬件测试时隐私保护被临时禁用")
    vision_service = VisionService(
        enabled=vision_config.enabled,
        index_or_path=vision_config.index_or_path,
        width=vision_config.width,
        height=vision_config.height,
        rotate_deg=vision_config.rotate_deg,
        flip=vision_config.flip,
        capture_interval_s=vision_config.capture_interval_s,
        jpeg_quality=vision_config.jpeg_quality,
        max_age_s=vision_config.max_age_s,
        enable_privacy_protection=False,  # 硬件测试时禁用
    )

    print("启动 VisionService...")
    vision_service.start()
    time.sleep(2)  # 等待摄像头初始化

    async def async_test():
        """异步测试 VisionService"""
        # 测试激活摄像头
        print("\n测试 activate_camera()...")
        success = await vision_service.activate_camera()
        print(f"activate_camera() 返回: {success}")

        # 测试获取帧
        print("\n测试 get_latest_jpeg_b64()...")
        for i in range(3):
            result = await vision_service.get_latest_jpeg_b64()
            if result:
                jpeg_b64, ts = result
                print(f"  尝试 {i+1}: 成功, b64长度={len(jpeg_b64)}, ts={ts}")
            else:
                print(f"  尝试 {i+1}: 无帧")
            await asyncio.sleep(1)

        # 获取摄像头统计
        print("\n摄像头统计:")
        stats = vision_service.get_camera_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")

    try:
        asyncio.run(async_test())
    finally:
        print("\n停止 VisionService...")
        vision_service.stop()


def main():
    print("=" * 60)
    print("摄像头硬件测试")
    print("=" * 60)
    print(f"平台: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")

    # 1. 测试 OpenCV 导入
    cv2 = test_opencv_import()
    if cv2 is None:
        print("\n错误: OpenCV 不可用，请安装: pip install opencv-python")
        return 1

    # 2. 测试不同摄像头索引
    if platform.system() == "Darwin":
        # macOS 上通常摄像头0被系统占用
        print("\n注意: macOS 系统上，摄像头索引 0 通常被系统占用")
        indices = test_camera_indices(cv2, max_index=3)
    elif platform.system() == "Linux":
        indices = test_camera_indices(cv2, max_index=5)
    else:
        indices = test_camera_indices(cv2, max_index=5)

    # 找出可用的索引
    available = [idx for idx, res in indices.items() if res["status"] == "ok"]
    if available:
        print(f"\n可用的摄像头索引: {available}")

        # 使用第一个可用的索引测试配置
        test_camera_config(cv2, index=available[0])
    else:
        print("\n警告: 没有找到可用的摄像头")

    # 3. 测试 VisionService（如果配置了）
    print("\n" + "=" * 60)
    print("是否测试 VisionService? (需要配置 LELAMP_VISION_ENABLED=true)")
    print("=" * 60)

    config = load_config()
    if config.vision_enabled:
        test_vision_service()
    else:
        print(f"\n跳过: LELAMP_VISION_ENABLED={config.vision_enabled}")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
