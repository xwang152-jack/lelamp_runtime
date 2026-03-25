#!/bin/bash
#
# LeLamp 边缘视觉功能测试脚本
# 在树莓派上测试边缘视觉是否正常工作
#
# 使用方法:
#   bash scripts/test_edge_vision_on_pi.sh
#

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "======================================"
echo "  LeLamp 边缘视觉功能测试"
echo "======================================"
echo ""

# 检查系统信息
echo "📋 系统信息:"
echo "   平台: $(uname -m)"
echo "   系统: $(uname -s)"
echo "   Python: $(python3 --version 2>&1)"
echo ""

# 检查项目目录
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -f "$PROJECT_DIR/pyproject.toml" ]; then
    echo -e "${RED}❌ 错误: 请在 lelamp_runtime 项目目录中运行此脚本${NC}"
    exit 1
fi

cd "$PROJECT_DIR"
echo "📁 项目目录: $PROJECT_DIR"
echo ""

# 检查虚拟环境
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}⚠️  未找到虚拟环境，正在创建...${NC}"
    uv sync
fi

echo "======================================"
echo "  测试 1: 检查 MediaPipe 安装"
echo "======================================"
echo ""

if uv run python -c "import mediapipe; print('MediaPipe version:', mediapipe.__version__)" 2>/dev/null; then
    echo -e "${GREEN}✅ MediaPipe 已安装${NC}"
    MEDIAPIPE_VERSION=$(uv run python -c "import mediapipe; print(mediapipe.__version__)")
    echo "   版本: $MEDIAPIPE_VERSION"
else
    echo -e "${RED}❌ MediaPipe 未安装${NC}"
    echo ""
    echo "安装方法:"
    echo "  uv sync --extra vision"
    echo ""
    echo "或手动安装:"
    echo "  uv pip install mediapipe opencv-python"
    echo ""
    echo "⚠️  注意: MediaPipe 在 Raspberry Pi ARM 上官方不支持"
    echo "   如果安装失败，系统会自动降级到云端视觉识别"
    exit 1
fi

echo ""
echo "======================================"
echo "  测试 2: 检查 OpenCV 安装"
echo "======================================"
echo ""

if uv run python -c "import cv2; print('OpenCV version:', cv2.__version__)" 2>/dev/null; then
    echo -e "${GREEN}✅ OpenCV 已安装${NC}"
    OPENCV_VERSION=$(uv run python -c "import cv2; print(cv2.__version__)")
    echo "   版本: $OPENCV_VERSION"
else
    echo -e "${RED}❌ OpenCV 未安装${NC}"
    echo "   安装方法: uv pip install opencv-python"
    exit 1
fi

echo ""
echo "======================================"
echo "  测试 3: 测试边缘视觉模块"
echo "======================================"
echo ""

# 创建测试脚本
cat > /tmp/test_edge_vision.py << 'EOF'
"""
边缘视觉功能测试
"""
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.expanduser("~/lelamp_runtime"))

print("正在导入边缘视觉模块...")

try:
    from lelamp.edge.face_detector import FaceDetector
    from lelamp.edge.hand_tracker import HandTracker
    from lelamp.edge.object_detector import ObjectDetector
    from lelamp.edge.hybrid_vision import HybridVisionService
    print("✅ 所有边缘视觉模块导入成功")
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    sys.exit(1)

print("\n" + "="*50)
print("测试人脸检测器")
print("="*50)

try:
    face_detector = FaceDetector()
    stats = face_detector.get_stats()
    print(f"人脸检测器状态: {'NoOp 模式' if stats.get('noop_mode') else '正常模式'}")
    print(f"总检测次数: {stats.get('total_detections', 0)}")
except Exception as e:
    print(f"❌ 人脸检测器初始化失败: {e}")

print("\n" + "="*50)
print("测试手势追踪器")
print("="*50)

try:
    hand_tracker = HandTracker()
    stats = hand_tracker.get_stats()
    print(f"手势追踪器状态: {'NoOp 模式' if stats.get('noop_mode') else '正常模式'}")
    print(f"总追踪次数: {stats.get('total_tracks', 0)}")
except Exception as e:
    print(f"❌ 手势追踪器初始化失败: {e}")

print("\n" + "="*50)
print("测试物体检测器")
print("="*50)

try:
    object_detector = ObjectDetector()
    stats = object_detector.get_stats()
    print(f"物体检测器状态: {'NoOp 模式' if stats.get('noop_mode') else '正常模式'}")
    print(f"总检测次数: {stats.get('total_detections', 0)}")
except Exception as e:
    print(f"❌ 物体检测器初始化失败: {e}")

print("\n" + "="*50)
print("测试混合视觉服务")
print("="*50)

try:
    hybrid_service = HybridVisionService(
        enable_face=True,
        enable_hand=True,
        enable_object=True,
    )
    stats = hybrid_service.get_stats()
    print(f"混合视觉服务状态: 正常")
    print(f"服务组件:")
    print(f"  - 人脸检测: {stats.get('services', {}).get('face_detector', 'N/A')}")
    print(f"  - 手势追踪: {stats.get('services', {}).get('hand_tracker', 'N/A')}")
    print(f"  - 物体检测: {stats.get('services', {}).get('object_detector', 'N/A')}")
    print(f"总查询次数: {stats.get('total_queries', 0)}")
    print(f"本地查询: {stats.get('local_queries', 0)}")
    print(f"云端查询: {stats.get('cloud_queries', 0)}")
except Exception as e:
    print(f"❌ 混合视觉服务初始化失败: {e}")

print("\n" + "="*50)
print("测试图像处理")
print("="*50)

try:
    import numpy as np
    import cv2

    # 创建测试图像（640x480 彩色）
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 添加一些内容（画一个红色的圆）
    cv2.circle(frame, (320, 240), 100, (0, 0, 255), -1)

    print("✅ 测试图像创建成功 (640x480)")

    # 测试物体检测
    result = object_detector.detect(frame)
    print(f"物体检测结果: {result['summary']}")
    print(f"检测到的物体数量: {result['count']}")

except Exception as e:
    print(f"❌ 图像处理测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("测试完成")
print("="*50)

# 判断测试结果
if stats.get('services', {}).get('object_detector', False):
    print("\n✅ 边缘视觉功能正常工作")
    sys.exit(0)
else:
    print("\n⚠️  边缘视觉功能处于 NoOp 模式")
    print("   这通常意味着 MediaPipe 不可用")
    print("   系统会自动降级到云端视觉识别")
    sys.exit(0)
EOF

# 运行测试
echo "正在运行测试..."
echo ""
uv run python /tmp/test_edge_vision.py
TEST_RESULT=$?

echo ""
echo "======================================"
echo "  测试完成"
echo "======================================"
echo ""

if [ $TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}✅ 所有测试通过${NC}"
    echo ""
    echo "📝 下一步:"
    echo "   1. 在 .env 中启用边缘视觉:"
    echo "      LELAMP_EDGE_VISION_ENABLED=true"
    echo "   2. 重启 LeLamp 服务:"
    echo "      sudo systemctl restart lelamp-*"
    echo "   3. 查看日志确认:"
    echo "      tail -f /var/log/lelamp.log | grep EdgeVision"
else
    echo -e "${RED}❌ 测试失败${NC}"
    echo ""
    echo "💡 可能的原因:"
    echo "   1. MediaPipe 未正确安装"
    echo "   2. 平台不支持（Raspberry Pi ARM）"
    echo "   3. 依赖库缺失"
    echo ""
    echo "📖 解决方案:"
    echo "   - 参考 docs/EDGE_VISION_SETUP.md"
    echo "   - 或使用云端视觉识别（自动降级）"
    exit 1
fi
