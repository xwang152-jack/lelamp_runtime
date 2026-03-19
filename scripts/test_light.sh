#!/bin/bash
# LeLamp 灯光控制测试脚本

PI_HOST="pi@192.168.0.104"

echo "================================================"
echo "LeLamp 灯光控制测试"
echo "================================================"
echo ""

echo "正在连接到树莓派..."
ssh $PI_HOST "cd ~/lelamp_runtime && sudo uv run python -c \'
import time
from lelamp.service.rgb.rgb_service import RGBService

print(\"🔆 开始灯光测试\")

# 创建并启动RGB服务
rgb = RGBService()
rgb.start()

# 测试不同颜色
tests = [
    ((255, 255, 255), \"白色 - 全亮度\", 3),
    ((255, 0, 0), \"红色\", 2),
    ((0, 255, 0), \"绿色\", 2),
    ((0, 0, 255), \"蓝色\", 2),
    ((255, 255, 0), \"黄色\", 2),
    ((255, 0, 255), \"紫色\", 2),
    ((0, 255, 255), \"青色\", 2),
    ((255, 165, 0), \"橙色\", 2),
    ((255, 255, 255), \"白色 - 待机\", 3),
    ((0, 0, 0), \"关闭\", 2),
]

for color, name, duration in tests:
    print(f\"\\n💡 设置{name}灯光\")
    rgb.dispatch(\"solid\", color)
    print(f\"   RGB值: {color}\")
    print(f\"   持续时间: {duration}秒\")
    time.sleep(duration)

# 停止服务
rgb.stop()
print(\"\\n✅ 灯光测试完成！\")
print(\"\\n如果所有灯光都正常亮起，说明台灯硬件工作正常。\")
'"

echo ""
echo "================================================"
echo "测试完成"
echo "================================================"
echo ""
echo "🌐 控制台灯的方法："
echo ""
echo "1. 通过API服务 (推荐):"
echo "   curl http://192.168.0.104:8000/health"
echo ""
echo "2. 直接Python脚本:"
echo "   ssh $PI_HOST 'cd ~/lelamp_runtime && sudo uv run python -c \"from lelamp.service.rgb.rgb_service import RGBService; rgb = RGBService(); rgb.start(); rgb.dispatch(\\\"solid\\\", (255, 255, 255)); import time; time.sleep(3); rgb.dispatch(\\\"solid\\\", (0, 0, 0)); rgb.stop()\"'"
echo ""
echo "3. 主程序模式 (语音交互):"
echo "   ssh $PI_HOST 'cd ~/lelamp_runtime && sudo uv run main.py console'"
echo ""
