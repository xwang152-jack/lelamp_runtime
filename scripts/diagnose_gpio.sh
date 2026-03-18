#!/bin/bash
#
# 诊断GPIO和RGB LED配置问题
#

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}  LeLamp GPIO诊断工具${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

# 1. 检查GPIO设备
echo -e "${YELLOW}📌 检查GPIO设备...${NC}"
GPIO_DEVICES=("/dev/gpiomem" "/dev/mem" "/dev/vcio")

for device in "${GPIO_DEVICES[@]}"; do
    if [ -e "$device" ]; then
        echo -e "  ${GREEN}✓${NC} $device 存在"
        ls -l "$device" 2>/dev/null | awk '{print "    权限: " $1 "  所有者: " $3 "  组: " $4}'
    else
        echo -e "  ${RED}✗${NC} $device 不存在"
    fi
done
echo ""

# 2. 检查用户组
echo -e "${YELLOW}👥 检查用户组权限...${NC}"
CURRENT_USER=${SUDO_USER:-$USER}
echo "  当前用户: $CURRENT_USER"

for group in video gpio; do
    if groups "$CURRENT_USER" 2>/dev/null | grep -q "$group"; then
        echo -e "  ${GREEN}✓${NC} 用户在 $group 组中"
    else
        echo -e "  ${RED}✗${NC} 用户不在 $group 组中"
    fi
done
echo ""

# 3. 检查udev规则
echo -e "${YELLOW}📝 检查udev规则...${NC}"
if [ -f /etc/udev/rules.d/99-lelamp-gpio.rules ]; then
    echo -e "  ${GREEN}✓${NC} LeLamp udev规则存在"
    echo "    内容:"
    cat /etc/udev/rules.d/99-lelamp-gpio.rules | sed 's/^/    /'
elif [ -f /etc/udev/rules.d/99-gpio.rules ]; then
    echo -e "  ${GREEN}✓${NC} 通用GPIO udev规则存在"
else
    echo -e "  ${RED}✗${NC} 没有找到GPIO udev规则"
fi
echo ""

# 4. 测试GPIO访问
echo -e "${YELLOW}🧪 测试GPIO访问...${NC}"
if [ -r /dev/gpiomem ]; then
    echo -e "  ${GREEN}✓${NC} /dev/gpiomem 可读"
else
    echo -e "  ${RED}✗${NC} /dev/gpiomem 不可读"
    echo "  ${YELLOW}建议: 使用sudo启动API服务器${NC}"
fi
echo ""

# 5. 检查Python库
echo -e "${YELLOW}🐍 检查Python库...${NC}"
cd /home/pi/lelamp_runtime 2>/dev/null || cd "$(dirname "$0")/.."

if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo -e "  ${GREEN}✓${NC} 虚拟环境已激活"

    # 检查rpi_ws281x库
    if python -c "import rpi_ws281x" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} rpi_ws281x 库已安装"

        # 测试导入
        python3 << 'EOF'
import sys
try:
    from rpi_ws281x import PixelStrip
    print("  ✓ PixelStrip类可用")

    # 尝试创建配置（不实际初始化）
    try:
        # 测试GPIO权限
        import os
        if os.access('/dev/gpiomem', os.R_OK):
            print("  ✓ GPIO内存可访问")
        else:
            print("  ✗ GPIO内存不可访问（需要sudo）")
    except Exception as e:
        print(f"  ✗ 错误: {e}")
except ImportError as e:
    print(f"  ✗ rpi_ws281x导入失败: {e}")
    sys.exit(1)
EOF
    else
        echo -e "  ${RED}✗${NC} rpi_ws281x 库未安装"
        echo "  ${YELLOW}运行: uv sync --extra hardware${NC}"
    fi
else
    echo -e "  ${RED}✗${NC} 虚拟环境不存在"
fi
echo ""

# 6. 检查API服务状态
echo -e "${YELLOW}🔍 检查API服务状态...${NC}"
if pgrep -f "uvicorn lelamp.api.app" > /dev/null; then
    echo -e "  ${GREEN}✓${NC} API服务正在运行"
    PID=$(pgrep -f "uvicorn lelamp.api.app")
    echo "    PID: $PID"
    ps -p "$PID" -o user,command | tail -1 | awk '{print "    用户: " $1}'
else
    echo -e "  ${RED}✗${NC} API服务未运行"
    echo "  ${YELLOW}启动: sudo bash scripts/start_api_with_gpio.sh${NC}"
fi
echo ""

# 7. 总结和建议
echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}  诊断总结${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

ISSUES_FOUND=0

# 检查问题
if [ ! -e /dev/gpiomem ]; then
    echo -e "${RED}❌ GPIO设备不存在${NC}"
    echo "   → 这不是树莓派，或GPIO内核模块未加载"
    ISSUES_FOUND=$((ISSUES_FOUND + 1))
fi

if [ ! -r /dev/gpiomem ]; then
    echo -e "${RED}❌ GPIO不可读${NC}"
    echo "   → ${YELLOW}解决方案: 使用sudo启动API服务器${NC}"
    echo "   → 命令: sudo bash scripts/start_api_with_gpio.sh"
    ISSUES_FOUND=$((ISSUES_FOUND + 1))
fi

if ! groups "$CURRENT_USER" 2>/dev/null | grep -q "gpio"; then
    echo -e "${YELLOW}⚠️  用户不在gpio组${NC}"
    echo "   → ${YELLOW}解决方案: sudo bash scripts/setup_gpio_permissions.sh${NC}"
    ISSUES_FOUND=$((ISSUES_FOUND + 1))
fi

if [ $ISSUES_FOUND -eq 0 ]; then
    echo -e "${GREEN}✅ 未发现明显问题${NC}"
    echo ""
    echo -e "${BLUE}启动API服务器:${NC}"
    echo "  sudo bash scripts/start_api_with_gpio.sh"
else
    echo -e "${RED}发现 $ISSUES_FOUND 个问题${NC}"
fi

echo ""
echo -e "${BLUE}更多信息:${NC}"
echo "  - GPIO权限配置: scripts/setup_gpio_permissions.sh"
echo "  - 启动服务器:   scripts/start_api_with_gpio.sh"
echo "  - Sudo免密配置: scripts/setup_sudoers.sh"
echo ""
