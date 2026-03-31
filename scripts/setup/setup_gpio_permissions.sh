#!/bin/bash
#
# 配置GPIO权限，让普通用户可以访问WS2812 LED
#
# 注意：即使配置了udev规则，WS2812库通常仍需要root权限
# 因此建议使用 sudo 启动API服务器
#

set -e  # 遇到错误立即退出

echo "========================================="
echo "  LeLamp GPIO权限配置"
echo "========================================="
echo ""

# 检查是否以root运行
if [ "$EUID" -ne 0 ]; then
    echo "❌ 此脚本需要root权限，请使用: sudo $0"
    exit 1
fi

# 获取当前用户（如果使用sudo）
CURRENT_USER=${SUDO_USER:-pi}

echo "当前用户: $CURRENT_USER"
echo ""

# 1. 创建udev规则文件
echo "📝 配置udev规则..."
cat > /etc/udev/rules.d/99-lelamp-gpio.rules <<EOF
# LeLamp GPIO访问规则
# 允许访问GPIO内存（WS2812 LED需要）
SUBSYSTEM=="bcm2835-gpiomem", OWNER="$CURRENT_USER", MODE="0660"
KERNEL=="gpiomem", MODE="0666"
SUBSYSTEM=="gpio*", MODE="0660"
# 允许访问LED设备
SUBSYSTEM=="leds", MODE="0660"
EOF

echo "✓ udev规则已创建: /etc/udev/rules.d/99-lelamp-gpio.rules"

# 2. 重新加载udev规则
echo "🔄 重新加载udev规则..."
udevadm control --reload-rules
udevadm trigger
echo "✓ udev规则已重新加载"

# 3. 将用户添加到必要的组
echo "👥 配置用户组权限..."
usermod -a -G video "$CURRENT_USER" 2>/dev/null || true
usermod -a -G gpio "$CURRENT_USER" 2>/dev/null || true
echo "✓ 用户 $CURRENT_USER 已添加到 video 和 gpio 组"

# 4. 设置LED控制权限（可选）
echo "💡 配置LED亮度控制权限..."
if [ -f /sys/class/leds/led1/brightness ]; then
    chmod 666 /sys/class/leds/led1/brightness 2>/dev/null || true
    echo "✓ LED亮度控制权限已设置"
fi

echo ""
echo "========================================="
echo "  ✅ GPIO权限配置完成！"
echo "========================================="
echo ""
echo "⚠️  重要提示:"
echo "  1. WS2812 LED库通常仍需要root权限"
echo "  2. 建议使用以下命令启动API服务器:"
echo "     sudo bash scripts/start_api_server.sh"
echo ""
echo "  3. 或者免密启动（配置sudoers）:"
echo "     sudo bash scripts/setup_sudoers.sh"
echo ""
echo "📌 请注销并重新登录以使组权限完全生效"
echo "   或运行: newgrp gpio"
echo ""
