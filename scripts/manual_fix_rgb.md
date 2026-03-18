#!/bin/bash
#
# 手动修复RGB控制问题的步骤清单
#

echo "========================================="
echo "  LeLamp RGB控制手动修复指南"
echo "========================================="
echo ""
echo "由于需要sudo权限,请手动执行以下步骤:"
echo ""

echo "步骤 1/4: 配置GPIO权限"
echo "  命令: sudo bash scripts/setup_gpio_permissions.sh"
echo ""

echo "步骤 2/4: 检查虚拟环境"
echo "  命令: ls -la .venv"
echo "  如果不存在: uv sync --extra hardware"
echo ""

echo "步骤 3/4: 停止旧的服务进程"
echo "  命令: sudo pkill -f 'uvicorn lelamp.api.app'"
echo ""

echo "步骤 4/4: 启动API服务器"
echo "  命令: sudo bash scripts/start_api_with_gpio.sh"
echo ""

echo "========================================="
echo "或者,在树莓派上直接运行:"
echo "  sudo bash scripts/fix_rgb_control.sh"
echo "========================================="
