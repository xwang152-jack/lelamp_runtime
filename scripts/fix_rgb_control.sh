#!/bin/bash
#
# 一键修复RGB灯光控制问题
#

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}  LeLamp RGB灯光控制一键修复${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

# 检查是否在正确的目录
if [ ! -f "lelamp/api/app.py" ]; then
    echo -e "${RED}❌ 错误: 请在 lelamp_runtime 目录下运行此脚本${NC}"
    exit 1
fi

echo -e "${YELLOW}步骤 1/4: 配置GPIO权限...${NC}"
if sudo bash scripts/setup_gpio_permissions.sh; then
    echo -e "${GREEN}✓ GPIO权限配置完成${NC}"
else
    echo -e "${RED}✗ GPIO权限配置失败${NC}"
fi
echo ""

echo -e "${YELLOW}步骤 2/4: 检查虚拟环境...${NC}"
if [ ! -d ".venv" ]; then
    echo "虚拟环境不存在,正在安装依赖..."
    if uv sync --extra hardware; then
        echo -e "${GREEN}✓ 依赖安装成功${NC}"
    else
        echo -e "${RED}✗ 依赖安装失败${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ 虚拟环境已存在${NC}"
fi
echo ""

echo -e "${YELLOW}步骤 3/4: 停止旧的服务进程...${NC}"
sudo pkill -f "uvicorn lelamp.api.app" 2>/dev/null || true
echo -e "${GREEN}✓ 旧进程已停止${NC}"
echo ""

echo -e "${YELLOW}步骤 4/4: 启动API服务器...${NC}"
echo ""
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}  API服务器启动中...${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo -e "提示: 按 Ctrl+C 停止服务器"
echo ""

# 启动服务器
sudo bash scripts/start_api_with_gpio.sh
