#!/bin/bash
#
# 启动LeLamp API服务器（需要sudo权限用于GPIO控制）
# 这是推荐的启动方式，确保RGB LED控制正常工作
#

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}  LeLamp API服务器启动${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""

# 切换到项目目录
cd /home/pi/lelamp_runtime || {
    echo -e "${RED}❌ 错误: 无法切换到项目目录 /home/pi/lelamp_runtime${NC}"
    exit 1
}

echo -e "${GREEN}✓${NC} 工作目录: $(pwd)"

# 检查虚拟环境
if [ ! -d ".venv" ]; then
    echo -e "${RED}❌ 错误: 虚拟环境 .venv 不存在${NC}"
    echo -e "${YELLOW}请先运行: uv sync${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} 虚拟环境已找到"

# 检查环境变量文件
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠️  警告: .env 文件不存在${NC}"
    echo -e "${YELLOW}请从 .env.example 复制并配置${NC}"
fi

# 激活虚拟环境
source .venv/bin/activate

echo -e "${GREEN}✓${NC} 虚拟环境已激活"
echo ""

# 检查GPIO权限
echo -e "${YELLOW}🔍 检查GPIO访问权限...${NC}"

if [ ! -e /dev/gpiomem ]; then
    echo -e "${RED}❌ 错误: /dev/gpiomem 不存在${NC}"
    echo -e "${YELLOW}GPIO权限未正确配置，RGB LED可能无法工作${NC}"
    echo -e "${YELLOW}请运行: sudo bash scripts/setup_gpio_permissions.sh${NC}"
else
    if [ -r /dev/gpiomem ]; then
        echo -e "${GREEN}✓${NC} /dev/gpiomem 可读"
    else
        echo -e "${YELLOW}⚠️  /dev/gpiomem 不可读（将使用sudo）${NC}"
    fi
fi

echo ""
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}  启动API服务器${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo -e "监听地址: ${GREEN}0.0.0.0:8000${NC}"
echo -e "API文档:   ${GREEN}http://localhost:8000/docs${NC}"
echo -e ""
echo -e "${YELLOW}按 Ctrl+C 停止服务器${NC}"
echo ""

# 使用 sudo 运行 uvicorn
# sudo -E 保留环境变量（特别是 PYTHONPATH 和其他配置）
exec sudo -E uv run uvicorn lelamp.api.app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level info \
    --access-log