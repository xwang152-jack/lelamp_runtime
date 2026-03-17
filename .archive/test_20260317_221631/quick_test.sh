#!/bin/bash

# LeLamp Phase 3 快速测试脚本
# 这是一个简化版本的测试脚本，用于快速验证功能

set -e

echo "🚀 LeLamp Phase 3 快速测试"
echo "========================="
echo ""

# 颜色
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 步骤 1: 安装依赖
echo -e "${YELLOW}1️⃣ 安装依赖...${NC}"
uv sync --extra api --extra dev --quiet
echo -e "${GREEN}✓ 依赖安装完成${NC}"
echo ""

# 步骤 2: 启动服务器
echo -e "${YELLOW}2️⃣ 启动 API 服务器...${NC}"
pkill -f "uvicorn lelamp.api.app:app" 2>/dev/null || true
uv run uvicorn lelamp.api.app:app --host 0.0.0.0 --port 8000 > /tmp/lelamp_api.log 2>&1 &
API_PID=$!
sleep 3

if curl -s http://localhost:8000/health > /dev/null; then
    echo -e "${GREEN}✓ API 服务器已启动 (PID: $API_PID)${NC}"
else
    echo -e "${RED}✗ 服务器启动失败${NC}"
    cat /tmp/lelamp_api.log
    exit 1
fi
echo ""

# 步骤 3: 运行测试
echo -e "${YELLOW}3️⃣ 运行测试...${NC}"
echo ""

echo "📊 数据库测试..."
uv run pytest lelamp/test/integration/test_database.py -v --tb=line -q 2>&1 | tail -5

echo ""
echo "🌐 API 测试..."
uv run pytest lelamp/test/integration/test_api.py -v --tb=line -q 2>&1 | tail -5

echo ""
echo "🔌 WebSocket 测试..."
uv run pytest lelamp/test/integration/test_websocket.py -v --tb=line -q 2>&1 | tail -5

echo ""
echo "🔄 E2E 测试..."
uv run pytest lelamp/test/integration/test_e2e.py -v --tb=line -q 2>&1 | tail -5

echo ""
echo -e "${GREEN}✓ 所有测试完成${NC}"
echo ""

# 步骤 4: 显示总结
echo -e "${YELLOW}4️⃣ 测试总结${NC}"
TOTAL_TESTS=$(uv run pytest lelamp/test/integration/ --collect-only -q 2>/dev/null | grep -E "test session starts|tests collected" | tail -1 | awk '{print $1}')
echo "总测试数: $TOTAL_TESTS"

# 显示测试状态
PASSED=$(uv run pytest lelamp/test/integration/ -q 2>&1 | grep -oP '\d+(?= passed)' || echo "0")
echo -e "通过: ${GREEN}$PASSED${NC}"

# 步骤 5: 清理
echo ""
read -p "是否停止 API 服务器? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    kill $API_PID 2>/dev/null || true
    echo -e "${GREEN}✓ 服务器已停止${NC}"
else
    echo "服务器继续运行 (PID: $API_PID)"
    echo "查看日志: tail -f /tmp/lelamp_api.log"
    echo "停止服务器: kill $API_PID"
fi

echo ""
echo -e "${GREEN}=========================${NC}"
echo -e "${GREEN}✓ 测试完成！${NC}"
echo -e "${GREEN}=========================${NC}"
echo ""
echo "📄 查看完整报告: cat TEST_RESULTS_PHASE3.md"
echo "📖 查看覆盖率: open htmlcov/index.html"
echo ""
