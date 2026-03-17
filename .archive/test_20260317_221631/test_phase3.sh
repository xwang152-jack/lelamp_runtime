#!/bin/bash

# LeLamp Phase 3 后端功能测试脚本
# 此脚本将指导你完成所有测试步骤

set -e  # 遇到错误立即退出

echo "================================"
echo "LeLamp Phase 3 功能测试指南"
echo "================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 步骤计数
STEP=0

# 打印步骤函数
print_step() {
    STEP=$((STEP + 1))
    echo ""
    echo -e "${YELLOW}步骤 $STEP: $1${NC}"
    echo "----------------------------------------"
}

# 打印成功函数
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# 打印信息函数
print_info() {
    echo -e "${NC}→ $1${NC}"
}

# 等待用户确认
wait_confirm() {
    echo ""
    read -p "按 Enter 继续..."
    echo ""
}

# ============================================
# 步骤 1: 环境准备
# ============================================
print_step "环境准备"

echo "检查 UV 包管理器..."
if ! command -v uv &> /dev/null; then
    echo -e "${RED}错误: UV 未安装${NC}"
    echo "请安装 UV: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
print_success "UV 已安装"

echo ""
echo "检查 Python 版本..."
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo "当前 Python 版本: $PYTHON_VERSION"
print_success "Python 可用"

echo ""
echo "安装 API 和测试依赖..."
uv sync --extra api --extra dev
print_success "依赖安装完成"

wait_confirm

# ============================================
# 步骤 2: 启动 API 服务器
# ============================================
print_step "启动 API 服务器"

echo "检查端口 8000 是否被占用..."
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${YELLOW}警告: 端口 8000 已被占用${NC}"
    read -p "是否关闭现有进程并继续? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "关闭现有进程..."
        lsof -ti:8000 | xargs kill -9 2>/dev/null || true
        print_success "进程已关闭"
    else
        echo "请手动关闭占用端口的进程后重试"
        exit 1
    fi
fi

echo ""
echo "启动 FastAPI 服务器..."
echo "服务器将在后台运行，日志保存在 /tmp/lelamp_api.log"
echo ""

# 启动服务器
uv run uvicorn lelamp.api.app:app --host 0.0.0.0 --port 8000 > /tmp/lelamp_api.log 2>&1 &
API_PID=$!
echo $API_PID > /tmp/lelamp_api.pid

# 等待服务器启动
echo "等待服务器启动..."
sleep 3

# 检查服务器是否启动成功
if kill -0 $API_PID 2>/dev/null; then
    print_success "API 服务器已启动 (PID: $API_PID)"
else
    echo -e "${RED}错误: API 服务器启动失败${NC}"
    echo "查看日志: cat /tmp/lelamp_api.log"
    exit 1
fi

echo ""
echo "测试服务器连接..."
HEALTH_RESPONSE=$(curl -s http://localhost:8000/health)
if echo "$HEALTH_RESPONSE" | grep -q "healthy"; then
    print_success "服务器响应正常"
    echo "响应: $HEALTH_RESPONSE"
else
    echo -e "${RED}错误: 服务器无响应${NC}"
    echo "查看日志: cat /tmp/lelamp_api.log"
    exit 1
fi

echo ""
echo "API 文档地址:"
echo "  - Swagger UI: http://localhost:8000/docs"
echo "  - ReDoc:      http://localhost:8000/redoc"

wait_confirm

# ============================================
# 步骤 3: 运行单元测试
# ============================================
print_step "运行单元测试"

echo "运行数据库单元测试..."
uv run pytest lelamp/test/integration/test_database.py -v --tb=short
print_success "数据库测试通过"

echo ""
echo "运行 API 单元测试..."
uv run pytest lelamp/test/integration/test_api.py -v --tb=short
print_success "API 测试通过"

echo ""
echo "运行 WebSocket 测试..."
uv run pytest lelamp/test/integration/test_websocket.py -v --tb=short
print_success "WebSocket 测试通过"

wait_confirm

# ============================================
# 步骤 4: 运行 E2E 测试
# ============================================
print_step "运行端到端集成测试"

echo "运行完整 E2E 测试套件..."
uv run pytest lelamp/test/integration/test_e2e.py -v --tb=short
print_success "E2E 测试通过"

wait_confirm

# ============================================
# 步骤 5: 运行所有测试并生成覆盖率报告
# ============================================
print_step "生成测试覆盖率报告"

echo "运行所有测试并生成覆盖率报告..."
uv run pytest --cov=lelamp.api --cov=lelamp.database --cov-report=html --cov-report=term

print_success "覆盖率报告已生成"
echo ""
echo "查看 HTML 覆盖率报告:"
echo "  file://$(pwd)/htmlcov/index.html"

if command -v open >/dev/null 2>&1; then
    read -p "是否在浏览器中打开报告? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        open htmlcov/index.html
    fi
fi

wait_confirm

# ============================================
# 步骤 6: 手动 API 测试
# ============================================
print_step "手动 API 测试（使用 curl）"

echo "测试 1: 健康检查"
print_info "curl -s http://localhost:8000/health | jq '.'"
curl -s http://localhost:8000/health | jq '.'
echo ""

wait_confirm

echo "测试 2: 获取设备列表"
print_info "curl -s http://localhost:8000/api/devices | jq '.'"
curl -s http://localhost:8000/api/devices | jq '.'
echo ""

wait_confirm

echo "测试 3: 获取设备状态"
print_info "curl -s http://localhost:8000/api/devices/lelamp_001/state | jq '.'"
curl -s http://localhost:8000/api/devices/lelamp_001/state | jq '.'
echo ""

wait_confirm

echo "测试 4: 发送命令"
print_info "curl -s -X POST http://localhost:8000/api/devices/lelamp_001/command \\"
print_info "  -H 'Content-Type: application/json' \\"
print_info "  -d '{\"type\":\"motor_move\",\"action\":\"move_joint\",\"params\":{\"joint_name\":\"base_yaw\",\"position\":45.0}}' | jq '.'"
curl -s -X POST http://localhost:8000/api/devices/lelamp_001/command \
  -H 'Content-Type: application/json' \
  -d '{"type":"motor_move","action":"move_joint","params":{"joint_name":"base_yaw","position":45.0}}' | jq '.'
echo ""

wait_confirm

echo "测试 5: 获取操作日志"
print_info "curl -s http://localhost:8000/api/devices/lelamp_001/operations | jq '.'"
curl -s http://localhost:8000/api/devices/lelamp_001/operations | jq '.'
echo ""

wait_confirm

# ============================================
# 步骤 7: WebSocket 测试
# ============================================
print_step "WebSocket 连接测试"

echo "创建 WebSocket 测试客户端..."
cat > /tmp/test_ws_client.py << 'EOF'
import asyncio
import websockets
import json

async def test_websocket():
    uri = "ws://localhost:8000/api/ws/lelamp_001"
    print(f"连接到 {uri}...")

    try:
        async with websockets.connect(uri) as websocket:
            print("✓ WebSocket 连接成功")

            # 接收连接确认
            message = await websocket.recv()
            data = json.loads(message)
            print(f"✓ 收到消息: {data['type']}")

            # 发送 ping
            ping_msg = {"type": "ping"}
            await websocket.send(json.dumps(ping_msg))
            print(f"✓ 发送 ping")

            # 接收 pong
            pong_msg = await websocket.recv()
            pong_data = json.loads(pong_msg)
            print(f"✓ 收到 {pong_data['type']}")

            # 订阅频道
            sub_msg = {"type": "subscribe", "channels": ["state", "events"]}
            await websocket.send(json.dumps(sub_msg))
            print(f"✓ 订阅频道: {sub_msg['channels']}")

            # 等待订阅确认
            sub_confirm = await websocket.recv()
            sub_data = json.loads(sub_confirm)
            print(f"✓ 订阅确认: {sub_data['type']}")

            print("\n✓ 所有 WebSocket 测试通过!")

    except Exception as e:
        print(f"✗ WebSocket 测试失败: {e}")
        return False

    return True

if __name__ == "__main__":
    success = asyncio.run(test_websocket())
    exit(0 if success else 1)
EOF

echo "运行 WebSocket 测试..."
uv run python /tmp/test_ws_client.py

if [ $? -eq 0 ]; then
    print_success "WebSocket 测试通过"
else
    echo -e "${RED}WebSocket 测试失败${NC}"
fi

wait_confirm

# ============================================
# 步骤 8: 性能测试（可选）
# ============================================
print_step "性能测试（可选）"

read -p "是否运行性能测试? 这将创建高负载 (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "使用 Apache Bench 进行性能测试..."

    if command -v ab >/dev/null 2>&1; then
        echo "测试 GET /api/devices 端点..."
        ab -n 1000 -c 10 http://localhost:8000/api/devices
        print_success "性能测试完成"
    else
        echo -e "${YELLOW}Apache Bench 未安装，跳过性能测试${NC}"
        echo "安装: sudo apt install apache2-utils"
    fi
else
    echo "跳过性能测试"
fi

wait_confirm

# ============================================
# 步骤 9: 查看日志
# ============================================
print_step "查看服务器日志"

echo "最近 20 行日志:"
echo "----------------------------------------"
tail -n 20 /tmp/lelamp_api.log
echo "----------------------------------------"

wait_confirm

# ============================================
# 步骤 10: 清理和总结
# ============================================
print_step "测试总结和清理"

echo ""
echo "========================================"
echo "测试完成总结"
echo "========================================"
echo ""

echo "✓ 环境准备完成"
echo "✓ API 服务器运行正常"
echo "✓ 单元测试通过"
echo "✓ 集成测试通过"
echo "✓ E2E 测试通过"
echo "✓ WebSocket 测试通过"
echo "✓ 手动 API 测试完成"
echo ""

echo "生成的文件:"
echo "  - htmlcov/index.html  (测试覆盖率报告)"
echo "  - /tmp/lelamp_api.log  (服务器日志)"
echo ""

read -p "是否停止 API 服务器? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "停止 API 服务器..."
    if [ -f /tmp/lelamp_api.pid ]; then
        kill $(cat /tmp/lelamp_api.pid) 2>/dev/null || true
        rm /tmp/lelamp_api.pid
        print_success "API 服务器已停止"
    else
        echo "未找到 PID 文件"
    fi
else
    echo ""
    echo "API 服务器继续运行:"
    echo "  - PID: $API_PID"
    echo "  - 日志: tail -f /tmp/lelamp_api.log"
    echo "  - 停止: kill $API_PID"
fi

echo ""
echo "========================================"
echo -e "${GREEN}所有测试完成!${NC}"
echo "========================================"
echo ""
echo "下一步建议:"
echo "  1. 查看测试覆盖率报告"
echo "  2. 检查 API 文档: http://localhost:8000/docs"
echo "  3. 阅读 API_DOCUMENTATION.md 了解完整 API"
echo "  4. 阅读 DEPLOYMENT_GUIDE.md 了解部署"
echo ""
