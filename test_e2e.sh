#!/bin/bash

# LeLamp 端到端测试脚本
# 测试完整的前后端集成

set -e

echo "================================"
echo "LeLamp 端到端测试"
echo "================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# 步骤计数
STEP=0

print_step() {
    STEP=$((STEP + 1))
    echo ""
    echo -e "${YELLOW}步骤 $STEP: $1${NC}"
    echo "----------------------------------------"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${BLUE}→ $1${NC}"
}

wait_confirm() {
    echo ""
    read -p "按 Enter 继续..."
    echo ""
}

# ============================================
# 步骤 1: 检查环境配置
# ============================================
print_step "检查环境配置"

cd /Users/jackwang/lelamp_runtime

# 检查 .env 文件
if [ ! -f ".env" ]; then
    echo -e "${RED}错误: .env 文件不存在${NC}"
    echo "请创建 .env 文件并配置以下变量："
    echo "  - LIVEKIT_URL"
    echo "  - LIVEKIT_API_KEY"
    echo "  - LIVEKIT_API_SECRET"
    echo "  - DEEPSEEK_API_KEY"
    echo "  - BAIDU_SPEECH_API_KEY"
    echo "  - BAIDU_SPEECH_SECRET_KEY"
    exit 1
fi

print_success ".env 文件存在"

# 检查开发模式配置
if ! grep -q "LELAMP_DEV_MODE" .env; then
    echo ""
    echo -e "${YELLOW}检测到未配置开发模式${NC}"
    echo "正在自动添加开发模式配置..."
    echo "LELAMP_DEV_MODE=1" >> .env
    print_success "已添加 LELAMP_DEV_MODE=1 到 .env"
fi

# 加载环境变量
export $(grep -v '^#' .env | xargs)

# 检查必需的环境变量
REQUIRED_VARS=("LIVEKIT_URL" "LIVEKIT_API_KEY" "LIVEKIT_API_SECRET")
MISSING_VARS=()

for VAR in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!VAR}" ]; then
        MISSING_VARS+=("$VAR")
    fi
done

if [ ${#MISSING_VARS[@]} -ne 0 ]; then
    echo -e "${RED}错误: 缺少必需的环境变量:${NC}"
    echo "  ${MISSING_VARS[*]}"
    exit 1
fi

print_success "环境变量配置正确"

echo ""
echo "当前配置:"
echo "  LIVEKIT_URL: $LIVEKIT_URL"
echo "  LIVEKIT_API_KEY: ${LIVEKIT_API_KEY:0:10}..."
echo "  LELAMP_DEV_MODE: ${LELAMP_DEV_MODE:-未设置}"

wait_confirm

# ============================================
# 步骤 2: 生成客户端 Token
# ============================================
print_step "生成客户端 Token"

print_info "运行 Token 生成脚本..."

# 切换到项目根目录
cd /Users/jackwang/lelamp_runtime

# 生成 Token
TOKEN_OUTPUT=$(uv run python scripts/generate_client_token.py 2>&1)

if [ $? -ne 0 ]; then
    echo -e "${RED}Token 生成失败${NC}"
    echo "$TOKEN_OUTPUT"
    exit 1
fi

# 提取 Room 和 Token
ROOM_NAME=$(echo "$TOKEN_OUTPUT" | grep "Room:" | awk '{print $2}')
USER_NAME=$(echo "$TOKEN_OUTPUT" | grep "User:" | awk '{print $2}')

# 提取 Token（在 "Token:" 和 "---" 之间）
CLIENT_TOKEN=$(echo "$TOKEN_OUTPUT" | sed -n '/^Token:$/,/^----$/{ /^Token:$/d; /^----$/d; p; }' | tr -d '\n')

# 从 .env 获取 LiveKit URL
load_dotenv () {
    if [ -f .env ]; then
        export $(grep -v '^#' .env | xargs)
    fi
}
load_dotenv
LIVEKIT_WS_URL="$LIVEKIT_URL"

if [ -z "$LIVEKIT_WS_URL" ] || [ -z "$CLIENT_TOKEN" ]; then
    echo -e "${RED}无法提取 Token 信息${NC}"
    echo "调试信息:"
    echo "  ROOM_NAME: ${ROOM_NAME:-未设置}"
    echo "  USER_NAME: ${USER_NAME:-未设置}"
    echo "  LIVEKIT_WS_URL: ${LIVEKIT_WS_URL:-未设置}"
    echo "  CLIENT_TOKEN 长度: ${#CLIENT_TOKEN}"
    exit 1
fi

print_success "Token 生成成功"

echo ""
echo "连接信息:"
echo "  Room: $ROOM_NAME"
echo "  User: $USER_NAME"
echo "  LiveKit URL: $LIVEKIT_WS_URL"
echo "  Token: ${CLIENT_TOKEN:0:50}..."

# 保存到文件供后续使用
cat > /tmp/lelamp_connection.txt << EOF
LiveKit URL: $LIVEKIT_WS_URL
Room: $ROOM_NAME
User: $USER_NAME
Token: $CLIENT_TOKEN
EOF

print_success "连接信息已保存到 /tmp/lelamp_connection.txt"

wait_confirm

# ============================================
# 步骤 3: 启动后端服务
# ============================================
print_step "启动后端服务"

echo "检查端口占用..."
if pgrep -f "python.*main.py" > /dev/null; then
    echo -e "${YELLOW}后端服务已在运行${NC}"

    read -p "是否重启后端服务? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pkill -f "python.*main.py"
        sleep 2
    else
        print_success "使用现有后端服务"
    fi
fi

if ! pgrep -f "python.*main.py" > /dev/null; then
    echo "启动后端服务（需要 sudo 权限）..."
    echo "注意: 后端服务将使用摄像头和麦克风"
    echo ""

    # 检查是否有 sudo 权限
    if [ "$EUID" -ne 0 ]; then
        echo "需要 sudo 权限来启动后端服务"
        echo "请输入密码以继续..."
    fi

    # 启动后端服务（后台）
    sudo uv run main.py console > /tmp/lelamp_backend.log 2>&1 &
    BACKEND_PID=$!
    echo $BACKEND_PID > /tmp/lelamp_backend.pid

    # 等待服务启动
    echo "等待后端服务启动..."
    sleep 5

    # 检查服务是否运行
    if pgrep -f "python.*main.py" > /dev/null; then
        print_success "后端服务已启动 (PID: $(pgrep -f 'python.*main.py' | head -1))"
    else
        echo -e "${RED}后端服务启动失败${NC}"
        echo "查看日志: cat /tmp/lelamp_backend.log"
        exit 1
    fi
fi

wait_confirm

# ============================================
# 步骤 4: 启动前端开发服务器
# ============================================
print_step "启动前端开发服务器"

cd /Users/jackwang/lelamp_runtime/web

# 检查端口占用
if lsof -ti:5173 >/dev/null 2>&1; then
    echo -e "${YELLOW}前端服务器已在运行${NC}"

    read -p "是否重启前端服务器? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        lsof -ti:5173 | xargs kill -9 2>/dev/null || true
        sleep 2
    else
        print_success "使用现有前端服务器"
    fi
fi

if ! lsof -ti:5173 >/dev/null 2>&1; then
    echo "启动前端开发服务器..."
    pnpm dev > /tmp/vite_dev.log 2>&1 &
    VITE_PID=$!
    echo $VITE_PID > /tmp/vite_pid.txt

    # 等待服务启动
    echo "等待前端服务启动..."
    sleep 5

    if curl -s http://localhost:5173 > /dev/null; then
        print_success "前端服务器已启动 (PID: $VITE_PID)"
    else
        echo -e "${RED}前端服务器启动失败${NC}"
        echo "查看日志: cat /tmp/vite_dev.log"
        exit 1
    fi
fi

wait_confirm

# ============================================
# 步骤 5: 打开浏览器
# ============================================
print_step "打开浏览器并连接"

if command -v open >/dev/null 2>&1; then
    echo "正在打开浏览器..."
    open http://localhost:5173
    print_success "浏览器已打开"
else
    echo "请手动在浏览器中访问: http://localhost:5173"
fi

echo ""
echo "========================================"
echo -e "${GREEN}连接信息${NC}"
echo "========================================"
echo ""
echo "Server URL:"
echo "  $LIVEKIT_WS_URL"
echo ""
echo "Token:"
echo "  $CLIENT_TOKEN"
echo ""
echo "提示: Token 已复制到剪贴板（如果支持）"
echo ""

# 尝试复制到剪贴板（macOS）
if command -v pbcopy >/dev/null 2>&1; then
    echo "$CLIENT_TOKEN" | pbcopy
    print_success "Token 已复制到剪贴板"
fi

wait_confirm

# ============================================
# 步骤 6: 测试指导
# ============================================
print_step "功能测试指导"

echo "请在浏览器中执行以下步骤："
echo ""

echo "1️⃣  连接设备"
echo "   - Server URL 已自动填充（或手动输入）"
echo "   - 粘贴 Token（已复制到剪贴板）"
echo "   - 点击'连接设备'按钮"
echo "   - 等待连接成功"
echo ""

read -p "是否已成功连接到设备? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "请检查连接信息是否正确"
    echo "查看后端日志: tail -f /tmp/lelamp_backend.log"
    exit 1
fi

echo ""
echo "2️⃣ 检查视频画面"
echo "   - 查看控制台页面中央的视频区域"
echo "   - 应该能看到摄像头画面"
echo "   - 隐私指示器应显示'摄像头已开启'"
echo "   - LED 指示灯应变为红色"
echo ""

read -p "是否能看到摄像头画面? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_success "摄像头画面正常！"
else
    echo ""
    echo "可能的原因:"
    echo "  1. 后端服务未正常启动"
    echo "  2. 摄像头被其他程序占用"
    echo "  3. 权限不足"
    echo ""
    echo "调试步骤:"
    echo "  1. 查看后端日志: tail -f /tmp/lelamp_backend.log"
    echo "  2. 检查摄像头: ls /dev/video*"
    echo "  3. 重启后端服务"
fi

echo ""
echo "3️⃣ 测试语音对话"
echo "   - 点击允许麦克风权限"
echo "   - 说话测试（如：你好）"
echo "   - 观察 LeLamp 是否响应"
echo ""

echo "4️⃣ 测试灯光控制"
echo "   - 点击预设颜色色块"
echo "   - 观察 LeLamp 灯光变化"
echo ""

echo "5️⃣ 测试快捷操作"
echo "   - 点击'👋 打招呼'"
echo "   - 观察 LeLamp 动作反应"
echo ""

# ============================================
# 步骤 7: 查看日志
# ============================================
print_step "查看服务日志"

echo "后端服务日志（最近 20 行）:"
echo "----------------------------------------"
tail -n 20 /tmp/lelamp_backend.log 2>/dev/null || echo "后端日志文件不存在"
echo "----------------------------------------"

echo ""
echo "前端服务日志（最近 20 行）:"
echo "----------------------------------------"
tail -n 20 /tmp/vite_dev.log 2>/dev/null || echo "前端日志文件不存在"
echo "----------------------------------------"

wait_confirm

# ============================================
# 步骤 8: 清理和总结
# ============================================
print_step "测试总结和清理"

echo ""
echo "========================================"
echo "端到端测试完成"
echo "========================================"
echo ""

echo "服务状态:"
if pgrep -f "python.*main.py" > /dev/null; then
    echo "  ✓ 后端服务: 运行中 (PID: $(pgrep -f 'python.*main.py' | head -1))"
else
    echo "  ✗ 后端服务: 未运行"
fi

if lsof -ti:5173 >/dev/null 2>&1; then
    echo "  ✓ 前端服务: 运行中 (PID: $(lsof -ti:5173))"
else
    echo "  ✗ 前端服务: 未运行"
fi

echo ""
read -p "是否停止所有服务? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "停止服务..."

    # 停止后端
    if pgrep -f "python.*main.py" > /dev/null; then
        sudo pkill -f "python.*main.py"
        print_success "后端服务已停止"
    fi

    # 停止前端
    if lsof -ti:5173 >/dev/null 2>&1; then
        kill $(lsof -ti:5173) 2>/dev/null || true
        print_success "前端服务已停止"
    fi

    # 清理 PID 文件
    rm -f /tmp/lelamp_backend.pid /tmp/vite_pid.txt
else
    echo ""
    echo "服务继续运行:"
    if pgrep -f "python.*main.py" > /dev/null; then
        echo "  - 后端: sudo pkill -f 'python.*main.py'"
    fi
    if lsof -ti:5173 >/dev/null 2>&1; then
        echo "  - 前端: kill $(lsof -ti:5173)"
    fi
    echo ""
    echo "查看日志:"
    echo "  - 后端: tail -f /tmp/lelamp_backend.log"
    echo "  - 前端: tail -f /tmp/vite_dev.log"
fi

echo ""
echo "========================================"
echo -e "${GREEN}端到端测试完成!${NC}"
echo "========================================"
echo ""
echo "📝 连接信息已保存: /tmp/lelamp_connection.txt"
echo "📚 相关文档:"
echo "  - 前端测试指南: web/TESTING_GUIDE.md"
echo "  - 后端开发计划: docs/BACKEND_DEVELOPMENT_PLAN.md"
echo ""
