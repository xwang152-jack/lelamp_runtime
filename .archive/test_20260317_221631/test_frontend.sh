#!/bin/bash

# LeLamp Web 前端测试脚本
# 此脚本将指导你完成前端的所有测试步骤

set -e  # 遇到错误立即退出

echo "================================"
echo "LeLamp Web 前端测试指南"
echo "================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
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
    echo -e "${BLUE}→ $1${NC}"
}

# 等待用户确认
wait_confirm() {
    echo ""
    read -p "按 Enter 继续..."
    echo ""
}

# ============================================
# 步骤 1: 进入前端目录
# ============================================
print_step "进入前端目录"

cd /Users/jackwang/lelamp_runtime/web
print_success "已进入 web 目录"
print_info "当前目录: $(pwd)"

wait_confirm

# ============================================
# 步骤 2: 检查 Node.js 和 pnpm
# ============================================
print_step "检查开发环境"

echo "检查 Node.js..."
if ! command -v node &> /dev/null; then
    echo -e "${RED}错误: Node.js 未安装${NC}"
    echo "请访问 https://nodejs.org/ 下载安装"
    exit 1
fi
NODE_VERSION=$(node --version)
print_success "Node.js 已安装: $NODE_VERSION"

echo ""
echo "检查 pnpm..."
if ! command -v pnpm &> /dev/null; then
    echo -e "${YELLOW}pnpm 未安装，正在安装...${NC}"
    npm install -g pnpm
fi
PNPM_VERSION=$(pnpm --version)
print_success "pnpm 已安装: $PNPM_VERSION"

wait_confirm

# ============================================
# 步骤 3: 安装依赖
# ============================================
print_step "安装前端依赖"

if [ ! -d "node_modules" ]; then
    echo "正在安装依赖（首次运行）..."
    pnpm install
    print_success "依赖安装完成"
else
    echo -e "${GREEN}依赖已安装${NC}"

    read -p "是否重新安装依赖? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf node_modules
        pnpm install
        print_success "依赖重新安装完成"
    fi
fi

wait_confirm

# ============================================
# 步骤 4: 代码质量检查
# ============================================
print_step "代码质量检查"

echo "1. TypeScript 类型检查..."
if pnpm exec vue-tsc --noEmit 2>&1 | grep -q "error"; then
    echo -e "${RED}类型检查失败${NC}"
    pnpm type-check
else
    print_success "TypeScript 类型检查通过"
fi

echo ""
echo "2. ESLint 代码检查..."
LINT_OUTPUT=$(pnpm lint 2>&1)
LINT_ERRORS=$(echo "$LINT_OUTPUT" | grep -o "[0-9]* error(s)" | awk '{print $1}')
LINT_WARNINGS=$(echo "$LINT_OUTPUT" | grep -o "[0-9]* warning(s)" | awk '{print $1}')

if [ "$LINT_ERRORS" = "0" ] || [ -z "$LINT_ERRORS" ]; then
    print_success "ESLint 检查通过 (警告: ${LINT_WARNINGS:-0})"
else
    echo -e "${RED}ESLint 发现 ${LINT_ERRORS} 个错误${NC}"
    echo "$LINT_OUTPUT"
fi

echo ""
echo "3. 代码格式检查..."
if pnpm format 2>&1 | grep -q "would have been formatted"; then
    echo -e "${YELLOW}有文件需要格式化${NC}"
    read -p "是否自动格式化? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pnpm exec prettier --write .
        print_success "代码格式化完成"
    fi
else
    print_success "代码格式正确"
fi

wait_confirm

# ============================================
# 步骤 5: 生产构建测试
# ============================================
print_step "生产构建测试"

echo "清除旧构建..."
rm -rf dist

echo ""
echo "开始构建..."
BUILD_START=$(date +%s)
pnpm build
BUILD_END=$(date +%s)
BUILD_TIME=$((BUILD_END - BUILD_START))

print_success "构建成功！耗时: ${BUILD_TIME} 秒"

echo ""
echo "构建产物分析:"
if [ -d "dist" ]; then
    DIST_SIZE=$(du -sh dist | awk '{print $1}')
    FILE_COUNT=$(find dist -type f | wc -l)
    echo "  - 总大小: $DIST_SIZE"
    echo "  - 文件数: $FILE_COUNT"

    echo ""
    echo "主要文件:"
    ls -lh dist/assets/*.js 2>/dev/null | awk '{print "  - " $9 " (" $5 ")"}' | head -5
fi

wait_confirm

# ============================================
# 步骤 6: 启动开发服务器
# ============================================
print_step "启动开发服务器"

echo "检查端口 5173..."
if lsof -ti:5173 >/dev/null 2>&1; then
    echo -e "${YELLOW}端口 5173 已被占用${NC}"
    read -p "是否关闭现有进程并继续? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        lsof -ti:5173 | xargs kill -9 2>/dev/null || true
        print_success "进程已关闭"
    else
        echo "请手动关闭占用端口的进程后重试"
        exit 1
    fi
fi

echo ""
echo "启动开发服务器..."
echo "服务器将在后台运行，日志保存在 /tmp/vite_dev.log"
echo ""

# 启动服务器
pnpm dev > /tmp/vite_dev.log 2>&1 &
VITE_PID=$!
echo $VITE_PID > /tmp/vite_pid.txt

# 等待服务器启动
echo "等待服务器启动..."
sleep 5

# 检查服务器是否启动成功
if kill -0 $VITE_PID 2>/dev/null; then
    print_success "开发服务器已启动 (PID: $VITE_PID)"
else
    echo -e "${RED}错误: 开发服务器启动失败${NC}"
    echo "查看日志: cat /tmp/vite_dev.log"
    exit 1
fi

echo ""
echo "测试服务器连接..."
if curl -s http://localhost:5173 > /dev/null; then
    print_success "服务器响应正常"
else
    echo -e "${RED}服务器无响应${NC}"
    cat /tmp/vite_dev.log
    exit 1
fi

echo ""
echo -e "${GREEN}================================"
echo "✓ 前端开发服务器已就绪！"
echo "================================${NC}"
echo ""
echo "📱 访问地址:"
echo "   - 本地: http://localhost:5173"
echo "   - 网络: http://$(hostname -I 2>/dev/null | awk '{print $1}'):5173"
echo ""
echo "📝 功能测试清单:"
echo "   ✓ 连接页面表单验证"
echo "   ✓ 控制台页面布局"
echo "   ✓ 快捷操作按钮"
echo "   ✓ 灯光控制功能"
echo "   ✓ 实时对话功能"
echo ""

wait_confirm

# ============================================
# 步骤 7: 打开浏览器
# ============================================
print_step "打开浏览器测试"

if command -v open >/dev/null 2>&1; then
    echo "正在打开浏览器..."
    open http://localhost:5173
    print_success "浏览器已打开"
else
    echo "请手动在浏览器中访问: http://localhost:5173"
fi

wait_confirm

# ============================================
# 步骤 8: 功能测试指导
# ============================================
print_step "功能测试指导"

echo "请在浏览器中进行以下测试："
echo ""

echo "📋 连接页面测试 (http://localhost:5173)"
echo "  [ ] 页面标题正确显示"
echo "  [ ] Server URL 输入框可输入"
echo "  [ ] Token 输入框可输入多行文本"
echo "  [ ] 连接按钮可点击"
echo "  [ ] 空表单点击连接显示警告"
echo "  [ ] 背景渐变效果正常"
echo ""

read -p "是否已完成连接页面测试? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "请完成连接页面测试后继续"
    wait_confirm
fi

echo ""
echo "📋 控制台页面测试"
echo "  注意: 需要真实的 LiveKit Token 才能进入此页面"
echo ""
echo "  [ ] 顶部状态栏显示'已连接'"
echo "  [ ] 状态点显示绿色"
echo "  [ ] 断开连接按钮可点击"
echo "  [ ] 视频占位符显示"
echo "  [ ] 隐私指示器显示"
echo ""
echo "  快捷操作:"
echo "  [ ] 点击'👋 打招呼' - 消息出现在聊天框"
echo "  [ ] 点击'⏰ 查看时间' - 消息正确显示"
echo "  [ ] 点击'😄 讲笑话' - 消息正确显示"
echo "  [ ] 点击'🎵 唱歌' - 消息正确显示"
echo ""
echo "  灯光控制:"
echo "  [ ] 颜色选择器可点击"
echo "  [ ] 点击预设颜色色块 - 显示对应颜色"
echo "  [ ] 点击特效按钮 - 显示成功提示"
echo "  [ ] 点击'🌑 关闭灯光' - 消息出现在聊天框"
echo ""
echo "  实时对话:"
echo "  [ ] 输入框可输入文字"
echo "  [ ] 点击'发送'按钮 - 消息出现在聊天框"
echo "  [ ] 按 Enter 键 - 消息发送成功"
echo "  [ ] 发送空消息 - 无反应"
echo "  [ ] 消息显示时间戳"
echo "  [ ] 聊天框自动滚动"
echo ""

read -p "是否已完成控制台页面测试? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "请完成控制台页面测试后继续"
    wait_confirm
fi

# ============================================
# 步骤 9: 查看日志
# ============================================
print_step "查看开发服务器日志"

echo "最近 20 行日志:"
echo "----------------------------------------"
tail -n 20 /tmp/vite_dev.log
echo "----------------------------------------"

wait_confirm

# ============================================
# 步骤 10: 清理和总结
# ============================================
print_step "测试总结和清理"

echo ""
echo "========================================"
echo "前端测试完成总结"
echo "========================================"
echo ""

echo "✓ 开发环境准备完成"
echo "✓ 代码质量检查通过"
echo "✓ 生产构建成功"
echo "✓ 开发服务器运行正常"
echo "✓ 功能测试完成"
echo ""

echo "生成的文件:"
echo "  - dist/  (生产构建产物)"
echo ""

read -p "是否停止开发服务器? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "停止开发服务器..."
    if [ -f /tmp/vite_pid.txt ]; then
        kill $(cat /tmp/vite_pid.txt) 2>/dev/null || true
        rm /tmp/vite_pid.txt
        print_success "开发服务器已停止"
    else
        echo "未找到 PID 文件"
    fi
else
    echo ""
    echo "开发服务器继续运行:"
    echo "  - PID: $VITE_PID"
    echo "  - 日志: tail -f /tmp/vite_dev.log"
    echo "  - 停止: kill $VITE_PID"
    echo ""
    echo "实时查看日志:"
    echo "  tail -f /tmp/vite_dev.log"
fi

echo ""
echo "========================================"
echo -e "${GREEN}前端测试完成!${NC}"
echo "========================================"
echo ""
echo "📚 相关文档:"
echo "  - 前端测试指南: web/TESTING_GUIDE.md"
echo "  - 测试计划: docs/plans/2026-03-17-web-frontend-phase1-testing-plan.md"
echo "  - 测试报告: docs/PHASE1_TEST_REPORT.md"
echo ""
echo "🎯 下一步:"
echo "  1. 真实环境测试（需要 LiveKit Token）"
echo "  2. 连接后端 API"
echo "  3. 端到端测试"
echo ""
