#!/bin/bash

# LeLamp 快速启动脚本 - 简化版
# 快速启动后端和前端，生成 Token

set -e

echo "================================"
echo "LeLamp 快速启动"
echo "================================"
echo ""

# 颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 1. 生成 Token
echo -e "${YELLOW}1️⃣ 生成 Token...${NC}"
cd /Users/jackwang/lelamp_runtime

TOKEN_OUTPUT=$(uv run python scripts/generate_client_token.py 2>&1)
ROOM=$(echo "$TOKEN_OUTPUT" | grep "Room:" | awk '{print $2}')
TOKEN=$(echo "$TOKEN_OUTPUT" | sed -n '/^Token:$/,/^----$/{ /^Token:$/d; /^----$/d; p; }' | tr -d '\n')

# 加载 .env
export $(grep -v '^#' .env | xargs)
URL="$LIVEKIT_URL"

echo -e "${GREEN}✓ Token 已生成${NC}"
echo ""
echo "连接信息:"
echo "  URL: $URL"
echo "  Room: $ROOM"
echo "  Token: ${TOKEN:0:80}..."
echo ""

# 2. 保存到文件
cat > /tmp/lelamp_quick_connect.txt << EOF
Server URL: $URL
Token: $TOKEN
EOF

echo -e "${BLUE}连接信息已保存到: /tmp/lelamp_quick_connect.txt${NC}"
echo ""

# 3. 复制 Token 到剪贴板
if command -v pbcopy >/dev/null 2>&1; then
    echo "$TOKEN" | pbcopy
    echo -e "${GREEN}✓ Token 已复制到剪贴板${NC}"
fi

echo ""
echo -e "${YELLOW}================================${NC}"
echo -e "${GREEN}✓ 准备完成！${NC}"
echo -e "${YELLOW}================================${NC}"
echo ""
echo "下一步:"
echo ""
echo "1️⃣  启动后端服务:"
echo "   cd /Users/jackwang/lelamp_runtime"
echo "   sudo uv run main.py console"
echo ""
echo "2️⃣  启动前端服务（新终端）:"
echo "   cd /Users/jackwang/lelamp_runtime/web"
echo "   pnpm dev"
echo ""
echo "3️⃣  连接设备:"
echo "   - 访问: http://localhost:5173"
echo "   - 粘贴 Token（Cmd+V）"
echo "   - 点击连接"
echo ""
