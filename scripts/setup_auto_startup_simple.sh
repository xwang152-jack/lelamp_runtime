#!/bin/bash
# LeLamp 简化版自动启动设置（仅后端 + 静态前端）

set -e

PI_HOST="pi@192.168.0.104"
PROJECT_DIR="~/lelamp_runtime"

echo "================================================"
echo "LeLamp 自动启动设置（简化版）"
echo "================================================"
echo ""

# 检查连接
echo "🔍 检查树莓派连接..."
if ! ssh $PI_HOST "echo '连接成功'" 2>/dev/null; then
    echo "❌ 无法连接到树莓派"
    exit 1
fi
echo "✅ 树莓派连接正常"
echo ""

# 1. 构建前端
echo "🏗️  1. 构建前端静态文件..."
echo "   这将在本地构建前端，然后部署到树莓派"
echo ""

# 检查前端是否已构建
if [ ! -d "web/dist" ]; then
    echo "   正在构建前端..."
    cd web
    npm run build
    cd ..
    echo "   ✅ 前端构建完成"
else
    echo "   ℹ️  前端已构建，跳过"
fi

echo ""

# 2. 复制前端文件到树莓派
echo "📦 2. 部署前端文件到树莓派..."
ssh $PI_HOST "mkdir -p $PROJECT_DIR/web/dist"
scp -r web/dist/* $PI_HOST:$PROJECT_DIR/web/dist/
echo "✅ 前端文件已部署"

echo ""

# 3. 创建后端服务（包含静态文件服务）
echo "📝 3. 创建 systemd 服务..."
ssh $PI_HOST "sudo tee /etc/systemd/system/lelamp-api.service > /dev/null << 'EOF'
[Unit]
Description=LeLamp API Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$PROJECT_DIR
Environment=PATH=/usr/local/bin:/usr/bin:/bin
EnvironmentFile=$PROJECT_DIR/.env
ExecStart=/usr/local/bin/uv run python -m uvicorn lelamp.api.app:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
"

echo "✅ 服务文件已创建"

# 4. 清理旧服务
echo "🔧 4. 清理旧服务..."
ssh $PI_HOST "sudo systemctl stop lelamp-api.service 2>/dev/null || true"
ssh $PI_HOST "sudo systemctl disable lelamp-api.service 2>/dev/null || true"

# 5. 重新加载并启用
echo "🔄 5. 配置自动启动..."
ssh $PI_HOST "sudo systemctl daemon-reload"
ssh $PI_HOST "sudo systemctl enable lelamp-api.service"
echo "✅ 服务已设置为开机自启"

# 6. 启动服务
echo "▶️  6. 启动服务..."
ssh $PI_HOST "sudo systemctl start lelamp-api.service"
echo "✅ 服务已启动"

echo ""
echo "================================================"
echo "✅ 自动启动设置完成！"
echo "================================================"
echo ""
echo "🌐 访问地址："
echo "   前端: http://192.168.0.104:8000/index.html"
echo "   API:  http://192.168.0.104:8000/api"
echo "   文档: http://192.168.0.104:8000/docs"
echo ""
echo "🎯 管理命令："
echo "   状态: ssh $PI_HOST 'sudo systemctl status lelamp-api.service'"
echo "   重启: ssh $PI_HOST 'sudo systemctl restart lelamp-api.service'"
echo "   日志: ssh $PI_HOST 'sudo journalctl -u lelamp-api.service -f'"
echo ""
