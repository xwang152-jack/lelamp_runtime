#!/bin/bash
# LeLamp 自动启动设置脚本

set -e

PI_HOST="pi@192.168.0.104"
PROJECT_DIR="/home/pi/lelamp_runtime"
FRONTEND_DIR="/home/pi/lelamp_runtime/web"

echo "================================================"
echo "LeLamp 自动启动设置"
echo "================================================"
echo ""

# 检查是否可以连接到树莓派
echo "🔍 检查树莓派连接..."
if ! ssh $PI_HOST "echo '连接成功'" 2>/dev/null; then
    echo "❌ 无法连接到树莓派 $PI_HOST"
    echo "请检查："
    echo "  1. 树莓派是否开机"
    echo "  2. 网络连接是否正常"
    echo "  3. SSH 密钥是否配置"
    exit 1
fi

echo "✅ 树莓派连接正常"
echo ""

# 1. 创建后端 systemd 服务
echo "📝 1. 创建后端 systemd 服务..."
ssh $PI_HOST "sudo tee /etc/systemd/system/lelamp-api.service > /dev/null << 'EOF'
[Unit]
Description=LeLamp API Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/home/pi/lelamp_runtime
Environment=PATH=/usr/local/bin:/usr/bin:/bin
EnvironmentFile=/home/pi/lelamp_runtime/.env
ExecStart=/usr/local/bin/uv run python -m uvicorn lelamp.api.app:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
"

echo "✅ 后端服务文件已创建"

# 2. 清理旧的前端服务（前后端已分离）
echo "🔧 2. 清理旧的前端服务..."
ssh $PI_HOST "sudo systemctl stop lelamp-frontend.service 2>/dev/null || true"
ssh $PI_HOST "sudo systemctl disable lelamp-frontend.service 2>/dev/null || true"

# 3. 停止并禁用旧的 lelamp-api 服务（如果存在）
echo "🔧 3. 清理旧服务..."
ssh $PI_HOST "sudo systemctl stop lelamp-api.service 2>/dev/null || true"
ssh $PI_HOST "sudo systemctl disable lelamp-api.service 2>/dev/null || true"

# 4. 重新加载 systemd
echo "🔄 4. 重新加载 systemd..."
ssh $PI_HOST "sudo systemctl daemon-reload"
echo "✅ systemd 已重新加载"

# 5. 启用服务
echo "🚀 5. 启用开机自启..."
ssh $PI_HOST "sudo systemctl enable lelamp-api.service"
echo "✅ 服务已设置为开机自启"

# 6. 启动服务
echo "▶️  6. 启动服务..."
ssh $PI_HOST "sudo systemctl start lelamp-api.service"
echo "✅ 服务已启动"

# 7. 检查服务状态
echo ""
echo "================================================"
echo "🔍 服务状态检查"
echo "================================================"
echo ""

echo "📊 后端服务状态:"
ssh $PI_HOST "sudo systemctl status lelamp-api.service --no-pager -l" | head -15
echo ""

# 8. 测试连接
echo "================================================"
echo "🧪 连接测试"
echo "================================================"
echo ""

echo "测试后端 API..."
sleep 2
if ssh $PI_HOST "curl -s http://localhost:8000/health > /dev/null"; then
    echo "✅ 后端 API 响应正常"
    echo "   地址: http://192.168.0.104:8000"
    echo "   文档: http://192.168.0.104:8000/docs"
else
    echo "❌ 后端 API 无响应，请检查日志"
fi

echo ""
echo ""
echo "================================================"
echo "✅ 自动启动设置完成！"
echo "================================================"
echo ""
echo "🎯 管理命令："
echo ""
echo "查看服务状态:"
echo "  ssh $PI_HOST 'sudo systemctl status lelamp-api.service'"
echo ""
echo "重启服务:"
echo "  ssh $PI_HOST 'sudo systemctl restart lelamp-api.service'"
echo ""
echo "查看日志:"
echo "  ssh $PI_HOST 'sudo journalctl -u lelamp-api.service -f'"
echo ""
echo "停止服务:"
echo "  ssh $PI_HOST 'sudo systemctl stop lelamp-api.service'"
echo ""
echo "禁用开机自启:"
echo "  ssh $PI_HOST 'sudo systemctl disable lelamp-api.service'"
echo ""
