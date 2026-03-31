#!/bin/bash
# LeLamp 完整服务设置脚本 - LiveKit + API + Frontend

set -e

PI_HOST="${PI_HOST:-pi@192.168.0.104}"
PROJECT_DIR="/home/pi/lelamp_runtime"

echo "================================================"
echo "LeLamp 完整服务设置"
echo "================================================"
echo ""
echo "将设置以下服务："
echo "  1. LiveKit 服务（语音交互）"
echo "  2. API 服务（后端）"
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

# 检查 tmux 是否安装（LiveKit 服务需要）
echo "🔍 检查 tmux 是否安装..."
if ! ssh $PI_HOST "which tmux" 2>/dev/null; then
    echo "📦 正在安装 tmux..."
    ssh $PI_HOST "sudo apt-get update && sudo apt-get install -y tmux"
else
    echo "✅ tmux 已安装"
fi
echo ""

# 停止所有旧服务
echo "🔧 1. 停止并清理旧服务..."
ssh $PI_HOST "sudo systemctl stop lelamp-livekit.service 2>/dev/null || true"
ssh $PI_HOST "sudo systemctl stop lelamp-api.service 2>/dev/null || true"
ssh $PI_HOST "sudo systemctl stop lelamp-frontend.service 2>/dev/null || true"
echo "   注意: lelamp-frontend.service 已废弃（前后端已分离）"
ssh $PI_HOST "sudo systemctl disable lelamp-livekit.service 2>/dev/null || true"
ssh $PI_HOST "sudo systemctl disable lelamp-api.service 2>/dev/null || true"
ssh $PI_HOST "sudo systemctl disable lelamp-frontend.service 2>/dev/null || true"
echo "✅ 旧服务已清理"
echo ""

# 创建 LiveKit 服务
echo "📝 2. 创建 LiveKit 服务..."
cat scripts/services/lelamp-livekit.service | ssh $PI_HOST "sudo tee /etc/systemd/system/lelamp-livekit.service > /dev/null"
echo "✅ LiveKit 服务文件已创建"
echo ""

# 创建 API 服务
echo "📝 3. 创建 API 服务..."
ssh $PI_HOST "sudo tee /etc/systemd/system/lelamp-api.service > /dev/null << 'EOF'
[Unit]
Description=LeLamp API Server (Web Control)
Documentation=https://github.com/xwang152-jack/lelamp_runtime
After=network.target lelamp-livekit.service

[Service]
Type=simple
User=root
WorkingDirectory=/home/pi/lelamp_runtime
Environment=PATH=/usr/local/bin:/usr/bin:/bin
EnvironmentFile=/home/pi/lelamp_runtime/.env
ExecStart=/usr/local/bin/uv run python -m uvicorn lelamp.api.app:app --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=5
StartLimitInterval=120
StartLimitBurst=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=lelamp-api

[Install]
WantedBy=multi-user.target
EOF
"
echo "✅ API 服务文件已创建"
echo ""

# 重新加载 systemd
echo "🔄 5. 重新加载 systemd..."
ssh $PI_HOST "sudo systemctl daemon-reload"
echo "✅ systemd 已重新加载"
echo ""

# 启用服务（按顺序）
echo "🚀 6. 启用开机自启..."
ssh $PI_HOST "sudo systemctl enable lelamp-livekit.service"
ssh $PI_HOST "sudo systemctl enable lelamp-api.service"
echo "✅ 所有服务已设置为开机自启"
echo ""

# 启动服务（按顺序）
echo "▶️  7. 启动服务..."
echo "启动 LiveKit 服务..."
ssh $PI_HOST "sudo systemctl start lelamp-livekit.service"
sleep 3

echo "启动 API 服务..."
ssh $PI_HOST "sudo systemctl start lelamp-api.service"
sleep 3

echo "✅ 所有服务已启动"
echo ""

# 检查服务状态
echo ""
echo "================================================"
echo "🔍 服务状态检查"
echo "================================================"
echo ""

echo "📊 LiveKit 服务状态:"
ssh $PI_HOST "sudo systemctl status lelamp-livekit.service --no-pager" | head -10
echo ""

echo "📊 API 服务状态:"
ssh $PI_HOST "sudo systemctl status lelamp-api.service --no-pager" | head -10
echo ""

# 测试连接
echo "================================================"
echo "🧪 连接测试"
echo "================================================"
echo ""

echo "测试 API 服务..."
sleep 2
if ssh $PI_HOST "curl -s http://localhost:8000/health > /dev/null"; then
    echo "✅ API 服务响应正常"
    echo "   地址: http://192.168.0.104:8000"
    echo "   文档: http://192.168.0.104:8000/docs"
else
    echo "⏳ API 服务启动中..."
    echo "   地址: http://192.168.0.104:8000"
fi

echo ""
echo "================================================"
echo "✅ 完整服务设置完成！"
echo "================================================"
echo ""
echo "🎯 服务概览："
echo ""
echo "📱 LiveKit 服务（语音交互）:"
echo "  状态: ssh $PI_HOST 'sudo systemctl status lelamp-livekit.service'"
echo "  日志: ssh $PI_HOST 'sudo journalctl -u lelamp-livekit.service -f'"
echo "  管理: ./scripts/services/livekit_service_manager.sh <command>"
echo ""
echo "🔌 API 服务（后端）:"
echo "  状态: ssh $PI_HOST 'sudo systemctl status lelamp-api.service'"
echo "  日志: ssh $PI_HOST 'sudo journalctl -u lelamp-api.service -f'"
echo "  地址: http://192.168.0.104:8000"
echo "  文档: http://192.168.0.104:8000/docs"
echo ""
echo "💡 前端（web/）已独立，请单独部署："
echo "  cd web && pnpm build  # 构建"
echo "  部署到 Nginx 或其他静态服务器"
echo ""
