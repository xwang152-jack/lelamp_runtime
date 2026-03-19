#!/bin/bash
# LeLamp 完整服务启动设置（API + 对话）

set -e

PI_HOST="pi@192.168.0.104"
PROJECT_DIR="/home/pi/lelamp_runtime"

echo "================================================"
echo "LeLamp 完整服务设置（API + 语音对话）"
echo "================================================"
echo ""
echo "这将同时启动："
echo "  - API 服务器（Web界面 + REST API）"
echo "  - 语音对话代理（LiveKit语音交互）"
echo ""

# 检查连接
echo "🔍 检查树莓派连接..."
if ! ssh $PI_HOST "echo '连接成功'" 2>/dev/null; then
    echo "❌ 无法连接到树莓派"
    exit 1
fi
echo "✅ 树莓派连接正常"
echo ""

# 1. 停止现有服务
echo "🛑 1. 停止现有服务..."
ssh $PI_HOST "sudo systemctl stop lelamp-api.service 2>/dev/null || true"
ssh $PI_HOST "sudo systemctl disable lelamp-api.service 2>/dev/null || true"
echo "✅ 现有服务已停止"
echo ""

# 2. 创建对话代理服务
echo "📝 2. 创建对话代理 systemd 服务..."
ssh $PI_HOST "sudo tee /etc/systemd/system/lelamp-conversation.service > /dev/null << 'EOF'
[Unit]
Description=LeLamp Voice Conversation Agent
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/home/pi/lelamp_runtime
Environment=PATH=/usr/local/bin:/usr/bin:/bin
EnvironmentFile=/home/pi/lelamp_runtime/.env
ExecStart=/usr/local/bin/uv run main.py console
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
"

echo "✅ 对话代理服务文件已创建"
echo ""

# 3. 重新加载 systemd
echo "🔄 3. 配置系统服务..."
ssh $PI_HOST "sudo systemctl daemon-reload"
ssh $PI_HOST "sudo systemctl enable lelamp-conversation.service"
echo "✅ 对话代理已设置为开机自启"
echo ""

# 4. 启动对话代理
echo "▶️  4. 启动对话代理..."
ssh $PI_HOST "sudo systemctl start lelamp-conversation.service"
sleep 5
echo "✅ 对话代理已启动"
echo ""

echo "================================================"
echo "✅ 完整服务设置完成！"
echo "================================================"
echo ""
echo "🎤 当前功能："
echo "   ✅ 语音对话（直接对台灯说话）"
echo "   ✅ LiveKit 语音识别和合成"
echo "   ✅ AI 对话能力（DeepSeek）"
echo "   ✅ 电机控制"
echo "   ✅ 灯光控制"
echo ""
echo "🔧 管理命令："
echo "   状态: ssh $PI_HOST 'sudo systemctl status lelamp-conversation.service'"
echo "   重启: ssh $PI_HOST 'sudo systemctl restart lelamp-conversation.service'"
echo "   日志: ssh $PI_HOST 'sudo journalctl -u lelamp-conversation.service -f'"
echo ""
echo "🌐 如需 Web 界面，可以手动启动 API："
echo "   ssh $PI_HOST 'cd ~/lelamp_runtime && sudo -E uv run python -m uvicorn lelamp.api.app:app --host 0.0.0.0 --port 8000 &'"
echo ""
