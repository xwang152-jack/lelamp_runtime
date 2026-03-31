#!/bin/bash
# LeLamp API 服务器自动启动设置

set -e

PI_HOST="pi@192.168.0.104"
PROJECT_DIR="/home/pi/lelamp_runtime"

echo "================================================"
echo "LeLamp API 服务器自动启动设置"
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

# 1. 停止现有服务
echo "🛑 1. 停止现有服务..."
ssh $PI_HOST "sudo systemctl stop lelamp-conversation.service 2>/dev/null || true"
ssh $PI_HOST "sudo systemctl disable lelamp-conversation.service 2>/dev/null || true"
ssh $PI_HOST "sudo systemctl stop lelamp-api.service 2>/dev/null || true"
echo "✅ 现有服务已停止"
echo ""

# 2. 创建 API 服务启动脚本
echo "📝 2. 创建 API 服务启动脚本..."
ssh $PI_HOST 'cat > /home/pi/lelamp_runtime/start_api.sh << '\''EOFSCRIPT'\''
#!/bin/bash
cd /home/pi/lelamp_runtime
exec /usr/local/bin/uv run python -m uvicorn lelamp.api.app:app --host 0.0.0.0 --port 8000
EOFSCRIPT
chmod +x /home/pi/lelamp_runtime/start_api.sh
'

echo "✅ 启动脚本已创建"
echo ""

# 3. 创建 systemd 服务
echo "📝 3. 创建 systemd 服务..."
ssh $PI_HOST 'sudo tee /etc/systemd/system/lelamp-api.service > /dev/null << '\''EOF'\''
[Unit]
Description=LeLamp API Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/home/pi/lelamp_runtime
Environment=PATH=/usr/local/bin:/usr/bin:/bin
EnvironmentFile=/home/pi/lelamp_runtime/.env
StandardInput=null
StandardOutput=journal
StandardError=journal
ExecStart=/home/pi/lelamp_runtime/start_api.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
'

echo "✅ 服务文件已创建"
echo ""

# 4. 重新加载并启用
echo "🔄 4. 配置自动启动..."
ssh $PI_HOST "sudo systemctl daemon-reload"
ssh $PI_HOST "sudo systemctl enable lelamp-api.service"
echo "✅ 服务已设置为开机自启"
echo ""

# 5. 启动服务
echo "▶️  5. 启动服务..."
ssh $PI_HOST "sudo systemctl start lelamp-api.service"
sleep 3
echo "✅ 服务已启动"
echo ""

# 6. 检查服务状态
echo "================================================"
echo "🔍 服务状态"
echo "================================================"
echo ""
ssh $PI_HOST "sudo systemctl status lelamp-api.service --no-pager" | head -15
echo ""

# 7. 测试连接
echo "================================================"
echo "🧪 连接测试"
echo "================================================"
echo ""
sleep 2
if ssh $PI_HOST "curl -s http://localhost:8000/health > /dev/null"; then
    echo "✅ API 服务器响应正常"
    echo ""
    echo "🌐 访问地址："
    echo "   Web 界面: http://192.168.0.104:8000/index.html"
    echo "   API 文档: http://192.168.0.104:8000/docs"
    echo "   API 根路径: http://192.168.0.104:8000/api"
else
    echo "⏳ API 服务器启动中，请稍候..."
    echo "   可以通过以下命令查看日志："
    echo "   ssh $PI_HOST 'sudo journalctl -u lelamp-api.service -f'"
fi

echo ""
echo "================================================"
echo "✅ API 服务器自动启动设置完成！"
echo "================================================"
echo ""
echo "🎯 管理命令："
echo "   状态: ssh $PI_HOST 'sudo systemctl status lelamp-api.service'"
echo "   重启: ssh $PI_HOST 'sudo systemctl restart lelamp-api.service'"
echo "   日志: ssh $PI_HOST 'sudo journalctl -u lelamp-api.service -f'"
echo ""
