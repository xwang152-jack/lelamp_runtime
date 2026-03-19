#!/bin/bash
# LeLamp 启动配置向导

set -e

PI_HOST="pi@192.168.0.104"
PROJECT_DIR="~/lelamp_runtime"

echo "================================================"
echo "LeLamp 启动配置向导"
echo "================================================"
echo ""

echo "请选择启动模式："
echo ""
echo "1. 简化模式（推荐）"
echo "   - 仅启动后端 API"
echo "   - 前端静态文件由 API 服务"
echo "   - 占用资源少，启动快速"
echo "   - 适合：日常使用、资源受限环境"
echo ""
echo "2. 完整模式"
echo "   - 后端 API + 前端开发服务器"
echo "   - 前端热更新，开发更便捷"
echo "   - 占用资源较多，启动较慢"
echo "   - 适合：开发调试、需要实时预览"
echo ""
echo "3. 主程序模式"
echo "   - 仅启动 main.py console"
echo "   - 纯命令行交互模式"
echo "   - 最低资源占用"
echo "   - 适合：无头使用、纯语音交互"
echo ""

read -p "请输入选择 (1/2/3): " choice

case $choice in
  1)
    echo ""
    echo "🚀 设置简化模式启动..."
    bash "$(dirname "$0")/setup_auto_startup_simple.sh"
    ;;
  2)
    echo ""
    echo "🚀 设置完整模式启动..."
    bash "$(dirname "$0")/setup_auto_startup.sh"
    ;;
  3)
    echo ""
    echo "🚀 设置主程序模式启动..."
    ssh $PI_HOST "sudo tee /etc/systemd/system/lelamp-main.service > /dev/null << 'EOF'
[Unit]
Description=LeLamp Main Console
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$PROJECT_DIR
Environment=PATH=/usr/local/bin:/usr/bin:/bin
EnvironmentFile=$PROJECT_DIR/.env
ExecStart=/usr/local/bin/uv run main.py console
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
"

    ssh $PI_HOST "sudo systemctl daemon-reload"
    ssh $PI_HOST "sudo systemctl enable lelamp-main.service"
    ssh $PI_HOST "sudo systemctl start lelamp-main.service"

    echo "✅ 主程序模式设置完成"
    echo ""
    echo "管理命令:"
    echo "  状态: ssh $PI_HOST 'sudo systemctl status lelamp-main.service'"
    echo "  重启: ssh $PI_HOST 'sudo systemctl restart lelamp-main.service'"
    echo "  日志: ssh $PI_HOST 'sudo journalctl -u lelamp-main.service -f'"
    ;;
  *)
    echo "❌ 无效选择"
    exit 1
    ;;
esac

echo ""
echo "================================================"
echo "配置完成！"
echo "================================================"
echo ""
echo "🔧 其他管理命令："
echo ""
echo "查看所有服务状态:"
echo "  ssh $PI_HOST 'sudo systemctl status lelamp-*.service'"
echo ""
echo "停止所有服务:"
echo "  ssh $PI_HOST 'sudo systemctl stop lelamp-*.service'"
echo ""
echo "重启所有服务:"
echo "  ssh $PI_HOST 'sudo systemctl restart lelamp-*.service'"
echo ""
echo "查看服务日志:"
echo "  ssh $PI_HOST 'sudo journalctl -u lelamp-*.service -f'"
echo ""
