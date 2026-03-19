#!/bin/bash
# LeLamp LiveKit Service 管理脚本

PI_HOST="${PI_HOST:-pi@192.168.0.104}"

case "$1" in
    status)
        echo "📊 服务状态："
        ssh $PI_HOST 'sudo systemctl status lelamp-livekit.service --no-pager -l' | head -15
        echo ""
        echo "🖥️  Tmux 会话："
        ssh $PI_HOST 'tmux list-sessions 2>/dev/null || echo "无活跃会话"'
        ;;

    logs)
        echo "📋 实时日志（Ctrl+C 退出）："
        ssh $PI_HOST 'sudo journalctl -u lelamp-livekit.service -f'
        ;;

    attach)
        echo "🔌 连接到 tmux 会话（Ctrl+B 然后 D 退出）："
        ssh $PI_HOST 'tmux attach -t livekit'
        ;;

    restart)
        echo "🔄 重启服务..."
        ssh $PI_HOST 'sudo systemctl restart lelamp-livekit.service'
        echo "✅ 服务已重启"
        echo ""
        echo "📊 服务状态："
        ssh $PI_HOST 'sudo systemctl status lelamp-livekit.service --no-pager' | head -10
        ;;

    stop)
        echo "⏹️  停止服务..."
        ssh $PI_HOST 'sudo systemctl stop lelamp-livekit.service'
        echo "✅ 服务已停止"
        ;;

    start)
        echo "▶️  启动服务..."
        ssh $PI_HOST 'sudo systemctl start lelamp-livekit.service'
        echo "✅ 服务已启动"
        echo ""
        echo "📊 服务状态："
        ssh $PI_HOST 'sudo systemctl status lelamp-livekit.service --no-pager' | head -10
        ;;

    enable)
        echo "🚀 启用开机自启..."
        ssh $PI_HOST 'sudo systemctl enable lelamp-livekit.service'
        echo "✅ 已启用开机自启"
        ;;

    disable)
        echo "🔒 禁用开机自启..."
        ssh $PI_HOST 'sudo systemctl disable lelamp-livekit.service'
        echo "✅ 已禁用开机自启"
        ;;

    *)
        echo "LeLamp LiveKit Service 管理脚本"
        echo ""
        echo "用法: $0 {status|logs|attach|restart|stop|start|enable|disable}"
        echo ""
        echo "命令说明："
        echo "  status  - 查看服务状态和 tmux 会话"
        echo "  logs    - 查看实时日志"
        echo "  attach  - 连接到 tmux 会话（查看实时输出）"
        echo "  restart - 重启服务"
        echo "  stop    - 停止服务"
        echo "  start   - 启动服务"
        echo "  enable  - 启用开机自启"
        echo "  disable - 禁用开机自启"
        echo ""
        echo "示例："
        echo "  $0 status     # 查看状态"
        echo "  $0 logs       # 查看日志"
        echo "  $0 attach     # 连接到 tmux"
        exit 1
        ;;
esac
