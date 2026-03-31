#!/bin/bash
# LeLamp LiveKit Tmux Service 设置脚本

set -e

PI_HOST="${PI_HOST:-pi@192.168.0.104}"
PROJECT_DIR="/home/pi/lelamp_runtime"
SERVICE_FILE="scripts/lelamp-livekit.service"

echo "================================================"
echo "LeLamp LiveKit Tmux Service 设置"
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

# 检查 tmux 是否安装
echo "🔍 检查 tmux 是否安装..."
if ! ssh $PI_HOST "which tmux" 2>/dev/null; then
    echo "📦 正在安装 tmux..."
    ssh $PI_HOST "sudo apt-get update && sudo apt-get install -y tmux"
else
    echo "✅ tmux 已安装"
fi
echo ""

# 停止并禁用旧的 LiveKit 服务（如果存在）
echo "🔧 1. 清理旧服务..."
ssh $PI_HOST "sudo systemctl stop lelamp-livekit.service 2>/dev/null || true"
ssh $PI_HOST "sudo systemctl disable lelamp-livekit.service 2>/dev/null || true"
# 同时停止可能存在的旧服务
ssh $PI_HOST "sudo systemctl stop lelamp-agent.service 2>/dev/null || true"
ssh $PI_HOST "sudo systemctl disable lelamp-agent.service 2>/dev/null || true"
echo "✅ 旧服务已清理"
echo ""

# 创建新的 systemd 服务文件
echo "📝 2. 创建 systemd 服务文件..."
cat $SERVICE_FILE | ssh $PI_HOST "sudo tee /etc/systemd/system/lelamp-livekit.service > /dev/null"
echo "✅ 服务文件已创建"
echo ""

# 重新加载 systemd
echo "🔄 3. 重新加载 systemd..."
ssh $PI_HOST "sudo systemctl daemon-reload"
echo "✅ systemd 已重新加载"
echo ""

# 启用服务
echo "🚀 4. 启用开机自启..."
ssh $PI_HOST "sudo systemctl enable lelamp-livekit.service"
echo "✅ 服务已设置为开机自启"
echo ""

# 启动服务
echo "▶️  5. 启动服务..."
ssh $PI_HOST "sudo systemctl start lelamp-livekit.service"
echo "✅ 服务已启动"
echo ""

# 等待服务启动
sleep 3

# 检查服务状态
echo ""
echo "================================================"
echo "🔍 服务状态检查"
echo "================================================"
echo ""
ssh $PI_HOST "sudo systemctl status lelamp-livekit.service --no-pager -l" | head -20
echo ""

# 检查 tmux 会话
echo "🔍 Tmux 会话状态:"
ssh $PI_HOST "tmux list-sessions 2>/dev/null || echo 'tmux 会话不存在或正在启动'"
echo ""

echo ""
echo "================================================"
echo "✅ LiveKit Tmux Service 设置完成！"
echo "================================================"
echo ""
echo "🎯 管理命令："
echo ""
echo "查看服务状态:"
echo "  ssh $PI_HOST 'sudo systemctl status lelamp-livekit.service'"
echo ""
echo "查看实时日志:"
echo "  ssh $PI_HOST 'sudo journalctl -u lelamp-livekit.service -f'"
echo ""
echo "连接到 tmux 会话（查看实时输出）:"
echo "  ssh $PI_HOST 'tmux attach -t livekit'"
echo "  退出 tmux 会话：按 Ctrl+B 然后 D"
echo ""
echo "重启服务:"
echo "  ssh $PI_HOST 'sudo systemctl restart lelamp-livekit.service'"
echo ""
echo "停止服务:"
echo "  ssh $PI_HOST 'sudo systemctl stop lelamp-livekit.service'"
echo ""
echo "禁用开机自启:"
echo "  ssh $PI_HOST 'sudo systemctl disable lelamp-livekit.service'"
echo ""
