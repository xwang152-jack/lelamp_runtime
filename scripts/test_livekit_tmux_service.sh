#!/bin/bash
# LiveKit Tmux Service 测试脚本

set -e

PI_HOST="${PI_HOST:-pi@192.168.0.104}"

echo "================================================"
echo "LeLamp LiveKit Tmux Service 测试"
echo "================================================"
echo ""

echo "🔍 测试 1: 检查服务状态"
if ssh $PI_HOST "sudo systemctl is-active lelamp-livekit.service" | grep -q "active"; then
    echo "✅ 服务运行中"
else
    echo "❌ 服务未运行"
    exit 1
fi
echo ""

echo "🔍 测试 2: 检查 tmux 会话"
if ssh $PI_HOST "tmux list-sessions | grep -q livekit"; then
    echo "✅ tmux 会话存在"
    ssh $PI_HOST "tmux list-sessions"
else
    echo "❌ tmux 会话不存在"
    exit 1
fi
echo ""

echo "🔍 测试 3: 检查服务日志"
echo "最近的日志："
ssh $PI_HOST "sudo journalctl -u lelamp-livekit.service -n 10 --no-pager"
echo ""

echo "🔍 测试 4: 测试服务重启"
echo "重启服务..."
ssh $PI_HOST "sudo systemctl restart lelamp-livekit.service"
sleep 5

if ssh $PI_HOST "sudo systemctl is-active lelamp-livekit.service" | grep -q "active"; then
    echo "✅ 服务重启成功"
else
    echo "❌ 服务重启失败"
    exit 1
fi
echo ""

echo "🔍 测试 5: 检查自动重启功能"
echo "杀死 tmux 会话..."
ssh $PI_HOST "tmux kill-session -t livekit" 2>/dev/null || true
sleep 10

if ssh $PI_HOST "sudo systemctl is-active lelamp-livekit.service" | grep -q "active"; then
    echo "✅ 自动重启功能正常"
else
    echo "❌ 自动重启功能失败"
    exit 1
fi
echo ""

echo "================================================"
echo "✅ 所有测试通过！"
echo "================================================"
