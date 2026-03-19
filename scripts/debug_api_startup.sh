#!/bin/bash
# API 启动诊断脚本 - 捕获详细日志

echo "================================================"
echo "LeLamp API 启动诊断"
echo "================================================"
echo ""

PI_HOST="pi@192.168.0.104"

echo "🔍 在树莓派上启用详细日志并启动 API..."
echo ""

ssh $PI_HOST "cd ~/lelamp_runtime && sudo -E uv run python -m uvicorn lelamp.api.app:app --host 0.0.0.0 --port 8000 --log-level debug" &
API_PID=$!

echo "API 进程 ID: $API_PID"
echo "等待 10 秒观察启动过程..."
echo ""

sleep 10

echo ""
echo "================================================"
echo "检查 API 进程状态..."
echo "================================================"

if ssh $PI_HOST "ps -p $API_PID > /dev/null"; then
    echo "✅ API 进程仍在运行 (PID: $API_PID)"
    echo ""
    echo "🔍 检查端口监听状态..."
    ssh $PI_HOST "sudo netstat -tlnp | grep :8000"

    echo ""
    echo "🌐 测试 API 端点..."
    ssh $PI_HOST "curl -s http://localhost:8000/health || echo '健康检查失败'"

    echo ""
    echo "✅ API 似乎正常运行，按 Ctrl+C 停止诊断"
    read

    # 清理
    kill $API_PID 2>/dev/null
    ssh $PI_HOST "sudo pkill -f 'uvicorn lelamp.api.app'" 2>/dev/null
else
    echo "❌ API 进程已停止"
    echo ""
    echo "🔍 获取详细的启动日志..."
    ssh $PI_HOST "cd ~/lelamp_runtime && sudo -E uv run python -m uvicorn lelamp.api.app:app --host 0.0.0.0 --port 8000 --log-level debug" &
    sleep 5
    echo ""
fi

echo ""
echo "================================================"
echo "诊断完成"
echo "================================================"
