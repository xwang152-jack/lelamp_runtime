#!/bin/bash
# LeLamp API 服务重启脚本

echo "🔄 正在重启 LeLamp API 服务..."

# 查找并停止所有 uvicorn 进程
echo "1️⃣ 停止现有服务..."
pkill -f "uvicorn lelamp.api.app:app" || echo "没有运行的服务需要停止"

# 等待进程完全停止
sleep 2

# 确认端口已释放
echo "2️⃣ 检查端口8000..."
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  端口8000仍被占用，强制清理..."
    sudo fuser -k 8000/tcp
    sleep 1
fi

# 启动新服务
echo "3️⃣ 启动API服务..."
cd ~/lelamp_runtime
nohup .venv/bin/python -m uvicorn lelamp.api.app:app \
    --host 0.0.0.0 \
    --port 8000 \
    > /tmp/lelamp-api.log 2>&1 &

# 等待服务启动
sleep 3

# 验证服务状态
echo "4️⃣ 验证服务状态..."
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ 服务启动成功！"
    echo "📍 API地址: http://$(hostname -I | awk '{print $1}'):8000"
    echo "📊 健康检查: http://localhost:8000/health"
    echo "📖 API文档: http://localhost:8000/docs"
else
    echo "❌ 服务启动失败，请检查日志："
    echo "tail -20 /tmp/lelamp-api.log"
    exit 1
fi

echo ""
echo "🎉 LeLamp API 服务重启完成！"
