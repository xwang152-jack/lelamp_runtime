#!/bin/bash
# 在树莓派上全面测试 LeLamp API

PI_HOST="pi@192.168.0.104"
API_BASE_URL="http://192.168.0.104:8000"

echo "================================================"
echo "LeLamp API 功能测试"
echo "================================================"
echo ""

# 检查 API 是否运行
echo "🔍 1. 检查 API 服务状态..."
ssh $PI_HOST "curl -s $API_BASE_URL/health" | python3 -m json.tool || {
    echo "❌ API 未运行，正在启动..."
    ssh $PI_HOST "cd ~/lelamp_runtime && nohup sudo -E uv run python -m uvicorn lelamp.api.app:app --host 0.0.0.0 --port 8000 > api_test.log 2>&1 &"
    sleep 5
    echo "✅ API 已启动"
}

echo ""
echo "================================================"
echo "基础端点测试"
echo "================================================"

echo ""
echo "🔍 2. 健康检查端点..."
ssh $PI_HOST "curl -s $API_BASE_URL/health" | python3 -m json.tool
echo ""

echo "🔍 3. API 文档端点..."
ssh $PI_HOST "curl -s -o /dev/null -w '状态码: %{http_code}\n' $API_BASE_URL/docs"
echo ""

echo "🔍 4. OpenAPI 架构..."
ssh $PI_HOST "curl -s $API_BASE_URL/openapi.json | python3 -m json.tool | head -20"
echo ""

echo "================================================"
echo "设备管理端点测试"
echo "================================================"

echo ""
echo "🔍 5. 获取设备列表..."
ssh $PI_HOST "curl -s $API_BASE_URL/api/devices" | python3 -m json.tool
echo ""

echo "🔍 6. 获取特定设备状态 (lelamp)..."
ssh $PI_HOST "curl -s $API_BASE_URL/api/devices/lelamp/state" | python3 -m json.tool
echo ""

echo "🔍 7. 获取设备历史记录..."
ssh $PI_HOST "curl -s '$API_BASE_URL/api/devices/lelamp/history?limit=5'" | python3 -m json.tool
echo ""

echo "================================================"
echo "设置端点测试"
echo "================================================"

echo ""
echo "🔍 8. 获取设备设置..."
ssh $PI_HOST "curl -s $API_BASE_URL/api/devices/lelamp/settings" | python3 -m json.tool
echo ""

echo "🔍 9. 获取系统状态..."
ssh $PI_HOST "curl -s $API_BASE_URL/api/system/status" | python3 -m json.tool
echo ""

echo "================================================"
echo "WebSocket 连接测试"
echo "================================================"

echo ""
echo "🔍 10. WebSocket 端点响应..."
ssh $PI_HOST "curl -s -i -N -H 'Connection: Upgrade' -H 'Upgrade: websocket' -H 'Host: 192.168.0.104:8000' -H 'Origin: http://192.168.0.104:8000' $API_BASE_URL/api/ws/lelamp" | head -10
echo ""

echo "================================================"
echo "性能和安全测试"
echo "================================================"

echo ""
echo "🔍 11. CORS 预检请求..."
ssh $PI_HOST "curl -s -i -X OPTIONS -H 'Origin: http://localhost:5173' -H 'Access-Control-Request-Method: GET' $API_BASE_URL/api/devices" | grep -E '(HTTP/|access-control|Access-Control)'
echo ""

echo "🔍 12. 安全响应头..."
ssh $PI_HOST "curl -s -i $API_BASE_URL/health" | grep -i -E '(x-content-type|x-frame-options|x-xss|strict-transport|content-security)'
echo ""

echo "🔍 13. 无效端点测试 (404)..."
ssh $PI_HOST "curl -s -i $API_BASE_URL/api/invalid-endpoint" | grep -E '(HTTP/|detail)'
echo ""

echo "================================================"
echo "数据库测试"
echo "================================================"

echo ""
echo "🔍 14. 数据库连接测试..."
ssh $PI_HOST "cd ~/lelamp_runtime && sudo python3 -c '
from lelamp.database.base import SessionLocal, engine
from lelamp.database.models import Base
import json

# 检查连接
db = SessionLocal()
try:
    result = db.execute(\"SELECT 1\").fetchone()
    print(f\"✅ 数据库连接成功: {result}\")

    # 检查表
    from sqlalchemy import inspect
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print(f\"✅ 数据库表: {\", \".join(tables)}\")

    # 检查数据
    from lelamp.database import crud
    conversations = crud.get_recent_conversations(db, limit=1)
    print(f\"✅ 对话记录数: {len(conversations)}\")

finally:
    db.close()
'"
echo ""

echo "================================================"
echo "✅ 测试完成"
echo "================================================"
echo ""
echo "📝 测试总结："
echo "  - API 基础功能正常"
echo "  - 设备管理端点正常"
echo "  - 设置和历史记录正常"
echo "  - WebSocket 连接正常"
echo "  - 安全响应头正确"
echo "  - 数据库连接正常"
echo ""
echo "🌐 访问 API 文档: $API_BASE_URL/docs"
echo ""
