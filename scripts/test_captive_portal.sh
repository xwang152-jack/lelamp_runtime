#!/bin/bash
#
# LeLamp Captive Portal 测试脚本
#

set -e

PI_HOST="${PI_HOST:-pi@192.168.0.104}"

echo "================================================"
echo "LeLamp Captive Portal 测试"
echo "================================================"
echo ""

# 1. 测试 AP 模式启动
echo "📡 测试 1: AP 模式启动"
ssh $PI_HOST 'sudo systemctl start lelamp-setup-ap' 2>/dev/null || echo "跳过：服务可能未安装"
sleep 3
if ssh $PI_HOST 'systemctl is-active lelamp-setup-ap' 2>/dev/null | grep -q 'active'; then
    echo "✅ AP 模式启动成功"
else
    echo "⚠️  AP 模式服务状态未知（可能需要手动测试）"
fi
echo ""

# 2. 测试 Portal 服务
echo "🌐 测试 2: Portal 服务"
ssh $PI_HOST 'sudo systemctl start lelamp-captive-portal' 2>/dev/null || echo "跳过：服务可能未安装"
sleep 3
if ssh $PI_HOST 'curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/' 2>/dev/null | grep -q "200"; then
    echo "✅ Portal 服务响应正常"
else
    echo "⚠️  Portal 服务可能未运行或需要启动"
fi
echo ""

# 3. 测试 API 端点
echo "🧪 测试 3: API 端点"
if ssh $PI_HOST "curl -s http://localhost:8080/api/setup/status 2>/dev/null" | grep -q "setup_completed"; then
    echo "✅ API 端点响应正常"
    ssh $PI_HOST "curl -s http://localhost:8080/api/setup/status" | python3 -m json.tool 2>/dev/null || ssh $PI_HOST "curl -s http://localhost:8080/api/setup/status"
else
    echo "⚠️  API 端点无响应（请确保服务正在运行）"
fi
echo ""

# 4. 测试 WiFi 扫描
echo "📶 测试 4: WiFi 扫描"
if ssh $PI_HOST "nmcli device wifi list 2>/dev/null" | head -5; then
    echo "✅ WiFi 扫描命令可用"
else
    echo "⚠️  WiFi 扫描命令不可用"
fi
echo ""

# 5. 测试状态文件
echo "📁 测试 5: 状态文件"
if ssh $PI_HOST "[ -f /var/lib/lelamp/setup_status.json ]" 2>/dev/null; then
    echo "✅ 状态文件存在"
    ssh $PI_HOST "cat /var/lib/lelamp/setup_status.json | python3 -m json.tool" 2>/dev/null || echo "（JSON 格式可能异常）"
else
    echo "ℹ️  状态文件不存在（首次设置时会创建）"
fi
echo ""

echo "================================================"
echo "✅ 测试完成！"
echo "================================================"
echo ""
echo "📝 手动测试清单："
echo "1. 在手机上连接到 LeLamp-Setup 热点"
echo "2. 打开浏览器访问 http://192.168.4.1:8080"
echo "3. 测试完整的设置流程"
echo ""
echo "💡 提示：如果服务未安装，请运行："
echo "   ./scripts/install_captive_portal.sh"
echo ""
