#!/bin/bash
#
# 部署脚本：在 Raspberry Pi 上拉取最新代码并重启服务
#

set -e  # 遇到错误立即退出

# 配置
PI_HOST="pi@192.168.0.104"
PROJECT_DIR="/home/pi/lelamp_runtime"  # 根据实际情况修改
SERVICE_NAME="lelamp-api"  # systemd 服务名称（如果有的话）

echo "========================================"
echo "部署到 Raspberry Pi"
echo "========================================"
echo "主机: $PI_HOST"
echo "项目目录: $PROJECT_DIR"
echo ""

# 1. 检查 SSH 连接
echo "1. 检查 SSH 连接..."
if ! ssh -o ConnectTimeout=5 "$PI_HOST" "echo 'SSH 连接成功'"; then
    echo "❌ 无法连接到 $PI_HOST"
    echo "请确保："
    echo "  - Raspberry Pi 已开机"
    echo "  - 网络连接正常"
    echo "  - SSH 密钥已配置"
    exit 1
fi

# 2. 拉取最新代码
echo ""
echo "2. 拉取最新代码..."
ssh "$PI_HOST" "cd $PROJECT_DIR && git pull origin main"

# 3. 安装/更新依赖（如果需要）
echo ""
echo "3. 检查依赖更新..."
ssh "$PI_HOST" "cd $PROJECT_DIR && uv sync"

# 4. 重启服务
echo ""
echo "4. 重启 API 服务..."
ssh "$PI_HOST" "sudo systemctl restart $SERVICE_NAME" || {
    echo "⚠️  systemd 服务重启失败，尝试手动重启..."
    echo "您可能需要在 Pi 上手动运行："
    echo "  sudo systemctl restart $SERVICE_NAME"
    echo "或者手动启动 API 服务器："
    echo "  cd $PROJECT_DIR && uv run scripts/run_api_server.py"
}

# 5. 检查服务状态
echo ""
echo "5. 检查服务状态..."
ssh "$PI_HOST" "systemctl is-active $SERVICE_NAME" && echo "✅ 服务运行正常" || echo "⚠️  服务未运行"

echo ""
echo "========================================"
echo "部署完成！"
echo "========================================"
echo ""
echo "您可以通过以下方式验证："
echo "  - 访问 http://$PI_HOST:8000/docs 查看 API 文档"
echo "  - 在前端设置界面尝试保存设置"
echo ""
