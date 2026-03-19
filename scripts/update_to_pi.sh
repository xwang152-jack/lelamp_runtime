#!/bin/bash
# 安全更新树莓派代码脚本

PI_HOST="pi@192.168.0.104"
PROJECT_DIR="~/lelamp_runtime"

echo "================================================"
echo "LeLamp Runtime - 树莓派代码更新工具"
echo "================================================"
echo ""

# 1. 备份树莓派上的本地修改
echo "📦 步骤 1: 备份树莓派上的本地修改..."
ssh $PI_HOST "cd $PROJECT_DIR && git stash push -m 'auto-backup-before-update-$(date +%Y%m%d_%H%M%S)'"

if [ $? -eq 0 ]; then
    echo "✅ 本地修改已备份到 git stash"
else
    echo "⚠️  备份失败，继续更新..."
fi

echo ""

# 2. 清理未跟踪的文件（临时备份）
echo "🧹 步骤 2: 清理未跟踪的文件..."
ssh $PI_HOST "cd $PROJECT_DIR && mkdir -p ../lelamp_backup_\$(date +%Y%m%d_%H%M%S) && find . -type f -not -path './.git/*' -not -path './.venv/*' -not -path './__pycache__/*' -not -path './*/__pycache__/*' -exec cp --parents {} ../lelamp_backup_\$(date +%Y%m%d_%H%M%S)/ \; 2>/dev/null || true"

echo "📥 步骤 3: 拉取最新代码..."
ssh $PI_HOST "cd $PROJECT_DIR && git clean -fd && git pull origin main"

if [ $? -eq 0 ]; then
    echo "✅ 代码拉取成功"
else
    echo "❌ 代码拉取失败"
    exit 1
fi

echo ""

# 4. 安装/更新依赖
echo "📦 步骤 4: 更新 Python 依赖..."
ssh $PI_HOST "cd $PROJECT_DIR && uv sync"

if [ $? -eq 0 ]; then
    echo "✅ 依赖更新成功"
else
    echo "⚠️  依赖更新失败（可能不需要更新）"
fi

echo ""

# 5. 检查服务状态
echo "🔍 步骤 5: 检查 LeLamp API 服务状态..."
ssh $PI_HOST "systemctl is-active lelamp-api.service" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "🔄 重启 LeLamp API 服务..."
    ssh $PI_HOST "sudo systemctl restart lelamp-api.service"
    sleep 2

    # 检查服务是否正常启动
    ssh $PI_HOST "systemctl is-active lelamp-api.service"
    if [ $? -eq 0 ]; then
        echo "✅ API 服务重启成功"
    else
        echo "❌ API 服务重启失败，检查日志："
        ssh $PI_HOST "sudo journalctl -u lelamp-api.service -n 20 --no-pager"
    fi
else
    echo "ℹ️  LeLamp API 服务未安装或未运行"
fi

echo ""
echo "================================================"
echo "✅ 更新完成！"
echo "================================================"
echo ""
echo "📝 更新内容："
echo "  - 修复电机服务关闭时的竞态条件"
echo "  - 添加 API 启动诊断工具"
echo "  - 添加服务关闭测试"
echo ""
echo "🚀 启动命令："
echo "  sudo uv run python -m uvicorn lelamp.api.app:app --host 0.0.0.0 --port 8000"
echo ""
echo "🔍 查看备份的修改："
echo "  ssh $PI_HOST 'cd $PROJECT_DIR && git stash list'"
echo ""
