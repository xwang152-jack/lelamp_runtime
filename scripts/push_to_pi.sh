#!/bin/bash
# 直接推送本地代码到树莓派

PI_HOST="pi@192.168.0.104"
PI_DIR="~/lelamp_runtime"

echo "================================================"
echo "LeLamp Runtime - 推送到树莓派"
echo "================================================"
echo ""

# 检查本地是否有未提交的更改
if [ -n "$(git status --porcelain)" ]; then
    echo "⚠️  检测到未提交的更改，请先提交："
    git status --short
    echo ""
    echo "请运行："
    echo "  git add ."
    echo "  git commit -m 'your message'"
    exit 1
fi

echo "✅ 本地仓库状态干净"
echo ""

# 1. 在树莓派上备份当前状态
echo "📦 在树莓派上备份当前状态..."
ssh $PI_HOST "cd $PI_DIR && git stash save 'auto-backup-$(date +%Y%m%d_%H%M%S)'" 2>/dev/null || echo "没有需要备份的更改"

echo ""

# 2. 在树莓派上重置到远程 main 分支
echo "🔄 在树莓派上重置代码..."
ssh $PI_HOST "cd $PI_DIR && git fetch origin && git reset --hard origin/main && git clean -fd"

if [ $? -eq 0 ]; then
    echo "✅ 代码同步成功"
else
    echo "❌ 代码同步失败"
    exit 1
fi

echo ""

# 3. 显示同步的提交
echo "📝 最新提交信息："
ssh $PI_HOST "cd $PI_DIR && git log -1 --oneline"

echo ""
echo "================================================"
echo "✅ 推送完成！"
echo "================================================"
echo ""
echo "🚀 在树莓派上启动 API："
echo "  ssh $PI_HOST"
echo "  cd $PI_DIR"
echo "  sudo uv run python -m uvicorn lelamp.api.app:app --host 0.0.0.0 --port 8000"
echo ""
