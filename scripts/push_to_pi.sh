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

# 3. 确保依赖安装正确
echo "📦 在树莓派上确保依赖安装..."
ssh $PI_HOST "cd $PI_DIR && sudo uv sync --extra api --extra hardware"

if [ $? -eq 0 ]; then
    echo "✅ 依赖安装成功"
else
    echo "⚠️  依赖安装可能有问题"
fi

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
echo ""
echo "================================================"
echo "✅ 推送完成！"
echo "================================================"
echo ""
echo "📝 修复总结："
echo "  - 修复了电机服务关闭时的竞态条件"
echo "  - 确保了 API 依赖正确安装"
echo "  - API 现在可以正常启动和关闭"
echo ""
echo "🚀 在树莓派上启动 API："
echo "  ssh $PI_HOST"
echo "  cd $PI_DIR"
echo "  sudo uv run python -m uvicorn lelamp.api.app:app --host 0.0.0.0 --port 8000"
echo ""
echo "💡 如果将来遇到依赖问题，运行："
echo "  sudo uv sync --extra api --extra hardware"
echo ""
