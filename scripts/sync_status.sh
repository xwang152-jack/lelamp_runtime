#!/bin/bash
# LeLamp Runtime 同步状态检查和同步工具

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PI_HOST="pi@192.168.0.104"
PI_DIR="~/lelamp_runtime"

echo "================================================"
echo "LeLamp Runtime - 仓库同步工具"
echo "================================================"
echo ""

# 检查本地状态
echo "🔍 1. 检查本地仓库状态..."
cd "$PROJECT_DIR"
LOCAL_STATUS=$(git status --porcelain)
LOCAL_COMMIT=$(git log -1 --oneline)
echo "   本地最新提交: $LOCAL_COMMIT"

if [ -n "$LOCAL_STATUS" ]; then
    echo "   ⚠️  有未提交的更改:"
    echo "$LOCAL_STATUS"
    echo ""
    read -p "是否要提交这些更改? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git add .
        read -p "请输入提交信息: " commit_msg
        git commit -m "$commit_msg"
        echo "   ✅ 更改已提交"
    fi
else
    echo "   ✅ 本地仓库干净"
fi

echo ""

# 检查 GitHub 同步
echo "🔍 2. 检查与 GitHub 的同步..."
BEHIND=$(git rev-list --count --left-right @{u}...HEAD 2>/dev/null | awk '{print $1}')
AHEAD=$(git rev-list --count --left-right @{u}...HEAD 2>/dev/null | awk '{print $2}')

if [ "$BEHIND" -gt 0 ]; then
    echo "   ⬇️  本地落后 GitHub $BEHIND 个提交"
    git pull origin main
    echo "   ✅ 已从 GitHub 拉取最新代码"
elif [ "$AHEAD" -gt 0 ]; then
    echo "   ⬆️  本地领先 GitHub $AHEAD 个提交"
    git push origin main
    echo "   ✅ 已推送到 GitHub"
else
    echo "   ✅ 与 GitHub 完全同步"
fi

echo ""

# 检查树莓派同步
echo "🔍 3. 检查与树莓派的同步..."
PI_COMMIT=$(ssh $PI_HOST "cd $PI_DIR && git log -1 --oneline" 2>/dev/null)
LOCAL_COMMIT_SHORT=$(git log -1 --oneline | awk '{print $1}')

if [ "$PI_COMMIT" != "$LOCAL_COMMIT_SHORT"* ]; then
    echo "   🔄 树莓派需要同步"
    echo "   树莓派: $PI_COMMIT"
    echo "   本地:   $LOCAL_COMMIT_SHORT"

    # 保存树莓派上的本地更改
    echo "   💾 保存树莓派本地更改..."
    ssh $PI_HOST "cd $PI_DIR && git stash push -m 'auto-sync-$(date +%Y%m%d_%H%M%S)'"

    # 同步代码
    echo "   📥 同步代码到树莓派..."
    ssh $PI_HOST "cd $PI_DIR && git fetch origin && git reset --hard origin/main"

    # 恢复本地更改
    echo "   💾 恢复树莓派本地更改..."
    ssh $PI_HOST "cd $PI_DIR && git stash pop"

    # 确保依赖
    echo "   📦 确保依赖安装..."
    ssh $PI_HOST "cd $PI_DIR && sudo uv sync --extra api --extra hardware"

    echo "   ✅ 树莓派同步完成"
else
    echo "   ✅ 与树莓派完全同步"
fi

echo ""
echo "================================================"
echo "✅ 同步完成！"
echo "================================================"
echo ""

# 显示最终状态
echo "📊 最终同步状态:"
echo ""
echo "本地:"
git log -1 --oneline
echo ""
echo "GitHub:"
git log -1 --oneline origin/main
echo ""
echo "树莓派:"
ssh $PI_HOST "cd $PI_DIR && git log -1 --oneline"
echo ""
echo "🎯 所有位置已同步到同一版本"
echo ""
