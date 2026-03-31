#!/bin/bash
#
# 快速同步代码到树莓派（通过推送，无需SSH服务）
#

set -e

PI_HOST="pi@192.168.0.104"
PI_DIR="/home/pi/lelamp_runtime"

echo "🚀 同步代码到树莓派..."
echo ""

# 检查是否有未提交的更改
if [ -n "$(git status --porcelain)" ]; then
    echo "📝 检测到未提交的更改，正在提交..."
    git add -A
    git commit -m "sync: auto-commit before push to Pi"
fi

# 推送到GitHub（树莓派可以从GitHub拉取）
echo "📦 推送到GitHub..."
git push origin main

# 在树莓派上从GitHub拉取
echo "⬇️  在树莓派上拉取更新..."
ssh "$PI_HOST" "cd $PI_DIR && git pull origin main"

# 更新工作目录中的文件（如果需要）
echo "🔄 更新工作目录..."
ssh "$PI_HOST" "cd $PI_DIR && git checkout HEAD -- ."

# 询问是否重启API服务
echo ""
read -p "是否要重启API服务？(y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🔄 重启API服务..."
    ssh "$PI_HOST" "pkill -f 'uvicorn lelamp.api.app:app'"
    sleep 2
    ssh "$PI_HOST" "cd $PI_DIR && source .venv/bin/activate && nohup python -m uvicorn lelamp.api.app:app --host 0.0.0.0 --port 8000 > api_server.log 2>&1 &"
    sleep 3

    if ssh "$PI_HOST" "ps aux | grep -q 'uvicorn lelamp.api.app:app'"; then
        echo "✅ API服务重启成功"
    else
        echo "❌ API服务重启失败"
        ssh "$PI_HOST" "tail -20 $PI_DIR/api_server.log"
    fi
fi

echo ""
echo "🎉 同步完成！"
echo ""
echo "提示："
echo "- 前端使用 Vite，会自动热重载"
echo "- 如果前端没有更新，请刷新浏览器 (Ctrl+F5)"
