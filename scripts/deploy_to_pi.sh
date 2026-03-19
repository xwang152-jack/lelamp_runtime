#!/bin/bash
# 完整部署脚本：构建前端 + 推送代码 + 设置自动启动

set -e

echo "================================================"
echo "LeLamp 完整部署工具"
echo "================================================"
echo ""

# 1. 构建前端
echo "🏗️  1. 构建前端..."
if [ ! -d "web/node_modules" ]; then
    echo "   安装前端依赖..."
    cd web
    npm install
    cd ..
fi

echo "   构建前端静态文件..."
cd web
npm run build
cd ..

echo "✅ 前端构建完成"
echo ""

# 2. 提交代码
echo "📝 2. 提交代码变更..."
if [ -n "$(git status --porcelain)" ]; then
    git add .
    echo "请输入提交信息（留空使用默认）:"
    read -r commit_msg
    if [ -z "$commit_msg" ]; then
        commit_msg="chore: 部署更新 $(date +%Y%m%d_%H%M%S)"
    fi
    git commit -m "$commit_msg"
    git push origin main
    echo "✅ 代码已提交并推送"
else
    echo "ℹ️  没有需要提交的更改"
fi

echo ""

# 3. 同步到树莓派
echo "📡 3. 同步代码到树莓派..."
if [ -x "./scripts/push_to_pi.sh" ]; then
    ./scripts/push_to_pi.sh
else
    echo "❌ 找不到推送脚本，请确保 scripts/push_to_pi.sh 存在"
    exit 1
fi

echo ""

# 4. 部署前端文件
echo "📦 4. 部署前端文件到树莓派..."
PI_HOST="pi@192.168.0.104"
PROJECT_DIR="~/lelamp_runtime"

ssh $PI_HOST "mkdir -p $PROJECT_DIR/web/dist"
scp -r web/dist/* $PI_HOST:$PROJECT_DIR/web/dist/
echo "✅ 前端文件已部署"

echo ""

# 5. 询问是否设置自动启动
echo "================================================"
echo "部署完成！"
echo "================================================"
echo ""
echo "是否要设置开机自动启动？"
echo ""
echo "1. 是 - 设置自动启动"
echo "2. 否 - 手动启动"
echo ""
read -p "请选择 (1/2): " auto_start

case $auto_start in
  1)
    echo ""
    bash "$(dirname "$0")/setup_lelamp_startup.sh"
    ;;
  *)
    echo ""
    echo "跳过自动启动设置"
    echo ""
    echo "手动启动命令:"
    echo "  ssh $PI_HOST 'cd $PROJECT_DIR && sudo -E uv run python -m uvicorn lelamp.api.app:app --host 0.0.0.0 --port 8000'"
    ;;
esac

echo ""
echo "================================================"
echo "🎉 部署完成！"
echo "================================================"
echo ""
echo "🌐 访问地址："
echo "   http://192.168.0.104:8000"
echo ""
