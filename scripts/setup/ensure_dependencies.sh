#!/bin/bash
# 确保 LeLamp Runtime 所有依赖都已安装

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "================================================"
echo "LeLamp Runtime - 依赖检查和安装"
echo "================================================"
echo ""

# 检测运行环境
if [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM="macos"
    EXTRAS="api dev"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PLATFORM="linux"
    EXTRAS="hardware api dev vision"
else
    echo "❌ 未知平台: $OSTYPE"
    exit 1
fi

echo "🔍 检测到平台: $PLATFORM"
echo "📦 将安装的 extras: $EXTRAS"
echo ""

# 检查 uv 是否安装
if ! command -v uv &> /dev/null; then
    echo "❌ uv 未安装，请先安装: https://docs.astral.sh/uv/"
    exit 1
fi

echo "✅ uv 已安装"
echo ""

# 检查是否在正确的目录
if [[ ! -f "$PROJECT_DIR/pyproject.toml" ]]; then
    echo "❌ 未找到 pyproject.toml，请在项目根目录运行此脚本"
    exit 1
fi

echo "✅ 项目目录正确: $PROJECT_DIR"
echo ""

# 构建 uv sync 命令
SYNC_CMD="uv sync"
for extra in $EXTRAS; do
    SYNC_CMD="$SYNC_CMD --extra $extra"
done

echo "🔧 执行命令: $SYNC_CMD"
echo ""

# 执行依赖安装
cd "$PROJECT_DIR"
eval $SYNC_CMD

if [[ $? -eq 0 ]]; then
    echo ""
    echo "================================================"
    echo "✅ 依赖安装完成！"
    echo "================================================"
    echo ""
    echo "🚀 启动命令："
    echo "  # 主程序 (console 模式)"
    echo "  sudo uv run main.py console"
    echo ""
    echo "  # API 服务器"
    echo "  sudo uv run python -m uvicorn lelamp.api.app:app --host 0.0.0.0 --port 8000"
    echo ""
else
    echo ""
    echo "================================================"
    echo "❌ 依赖安装失败"
    echo "================================================"
    exit 1
fi
