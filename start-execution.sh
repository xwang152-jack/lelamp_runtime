#!/bin/bash

# LeLamp Runtime 优化执行会话启动脚本

echo "🚀 启动 LeLamp Runtime 优化执行会话"
echo "=================================="
echo ""
echo "📍 当前位置: $(pwd)"
echo "🌿 分支: $(git branch --show-current)"
echo "📝 基础提交: $(git rev-parse --short HEAD)"
echo ""
echo "📋 实施计划: docs/plans/2025-03-19-comprehensive-optimization-plan.md"
echo ""
echo "🎯 执行阶段:"
echo "   Phase 1: 安全基础设施 (1-2 周)"
echo "   Phase 2: 功能完善 (2-3 周)"
echo "   Phase 3: 性能优化 (1-2 周)"
echo ""
echo "✅ 环境检查..."
echo ""

# 检查 Python 版本
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "🐍 Python 版本: $python_version"

# 检查 UV
if command -v uv &> /dev/null; then
    uv_version=$(uv --version)
    echo "📦 UV 版本: $uv_version"
else
    echo "❌ UV 未安装"
    echo "请安装: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# 检查依赖
echo ""
echo "📦 检查依赖..."
if uv run pytest --version &> /dev/null; then
    echo "✅ 测试依赖已安装"
else
    echo "❌ 测试依赖缺失，正在安装..."
    uv sync --extra dev
fi

echo ""
echo "🎉 环境就绪！"
echo ""
echo "📖 开始执行，请告诉 Claude Code:"
echo ""
echo "   '我将执行 LeLamp Runtime 的全面优化计划。"
echo "    请使用 superpowers:executing-plans 技能来执行："
echo "    docs/plans/2025-03-19-comprehensive-optimization-plan.md'"
echo ""
echo "💡 提示: 查看 EXECUTION_SESSION.md 了解详细指令"
echo ""
