#!/bin/bash

# LeLamp Runtime 优化执行会话启动脚本

echo "🚀 启动 LeLamp Runtime 优化执行会话"
echo "=========================================="
echo ""

# 检查当前目录
current_dir=$(pwd)
expected_dir="/Users/jackwang/lelamp_runtime/.claude/worktrees/optimization-implementation"

if [ "$current_dir" != "$expected_dir" ]; then
    echo "⚠️  当前不在执行目录"
    echo "📍 当前目录: $current_dir"
    echo "🎯 目标目录: $expected_dir"
    echo ""
    echo "正在切换到执行目录..."
    cd "$expected_dir" || exit 1
    echo "✅ 已切换到执行目录"
    echo ""
fi

# 检查 git 状态
echo "🌿 Git 状态:"
git branch --show-current
git rev-parse --short HEAD
echo ""

# 检查关键文件
echo "📋 检查执行文件..."
if [ -f "EXECUTION_SESSION.md" ]; then
    echo "✅ EXECUTION_SESSION.md 存在"
else
    echo "❌ EXECUTION_SESSION.md 缺失"
fi

if [ -f "START_PROMPT.md" ]; then
    echo "✅ START_PROMPT.md 存在"
else
    echo "❌ START_PROMPT.md 缺失"
fi

if [ -f "docs/plans/2025-03-19-comprehensive-optimization-plan.md" ]; then
    echo "✅ 实施计划存在"
else
    echo "❌ 实施计划缺失"
fi

echo ""
echo "🎉 执行环境就绪！"
echo ""

# 显示启动指令
echo "📖 新会话启动指令:"
echo "===================="
echo ""
cat "START_PROMPT.md"
echo ""

# 提供下一步操作建议
echo "🚀 下一步操作:"
echo "============="
echo ""
echo "1. 保持当前终端打开（用于监控进度）"
echo "2. 打开新的终端窗口"
echo "3. 在新终端中运行: cd $expected_dir"
echo "4. 启动新的 Claude Code 会话"
echo "5. 粘贴上面的启动指令"
echo ""

# 提供监控命令
echo "📊 监控命令 (在本会话中使用):"
echo "================================"
echo ""
echo "# 查看执行分支进度"
echo "git log origin/optimization-implementation --oneline --since='1 hour ago'"
echo ""
echo "# 查看文件变更"
echo "git diff main..origin/optimization-implementation --stat"
echo ""
echo "# 查看具体代码变更"
echo "git diff main..origin/optimization-implementation -- lelamp/database/ lelamp/api/"
echo ""

# 提供问题排查
echo "🆘 问题排查:"
echo "=========="
echo ""
echo "如果新会话遇到问题："
echo "1. 查看 EXECUTION_SESSION.md 了解详细指令"
echo "2. 查看实施计划中的 Task 说明"
echo "3. 回到主会话寻求帮助"
echo ""

echo "✨ 准备就绪！请在新会话中开始执行优化计划。"
echo ""
