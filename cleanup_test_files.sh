#!/bin/bash

# LeLamp 测试文件清理脚本
# 清理测试过程中创建的临时文档和脚本

echo "================================"
echo "LeLamp 测试文件清理"
echo "================================"
echo ""

cd /Users/jackwang/lelamp_runtime

# 创建归档目录
ARCHIVE_DIR=".archive/test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ARCHIVE_DIR"

echo "📁 归档目录: $ARCHIVE_DIR"
echo ""

# 1. 归档测试相关的 Markdown 文档
echo "📄 归档测试文档..."
TEST_DOCS=(
    "ALL_FIXED.md"
    "CAMERA_GUIDE.md"
    "CAMERA_QUICKSTART.md"
    "FIXED_LICENSE_ISSUE.md"
    "FRONTEND_TEST_SUMMARY.md"
    "LICENSE_SETUP.md"
    "MACOS_LIMITATION.md"
    "TEST_GUIDE.md"
    "TEST_RESULTS_PHASE3.md"
    "TESTING_QUICKSTART.md"
)

for doc in "${TEST_DOCS[@]}"; do
    if [ -f "$doc" ]; then
        mv "$doc" "$ARCHIVE_DIR/"
        echo "  ✓ $doc"
    fi
done

echo ""

# 2. 归档测试脚本
echo "🔧 归档测试脚本..."
TEST_SCRIPTS=(
    "fix_env.sh"
    "quick_test.sh"
    "test_frontend.sh"
    "test_phase3.sh"
)

for script in "${TEST_SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        mv "$script" "$ARCHIVE_DIR/"
        echo "  ✓ $script"
    fi
done

echo ""

# 3. 保留有用的脚本
echo "✅ 保留的有用脚本:"
echo "  ✓ quick_start.sh - Token 生成工具"
echo "  ✓ test_e2e.sh - 端到端测试脚本"
echo ""

# 4. 清理临时文件
echo "🗑️  清理临时文件..."
TEMP_FILES=(
    "/tmp/lelamp_quick_connect.txt"
)

for temp_file in "${TEMP_FILES[@]}"; do
    if [ -f "$temp_file" ]; then
        rm -f "$temp_file"
        echo "  ✓ $temp_file"
    fi
done

echo ""
echo "================================"
echo "✅ 清理完成！"
echo "================================"
echo ""
echo "📋 归档位置: $ARCHIVE_DIR"
echo ""
echo "📁 归档内容:"
ls -lh "$ARCHIVE_DIR"
echo ""
echo "💡 提示: 如需恢复任何文件，可以从归档目录中复制"
echo ""
echo "🔧 下一步:"
echo "  - 开发前端: cd web && pnpm dev"
echo "  - 生成 Token: ./quick_start.sh"
echo "  - 部署到 Pi: 参考 docs/USER_GUIDE.md"
echo ""
