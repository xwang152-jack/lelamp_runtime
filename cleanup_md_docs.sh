#!/bin/bash

# LeLamp Markdown 文档清理脚本
# 清理根目录下不必要的临时文档，保留核心文档

echo "================================"
echo "LeLamp Markdown 文档清理"
echo "================================"
echo ""

cd /Users/jackwang/lelamp_runtime

# 创建归档目录
ARCHIVE_DIR=".archive/docs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ARCHIVE_DIR"

echo "📁 归档目录: $ARCHIVE_DIR"
echo ""

# 1. 归档 Phase 3 临时文档（已有完整版本在 docs/）
echo "📄 归档 Phase 3 临时文档（docs/ 下已有完整版本）..."
PHASE3_DOCS=(
    "PHASE3_ACCEPTANCE_CHECKLIST.md"
    "PHASE3_FINAL_SUMMARY.md"
)

for doc in "${PHASE3_DOCS[@]}"; do
    if [ -f "$doc" ]; then
        mv "$doc" "$ARCHIVE_DIR/"
        echo "  ✓ $doc → docs/PHASE3_COMPLETION_REPORT.md（完整版本）"
    fi
done

echo ""

# 2. 归档临时参考文档
echo "📄 归档临时参考文档..."
TEMP_DOCS=(
    "COMMANDS_REFERENCE.md"
    "CLEANUP_SUMMARY.md"
)

for doc in "${TEMP_DOCS[@]}"; do
    if [ -f "$doc" ]; then
        mv "$doc" "$ARCHIVE_DIR/"
        echo "  ✓ $doc（内容已整合到 QUICK_REFERENCE.md）"
    fi
done

echo ""

# 3. 归档空的或占位文件
echo "📄 归档空文件..."
EMPTY_DOCS=(
    "SPONSORS.md"
    "CONTRIBUTORS.md"
)

for doc in "${EMPTY_DOCS[@]}"; do
    if [ -f "$doc" ]; then
        # 检查文件是否为空或只有几行
        if [ $(wc -l < "$doc") -lt 5 ]; then
            mv "$doc" "$ARCHIVE_DIR/"
            echo "  ✓ $doc（空文件或占位文件）"
        fi
    fi
done

echo ""
echo "================================"
echo "✅ 保留的核心文档"
echo "================================"
echo ""
echo "📄 项目文档:"
echo "  ✓ README.md - 项目主文档"
echo "  ✓ CLAUDE.md - Claude Code 项目指南"
echo "  ✓ QUICK_REFERENCE.md - 快速参考"
echo ""
echo "📚 完整文档目录:"
echo "  📁 docs/ - 所有详细文档"
echo "    - USER_GUIDE.md - 用户指南"
echo "    - TESTING_CHECKLIST.md - 测试清单"
echo "    - PHASE3_COMPLETION_REPORT.md - Phase 3 完整报告"
echo "    - COMMERCIAL_APP_ARCHITECTURE.md - 商业化架构"
echo "    - PRODUCT_*.md - 产品相关文档"
echo ""
echo "================================"
echo "✅ 清理完成！"
echo "================================"
echo ""
echo "📋 归档位置: $ARCHIVE_DIR"
echo ""

# 显示归档内容
if [ "$(ls -A $ARCHIVE_DIR 2>/dev/null)" ]; then
    echo "📁 归档内容:"
    ls -lh "$ARCHIVE_DIR"
else
    echo "（无需归档的文件）"
fi

echo ""
echo "💡 提示:"
echo "  - 核心文档已保留在根目录"
echo "  - 完整文档位于 docs/ 目录"
echo "  - 临时文档已安全归档"
echo ""
