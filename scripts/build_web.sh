#!/usr/bin/env bash
# 构建 Vue 前端并验证产物
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WEB_DIR="$PROJECT_ROOT/web"

echo "==> 构建 Vue 前端..."
cd "$WEB_DIR"
pnpm build

if [ -f "$WEB_DIR/dist/index.html" ]; then
    echo "==> 构建成功: $WEB_DIR/dist/index.html"
else
    echo "==> 错误: 构建产物 $WEB_DIR/dist/index.html 不存在"
    exit 1
fi
