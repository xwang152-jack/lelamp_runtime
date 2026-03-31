#!/bin/bash
set -e

# 检查是否安装了 Nuitka
if ! python3 -m nuitka --version &> /dev/null; then
    echo "Error: Nuitka 未安装。请运行: pip install nuitka ordered-set"
    exit 1
fi

echo ">>> 开始编译 LeLamp Runtime (main.py)..."
echo ">>> 注意：首次编译可能需要下载 C 编译器 (ccache/gcc) 并花费较长时间。"

# 清理旧构建
rm -rf dist build

# Nuitka 编译配置
# --standalone: 生成包含 Python 运行时的独立目录 (无需目标机器安装 Python)
# --include-package=lelamp: 强制包含项目源码包
# --follow-imports: 自动分析第三方依赖
# --show-progress: 显示进度
# --output-dir=dist: 输出目录

python3 -m nuitka \
    --standalone \
    --follow-imports \
    --include-package=lelamp \
    --include-package-data=lelamp \
    --output-dir=dist \
    --output-filename=lelamp-runtime \
    main.py

echo ">>> 编译完成！"
echo ">>> 可执行程序位于: dist/main.dist/lelamp-runtime"
echo ">>> 您可以将 dist/main.dist 目录打包分发到目标机器运行。"
