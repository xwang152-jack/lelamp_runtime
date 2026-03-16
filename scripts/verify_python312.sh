#!/bin/bash
# Python 3.12 安装后验证和配置脚本

set -e

echo "======================================"
echo "Python 3.12 安装验证"
echo "======================================"

# 检查 Python 3.12 是否安装成功
if command -v python3.12 &> /dev/null; then
    echo "✓ Python 3.12 已安装"
    python3.12 --version
else
    echo "✗ Python 3.12 未找到"
    echo "请运行: brew install python@3.12"
    exit 1
fi

# 检查 pip3.12
if command -v pip3.12 &> /dev/null; then
    echo "✓ pip3.12 可用"
    pip3.12 --version
else
    echo "✗ pip3.12 未找到"
fi

echo ""
echo "======================================"
echo "配置 Python 3.12 为默认版本（可选）"
echo "======================================"

echo "如果你想将 Python 3.12 设置为默认 python3，可以运行："
echo ""
echo "  # 添加到 ~/.zshrc 或 ~/.bash_profile"
echo "  export PATH=\"/opt/homebrew/opt/python@3.12/bin:\$PATH\""
echo ""
echo "  # 或创建别名"
echo "  alias python3=python3.12"
echo "  alias pip3=pip3.12"
echo ""

echo "======================================"
echo "UV 配置"
echo "======================================"

# 检查 UV 是否使用 Python 3.12
echo "当前 UV 配置:"
uv python list | grep "3.12" || echo "UV 将自动下载 Python 3.12"

echo ""
echo "======================================"
echo "验证项目依赖"
echo "======================================"

cd "$(dirname "$0")"
echo "使用 UV 同步依赖（Python 3.12）..."
uv sync --python 3.12 || {
    echo "依赖同步失败，请检查 pyproject.toml"
    exit 1
}

echo ""
echo "✓ 所有检查通过！"
echo ""
echo "现在可以使用以下命令运行项目："
echo "  uv run python main.py console"
