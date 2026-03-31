#!/bin/bash
#
# 启动LeLamp API服务器（需要sudo权限用于GPIO控制）
#

cd /home/pi/lelamp_runtime || exit 1

# 检查虚拟环境是否存在
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✓ 虚拟环境已激活"
else
    echo "❌ 错误: 虚拟环境 .venv 不存在"
    echo "请先运行: uv sync"
    exit 1
fi

echo "启动LeLamp API服务器（需要sudo权限以访问GPIO）..."
echo "提示: RGB LED控制需要root权限访问GPIO内存"
echo ""

# 使用 sudo 运行 uvicorn
# 注意: 需要 sudo -E 来保留环境变量(特别是 PYTHONPATH)
sudo -E uv run uvicorn lelamp.api.app:app --host 0.0.0.0 --port 8000 --log-level info
