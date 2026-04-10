#!/bin/bash
#
# 同步代码到树莓派（通过 rsync 直推）
#
# 用法: bash scripts/tools/sync_to_pi.sh [PI_HOST]
# 示例: bash scripts/tools/sync_to_pi.sh pi@192.168.3.129
#

set -e

PI_HOST="${1:-pi@192.168.3.129}"
PI_DIR="/home/pi/lelamp_runtime"

echo "=== 同步代码到树莓派 ==="
echo "目标: $PI_HOST:$PI_DIR"
echo ""

# 1. 测试 SSH 连接
echo "[1/4] 检查 SSH 连接..."
if ! ssh -o ConnectTimeout=5 "$PI_HOST" "echo ok" > /dev/null 2>&1; then
    echo "错误: 无法连接 $PI_HOST"
    exit 1
fi
echo "  SSH 连接正常"

# 1.5 清理残留的 main.py 进程（防止手动调试遗留进程抢占资源）
echo "[1.5/4] 清理残留进程..."
ssh "$PI_HOST" "
    # 找出所有 main.py 的 python3 进程，排除 systemd 管理的进程
    PIDS=\$(ps aux | grep 'main.py' | grep -v grep | grep -v 'tmux\|systemd' | awk '{print \$2}')
    if [ -n \"\$PIDS\" ]; then
        echo \"  终止残留进程: \$PIDS\"
        sudo kill -9 \$PIDS 2>/dev/null || true
        sleep 1
    else
        echo '  无残留进程'
    fi
"

# 2. rsync 同步文件
echo "[2/4] 同步文件..."
rsync -avz \
    --exclude='.git/' \
    --exclude='.venv/' \
    --exclude='__pycache__/' \
    --exclude='.env' \
    --exclude='.DS_Store' \
    --exclude='*.pyc' \
    --exclude='.coverage' \
    --exclude='htmlcov/' \
    --exclude='.pytest_cache/' \
    --exclude='.ruff_cache/' \
    --exclude='lelamp.db' \
    --exclude='*.egg-info/' \
    --exclude='node_modules/' \
    --exclude='web/node_modules/' \
    --exclude='web/dist/' \
    --exclude='.archive/' \
    --exclude='*.tar.gz' \
    /Users/jackwang/lelamp_runtime/ \
    "$PI_HOST:$PI_DIR/"
echo "  文件同步完成"

# 3. 同步 Python 依赖
echo "[3/4] 同步 Python 依赖..."
ssh "$PI_HOST" "cd $PI_DIR && uv sync --extra api 2>&1"
echo "  依赖同步完成"

# 4. 重启服务
echo "[4/4] 重启服务..."
ssh "$PI_HOST" "sudo systemctl restart lelamp-api.service lelamp-captive-portal.service 2>&1"
sleep 3

# 验证
if ssh "$PI_HOST" "systemctl is-active --quiet lelamp-api.service"; then
    echo "  API 服务启动成功"
else
    echo "  警告: API 服务启动失败，查看日志:"
    ssh "$PI_HOST" "journalctl -u lelamp-api.service --no-pager -n 5"
fi

if ssh "$PI_HOST" "systemctl is-active --quiet lelamp-captive-portal.service"; then
    echo "  Captive Portal 启动成功"
else
    echo "  警告: Captive Portal 启动失败，查看日志:"
    ssh "$PI_HOST" "journalctl -u lelamp-captive-portal.service --no-pager -n 5"
fi

echo ""
echo "=== 同步完成 ==="
echo "访问: http://${PI_HOST#*@}:8000"
