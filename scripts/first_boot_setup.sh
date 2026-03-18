#!/bin/bash
#
# LeLamp 首次启动检测脚本
# 检测设备是否需要进入设置模式，并在需要时启动 AP 模式
#
# 安装: sudo cp first_boot_setup.sh /usr/local/bin/lelamp-first-boot
#        sudo chmod +x /usr/local/bin/lelamp-first-boot
# 添加到 systemd: 在 lelamp-api.service 的 ExecStartPre 中调用

set -e

# 配置
STATUS_FILE="/var/lib/lelamp/setup_status.json"
STATE_DIR="/var/lib/lelamp"
LOG_FILE="/var/log/lelamp/setup.log"

# 确保日志目录存在
mkdir -p "$(dirname "$LOG_FILE")"

# 日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "Starting first boot setup check..."

# 确保状态目录存在
mkdir -p "$STATE_DIR"

# 检查是否已完成配置
if [ -f "$STATUS_FILE" ]; then
    # 检查 setup_completed 字段
    if grep -q '"setup_completed": true' "$STATUS_FILE" 2>/dev/null; then
        log "Setup already completed, skipping AP mode"
        exit 0
    fi
fi

# 检查 WiFi 连接状态
# 如果已连接到 WiFi，则认为不需要进入设置模式
if nmcli -t -f ACTIVE,SSID connection show --active | grep -q '^yes:'; then
    CURRENT_SSID=$(nmcli -t -f ACTIVE,SSID connection show --active | grep '^yes:' | cut -d: -f2)
    log "WiFi already connected to: $CURRENT_SSID"

    # 更新状态文件
    cat > "$STATUS_FILE" << EOF
{
  "setup_completed": true,
  "setup_completed_at": "$(date -u +%Y-%m-%dT%H:%M:%S)",
  "wifi_ssid": "$CURRENT_SSID",
  "last_mode": "client",
  "last_updated": "$(date -u +%Y-%m-%dT%H:%M:%S)"
}
EOF

    exit 0
fi

# 未检测到 WiFi 配置，需要进入设置模式
log "No WiFi configuration detected, starting setup mode..."

# 启动 AP 模式
# 这里我们通过调用 API 来启动 AP 模式
# 假设 API 服务将在稍后启动

# 标记需要设置
cat > "$STATUS_FILE" << EOF
{
  "setup_completed": false,
  "setup_completed_at": null,
  "wifi_ssid": null,
  "last_mode": "ap",
  "setup_required_at": "$(date -u +%Y-%m-%dT%H:%M:%S)",
  "last_updated": "$(date -u +%Y-%m-%dT%H:%M:%S)"
}
EOF

log "Setup mode enabled, waiting for API service to start AP mode..."

exit 0
