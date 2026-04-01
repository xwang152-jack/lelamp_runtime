#!/bin/bash
#
# LeLamp 首次启动检测脚本
# 检测设备是否需要进入设置模式，并在需要时启动 Captive Portal
#
# 安装: sudo cp first_boot_setup.sh /usr/local/bin/lelamp-first-boot
#        sudo chmod +x /usr/local/bin/lelamp-first-boot
# 调用: 在 systemd 服务启动前执行此脚本

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
    # 使用 Python 解析 JSON（更可靠）
    if command -v python3 &> /dev/null; then
        SETUP_COMPLETED=$(python3 -c "
import json
try:
    with open('$STATUS_FILE', 'r') as f:
        data = json.load(f)
        print('true' if data.get('setup_completed', False) else 'false')
except:
    print('false')
" 2>/dev/null || echo "false")

        if [ "$SETUP_COMPLETED" = "true" ]; then
            log "Setup already completed, skipping AP mode"
            exit 0
        fi
    else
        # 回退到 grep（不推荐）
        if grep -q '"setup_completed": true' "$STATUS_FILE" 2>/dev/null; then
            log "Setup already completed, skipping AP mode"
            exit 0
        fi
    fi
fi

# 检查 WiFi 连接状态
# 如果已连接到 WiFi，则标记为已完成设置
if nmcli -t -f ACTIVE,SSID connection show --active | grep -q '^yes:'; then
    CURRENT_SSID=$(nmcli -t -f ACTIVE,SSID connection show --active | grep '^yes:' | cut -d: -f2)
    CURRENT_IP=$(hostname -I | awk '{print $1}')
    log "WiFi already connected to: $CURRENT_SSID (IP: $CURRENT_IP)"

    # 更新状态文件
    cat > "$STATUS_FILE" << EOF
{
  "setup_completed": true,
  "setup_completed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "wifi_ssid": "$CURRENT_SSID",
  "last_ip_address": "$CURRENT_IP",
  "last_mode": "client",
  "last_updated": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

    log "Setup marked as completed"
    exit 0
fi

# 未检测到 WiFi 配置，需要进入设置模式
log "No WiFi configuration detected, entering setup mode..."

# 更新状态文件
cat > "$STATUS_FILE" << EOF
{
  "setup_completed": false,
  "setup_started_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "setup_completed_at": null,
  "wifi_ssid": null,
  "last_ip_address": null,
  "current_step": "welcome",
  "connection_attempts": 0,
  "error_message": null,
  "last_mode": "ap",
  "ap_mode_count": 1,
  "last_updated": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

# 从 setup_status.json 读取 AP 密码
AP_PASSWORD=$(python3 -c "
import json
try:
    with open('$STATUS_FILE', 'r') as f:
        data = json.load(f)
    print(data.get('ap_password', 'lelamp123'))
except:
    print('lelamp123')
" 2>/dev/null || echo "lelamp123")

# 更新 hostapd 配置中的密码
if [ -f "/etc/hostapd.conf" ]; then
    sed -i "s/^wpa_passphrase=.*/wpa_passphrase=$AP_PASSWORD/" /etc/hostapd.conf
fi

# 启动 AP 模式服务
log "Starting AP mode service (password from factory config)..."
if systemctl start lelamp-setup-ap.service 2>/dev/null; then
    log "AP mode service started (password: $AP_PASSWORD)"
else
    log "Warning: Failed to start AP mode service (may not be installed)"
fi

# 启动 Captive Portal 服务
log "Starting Captive Portal service..."
if systemctl start lelamp-captive-portal.service 2>/dev/null; then
    log "Captive Portal service started"
else
    log "Warning: Failed to start Captive Portal service (may not be installed)"
fi

log "Setup mode activated. User should connect to 'LeLamp-Setup' hotspot."
log "Access the setup portal at http://192.168.4.1:8080"

exit 0
