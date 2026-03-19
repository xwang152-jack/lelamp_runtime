#!/bin/bash
#
# LeLamp Captive Portal 安装脚本
# 安装必要的系统依赖并配置服务
#

set -e

PI_HOST="${PI_HOST:-pi@192.168.0.104}"

echo "================================================"
echo "LeLamp Captive Portal 安装"
echo "================================================"
echo ""

# 检查是否可以连接到树莓派
echo "🔍 检查树莓派连接..."
if ! ssh $PI_HOST "echo '连接成功'" 2>/dev/null; then
    echo "❌ 无法连接到树莓派 $PI_HOST"
    exit 1
fi
echo "✅ 树莓派连接正常"
echo ""

# 1. 安装系统依赖
echo "📦 1. 安装系统依赖..."
ssh $PI_HOST 'sudo apt-get update'
ssh $PI_HOST 'sudo apt-get install -y hostapd dnsmasq iw wireless-tools'
echo "✅ 系统依赖安装完成"
echo ""

# 2. 配置 hostapd
echo "🔧 2. 配置 hostapd..."
ssh $PI_HOST 'sudo systemctl unmask hostapd'
ssh $PI_HOST 'sudo systemctl enable hostapd'
echo "✅ hostapd 配置完成"
echo ""

# 3. 创建辅助脚本
echo "📜 3. 创建辅助脚本..."

# AP 启动脚本
ssh $PI_HOST 'sudo tee /usr/local/bin/lelamp-start-ap > /dev/null' << 'ENDSCRIPT'
#!/bin/bash
# LeLamp AP 模式启动脚本

set -e

LOG_FILE="/var/log/lelamp/ap.log"
mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "Starting AP mode..."

# 停止现有的 WiFi 连接
log "Stopping existing WiFi connections..."
nmcli device disconnect wlan0 2>/dev/null || true

# 配置静态 IP
log "Configuring static IP..."
ip addr flush dev wlan0 2>/dev/null || true
ip addr add 192.168.4.1/24 dev wlan0

# 启用 IP 转发
echo 1 > /proc/sys/net/ipv4/ip_forward

# 启动 hostapd
log "Starting hostapd..."
hostapd -B /etc/hostapd.conf 2>&1 | tee -a "$LOG_FILE" || true

# 启动 dnsmasq
log "Starting dnsmasq..."
systemctl restart dnsmasq

log "AP mode started successfully"
ENDSCRIPT

ssh $PI_HOST 'sudo chmod +x /usr/local/bin/lelamp-start-ap'

# AP 停止脚本
ssh $PI_HOST 'sudo tee /usr/local/bin/lelamp-stop-ap > /dev/null' << 'ENDSCRIPT'
#!/bin/bash
# LeLamp AP 模式停止脚本

set -e

LOG_FILE="/var/log/lelamp/ap.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG_FILE"
}

log "Stopping AP mode..."

# 停止 hostapd
pkill hostapd 2>/dev/null || true

# 停止 dnsmasq
systemctl stop dnsmasq 2>/dev/null || true

# 清除 IP 配置
ip addr flush dev wlan0 2>/dev/null || true

log "AP mode stopped"
ENDSCRIPT

ssh $PI_HOST 'sudo chmod +x /usr/local/bin/lelamp-stop-ap'
echo "✅ 辅助脚本创建完成"
echo ""

# 4. 创建 hostapd 配置
echo "📝 4. 创建 hostapd 配置..."
ssh $PI_HOST 'sudo tee /etc/hostapd.conf > /dev/null' << 'EOF'
# LeLamp Hostapd Configuration
interface=wlan0
driver=nl80211
ssid=LeLamp-Setup
hw_mode=g
channel=6
wmm_enabled=0
macaddr_acl=0
auth_algs=1
ignore_broadcast_ssid=0
wpa=2
wpa_passphrase=lelamp123
wpa_key_mgmt=WPA-PSK
rsn_pairwise=CCMP
EOF
echo "✅ hostapd 配置创建完成"
echo ""

# 5. 部署 systemd 服务文件
echo "📜 5. 部署 systemd 服务..."
cat scripts/lelamp-setup-ap.service | ssh $PI_HOST 'sudo tee /etc/systemd/system/lelamp-setup-ap.service > /dev/null'
cat scripts/lelamp-captive-portal.service | ssh $PI_HOST 'sudo tee /etc/systemd/system/lelamp-captive-portal.service > /dev/null'
echo "✅ 服务文件部署完成"
echo ""

# 6. 创建状态目录
echo "📁 6. 创建状态目录..."
ssh $PI_HOST 'sudo mkdir -p /var/lib/lelamp /var/log/lelamp'
ssh $PI_HOST 'sudo chmod 755 /var/lib/lelamp /var/log/lelamp'
echo "✅ 状态目录创建完成"
echo ""

# 7. 重新加载 systemd
echo "🔄 7. 重新加载 systemd..."
ssh $PI_HOST 'sudo systemctl daemon-reload'
echo "✅ systemd 已重新加载"
echo ""

# 8. 启用服务
echo "🚀 8. 启用开机自启..."
ssh $PI_HOST 'sudo systemctl enable lelamp-setup-ap.service'
ssh $PI_HOST 'sudo systemctl enable lelamp-captive-portal.service'
echo "✅ 服务已设置为开机自启"
echo ""

echo ""
echo "================================================"
echo "✅ Captive Portal 安装完成！"
echo "================================================"
echo ""
echo "🎯 管理命令："
echo ""
echo "启动 AP 模式:"
echo "  ssh $PI_HOST 'sudo systemctl start lelamp-setup-ap'"
echo ""
echo "启动 Portal 服务:"
echo "  ssh $PI_HOST 'sudo systemctl start lelamp-captive-portal'"
echo ""
echo "查看服务状态:"
echo "  ssh $PI_HOST 'sudo systemctl status lelamp-captive-portal'"
echo ""
echo "查看日志:"
echo "  ssh $PI_HOST 'sudo journalctl -u lelamp-captive-portal -f'"
echo ""
echo "测试 WiFi 扫描:"
echo "  ssh $PI_HOST 'nmcli device wifi list'"
echo ""
echo "⚠️  注意：首次设置时需要清除状态文件才能进入设置模式："
echo "  ssh $PI_HOST 'sudo rm /var/lib/lelamp/setup_status.json'"
echo ""
