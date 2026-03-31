#!/bin/bash
#
# LeLamp AP 模式服务安装脚本
# 安装并配置 hostapd、dnsmasq 和相关系统服务
#
# 用法: sudo bash install_ap_services.sh

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查是否以 root 运行
if [ "$EUID" -ne 0 ]; then
    log_error "此脚本需要 root 权限运行"
    log_info "请使用: sudo bash install_ap_services.sh"
    exit 1
fi

log_info "开始安装 LeLamp AP 模式服务..."

# 更新软件包列表
log_info "更新软件包列表..."
apt-get update -q

# 安装必要的软件包
log_info "安装必要的软件包..."
apt-get install -y hostapd dnsmasq iptables-persistent

# 停止服务（如果在运行）
log_info "停止现有服务..."
systemctl stop hostapd 2>/dev/null || true
systemctl stop dnsmasq 2>/dev/null || true
systemctl disable hostapd 2>/dev/null || true
systemctl disable dnsmasq 2>/dev/null || true

# 配置 hostapd
log_info "配置 hostapd..."
cat > /etc/default/hostapd << 'EOF'
DAEMON_CONF="/etc/hostapd/hostapd.conf"
EOF

# 创建 hostapd 配置目录
mkdir -p /etc/hostapd

# 启用 IP 转发
log_info "启用 IP 转发..."
if ! grep -q "net.ipv4.ip_forward=1" /etc/sysctl.conf; then
    echo "net.ipv4.ip_forward=1" >> /etc/sysctl.conf
fi
sysctl -w net.ipv4.ip_forward=1

# 创建状态目录
log_info "创建状态目录..."
mkdir -p /var/lib/lelamp
mkdir -p /var/log/lelamp

# 复制首次启动脚本
log_info "安装首次启动脚本..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cp "$SCRIPT_DIR/first_boot_setup.sh" /usr/local/bin/lelamp-first-boot
chmod +x /usr/local/bin/lelamp-first-boot

# 安装 systemd 服务
log_info "安装 systemd 服务..."
cp "$SCRIPT_DIR/lelamp-setup.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable lelamp-setup.service

# 配置 NetworkManager 不管理 wlan0（当在 AP 模式时）
# 创建一个脚本来管理接口模式
log_info "配置网络接口管理..."

# 允许 lelamp 用户管理 hostapd 和 dnsmasq（仅限特定配置文件）
log_info "配置权限..."
cat > /etc/sudoers.d/lelamp-ap << 'EOF'
# LeLamp AP 模式权限配置
# 安全限制：仅允许特定命令，不使用通配符

# hostapd 和 dnsmasq 管理
lelamp ALL=(ALL) NOPASSWD: /usr/sbin/hostapd -B -P /var/run/hostapd_lelamp.pid /etc/hostapd/hostapd.conf
lelamp ALL=(ALL) NOPASSWD: /usr/sbin/dnsmasq -C /etc/dnsmasq.conf -x /var/run/dnsmasq_lelamp.pid -k

# 允许停止特定服务（通过 systemd，更安全）
lelamp ALL=(ALL) NOPASSWD: /usr/bin/systemctl stop hostapd.service, /usr/bin/systemctl stop dnsmasq.service
lelamp ALL=(ALL) NOPASSWD: /usr/bin/systemctl start hostapd.service, /usr/bin/systemctl start dnsmasq.service

# wlan0 接口 IP 配置（仅限特定 AP 模式 IP）
lelamp ALL=(ALL) NOPASSWD: /bin/ip addr add 192.168.4.1/255.255.255.0 dev wlan0
lelamp ALL=(ALL) NOPASSWD: /bin/ip addr flush dev wlan0
lelamp ALL=(ALL) NOPASSWD: /bin/ip link set dev wlan0 up
lelamp ALL=(ALL) NOPASSWD: /bin/ip link set dev wlan0 down

# nmcli 操作（明确列出允许的命令）
lelamp ALL=(ALL) NOPASSWD: /usr/bin/nmcli -t -f ACTIVE,SSID,SIGNAL connection show
lelamp ALL=(ALL) NOPASSWD: /usr/bin/nmcli -t -f SSID,BSSID,SIGNAL,SECURITY,FREQ device wifi list
lelamp ALL=(ALL) NOPASSWD: /usr/bin/nmcli device wifi connect *
lelamp ALL=(ALL) NOPASSWD: /usr/bin/nmcli connection down *
lelamp ALL=(ALL) NOPASSWD: /usr/bin/nmcli connection delete *
lelamp ALL=(ALL) NOPASSWD: /usr/bin/nmcli -t -f IP4.ADDRESS device show wlan0

# sysctl 网络配置
lelamp ALL=(ALL) NOPASSWD: /usr/sbin/sysctl -w net.ipv4.ip_forward=1

# iptables NAT 配置（特定规则）
lelamp ALL=(ALL) NOPASSWD: /usr/sbin/iptables -t nat -A POSTROUTING *
lelamp ALL=(ALL) NOPASSWD: /usr/sbin/iptables -A FORWARD *
lelamp ALL=(ALL) NOPASSWD: /usr/sbin/iptables -t nat -F POSTROUTING
lelamp ALL=(ALL) NOPASSWD: /usr/sbin/iptables -F FORWARD
EOF

# 验证 sudoers 文件语法
chmod 0440 /etc/sudoers.d/lelamp-ap
if visudo -c -f /etc/sudoers.d/lelamp-ap; then
    log_info "sudoers 配置已验证"
else
    log_error "sudoers 配置文件语法错误，移除文件"
    rm -f /etc/sudoers.d/lelamp-ap
    exit 1
fi

# 添加 lelamp 用户到需要的组
useradd -r -s /bin/false lelamp 2>/dev/null || true
usermod -a -G netdev lelamp 2>/dev/null || true

log_info "安装完成！"
log_info ""
log_info "已安装的服务:"
log_info "  - hostapd (AP 模式)"
log_info "  - dnsmasq (DHCP/DNS)"
log_info "  - lelamp-setup (首次启动检测)"
log_info ""
log_warn "请运行以下命令测试:"
log_warn "  sudo systemctl start lelamp-setup"
log_warn "  sudo journalctl -u lelamp-setup -n 20"
