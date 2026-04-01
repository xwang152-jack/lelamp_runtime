#!/bin/bash
#
# LeLamp 出厂预配置脚本
#
# 在树莓派上执行，一次性完成所有出厂配置：
# - 生成 device_id / device_secret / license_key / jwt_secret
# - 写入 .env 和 setup_status.json
# - 安装并启用 systemd 服务
#
# 使用方法:
#   sudo bash scripts/factory/prepare_factory_env.sh
#

set -e

# 配置
PROJECT_DIR="${PROJECT_DIR:-/opt/lelamp}"
ENV_FILE="${PROJECT_DIR}/.env"
STATUS_FILE="/var/lib/lelamp/setup_status.json"
STATE_DIR="/var/lib/lelamp"
LOG_FILE="/var/log/lelamp/factory_setup.log"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR] $*${NC}" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[OK] $*${NC}" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[WARN] $*${NC}" | tee -a "$LOG_FILE"
}

# 确保日志目录
mkdir -p "$(dirname "$LOG_FILE")" "$STATE_DIR"

log "=========================================="
log "LeLamp 出厂预配置"
log "=========================================="

# 检查项目目录
if [ ! -d "$PROJECT_DIR" ]; then
    error "项目目录不存在: $PROJECT_DIR"
    exit 1
fi

cd "$PROJECT_DIR"

# 检查 Python 环境
if ! command -v uv &> /dev/null; then
    error "uv 未安装，请先安装 uv"
    exit 1
fi

# ============================================================================
# 1. 生成密钥
# ============================================================================

log ""
log "--- 步骤 1: 生成设备密钥 ---"

# 获取 device_id
DEVICE_ID=$(uv run python -c "from lelamp.utils.security import get_device_id; print(get_device_id())")
success "设备 ID: $DEVICE_ID"

# 生成 device_secret
DEVICE_SECRET=$(python3 -c "import secrets; print(secrets.token_hex(8))")
success "设备密钥: $DEVICE_SECRET"

# 生成 JWT_SECRET
JWT_SECRET=$(python3 -c "import secrets; print(secrets.token_hex(32))")
success "JWT 密钥: ${JWT_SECRET:0:16}..."

# 生成 AP 热点密码
AP_PASSWORD=$(python3 -c "import secrets; print(secrets.token_urlsafe(8))")
success "AP 热点密码: $AP_PASSWORD"

# ============================================================================
# 2. 检查/生成 License
# ============================================================================

log ""
log "--- 步骤 2: License 配置 ---"

# 检查是否已有 LICENSE_SECRET
LICENSE_SECRET=$(grep "^LELAMP_LICENSE_SECRET=" "$ENV_FILE" 2>/dev/null | cut -d= -f2 || true)

if [ -z "$LICENSE_SECRET" ]; then
    warn "LELAMP_LICENSE_SECRET 未设置"
    warn "请先运行: uv run python scripts/tools/generate_license.py --generate-secret"
    warn "然后将密钥添加到 .env 文件后重新运行此脚本"
    exit 1
fi

success "License 密钥已配置: ${LICENSE_SECRET:0:16}..."

# 生成 License Key
LICENSE_KEY=$(uv run python -c "
from lelamp.utils.security import generate_license_key
import os
os.environ.setdefault('LELAMP_LICENSE_SECRET', '$LICENSE_SECRET')
print(generate_license_key('$DEVICE_ID', '$LICENSE_SECRET'))
")
success "License Key: $LICENSE_KEY"

# ============================================================================
# 3. 写入 .env 文件
# ============================================================================

log ""
log "--- 步骤 3: 写入 .env ---"

# 确保基础 .env 存在
if [ ! -f "$ENV_FILE" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example "$ENV_FILE"
        warn "从 .env.example 创建 .env，请手动填写 API 密钥"
    else
        touch "$ENV_FILE"
    fi
fi

# 更新/添加配置项
update_env() {
    local key="$1" value="$2"
    if grep -q "^${key}=" "$ENV_FILE" 2>/dev/null; then
        sed -i "s|^${key}=.*|${key}=${value}|" "$ENV_FILE"
    else
        echo "${key}=${value}" >> "$ENV_FILE"
    fi
}

update_env "LELAMP_DEVICE_SECRET" "$DEVICE_SECRET"
update_env "LELAMP_LICENSE_KEY" "$LICENSE_KEY"
update_env "LELAMP_JWT_SECRET" "$JWT_SECRET"
update_env "LELAMP_DEV_MODE" "0"
update_env "LELAMP_AP_PASSWORD" "$AP_PASSWORD"

success ".env 已更新"

# ============================================================================
# 4. 初始化 setup_status.json
# ============================================================================

log ""
log "--- 步骤 4: 初始化配置状态 ---"

cat > "$STATUS_FILE" << EOF
{
  "setup_completed": false,
  "setup_completed_at": null,
  "wifi_ssid": null,
  "device_id": "$DEVICE_ID",
  "device_secret": "$DEVICE_SECRET",
  "ap_password": "$AP_PASSWORD",
  "factory_configured_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "last_mode": "ap",
  "last_updated": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

success "setup_status.json 已初始化"

# ============================================================================
# 5. 安装 systemd 服务
# ============================================================================

log ""
log "--- 步骤 5: 安装 systemd 服务 ---"

# 安装 Captive Portal 相关服务
if [ -f "scripts/services/install_captive_portal.sh" ]; then
    # 不直接运行远程安装脚本（需要 PI_HOST），而是本地安装
    log "安装 hostapd 和 dnsmasq..."
    apt-get update -qq
    apt-get install -y -qq hostapd dnsmasq iw wireless-tools 2>/dev/null || true

    log "部署 systemd 服务文件..."
    mkdir -p /etc/systemd/system
    for svc in scripts/services/*.service; do
        if [ -f "$svc" ]; then
            cp "$svc" /etc/systemd/system/
            log "  已部署: $(basename $svc)"
        fi
    done

    systemctl daemon-reload
    systemctl enable lelamp-setup-ap.service 2>/dev/null || warn "lelamp-setup-ap.service 未找到"
    systemctl enable lelamp-captive-portal.service 2>/dev/null || warn "lelamp-captive-portal.service 未找到"
    success "systemd 服务已安装"
else
    warn "install_captive_portal.sh 不存在，跳过服务安装"
fi

# ============================================================================
# 6. 安装 LeLamp API 服务
# ============================================================================

log ""
log "--- 步骤 6: 安装 LeLamp API 服务 ---"

# 安装 first_boot 脚本
if [ -f "scripts/setup/first_boot_setup.sh" ]; then
    cp scripts/setup/first_boot_setup.sh /usr/local/bin/lelamp-first-boot
    chmod +x /usr/local/bin/lelamp-first-boot
    success "first_boot_setup.sh 已安装"
fi

# ============================================================================
# 完成
# ============================================================================

log ""
log "=========================================="
success "出厂预配置完成！"
log "=========================================="
log ""
log "设备信息摘要："
log "  设备 ID:     $DEVICE_ID"
log "  设备密钥:    $DEVICE_SECRET"
log "  AP 密码:     $AP_PASSWORD"
log "  License Key: $LICENSE_KEY"
log ""
warn "请将以上信息记录到出厂登记表"
warn "首次开机将自动进入 AP 设置模式"
log ""
