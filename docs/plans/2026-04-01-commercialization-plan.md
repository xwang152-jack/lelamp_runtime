# 商业化部署 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现出厂预配置脚本、Setup Wizard 注册/登录步骤、自动设备绑定后端，打通"开箱即用"用户旅程。

**Architecture:** 后端新增 `/api/auth/auto-bind` 端点，后端自动从 `setup_status.json` 读取 device_id 和 device_secret 完成绑定。前端 SetupWizard 从 5 步扩展为 6 步，在 WiFi 连接成功后加入注册/登录+自动绑定步骤。出厂脚本 `scripts/factory/prepare_factory_env.sh` 一次性生成所有密钥并写入 .env。

**Tech Stack:** Python/FastAPI (后端), Vue 3 + Element Plus + TypeScript (前端), Bash (部署脚本)

**Design doc:** `docs/plans/2026-04-01-commercialization-design.md`

---

### Task 1: 后端 — 新增 auto-bind 服务方法

**Files:**
- Modify: `lelamp/api/services/auth_service.py:131-166`

**Step 1: 在 AuthService 中新增 `auto_bind_device` 静态方法**

在 `auth_service.py` 的 `AuthService` 类中，`bind_device` 方法之后，新增：

```python
@staticmethod
def auto_bind_device(db: Session, user_id: int) -> DeviceBinding:
    """
    自动绑定当前设备（首次配置流程使用）

    自动从 setup_status.json 读取 device_id 和 device_secret，
    无需用户手动输入。已绑定时返回现有绑定。
    """
    import json
    from pathlib import Path

    # 读取 setup_status.json 获取 device_secret
    status_file = Path("/var/lib/lelamp/setup_status.json")
    device_secret = None
    if status_file.exists():
        try:
            data = json.loads(status_file.read_text())
            device_secret = data.get("device_secret")
        except Exception:
            pass

    # 也检查环境变量
    if not device_secret:
        device_secret = os.getenv("LELAMP_DEVICE_SECRET")

    if not device_secret:
        raise ValueError("设备密钥未配置，无法自动绑定")

    # 获取 device_id
    from lelamp.config import load_config
    config = load_config()
    device_id = config.lamp_id

    # 检查是否已绑定（已绑定时直接返回现有绑定，不报错）
    existing = db.query(DeviceBinding).filter(
        DeviceBinding.user_id == user_id,
        DeviceBinding.device_id == device_id
    ).first()
    if existing:
        return existing

    # 创建新绑定
    binding = DeviceBinding(
        user_id=user_id,
        device_id=device_id,
        device_secret="",
        permission_level="admin"
    )
    db.add(binding)
    db.commit()
    db.refresh(binding)
    return binding
```

**Step 2: Commit**

```bash
git add lelamp/api/services/auth_service.py
git commit -m "feat(api): add auto_bind_device service method"
```

---

### Task 2: 后端 — 新增 auto-bind API 端点

**Files:**
- Modify: `lelamp/api/routes/auth.py:198` (在 `bind_device` 之后)

**Step 1: 在 `auth.py` 路由文件中新增 `/auto-bind` 端点**

在 `bind_device` 端点之后、`refresh_token` 端点之前，新增：

```python
@router.post("/auto-bind", response_model=DeviceBindResponse)
async def auto_bind_device(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    自动绑定当前设备

    首次配置流程使用。后端自动从 setup_status.json 读取
    device_id 和 device_secret，用户登录后无需手动输入。
    """
    payload = AuthService.verify_token(token, "access")
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

    user_id = payload.get("user_id")
    try:
        binding = AuthService.auto_bind_device(db, user_id=user_id)

        logger.info(f"自动绑定设备 {binding.device_id} 到用户 {user_id}")
        return DeviceBindResponse(
            device_id=binding.device_id,
            permission_level=binding.permission_level,
            bound_at=binding.bound_at.isoformat()
        )

    except ValueError as e:
        logger.warning(f"自动绑定失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"自动绑定错误: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )
```

**Step 2: Commit**

```bash
git add lelamp/api/routes/auth.py
git commit -m "feat(api): add /auto-bind endpoint for setup wizard"
```

---

### Task 3: 后端 — 改造 GET /api/system/device 无需认证

**Files:**
- Modify: `lelamp/api/routes/system.py:394-442`

**Step 1: 改造 `get_device_info` 端点**

替换现有的 `get_device_info` 函数。关键变化：
- 不再返回 `device_secret`（安全考虑，设计文档要求）
- 新增 `setup_completed` 字段（从 onboarding_manager 读取）
- 新增 `ip_address` 字段
- 新增 `user_bound` 字段（数据库查询，但不需认证）

```python
@router.get("/device")
async def get_device_info(token: str | None = None) -> dict:
    """
    获取设备信息（无需认证，局域网内可用）

    所有用户均可查看设备基本信息。已认证时额外返回绑定状态。
    不返回 device_secret 等敏感信息。
    """
    from lelamp.api.services.auth_service import AuthService
    from lelamp.config import load_config

    config = load_config()

    result = {
        "device_id": config.lamp_id,
        "hostname": socket.gethostname(),
        "model": "LeLamp",
        "version": "0.1.0",
        "setup_completed": False,
    }

    # 读取配置状态
    try:
        setup_status = await onboarding_manager.get_setup_status()
        result["setup_completed"] = setup_status.get("setup_completed", False)
        result["configured_wifi"] = setup_status.get("wifi_ssid")
    except Exception:
        pass

    # 获取 IP 地址
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        result["ip_address"] = s.getsockname()[0]
        s.close()
    except Exception:
        pass

    # 已认证用户：返回个人绑定状态
    if token:
        payload = AuthService.verify_token(token, "access")
        if payload:
            user_id = payload.get("user_id")
            if user_id:
                from lelamp.database.models_auth import DeviceBinding
                from lelamp.database.session import SessionLocal
                db = SessionLocal()
                try:
                    binding = db.query(DeviceBinding).filter(
                        DeviceBinding.user_id == user_id,
                        DeviceBinding.device_id == config.lamp_id,
                    ).first()
                    result["is_bound"] = binding is not None
                    if binding:
                        result["permission_level"] = binding.permission_level
                finally:
                    db.close()

    return result
```

**Step 2: Commit**

```bash
git add lelamp/api/routes/system.py
git commit -m "feat(api): make /device endpoint public, remove secret from response"
```

---

### Task 4: 前端 — 新增 auto-bind API 调用

**Files:**
- Modify: `web/src/api/auth.ts:143-152` (在 `bindDevice` 之后)
- Modify: `web/src/types/auth.ts` (如需新增类型)

**Step 1: 在 `web/src/api/auth.ts` 中新增 `autoBindDevice` 函数**

在 `bindDevice` 函数之后新增：

```typescript
/**
 * 自动绑定当前设备（首次配置流程使用）
 */
export async function autoBindDevice(token: string): Promise<DeviceBindResponse> {
  const response = await fetch(
    `${getApiBase()}/api/auth/auto-bind`,
    getFetchConfig('POST', undefined, token)
  )
  return handleResponse<DeviceBindResponse>(response)
}
```

**Step 2: Commit**

```bash
git add web/src/api/auth.ts
git commit -m "feat(web): add autoBindDevice API call"
```

---

### Task 5: 前端 — Auth Store 新增 autoBindDevice action

**Files:**
- Modify: `web/src/stores/auth.ts`

**Step 1: 在 auth store 中新增 `autoBindDevice` 方法**

找到 `bindDevice` 函数（约 L217），在其后新增：

```typescript
async function autoBindDevice(): Promise<{ success: boolean; error?: string }> {
  if (!accessToken.value) {
    return { success: false, error: '未登录' }
  }
  try {
    await authApi.autoBindDevice(accessToken.value)
    return { success: true }
  } catch (e: any) {
    return { success: false, error: e.message || '自动绑定失败' }
  }
}
```

然后在 return 语句中导出 `autoBindDevice`。

**Step 2: Commit**

```bash
git add web/src/stores/auth.ts
git commit -m "feat(web): add autoBindDevice to auth store"
```

---

### Task 6: 前端 — SetupWizard 扩展为 6 步（含注册/登录）

**Files:**
- Modify: `web/src/views/SetupWizardView.vue`

这是最大的改动。当前 SetupWizard 有 5 步（欢迎→WiFi→密码→连接→完成），需要扩展为 6 步，在"连接成功"和"完成"之间插入"注册/登录"步骤。

**Step 1: 改造步骤定义和模板**

将 `steps` 从 5 步改为 6 步：

```typescript
const steps = ['欢迎', '选择 WiFi', '输入密码', '连接中', '注册 / 登录', '完成']
```

在步骤 4（连接验证）成功后，不再直接跳到完成，而是跳到步骤 5（注册/登录）。

在步骤 5 的模板中，新增注册/登录表单：

```html
<!-- 步骤 5: 注册/登录 -->
<div v-if="currentStep === 4" class="auth-step">
  <h2>创建账号或登录</h2>
  <p class="step-description">绑定设备需要登录账号，首次使用请注册</p>

  <el-tabs v-model="authTab" class="auth-tabs">
    <el-tab-pane label="登录" name="login">
      <el-form label-position="top" @submit.prevent="handleAuthLogin">
        <el-form-item label="用户名">
          <el-input v-model="loginForm.username" placeholder="请输入用户名" size="large" />
        </el-form-item>
        <el-form-item label="密码">
          <el-input v-model="loginForm.password" type="password" placeholder="请输入密码"
                    show-password size="large" @keyup.enter="handleAuthLogin" />
        </el-form-item>
        <el-button type="primary" class="auth-btn" :loading="authLoading" @click="handleAuthLogin">
          登录并绑定设备
        </el-button>
      </el-form>
    </el-tab-pane>

    <el-tab-pane label="注册" name="register">
      <el-form label-position="top" @submit.prevent="handleAuthRegister">
        <el-form-item label="用户名">
          <el-input v-model="registerForm.username" placeholder="3-50 个字符" size="large" />
        </el-form-item>
        <el-form-item label="邮箱">
          <el-input v-model="registerForm.email" type="email" placeholder="your@email.com" size="large" />
        </el-form-item>
        <el-form-item label="密码">
          <el-input v-model="registerForm.password" type="password" placeholder="至少 6 个字符"
                    show-password size="large" />
        </el-form-item>
        <el-button type="primary" class="auth-btn" :loading="authLoading" @click="handleAuthRegister">
          注册并绑定设备
        </el-button>
      </el-form>
    </el-tab-pane>
  </el-tabs>
</div>
```

完成步骤（原步骤 5）改为步骤 6（`currentStep === 5`）。

**Step 2: 新增 auth 相关状态和方法**

```typescript
import { useAuthStore } from '@/stores'

const authStore = useAuthStore()
const authTab = ref('login')
const authLoading = ref(false)

const loginForm = reactive({ username: '', password: '' })
const registerForm = reactive({ username: '', email: '', password: '' })

async function handleAuthLogin() {
  if (!loginForm.username || !loginForm.password) {
    ElMessage.warning('请输入用户名和密码')
    return
  }
  authLoading.value = true
  try {
    await authStore.login(loginForm.username, loginForm.password)
    // 自动绑定设备
    const result = await authStore.autoBindDevice()
    if (result.success) {
      currentStep.value = 5  // 跳到完成步骤
      await completeSetup()
    } else {
      ElMessage.error(result.error || '设备绑定失败')
    }
  } catch (e: any) {
    ElMessage.error(e.message || '登录失败')
  } finally {
    authLoading.value = false
  }
}

async function handleAuthRegister() {
  if (!registerForm.username || !registerForm.email || !registerForm.password) {
    ElMessage.warning('请填写所有字段')
    return
  }
  authLoading.value = true
  try {
    await authStore.register(registerForm.username, registerForm.email, registerForm.password)
    // 注册后自动绑定设备
    const result = await authStore.autoBindDevice()
    if (result.success) {
      currentStep.value = 5
      await completeSetup()
    } else {
      ElMessage.error(result.error || '设备绑定失败')
    }
  } catch (e: any) {
    ElMessage.error(e.message || '注册失败')
  } finally {
    authLoading.value = false
  }
}
```

**Step 3: 修改 WiFi 连接成功后的跳转**

在 `handleConnect` 函数中，WiFi 连接成功后不再跳到 `currentStep = 4`（完成），而是跳到 `currentStep = 4`（注册/登录，即新的步骤 5，因为索引从 0 开始）。

将原来 `completeSetup` 的调用移到 auth 成功后执行。

**Step 4: 修改完成步骤的模板**

将 `v-if="currentStep === 4"` 改为 `v-if="currentStep === 5"`。

完成步骤增加显示设备访问地址：

```html
<div v-if="currentStep === 5" class="complete-step">
  <div class="success-icon">✓</div>
  <h2>配置完成！</h2>
  <p>LeLamp 即将重启并连接到您的 WiFi</p>
  <div class="access-info">
    <p>设备访问地址：</p>
    <p class="access-url">http://lelamp.local:8000</p>
  </div>
  <div class="countdown">
    <span>{{ countdown }}</span> 秒后重启...
  </div>
</div>
```

**Step 5: 修改底部导航的步骤判断**

- `currentStep < 3` 保持不变（欢迎、WiFi、密码）
- `currentStep === 2` 保持不变（连接按钮在密码步骤）
- 连接成功后进入步骤 4（auth），auth 完成后自动进入步骤 5（完成），不需要底部导航按钮

**Step 6: Commit**

```bash
git add web/src/views/SetupWizardView.vue
git commit -m "feat(web): extend setup wizard with auth step for device binding"
```

---

### Task 7: 前端 — 完成步骤增加设备信息显示

**Files:**
- Modify: `web/src/views/SetupWizardView.vue`

**Step 1: 在组件加载时获取设备信息**

```typescript
const deviceInfo = ref<{ device_id: string; hostname: string } | null>(null)

onMounted(async () => {
  handleScan()
  try {
    const response = await axios.get(`${API_BASE}/api/system/device`)
    deviceInfo.value = response.data
  } catch {
    // ignore
  }
})
```

**Step 2: 在欢迎步骤和完成步骤中展示设备信息**

在欢迎页面的 `p.hint` 之后加入：

```html
<div v-if="deviceInfo" class="device-info-badge">
  <span class="badge-label">设备</span>
  <span class="badge-value">{{ deviceInfo.hostname || deviceInfo.device_id }}</span>
</div>
```

**Step 3: Commit**

```bash
git add web/src/views/SetupWizardView.vue
git commit -m "feat(web): show device info in setup wizard"
```

---

### Task 8: 出厂预配置脚本

**Files:**
- Create: `scripts/factory/prepare_factory_env.sh`

**Step 1: 创建出厂预配置脚本**

```bash
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
```

**Step 2: Commit**

```bash
mkdir -p scripts/factory
git add scripts/factory/prepare_factory_env.sh
git commit -m "feat(factory): add prepare_factory_env.sh for out-of-box setup"
```

---

### Task 9: 改造 first_boot_setup.sh 使用动态 AP 密码

**Files:**
- Modify: `scripts/setup/first_boot_setup.sh:100-117`

**Step 1: 改造 AP 启动逻辑，从 setup_status.json 读取动态密码**

替换 `first_boot_setup.sh` 中的 AP 启动部分（约 L100-117），改为从 `setup_status.json` 读取 AP 密码，并传递给 AP 服务：

```bash
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
```

**Step 2: Commit**

```bash
git add scripts/setup/first_boot_setup.sh
git commit -m "feat(setup): use dynamic AP password from factory config"
```

---

### Task 10: 构建前端并验证

**Step 1: 构建前端**

```bash
cd web && pnpm install && pnpm build
```

**Step 2: 验证后端启动无报错**

```bash
cd /Users/jackwang/lelamp_runtime
uv run python -c "from lelamp.api.routes.auth import router; print('auth routes OK')"
uv run python -c "from lelamp.api.routes.system import router; print('system routes OK')"
uv run python -c "from lelamp.api.services.auth_service import AuthService; print('AuthService OK')"
```

**Step 3: Commit (如有修复)**

```bash
git add -A
git commit -m "fix: address any build or import issues"
```

---

## 任务依赖关系

```
Task 1 (auto-bind service) → Task 2 (auto-bind endpoint)
Task 2 → Task 4 (frontend API) → Task 5 (auth store) → Task 6 (setup wizard)
Task 3 (device endpoint) 独立
Task 7 (device info display) 依赖 Task 6
Task 8 (factory script) 独立
Task 9 (first_boot) 独立
Task 10 (build & verify) 依赖所有前置任务
```

## 并行执行建议

可并行执行：
- **组 A**: Task 1 → 2 → 4 → 5 → 6 → 7（后端 + 前端主链路）
- **组 B**: Task 3（device endpoint 改造，独立）
- **组 C**: Task 8 → 9（脚本，独立）
- **最后**: Task 10（统一验证）
