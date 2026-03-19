<template>
  <div class="profile-view">
    <!-- Header -->
    <header class="profile-header">
      <div class="header-content">
        <button class="back-btn" @click="handleBack">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M19 12H5M12 19l-7-7 7-7" />
          </svg>
          <span>返回</span>
        </button>
        <h1 class="header-title">个人中心</h1>
        <div class="header-spacer"></div>
      </div>
    </header>

    <!-- Main Content -->
    <main class="profile-main">
      <!-- Profile Card -->
      <div class="profile-card">
        <div class="card-decoration"></div>

        <div class="profile-header">
          <div class="profile-avatar">
            <span class="avatar-text">{{ userInitial }}</span>
          </div>
          <div class="profile-info">
            <h2 class="profile-name">{{ authStore.user?.username }}</h2>
            <p class="profile-email">{{ authStore.user?.email }}</p>
          </div>
          <div class="profile-badges">
            <el-tag v-if="authStore.isAdmin" type="warning">
              <span class="badge-icon">👑</span>
              管理员
            </el-tag>
            <el-tag v-else type="info">免费用户</el-tag>
          </div>
        </div>

        <!-- Stats -->
        <div class="profile-stats">
          <div class="stat-item">
            <span class="stat-icon">📅</span>
            <div class="stat-info">
              <span class="stat-label">注册时间</span>
              <span class="stat-value">{{ formatDate(authStore.user?.created_at) }}</span>
            </div>
          </div>
          <div class="stat-item">
            <span class="stat-icon">📱</span>
            <div class="stat-info">
              <span class="stat-label">绑定设备</span>
              <span class="stat-value">{{ deviceCount }} 台</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Account Settings -->
      <div class="settings-section">
        <h3 class="section-title">账号设置</h3>

        <div class="settings-list">
          <div class="setting-item" @click="showEmailDialog = true">
            <div class="setting-left">
              <span class="setting-icon">📧</span>
              <div class="setting-info">
                <span class="setting-label">邮箱地址</span>
                <span class="setting-desc">{{ authStore.user?.email }}</span>
              </div>
            </div>
            <svg class="setting-arrow" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M9 18l6-6-6-6" />
            </svg>
          </div>

          <div class="setting-item" @click="showPasswordDialog = true">
            <div class="setting-left">
              <span class="setting-icon">🔒</span>
              <div class="setting-info">
                <span class="setting-label">修改密码</span>
                <span class="setting-desc">定期修改密码以保护账号安全</span>
              </div>
            </div>
            <svg class="setting-arrow" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M9 18l6-6-6-6" />
            </svg>
          </div>
        </div>
      </div>

      <!-- Linked Devices -->
      <div class="settings-section">
        <h3 class="section-title">
          <span>已绑定设备</span>
          <el-button link type="primary" @click="goToDevices">
            管理
            <svg class="link-arrow" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M9 18l6-6-6-6" />
            </svg>
          </el-button>
        </h3>

        <div v-if="loadingDevices" class="devices-loading">
          <el-skeleton :rows="2" animated />
        </div>

        <div v-else-if="devices.length === 0" class="devices-empty">
          <span class="empty-icon">📱</span>
          <p>还没有绑定设备</p>
          <el-button type="primary" @click="goToDevices">绑定设备</el-button>
        </div>

        <div v-else class="devices-list">
          <div v-for="device in devices" :key="device.device_id" class="device-item">
            <span class="device-icon">🪔</span>
            <div class="device-info">
              <span class="device-name">{{ device.device_id }}</span>
              <span class="device-permission">{{ getPermissionLabel(device.permission_level) }}</span>
            </div>
            <el-tag size="small" type="success">已绑定</el-tag>
          </div>
        </div>
      </div>

      <!-- Logout Button -->
      <div class="logout-section">
        <el-button
          type="danger"
          plain
          class="logout-btn"
          @click="handleLogout"
        >
          <span class="btn-icon">🚪</span>
          退出登录
        </el-button>
      </div>
    </main>

    <!-- Email Dialog -->
    <el-dialog
      v-model="showEmailDialog"
      title="修改邮箱"
      width="90%"
      :style="{ maxWidth: '400px' }"
    >
      <el-form :model="emailForm" label-position="top">
        <el-form-item label="新邮箱地址">
          <el-input v-model="emailForm.email" type="email" placeholder="请输入新邮箱" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showEmailDialog = false">取消</el-button>
        <el-button type="primary" :loading="loading" @click="handleUpdateEmail">
          确认修改
        </el-button>
      </template>
    </el-dialog>

    <!-- Password Dialog -->
    <el-dialog
      v-model="showPasswordDialog"
      title="修改密码"
      width="90%"
      :style="{ maxWidth: '400px' }"
    >
      <el-form :model="passwordForm" :rules="passwordRules" ref="passwordFormRef" label-position="top">
        <el-form-item label="当前密码" prop="current_password">
          <el-input v-model="passwordForm.current_password" type="password" show-password />
        </el-form-item>
        <el-form-item label="新密码" prop="new_password">
          <el-input v-model="passwordForm.new_password" type="password" show-password />
        </el-form-item>
        <el-form-item label="确认新密码" prop="confirm_password">
          <el-input v-model="passwordForm.confirm_password" type="password" show-password />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showPasswordDialog = false">取消</el-button>
        <el-button type="primary" :loading="loading" @click="handleUpdatePassword">
          确认修改
        </el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox, type FormInstance, type FormRules } from 'element-plus'
import { useAuthStore } from '@/stores'
import type { DeviceBindResponse } from '@/types/auth'
import { getPermissionLabel, formatDate, getUserInitial } from '@/utils/device'

const router = useRouter()
const authStore = useAuthStore()

const showEmailDialog = ref(false)
const showPasswordDialog = ref(false)
const loading = ref(false)
const loadingDevices = ref(false)
const devices = ref<DeviceBindResponse[]>([])
const passwordFormRef = ref<FormInstance>()

const emailForm = reactive({ email: authStore.user?.email || '' })

const passwordForm = reactive({
  current_password: '',
  new_password: '',
  confirm_password: ''
})

const validateConfirmPassword = (_rule: any, value: string, callback: any) => {
  if (value === '') {
    callback(new Error('请再次输入新密码'))
  } else if (value !== passwordForm.new_password) {
    callback(new Error('两次输入的密码不一致'))
  } else {
    callback()
  }
}

const passwordRules: FormRules = {
  current_password: [
    { required: true, message: '请输入当前密码', trigger: 'blur' }
  ],
  new_password: [
    { required: true, message: '请输入新密码', trigger: 'blur' },
    { min: 6, message: '密码长度至少为 6 个字符', trigger: 'blur' }
  ],
  confirm_password: [
    { required: true, message: '请确认新密码', trigger: 'blur' },
    { validator: validateConfirmPassword, trigger: 'blur' }
  ]
}

const userInitial = computed(() => {
  return getUserInitial(authStore.user?.username)
})

const deviceCount = computed(() => devices.value.length)

async function loadDevices() {
  loadingDevices.value = true
  try {
    devices.value = await authStore.getUserDevices()
  } catch (e) {
    console.error('Failed to load devices:', e)
  } finally {
    loadingDevices.value = false
  }
}

async function handleUpdateEmail() {
  if (!emailForm.email) {
    ElMessage.warning('请输入邮箱地址')
    return
  }

  loading.value = true
  try {
    const result = await authStore.updateProfile({ email: emailForm.email })
    if (result.success) {
      ElMessage.success('邮箱修改成功')
      showEmailDialog.value = false
    } else {
      ElMessage.error(result.error || '修改失败')
    }
  } catch (e) {
    ElMessage.error('修改失败，请稍后重试')
  } finally {
    loading.value = false
  }
}

async function handleUpdatePassword() {
  if (!passwordFormRef.value) return

  await passwordFormRef.value.validate(async (valid) => {
    if (!valid) return

    loading.value = true
    try {
      const result = await authStore.updateProfile({
        current_password: passwordForm.current_password,
        new_password: passwordForm.new_password
      })
      if (result.success) {
        ElMessage.success('密码修改成功，请重新登录')
        showPasswordDialog.value = false
        authStore.logout()
        router.push('/auth')
      } else {
        ElMessage.error(result.error || '修改失败')
      }
    } catch (e) {
      ElMessage.error('修改失败，请稍后重试')
    } finally {
      loading.value = false
    }
  })
}

async function handleLogout() {
  try {
    await ElMessageBox.confirm(
      '确定要退出登录吗？',
      '退出登录',
      {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )

    authStore.logout()
    ElMessage.success('已退出登录')
    router.push('/connect')
  } catch {
    // User cancelled
  }
}

function handleBack() {
  router.back()
}

function goToDevices() {
  router.push('/devices')
}

onMounted(() => {
  loadDevices()
})
</script>

<style lang="scss" scoped>
.profile-view {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  background: linear-gradient(180deg, var(--lelamp-bg-cream) 0%, #FFE8D6 100%);
}

/* === Header === */
.profile-header {
  position: sticky;
  top: 0;
  z-index: 10;
  background: rgba(255, 255, 255, 0.8);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid rgba(0, 0, 0, 0.05);
}

.header-content {
  display: flex;
  align-items: center;
  gap: var(--lelamp-space-md);
  padding: var(--lelamp-space-md) var(--lelamp-space-lg);
  max-width: 600px;
  margin: 0 auto;
}

.back-btn {
  display: flex;
  align-items: center;
  gap: var(--lelamp-space-xs);
  padding: var(--lelamp-space-sm) var(--lelamp-space-md);
  background: var(--lelamp-bg-gray);
  border: none;
  border-radius: var(--lelamp-radius-full);
  font-weight: 600;
  font-size: 0.875rem;
  color: var(--lelamp-text-secondary);
  cursor: pointer;
  transition: all var(--lelamp-transition-bounce);

  svg {
    width: 18px;
    height: 18px;
  }

  &:hover {
    background: var(--lelamp-peach-light);
    color: var(--lelamp-peach-dark);
    transform: translateX(-3px);
  }
}

.header-title {
  font-family: var(--lelamp-font-display);
  font-size: 1.25rem;
  font-weight: 700;
  color: var(--lelamp-text-primary);
  margin: 0;
}

.header-spacer {
  width: 80px;
}

/* === Main Content === */
.profile-main {
  flex: 1;
  max-width: 600px;
  width: 100%;
  margin: 0 auto;
  padding: var(--lelamp-space-lg);
  display: flex;
  flex-direction: column;
  gap: var(--lelamp-space-lg);
}

/* === Profile Card === */
.profile-card {
  background: var(--lelamp-bg-white);
  border-radius: var(--lelamp-radius-xl);
  padding: var(--lelamp-space-xl);
  box-shadow: var(--lelamp-shadow-md);
  position: relative;
  overflow: hidden;
}

.card-decoration {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 4px;
  background: linear-gradient(90deg, var(--lelamp-peach), var(--lelamp-sunny), var(--lelamp-mint), var(--lelamp-sky));
}

.profile-header {
  display: flex;
  align-items: center;
  gap: var(--lelamp-space-md);
  margin-bottom: var(--lelamp-space-lg);
}

.profile-avatar {
  width: 72px;
  height: 72px;
  border-radius: 50%;
  background: linear-gradient(135deg, var(--lelamp-peach) 0%, var(--lelamp-coral) 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 4px 12px rgba(255, 107, 138, 0.3);
}

.avatar-text {
  font-size: 2rem;
  font-weight: 700;
  color: var(--lelamp-bg-white);
}

.profile-info {
  flex: 1;
}

.profile-name {
  font-size: 1.25rem;
  font-weight: 700;
  color: var(--lelamp-text-primary);
  margin: 0 0 var(--lelamp-space-xs);
}

.profile-email {
  font-size: 0.875rem;
  color: var(--lelamp-text-secondary);
  margin: 0;
}

.profile-badges {
  display: flex;
  gap: var(--lelamp-space-xs);
}

.badge-icon {
  margin-right: var(--lelamp-space-xs);
}

/* === Stats === */
.profile-stats {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--lelamp-space-md);
}

.stat-item {
  display: flex;
  align-items: center;
  gap: var(--lelamp-space-md);
  padding: var(--lelamp-space-md);
  background: var(--lelamp-bg-gray);
  border-radius: var(--lelamp-radius-lg);
}

.stat-icon {
  font-size: 1.5rem;
}

.stat-info {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.stat-label {
  font-size: 0.75rem;
  color: var(--lelamp-text-secondary);
}

.stat-value {
  font-size: 0.938rem;
  font-weight: 600;
  color: var(--lelamp-text-primary);
}

/* === Settings Section === */
.settings-section {
  background: var(--lelamp-bg-white);
  border-radius: var(--lelamp-radius-xl);
  padding: var(--lelamp-space-lg);
  box-shadow: var(--lelamp-shadow-sm);
}

.section-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 1rem;
  font-weight: 700;
  color: var(--lelamp-text-primary);
  margin: 0 0 var(--lelamp-space-md);
}

.link-arrow {
  width: 16px;
  height: 16px;
  margin-left: var(--lelamp-space-xs);
}

.settings-list {
  display: flex;
  flex-direction: column;
}

.setting-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: var(--lelamp-space-md);
  border-radius: var(--lelamp-radius-md);
  cursor: pointer;
  transition: background var(--lelamp-transition-normal);

  &:hover {
    background: var(--lelamp-bg-gray);
  }

  &:not(:last-child) {
    margin-bottom: var(--lelamp-space-sm);
  }
}

.setting-left {
  display: flex;
  align-items: center;
  gap: var(--lelamp-space-md);
}

.setting-icon {
  font-size: 1.5rem;
}

.setting-info {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.setting-label {
  font-size: 0.938rem;
  font-weight: 600;
  color: var(--lelamp-text-primary);
}

.setting-desc {
  font-size: 0.813rem;
  color: var(--lelamp-text-secondary);
}

.setting-arrow {
  width: 20px;
  height: 20px;
  color: var(--lelamp-text-tertiary);
}

/* === Devices List === */
.devices-loading,
.devices-empty {
  padding: var(--lelamp-space-xl);
  text-align: center;
}

.empty-icon {
  display: block;
  font-size: 3rem;
  margin-bottom: var(--lelamp-space-md);
}

.devices-empty p {
  font-size: 0.875rem;
  color: var(--lelamp-text-secondary);
  margin: 0 0 var(--lelamp-space-md);
}

.devices-list {
  display: flex;
  flex-direction: column;
  gap: var(--lelamp-space-sm);
}

.device-item {
  display: flex;
  align-items: center;
  gap: var(--lelamp-space-md);
  padding: var(--lelamp-space-md);
  background: var(--lelamp-bg-gray);
  border-radius: var(--lelamp-radius-md);
}

.device-icon {
  font-size: 1.5rem;
}

.device-info {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.device-name {
  font-size: 0.938rem;
  font-weight: 600;
  color: var(--lelamp-text-primary);
}

.device-permission {
  font-size: 0.75rem;
  color: var(--lelamp-text-secondary);
}

/* === Logout Section === */
.logout-section {
  padding: var(--lelamp-space-lg) 0;
}

.logout-btn {
  width: 100%;
  height: 48px;
  font-size: 0.938rem;
  font-weight: 600;
  border-radius: var(--lelamp-radius-lg);
  border-color: var(--lelamp-coral);
  color: var(--lelamp-coral);

  &:hover {
    background: rgba(255, 107, 138, 0.1);
  }
}

.btn-icon {
  margin-right: var(--lelamp-space-xs);
  font-size: 1rem;
}

@media (max-width: 480px) {
  .profile-main {
    padding: var(--lelamp-space-md);
  }

  .header-spacer {
    display: none;
  }

  .back-btn span {
    display: none;
  }

  .profile-avatar {
    width: 56px;
    height: 56px;
  }

  .avatar-text {
    font-size: 1.5rem;
  }

  .profile-stats {
    grid-template-columns: 1fr;
  }
}
</style>
