<template>
  <div class="device-manage-view">
    <!-- Header -->
    <header class="device-header">
      <div class="header-content">
        <button class="back-btn" @click="handleBack">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M19 12H5M12 19l-7-7 7-7" />
          </svg>
          <span>返回</span>
        </button>
        <h1 class="header-title">设备管理</h1>
        <div class="header-spacer"></div>
      </div>
    </header>

    <!-- Main Content -->
    <main class="device-main">
      <!-- Bind New Device Card -->
      <div class="bind-card">
        <div class="card-header">
          <span class="card-icon">🔗</span>
          <div class="card-info">
            <h3 class="card-title">绑定新设备</h3>
            <p class="card-desc">输入设备信息以绑定到您的账户</p>
          </div>
        </div>

        <el-form
          ref="bindFormRef"
          :model="bindForm"
          :rules="bindRules"
          label-position="top"
          size="large"
        >
          <el-form-item prop="device_id">
            <el-input
              v-model="bindForm.device_id"
              placeholder="设备 ID (例如: lelamp)"
              :prefix-icon="Lock"
            />
          </el-form-item>

          <el-form-item prop="device_secret">
            <el-input
              v-model="bindForm.device_secret"
              type="password"
              placeholder="设备密钥"
              :prefix-icon="Lock"
              show-password
            />
          </el-form-item>

          <el-button
            type="primary"
            class="bind-btn"
            :loading="binding"
            @click="handleBind"
          >
            绑定设备
          </el-button>
        </el-form>
      </div>

      <!-- Devices List -->
      <div class="devices-section">
        <div class="section-header">
          <h3 class="section-title">已绑定设备</h3>
          <el-button
            link
            type="primary"
            :loading="loadingDevices"
            @click="loadDevices"
          >
            <svg class="refresh-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M23 4v6h-6M1 20v-6h6" />
              <path d="M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15" />
            </svg>
            刷新
          </el-button>
        </div>

        <!-- Loading State -->
        <div v-if="loadingDevices && devices.length === 0" class="devices-loading">
          <el-skeleton :rows="3" animated />
        </div>

        <!-- Empty State -->
        <div v-else-if="devices.length === 0" class="devices-empty">
          <span class="empty-icon">📱</span>
          <p class="empty-text">还没有绑定任何设备</p>
          <p class="empty-hint">在上方输入设备信息进行绑定</p>
        </div>

        <!-- Devices List -->
        <div v-else class="devices-list">
          <div
            v-for="device in devices"
            :key="device.device_id"
            class="device-item"
          >
            <div class="device-left">
              <div class="device-avatar">
                <span class="avatar-icon">🪔</span>
              </div>
              <div class="device-info">
                <h4 class="device-name">{{ device.device_id }}</h4>
                <div class="device-meta">
                  <span class="device-permission">
                    {{ getPermissionLabel(device.permission_level) }}
                  </span>
                  <span class="device-bound">
                    绑定于 {{ formatDate(device.bound_at) }}
                  </span>
                </div>
              </div>
            </div>
            <el-button
              type="danger"
              size="small"
              plain
              @click="handleUnbind(device.device_id)"
            >
              解绑
            </el-button>
          </div>
        </div>
      </div>

      <!-- Help Section -->
      <div class="help-section">
        <div class="help-card">
          <span class="help-icon">💡</span>
          <div class="help-content">
            <h4 class="help-title">如何获取设备信息？</h4>
            <ul class="help-list">
              <li>设备 ID 通常是设备的唯一标识符</li>
              <li>设备密钥可以在设备的设置页面找到</li>
              <li>如果是本地设备，可以使用默认 ID: lelamp</li>
            </ul>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox, type FormInstance, type FormRules } from 'element-plus'
import { Lock } from '@element-plus/icons-vue'
import { useAuthStore } from '@/stores'
import type { DeviceBindResponse } from '@/types/auth'
import { getPermissionLabel, formatDate } from '@/utils/device'

const router = useRouter()
const authStore = useAuthStore()

const bindFormRef = ref<FormInstance>()
const binding = ref(false)
const loadingDevices = ref(false)
const devices = ref<DeviceBindResponse[]>([])

const bindForm = reactive({
  device_id: '',
  device_secret: ''
})

const bindRules: FormRules = {
  device_id: [
    { required: true, message: '请输入设备 ID', trigger: 'blur' },
    { min: 2, max: 50, message: '设备 ID 长度应为 2-50 个字符', trigger: 'blur' }
  ],
  device_secret: [
    { required: true, message: '请输入设备密钥', trigger: 'blur' }
  ]
}

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

async function handleBind() {
  if (!bindFormRef.value) return

  await bindFormRef.value.validate(async (valid) => {
    if (!valid) return

    binding.value = true
    try {
      const result = await authStore.bindDevice(
        bindForm.device_id,
        bindForm.device_secret
      )
      if (result.success) {
        ElMessage.success('设备绑定成功')
        bindForm.device_id = ''
        bindForm.device_secret = ''
        bindFormRef.value?.resetFields()
        await loadDevices()
      } else {
        ElMessage.error(result.error || '设备绑定失败')
      }
    } catch (e) {
      ElMessage.error('设备绑定失败，请检查设备信息是否正确')
    } finally {
      binding.value = false
    }
  })
}

async function handleUnbind(deviceId: string) {
  try {
    await ElMessageBox.confirm(
      `确定要解绑设备 "${deviceId}" 吗？解绑后将无法控制该设备。`,
      '解绑设备',
      {
        confirmButtonText: '确定解绑',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )

    const result = await authStore.unbindDevice(deviceId)
    if (result.success) {
      ElMessage.success('设备已解绑')
      await loadDevices()
    } else {
      ElMessage.error(result.error || '解绑失败')
    }
  } catch (e) {
    // User cancelled
  }
}

function handleBack() {
  router.back()
}

onMounted(() => {
  loadDevices()
})
</script>

<style lang="scss" scoped>
.device-manage-view {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  background: linear-gradient(180deg, var(--lelamp-bg-cream) 0%, #FFE8D6 100%);
}

/* === Header === */
.device-header {
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
.device-main {
  flex: 1;
  max-width: 600px;
  width: 100%;
  margin: 0 auto;
  padding: var(--lelamp-space-lg);
  display: flex;
  flex-direction: column;
  gap: var(--lelamp-space-lg);
}

/* === Bind Card === */
.bind-card {
  background: var(--lelamp-bg-white);
  border-radius: var(--lelamp-radius-xl);
  padding: var(--lelamp-space-lg);
  box-shadow: var(--lelamp-shadow-md);
}

.card-header {
  display: flex;
  align-items: center;
  gap: var(--lelamp-space-md);
  margin-bottom: var(--lelamp-space-lg);
}

.card-icon {
  font-size: 2rem;
}

.card-info {
  flex: 1;
}

.card-title {
  font-size: 1.125rem;
  font-weight: 700;
  color: var(--lelamp-text-primary);
  margin: 0 0 var(--lelamp-space-xs);
}

.card-desc {
  font-size: 0.875rem;
  color: var(--lelamp-text-secondary);
  margin: 0;
}

:deep(.el-form-item) {
  margin-bottom: var(--lelamp-space-md);
}

:deep(.el-input__wrapper) {
  border-radius: var(--lelamp-radius-lg);
  padding: var(--lelamp-space-md) var(--lelamp-space-lg);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);

  &:hover {
    box-shadow: 0 2px 12px rgba(255, 154, 118, 0.15);
  }

  &.is-focus {
    box-shadow: 0 0 0 3px rgba(255, 154, 118, 0.2);
  }
}

.bind-btn {
  width: 100%;
  height: 48px;
  font-size: 1rem;
  font-weight: 700;
  border-radius: var(--lelamp-radius-lg);
  background: linear-gradient(135deg, var(--lelamp-peach) 0%, var(--lelamp-coral) 100%);
  border: none;
  box-shadow: 0 4px 16px rgba(255, 107, 138, 0.3);
  margin-top: var(--lelamp-space-sm);

  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(255, 107, 138, 0.4);
  }

  &:active {
    transform: translateY(0);
  }
}

/* === Devices Section === */
.devices-section {
  background: var(--lelamp-bg-white);
  border-radius: var(--lelamp-radius-xl);
  padding: var(--lelamp-space-lg);
  box-shadow: var(--lelamp-shadow-sm);
}

.section-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: var(--lelamp-space-md);
}

.section-title {
  font-size: 1rem;
  font-weight: 700;
  color: var(--lelamp-text-primary);
  margin: 0;
}

.refresh-icon {
  width: 16px;
  height: 16px;
  margin-right: var(--lelamp-space-xs);
}

.devices-loading,
.devices-empty {
  padding: var(--lelamp-space-xl);
  text-align: center;
}

.empty-icon {
  display: block;
  font-size: 3rem;
  margin-bottom: var(--lelamp-space-md);
  opacity: 0.5;
}

.empty-text {
  font-size: 1rem;
  font-weight: 600;
  color: var(--lelamp-text-secondary);
  margin: 0 0 var(--lelamp-space-xs);
}

.empty-hint {
  font-size: 0.875rem;
  color: var(--lelamp-text-tertiary);
  margin: 0;
}

.devices-list {
  display: flex;
  flex-direction: column;
  gap: var(--lelamp-space-sm);
}

.device-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: var(--lelamp-space-md);
  background: var(--lelamp-bg-gray);
  border-radius: var(--lelamp-radius-lg);
  transition: all var(--lelamp-transition-normal);

  &:hover {
    background: rgba(255, 154, 118, 0.1);
  }
}

.device-left {
  display: flex;
  align-items: center;
  gap: var(--lelamp-space-md);
}

.device-avatar {
  width: 48px;
  height: 48px;
  border-radius: var(--lelamp-radius-lg);
  background: linear-gradient(135deg, var(--lelamp-peach) 0%, var(--lelamp-sunny) 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 2px 8px rgba(255, 154, 118, 0.2);
}

.avatar-icon {
  font-size: 1.5rem;
}

.device-info {
  display: flex;
  flex-direction: column;
  gap: var(--lelamp-space-xs);
}

.device-name {
  font-size: 1rem;
  font-weight: 700;
  color: var(--lelamp-text-primary);
  margin: 0;
}

.device-meta {
  display: flex;
  align-items: center;
  gap: var(--lelamp-space-sm);
}

.device-permission {
  font-size: 0.75rem;
  padding: 2px var(--lelamp-space-xs);
  background: rgba(255, 154, 118, 0.2);
  color: var(--lelamp-peach-dark);
  border-radius: var(--lelamp-radius-sm);
  font-weight: 600;
}

.device-bound {
  font-size: 0.75rem;
  color: var(--lelamp-text-tertiary);
}

/* === Help Section === */
.help-section {
  margin-top: var(--lelamp-space-md);
}

.help-card {
  display: flex;
  gap: var(--lelamp-space-md);
  padding: var(--lelamp-space-lg);
  background: linear-gradient(135deg, rgba(255, 217, 61, 0.15), rgba(255, 154, 118, 0.1));
  border: 2px dashed rgba(255, 154, 118, 0.3);
  border-radius: var(--lelamp-radius-xl);
}

.help-icon {
  font-size: 2rem;
}

.help-content {
  flex: 1;
}

.help-title {
  font-size: 1rem;
  font-weight: 700;
  color: var(--lelamp-text-primary);
  margin: 0 0 var(--lelamp-space-sm);
}

.help-list {
  margin: 0;
  padding-left: var(--lelamp-space-lg);
  font-size: 0.875rem;
  color: var(--lelamp-text-secondary);

  li {
    margin-bottom: var(--lelamp-space-xs);

    &:last-child {
      margin-bottom: 0;
    }
  }
}

@media (max-width: 480px) {
  .device-main {
    padding: var(--lelamp-space-md);
  }

  .header-spacer {
    display: none;
  }

  .back-btn span {
    display: none;
  }

  .device-avatar {
    width: 40px;
    height: 40px;
  }

  .avatar-icon {
    font-size: 1.25rem;
  }

  .help-card {
    flex-direction: column;
  }
}
</style>
