<template>
  <div class="connect-view">
    <!-- Floating decorative elements -->
    <div class="floating-shapes">
      <div class="shape shape-1"></div>
      <div class="shape shape-2"></div>
      <div class="shape shape-3"></div>
    </div>

    <!-- Auth Entry Button -->
    <button class="auth-entry-btn" @click="handleAuthClick">
      <span v-if="!isAuthenticated" class="auth-content">
        <span class="auth-icon">👤</span>
        <span class="auth-text">登录 / 注册</span>
      </span>
      <div v-else class="user-avatar" @click.stop="handleAvatarClick">
        <span class="avatar-text">{{ userInitial }}</span>
      </div>
    </button>

    <div class="connect-container">
      <!-- Robot Mascot -->
      <div class="robot-mascot" :class="{ 'excited': loading, 'connected': isConnected }">
        <div class="robot-body">
          <div class="robot-head">
            <div class="robot-eyes">
              <div class="eye left" :class="{ 'blink': isBlinking }">
                <div class="pupil"></div>
              </div>
              <div class="eye right" :class="{ 'blink': isBlinking }">
                <div class="pupil"></div>
              </div>
            </div>
            <div class="robot-antenna">
              <div class="antenna-ball" :class="{ 'glow': loading }"></div>
            </div>
          </div>
          <div class="robot-base">
            <div class="lamp-light" :class="{ 'on': isConnected }"></div>
          </div>
        </div>
        <div class="robot-shadow"></div>
      </div>

      <!-- Welcome Text -->
      <div class="welcome-section">
        <h1 class="welcome-title">
          <span class="wave">👋</span>
          你好呀！我是 LeLamp
        </h1>
        <p class="welcome-subtitle">你爱学习的智能台灯朋友</p>
      </div>

      <!-- Connection Card -->
      <div class="connection-card">
        <div class="card-decoration"></div>

        <!-- Status indicator -->
        <div class="connection-status" :class="statusClass">
          <div class="status-dot"></div>
          <span class="status-text">{{ statusText }}</span>
        </div>

        <!-- Connection form -->
        <div class="connection-form">
          <div class="input-group">
            <label class="input-label">
              <span class="label-icon">🔗</span>
              服务器地址
            </label>
            <div class="input-wrapper">
              <input
                v-model="form.serverUrl"
                type="text"
                class="le-input"
                placeholder="自动检测中..."
                readonly
              />
              <div class="input-glow"></div>
            </div>
            <div class="input-hint">
              <span class="hint-icon">✨</span>
              {{ form.serverUrl || '正在自动配置...' }}
            </div>
          </div>

          <!-- Connect Button -->
          <button
            class="connect-btn"
            :class="{ 'loading': loading, 'success': isConnected }"
            @click="handleConnect"
            :disabled="loading || isConnected"
          >
            <span v-if="!loading && !isConnected" class="btn-content">
              <span class="btn-icon">🪔</span>
              点亮我的灯
            </span>
            <span v-else-if="loading" class="btn-content loading">
              <span class="spinner"></span>
              连接中...
            </span>
            <span v-else class="btn-content success">
              <span class="check-icon">✓</span>
              已连接
            </span>
          </button>
        </div>

        <!-- WiFi Status -->
        <div class="wifi-status">
          <div class="wifi-indicator" :class="{ 'active': connectedWiFi }">
            <svg class="wifi-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M12 20v-2M5 12.55a11 11 0 0 1 14.08 0M8.5 8.5a6 6 0 0 1 7 0" />
              <path d="M16.5 7a10 10 0 0 0-9 0" />
            </svg>
          </div>
          <span class="wifi-text">{{ connectedWiFi || '正在检测 WiFi...' }}</span>
        </div>
      </div>

      <!-- Device Info Card (shown after loading device info) -->
      <div v-if="deviceInfo" class="device-info-card">
        <div class="device-info-header">
          <span class="device-info-icon">📡</span>
          <span class="device-info-title">设备信息</span>
          <span v-if="deviceInfo.is_bound" class="badge badge-bound">已绑定</span>
        </div>
        <div class="device-info-body">
          <div class="info-row">
            <span class="info-label">设备 ID</span>
            <code class="info-value">{{ deviceInfo.device_id }}</code>
          </div>
          <div class="info-row">
            <span class="info-label">绑定密钥</span>
            <div class="secret-value">
              <code v-if="showSecret">{{ deviceInfo.device_secret }}</code>
              <code v-else>••••••••••••••••</code>
              <button class="copy-btn" @click="toggleSecret" :title="showSecret ? '隐藏' : '显示'">
                {{ showSecret ? '🙈' : '👁' }}
              </button>
              <button class="copy-btn" @click="copySecret" title="复制">
                📋
              </button>
            </div>
          </div>
          <button
            v-if="isAuthenticated && !deviceInfo.is_bound && deviceInfo.device_secret"
            class="bind-btn"
            @click="handleBindDevice"
          >
            🔗 绑定此设备
          </button>
        </div>
      </div>

      <!-- Help Tips -->
      <div class="help-tips">
        <div class="tip-item" v-for="(tip, index) in tips" :key="index" :style="{ animationDelay: `${index * 0.1}s` }">
          <span class="tip-number">{{ index + 1 }}</span>
          <span class="tip-text">{{ tip }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted, onUnmounted, computed } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useWebSocket } from '@/composables/useWebSocket'
import { useAuthStore } from '@/stores'
import { getUserInitial } from '@/utils/device'

const router = useRouter()
const { connect } = useWebSocket()
const authStore = useAuthStore()

const loading = ref(false)
const isConnected = ref(false)
const isBlinking = ref(false)
const connectedWiFi = ref('')
const deviceInfo = ref<Record<string, any> | null>(null)
const showSecret = ref(false)

const form = reactive({
  serverUrl: '',
  deviceName: 'LeLamp'
})

const tips = [
  '确保我和你在同一个 WiFi 下',
  '首次使用需要点一下连接按钮',
  '连接后我就能陪你学习啦~'
]

// Auto-blink animation
let blinkInterval: number

const statusClass = computed(() => {
  if (isConnected.value) return 'connected'
  if (loading.value) return 'connecting'
  return 'idle'
})

const statusText = computed(() => {
  if (isConnected.value) return '已连接'
  if (loading.value) return '正在连接...'
  return '等待连接'
})

// Auth computed properties
const isAuthenticated = computed(() => authStore.isAuthenticated)

const userInitial = computed(() => {
  return getUserInitial(authStore.user?.username)
})

// Auth handlers
function handleAuthClick() {
  if (isAuthenticated.value) {
    // Show user menu or go to profile
    router.push('/profile')
  } else {
    // Go to auth page
    router.push('/auth')
  }
}

function handleAvatarClick() {
  // Show dropdown menu or navigate to profile
  router.push('/profile')
}

const autoConfigureUrl = () => {
  const protocol = window.location.protocol
  const hostname = window.location.hostname
  form.serverUrl = `${protocol}//${hostname}:8000`
}

const checkNetworkStatus = async () => {
  try {
    const response = await fetch(`${form.serverUrl}/api/system/setup/status`)
    if (response.ok) {
      const data = await response.json()
      if (data.configured_wifi) {
        connectedWiFi.value = data.configured_wifi
      }
    }
  } catch (error) {
    console.warn('Failed to check network status:', error)
  }
}

const fetchDeviceInfo = async () => {
  try {
    const token = authStore.accessToken
    const url = new URL(`${form.serverUrl}/api/system/device`)
    if (token) url.searchParams.set('token', token)
    const response = await fetch(url.toString())
    if (response.ok) {
      deviceInfo.value = await response.json()
    }
  } catch (error) {
    console.warn('Failed to fetch device info:', error)
  }
}

const toggleSecret = () => {
  showSecret.value = !showSecret.value
}

const copySecret = async () => {
  if (deviceInfo.value?.device_secret) {
    try {
      await navigator.clipboard.writeText(deviceInfo.value.device_secret)
      ElMessage.success('密钥已复制')
    } catch {
      ElMessage.error('复制失败')
    }
  }
}

const handleBindDevice = async () => {
  if (!deviceInfo.value) return
  try {
    const response = await fetch(`${form.serverUrl}/api/auth/bind-device`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${authStore.accessToken}`,
      },
      body: JSON.stringify({
        device_id: deviceInfo.value.device_id,
        device_secret: deviceInfo.value.device_secret,
      }),
    })
    if (response.ok) {
      deviceInfo.value.is_bound = true
      ElMessage.success('设备绑定成功！')
    } else {
      const data = await response.json()
      ElMessage.error(data.detail || '绑定失败')
    }
  } catch {
    ElMessage.error('绑定请求失败')
  }
}

async function handleConnect() {
  if (!form.serverUrl) {
    ElMessage.warning('请先配置服务器地址')
    return
  }

  loading.value = true
  try {
    await connect(form.serverUrl, 'lelamp')
    isConnected.value = true
    ElMessage({
      message: '连接成功！我们现在是朋友啦~',
      type: 'success',
      duration: 2000
    })

    // Navigate to room after a short delay
    setTimeout(() => {
      router.push('/room')
    }, 800)
  } catch (error) {
    isConnected.value = false
    ElMessage.error('连接失败，请检查设备是否在线')
    console.error(error)
  } finally {
    loading.value = false
  }
}

// Start blink animation
const startBlinkAnimation = () => {
  blinkInterval = window.setInterval(() => {
    isBlinking.value = true
    setTimeout(() => {
      isBlinking.value = false
    }, 150)
  }, 4000)
}

onMounted(() => {
  autoConfigureUrl()
  checkNetworkStatus()
  fetchDeviceInfo()
  startBlinkAnimation()
})

onUnmounted(() => {
  if (blinkInterval) {
    clearInterval(blinkInterval)
  }
})
</script>

<style lang="scss" scoped>
.connect-view {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: var(--lelamp-space-lg);
  position: relative;
  overflow: hidden;
  background: linear-gradient(135deg, #FFF8F0 0%, #FFE5D9 50%, #FFF0E0 100%);
}

/* === Floating Background Shapes === */
.floating-shapes {
  position: absolute;
  inset: 0;
  overflow: hidden;
  pointer-events: none;
}

.shape {
  position: absolute;
  border-radius: 50%;
  opacity: 0.6;
  animation: float 6s ease-in-out infinite;
}

.shape-1 {
  width: 300px;
  height: 300px;
  background: radial-gradient(circle, var(--lelamp-peach-light) 0%, transparent 70%);
  top: -100px;
  right: -50px;
  animation-delay: 0s;
}

.shape-2 {
  width: 200px;
  height: 200px;
  background: radial-gradient(circle, var(--lelamp-sunny-light) 0%, transparent 70%);
  bottom: 100px;
  left: -50px;
  animation-delay: 2s;
}

.shape-3 {
  width: 150px;
  height: 150px;
  background: radial-gradient(circle, var(--lelamp-mint-light) 0%, transparent 70%);
  bottom: -30px;
  right: 20%;
  animation-delay: 4s;
}

/* === Main Container === */
.connect-container {
  max-width: 480px;
  width: 100%;
  position: relative;
  z-index: 1;
}

/* === Robot Mascot === */
.robot-mascot {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-bottom: var(--lelamp-space-lg);
  transition: transform var(--lelamp-transition-bounce);

  &.excited {
    animation: wiggle 0.5s ease-in-out;
  }

  &.connected .robot-base .lamp-light {
    background: var(--lelamp-sunny);
    box-shadow: 0 0 60px var(--lelamp-sunny), 0 0 100px var(--lelamp-sunny-light);
  }
}

.robot-body {
  position: relative;
}

.robot-head {
  position: relative;
  width: 120px;
  height: 100px;
  background: linear-gradient(180deg, var(--lelamp-peach) 0%, var(--lelamp-peach-dark) 100%);
  border-radius: 50% 50% 40% 40%;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 8px 24px rgba(255, 154, 118, 0.3);
}

.robot-eyes {
  display: flex;
  gap: var(--lelamp-space-md);
  margin-top: -8px;
}

.eye {
  width: 32px;
  height: 36px;
  background: var(--lelamp-bg-white);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.1);
  transition: all var(--lelamp-transition-fast);

  &.blink {
    height: 4px !important;
    margin-top: 16px;
  }

  .pupil {
    width: 16px;
    height: 16px;
    background: var(--lelamp-text-primary);
    border-radius: 50%;
    transition: transform var(--lelamp-transition-fast);
  }
}

.robot-mascot:hover .eye .pupil {
  transform: scale(1.2);
}

.robot-antenna {
  position: absolute;
  top: -20px;
  left: 50%;
  transform: translateX(-50%);
  width: 4px;
  height: 24px;
  background: var(--lelamp-peach-dark);
  border-radius: 2px;
}

.antenna-ball {
  position: absolute;
  top: -12px;
  left: 50%;
  transform: translateX(-50%);
  width: 16px;
  height: 16px;
  background: var(--lelamp-coral);
  border-radius: 50%;
  transition: all var(--lelamp-transition-normal);

  &.glow {
    animation: pulse-glow 1s ease-in-out infinite;
  }
}

.robot-base {
  width: 80px;
  height: 30px;
  background: linear-gradient(180deg, var(--lelamp-peach-dark) 0%, #D67A6A 100%);
  border-radius: 0 0 40px 40px;
  margin: -8px auto 0;
  position: relative;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.lamp-light {
  position: absolute;
  top: 8px;
  left: 50%;
  transform: translateX(-50%);
  width: 40px;
  height: 40px;
  background: rgba(255, 255, 255, 0.3);
  border-radius: 50%;
  transition: all var(--lelamp-transition-slow);
}

.robot-shadow {
  width: 80px;
  height: 12px;
  background: rgba(0, 0, 0, 0.1);
  border-radius: 50%;
  margin-top: 8px;
  filter: blur(4px);
}

/* === Welcome Section === */
.welcome-section {
  text-align: center;
  margin-bottom: var(--lelamp-space-xl);
}

.welcome-title {
  font-size: 1.75rem;
  font-weight: 700;
  color: var(--lelamp-text-primary);
  margin-bottom: var(--lelamp-space-sm);
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--lelamp-space-sm);
}

.wave {
  display: inline-block;
  animation: wave 2s ease-in-out infinite;
  transform-origin: 70% 70%;
}

@keyframes wave {
  0%, 100% { transform: rotate(0deg); }
  10% { transform: rotate(14deg); }
  20% { transform: rotate(-8deg); }
  30% { transform: rotate(14deg); }
  40% { transform: rotate(-4deg); }
  50% { transform: rotate(10deg); }
}

.welcome-subtitle {
  font-size: 1rem;
  color: var(--lelamp-text-secondary);
}

/* === Connection Card === */
.connection-card {
  background: var(--lelamp-bg-white);
  border-radius: var(--lelamp-radius-xl);
  padding: var(--lelamp-space-lg);
  box-shadow: var(--lelamp-shadow-lg);
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

/* === Status Indicator === */
.connection-status {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--lelamp-space-sm);
  padding: var(--lelamp-space-sm) var(--lelamp-space-md);
  border-radius: var(--lelamp-radius-full);
  margin-bottom: var(--lelamp-space-lg);
  font-size: 0.875rem;
  font-weight: 600;
  transition: all var(--lelamp-transition-normal);

  &.idle {
    background: var(--lelamp-bg-gray);
    color: var(--lelamp-text-secondary);

    .status-dot {
      background: var(--lelamp-text-tertiary);
    }
  }

  &.connecting {
    background: rgba(255, 217, 61, 0.15);
    color: var(--lelamp-sunny-dark);

    .status-dot {
      background: var(--lelamp-sunny);
      animation: pulse 1s ease-in-out infinite;
    }
  }

  &.connected {
    background: rgba(107, 203, 119, 0.15);
    color: var(--lelamp-mint-dark);

    .status-dot {
      background: var(--lelamp-mint);
    }
  }
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

/* === Input Group === */
.input-group {
  margin-bottom: var(--lelamp-space-lg);
}

.input-label {
  display: flex;
  align-items: center;
  gap: var(--lelamp-space-sm);
  font-size: 0.875rem;
  font-weight: 600;
  color: var(--lelamp-text-primary);
  margin-bottom: var(--lelamp-space-sm);
}

.label-icon {
  font-size: 1.125rem;
}

.input-wrapper {
  position: relative;
}

.le-input {
  width: 100%;
  padding: var(--lelamp-space-md) var(--lelamp-space-lg);
  font-size: 1rem;
  font-family: var(--lelamp-font-body);
  background: var(--lelamp-bg-gray);
  border: 2px solid transparent;
  border-radius: var(--lelamp-radius-md);
  color: var(--lelamp-text-primary);
  transition: all var(--lelamp-transition-normal);

  &:focus {
    outline: none;
    background: var(--lelamp-bg-white);
    border-color: var(--lelamp-peach);
    box-shadow: 0 0 0 4px rgba(255, 154, 118, 0.1);
  }

  &::placeholder {
    color: var(--lelamp-text-tertiary);
  }
}

.input-glow {
  position: absolute;
  inset: -2px;
  border-radius: var(--lelamp-radius-md);
  background: linear-gradient(90deg, var(--lelamp-peach), var(--lelamp-sunny), var(--lelamp-mint));
  z-index: -1;
  opacity: 0;
  transition: opacity var(--lelamp-transition-normal);
}

.le-input:focus + .input-glow {
  opacity: 0.5;
}

.input-hint {
  display: flex;
  align-items: center;
  gap: var(--lelamp-space-xs);
  margin-top: var(--lelamp-space-sm);
  font-size: 0.813rem;
  color: var(--lelamp-text-secondary);
}

.hint-icon {
  animation: wiggle 2s ease-in-out infinite;
}

/* === Connect Button === */
.connect-btn {
  width: 100%;
  padding: var(--lelamp-space-md) var(--lelamp-space-lg);
  font-size: 1.125rem;
  font-weight: 700;
  font-family: var(--lelamp-font-display);
  background: linear-gradient(135deg, var(--lelamp-peach) 0%, var(--lelamp-coral) 100%);
  color: var(--lelamp-bg-white);
  border: none;
  border-radius: var(--lelamp-radius-lg);
  cursor: pointer;
  transition: all var(--lelamp-transition-bounce);
  box-shadow: 0 4px 16px rgba(255, 107, 138, 0.3);
  position: relative;
  overflow: hidden;

  &:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(255, 107, 138, 0.4);
  }

  &:active:not(:disabled) {
    transform: translateY(0);
  }

  &:disabled {
    opacity: 0.8;
    cursor: not-allowed;
  }

  &.success {
    background: linear-gradient(135deg, var(--lelamp-mint) 0%, #4FA85C 100%);
    box-shadow: 0 4px 16px rgba(107, 203, 119, 0.3);
  }
}

.btn-content {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--lelamp-space-sm);
}

.btn-icon {
  font-size: 1.25rem;
}

.spinner {
  width: 20px;
  height: 20px;
  border: 3px solid rgba(255, 255, 255, 0.3);
  border-top-color: var(--lelamp-bg-white);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.check-icon {
  font-size: 1.25rem;
  font-weight: bold;
}

/* === WiFi Status === */
.wifi-status {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--lelamp-space-sm);
  padding-top: var(--lelamp-space-md);
  border-top: 1px dashed rgba(0, 0, 0, 0.1);
}

.wifi-indicator {
  width: 24px;
  height: 24px;
  color: var(--lelamp-text-tertiary);
  transition: all var(--lelamp-transition-normal);

  &.active {
    color: var(--lelamp-mint);
  }
}

.wifi-icon {
  width: 100%;
  height: 100%;
}

.wifi-text {
  font-size: 0.875rem;
  color: var(--lelamp-text-secondary);
}

/* === Help Tips === */
.help-tips {
  margin-top: var(--lelamp-space-xl);
  display: flex;
  flex-direction: column;
  gap: var(--lelamp-space-sm);
}

/* === Device Info Card === */
.device-info-card {
  margin-top: var(--lelamp-space-lg);
  background: var(--lelamp-bg-white);
  border-radius: var(--lelamp-radius-xl);
  padding: var(--lelamp-space-md) var(--lelamp-space-lg);
  box-shadow: var(--lelamp-shadow-md);
  animation: slide-up 0.5s ease-out backwards;
  animation-delay: 0.2s;
}

.device-info-header {
  display: flex;
  align-items: center;
  gap: var(--lelamp-space-sm);
  margin-bottom: var(--lelamp-space-md);
}

.device-info-icon {
  font-size: 1.125rem;
}

.device-info-title {
  font-size: 0.938rem;
  font-weight: 700;
  color: var(--lelamp-text-primary);
  flex: 1;
}

.badge {
  font-size: 0.75rem;
  font-weight: 600;
  padding: 2px 10px;
  border-radius: var(--lelamp-radius-full);
}

.badge-bound {
  background: rgba(107, 203, 119, 0.15);
  color: var(--lelamp-mint-dark);
}

.device-info-body {
  display: flex;
  flex-direction: column;
  gap: var(--lelamp-space-sm);
}

.info-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--lelamp-space-md);
}

.info-label {
  font-size: 0.813rem;
  color: var(--lelamp-text-secondary);
  white-space: nowrap;
}

.info-value {
  font-size: 0.813rem;
  font-family: monospace;
  color: var(--lelamp-text-primary);
  background: var(--lelamp-bg-gray);
  padding: 2px 8px;
  border-radius: var(--lelamp-radius-sm);
}

.secret-value {
  display: flex;
  align-items: center;
  gap: var(--lelamp-space-xs);
}

.secret-value code {
  font-size: 0.813rem;
  font-family: monospace;
  color: var(--lelamp-text-primary);
  background: var(--lelamp-bg-gray);
  padding: 2px 8px;
  border-radius: var(--lelamp-radius-sm);
  flex: 1;
  text-align: center;
}

.copy-btn {
  width: 28px;
  height: 28px;
  display: flex;
  align-items: center;
  justify-content: center;
  border: none;
  background: var(--lelamp-bg-gray);
  border-radius: var(--lelamp-radius-sm);
  cursor: pointer;
  font-size: 0.875rem;
  transition: background var(--lelamp-transition-fast);

  &:hover {
    background: var(--lelamp-peach-light);
  }
}

.bind-btn {
  width: 100%;
  margin-top: var(--lelamp-space-sm);
  padding: var(--lelamp-space-sm) var(--lelamp-space-md);
  font-size: 0.875rem;
  font-weight: 600;
  font-family: var(--lelamp-font-body);
  background: linear-gradient(135deg, var(--lelamp-peach) 0%, var(--lelamp-coral) 100%);
  color: var(--lelamp-bg-white);
  border: none;
  border-radius: var(--lelamp-radius-md);
  cursor: pointer;
  transition: all var(--lelamp-transition-bounce);
  box-shadow: 0 2px 8px rgba(255, 107, 138, 0.2);

  &:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(255, 107, 138, 0.3);
  }
}

.tip-item {
  display: flex;
  align-items: center;
  gap: var(--lelamp-space-md);
  padding: var(--lelamp-space-md);
  background: var(--lelamp-bg-white);
  border-radius: var(--lelamp-radius-md);
  box-shadow: var(--lelamp-shadow-sm);
  animation: slide-up 0.5s ease-out backwards;
}

.tip-number {
  display: flex;
  align-items: center;
  justify-content: center;
  min-width: 28px;
  height: 28px;
  background: linear-gradient(135deg, var(--lelamp-peach-light), var(--lelamp-sunny-light));
  color: var(--lelamp-text-primary);
  font-weight: 700;
  font-size: 0.875rem;
  border-radius: 50%;
}

.tip-text {
  font-size: 0.875rem;
  color: var(--lelamp-text-secondary);
}

/* === Responsive === */
@media (max-width: 480px) {
  .connect-view {
    padding: var(--lelamp-space-md);
  }

  .welcome-title {
    font-size: 1.5rem;
  }

  .connection-card {
    padding: var(--lelamp-space-md);
  }

  .robot-head {
    width: 100px;
    height: 85px;
  }

  .eye {
    width: 26px;
    height: 30px;

    .pupil {
      width: 14px;
      height: 14px;
    }
  }
}

/* === Auth Entry Button === */
.auth-entry-btn {
  position: fixed;
  top: var(--lelamp-space-lg);
  right: var(--lelamp-space-lg);
  z-index: 100;
  padding: var(--lelamp-space-sm) var(--lelamp-space-md);
  background: var(--lelamp-bg-white);
  border: 2px solid rgba(0, 0, 0, 0.08);
  border-radius: var(--lelamp-radius-full);
  cursor: pointer;
  transition: all var(--lelamp-transition-bounce);
  box-shadow: var(--lelamp-shadow-md);

  &:hover {
    border-color: var(--lelamp-peach);
    transform: translateY(-2px);
    box-shadow: var(--lelamp-shadow-lg);
  }

  &:active {
    transform: translateY(0);
  }
}

.auth-content {
  display: flex;
  align-items: center;
  gap: var(--lelamp-space-sm);
}

.auth-icon {
  font-size: 1.25rem;
}

.auth-text {
  font-size: 0.875rem;
  font-weight: 600;
  color: var(--lelamp-text-primary);
}

.user-avatar {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background: linear-gradient(135deg, var(--lelamp-peach) 0%, var(--lelamp-coral) 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 4px 12px rgba(255, 107, 138, 0.3);
}

.avatar-text {
  font-size: 1rem;
  font-weight: 700;
  color: var(--lelamp-bg-white);
}

@media (max-width: 480px) {
  .auth-entry-btn {
    top: var(--lelamp-space-md);
    right: var(--lelamp-space-md);
    padding: var(--lelamp-space-xs) var(--lelamp-space-sm);
  }

  .auth-text {
    display: none;
  }

  .user-avatar {
    width: 36px;
    height: 36px;
  }

  .avatar-text {
    font-size: 0.875rem;
  }
}
</style>
