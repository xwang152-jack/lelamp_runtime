<template>
  <div class="setup-wizard">
    <!-- 进度指示器 -->
    <div class="progress-bar">
      <div
        v-for="(step, index) in steps"
        :key="index"
        class="progress-step"
        :class="{
          active: currentStep === index,
          completed: index < currentStep
        }"
      >
        <div class="step-number">{{ index + 1 }}</div>
        <div class="step-label">{{ step }}</div>
      </div>
    </div>

    <!-- 步骤内容 -->
    <div class="step-content">
      <!-- 步骤 1: 欢迎页面 -->
      <div v-if="currentStep === 0" class="welcome-step">
        <div class="welcome-icon">🪔</div>
        <h1>欢迎使用 LeLamp</h1>
        <p>让我们来配置您的智能台灯</p>
        <p class="hint">这个过程只需要 2-3 分钟</p>
      </div>

      <!-- 步骤 2: WiFi 扫描 -->
      <div v-if="currentStep === 1" class="wifi-scan-step">
        <h2>选择您的 WiFi 网络</h2>
        <p class="step-description">正在扫描附近的 WiFi 网络...</p>

        <el-button
          type="primary"
          :icon="Refresh"
          :loading="scanning"
          @click="handleScan"
          class="scan-button"
        >
          {{ scanning ? '扫描中...' : '重新扫描' }}
        </el-button>

        <div v-if="networks.length > 0" class="networks-list">
          <div
            v-for="network in sortedNetworks"
            :key="network.bssid"
            class="network-item"
            :class="{ selected: selectedNetwork?.bssid === network.bssid }"
            @click="selectNetwork(network)"
          >
            <div class="network-info">
              <div class="network-name">{{ network.ssid }}</div>
              <div class="network-meta">
                <span>{{ network.security === 'open' ? '开放' : '加密' }}</span>
                <span>{{ network.frequency }}</span>
              </div>
            </div>
            <div class="network-signal">
              <el-icon class="signal-icon" :style="{ opacity: getSignalOpacity(network.signal_strength) }">
                <Signal />
              </el-icon>
            </div>
          </div>
        </div>
        <el-empty v-else-if="!scanning" description="未找到 WiFi 网络，请点击扫描按钮" />
      </div>

      <!-- 步骤 3: 输入密码 -->
      <div v-if="currentStep === 2" class="password-step">
        <h2>输入 WiFi 密码</h2>
        <div class="selected-network">
          <div class="network-ssid">{{ selectedNetwork?.ssid }}</div>
          <div class="network-security">{{ selectedNetwork?.security === 'open' ? '开放网络' : '需要密码' }}</div>
        </div>

        <el-form v-if="selectedNetwork?.security !== 'open'" label-position="top">
          <el-form-item label="密码">
            <el-input
              v-model="wifiPassword"
              type="password"
              placeholder="请输入 WiFi 密码"
              show-password
              size="large"
              @keyup.enter="handleNext"
            />
          </el-form-item>
        </el-form>
        <el-alert v-else type="success" :closable="false" show-icon>
          这是一个开放网络，无需密码
        </el-alert>
      </div>

      <!-- 步骤 4: 连接验证 -->
      <div v-if="currentStep === 3" class="connecting-step">
        <div class="connecting-animation">
          <div class="spinner"></div>
        </div>
        <h2>{{ connectingStatus.title }}</h2>
        <p>{{ connectingStatus.message }}</p>

        <div v-if="connectionError" class="error-message">
          <el-alert type="error" :title="connectionError" show-icon />
          <el-button @click="handleRetry" class="retry-button">重试</el-button>
        </div>
      </div>

      <!-- 步骤 5: 完成 -->
      <div v-if="currentStep === 4" class="complete-step">
        <div class="success-icon">✓</div>
        <h2>配置完成！</h2>
        <p>LeLamp 即将重启并连接到您的 WiFi</p>
        <div class="countdown">
          <span>{{ countdown }}</span> 秒后重启...
        </div>
      </div>
    </div>

    <!-- 底部导航 -->
    <div class="wizard-footer">
      <el-button
        v-if="currentStep > 0 && currentStep < 4"
        @click="handlePrevious"
        :disabled="connecting"
      >
        上一步
      </el-button>
      <el-button
        v-if="currentStep < 3"
        type="primary"
        @click="handleNext"
        :disabled="!canProceed"
      >
        下一步
      </el-button>
      <el-button
        v-if="currentStep === 2"
        type="primary"
        @click="handleConnect"
        :loading="connecting"
        :disabled="!canConnect"
      >
        连接
      </el-button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { Refresh } from '@element-plus/icons-vue'
import axios from 'axios'
import type { WiFiNetwork } from '@/types/settings'

const API_BASE = import.meta.env.VITE_API_BASE_URL || ''

// 添加请求超时配置
// @ts-ignore
const apiClient = axios.create({
  timeout: 30000,  // 30 秒超时
  headers: {
    'Content-Type': 'application/json'
  }
})

// 步骤定义
const steps = ['欢迎', '选择 WiFi', '输入密码', '连接中', '完成']
const currentStep = ref(0)

// WiFi 相关状态
const scanning = ref(false)
const networks = ref<WiFiNetwork[]>([])
const selectedNetwork = ref<WiFiNetwork | null>(null)
const wifiPassword = ref('')

// 连接状态
const connecting = ref(false)
const connectingStatus = ref({ title: '正在连接...', message: '请稍候' })
const connectionError = ref('')
const countdown = ref(5)

// 计算属性
const sortedNetworks = computed(() => {
  return [...networks.value].sort((a, b) => b.signal_strength - a.signal_strength)
})

const canProceed = computed(() => {
  if (currentStep.value === 1) return selectedNetwork.value !== null
  if (currentStep.value === 2) {
    if (selectedNetwork.value?.security === 'open') return true
    return wifiPassword.value.length > 0
  }
  return true
})

const canConnect = computed(() => {
  if (selectedNetwork.value?.security === 'open') return true
  return wifiPassword.value.length > 0
})

// 获取信号强度不透明度
function getSignalOpacity(strength: number): number {
  return Math.max(0.2, strength / 100)
}

// 扫描网络
async function handleScan() {
  scanning.value = true
  try {
    const response = await axios.get(`${API_BASE}/api/system/wifi/scan`)
    networks.value = response.data.networks || []
    if (networks.value.length === 0) {
      ElMessage.warning('未找到可用的 WiFi 网络')
    }
  } catch (error) {
    ElMessage.error('WiFi 扫描失败，请检查设备是否支持 WiFi')
  } finally {
    scanning.value = false
  }
}

// 选择网络
function selectNetwork(network: WiFiNetwork) {
  selectedNetwork.value = network
  wifiPassword.value = ''
}

// 下一步
function handleNext() {
  if (currentStep.value === 1 && !selectedNetwork.value) {
    ElMessage.warning('请先选择一个 WiFi 网络')
    return
  }
  if (currentStep.value === 2 && !canProceed.value) {
    ElMessage.warning('请输入 WiFi 密码')
    return
  }
  currentStep.value++
}

// 上一步
function handlePrevious() {
  if (currentStep.value > 0) {
    currentStep.value--
  }
}

// 连接 WiFi
async function handleConnect() {
  if (!selectedNetwork.value) return

  connecting.value = true
  connectionError.value = ''
  currentStep.value = 3

  connectingStatus.value = {
    title: '正在连接 WiFi...',
    message: `正在连接到 ${selectedNetwork.value.ssid}`
  }

  try {
    // 连接 WiFi
    const connectResponse = await axios.post(`${API_BASE}/api/system/wifi/connect`, {
      ssid: selectedNetwork.value.ssid,
      password: selectedNetwork.value.security === 'open' ? undefined : wifiPassword.value
    })

    if (!connectResponse.data.success) {
      throw new Error(connectResponse.data.message || '连接失败')
    }

    connectingStatus.value = {
      title: '连接成功！',
      message: '正在保存配置...'
    }

    // 等待一小段时间
    await new Promise(resolve => setTimeout(resolve, 1000))

    // 完成配置
    currentStep.value = 4

    // 开始倒计时
    await completeSetup()

  } catch (error: any) {
    connectionError.value = error.response?.data?.detail || error.message || '连接失败，请检查密码是否正确'
    connecting.value = false
  }
}

// 完成配置
async function completeSetup() {
  try {
    const response = await axios.post(`${API_BASE}/api/setup/complete`, {
      wifi_ssid: selectedNetwork.value?.ssid,
      restart_delay: countdown.value
    })

    if (response.data.success) {
      // 开始倒计时
      const timer = setInterval(() => {
        countdown.value--
        if (countdown.value <= 0) {
          clearInterval(timer)
        }
      }, 1000)
    }
  } catch (error) {
    ElMessage.error('保存配置失败')
  }
}

// 重试
function handleRetry() {
  connectionError.value = ''
  currentStep.value = 2
}

// 初始化
onMounted(() => {
  // 自动开始扫描
  handleScan()
})
</script>

<style lang="scss" scoped>
.setup-wizard {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 20px;
}

.progress-bar {
  display: flex;
  justify-content: center;
  gap: 40px;
  padding: 30px 20px;
  max-width: 600px;
  margin: 0 auto;
  width: 100%;
}

.progress-step {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  opacity: 0.5;
  transition: opacity 0.3s;

  &.active {
    opacity: 1;
  }

  &.completed {
    opacity: 0.8;
  }
}

.step-number {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.3);
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--lelamp-bg-white);
  font-weight: bold;
}

.step-label {
  color: var(--lelamp-bg-white);
  font-size: 14px;
}

.step-content {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 20px;
}

.welcome-step {
  text-align: center;
  color: var(--lelamp-bg-white);

  .welcome-icon {
    font-size: 80px;
    margin-bottom: 20px;
  }

  h1 {
    font-size: 32px;
    margin-bottom: 10px;
  }

  p {
    font-size: 18px;
    opacity: 0.9;
  }

  .hint {
    font-size: 14px;
    opacity: 0.7;
    margin-top: 20px;
  }
}

.wifi-scan-step {
  background: var(--lelamp-bg-white);
  border-radius: 16px;
  padding: 30px;
  max-width: 500px;
  width: 100%;

  h2 {
    margin-top: 0;
    color: var(--lelamp-text-primary);
  }

  .step-description {
    color: var(--lelamp-text-secondary);
    margin-bottom: 20px;
  }

  .scan-button {
    width: 100%;
    margin-bottom: 20px;
  }
}

.networks-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
  max-height: 400px;
  overflow-y: auto;
}

.network-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px;
  border: 2px solid var(--lelamp-bg-gray);
  border-radius: 12px;
  cursor: pointer;
  transition: all 0.2s;

  &:hover {
    border-color: var(--lelamp-sky);
    background: var(--lelamp-bg-gray);
  }

  &.selected {
    border-color: var(--lelamp-sky);
    background: var(--lelamp-sky-light);
  }

  .network-info {
    flex: 1;

    .network-name {
      font-weight: 500;
      font-size: 16px;
      color: var(--lelamp-text-primary);
      margin-bottom: 4px;
    }

    .network-meta {
      display: flex;
      gap: 12px;
      font-size: 12px;
      color: var(--lelamp-text-tertiary);
    }
  }

  .network-signal {
    .signal-icon {
      font-size: 24px;
      color: var(--lelamp-mint);
    }
  }
}

.password-step {
  background: var(--lelamp-bg-white);
  border-radius: 16px;
  padding: 30px;
  max-width: 450px;
  width: 100%;

  h2 {
    margin-top: 0;
    color: var(--lelamp-text-primary);
  }

  .selected-network {
    padding: 16px;
    background: var(--lelamp-bg-gray);
    border-radius: 8px;
    margin-bottom: 24px;

    .network-ssid {
      font-size: 18px;
      font-weight: 500;
      color: var(--lelamp-text-primary);
    }

    .network-security {
      font-size: 14px;
      color: var(--lelamp-text-secondary);
      margin-top: 4px;
    }
  }
}

.connecting-step {
  text-align: center;
  color: var(--lelamp-text-primary);

  .connecting-animation {
    margin-bottom: 30px;

    .spinner {
      width: 60px;
      height: 60px;
      border: 4px solid rgba(255, 255, 255, 0.3);
      border-top-color: var(--lelamp-bg-white);
      border-radius: 50%;
      animation: spin 1s linear infinite;
      margin: 0 auto;
    }
  }

  h2 {
    font-size: 24px;
    margin-bottom: 10px;
  }

  .error-message {
    margin-top: 30px;

    .retry-button {
      margin-top: 16px;
    }
  }
}

.complete-step {
  text-align: center;
  color: var(--lelamp-bg-white);

  .success-icon {
    width: 80px;
    height: 80px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.2);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 48px;
    margin: 0 auto 20px;
  }

  h2 {
    font-size: 28px;
    margin-bottom: 10px;
  }

  .countdown {
    font-size: 48px;
    font-weight: bold;
    margin-top: 20px;
  }
}

.wizard-footer {
  display: flex;
  justify-content: center;
  gap: 16px;
  padding: 20px;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}
</style>
