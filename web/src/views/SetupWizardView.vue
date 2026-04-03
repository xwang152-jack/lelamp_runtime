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
        <p class="hint">这个过程只需要 1-2 分钟</p>
        <div v-if="deviceInfo" class="device-info-badge">
          <span class="badge-label">设备</span>
          <span class="badge-value">{{ deviceInfo.hostname || deviceInfo.device_id }}</span>
        </div>
      </div>

      <!-- 步骤 2: 选择 WiFi + 输入密码（合并） -->
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
          >
            <!-- 网络基本信息行 -->
            <div class="network-row" @click="selectNetwork(network)">
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

            <!-- 展开的密码输入区域 -->
            <div v-if="selectedNetwork?.bssid === network.bssid" class="network-expand">
              <el-form label-position="top" v-if="network.security !== 'open'">
                <el-form-item label="密码">
                  <el-input
                    v-model="wifiPassword"
                    type="password"
                    placeholder="请输入 WiFi 密码"
                    show-password
                    size="large"
                    @keyup.enter="handleConnect"
                  />
                </el-form-item>
              </el-form>
              <el-alert v-else type="success" :closable="false" show-icon>
                这是一个开放网络，无需密码
              </el-alert>
              <el-button
                type="primary"
                class="connect-inline-btn"
                :loading="connecting"
                @click="handleConnect"
              >
                连接
              </el-button>
            </div>
          </div>
        </div>
        <el-empty v-else-if="!scanning" description="未找到 WiFi 网络，请点击扫描按钮" />
      </div>

      <!-- 步骤 3: 连接中 -->
      <div v-if="currentStep === 2" class="connecting-step">
        <h2>{{ connectionError ? '连接失败' : '正在连接...' }}</h2>

        <!-- 实时进度列表 -->
        <div class="progress-list" v-if="progressItems.length > 0 && !connectionError">
          <div
            v-for="(item, i) in progressItems"
            :key="i"
            class="progress-item"
            :class="item.status"
          >
            <span class="progress-icon">
              <span v-if="item.status === 'done'">✓</span>
              <span v-else-if="item.status === 'running'" class="mini-spinner"></span>
              <span v-else-if="item.status === 'error'">✗</span>
              <span v-else>○</span>
            </span>
            <span class="progress-text">{{ item.text }}</span>
          </div>
        </div>

        <!-- 静态 spinner（初始状态） -->
        <div v-else-if="!connectionError" class="connecting-animation">
          <div class="spinner"></div>
          <p>{{ connectingStatus.message }}</p>
        </div>

        <!-- 错误状态 -->
        <div v-if="connectionError" class="error-message">
          <el-alert type="error" :title="connectionError" show-icon :closable="false" />
          <el-button @click="handleRetry" class="retry-button" type="primary">重新选择网络</el-button>
        </div>
      </div>

      <!-- 步骤 4: 完成 -->
      <div v-if="currentStep === 3" class="complete-step">
        <div class="success-icon">✓</div>
        <h2>配置完成！</h2>
        <p class="complete-ssid">WiFi 连接成功：{{ completedSsid }}</p>
        <p class="complete-hint">设备正在切换到 WiFi 网络...</p>
      </div>
    </div>

    <!-- 底部导航 -->
    <div class="wizard-footer">
      <el-button
        v-if="currentStep === 1"
        @click="handlePrevious"
        :disabled="connecting"
      >
        上一步
      </el-button>
    </div>

    <!-- 恢复确认对话框 -->
    <el-dialog
      v-model="showRecoveryDialog"
      title="检测到已连接的 WiFi"
      width="90%"
      :close-on-click-modal="false"
      :show-close="false"
    >
      <p>设备已连接到 WiFi：<strong>{{ recoveryInfo?.current_ssid }}</strong></p>
      <p>是否跳过 WiFi 配置，直接完成设置？</p>
      <template #footer>
        <el-button @click="rejectRecovery">重新配置</el-button>
        <el-button type="primary" @click="acceptRecovery">是，继续</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { formatApiError } from '@/utils/errorMessages'
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

// 步骤定义（4 步）
const steps = ['欢迎', '选择 WiFi', '连接中', '完成']
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

// 完成状态
const completedSsid = ref('')

// 设备信息
const deviceInfo = ref<{ device_id: string; hostname: string } | null>(null)

// 恢复检测
const recoveryChecked = ref(false)
const showRecoveryDialog = ref(false)
const recoveryInfo = ref<{ skip_to_step: number; current_ssid: string } | null>(null)

// 实时进度（步骤 3）
interface ProgressItem {
  text: string
  status: 'pending' | 'running' | 'done' | 'error'
}
const progressItems = ref<ProgressItem[]>([])
let setupWs: WebSocket | null = null

// 计算属性
const sortedNetworks = computed(() => {
  return [...networks.value].sort((a, b) => b.signal_strength - a.signal_strength)
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

// 上一步
function handlePrevious() {
  if (currentStep.value > 0) {
    currentStep.value--
  }
}

// 连接 WiFi
async function handleConnect() {
  if (!selectedNetwork.value) return

  // 验证密码
  if (selectedNetwork.value.security !== 'open' && wifiPassword.value.length === 0) {
    ElMessage.warning('请输入 WiFi 密码')
    return
  }

  connecting.value = true
  connectionError.value = ''
  currentStep.value = 2
  initProgress()
  connectSetupWs()

  try {
    const connectResponse = await apiClient.post(`${API_BASE}/api/system/wifi/connect`, {
      ssid: selectedNetwork.value.ssid,
      password: selectedNetwork.value.security === 'open' ? undefined : wifiPassword.value
    })

    if (!connectResponse.data.success) {
      throw new Error(connectResponse.data.message || '连接失败')
    }

    if (connectResponse.data.network_ok === false) {
      connectionError.value = 'WiFi 已连接，但无法访问互联网，请检查路由器设置'
      connecting.value = false
      return
    }

    connectingStatus.value = { title: '连接成功！', message: '正在完成配置...' }
    await new Promise(resolve => setTimeout(resolve, 800))

    // 调用 setup/complete
    const ssid = selectedNetwork.value.ssid
    await apiClient.post(`${API_BASE}/api/system/setup/complete`, {
      wifi_ssid: ssid,
      restart_delay: 0
    })

    completedSsid.value = ssid
    currentStep.value = 3
  } catch (error: any) {
    connectionError.value = formatApiError(error, 'wifi')
  } finally {
    connecting.value = false
    disconnectSetupWs()
  }
}

// 重试
function handleRetry() {
  connectionError.value = ''
  currentStep.value = 1
}

// 恢复确认
async function acceptRecovery() {
  showRecoveryDialog.value = false
  if (recoveryInfo.value) {
    const ssid = recoveryInfo.value.current_ssid
    selectedNetwork.value = {
      ssid: ssid,
      bssid: '',
      signal_strength: 100,
      security: 'wpa2',
      frequency: '2.4GHz',
      is_hidden: false,
    }
    // 直接调用 setup/complete，跳到完成步骤
    try {
      await apiClient.post(`${API_BASE}/api/system/setup/complete`, {
        wifi_ssid: ssid,
        restart_delay: 0
      })
      completedSsid.value = ssid
      currentStep.value = 3
    } catch (error: any) {
      ElMessage.error(formatApiError(error, 'wifi'))
    }
  }
}

function rejectRecovery() {
  showRecoveryDialog.value = false
  handleScan()
}

// WebSocket 驱动的进度
function initProgress() {
  progressItems.value = [
    { text: '正在发送连接请求...', status: 'running' },
    { text: '正在连接 WiFi...', status: 'pending' },
    { text: '验证网络连通性...', status: 'pending' },
  ]
}

function handleProgressEvent(event: { event: string; attempt?: number; retry_in?: number }) {
  switch (event.event) {
    case 'wifi_connecting':
      progressItems.value[0].status = 'done'
      progressItems.value[1].status = 'running'
      progressItems.value[1].text = `正在连接 WiFi（第 ${event.attempt}/${3} 次）...`
      break
    case 'wifi_connected':
      progressItems.value[1].status = 'done'
      progressItems.value[2].status = 'running'
      break
    case 'wifi_failed':
      if (event.retry_in) {
        progressItems.value[1].text = `连接失败，${event.retry_in} 秒后重试...`
      }
      break
    case 'network_ok':
      progressItems.value[2].status = 'done'
      break
    case 'network_failed':
      progressItems.value[2].status = 'error'
      progressItems.value[2].text = 'WiFi 已连接，但无法访问互联网'
      connectionError.value = 'WiFi 已连接，但无法访问互联网，请检查路由器设置'
      break
  }
}

function connectSetupWs() {
  const wsUrl = `${API_BASE.replace('http', 'ws')}/api/ws/setup`
  setupWs = new WebSocket(wsUrl)
  setupWs.onmessage = (e) => {
    try {
      const event = JSON.parse(e.data)
      handleProgressEvent(event)
    } catch { /* ignore */ }
  }
  setupWs.onerror = () => { /* ignore */ }
}

function disconnectSetupWs() {
  if (setupWs) {
    setupWs.close()
    setupWs = null
  }
}

// 初始化
onMounted(async () => {
  // 获取设备信息
  try {
    const response = await axios.get(`${API_BASE}/api/system/device`)
    deviceInfo.value = response.data
  } catch {
    // ignore
  }

  // 检查是否可以从中断处恢复
  try {
    const recovery = await axios.get(`${API_BASE}/api/system/setup/recovery`)
    recoveryChecked.value = true
    if (recovery.data.can_recover) {
      recoveryInfo.value = recovery.data
      showRecoveryDialog.value = true
    } else {
      handleScan()
    }
  } catch {
    recoveryChecked.value = true
    handleScan()
  }
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

.progress-list {
  margin: 20px 0;
  text-align: left;

  .progress-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 0;
    opacity: 0.5;
    color: white;

    &.running, &.done, &.error { opacity: 1; }
    &.done .progress-icon { color: #67c23a; }
    &.error .progress-icon { color: #f56c6c; }
  }

  .mini-spinner {
    display: inline-block;
    width: 14px;
    height: 14px;
    border: 2px solid rgba(255,255,255,0.3);
    border-top-color: white;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
}

.progress-bar {
  display: flex;
  justify-content: center;
  gap: 40px;
  padding: 30px 20px;
  max-width: 700px;
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

.device-info-badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  margin-top: 24px;
  padding: 8px 16px;
  background: rgba(255, 255, 255, 0.15);
  border-radius: 20px;

  .badge-label {
    font-size: 12px;
    opacity: 0.7;
  }

  .badge-value {
    font-size: 14px;
    font-weight: 500;
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
  border: 2px solid var(--lelamp-bg-gray);
  border-radius: 12px;
  transition: all 0.2s;

  &.selected {
    border-color: var(--lelamp-sky);
    background: var(--lelamp-sky-light);
  }

  .network-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px;
    cursor: pointer;

    &:hover {
      opacity: 0.85;
    }
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

  .network-expand {
    padding: 0 16px 16px 16px;
    border-top: 1px solid var(--lelamp-bg-gray);

    .connect-inline-btn {
      width: 100%;
      margin-top: 12px;
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
    color: #67c23a;
  }

  h2 {
    font-size: 28px;
    margin-bottom: 10px;
  }

  .complete-ssid {
    font-size: 18px;
    opacity: 0.9;
    margin-bottom: 8px;
  }

  .complete-hint {
    font-size: 14px;
    opacity: 0.7;
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
