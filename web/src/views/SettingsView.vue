<template>
  <div class="settings-view">
    <!-- Animated Background -->
    <div class="settings-background">
      <div class="bg-gradient"></div>
      <div class="floating-gears">
        <div v-for="i in 4" :key="i" class="gear" :class="`gear-${i}`" :style="{ '--delay': `${i * 1.5}s` }">
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 15a3 3 0 100-6 3 3 0 000 6z" />
            <path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-2 2 2 2 0 01-2-2v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83 0 2 2 0 010-2.83l.06-.06a1.65 1.65 0 00.33-1.82 1.65 1.65 0 00-1.51-1H3a2 2 0 01-2-2 2 2 0 012-2h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 010-2.83 2 2 0 012.83 0l.06.06a1.65 1.65 0 001.82.33H9a1.65 1.65 0 001-1.51V3a2 2 0 012-2 2 2 0 012 2v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 0 2 2 0 010 2.83l-.06.06a1.65 1.65 0 00-.33 1.82V9a1.65 1.65 0 001.51 1H21a2 2 0 012 2 2 2 0 01-2 2h-.09a1.65 1.65 0 00-1.51 1z" />
          </svg>
        </div>
      </div>
    </div>

    <!-- Header -->
    <header class="settings-header">
      <div class="header-content">
        <button class="back-btn" @click="handleBack">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M19 12H5M12 19l-7-7 7-7" />
          </svg>
          <span>返回</span>
        </button>
        <div class="header-info">
          <h1 class="header-title">
            <span class="title-emoji">⚙️</span>
            系统设置
          </h1>
          <p class="device-badge">
            <span class="badge-icon">🪔</span>
            <span class="badge-text">LeLamp 智能台灯</span>
            <span class="badge-id">{{ lampId }}</span>
          </p>
        </div>
        <div class="header-spacer"></div>
      </div>
    </header>

    <!-- Pending Changes Alert -->
    <div v-if="settingsStore.hasPendingChanges" class="pending-alert" :class="{ 'shake': shouldShake }">
      <div class="alert-content">
        <div class="alert-icon">⚠️</div>
        <div class="alert-text">
          <span class="alert-title">配置已更改</span>
          <span class="alert-desc">需要重启服务才能生效</span>
        </div>
        <button class="restart-btn" @click="showRestartDialog">
          <span class="btn-icon">🔄</span>
          立即重启
        </button>
      </div>
    </div>

    <!-- Main Content -->
    <main class="settings-main">
      <!-- Navigation Tabs -->
      <nav class="settings-nav">
        <div class="nav-tabs">
          <button
            v-for="tab in tabs"
            :key="tab.key"
            :class="['nav-tab', { 'active': activeTab === tab.key }]"
            @click="handleTabSelect(tab.key)"
          >
            <span class="tab-emoji">{{ tab.emoji }}</span>
            <span class="tab-label">{{ tab.label }}</span>
          </button>
        </div>
      </nav>

      <!-- Content Area -->
      <div class="settings-content">
        <transition name="slide-fade" mode="out-in">
          <component :is="currentComponent" :key="activeTab" />
        </transition>
      </div>
    </main>

    <!-- Restart Dialog -->
    <transition name="dialog-fade">
      <div v-if="restartDialogVisible" class="dialog-overlay" @click.self="cancelRestart">
        <div class="dialog-box">
          <div class="dialog-header">
            <div class="dialog-icon">🤖</div>
            <h3 class="dialog-title">重启服务</h3>
            <p class="dialog-desc">重启会中断当前连接，确定要重启吗？</p>
          </div>

          <div v-if="restartCountdown > 0" class="countdown-box">
            <div class="countdown-ring">
              <svg class="countdown-svg" viewBox="0 0 100 100">
                <circle cx="50" cy="50" r="45" fill="none" stroke="rgba(255, 154, 118, 0.2)" stroke-width="8" />
                <circle
                  cx="50" cy="50" r="45"
                  fill="none"
                  stroke="var(--lelamp-peach)"
                  stroke-width="8"
                  stroke-linecap="round"
                  :stroke-dasharray="283"
                  :stroke-dashoffset="283 * (1 - restartCountdown / 3)"
                  transform="rotate(-90 50 50)"
                  class="countdown-progress"
                />
              </svg>
              <span class="countdown-number">{{ restartCountdown }}</span>
            </div>
            <p class="countdown-text">服务将在 {{ restartCountdown }} 秒后重启...</p>
          </div>

          <div class="dialog-actions">
            <button class="dialog-btn cancel" @click="cancelRestart" :disabled="restarting">
              取消
            </button>
            <button class="dialog-btn confirm" @click="confirmRestart" :disabled="restarting || restartCountdown > 0">
              <span v-if="restarting" class="btn-spinner"></span>
              <span v-else>确认重启</span>
            </button>
          </div>
        </div>
      </div>
    </transition>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useSettingsStore } from '@/stores'

// Import all settings components
import WiFiSettings from '@/components/settings/WiFiSettings.vue'
import LLMConfig from '@/components/settings/LLMConfig.vue'
import VisionConfig from '@/components/settings/VisionConfig.vue'
import CameraConfig from '@/components/settings/CameraConfig.vue'
import SpeechConfig from '@/components/settings/SpeechConfig.vue'
import HardwareConfig from '@/components/settings/HardwareConfig.vue'
import BehaviorConfig from '@/components/settings/BehaviorConfig.vue'
import UIConfig from '@/components/settings/UIConfig.vue'
import AccountSettings from '@/components/settings/AccountSettings.vue'
import { triggerRestart } from '@/api/settings'

const router = useRouter()
const route = useRoute()
const settingsStore = useSettingsStore()

const activeTab = ref('wifi')
const lampId = ref<string>('')
const restartDialogVisible = ref(false)
const restartCountdown = ref(0)
const restarting = ref(false)
const shouldShake = ref(false)

const tabs = [
  { key: 'wifi', emoji: '📶', label: 'WiFi' },
  { key: 'llm', emoji: '🧠', label: 'AI 大脑' },
  { key: 'vision', emoji: '👁️', label: '视觉识别' },
  { key: 'camera', emoji: '📷', label: '摄像头' },
  { key: 'speech', emoji: '🎤', label: '语音' },
  { key: 'hardware', emoji: '🔧', label: '硬件' },
  { key: 'behavior', emoji: '🎭', label: '性格' },
  { key: 'ui', emoji: '🎨', label: '界面' },
  { key: 'account', emoji: '👤', label: '账户' }
]

const components = {
  wifi: WiFiSettings,
  llm: LLMConfig,
  vision: VisionConfig,
  camera: CameraConfig,
  speech: SpeechConfig,
  hardware: HardwareConfig,
  behavior: BehaviorConfig,
  ui: UIConfig,
  account: AccountSettings
}

const currentComponent = computed(() => components[activeTab.value as keyof typeof components])

onMounted(async () => {
  lampId.value = (route.query.lamp_id as string) || 'lelamp'
  settingsStore.setLampId(lampId.value)
  activeTab.value = (route.query.tab as string) || 'wifi'

  try {
    await settingsStore.fetchSettings()
    await settingsStore.fetchWiFiStatus()
  } catch (e) {
    ElMessage.error('加载设置失败')
  }
})

function handleTabSelect(key: string) {
  activeTab.value = key
}

function handleBack() {
  router.push('/room')
}

function showRestartDialog() {
  restartDialogVisible.value = true
  restartCountdown.value = 3
  const timer = setInterval(() => {
    restartCountdown.value--
    if (restartCountdown.value <= 0) {
      clearInterval(timer)
    }
  }, 1000)
}

function cancelRestart() {
  restartDialogVisible.value = false
  restartCountdown.value = 0
}

async function confirmRestart() {
  restarting.value = true
  try {
    const result = await triggerRestart({ delay_seconds: 3, reason: '用户从设置页面发起重启' })
    ElMessage.success(result.message)
    restartDialogVisible.value = false
    setTimeout(() => {
      router.push('/connect')
    }, 2000)
  } catch (e) {
    ElMessage.error('重启请求失败')
    restarting.value = false
  }
}
</script>

<style lang="scss" scoped>
.settings-view {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  position: relative;
}

/* === Background === */
.settings-background {
  position: fixed;
  inset: 0;
  z-index: 0;
  pointer-events: none;
}

.bg-gradient {
  position: absolute;
  inset: 0;
  background: linear-gradient(180deg, var(--lelamp-bg-cream) 0%, #FFE8D6 100%);
}

.floating-gears {
  position: absolute;
  inset: 0;
  opacity: 0.03;
}

.gear {
  position: absolute;
  animation: spin 20s linear infinite;
  animation-delay: var(--delay);

  &.gear-1 {
    width: 120px;
    height: 120px;
    top: 10%;
    left: 5%;
    color: var(--lelamp-peach);
  }

  &.gear-2 {
    width: 80px;
    height: 80px;
    top: 60%;
    right: 10%;
    animation-direction: reverse;
    color: var(--lelamp-sunny);
  }

  &.gear-3 {
    width: 60px;
    height: 60px;
    bottom: 20%;
    left: 15%;
    color: var(--lelamp-mint);
  }

  &.gear-4 {
    width: 100px;
    height: 100px;
    top: 30%;
    right: 25%;
    color: var(--lelamp-sky);
  }
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

/* === Header === */
.settings-header {
  position: relative;
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
  max-width: 1200px;
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

  &:active {
    transform: translateX(-1px);
  }
}

.header-info {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: var(--lelamp-space-xs);
}

.header-title {
  display: flex;
  align-items: center;
  gap: var(--lelamp-space-sm);
  font-family: var(--lelamp-font-display);
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--lelamp-text-primary);
  margin: 0;
}

.title-emoji {
  animation: float 3s ease-in-out infinite;
}

.device-badge {
  display: inline-flex;
  align-items: center;
  gap: var(--lelamp-space-sm);
  padding: var(--lelamp-space-xs) var(--lelamp-space-md);
  background: rgba(255, 154, 118, 0.1);
  border-radius: var(--lelamp-radius-full);
  font-size: 0.813rem;
  color: var(--lelamp-peach-dark);
}

.badge-icon {
  font-size: 1rem;
}

.badge-text {
  font-weight: 600;
}

.badge-id {
  font-family: var(--lelamp-font-mono);
  font-size: 0.75rem;
  opacity: 0.7;
  padding-left: var(--lelamp-space-xs);
  border-left: 1px solid var(--lelamp-peach);
}

.header-spacer {
  width: 80px;
}

/* === Pending Alert === */
.pending-alert {
  position: relative;
  z-index: 10;
  padding: 0 var(--lelamp-space-lg);
  margin-top: var(--lelamp-space-md);
}

.pending-alert.shake {
  animation: shake 0.5s ease-in-out;
}

@keyframes shake {
  0%, 100% { transform: translateX(0); }
  25% { transform: translateX(-5px); }
  75% { transform: translateX(5px); }
}

.alert-content {
  display: flex;
  align-items: center;
  gap: var(--lelamp-space-md);
  padding: var(--lelamp-space-md) var(--lelamp-space-lg);
  background: linear-gradient(135deg, rgba(255, 217, 61, 0.2), rgba(255, 154, 118, 0.15));
  border: 2px solid var(--lelamp-sunny);
  border-radius: var(--lelamp-radius-lg);
  box-shadow: var(--lelamp-shadow-md);
}

.alert-icon {
  font-size: 1.5rem;
  animation: wiggle 2s ease-in-out infinite;
}

.alert-text {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.alert-title {
  font-weight: 700;
  font-size: 0.938rem;
  color: var(--lelamp-text-primary);
}

.alert-desc {
  font-size: 0.813rem;
  color: var(--lelamp-text-secondary);
}

.restart-btn {
  display: flex;
  align-items: center;
  gap: var(--lelamp-space-sm);
  padding: var(--lelamp-space-sm) var(--lelamp-space-lg);
  background: linear-gradient(135deg, var(--lelamp-sunny), var(--lelamp-peach));
  border: none;
  border-radius: var(--lelamp-radius-md);
  font-weight: 700;
  font-size: 0.875rem;
  color: var(--lelamp-bg-white);
  cursor: pointer;
  transition: all var(--lelamp-transition-bounce);
  white-space: nowrap;

  &:hover {
    transform: translateY(-2px);
    box-shadow: var(--lelamp-shadow-md);
  }

  &:active {
    transform: translateY(0);
  }
}

.btn-icon {
  font-size: 1rem;
}

/* === Main Content === */
.settings-main {
  position: relative;
  z-index: 1;
  flex: 1;
  display: flex;
  flex-direction: column;
  max-width: 1200px;
  width: 100%;
  margin: 0 auto;
  padding: var(--lelamp-space-lg);
}

/* === Navigation Tabs === */
.settings-nav {
  margin-bottom: var(--lelamp-space-lg);
}

.nav-tabs {
  display: flex;
  gap: var(--lelamp-space-sm);
  padding: var(--lelamp-space-sm);
  background: var(--lelamp-bg-white);
  border-radius: var(--lelamp-radius-xl);
  box-shadow: var(--lelamp-shadow-sm);
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;

  &::-webkit-scrollbar {
    display: none;
  }
}

.nav-tab {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--lelamp-space-xs);
  padding: var(--lelamp-space-sm) var(--lelamp-space-md);
  background: transparent;
  border: none;
  border-radius: var(--lelamp-radius-md);
  cursor: pointer;
  transition: all var(--lelamp-transition-bounce);
  white-space: nowrap;
  min-width: 70px;

  .tab-emoji {
    font-size: 1.5rem;
    transition: transform var(--lelamp-transition-normal);
  }

  .tab-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--lelamp-text-secondary);
  }

  &:hover {
    background: var(--lelamp-bg-gray);

    .tab-emoji {
      transform: scale(1.2);
    }
  }

  &.active {
    background: linear-gradient(135deg, var(--lelamp-peach-light), var(--lelamp-sunny-light));

    .tab-emoji {
      transform: scale(1.1);
    }

    .tab-label {
      color: var(--lelamp-peach-dark);
    }
  }
}

/* === Content Area === */
.settings-content {
  flex: 1;
  background: var(--lelamp-bg-white);
  border-radius: var(--lelamp-radius-xl);
  padding: var(--lelamp-space-xl);
  box-shadow: var(--lelamp-shadow-md);
  overflow: hidden;
  position: relative;
  z-index: 10;

  /* 确保内部 Element Plus 组件正常显示 */
  :deep(.el-card) {
    background: transparent;
    border: none;
    box-shadow: none;
  }

  :deep(.el-card__header) {
    background: transparent;
    border-bottom: 1px dashed rgba(0, 0, 0, 0.1);
    padding: var(--lelamp-space-md) 0;
  }

  :deep(.el-card__body) {
    padding: var(--lelamp-space-md) 0;
  }

  /* 修复描述列表样式 */
  :deep(.el-descriptions) {
    --el-description-border-color: rgba(0, 0, 0, 0.05);
  }

  :deep(.el-descriptions__label) {
    font-weight: 600;
    color: var(--lelamp-text-secondary);
  }

  /* 修复按钮样式 - 只对主要按钮应用 hover 效果 */
  :deep(.el-button:hover) {
    transform: translateY(-1px);
  }

  :deep(.el-button:active) {
    transform: translateY(0);
  }

  /* 修复进度条样式 */
  :deep(.el-progress) {
    --el-progress-border-radius: var(--lelamp-radius-full);
  }

  /* 修复表单样式 */
  :deep(.el-form-item__label) {
    font-weight: 600;
    color: var(--lelamp-text-primary);
  }

  :deep(.el-input__wrapper) {
    border-radius: var(--lelamp-radius-md) !important;
    transition: all var(--lelamp-transition-normal);

    &:hover {
      box-shadow: 0 0 0 2px var(--lelamp-peach-light) !important;
    }

    &.is-focus {
      box-shadow: 0 0 0 3px var(--lelamp-sunny) !important;
    }
  }

  :deep(.el-switch.is-checked .el-switch__core) {
    background-color: var(--lelamp-peach);
    border-color: var(--lelamp-peach);
  }

  :deep(.el-radio__input.is-checked .el-radio__inner) {
    background-color: var(--lelamp-peach);
    border-color: var(--lelamp-peach);
  }

  :deep(.el-radio__label) {
    color: var(--lelamp-text-primary);
  }
}

/* === Transitions === */
.slide-fade-enter-active {
  transition: all 0.3s ease-out;
}

.slide-fade-leave-active {
  transition: all 0.2s ease-in;
}

.slide-fade-enter-from {
  opacity: 0;
  transform: translateX(20px);
}

.slide-fade-leave-to {
  opacity: 0;
  transform: translateX(-20px);
}

/* === Dialog === */
.dialog-overlay {
  position: fixed;
  inset: 0;
  z-index: 1000;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: var(--lelamp-space-lg);
  background: rgba(0, 0, 0, 0.5);
  backdrop-filter: blur(4px);
}

.dialog-fade-enter-active,
.dialog-fade-leave-active {
  transition: opacity 0.2s ease;
}

.dialog-fade-enter-from,
.dialog-fade-leave-to {
  opacity: 0;
}

.dialog-box {
  width: 100%;
  max-width: 400px;
  background: var(--lelamp-bg-white);
  border-radius: var(--lelamp-radius-xl);
  padding: var(--lelamp-space-xl);
  box-shadow: var(--lelamp-shadow-xl);
  animation: bounce-in 0.3s ease-out;
}

.dialog-header {
  text-align: center;
  margin-bottom: var(--lelamp-space-lg);
}

.dialog-icon {
  font-size: 3rem;
  margin-bottom: var(--lelamp-space-sm);
}

.dialog-title {
  font-family: var(--lelamp-font-display);
  font-size: 1.25rem;
  font-weight: 700;
  color: var(--lelamp-text-primary);
  margin: 0 0 var(--lelamp-space-sm);
}

.dialog-desc {
  font-size: 0.938rem;
  color: var(--lelamp-text-secondary);
  margin: 0;
}

.countdown-box {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--lelamp-space-md);
  margin-bottom: var(--lelamp-space-lg);
}

.countdown-ring {
  position: relative;
  width: 100px;
  height: 100px;
}

.countdown-svg {
  width: 100%;
  height: 100%;
}

.countdown-progress {
  transition: stroke-dashoffset 1s linear;
}

.countdown-number {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  font-family: var(--lelamp-font-display);
  font-size: 2rem;
  font-weight: 700;
  color: var(--lelamp-peach);
}

.countdown-text {
  font-size: 0.875rem;
  color: var(--lelamp-text-secondary);
  margin: 0;
}

.dialog-actions {
  display: flex;
  gap: var(--lelamp-space-sm);
}

.dialog-btn {
  flex: 1;
  padding: var(--lelamp-space-md);
  font-size: 0.938rem;
  font-weight: 600;
  font-family: var(--lelamp-font-body);
  border: none;
  border-radius: var(--lelamp-radius-md);
  cursor: pointer;
  transition: all var(--lelamp-transition-bounce);

  &.cancel {
    background: var(--lelamp-bg-gray);
    color: var(--lelamp-text-secondary);

    &:hover:not(:disabled) {
      background: #e5e7eb;
    }
  }

  &.confirm {
    background: linear-gradient(135deg, var(--lelamp-coral), var(--lelamp-peach));
    color: var(--lelamp-bg-white);

    &:hover:not(:disabled) {
      transform: translateY(-2px);
      box-shadow: var(--lelamp-shadow-md);
    }

    &:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }
  }

  &:active:not(:disabled) {
    transform: translateY(0);
  }
}

.btn-spinner {
  display: inline-block;
  width: 16px;
  height: 16px;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-top-color: var(--lelamp-bg-white);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

/* === Responsive === */
@media (max-width: 768px) {
  .header-content {
    padding: var(--lelamp-space-sm) var(--lelamp-space-md);
  }

  .header-spacer {
    display: none;
  }

  .back-btn span {
    display: none;
  }

  .header-title {
    font-size: 1.25rem;
  }

  .settings-main {
    padding: var(--lelamp-space-md);
  }

  .nav-tabs {
    padding: var(--leamp-space-xs);
  }

  .nav-tab {
    min-width: 60px;
    padding: var(--leamp-space-xs) var(--leamp-space-sm);

    .tab-emoji {
      font-size: 1.25rem;
    }

    .tab-label {
      font-size: 0.688rem;
    }
  }

  .settings-content {
    padding: var(--lelamp-space-md);
  }

  .alert-content {
    flex-wrap: wrap;
    padding: var(--lelamp-space-sm) var(--lelamp-space-md);
  }

  .restart-btn span:not(.btn-icon) {
    display: none;
  }

  .dialog-overlay {
    padding: var(--lelamp-space-md);
  }
}
</style>
