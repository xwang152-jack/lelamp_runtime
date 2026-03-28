<template>
  <div class="camera-panel">
    <div class="camera-header">
      <h3 class="camera-title">
        <span class="title-icon">📷</span>
        摄像头
      </h3>
      <div class="camera-status" :class="statusClass">
        <span class="status-dot"></span>
        <span class="status-text">{{ statusText }}</span>
      </div>
    </div>

    <div class="camera-viewport">
      <!-- 激活状态 - 显示实时画面 -->
      <img
        v-if="cameraActive && currentFrame"
        :src="frameSrc"
        :alt="'摄像头画面'"
        class="camera-frame"
        @error="handleFrameError"
      />

      <!-- 未激活状态 - 显示隐私占位符 -->
      <div v-else class="camera-placeholder">
        <div class="placeholder-content">
          <span class="placeholder-icon">📷</span>
          <p class="placeholder-text">
            {{ cameraActive ? '等待画面...' : '摄像头未激活' }}
          </p>
          <button
            v-if="!cameraActive"
            class="activate-btn"
            @click="requestCameraActivation"
          >
            <span class="btn-icon">🔓</span>
            <span>开启摄像头</span>
          </button>
        </div>
        <div class="placeholder-bg">
          <div v-for="i in 8" :key="i" class="bg-line" :style="{ '--delay': `${i * 0.1}s` }"></div>
        </div>
      </div>

      <!-- 加载指示器 -->
      <div v-if="cameraActive && !currentFrame" class="camera-loading">
        <div class="loading-spinner"></div>
      </div>
    </div>

    <!-- 画面信息 -->
    <div v-if="cameraActive && (frameInfo.width || frameInfo.height)" class="camera-info">
      <span class="info-item">{{ frameInfo.width }}x{{ frameInfo.height }}</span>
      <span class="info-item" v-if="frameInfo.timestamp">{{ formatTime(frameInfo.timestamp) }}</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { useDeviceStore } from '@/stores'
import { useWebSocket } from '@/composables/useWebSocket'

const deviceStore = useDeviceStore()
const { sendCommand } = useWebSocket()

const currentFrame = ref<string | null>(null)
const frameInfo = ref<{ width?: number; height?: number; timestamp?: number }>({})
const frameError = ref(false)

// 监听摄像头状态
const cameraActive = computed(() => deviceStore.cameraActive)

// 计算状态类
const statusClass = computed(() => {
  if (!cameraActive.value) return 'inactive'
  if (frameError.value) return 'error'
  if (!currentFrame.value) return 'loading'
  return 'active'
})

// 状态文本
const statusText = computed(() => {
  if (!cameraActive.value) return '未激活'
  if (frameError.value) return '错误'
  if (!currentFrame.value) return '加载中'
  return '实时'
})

// 图片源
const frameSrc = computed(() => {
  if (!currentFrame.value) return ''
  return `data:image/jpeg;base64,${currentFrame.value}`
})

// 请求激活摄像头
function requestCameraActivation() {
  sendCommand('activate_camera', {})
}

// 处理帧错误
function handleFrameError() {
  frameError.value = true
  console.error('Camera frame failed to load')
  // 3秒后重置错误状态
  setTimeout(() => {
    frameError.value = false
  }, 3000)
}

// 格式化时间
function formatTime(timestamp: number): string {
  const date = new Date(timestamp)
  return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

// 暴露更新画面的方法（供外部调用）
function updateFrame(frameB64: string, info: { width?: number; height?: number; timestamp?: number }) {
  currentFrame.value = frameB64
  frameInfo.value = info
  frameError.value = false
}

// 清除画面
function clearFrame() {
  currentFrame.value = null
  frameInfo.value = {}
}

// 监听摄像头状态变化，清空画面
watch(cameraActive, (active) => {
  if (!active) {
    clearFrame()
  }
})

defineExpose({
  updateFrame,
  clearFrame
})
</script>

<style lang="scss" scoped>
.camera-panel {
  display: flex;
  flex-direction: column;
  gap: var(--lelamp-space-sm);
}

/* === Header === */
.camera-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: var(--lelamp-space-md) var(--lelamp-space-lg);
  background: var(--lelamp-bg-white);
  border-radius: var(--lelamp-radius-xl) var(--lelamp-radius-xl) 0 0;
  border-bottom: 1px solid rgba(0, 0, 0, 0.05);
}

.camera-title {
  display: flex;
  align-items: center;
  gap: var(--lelamp-space-sm);
  font-family: var(--lelamp-font-display);
  font-size: 1.125rem;
  font-weight: 700;
  color: var(--lelamp-text-primary);
}

.title-icon {
  font-size: 1.25rem;
}

.camera-status {
  display: flex;
  align-items: center;
  gap: var(--lelamp-space-sm);
  padding: var(--lelamp-space-xs) var(--lelamp-space-md);
  border-radius: var(--lelamp-radius-full);
  font-size: 0.75rem;
  font-weight: 600;

  &.active {
    background: rgba(107, 203, 119, 0.15);
    color: var(--lelamp-mint-dark);
    .status-dot { background: var(--lelamp-mint); }
  }

  &.loading {
    background: rgba(255, 217, 61, 0.15);
    color: var(--lelamp-sunny-dark);
    .status-dot {
      background: var(--lelamp-sunny);
      animation: pulse 1s ease-in-out infinite;
    }
  }

  &.inactive {
    background: rgba(150, 150, 150, 0.15);
    color: var(--lelamp-text-secondary);
    .status-dot { background: var(--lelamp-text-tertiary); }
  }

  &.error {
    background: rgba(255, 107, 138, 0.15);
    color: var(--lelamp-coral-dark);
    .status-dot { background: var(--lelamp-coral); }
  }
}

.status-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

/* === Viewport === */
.camera-viewport {
  position: relative;
  width: 100%;
  aspect-ratio: 16 / 9;
  background: var(--lelamp-bg-gray);
  border-radius: 0 0 var(--lelamp-radius-xl) var(--lelamp-radius-xl);
  overflow: hidden;
  box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.05);
}

.camera-frame {
  width: 100%;
  height: 100%;
  object-fit: contain;
  background: #000;
}

/* === Placeholder === */
.camera-placeholder {
  position: relative;
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #f5f7fa 0%, #e8eef5 100%);
}

.placeholder-content {
  position: relative;
  z-index: 2;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--lelamp-space-md);
  text-align: center;
}

.placeholder-icon {
  font-size: 3rem;
  opacity: 0.4;
}

.placeholder-text {
  font-size: 0.938rem;
  color: var(--lelamp-text-secondary);
  margin: 0;
}

.activate-btn {
  display: flex;
  align-items: center;
  gap: var(--lelamp-space-sm);
  padding: var(--lelamp-space-sm) var(--lelamp-space-lg);
  background: linear-gradient(135deg, var(--lelamp-peach), var(--lelamp-coral));
  border: none;
  border-radius: var(--lelamp-radius-full);
  font-size: 0.875rem;
  font-weight: 600;
  color: var(--lelamp-bg-white);
  cursor: pointer;
  transition: all var(--lelamp-transition-bounce);
  box-shadow: 0 2px 8px rgba(255, 107, 138, 0.3);

  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(255, 107, 138, 0.4);
  }

  &:active {
    transform: translateY(0);
  }
}

.btn-icon {
  font-size: 1rem;
}

.placeholder-bg {
  position: absolute;
  inset: 0;
  overflow: hidden;
  opacity: 0.5;
}

.bg-line {
  position: absolute;
  height: 1px;
  width: 100%;
  background: linear-gradient(90deg, transparent, rgba(255, 154, 118, 0.3), transparent);
  animation: scan 3s ease-in-out infinite;
  animation-delay: var(--delay);

  &:nth-child(1) { top: 10%; }
  &:nth-child(2) { top: 25%; }
  &:nth-child(3) { top: 40%; }
  &:nth-child(4) { top: 55%; }
  &:nth-child(5) { top: 70%; }
  &:nth-child(6) { top: 85%; }
  &:nth-child(7) { top: 95%; }
  &:nth-child(8) { top: 50%; }
}

@keyframes scan {
  0% { transform: translateX(-100%); opacity: 0; }
  50% { opacity: 1; }
  100% { transform: translateX(100%); opacity: 0; }
}

/* === Loading === */
.camera-loading {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(0, 0, 0, 0.05);
}

.loading-spinner {
  width: 40px;
  height: 40px;
  border: 3px solid var(--lelamp-bg-gray);
  border-top-color: var(--lelamp-peach);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

/* === Info Bar === */
.camera-info {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--lelamp-space-md);
  padding: var(--lelamp-space-xs) var(--lelamp-space-md);
  background: var(--lelamp-bg-white);
  border-radius: var(--lelamp-radius-lg);
  font-size: 0.75rem;
  color: var(--lelamp-text-tertiary);
}

.info-item {
  display: flex;
  align-items: center;
  gap: 4px;

  &::before {
    content: '';
    width: 4px;
    height: 4px;
    border-radius: 50%;
    background: var(--lelamp-peach);
  }
}

/* === Responsive === */
@media (max-width: 900px) {
  .camera-viewport {
    aspect-ratio: 4 / 3;
  }
}

@media (max-width: 480px) {
  .camera-header {
    padding: var(--lelamp-space-sm) var(--lelamp-space-md);
  }

  .camera-title {
    font-size: 1rem;
  }

  .placeholder-icon {
    font-size: 2.5rem;
  }

  .placeholder-text {
    font-size: 0.813rem;
  }
}
</style>
