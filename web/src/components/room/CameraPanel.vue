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
        @load="drawDetections"
      />
      <!-- 检测结果标注层 -->
      <canvas
        v-if="cameraActive && currentFrame"
        ref="overlayCanvas"
        class="detection-overlay"
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
const lastDetections = ref<any>(null)
const overlayCanvas = ref<HTMLCanvasElement | null>(null)

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
function updateFrame(
  frameB64: string,
  info: { width?: number; height?: number; timestamp?: number },
  detections?: any
) {
  currentFrame.value = frameB64
  frameInfo.value = info
  frameError.value = false
  lastDetections.value = detections || null
}

// 清除画面
function clearFrame() {
  currentFrame.value = null
  frameInfo.value = {}
  lastDetections.value = null
}

// 在 canvas overlay 上绘制检测结果标注
function drawDetections() {
  const canvas = overlayCanvas.value
  const img = document.querySelector('.camera-frame') as HTMLImageElement
  if (!canvas || !img || !lastDetections.value) return

  // 使 canvas 尺寸匹配 img 显示尺寸
  const rect = img.getBoundingClientRect()
  canvas.width = rect.width
  canvas.height = rect.height

  const ctx = canvas.getContext('2d')
  if (!ctx) return

  ctx.clearRect(0, 0, canvas.width, canvas.height)

  const det = lastDetections.value
  const W = canvas.width
  const H = canvas.height

  // ========== 人脸：矩形框 ==========
  if (det.faces?.length) {
    for (const face of det.faces) {
      if (face.x == null || face.y == null) continue
      const fx = face.x * W
      const fy = face.y * H
      const fw = (face.w ?? 0) * W
      const fh = (face.h ?? 0) * H

      // 绿色矩形框
      ctx.strokeStyle = '#4ade80'
      ctx.lineWidth = 2
      ctx.strokeRect(fx, fy, fw, fh)

      // 角标记（更醒目）
      const cornerLen = Math.min(fw, fh) * 0.15
      ctx.lineWidth = 3
      ctx.strokeStyle = '#22c55e'
      // 左上
      ctx.beginPath(); ctx.moveTo(fx, fy + cornerLen); ctx.lineTo(fx, fy); ctx.lineTo(fx + cornerLen, fy); ctx.stroke()
      // 右上
      ctx.beginPath(); ctx.moveTo(fx + fw - cornerLen, fy); ctx.lineTo(fx + fw, fy); ctx.lineTo(fx + fw, fy + cornerLen); ctx.stroke()
      // 左下
      ctx.beginPath(); ctx.moveTo(fx, fy + fh - cornerLen); ctx.lineTo(fx, fy + fh); ctx.lineTo(fx + cornerLen, fy + fh); ctx.stroke()
      // 右下
      ctx.beginPath(); ctx.moveTo(fx + fw - cornerLen, fy + fh); ctx.lineTo(fx + fw, fy + fh); ctx.lineTo(fx + fw, fy + fh - cornerLen); ctx.stroke()

      // 置信度标签
      ctx.font = '11px monospace'
      ctx.fillStyle = '#22c55e'
      const label = `Face ${(face.confidence * 100).toFixed(0)}%`
      const textW = ctx.measureText(label).width
      ctx.fillStyle = 'rgba(0,0,0,0.5)'
      ctx.fillRect(fx, fy - 16, textW + 8, 16)
      ctx.fillStyle = '#4ade80'
      ctx.fillText(label, fx + 4, fy - 4)
    }
  }

  // ========== 手部：关节连线和关键点 ==========
  if (det.hands?.length) {
    // 手部骨骼连接关系（MediaPipe 21 点标准）
    const CONNECTIONS = [
      // 手指
      [0,1],[1,2],[2,3],[3,4],         // 拇指
      [0,5],[5,6],[6,7],[7,8],         // 食指
      [0,9],[9,10],[10,11],[11,12],    // 中指
      [0,13],[13,14],[14,15],[15,16],  // 无名指
      [0,17],[17,18],[18,19],[19,20],  // 小指
      // 掌骨横连
      [5,9],[9,13],[13,17],
    ]

    for (const hand of det.hands) {
      if (!hand.landmarks?.length) continue
      const pts = hand.landmarks

      // 绘制骨骼连线
      ctx.strokeStyle = '#22d3ee'
      ctx.lineWidth = 1.5
      for (const [a, b] of CONNECTIONS) {
        if (a < pts.length && b < pts.length) {
          ctx.beginPath()
          ctx.moveTo(pts[a].x * W, pts[a].y * H)
          ctx.lineTo(pts[b].x * W, pts[b].y * H)
          ctx.stroke()
        }
      }

      // 绘制关键点
      for (let i = 0; i < pts.length; i++) {
        const px = pts[i].x * W
        const py = pts[i].y * H
        // 指尖用大圆点，其他用小圆点
        const isTip = [4, 8, 12, 16, 20].includes(i)
        const r = isTip ? 4 : 2.5
        ctx.beginPath()
        ctx.arc(px, py, r, 0, Math.PI * 2)
        ctx.fillStyle = isTip ? '#06b6d4' : '#67e8f9'
        ctx.fill()
      }

      // 手势标签（手腕位置上方）
      const wrist = pts[0]
      if (hand.gesture) {
        ctx.font = 'bold 12px sans-serif'
        const gestureLabel = hand.gesture.replace(/_/g, ' ').toUpperCase()
        const gestureW = ctx.measureText(gestureLabel).width
        const gx = wrist.x * W - gestureW / 2
        const gy = wrist.y * H + 18
        ctx.fillStyle = 'rgba(0,0,0,0.6)'
        ctx.fillRect(gx - 4, gy - 12, gestureW + 8, 16)
        ctx.fillStyle = '#22d3ee'
        ctx.fillText(gestureLabel, gx, gy)

        // 左右手标识
        if (hand.handedness) {
          ctx.font = '10px sans-serif'
          ctx.fillStyle = '#a5f3fc'
          ctx.fillText(hand.handedness, wrist.x * W - 10, gy + 14)
        }
      }
    }
  }

  // ========== 在场状态指示器（右上角）==========
  const present = det.presence
  ctx.beginPath()
  ctx.arc(W - 16, 16, 6, 0, Math.PI * 2)
  ctx.fillStyle = present ? '#4ade80' : '#f87171'
  ctx.fill()
  ctx.strokeStyle = 'rgba(255,255,255,0.8)'
  ctx.lineWidth = 1
  ctx.stroke()
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

.detection-overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  z-index: 2;
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
