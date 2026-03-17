<template>
  <div class="privacy-indicator">
    <div class="privacy-header">
      <span>🔒 隐私保护</span>
    </div>
    <div class="camera-status">
      <span
        class="camera-led"
        :class="{ active: isActive }"
      />
      <span class="status-text">{{ isActive ? '摄像头已开启' : '摄像头已关闭' }}</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

interface Props {
  active: boolean
}

const props = defineProps<Props>()
const isActive = computed(() => props.active)
</script>

<style lang="scss" scoped>
.privacy-indicator {
  padding: 12px 16px;
  background: rgba(0, 0, 0, 0.7);
  border-radius: 8px;
  color: white;
  font-size: 14px;

  .privacy-header {
    font-weight: 500;
    margin-bottom: 6px;
  }

  .camera-status {
    display: flex;
    align-items: center;
    gap: 8px;

    .camera-led {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: #666;
      transition: all 0.3s;

      &.active {
        background: #ff4444;
        box-shadow: 0 0 10px #ff4444;
        animation: pulse 2s infinite;
      }
    }

    .status-text {
      font-size: 13px;
      opacity: 0.9;
    }
  }
}

@keyframes pulse {
  0%,
  100% {
    opacity: 1;
  }
  50% {
    opacity: 0.5;
  }
}
</style>
