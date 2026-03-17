<template>
  <div class="status-indicator">
    <span
      class="status-dot"
      :class="statusClass"
    />
    <span class="status-text">{{ statusText }}</span>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

interface Props {
  status: 'online' | 'offline' | 'connecting' | 'error'
}

const props = defineProps<Props>()

const statusClass = computed(() => props.status)
const statusText = computed(() => {
  const map = {
    online: '在线',
    offline: '离线',
    connecting: '连接中',
    error: '错误'
  }
  return map[props.status]
})
</script>

<style lang="scss" scoped>
.status-indicator {
  display: flex;
  align-items: center;
  gap: 6px;

  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;

    &.online {
      background: var(--color-success);
    }
    &.offline {
      background: var(--color-info);
    }
    &.connecting {
      background: var(--color-warning);
      animation: blink 1s infinite;
    }
    &.error {
      background: var(--color-danger);
    }
  }

  .status-text {
    font-size: 14px;
    color: #606266;
  }
}

@keyframes blink {
  0%,
  100% {
    opacity: 1;
  }
  50% {
    opacity: 0.3;
  }
}
</style>
