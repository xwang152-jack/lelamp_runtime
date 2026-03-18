<template>
  <div class="room-view">
    <div class="room-header">
      <div class="status">
        <span :class="['status-dot', connectionStatusClass]" />
        <span>{{ connectionStatusText }}</span>
      </div>
      <div class="header-actions">
        <el-button @click="handleSettings">
          <el-icon><Setting /></el-icon>
          设置
        </el-button>
        <el-button type="danger" @click="handleDisconnect"> 断开连接 </el-button>
      </div>
    </div>

    <div class="room-content">
      <div class="control-section">
        <div class="panel">
          <h3>⚡ 快捷操作</h3>
          <div class="button-grid">
            <el-button @click="sendChatAction('你好')"> 👋 打招呼 </el-button>
            <el-button @click="sendChatAction('现在几点了')"> ⏰ 查看时间 </el-button>
            <el-button @click="sendChatAction('讲个笑话')"> 😄 讲笑话 </el-button>
            <el-button @click="sendChatAction('唱首歌')"> 🎵 唱歌 </el-button>
          </div>
        </div>

        <LightPanel />
      </div>

      <div class="chat-section">
        <h3>💬 实时对话</h3>
        <div ref="messagesContainer" class="messages">
          <div v-for="msg in messages" :key="msg.id" :class="['message', msg.sender]">
            <div class="message-content">
              {{ msg.content }}
            </div>
            <div class="message-time">
              {{ formatTime(msg.timestamp) }}
            </div>
          </div>
          <div v-if="messages.length === 0" class="empty-hint">开始对话吧...</div>
        </div>
        <div class="input-area">
          <el-input v-model="inputText" placeholder="输入消息..." @keyup.enter="sendMessage" />
          <el-button type="primary" @click="sendMessage"> 发送 </el-button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, nextTick, watch } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { Setting } from '@element-plus/icons-vue'
import { useWebSocket } from '@/composables/useWebSocket'
import { useChatStore, useConnectionStore } from '@/stores'
import LightPanel from '@/components/room/LightPanel.vue'

const router = useRouter()
const { disconnect, sendChat: sendWSChat } = useWebSocket()
const chatStore = useChatStore()
const connectionStore = useConnectionStore()

const inputText = ref('')
const messagesContainer = ref<HTMLElement>()

const messages = computed(() => chatStore.messages)

// 连接状态显示
const connectionStatusClass = computed(() => {
  switch (connectionStore.connectionStatus) {
    case 'connected': return 'online'
    case 'connecting': return 'connecting'
    case 'disconnected': return 'offline'
    case 'error': return 'error'
    default: return 'offline'
  }
})

const connectionStatusText = computed(() => {
  switch (connectionStore.connectionStatus) {
    case 'connected': return '已连接'
    case 'connecting': return '连接中...'
    case 'disconnected': return '未连接'
    case 'error': return '连接错误'
    default: return '未知状态'
  }
})

function scrollToBottom() {
  nextTick(() => {
    if (messagesContainer.value) {
      messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
    }
  })
}

async function handleDisconnect() {
  await disconnect()
  router.push('/connect')
}

function handleSettings() {
  router.push('/settings')
}

function sendChatAction(text: string) {
  sendWSChat(text)
  // 不再显示虚假成功消息，等待后端响应
  scrollToBottom()
}

function sendMessage() {
  if (!inputText.value.trim()) return
  sendChatAction(inputText.value)
  inputText.value = ''
}

function formatTime(timestamp: number): string {
  const date = new Date(timestamp)
  return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
}

onMounted(() => {
  scrollToBottom()
})

watch(
  () => chatStore.messages,
  () => {
    scrollToBottom()
  },
  { deep: true }
)
</script>

<style lang="scss" scoped>
.room-view {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: var(--lelamp-bg);
}

.room-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 24px;
  background: white;
  box-shadow: var(--shadow-sm);

  .status {
    display: flex;
    align-items: center;
    gap: 8px;

    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      &.online {
        background: var(--color-success);
      }
    }
  }

  .header-actions {
    display: flex;
    gap: 8px;
  }
}

.room-content {
  display: grid;
  grid-template-columns: 400px 1fr;
  gap: 24px;
  padding: 24px;
  flex: 1;
  overflow: hidden;
}

.control-section {
  display: flex;
  flex-direction: column;
  gap: 16px;
  overflow-y: auto;
}

.panel {
  padding: 20px;
  background: white;
  border-radius: var(--radius-md);
  box-shadow: var(--shadow-sm);

  h3 {
    margin-bottom: 16px;
  }
}

.button-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}

.chat-section {
  display: flex;
  flex-direction: column;
  padding: 20px;
  background: white;
  border-radius: var(--radius-md);
  box-shadow: var(--shadow-sm);

  h3 {
    margin-bottom: 16px;
  }

  .messages {
    flex: 1;
    overflow-y: auto;
    margin-bottom: 16px;
    padding: 16px;
    background: #f5f7fa;
    border-radius: 8px;

    .message {
      margin-bottom: 12px;

      .message-content {
        padding: 8px 12px;
        border-radius: 8px;
        max-width: 70%;
        word-break: break-word;
      }

      .message-time {
        font-size: 11px;
        color: #999;
        margin-top: 4px;
      }

      &.user {
        text-align: right;

        .message-content {
          background: var(--lelamp-brand);
          color: white;
          margin-left: auto;
        }
      }

      &.agent {
        text-align: left;

        .message-content {
          background: white;
          color: #333;
          box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
        }
      }
    }

    .empty-hint {
      text-align: center;
      color: #999;
      font-size: 14px;
      padding: 40px 0;
    }
  }

  .input-area {
    display: flex;
    gap: 8px;
  }
}
</style>
