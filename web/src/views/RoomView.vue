<template>
  <div class="room-view">
    <div class="room-header">
      <div class="status">
        <span class="status-dot online" />
        <span>已连接</span>
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
      <div class="video-section">
        <div class="video-placeholder">📹</div>
        <p>等待摄像头画面...</p>
        <PrivacyIndicator :active="cameraActive" />
      </div>

      <div class="control-section">
        <div class="panel">
          <h3>⚡ 快捷操作</h3>
          <div class="button-grid">
            <el-button @click="sendChat('你好')"> 👋 打招呼 </el-button>
            <el-button @click="sendChat('现在几点了')"> ⏰ 查看时间 </el-button>
            <el-button @click="sendChat('讲个笑话')"> 😄 讲笑话 </el-button>
            <el-button @click="sendChat('唱首歌')"> 🎵 唱歌 </el-button>
          </div>
        </div>

        <LightPanel />
      </div>
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
</template>

<script setup lang="ts">
import { ref, computed, onMounted, nextTick, watch } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { Setting } from '@element-plus/icons-vue'
import { useLiveKit } from '@/composables/useLiveKit'
import { useChatStore, useDeviceStore } from '@/stores'
import PrivacyIndicator from '@/components/common/PrivacyIndicator.vue'
import LightPanel from '@/components/room/LightPanel.vue'

const router = useRouter()
const { disconnect, sendChat: sendLiveKitChat } = useLiveKit()
const chatStore = useChatStore()
const deviceStore = useDeviceStore()

const inputText = ref('')
const messagesContainer = ref<HTMLElement>()

const messages = computed(() => chatStore.messages)
const cameraActive = computed(() => deviceStore.cameraActive)

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

function sendChat(text: string) {
  sendLiveKitChat(text)
  ElMessage.success(`发送: ${text}`)
  scrollToBottom()
}

function sendMessage() {
  if (!inputText.value.trim()) return
  sendChat(inputText.value)
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
      }
    }
  }
}

.room-content {
  display: grid;
  grid-template-columns: 1fr 400px;
  gap: 24px;
  padding: 24px;
  flex: 1;
  overflow: hidden;
}

.video-section {
  background: black;
  border-radius: var(--radius-lg);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: white;

  .video-placeholder {
    font-size: 64px;
  }
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
  padding: 16px 24px;
  background: white;
  border-top: 1px solid #eee;

  h3 {
    margin-bottom: 12px;
  }

  .messages {
    height: 150px;
    overflow-y: auto;
    margin-bottom: 12px;
    padding: 12px;
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
