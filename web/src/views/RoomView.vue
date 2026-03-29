<template>
  <div class="room-view">
    <!-- Animated Background -->
    <div class="room-background">
      <div class="bg-gradient"></div>
      <div class="floating-lights">
        <div v-for="i in 6" :key="i" class="light-bubble" :style="{ '--delay': `${i * 0.5}s` }"></div>
      </div>
    </div>

    <!-- Header -->
    <header class="room-header">
      <div class="header-left">
        <div class="lelamp-logo">
          <span class="logo-emoji">🪔</span>
          <span class="logo-text">LeLamp</span>
        </div>
        <div class="connection-badge" :class="statusClass">
          <div class="badge-dot"></div>
          <span class="badge-text">{{ statusText }}</span>
        </div>
      </div>
      <div class="header-actions">
        <!-- User Avatar (when logged in) -->
        <button
          v-if="isAuthenticated"
          class="user-avatar-btn"
          @click="handleProfileClick"
          :title="authStore.user?.username || '个人中心'"
        >
          <span class="avatar-text">{{ userInitial }}</span>
        </button>
        <button class="icon-btn" @click="handleSettings" title="设置">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="3" />
            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
          </svg>
        </button>
        <button class="icon-btn danger" @click="handleDisconnect" title="断开连接">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
            <polyline points="16 17 21 12 16 7" />
            <line x1="21" y1="12" x2="9" y2="12" />
          </svg>
        </button>
      </div>
    </header>

    <!-- Main Content -->
    <main class="room-main">
      <!-- Left Panel - Controls -->
      <div class="control-panel">
        <!-- Quick Actions -->
        <section class="panel-section">
          <h3 class="section-title">
            <span class="title-icon">⚡</span>
            快捷指令
          </h3>
          <div class="action-grid">
            <button
              v-for="action in quickActions"
              :key="action.id"
              class="action-card"
              @click="sendChatAction(action.text)"
            >
              <span class="action-emoji">{{ action.emoji }}</span>
              <span class="action-label">{{ action.label }}</span>
            </button>
          </div>
        </section>

        <!-- Light Control -->
        <section class="panel-section">
          <h3 class="section-title">
            <span class="title-icon">💡</span>
            灯光魔法
          </h3>
          <LightPanel />
        </section>
      </div>

      <!-- Right Panel - Camera & Chat -->
      <div class="right-panel">
        <!-- Camera Panel -->
        <CameraPanel ref="cameraPanelRef" />

        <!-- Chat Panel -->
        <div class="chat-panel">
          <div class="chat-header">
          <h3 class="chat-title">
            <span class="title-icon">💬</span>
            对话
          </h3>
          <div class="typing-indicator" v-if="isTyping">
            <span></span>
            <span></span>
            <span></span>
          </div>
        </div>

        <!-- Messages -->
        <div ref="messagesContainer" class="messages-container">
          <div v-if="messages.length === 0" class="empty-state">
            <div class="empty-emoji">🤖</div>
            <p>和我打个招呼吧~</p>
          </div>
          <div
            v-for="msg in messages"
            :key="msg.id"
            :class="['message', msg.sender]"
          >
            <div class="message-avatar">
              <span v-if="msg.sender === 'user'">👤</span>
              <span v-else>🪔</span>
            </div>
            <div class="message-content">
              <p>{{ msg.content }}</p>
            </div>
            <div class="message-time">{{ formatTime(msg.timestamp) }}</div>
          </div>
        </div>

        <!-- Input -->
        <div class="chat-input">
          <input
            v-model="inputText"
            type="text"
            class="message-input"
            placeholder="说点什么..."
            @keyup.enter="sendMessage"
          />
          <button
            class="send-btn"
            :class="{ 'has-content': inputText.trim() }"
            @click="sendMessage"
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <line x1="22" y1="2" x2="11" y2="13" />
              <polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
          </button>
        </div>
      </div>
      </div>
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, nextTick, watch, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { useChatStore, useConnectionStore, useDeviceStore, useAuthStore } from '@/stores'
import { useWebSocket } from '@/composables/useWebSocket'
import LightPanel from '@/components/room/LightPanel.vue'
import CameraPanel from '@/components/room/CameraPanel.vue'
import { getUserInitial } from '@/utils/device'

const router = useRouter()
const { disconnect, sendChat: sendWSChat, onCameraFrame, offCameraFrame } = useWebSocket()
const chatStore = useChatStore()
const connectionStore = useConnectionStore()
const deviceStore = useDeviceStore()
const authStore = useAuthStore()

const inputText = ref('')
const messagesContainer = ref<HTMLElement>()
const cameraPanelRef = ref<InstanceType<typeof CameraPanel> | null>(null)

const quickActions = [
  { id: 1, emoji: '👋', label: '打招呼', text: '你好' },
  { id: 2, emoji: '⏰', label: '几点了', text: '现在几点了' },
  { id: 3, emoji: '😄', label: '讲笑话', text: '讲个笑话' },
  { id: 4, emoji: '🎵', label: '唱首歌', text: '唱首歌' },
  { id: 5, emoji: '🌈', label: '彩虹灯', text: '帮我开彩虹灯' },
  { id: 6, emoji: '😴', label: '晚安', text: '晚安' }
]

const messages = computed(() => chatStore.messages)

const isTyping = computed(() => deviceStore.conversationState === 'thinking')

const statusClass = computed(() => connectionStore.connectionStatus)
const statusText = computed(() => {
  switch (connectionStore.connectionStatus) {
    case 'connected': return '已连接'
    case 'connecting': return '连接中...'
    case 'disconnected': return '未连接'
    case 'error': return '连接错误'
    default: return '未知'
  }
})

// Auth computed properties
const isAuthenticated = computed(() => authStore.isAuthenticated)
const userInitial = computed(() => {
  return getUserInitial(authStore.user?.username)
})

function scrollToBottom() {
  nextTick(() => {
    if (messagesContainer.value) {
      messagesContainer.value.scrollTo({
        top: messagesContainer.value.scrollHeight,
        behavior: 'smooth'
      })
    }
  })
}

async function handleDisconnect() {
  await disconnect()
  sessionStorage.removeItem('lelamp_connected')
  router.push('/connect')
}

function handleSettings() {
  router.push('/settings')
}

function handleProfileClick() {
  router.push('/profile')
}

function sendChatAction(text: string) {
  sendWSChat(text)
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

  // 注册摄像头帧回调
  onCameraFrame((frameB64, info, detections) => {
    if (cameraPanelRef.value) {
      cameraPanelRef.value.updateFrame(frameB64, info, detections)
    }
  })
})

onUnmounted(() => {
  // 清理摄像头帧回调
  offCameraFrame()
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
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  position: relative;
}

/* === Background === */
.room-background {
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

.floating-lights {
  position: absolute;
  inset: 0;
}

.light-bubble {
  position: absolute;
  border-radius: 50%;
  background: radial-gradient(circle, var(--lelamp-peach-light) 0%, transparent 70%);
  opacity: 0.4;
  animation: float 8s ease-in-out infinite;
  animation-delay: var(--delay);

  &:nth-child(1) { width: 200px; height: 200px; top: 10%; left: 5%; }
  &:nth-child(2) { width: 150px; height: 150px; top: 60%; right: 10%; animation-delay: 2s; }
  &:nth-child(3) { width: 100px; height: 100px; top: 30%; right: 30%; animation-delay: 4s; }
  &:nth-child(4) { width: 180px; height: 180px; bottom: 20%; left: 20%; animation-delay: 1s; }
  &:nth-child(5) { width: 120px; height: 120px; top: 15%; right: 5%; animation-delay: 3s; }
  &:nth-child(6) { width: 80px; height: 80px; bottom: 10%; right: 40%; animation-delay: 5s; }
}

/* === Header === */
.room-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: var(--lelamp-space-md) var(--lelamp-space-lg);
  background: rgba(255, 255, 255, 0.8);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid rgba(0, 0, 0, 0.05);
  position: relative;
  z-index: 10;
}

.header-left {
  display: flex;
  align-items: center;
  gap: var(--lelamp-space-lg);
}

.lelamp-logo {
  display: flex;
  align-items: center;
  gap: var(--lelamp-space-sm);
}

.logo-emoji {
  font-size: 1.75rem;
  animation: float 3s ease-in-out infinite;
}

.logo-text {
  font-family: var(--lelamp-font-display);
  font-size: 1.5rem;
  font-weight: 700;
  background: linear-gradient(135deg, var(--lelamp-peach), var(--lelamp-coral));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.connection-badge {
  display: flex;
  align-items: center;
  gap: var(--lelamp-space-sm);
  padding: var(--lelamp-space-xs) var(--lelamp-space-md);
  border-radius: var(--lelamp-radius-full);
  font-size: 0.813rem;
  font-weight: 600;

  &.connected {
    background: rgba(107, 203, 119, 0.15);
    color: var(--lelamp-mint-dark);
    .badge-dot { background: var(--lelamp-mint); }
  }

  &.connecting {
    background: rgba(255, 217, 61, 0.15);
    color: var(--lelamp-sunny-dark);
    .badge-dot { background: var(--lelamp-sunny); animation: pulse 1s ease-in-out infinite; }
  }

  &.disconnected, &.error {
    background: rgba(255, 107, 138, 0.15);
    color: var(--lelamp-coral-dark);
    .badge-dot { background: var(--lelamp-coral); }
  }
}

.badge-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

.header-actions {
  display: flex;
  gap: var(--lelamp-space-sm);
}

.icon-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 40px;
  height: 40px;
  background: var(--lelamp-bg-white);
  border: 1px solid rgba(0, 0, 0, 0.08);
  border-radius: var(--lelamp-radius-md);
  color: var(--lelamp-text-secondary);
  cursor: pointer;
  transition: all var(--lelamp-transition-normal);

  svg {
    width: 20px;
    height: 20px;
  }

  &:hover {
    background: var(--lelamp-bg-gray);
    color: var(--lelamp-text-primary);
    transform: translateY(-2px);
    box-shadow: var(--lelamp-shadow-sm);
  }

  &:active {
    transform: translateY(0);
  }

  &.danger:hover {
    background: var(--lelamp-coral-light);
    color: var(--lelamp-coral-dark);
    border-color: var(--lelamp-coral);
  }
}

/* === User Avatar Button === */
.user-avatar-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 40px;
  height: 40px;
  background: linear-gradient(135deg, var(--lelamp-peach) 0%, var(--lelamp-coral) 100%);
  border: 2px solid rgba(255, 255, 255, 0.8);
  border-radius: var(--lelamp-radius-md);
  cursor: pointer;
  transition: all var(--lelamp-transition-normal);
  box-shadow: 0 2px 8px rgba(255, 107, 138, 0.3);

  &:hover {
    transform: translateY(-2px) scale(1.05);
    box-shadow: 0 4px 12px rgba(255, 107, 138, 0.4);
  }

  &:active {
    transform: translateY(0) scale(1);
  }
}

.avatar-text {
  font-size: 1rem;
  font-weight: 700;
  color: var(--lelamp-bg-white);
}

/* === Main Content === */
.room-main {
  display: grid;
  grid-template-columns: 360px 1fr;
  gap: var(--lelamp-space-lg);
  padding: var(--lelamp-space-lg);
  flex: 1;
  position: relative;
  z-index: 1;
  max-width: 1400px;
  margin: 0 auto;
  width: 100%;
}

/* === Control Panel === */
.control-panel {
  display: flex;
  flex-direction: column;
  gap: var(--lelamp-space-md);
}

.panel-section {
  background: var(--lelamp-bg-white);
  border-radius: var(--lelamp-radius-xl);
  padding: var(--lelamp-space-lg);
  box-shadow: var(--lelamp-shadow-md);
}

.section-title {
  display: flex;
  align-items: center;
  gap: var(--lelamp-space-sm);
  font-family: var(--lelamp-font-display);
  font-size: 1.125rem;
  font-weight: 700;
  color: var(--lelamp-text-primary);
  margin-bottom: var(--lelamp-space-md);
}

.title-icon {
  font-size: 1.25rem;
}

.action-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: var(--lelamp-space-sm);
}

.action-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--lelamp-space-xs);
  padding: var(--lelamp-space-md);
  background: var(--lelamp-bg-gray);
  border: 2px solid transparent;
  border-radius: var(--lelamp-radius-md);
  cursor: pointer;
  transition: all var(--lelamp-transition-bounce);

  &:hover {
    background: var(--lelamp-bg-white);
    border-color: var(--lelamp-peach-light);
    transform: translateY(-4px);
    box-shadow: var(--lelamp-shadow-md);
  }

  &:active {
    transform: translateY(-2px);
  }
}

.action-emoji {
  font-size: 1.5rem;
}

.action-label {
  font-size: 0.813rem;
  font-weight: 600;
  color: var(--lelamp-text-secondary);
}

/* === Chat Panel === */
.right-panel {
  display: flex;
  flex-direction: column;
  gap: var(--lelamp-space-md);
  flex: 1;
  min-height: 0;
}

.chat-panel {
  background: var(--lelamp-bg-white);
  border-radius: var(--lelamp-radius-xl);
  box-shadow: var(--lelamp-shadow-md);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  flex: 1;
  min-height: 0;
}

.chat-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: var(--lelamp-space-md) var(--lelamp-space-lg);
  border-bottom: 1px solid rgba(0, 0, 0, 0.05);
}

.chat-title {
  display: flex;
  align-items: center;
  gap: var(--lelamp-space-sm);
  font-family: var(--lelamp-font-display);
  font-size: 1.125rem;
  font-weight: 700;
  color: var(--lelamp-text-primary);
}

.typing-indicator {
  display: flex;
  gap: 4px;
  padding: var(--lelamp-space-sm) var(--lelamp-space-md);
  background: var(--lelamp-bg-gray);
  border-radius: var(--lelamp-radius-full);

  span {
    width: 8px;
    height: 8px;
    background: var(--lelamp-peach);
    border-radius: 50%;
    animation: typing 1.4s ease-in-out infinite;

    &:nth-child(2) { animation-delay: 0.2s; }
    &:nth-child(3) { animation-delay: 0.4s; }
  }
}

@keyframes typing {
  0%, 60%, 100% { transform: translateY(0); }
  30% { transform: translateY(-6px); }
}

.messages-container {
  flex: 1;
  overflow-y: auto;
  padding: var(--lelamp-space-lg);
  background: var(--lelamp-bg-gray);
  min-height: 0;
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: var(--lelamp-text-tertiary);
  text-align: center;

  .empty-emoji {
    font-size: 4rem;
    margin-bottom: var(--lelamp-space-md);
    opacity: 0.5;
  }

  p {
    font-size: 1rem;
  }
}

.message {
  display: flex;
  gap: var(--lelamp-space-sm);
  margin-bottom: var(--lelamp-space-md);
  animation: slide-up 0.3s ease-out;

  &.user {
    flex-direction: row-reverse;
  }
}

.message-avatar {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 36px;
  font-size: 1.25rem;
  flex-shrink: 0;
}

.message-content {
  max-width: 70%;
}

.message-content p {
  padding: var(--lelamp-space-sm) var(--lelamp-space-md);
  border-radius: var(--lelamp-radius-lg);
  font-size: 0.938rem;
  line-height: 1.5;
  word-break: break-word;
}

.user .message-content p {
  background: linear-gradient(135deg, var(--lelamp-peach), var(--lelamp-coral));
  color: var(--lelamp-bg-white);
  border-bottom-right-radius: var(--lelamp-space-xs);
}

.agent .message-content p {
  background: var(--lelamp-bg-white);
  color: var(--lelamp-text-primary);
  box-shadow: var(--lelamp-shadow-sm);
  border-bottom-left-radius: var(--lelamp-space-xs);
}

.message-time {
  font-size: 0.688rem;
  color: var(--lelamp-text-tertiary);
  margin-top: 2px;
}

.chat-input {
  display: flex;
  gap: var(--lelamp-space-sm);
  padding: var(--lelamp-space-md) var(--lelamp-space-lg);
  background: var(--lelamp-bg-white);
  border-top: 1px solid rgba(0, 0, 0, 0.05);
}

.message-input {
  flex: 1;
  padding: var(--lelamp-space-sm) var(--lelamp-space-md);
  font-size: 0.938rem;
  background: var(--lelamp-bg-gray);
  border: 2px solid transparent;
  border-radius: var(--leamp-radius-full);
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

.send-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 44px;
  height: 44px;
  background: var(--lelamp-bg-gray);
  border: none;
  border-radius: 50%;
  cursor: pointer;
  transition: all var(--lelamp-transition-bounce);
  color: var(--lelamp-text-tertiary);

  svg {
    width: 20px;
    height: 20px;
  }

  &:hover {
    transform: scale(1.1);
  }

  &.has-content {
    background: linear-gradient(135deg, var(--lelamp-peach), var(--lelamp-coral));
    color: var(--lelamp-bg-white);

    &:hover {
      transform: scale(1.1);
      box-shadow: 0 4px 12px rgba(255, 107, 138, 0.4);
    }
  }
}

/* === Responsive === */
@media (max-width: 900px) {
  .room-main {
    grid-template-columns: 1fr;
  }

  .control-panel {
    order: 2;
  }

  .chat-panel {
    order: 1;
    height: 400px;
  }

  .action-grid {
    grid-template-columns: repeat(6, 1fr);
  }

  .action-card {
    padding: var(--lelamp-space-sm);
  }

  .action-emoji {
    font-size: 1.25rem;
  }

  .action-label {
    font-size: 0.688rem;
  }
}

@media (max-width: 480px) {
  .room-header {
    padding: var(--lelamp-space-sm) var(--lelamp-space-md);
  }

  .logo-text {
    display: none;
  }

  .room-main {
    padding: var(--lelamp-space-md);
  }

  .action-grid {
    grid-template-columns: repeat(3, 1fr);
  }
}
</style>
