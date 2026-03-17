<template>
  <div class="room-view">
    <div class="room-header">
      <div class="status">
        <span class="status-dot online"></span>
        <span>已连接</span>
      </div>
      <el-button @click="handleDisconnect" type="danger">断开连接</el-button>
    </div>

    <div class="room-content">
      <div class="video-section">
        <div class="video-placeholder">📹</div>
        <p>等待摄像头画面...</p>
      </div>

      <div class="control-section">
        <div class="panel">
          <h3>⚡ 快捷操作</h3>
          <div class="button-grid">
            <el-button @click="sendChat('你好')">👋 打招呼</el-button>
            <el-button @click="sendChat('现在几点了')">⏰ 查看时间</el-button>
            <el-button @click="sendChat('讲个笑话')">😄 讲笑话</el-button>
            <el-button @click="sendChat('唱首歌')">🎵 唱歌</el-button>
          </div>
        </div>

        <div class="panel">
          <h3>💡 灯光魔法</h3>
          <div class="button-grid">
            <el-button @click="setEffect('breathing')">💗 呼吸灯</el-button>
            <el-button @click="setEffect('rainbow')">🌈 彩虹</el-button>
            <el-button @click="setEffect('wave')">🌊 波浪</el-button>
          </div>
        </div>
      </div>
    </div>

    <div class="chat-section">
      <h3>💬 实时对话</h3>
      <div class="messages"></div>
      <div class="input-area">
        <el-input v-model="inputText" @keyup.enter="sendMessage" placeholder="输入消息..." />
        <el-button type="primary" @click="sendMessage">发送</el-button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useLiveKit } from '@/composables/useLiveKit'
import { useChatStore } from '@/stores'

const router = useRouter()
const { disconnect, sendCommand } = useLiveKit()
const chatStore = useChatStore()

const inputText = ref('')

async function handleDisconnect() {
  await disconnect()
  router.push('/connect')
}

function sendChat(text: string) {
  chatStore.addMessage('user', text)
  ElMessage.success(`发送: ${text}`)
}

function setEffect(effect: string) {
  sendCommand(`rgb_effect_${effect}`, {})
  ElMessage.success(`设置特效: ${effect}`)
}

function sendMessage() {
  if (!inputText.value.trim()) return
  sendChat(inputText.value)
  inputText.value = ''
}
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
  }

  .input-area {
    display: flex;
    gap: 8px;
  }
}
</style>
