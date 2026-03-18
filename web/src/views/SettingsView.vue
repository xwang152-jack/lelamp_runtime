<template>
  <div class="settings-view">
    <div class="settings-header">
      <div class="header-left">
        <h2>系统设置</h2>
        <p v-if="lampId" class="lamp-id">设备: {{ lampId }}</p>
      </div>
      <el-button @click="handleBack">返回</el-button>
    </div>

    <!-- 重启提示 -->
    <el-alert
      v-if="settingsStore.hasPendingChanges"
      type="warning"
      :closable="false"
      show-icon
      class="restart-alert"
    >
      <template #title>
        <span>配置修改后需要重启服务才能生效</span>
        <el-button type="primary" size="small" @click="showRestartDialog">
          立即重启
        </el-button>
      </template>
    </el-alert>

    <div class="settings-content">
      <!-- 左侧菜单 -->
      <el-aside width="200px" class="settings-sidebar">
        <el-menu :default-active="activeTab" @select="handleTabSelect">
          <el-menu-item index="wifi">
            <el-icon><Connection /></el-icon>
            <span>WiFi 网络</span>
          </el-menu-item>
          <el-menu-item index="llm">
            <el-icon><ChatDotRound /></el-icon>
            <span>LLM 模型</span>
          </el-menu-item>
          <el-menu-item index="vision">
            <el-icon><View /></el-icon>
            <span>视觉识别</span>
          </el-menu-item>
          <el-menu-item index="camera">
            <el-icon><VideoCamera /></el-icon>
            <span>摄像头</span>
          </el-menu-item>
          <el-menu-item index="speech">
            <el-icon><Microphone /></el-icon>
            <span>语音配置</span>
          </el-menu-item>
          <el-menu-item index="hardware">
            <el-icon><Setting /></el-icon>
            <span>硬件配置</span>
          </el-menu-item>
          <el-menu-item index="behavior">
            <el-icon><Operation /></el-icon>
            <span>行为配置</span>
          </el-menu-item>
          <el-menu-item index="ui">
            <el-icon><Brush /></el-icon>
            <span>界面设置</span>
          </el-menu-item>
        </el-menu>
      </el-aside>

      <!-- 右侧内容区 -->
      <el-main class="settings-main">
        <WiFiSettings v-if="activeTab === 'wifi'" />
        <LLMConfig v-else-if="activeTab === 'llm'" />
        <VisionConfig v-else-if="activeTab === 'vision'" />
        <CameraConfig v-else-if="activeTab === 'camera'" />
        <SpeechConfig v-else-if="activeTab === 'speech'" />
        <HardwareConfig v-else-if="activeTab === 'hardware'" />
        <BehaviorConfig v-else-if="activeTab === 'behavior'" />
        <UIConfig v-else-if="activeTab === 'ui'" />
      </el-main>
    </div>

    <!-- 重启确认对话框 -->
    <el-dialog v-model="restartDialogVisible" title="重启服务" width="400px">
      <p>重启服务将中断当前连接，是否继续？</p>
      <p v-if="restartCountdown > 0" class="countdown">
        服务将在 {{ restartCountdown }} 秒后重启...
      </p>
      <template #footer>
        <el-button @click="cancelRestart">取消</el-button>
        <el-button type="danger" :disabled="restarting" @click="confirmRestart">
          确认重启
        </el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import {
  Connection,
  ChatDotRound,
  View,
  VideoCamera,
  Microphone,
  Setting,
  Operation,
  Brush,
} from '@element-plus/icons-vue'
import { useSettingsStore } from '@/stores'
import WiFiSettings from '@/components/settings/WiFiSettings.vue'
import LLMConfig from '@/components/settings/LLMConfig.vue'
import VisionConfig from '@/components/settings/VisionConfig.vue'
import CameraConfig from '@/components/settings/CameraConfig.vue'
import SpeechConfig from '@/components/settings/SpeechConfig.vue'
import HardwareConfig from '@/components/settings/HardwareConfig.vue'
import BehaviorConfig from '@/components/settings/BehaviorConfig.vue'
import UIConfig from '@/components/settings/UIConfig.vue'
import { triggerRestart } from '@/api/settings'

const router = useRouter()
const route = useRoute()
const settingsStore = useSettingsStore()

const activeTab = ref('wifi')
const lampId = ref<string>('')
const restartDialogVisible = ref(false)
const restartCountdown = ref(0)
const restarting = ref(false)

onMounted(async () => {
  // 从路由查询参数获取 lamp_id
  lampId.value = (route.query.lamp_id as string) || 'lelamp'
  settingsStore.setLampId(lampId.value)

  // 获取初始标签
  activeTab.value = (route.query.tab as string) || 'wifi'

  try {
    await settingsStore.fetchSettings()
    await settingsStore.fetchWiFiStatus()
  } catch (e) {
    ElMessage.error('加载设置失败')
  }
})

function handleTabSelect(index: string) {
  activeTab.value = index
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

    // 延迟返回到连接页面
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
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: var(--lelamp-bg);
}

.settings-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 24px;
  background: white;
  box-shadow: var(--shadow-sm);

  .header-left {
    h2 {
      margin: 0;
      font-size: 20px;
      color: #333;
    }

    .lamp-id {
      margin: 4px 0 0;
      font-size: 12px;
      color: #999;
    }
  }
}

.restart-alert {
  margin: 16px 24px;

  :deep(.el-alert__title) {
    display: flex;
    align-items: center;
    gap: 12px;
  }
}

.settings-content {
  display: flex;
  flex: 1;
  overflow: hidden;
}

.settings-sidebar {
  background: white;
  border-right: 1px solid #eee;
  overflow-y: auto;

  .el-menu {
    border-right: none;
  }
}

.settings-main {
  padding: 24px;
  overflow-y: auto;
}

.countdown {
  color: #e6a23c;
  font-size: 14px;
  margin-top: 12px;
}
</style>
