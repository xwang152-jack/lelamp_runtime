<template>
  <div class="wifi-settings">
    <!-- 当前连接状态 -->
    <el-card header="当前连接" class="status-card">
      <el-descriptions :column="2" border>
        <el-descriptions-item label="状态">
          <el-tag :type="wifiStatus?.connected ? 'success' : 'danger'">
            {{ wifiStatus?.connected ? '已连接' : '未连接' }}
          </el-tag>
        </el-descriptions-item>
        <el-descriptions-item label="网络名称">
          {{ wifiStatus?.ssid || '-' }}
        </el-descriptions-item>
        <el-descriptions-item label="信号强度">
          <template v-if="wifiStatus?.signal_strength !== null">
            <el-progress
              :percentage="wifiStatus.signal_strength"
              :color="getSignalColor(wifiStatus.signal_strength)"
              :stroke-width="12"
            />
          </template>
          <template v-else>-</template>
        </el-descriptions-item>
        <el-descriptions-item label="IP 地址">
          {{ wifiStatus?.ip_address || '-' }}
        </el-descriptions-item>
      </el-descriptions>

      <template #extra>
        <el-button
          v-if="wifiStatus?.connected"
          type="danger"
          size="small"
          @click="handleDisconnect"
        >
          断开连接
        </el-button>
      </template>
    </el-card>

    <!-- 可用网络 -->
    <el-card header="可用网络" class="networks-card">
      <div class="scan-header">
        <el-alert
          type="info"
          :closable="false"
          show-icon
        >
          <template #default>
            <span>点击"扫描网络"查看附近的 WiFi 热点</span>
          </template>
        </el-alert>
      </div>

      <el-button
        type="primary"
        :icon="Refresh"
        :loading="wifiScanning"
        @click="handleScan"
      >
        扫描网络
      </el-button>

      <div v-if="networks.length > 0" class="networks-list">
        <div
          v-for="network in networks"
          :key="network.bssid"
          class="network-item"
          :class="{ connected: network.ssid === wifiStatus?.ssid }"
        >
          <div class="network-info">
            <div class="network-name">
              {{ network.ssid }}
              <el-tag v-if="network.is_hidden" size="small" type="info">隐藏</el-tag>
            </div>
            <div class="network-meta">
              <span>{{ network.security }}</span>
              <span>{{ network.frequency }}</span>
            </div>
          </div>
          <div class="network-signal">
            <el-progress
              :percentage="network.signal_strength"
              :color="getSignalColor(network.signal_strength)"
              :show-text="false"
              :stroke-width="8"
            />
            <span class="signal-text">{{ network.signal_strength }}%</span>
          </div>
          <div class="network-action">
            <el-button
              size="small"
              :disabled="network.ssid === wifiStatus?.ssid"
              @click="showConnectDialog(network)"
            >
              {{ network.ssid === wifiStatus?.ssid ? '已连接' : '连接' }}
            </el-button>
          </div>
        </div>
      </div>
      <el-empty v-else-if="!wifiScanning" description="暂无网络，请点击扫描" :image-size="100" />
    </el-card>

    <!-- 连接对话框 -->
    <el-dialog
      v-model="connectDialogVisible"
      :title="`连接到 ${selectedNetwork?.ssid}`"
      width="400px"
    >
      <el-form label-width="80px">
        <el-form-item label="网络名称">
          <el-input :value="selectedNetwork?.ssid" disabled />
        </el-form-item>
        <el-form-item
          v-if="selectedNetwork?.security !== 'open' && selectedNetwork?.security !== ''"
          label="密码"
        >
          <el-input
            v-model="wifiPassword"
            type="password"
            placeholder="请输入 WiFi 密码"
            show-password
            @keyup.enter="handleConnect"
          />
        </el-form-item>
        <el-form-item v-else>
          <el-alert type="success" :closable="false">
            这是一个开放网络，无需密码
          </el-alert>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="connectDialogVisible = false">取消</el-button>
        <el-button type="primary" @click="handleConnect">连接</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { Refresh } from '@element-plus/icons-vue'
import { useSettingsStore } from '@/stores'
import type { WiFiNetwork } from '@/types/settings'

const settingsStore = useSettingsStore()

const connectDialogVisible = ref(false)
const selectedNetwork = ref<WiFiNetwork | null>(null)
const wifiPassword = ref('')

const wifiStatus = computed(() => settingsStore.wifiStatus)
const wifiScanning = computed(() => settingsStore.wifiScanning)
const networks = computed(() => settingsStore.wifiNetworks)

onMounted(async () => {
  await refreshStatus()
})

async function refreshStatus() {
  try {
    await settingsStore.fetchWiFiStatus()
  } catch (e) {
    console.error('获取 WiFi 状态失败:', e)
  }
}

async function handleScan() {
  try {
    await settingsStore.scanWiFi()
    ElMessage.success('扫描完成')
  } catch (e) {
    ElMessage.error('WiFi 扫描失败，请确保设备支持 WiFi')
  }
}

function showConnectDialog(network: WiFiNetwork) {
  selectedNetwork.value = network
  wifiPassword.value = ''
  connectDialogVisible.value = true
}

async function handleConnect() {
  if (!selectedNetwork.value) return

  try {
    const result = await settingsStore.connectToWiFi(
      selectedNetwork.value.ssid,
      wifiPassword.value || undefined
    )

    if (result.success) {
      ElMessage.success(`已连接到 ${selectedNetwork.value.ssid}`)
      connectDialogVisible.value = false
      await refreshStatus()
    } else {
      ElMessage.error(result.message || '连接失败')
    }
  } catch (e) {
    ElMessage.error(e instanceof Error ? e.message : '连接失败')
  }
}

async function handleDisconnect() {
  try {
    const result = await settingsStore.disconnectFromWiFi()
    if (result.success) {
      ElMessage.success('已断开连接')
      await refreshStatus()
    } else {
      ElMessage.error('断开连接失败')
    }
  } catch (e) {
    ElMessage.error('断开连接失败')
  }
}

function getSignalColor(strength: number): string {
  if (strength >= 60) return '#67c23a'
  if (strength >= 30) return '#e6a23c'
  return '#f56c6c'
}
</script>

<style lang="scss" scoped>
.wifi-settings {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.status-card {
  :deep(.el-card__header) {
    padding: 12px 16px;
  }
}

.networks-card {
  :deep(.el-card__header) {
    padding: 12px 16px;
  }

  .scan-header {
    margin-bottom: 16px;
  }
}

.networks-list {
  margin-top: 16px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.network-item {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 12px;
  border: 1px solid #eee;
  border-radius: 8px;
  background: #fafafa;
  transition: all 0.2s;

  &:hover {
    background: #f0f0f0;
  }

  &.connected {
    border-color: var(--el-color-success);
    background: #f0f9ff;
  }

  .network-info {
    flex: 1;
    min-width: 0;

    .network-name {
      font-weight: 500;
      margin-bottom: 4px;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .network-meta {
      display: flex;
      gap: 12px;
      font-size: 12px;
      color: #999;
    }
  }

  .network-signal {
    display: flex;
    align-items: center;
    gap: 8px;
    width: 100px;

    .signal-text {
      font-size: 12px;
      color: #666;
      min-width: 35px;
    }
  }

  .network-action {
    flex-shrink: 0;
  }
}
</style>
