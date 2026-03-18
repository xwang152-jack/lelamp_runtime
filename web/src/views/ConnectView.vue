<template>
  <div class="connect-view">
    <div class="connect-card">
      <div class="card-header">
        <div class="logo-icon">🪔</div>
        <h1>欢迎使用 LeLamp</h1>
        <p>智能台灯,陪伴成长</p>
      </div>

      <!-- 帮助提示 -->
      <el-alert
        type="info"
        show-icon
        :closable="false"
        class="help-alert"
      >
        <template #title>
          <strong>如何连接您的 LeLamp?</strong>
        </template>
        <div class="help-content">
          <div class="help-step">
            <span class="step-number">1</span>
            <span>确保 LeLamp 和您在同一 WiFi 下</span>
          </div>
          <div class="help-step">
            <span class="step-number">2</span>
            <span>API 地址已自动配置,直接点击连接即可</span>
          </div>
          <div class="help-step">
            <span class="step-number">3</span>
            <span>首次使用? 设备背面有 6 位连接码</span>
          </div>
        </div>
      </el-alert>

      <el-form :model="form" label-position="top" class="connect-form">
        <!-- 简化的连接选项 -->
        <el-tabs v-model="activeTab" class="connect-tabs">
          <!-- 快速连接 (推荐) -->
          <el-tab-pane label="快速连接" name="quick">
            <div class="tab-content">
              <p class="tab-description">
                <el-icon><InfoFilled /></el-icon>
                自动检测同一 WiFi 下的 LeLamp 设备
              </p>

              <el-form-item label="设备名称">
                <el-input
                  v-model="form.deviceName"
                  placeholder="LeLamp (自动检测)"
                  clearable
                  readonly
                  disabled
                >
                  <template #prefix>
                    <span class="input-icon">💡</span>
                  </template>
                </el-input>
              </el-form-item>

              <el-form-item label="API 地址">
                <el-input
                  v-model="form.serverUrl"
                  placeholder="已自动配置"
                  clearable
                >
                  <template #prefix>
                    <span class="input-icon">🌐</span>
                  </template>
                </el-input>
                <div class="hint success">
                  ✅ 已自动配置为: {{ form.serverUrl }}
                </div>
              </el-form-item>

              <el-form-item>
                <el-button
                  type="primary"
                  size="large"
                  :loading="loading"
                  style="width: 100%"
                  @click="handleConnect"
                >
                  <el-icon class="btn-icon"><Connection /></el-icon>
                  连接设备
                </el-button>
              </el-form-item>
            </div>
          </el-tab-pane>

          <!-- 设备码连接 -->
          <el-tab-pane label="设备码连接" name="code">
            <div class="tab-content">
              <p class="tab-description">
                <el-icon><Key /></el-icon>
                输入设备背面的 6 位连接码
              </p>

              <el-form-item label="设备码">
                <el-input
                  v-model="form.deviceCode"
                  placeholder="ABC123"
                  maxlength="6"
                  size="large"
                  clearable
                  style="font-family: 'Courier New', monospace; font-size: 18px; letter-spacing: 2px;"
                  @input="handleDeviceCodeInput"
                >
                  <template #prefix>
                    <span class="input-icon">🔢</span>
                  </template>
                </el-input>
                <div class="hint">
                  💡 设备码格式: 6 位大写字母 (例如: ABC123)
                </div>
              </el-form-item>

              <el-form-item>
                <el-button
                  type="primary"
                  size="large"
                  :loading="loading"
                  :disabled="!isValidDeviceCode"
                  style="width: 100%"
                  @click="handleConnectByCode"
                >
                  <el-icon class="btn-icon"><Connection /></el-icon>
                  通过设备码连接
                </el-button>
              </el-form-item>
            </div>
          </el-tab-pane>

          <!-- 高级选项 -->
          <el-tab-pane label="高级选项" name="advanced">
            <div class="tab-content">
              <p class="tab-description">
                <el-icon><Setting /></el-icon>
                手动配置 API 地址 (适用于特殊网络环境)
              </p>

              <el-form-item label="API Server URL">
                <el-input
                  v-model="form.serverUrl"
                  placeholder="http://lelamp.local:8000"
                  clearable
                >
                  <template #prefix>
                    <span class="input-icon">🌐</span>
                  </template>
                </el-input>
                <div class="hint">
                  💡 默认: http://localhost:8000<br>
                  如果 LeLamp 在同一网络,使用: http://192.168.0.104:8000
                </div>
              </el-form-item>

              <el-form-item>
                <el-button
                  type="primary"
                  size="large"
                  :loading="loading"
                  style="width: 100%"
                  @click="handleConnect"
                >
                  <el-icon class="btn-icon"><Connection /></el-icon>
                  连接设备
                </el-button>
              </el-form-item>
            </div>
          </el-tab-pane>
        </el-tabs>
      </el-form>

      <!-- 底部提示 -->
      <div class="footer-tips">
        <el-tag type="success" effect="plain" size="small">
          <el-icon><CircleCheck /></el-icon>
          自动检测已启用
        </el-tag>
        <el-tag type="info" effect="plain" size="small">
          <el-icon><Wifi /></el-icon>
          WiFi: {{ connectedWiFi || '检测中...' }}
        </el-tag>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import {
  Connection,
  InfoFilled,
  Key,
  Setting,
  CircleCheck,
  Wifi
} from '@element-plus/icons-vue'
import { useWebSocket } from '@/composables/useWebSocket'
import axios from 'axios'

const router = useRouter()
const { connect } = useWebSocket()

const loading = ref(false)
const activeTab = ref('quick')
const connectedWiFi = ref('')

// 表单数据
const form = reactive({
  serverUrl: '',
  deviceName: 'LeLamp',
  deviceCode: ''
})

// 设备码验证
const isValidDeviceCode = computed(() => {
  return /^[A-Z0-9]{6}$/.test(form.deviceCode)
})

// 自动配置 API URL
const autoConfigureUrl = () => {
  const protocol = window.location.protocol
  const hostname = window.location.hostname
  form.serverUrl = `${protocol}//${hostname}:8000`
  form.deviceName = 'LeLamp (自动检测)'
}

// 设备码输入处理
const handleDeviceCodeInput = (value: string) => {
  form.deviceCode = value.toUpperCase().replace(/[^A-Z0-9]/g, '')
}

// 连接状态
const connectionStatus = ref<'checking' | 'connected' | 'disconnected'>('checking')

// 检查网络连接状态
const checkNetworkStatus = async () => {
  try {
    const response = await fetch(`${form.serverUrl}/api/system/setup/status`)
    if (response.ok) {
      const data = await response.json()
      if (data.configured_wifi) {
        connectedWiFi.value = data.configured_wifi
        connectionStatus.value = 'connected'
      }
    }
  } catch (error) {
    connectionStatus.value = 'disconnected'
  }
}

// 通过设备码连接
async function handleConnectByCode() {
  if (!isValidDeviceCode.value) {
    ElMessage.warning('请输入正确的 6 位设备码')
    return
  }

  loading.value = true
  try {
    // TODO: 实现设备码到 IP 的映射
    // const apiUrl = await resolveDeviceCode(form.deviceCode)

    // 临时方案: 使用设备码作为提示
    ElMessage.info(`设备码功能开发中,当前使用自动配置连接`)

    // 仍然使用自动配置的 URL 连接
    await connect(form.serverUrl, 'lelamp')
    ElMessage.success('连接成功!')
    router.push('/room')
  } catch (error) {
    ElMessage.error('连接失败,请检查设备是否在线')
    console.error(error)
  } finally {
    loading.value = false
  }
}

// 正常连接
async function handleConnect() {
  const resolvedUrl = form.serverUrl

  if (!resolvedUrl) {
    ElMessage.warning('请配置 API Server URL')
    return
  }

  loading.value = true
  try {
    await connect(resolvedUrl, 'lelamp')
    ElMessage.success('连接成功!')
    router.push('/room')
  } catch (error) {
    ElMessage.error('连接失败,请检查:')
    console.error(error)

    // 显示详细错误信息
    if (error instanceof Error) {
      ElMessage({
        message: `错误详情: ${error.message}`,
        type: 'error',
        duration: 5000
      })
    }
  } finally {
    loading.value = false
  }
}

// 初始化
onMounted(() => {
  autoConfigureUrl()
  checkNetworkStatus()
})
</script>

<style lang="scss" scoped>
.connect-view {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 20px;
}

.connect-card {
  width: 100%;
  max-width: 600px;
  padding: 40px;
  background: white;
  border-radius: 16px;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
}

.card-header {
  text-align: center;
  margin-bottom: 32px;

  .logo-icon {
    font-size: 64px;
    margin-bottom: 16px;
    animation: float 3s ease-in-out infinite;
  }

  h1 {
    font-size: 28px;
    margin-bottom: 8px;
    color: #333;
  }

  p {
    color: #666;
    font-size: 16px;
  }
}

@keyframes float {
  0%, 100% {
    transform: translateY(0);
  }
  50% {
    transform: translateY(-10px);
  }
}

.help-alert {
  margin-bottom: 24px;

  :deep(.el-alert__content) {
    font-size: 14px;
  }
}

.help-content {
  margin-top: 12px;
}

.help-step {
  display: flex;
  align-items: center;
  margin-bottom: 8px;
  font-size: 14px;
  line-height: 1.6;

  &:last-child {
    margin-bottom: 0;
  }
}

.step-number {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  background: #667eea;
  color: white;
  border-radius: 50%;
  font-size: 12px;
  font-weight: bold;
  margin-right: 12px;
  flex-shrink: 0;
}

.connect-form {
  margin-top: 24px;
}

.connect-tabs {
  :deep(.el-tabs__header) {
    margin-bottom: 24px;
  }
}

.tab-content {
  padding: 16px 0;
}

.tab-description {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px;
  background: #f5f7fa;
  border-radius: 8px;
  margin-bottom: 20px;
  color: #606266;
  font-size: 14px;

  .el-icon {
    color: #409eff;
  }
}

.hint {
  margin-top: 8px;
  font-size: 13px;
  color: #666;
  line-height: 1.6;

  code {
    padding: 2px 6px;
    background: #f5f5f5;
    border-radius: 4px;
    font-family: 'Courier New', monospace;
    font-size: 12px;
  }

  &.success {
    color: #67c23a;
    background: #f0f9ff;
    padding: 12px;
    border-radius: 8px;
    border-left: 3px solid #67c23a;
    font-weight: 500;
  }

  &.warning {
    color: #e6a23c;
    background: #fef9f0;
    padding: 12px;
    border-radius: 8px;
    border-left: 3px solid #e6a23c;
  }
}

.input-icon {
  font-size: 18px;
}

.btn-icon {
  margin-right: 8px;
}

.footer-tips {
  display: flex;
  justify-content: center;
  gap: 12px;
  margin-top: 24px;
  padding-top: 24px;
  border-top: 1px solid #eee;

  .el-tag {
    display: flex;
    align-items: center;
    gap: 4px;

    .el-icon {
      font-size: 14px;
    }
  }
}

// 响应式设计
@media (max-width: 768px) {
  .connect-card {
    padding: 24px;
  }

  .card-header {
    h1 {
      font-size: 24px;
    }

    .logo-icon {
      font-size: 48px;
    }
  }

  .footer-tips {
    flex-direction: column;
    gap: 8px;
  }

  .help-step {
    font-size: 13px;
  }
}
</style>
