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
            连接设备
          </el-button>
        </el-form-item>
      </el-form>

      <!-- 底部提示 -->
      <div class="footer-tips">
        <el-tag type="success" effect="plain" size="small">
          ✅ 自动检测已启用
        </el-tag>
        <el-tag type="info" effect="plain" size="small">
          📶 WiFi: {{ connectedWiFi || '检测中...' }}
        </el-tag>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useWebSocket } from '@/composables/useWebSocket'

const router = useRouter()
const { connect } = useWebSocket()

const loading = ref(false)
const connectedWiFi = ref('')

// 表单数据
const form = reactive({
  serverUrl: '',
  deviceName: 'LeLamp'
})

// 自动配置 API URL
const autoConfigureUrl = () => {
  const protocol = window.location.protocol
  const hostname = window.location.hostname
  form.serverUrl = `${protocol}//${hostname}:8000`
}

// 检查网络连接状态
const checkNetworkStatus = async () => {
  try {
    const response = await fetch(`${form.serverUrl}/api/system/setup/status`)
    if (response.ok) {
      const data = await response.json()
      if (data.configured_wifi) {
        connectedWiFi.value = data.configured_wifi
      }
    }
  } catch (error) {
    console.warn('Failed to check network status:', error)
  }
}

// 连接设备
async function handleConnect() {
  if (!form.serverUrl) {
    ElMessage.warning('请配置 API 地址')
    return
  }

  loading.value = true
  try {
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

.hint {
  margin-top: 8px;
  font-size: 13px;
  color: #666;
  line-height: 1.6;

  &.success {
    color: #67c23a;
    background: #f0f9ff;
    padding: 12px;
    border-radius: 8px;
    border-left: 3px solid #67c23a;
    font-weight: 500;
  }
}

.input-icon {
  font-size: 18px;
}

.footer-tips {
  display: flex;
  justify-content: center;
  gap: 12px;
  margin-top: 24px;
  padding-top: 24px;
  border-top: 1px solid #eee;
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
