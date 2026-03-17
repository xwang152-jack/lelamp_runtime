<template>
  <div class="connect-view">
    <div class="connect-card">
      <div class="card-header">
        <h1>🪔 LeLamp Web Client</h1>
        <p>智能台灯，陪伴成长</p>
      </div>

      <el-form
        :model="form"
        label-position="top"
      >
        <el-form-item label="LiveKit Server URL">
          <el-input
            v-model="form.serverUrl"
            placeholder="wss://your-project.livekit.cloud"
          />
        </el-form-item>

        <el-form-item label="Access Token">
          <el-input
            v-model="form.token"
            type="textarea"
            :rows="4"
            placeholder="粘贴生成的 Token..."
          />
          <div class="hint">
            运行 <code>python3 scripts/generate_client_token.py</code> 获取 Token
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
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useLiveKit } from '@/composables/useLiveKit'

const router = useRouter()
const { connect } = useLiveKit()

const loading = ref(false)
const form = reactive({
  serverUrl: import.meta.env.VITE_LIVEKIT_URL || '',
  token: ''
})

async function handleConnect() {
  if (!form.serverUrl || !form.token) {
    ElMessage.warning('请填写完整信息')
    return
  }

  loading.value = true
  try {
    await connect(form.serverUrl, form.token)
    ElMessage.success('连接成功')
    router.push('/room')
  } catch (error) {
    ElMessage.error('连接失败，请检查配置')
    console.error(error)
  } finally {
    loading.value = false
  }
}
</script>

<style lang="scss" scoped>
.connect-view {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.connect-card {
  width: 90%;
  max-width: 500px;
  padding: 40px;
  background: white;
  border-radius: 16px;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
}

.card-header {
  text-align: center;
  margin-bottom: 32px;

  h1 {
    font-size: 28px;
    margin-bottom: 8px;
  }

  p {
    color: #666;
    font-size: 14px;
  }
}

.hint {
  margin-top: 8px;
  font-size: 12px;
  color: #999;

  code {
    padding: 2px 6px;
    background: #f5f5f5;
    border-radius: 4px;
    font-family: 'Courier New', monospace;
  }
}
</style>
