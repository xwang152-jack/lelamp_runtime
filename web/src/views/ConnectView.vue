<template>
  <div class="connect-view">
    <div class="connect-card">
      <div class="card-header">
        <h1>🪔 LeLamp Web Client</h1>
        <p>智能台灯，陪伴成长</p>
      </div>

      <el-form :model="form" label-position="top">
        <el-form-item label="API Server URL">
          <el-input
            v-model="form.serverUrl"
            placeholder="http://localhost:8000"
            clearable
          >
            <template #prefix>
              <span class="input-icon">🌐</span>
            </template>
          </el-input>
          <div class="hint">
            💡 请输入 LeLamp 后端 API 地址（默认 http://localhost:8000）
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
import { useWebSocket } from '@/composables/useWebSocket'

const router = useRouter()
const { connect } = useWebSocket()

const loading = ref(false)

const form = reactive({
  serverUrl: 'http://localhost:8000'
})

async function handleConnect() {
  const resolvedUrl = form.serverUrl

  if (!resolvedUrl) {
    ElMessage.warning('请配置 API Server URL')
    return
  }

  loading.value = true
  try {
    await connect(resolvedUrl, 'lelamp')
    ElMessage.success('连接成功')
    router.push('/room') // Note: We might want to rename this view later, but keeping it for now
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

  &.success {
    color: #67c23a;
    background: #f0f9ff;
    padding: 8px;
    border-radius: 4px;
    border-left: 3px solid #67c23a;
  }

  &.warning {
    color: #e6a23c;
    background: #fef9f0;
    padding: 8px;
    border-radius: 4px;
    border-left: 3px solid #e6a23c;
  }
}

.input-icon {
  font-size: 16px;
}
</style>
