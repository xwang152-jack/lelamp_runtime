<template>
  <div class="vision-config">
    <el-card header="视觉识别配置">
      <el-form
        ref="formRef"
        :model="form"
        label-width="140px"
        label-position="left"
      >
        <el-form-item label="启用视觉服务">
          <el-switch v-model="form.vision_enabled" />
          <span class="form-hint">关闭后将禁用摄像头识别功能</span>
        </el-form-item>

        <el-form-item label="模型名称">
          <el-input
            v-model="form.modelscope_model"
            placeholder="Qwen/Qwen3-VL-235B-A22B-Instruct"
          />
          <span class="form-hint">ModelScope 视觉模型名称</span>
        </el-form-item>

        <el-form-item label="API Key">
          <el-input
            v-model="form.modelscope_api_key"
            type="password"
            placeholder="输入新的 API Key"
            show-password
          />
          <span class="form-hint">
            当前状态:
            <el-tag :type="apiKeyConfigured ? 'success' : 'warning'" size="small">
              {{ apiKeyConfigured ? '已配置' : '未配置' }}
            </el-tag>
          </span>
        </el-form-item>

        <el-form-item label="超时时间">
          <el-input-number
            v-model="form.modelscope_timeout_s"
            :min="1"
            :max="300"
            :step="1"
          />
          <span class="form-hint">API 请求超时时间（秒）</span>
        </el-form-item>

        <el-form-item>
          <el-button type="primary" @click="handleSave" :loading="saving">
            保存配置
          </el-button>
          <el-button @click="handleReset">重置</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <!-- 边缘视觉配置 -->
    <el-card header="边缘视觉 (Edge Vision)" class="edge-vision-card">
      <template #header>
        <div class="card-header">
          <span>边缘视觉 (Edge Vision)</span>
          <el-tag type="success" size="small">低延迟</el-tag>
        </div>
      </template>
      
      <el-form
        :model="edgeForm"
        label-width="140px"
        label-position="left"
      >
        <el-form-item label="启用边缘视觉">
          <el-switch v-model="edgeForm.edge_vision_enabled" />
          <span class="form-hint">启用本地 AI 推理，降低延迟并保护隐私</span>
        </el-form-item>

        <el-form-item label="优先本地推理">
          <el-switch 
            v-model="edgeForm.edge_vision_prefer_local" 
            :disabled="!edgeForm.edge_vision_enabled"
          />
          <span class="form-hint">简单问题优先使用本地模型，降低云端 API 调用</span>
        </el-form-item>

        <el-form-item label="本地置信度阈值">
          <el-slider
            v-model="edgeForm.edge_vision_local_threshold"
            :min="0.1"
            :max="1.0"
            :step="0.1"
            :disabled="!edgeForm.edge_vision_enabled"
            show-input
            :show-input-controls="false"
          />
          <span class="form-hint">高于此阈值时直接返回本地结果，低于则调用云端</span>
        </el-form-item>

        <el-form-item>
          <el-button 
            type="primary" 
            @click="handleEdgeSave" 
            :loading="edgeSaving"
            :disabled="!edgeForm.edge_vision_enabled"
          >
            保存边缘视觉配置
          </el-button>
        </el-form-item>
      </el-form>

      <el-divider />

      <div class="edge-features">
        <h4>功能特性</h4>
        <el-row :gutter="16">
          <el-col :span="8">
            <div class="feature-item">
              <el-icon :size="24" color="#67c23a"><User /></el-icon>
              <div class="feature-info">
                <span class="feature-title">人脸检测</span>
                <span class="feature-desc">用户在场检测 < 50ms</span>
              </div>
            </div>
          </el-col>
          <el-col :span="8">
            <div class="feature-item">
              <el-icon :size="24" color="#409eff"><Pointer /></el-icon>
              <div class="feature-info">
                <span class="feature-title">手势追踪</span>
                <span class="feature-desc">8种手势识别 < 100ms</span>
              </div>
            </div>
          </el-col>
          <el-col :span="8">
            <div class="feature-item">
              <el-icon :size="24" color="#e6a23c"><Box /></el-icon>
              <div class="feature-info">
                <span class="feature-title">物体检测</span>
                <span class="feature-desc">80类COCO物体 < 300ms</span>
              </div>
            </div>
          </el-col>
        </el-row>
      </div>
    </el-card>

    <el-alert
      type="info"
      :closable="false"
      show-icon
      class="info-alert"
    >
      <template #default>
        <p>视觉识别功能需要摄像头支持。配置修改后需要重启服务才能生效。</p>
        <p>边缘视觉需要安装 MediaPipe：pip install mediapipe opencv-python</p>
      </template>
    </el-alert>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { User, Pointer, Box } from '@element-plus/icons-vue'
import { useSettingsStore } from '@/stores'

const settingsStore = useSettingsStore()
const saving = ref(false)
const edgeSaving = ref(false)

const form = ref({
  vision_enabled: true,
  modelscope_model: '',
  modelscope_api_key: '',
  modelscope_timeout_s: 60,
})

const edgeForm = ref({
  edge_vision_enabled: false,
  edge_vision_prefer_local: true,
  edge_vision_local_threshold: 0.7,
})

const apiKeyConfigured = computed(() => settingsStore.settings?.modelscope_api_key_configured || false)

onMounted(async () => {
  await loadSettings()
})

async function loadSettings() {
  const settings = settingsStore.settings
  if (settings) {
    form.value = {
      vision_enabled: settings.vision_enabled ?? true,
      modelscope_model: settings.modelscope_model || '',
      modelscope_api_key: '',
      modelscope_timeout_s: settings.modelscope_timeout_s || 60,
    }
    edgeForm.value = {
      edge_vision_enabled: settings.edge_vision_enabled ?? false,
      edge_vision_prefer_local: settings.edge_vision_prefer_local ?? true,
      edge_vision_local_threshold: settings.edge_vision_local_threshold ?? 0.7,
    }
  }
}

async function handleSave() {
  saving.value = true

  try {
    const updates: any = {
      vision_enabled: form.value.vision_enabled,
      modelscope_model: form.value.modelscope_model || undefined,
      modelscope_timeout_s: form.value.modelscope_timeout_s,
    }

    if (form.value.modelscope_api_key) {
      updates.modelscope_api_key = form.value.modelscope_api_key
    }

    await settingsStore.saveSettings(updates)
    ElMessage.success('配置已保存，重启后生效')
  } catch (e) {
    ElMessage.error('保存失败')
  } finally {
    saving.value = false
  }
}

async function handleEdgeSave() {
  edgeSaving.value = true

  try {
    const updates: any = {
      edge_vision_enabled: edgeForm.value.edge_vision_enabled,
      edge_vision_prefer_local: edgeForm.value.edge_vision_prefer_local,
      edge_vision_local_threshold: edgeForm.value.edge_vision_local_threshold,
    }

    await settingsStore.saveSettings(updates)
    ElMessage.success('边缘视觉配置已保存，重启后生效')
  } catch (e) {
    ElMessage.error('保存失败')
  } finally {
    edgeSaving.value = false
  }
}

function handleReset() {
  loadSettings()
}
</script>

<style lang="scss" scoped>
.vision-config {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.form-hint {
  display: block;
  margin-top: 4px;
  font-size: 12px;
  color: var(--lelamp-text-tertiary);
}

.info-alert p {
  margin: 4px 0;
}

.edge-vision-card {
  .card-header {
    display: flex;
    align-items: center;
    gap: 8px;
  }
}

.edge-features {
  h4 {
    margin: 0 0 16px 0;
    font-size: 14px;
    font-weight: 500;
    color: var(--el-text-color-primary);
  }

  .feature-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px;
    background: var(--el-fill-color-light);
    border-radius: 8px;

    .feature-info {
      display: flex;
      flex-direction: column;
      gap: 4px;

      .feature-title {
        font-size: 14px;
        font-weight: 500;
        color: var(--el-text-color-primary);
      }

      .feature-desc {
        font-size: 12px;
        color: var(--el-text-color-secondary);
      }
    }
  }
}
</style>
