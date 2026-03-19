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

    <el-alert
      type="info"
      :closable="false"
      show-icon
      class="info-alert"
    >
      <template #default>
        <p>视觉识别功能需要摄像头支持。配置修改后需要重启服务才能生效。</p>
      </template>
    </el-alert>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { useSettingsStore } from '@/stores'

const settingsStore = useSettingsStore()
const saving = ref(false)

const form = ref({
  vision_enabled: true,
  modelscope_model: '',
  modelscope_api_key: '',
  modelscope_timeout_s: 60,
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
</style>
