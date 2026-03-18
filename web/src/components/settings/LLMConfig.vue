<template>
  <div class="llm-config">
    <el-card header="LLM 模型配置">
      <el-form
        ref="formRef"
        :model="form"
        label-width="140px"
        label-position="left"
      >
        <el-form-item label="模型名称">
          <el-input
            v-model="form.deepseek_model"
            placeholder="deepseek-chat"
          />
          <span class="form-hint">DeepSeek API 使用的模型名称</span>
        </el-form-item>

        <el-form-item label="API 地址">
          <el-input
            v-model="form.deepseek_base_url"
            placeholder="https://api.deepseek.com"
          />
          <span class="form-hint">DeepSeek API 基础 URL</span>
        </el-form-item>

        <el-form-item label="API Key">
          <el-input
            v-model="form.deepseek_api_key"
            type="password"
            placeholder="输入新的 API Key"
            show-password
          />
          <span class="form-hint">
            当前状态:
            <el-tag :type="apiKeyConfigured ? 'success' : 'warning'" size="small">
              {{ apiKeyConfigured ? '已配置' : '未配置' }}
            </el-tag>
            <span v-if="maskedKey" class="masked-key">{{ maskedKey }}</span>
          </span>
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
        <p>LLM 配置修改后需要重启服务才能生效。</p>
        <p>获取 API Key: <a href="https://platform.deepseek.com" target="_blank">https://platform.deepseek.com</a></p>
      </template>
    </el-alert>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { useSettingsStore } from '@/stores'

const settingsStore = useSettingsStore()
const formRef = ref()
const saving = ref(false)

const form = ref({
  deepseek_model: '',
  deepseek_base_url: '',
  deepseek_api_key: '',
})

const apiKeyConfigured = computed(() => settingsStore.settings?.deepseek_api_key_configured || false)
const maskedKey = computed(() => settingsStore.settings?.deepseek_api_key_masked || '')

onMounted(async () => {
  await loadSettings()
})

async function loadSettings() {
  const settings = settingsStore.settings
  if (settings) {
    form.value = {
      deepseek_model: settings.deepseek_model || '',
      deepseek_base_url: settings.deepseek_base_url || '',
      deepseek_api_key: '', // 不加载现有密钥
    }
  }
}

async function handleSave() {
  saving.value = true

  try {
    const updates: any = {
      deepseek_model: form.value.deepseek_model || undefined,
      deepseek_base_url: form.value.deepseek_base_url || undefined,
    }

    // 只有输入了新密钥才更新
    if (form.value.deepseek_api_key) {
      updates.deepseek_api_key = form.value.deepseek_api_key
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
.llm-config {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.form-hint {
  display: block;
  margin-top: 4px;
  font-size: 12px;
  color: #999;

  .masked-key {
    margin-left: 8px;
    font-family: monospace;
    color: #666;
  }

  a {
    color: var(--el-color-primary);
    text-decoration: none;

    &:hover {
      text-decoration: underline;
    }
  }
}

.info-alert {
  p {
    margin: 4px 0;
  }
}
</style>
