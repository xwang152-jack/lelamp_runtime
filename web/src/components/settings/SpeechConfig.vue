<template>
  <div class="speech-config">
    <el-card header="语音配置">
      <el-form
        ref="formRef"
        :model="form"
        label-width="200px"
        label-position="left"
      >
        <el-form-item label="TTS 音调/语速">
          <el-slider
            v-model="form.baidu_tts_per"
            :min="0"
            :max="15"
            :step="1"
            :marks="ttsMarks"
            show-input
          />
          <span class="form-hint">百度语音合成参数，影响音调和语速</span>
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
        <p>语音配置修改后需要重启服务才能生效。</p>
      </template>
    </el-alert>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { useSettingsStore } from '@/stores'

const settingsStore = useSettingsStore()
const saving = ref(false)

const ttsMarks = {
  0: '低',
  5: '中',
  10: '高',
  15: '最高',
}

const form = ref({
  baidu_tts_per: 4,
})

onMounted(async () => {
  await loadSettings()
})

async function loadSettings() {
  const settings = settingsStore.settings
  if (settings) {
    form.value = {
      baidu_tts_per: settings.baidu_tts_per || 4,
    }
  }
}

async function handleSave() {
  saving.value = true

  try {
    await settingsStore.saveSettings({
      baidu_tts_per: form.value.baidu_tts_per,
    })

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
.speech-config {
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
