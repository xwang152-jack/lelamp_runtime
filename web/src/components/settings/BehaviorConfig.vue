<template>
  <div class="behavior-config">
    <el-card header="行为配置">
      <el-form
        ref="formRef"
        :model="form"
        label-width="160px"
        label-position="left"
      >
        <el-form-item label="开机问候语">
          <el-input
            v-model="form.greeting_text"
            type="textarea"
            :rows="2"
            placeholder="你好！我是 LeLamp，你的智能台灯。"
            maxlength="500"
            show-word-limit
          />
          <span class="form-hint">设备启动时的欢迎语</span>
        </el-form-item>

        <el-form-item label="噪音消除">
          <el-switch v-model="form.noise_cancellation" />
          <span class="form-hint">启用语音输入的噪音消除功能</span>
        </el-form-item>

        <el-form-item label="动作冷却时间">
          <el-input-number
            v-model="form.motion_cooldown_s"
            :min="0.5"
            :max="30"
            :step="0.5"
            :precision="1"
          />
          <span class="form-hint">连续动作之间的最小间隔时间（秒）</span>
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
        <p>行为配置修改后需要重启服务才能生效。</p>
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

const form = ref({
  greeting_text: '',
  noise_cancellation: true,
  motion_cooldown_s: 2.0,
})

onMounted(async () => {
  await loadSettings()
})

async function loadSettings() {
  const settings = settingsStore.settings
  if (settings) {
    form.value = {
      greeting_text: settings.greeting_text || '',
      noise_cancellation: settings.noise_cancellation ?? true,
      motion_cooldown_s: settings.motion_cooldown_s ?? 2.0,
    }
  }
}

async function handleSave() {
  saving.value = true

  try {
    await settingsStore.saveSettings({
      greeting_text: form.value.greeting_text || undefined,
      noise_cancellation: form.value.noise_cancellation,
      motion_cooldown_s: form.value.motion_cooldown_s,
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
.behavior-config {
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
