<template>
  <div class="ui-config">
    <el-card header="界面设置">
      <el-form
        ref="formRef"
        :model="form"
        label-width="140px"
        label-position="left"
      >
        <el-form-item label="主题">
          <el-radio-group v-model="form.theme">
            <el-radio value="light">浅色</el-radio>
            <el-radio value="dark">深色</el-radio>
          </el-radio-group>
          <span class="form-hint">界面颜色主题</span>
        </el-form-item>

        <el-form-item label="语言">
          <el-radio-group v-model="form.language">
            <el-radio value="zh">简体中文</el-radio>
            <el-radio value="en">English</el-radio>
          </el-radio-group>
          <span class="form-hint">界面显示语言</span>
        </el-form-item>

        <el-form-item label="通知">
          <el-switch v-model="form.notifications_enabled" />
          <span class="form-hint">启用系统通知提示</span>
        </el-form-item>

        <el-form-item label="界面亮度">
          <el-slider
            v-model="form.brightness_level"
            :min="0"
            :max="100"
            :step="5"
            :marks="brightnessMarks"
            show-input
          />
          <span class="form-hint">界面亮度级别（0-100）</span>
        </el-form-item>

        <el-form-item label="音量">
          <el-slider
            v-model="form.volume_level"
            :min="0"
            :max="100"
            :step="5"
            :marks="volumeMarks"
            show-input
          />
          <span class="form-hint">系统音量级别（0-100）</span>
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
        <p>界面设置会立即生效，无需重启服务。</p>
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

const brightnessMarks = {
  0: '暗',
  50: '中',
  100: '亮',
}

const volumeMarks = {
  0: '静音',
  50: '中',
  100: '最大',
}

const form = ref({
  theme: 'light',
  language: 'zh',
  notifications_enabled: true,
  brightness_level: 25,
  volume_level: 50,
})

onMounted(async () => {
  await loadSettings()
})

async function loadSettings() {
  const settings = settingsStore.settings
  if (settings) {
    form.value = {
      theme: settings.theme || 'light',
      language: settings.language || 'zh',
      notifications_enabled: settings.notifications_enabled ?? true,
      brightness_level: settings.brightness_level || 25,
      volume_level: settings.volume_level || 50,
    }
  }
}

async function handleSave() {
  saving.value = true

  try {
    await settingsStore.saveSettings({
      theme: form.value.theme,
      language: form.value.language,
      notifications_enabled: form.value.notifications_enabled,
      brightness_level: form.value.brightness_level,
      volume_level: form.value.volume_level,
    })

    ElMessage.success('配置已保存')
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
.ui-config {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.form-hint {
  display: block;
  margin-top: 4px;
  font-size: 12px;
  color: #999;
}

.info-alert p {
  margin: 4px 0;
}
</style>
