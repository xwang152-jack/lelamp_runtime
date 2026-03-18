<template>
  <div class="hardware-config">
    <el-card header="硬件配置">
      <el-form
        ref="formRef"
        :model="form"
        label-width="140px"
        label-position="left"
      >
        <el-form-item label="LED 亮度">
          <el-slider
            v-model="form.led_brightness"
            :min="0"
            :max="100"
            :step="5"
            :marks="brightnessMarks"
            show-input
          />
          <span class="form-hint">LED 矩阵默认亮度（0-100）</span>
        </el-form-item>

        <el-form-item label="串口">
          <el-input
            v-model="form.lamp_port"
            placeholder="/dev/ttyACM0"
          />
          <span class="form-hint">电机控制串口设备路径</span>
        </el-form-item>

        <el-form-item label="设备 ID">
          <el-input
            v-model="form.lamp_id"
            placeholder="lelamp"
          />
          <span class="form-hint">设备唯一标识符</span>
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
      type="warning"
      :closable="false"
      show-icon
      class="info-alert"
    >
      <template #default>
        <p>硬件配置修改后需要重启服务才能生效。</p>
        <p>修改串口或设备 ID 可能导致设备无法正常连接，请谨慎操作。</p>
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
  0: '关闭',
  25: '低',
  50: '中',
  75: '高',
  100: '最亮',
}

const form = ref({
  led_brightness: 25,
  lamp_port: '',
  lamp_id: '',
})

onMounted(async () => {
  await loadSettings()
})

async function loadSettings() {
  const settings = settingsStore.settings
  if (settings) {
    form.value = {
      led_brightness: settings.led_brightness || 25,
      lamp_port: settings.lamp_port || '/dev/ttyACM0',
      lamp_id: settings.lamp_id || 'lelamp',
    }
  }
}

async function handleSave() {
  saving.value = true

  try {
    await settingsStore.saveSettings({
      led_brightness: form.value.led_brightness,
      lamp_port: form.value.lamp_port || undefined,
      lamp_id: form.value.lamp_id || undefined,
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
.hardware-config {
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
