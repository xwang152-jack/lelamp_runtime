<template>
  <div class="camera-config">
    <el-card header="摄像头配置">
      <el-form
        ref="formRef"
        :model="form"
        label-width="140px"
        label-position="left"
      >
        <el-form-item label="分辨率">
          <el-select v-model="form.resolution" placeholder="选择分辨率">
            <el-option label="1920 x 1080 (FHD)" value="1920x1080" />
            <el-option label="1280 x 720 (HD)" value="1280x720" />
            <el-option label="1024 x 768" value="1024x768" />
            <el-option label="640 x 480 (VGA)" value="640x480" />
          </el-select>
          <span class="form-hint">摄像头捕获分辨率</span>
        </el-form-item>

        <el-form-item label="旋转角度">
          <el-slider
            v-model="form.camera_rotate_deg"
            :min="0"
            :max="360"
            :step="90"
            :marks="rotationMarks"
            show-stops
          />
          <span class="form-hint">图像旋转角度（度）</span>
        </el-form-item>

        <el-form-item label="镜像翻转">
          <el-select v-model="form.camera_flip" placeholder="选择翻转模式">
            <el-option label="不翻转" value="none" />
            <el-option label="水平翻转" value="horizontal" />
            <el-option label="垂直翻转" value="vertical" />
            <el-option label="双向翻转" value="both" />
          </el-select>
          <span class="form-hint">摄像头图像镜像设置</span>
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
        <p>摄像头配置修改后需要重启服务才能生效。</p>
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

const rotationMarks = {
  0: '0°',
  90: '90°',
  180: '180°',
  270: '270°',
  360: '360°',
}

const form = ref({
  resolution: '1024x768',
  camera_rotate_deg: 0,
  camera_flip: 'none',
})

onMounted(async () => {
  await loadSettings()
})

async function loadSettings() {
  const settings = settingsStore.settings
  if (settings) {
    form.value = {
      resolution: `${settings.camera_width}x${settings.camera_height}`,
      camera_rotate_deg: settings.camera_rotate_deg || 0,
      camera_flip: settings.camera_flip || 'none',
    }
  }
}

async function handleSave() {
  saving.value = true

  try {
    // 验证分辨率格式
    const parts = form.value.resolution.split('x')
    if (parts.length !== 2) {
      ElMessage.error('无效的分辨率格式')
      return
    }
    const [width, height] = parts.map(Number)
    if (isNaN(width) || isNaN(height) || width <= 0 || height <= 0) {
      ElMessage.error('分辨率必须为正整数')
      return
    }

    await settingsStore.saveSettings({
      camera_width: width,
      camera_height: height,
      camera_rotate_deg: form.value.camera_rotate_deg,
      camera_flip: form.value.camera_flip as 'none' | 'horizontal' | 'vertical' | 'both',
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
.camera-config {
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
