<template>
  <div class="light-panel">
    <h3>💡 灯光魔法</h3>

    <!-- 颜色选择器 -->
    <div class="color-picker-section">
      <label>选择颜色:</label>
      <div class="color-input-row">
        <input
          ref="colorInput"
          type="color"
          v-model="selectedColor"
          @change="setCustomColor"
        />
        <el-button size="small" @click="setCustomColor">设置</el-button>
      </div>
    </div>

    <!-- 预设颜色 -->
    <div class="preset-colors">
      <h4>快速颜色</h4>
      <div class="color-grid">
        <div
          v-for="color in presetColors"
          :key="color.name"
          class="color-swatch"
          :style="{ backgroundColor: color.hex }"
          :title="color.name"
          @click="setPresetColor(color)"
        />
      </div>
    </div>

    <!-- 特效动画 -->
    <div class="effects-section">
      <h4>特效动画</h4>
      <div class="effects-grid">
        <el-button
          v-for="effect in effects"
          :key="effect.key"
          size="small"
          @click="setEffect(effect.key)"
        >
          {{ effect.emoji }} {{ effect.label }}
        </el-button>
      </div>
    </div>

    <!-- 关灯 -->
    <div class="actions">
      <el-button type="danger" size="small" @click="turnOffLight">
        🌑 关闭灯光
      </el-button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { ElMessage } from 'element-plus'
import { useLiveKit } from '@/composables/useLiveKit'

interface PresetColor {
  name: string
  hex: string
  rgb: { r: number; g: number; b: number }
}

const { sendCommand, sendChat } = useLiveKit()

const colorInput = ref<HTMLInputElement>()
const selectedColor = ref('#FFB6C1')

const presetColors: PresetColor[] = [
  { name: '暖红', hex: '#FF6B6B', rgb: { r: 255, g: 107, b: 107 } },
  { name: '粉红', hex: '#FFB6C1', rgb: { r: 255, g: 182, b: 193 } },
  { name: '橙色', hex: '#FFA500', rgb: { r: 255, g: 165, b: 0 } },
  { name: '金黄', hex: '#FFD700', rgb: { r: 255, g: 215, b: 0 } },
  { name: '浅绿', hex: '#90EE90', rgb: { r: 144, g: 238, b: 144 } },
  { name: '天蓝', hex: '#87CEEB', rgb: { r: 135, g: 206, b: 235 } },
  { name: '紫色', hex: '#9370DB', rgb: { r: 147, g: 112, b: 219 } },
  { name: '暖白', hex: '#FFF4E5', rgb: { r: 255, g: 244, b: 229 } }
]

const effects = [
  { key: 'breathing', label: '呼吸灯', emoji: '💗' },
  { key: 'rainbow', label: '彩虹', emoji: '🌈' },
  { key: 'wave', label: '波浪', emoji: '🌊' },
  { key: 'fire', label: '火焰', emoji: '🔥' },
  { key: 'fireworks', label: '烟花', emoji: '🎆' },
  { key: 'starry', label: '星空', emoji: '⭐' }
]

function setCustomColor() {
  const hex = selectedColor.value
  const r = parseInt(hex.slice(1, 3), 16)
  const g = parseInt(hex.slice(3, 5), 16)
  const b = parseInt(hex.slice(5, 7), 16)
  sendCommand('set_rgb_solid', { r, g, b })
  ElMessage.success('设置颜色')
}

function setPresetColor(color: PresetColor) {
  sendCommand('set_rgb_solid', color.rgb)
  selectedColor.value = color.hex
  ElMessage.success(`设置颜色: ${color.name}`)
}

function setEffect(effect: string) {
  sendCommand(`rgb_effect_${effect}`, {})
  ElMessage.success(`设置特效: ${effect}`)
}

function turnOffLight() {
  sendChat('关闭补光灯')
}
</script>

<style lang="scss" scoped>
.light-panel {
  padding: 20px;
  background: white;
  border-radius: var(--radius-md);
  box-shadow: var(--shadow-sm);

  h3 {
    margin-bottom: 16px;
  }

  h4 {
    font-size: 14px;
    margin: 16px 0 8px;
    color: #606266;
  }

  .color-picker-section {
    margin-bottom: 16px;

    label {
      display: block;
      font-size: 14px;
      margin-bottom: 8px;
      color: #606266;
    }

    .color-input-row {
      display: flex;
      gap: 8px;
      align-items: center;

      input[type="color"] {
        width: 60px;
        height: 32px;
        border: 1px solid #dcdfe6;
        border-radius: 4px;
        cursor: pointer;
      }
    }
  }

  .preset-colors {
    .color-grid {
      display: grid;
      grid-template-columns: repeat(8, 1fr);
      gap: 6px;

      .color-swatch {
        width: 100%;
        aspect-ratio: 1;
        border-radius: 4px;
        cursor: pointer;
        transition: transform 0.2s;
        border: 2px solid transparent;

        &:hover {
          transform: scale(1.1);
          border-color: var(--lelamp-brand);
        }
      }
    }
  }

  .effects-section {
    .effects-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 6px;
    }
  }

  .actions {
    margin-top: 16px;
    padding-top: 16px;
    border-top: 1px solid #eee;
  }
}
</style>
