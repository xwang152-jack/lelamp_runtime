<template>
  <div class="light-panel">
    <!-- Color Picker Section -->
    <div class="color-section">
      <div class="section-header">
        <span class="section-icon">🎨</span>
        <span class="section-title">选择颜色</span>
      </div>

      <!-- Current Color Display -->
      <div class="current-color" :style="{ backgroundColor: displayColor }">
        <span class="color-label">{{ selectedColor }}</span>
      </div>

      <!-- Color Picker Input -->
      <div class="color-input-wrapper">
        <input
          ref="colorInput"
          v-model="selectedColor"
          type="color"
          @input="updateColorFromInput"
          class="color-input"
        />
        <button class="apply-btn" @click="setCustomColor">
          <span>应用</span>
        </button>
      </div>
    </div>

    <!-- Preset Colors -->
    <div class="preset-section">
      <div class="section-header">
        <span class="section-icon">✨</span>
        <span class="section-title">快速色彩</span>
      </div>
      <div class="color-grid">
        <div
          v-for="color in presetColors"
          :key="color.name"
          class="color-swatch"
          :class="{ 'active': isActiveColor(color) }"
          :style="{ backgroundColor: color.hex }"
          :title="color.name"
          @click="setPresetColor(color)"
        >
          <span class="swatch-glow"></span>
        </div>
      </div>
    </div>

    <!-- Effects Section -->
    <div class="effects-section">
      <div class="section-header">
        <span class="section-icon">🌟</span>
        <span class="section-title">特效动画</span>
      </div>
      <div class="effects-grid">
        <button
          v-for="effect in effects"
          :key="effect.key"
          class="effect-btn"
          @click="setEffect(effect.key)"
        >
          <span class="effect-emoji">{{ effect.emoji }}</span>
          <span class="effect-label">{{ effect.label }}</span>
        </button>
      </div>
    </div>

    <!-- Power Button -->
    <button class="power-btn" @click="turnOffLight">
      <span class="power-icon">🌙</span>
      <span class="power-label">关闭灯光</span>
    </button>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useWebSocket } from '@/composables/useWebSocket'

interface PresetColor {
  name: string
  hex: string
  rgb: { r: number; g: number; b: number }
}

const { sendCommand, sendChat } = useWebSocket()

const colorInput = ref<HTMLInputElement>()
const selectedColor = ref('#FFB6C1')

const presetColors: PresetColor[] = [
  { name: '暖红', hex: '#FF6B6B', rgb: { r: 255, g: 107, b: 107 } },
  { name: '粉红', hex: '#FFB6C1', rgb: { r: 255, g: 182, b: 193 } },
  { name: '蜜桃', hex: '#FF9A76', rgb: { r: 255, g: 154, b: 118 } },
  { name: '橙子', hex: '#FFA500', rgb: { r: 255, g: 165, b: 0 } },
  { name: '金黄', hex: '#FFD93D', rgb: { r: 255, g: 217, b: 61 } },
  { name: '草绿', hex: '#6BCB77', rgb: { r: 107, g: 203, b: 119 } },
  { name: '天空', hex: '#74B9FF', rgb: { r: 116, g: 185, b: 255 } },
  { name: '葡萄', hex: '#9370DB', rgb: { r: 147, g: 112, b: 219 } },
  { name: '樱花', hex: '#FFB7CE', rgb: { r: 255, g: 183, b: 206 } },
  { name: '薄荷', hex: '#98FFB8', rgb: { r: 152, g: 255, b: 184 } },
  { name: '薰衣草', hex: '#B8A5E8', rgb: { r: 184, g: 165, b: 232 } },
  { name: '暖白', hex: '#FFF4E5', rgb: { r: 255, g: 244, b: 229 } }
]

const effects = [
  { key: 'breathing', label: '呼吸', emoji: '💗' },
  { key: 'rainbow', label: '彩虹', emoji: '🌈' },
  { key: 'wave', label: '波浪', emoji: '🌊' },
  { key: 'fire', label: '火焰', emoji: '🔥' }
]

const displayColor = computed(() => selectedColor.value)

function isActiveColor(color: PresetColor): boolean {
  return color.hex.toLowerCase() === selectedColor.value.toLowerCase()
}

function updateColorFromInput() {
  // Just update the display, wait for apply button
}

function setCustomColor() {
  const hex = selectedColor.value
  const r = parseInt(hex.slice(1, 3), 16)
  const g = parseInt(hex.slice(3, 5), 16)
  const b = parseInt(hex.slice(5, 7), 16)
  sendCommand('set_rgb_solid', { r, g, b })
}

function setPresetColor(color: PresetColor) {
  sendCommand('set_rgb_solid', color.rgb)
  selectedColor.value = color.hex
}

function setEffect(effect: string) {
  sendCommand(`rgb_effect_${effect}`, {})
}

function turnOffLight() {
  sendChat('关闭补光灯')
}
</script>

<style lang="scss" scoped>
.light-panel {
  display: flex;
  flex-direction: column;
  gap: var(--lelamp-space-lg);
}

/* === Section Header === */
.section-header {
  display: flex;
  align-items: center;
  gap: var(--lelamp-space-sm);
  margin-bottom: var(--lelamp-space-md);
}

.section-icon {
  font-size: 1.125rem;
}

.section-title {
  font-size: 0.938rem;
  font-weight: 700;
  color: var(--lelamp-text-primary);
}

/* === Color Section === */
.color-section {
  display: flex;
  flex-direction: column;
  gap: var(--lelamp-space-md);
}

.current-color {
  height: 60px;
  border-radius: var(--lelamp-radius-md);
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.1);
  position: relative;
  overflow: hidden;
  transition: all var(--lelamp-transition-normal);

  &::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.3) 0%, transparent 50%);
  }
}

.color-label {
  position: relative;
  font-family: var(--lelamp-font-mono);
  font-size: 0.813rem;
  font-weight: 600;
  color: var(--lelamp-bg-white);
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
  padding: var(--lelamp-space-xs) var(--lelamp-space-sm);
  background: rgba(0, 0, 0, 0.2);
  border-radius: var(--lelamp-radius-sm);
}

.color-input-wrapper {
  display: flex;
  gap: var(--lelamp-space-sm);
}

.color-input {
  flex: 1;
  height: 44px;
  border: none;
  border-radius: var(--lelamp-radius-md);
  cursor: pointer;
  padding: 0 var(--lelamp-space-sm);

  &::-webkit-color-swatch-wrapper {
    padding: 0;
  }

  &::-webkit-color-swatch {
    border: none;
    border-radius: var(--lelamp-radius-sm);
  }
}

.apply-btn {
  padding: 0 var(--lelamp-space-lg);
  background: var(--lelamp-bg-gray);
  border: none;
  border-radius: var(--lelamp-radius-md);
  font-weight: 600;
  font-size: 0.875rem;
  color: var(--lelamp-text-primary);
  cursor: pointer;
  transition: all var(--lelamp-transition-bounce);

  &:hover {
    background: var(--lelamp-peach);
    color: var(--lelamp-bg-white);
    transform: translateY(-2px);
    box-shadow: var(--lelamp-shadow-sm);
  }

  &:active {
    transform: translateY(0);
  }
}

/* === Preset Colors === */
.preset-section {
  display: flex;
  flex-direction: column;
  gap: var(--lelamp-space-md);
}

.color-grid {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: var(--lelamp-space-sm);
}

.color-swatch {
  aspect-ratio: 1;
  border-radius: var(--lelamp-radius-md);
  cursor: pointer;
  position: relative;
  transition: all var(--lelamp-transition-bounce);
  border: 3px solid transparent;
  box-shadow: var(--lelamp-shadow-sm);

  &:hover {
    transform: scale(1.15);
    box-shadow: var(--lelamp-shadow-md);
  }

  &.active {
    border-color: var(--lelamp-bg-white);
    box-shadow: 0 0 0 3px var(--lelamp-peach);
    transform: scale(1.1);
  }
}

.swatch-glow {
  position: absolute;
  inset: 0;
  border-radius: var(--lelamp-radius-sm);
  background: radial-gradient(circle at 30% 30%, rgba(255, 255, 255, 0.4) 0%, transparent 50%);
}

/* === Effects Section === */
.effects-section {
  display: flex;
  flex-direction: column;
  gap: var(--lelamp-space-md);
}

.effects-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: var(--lelamp-space-sm);
}

.effect-btn {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--lelamp-space-xs);
  padding: var(--lelamp-space-sm);
  background: var(--lelamp-bg-gray);
  border: 2px solid transparent;
  border-radius: var(--lelamp-radius-md);
  cursor: pointer;
  transition: all var(--lelamp-transition-bounce);

  &:hover {
    background: var(--lelamp-bg-white);
    border-color: var(--lelamp-peach-light);
    transform: translateY(-4px);
    box-shadow: var(--lelamp-shadow-md);
  }

  &:active {
    transform: translateY(-2px);
  }
}

.effect-emoji {
  font-size: 1.25rem;
}

.effect-label {
  font-size: 0.688rem;
  font-weight: 600;
  color: var(--lelamp-text-secondary);
}

/* === Power Button === */
.power-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--lelamp-space-sm);
  padding: var(--lelamp-space-md);
  background: var(--lelamp-bg-gray);
  border: none;
  border-radius: var(--lelamp-radius-md);
  font-weight: 600;
  font-size: 0.875rem;
  color: var(--lelamp-text-secondary);
  cursor: pointer;
  transition: all var(--lelamp-transition-bounce);

  &:hover {
    background: rgba(255, 107, 138, 0.15);
    color: var(--lelamp-coral-dark);
    transform: translateY(-2px);
    box-shadow: var(--lelamp-shadow-sm);
  }

  &:active {
    transform: translateY(0);
  }
}

.power-icon {
  font-size: 1.125rem;
}

/* === Responsive === */
@media (max-width: 480px) {
  .color-grid {
    grid-template-columns: repeat(5, 1fr);
  }

  .effects-grid {
    grid-template-columns: repeat(4, 1fr);
  }

  .effect-label {
    display: none;
  }
}
</style>
