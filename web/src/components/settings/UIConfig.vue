<template>
  <div class="ui-config">
    <div class="config-header">
      <div class="header-icon">🎨</div>
      <div class="header-text">
        <h2 class="header-title">界面设置</h2>
        <p class="header-desc">自定义你的 LeLamp 外观</p>
      </div>
    </div>

    <div class="config-sections">
      <!-- Theme Section -->
      <section class="config-section">
        <h3 class="section-title">
          <span class="title-icon">🌗</span>
          主题模式
        </h3>
        <div class="option-grid">
          <button
            v-for="theme in themes"
            :key="theme.value"
            :class="['option-card', { 'active': form.theme === theme.value }]"
            @click="selectTheme(theme.value)"
          >
            <span class="option-emoji">{{ theme.emoji }}</span>
            <span class="option-label">{{ theme.label }}</span>
            <div class="option-check" v-if="form.theme === theme.value">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3">
                <polyline points="20 6 9 17 4 12" />
              </svg>
            </div>
          </button>
        </div>
      </section>

      <!-- Language Section -->
      <section class="config-section">
        <h3 class="section-title">
          <span class="title-icon">🌐</span>
          语言设置
        </h3>
        <div class="option-list">
          <button
            v-for="lang in languages"
            :key="lang.value"
            :class="['option-row', { 'active': form.language === lang.value }]"
            @click="form.language = lang.value"
          >
            <span class="lang-emoji">{{ lang.emoji }}</span>
            <span class="lang-label">{{ lang.label }}</span>
            <div class="lang-check" v-if="form.language === lang.value">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3">
                <polyline points="20 6 9 17 4 12" />
              </svg>
            </div>
          </button>
        </div>
      </section>

      <!-- Brightness Section -->
      <section class="config-section">
        <h3 class="section-title">
          <span class="title-icon">💡</span>
          界面亮度
        </h3>
        <div class="brightness-control">
          <div class="brightness-slider">
            <input
              v-model.number="form.brightness_level"
              type="range"
              min="10"
              max="100"
              class="slider-input"
            />
            <div class="slider-track" :style="{ background: brightnessGradient }"></div>
          </div>
          <div class="brightness-value">{{ form.brightness_level }}%</div>
        </div>
      </section>

      <!-- Notifications Section -->
      <section class="config-section">
        <h3 class="section-title">
          <span class="title-icon">🔔</span>
          通知设置
        </h3>
        <div class="toggle-group">
          <label class="toggle-row">
            <input type="checkbox" v-model="form.notifications_enabled" class="toggle-input">
            <div class="toggle-switch">
              <div class="toggle-thumb"></div>
            </div>
            <span class="toggle-label">启用通知</span>
          </label>
        </div>
      </section>
    </div>

    <!-- Action Buttons -->
    <div class="config-actions">
      <button class="action-btn save" @click="handleSave">
        <span class="btn-icon">💾</span>
        保存设置
      </button>
      <button class="action-btn reset" @click="handleReset">
        <span class="btn-icon">🔄</span>
        恢复默认
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { useThemeStore } from '@/stores'

const themeStore = useThemeStore()

const form = ref({
  theme: 'light',
  language: 'zh',
  brightness_level: 25,
  notifications_enabled: true
})

// 组件挂载时同步当前主题
onMounted(() => {
  form.value.theme = themeStore.current
})

const themes = [
  { value: 'light', emoji: '☀️', label: '浅色' },
  { value: 'dark', emoji: '🌙', label: '深色' },
  { value: 'auto', emoji: '🌗', label: '自动' }
]

const languages = [
  { value: 'zh', emoji: '🇨🇳', label: '简体中文' },
  { value: 'en', emoji: '🇺🇸', label: 'English' }
]

const brightnessGradient = computed(() => {
  const value = form.value.brightness_level
  return `linear-gradient(90deg, var(--lelamp-peach) ${value}%, #e5e7eb ${value}%)`
})

function selectTheme(value: string) {
  form.value.theme = value
  // 实际应用主题
  themeStore.setTheme(value as 'light' | 'dark' | 'auto')
  ElMessage.success(`已切换到${themes.find(t => t.value === value)?.label || value}主题`)
}

async function handleSave() {
  try {
    ElMessage.success('设置已保存')
  } catch (error) {
    ElMessage.error('保存失败')
  }
}

function handleReset() {
  form.value = {
    theme: 'light',
    language: 'zh',
    brightness_level: 25,
    notifications_enabled: true
  }
  ElMessage.info('已恢复默认设置')
}
</script>

<style lang="scss" scoped>
.ui-config {
  display: flex;
  flex-direction: column;
  gap: var(--lelamp-space-lg);
}

/* === Header === */
.config-header {
  display: flex;
  align-items: center;
  gap: var(--lelamp-space-sm);
  padding-bottom: var(--lelamp-space-md);
  border-bottom: 1px dashed rgba(0, 0, 0, 0.1);
}

.header-icon {
  font-size: 1.75rem;  /* 减小 */
}

.header-text {
  flex: 1;
}

.header-title {
  font-family: var(--lelamp-font-display);
  font-size: 1.25rem;  /* 减小 */
  font-weight: 700;
  color: var(--lelamp-text-primary);
  margin: 0;
}

.header-desc {
  font-size: 0.875rem;
  color: var(--lelamp-text-secondary);
  margin: 0;
}

/* === Sections === */
.config-sections {
  display: flex;
  flex-direction: column;
  gap: var(--lelamp-space-md);
}

.config-section {
  display: flex;
  flex-direction: column;
  gap: var(--lelamp-space-sm);
}

.section-title {
  display: flex;
  align-items: center;
  gap: var(--lelamp-space-xs);
  font-family: var(--lelamp-font-display);
  font-size: 0.938rem;
  font-weight: 700;
  color: var(--lelamp-text-primary);
  margin: 0;
}

.title-icon {
  font-size: 1rem;
}

/* === Option Grid === */
.option-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: var(--lelamp-space-xs);
}

.option-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
  padding: var(--lelamp-space-sm);  /* 减小内边距 */
  background: var(--lelamp-bg-gray);
  border: 2px solid transparent;
  border-radius: var(--lelamp-radius-md);  /* 减小圆角 */
  cursor: pointer;
  transition: all var(--lelamp-transition-normal);
  position: relative;

  &:hover {
    background: var(--lelamp-bg-white);
    border-color: var(--lelamp-peach-light);
    transform: translateY(-2px);
    box-shadow: var(--lelamp-shadow-sm);
  }

  &.active {
    background: linear-gradient(135deg, var(--lelamp-peach-light), var(--lelamp-sunny-light));
    border-color: var(--lelamp-peach);
  }

  &:active {
    transform: translateY(0);
  }
}

.option-emoji {
  font-size: 1.5rem;  /* 减小 */
}

.option-label {
  font-size: 0.75rem;  /* 减小 */
  font-weight: 600;
  color: var(--lelamp-text-primary);
}

.option-check {
  position: absolute;
  top: 4px;
  right: 4px;
  width: 16px;  /* 减小 */
  height: 16px;
  background: var(--lelamp-bg-white);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;

  svg {
    width: 10px;
    height: 10px;
    color: var(--lelamp-peach);
  }
}

/* === Option List === */
.option-list {
  display: flex;
  flex-direction: column;
  gap: var(--lelamp-space-xs);
}

.option-row {
  display: flex;
  align-items: center;
  gap: var(--lelamp-space-sm);
  padding: var(--lelamp-space-sm);
  background: var(--lelamp-bg-gray);
  border: 2px solid transparent;
  border-radius: var(--lelamp-radius-sm);  /* 减小圆角 */
  cursor: pointer;
  transition: all var(--lelamp-transition-normal);

  &:hover {
    background: var(--lelamp-bg-white);
    border-color: var(--lelamp-peach-light);
  }

  &.active {
    background: linear-gradient(135deg, var(--lelamp-peach-light), var(--lelamp-sunny-light));
    border-color: var(--lelamp-peach);
  }
}

.lang-emoji {
  font-size: 1.25rem;  /* 减小 */
}

.lang-label {
  flex: 1;
  font-weight: 600;
  font-size: 0.875rem;  /* 减小 */
  color: var(--lelamp-text-primary);
  text-align: left;
}

.lang-check {
  width: 20px;  /* 减小 */
  height: 20px;
  background: var(--lelamp-bg-white);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;

  svg {
    width: 12px;
    height: 12px;
    color: var(--lelamp-peach);
  }
}

/* === Brightness Slider === */
.brightness-control {
  display: flex;
  align-items: center;
  gap: var(--lelamp-space-sm);
  padding: var(--lelamp-space-sm);
  background: var(--lelamp-bg-gray);
  border-radius: var(--lelamp-radius-md);
}

.brightness-slider {
  flex: 1;
  position: relative;
  height: 24px;
}

.slider-input {
  width: 100%;
  height: 6px;
  -webkit-appearance: none;
  appearance: none;
  background: transparent;
  cursor: pointer;
  position: relative;
  z-index: 2;

  &::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 18px;
    height: 18px;
    background: var(--lelamp-bg-white);
    border: 3px solid var(--lelamp-peach);
    border-radius: 50%;
    cursor: grab;
    box-shadow: var(--lelamp-shadow-sm);
    transition: transform var(--lelamp-transition-normal);

    &:hover {
      transform: scale(1.1);
    }

    &:active {
      cursor: grabbing;
      transform: scale(1.05);
    }
  }

  &::-moz-range-thumb {
    width: 18px;
    height: 18px;
    background: var(--lelamp-bg-white);
    border: 3px solid var(--lelamp-peach);
    border-radius: 50%;
    cursor: grab;
    box-shadow: var(--lelamp-shadow-sm);
  }

  &::-webkit-slider-runnable-track {
    height: 6px;
    border-radius: var(--lelamp-radius-full);
  }
}

.slider-track {
  position: absolute;
  top: 50%;
  left: 0;
  right: 0;
  height: 6px;
  transform: translateY(-50%);
  border-radius: var(--lelamp-radius-full);
  pointer-events: none;
}

.brightness-value {
  min-width: 40px;
  text-align: center;
  font-family: var(--lelamp-font-display);
  font-size: 0.938rem;  /* 减小 */
  font-weight: 700;
  color: var(--lelamp-peach);
}

/* === Toggle === */
.toggle-group {
  display: flex;
  flex-direction: column;
  gap: var(--lelamp-space-xs);
}

.toggle-row {
  display: flex;
  align-items: center;
  gap: var(--lelamp-space-sm);
  padding: var(--lelamp-space-sm);
  background: var(--lelamp-bg-gray);
  border-radius: var(--lelamp-radius-sm);
  cursor: pointer;
  user-select: none;

  &:hover {
    background: var(--lelamp-bg-white);
  }
}

.toggle-input {
  display: none;
}

.toggle-switch {
  position: relative;
  width: 40px;  /* 减小 */
  height: 24px;  /* 减小 */
  background: #d1d5db;
  border-radius: var(--lelamp-radius-full);
  transition: background var(--lelamp-transition-normal);
}

.toggle-thumb {
  position: absolute;
  top: 2px;
  left: 2px;
  width: 20px;  /* 减小 */
  height: 20px;
  background: var(--lelamp-bg-white);
  border-radius: 50%;
  box-shadow: var(--lelamp-shadow-sm);
  transition: transform var(--lelamp-transition-normal);
}

.toggle-input:checked + .toggle-switch {
  background: var(--lelamp-peach);
}

.toggle-input:checked + .toggle-switch .toggle-thumb {
  transform: translateX(16px);
}

.toggle-label {
  font-weight: 600;
  font-size: 0.875rem;  /* 减小 */
  color: var(--lelamp-text-primary);
}

/* === Action Buttons === */
.config-actions {
  display: flex;
  gap: var(--lelamp-space-sm);
  padding-top: var(--lelamp-space-md);
  border-top: 1px dashed rgba(0, 0, 0, 0.1);
}

.action-btn {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--lelamp-space-xs);
  padding: var(--lelamp-space-sm);  /* 减小内边距 */
  font-family: var(--lelamp-font-display);
  font-size: 0.938rem;  /* 减小 */
  font-weight: 700;
  border: none;
  border-radius: var(--lelamp-radius-lg);
  cursor: pointer;
  transition: all var(--lelamp-transition-bounce);

  &.save {
    background: linear-gradient(135deg, var(--lelamp-peach), var(--lelamp-coral));
    color: var(--lelamp-bg-white);
    box-shadow: 0 4px 16px rgba(255, 107, 138, 0.3);

    &:hover {
      transform: translateY(-2px);
      box-shadow: 0 8px 24px rgba(255, 107, 138, 0.4);
    }
  }

  &.reset {
    background: var(--lelamp-bg-gray);
    color: var(--lelamp-text-secondary);

    &:hover {
      background: #e5e7eb;
      transform: translateY(-2px);
    }
  }

  &:active {
    transform: translateY(0);
  }
}

.btn-icon {
  font-size: 1rem;  /* 减小 */
}

/* === Responsive === */
@media (max-width: 480px) {
  .option-grid {
    grid-template-columns: 1fr;
  }

  .config-actions {
    flex-direction: column;
  }

  .brightness-value {
    min-width: 35px;
    font-size: 0.875rem;
  }
}
</style>
