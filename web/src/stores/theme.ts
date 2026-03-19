import { defineStore } from 'pinia'
import { ref, watch } from 'vue'

export type Theme = 'light' | 'dark' | 'auto'

export const useThemeStore = defineStore('theme', () => {
  // 从 localStorage 读取，或使用默认值
  const stored = localStorage.getItem('lelamp-theme') as Theme
  const current = ref<Theme>(stored || 'light')
  const systemDark = ref(false)

  // 初始化：检测系统主题偏好
  const initTheme = () => {
    // 检测系统主题偏好
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
    systemDark.value = mediaQuery.matches
    applyTheme(current.value)

    // 监听系统主题变化
    mediaQuery.addEventListener('change', (e) => {
      systemDark.value = e.matches
      if (current.value === 'auto') {
        applyTheme('auto')
      }
    })
  }

  // 应用主题到页面
  const applyTheme = (theme: Theme) => {
    const root = document.documentElement
    const isDark = theme === 'dark' || (theme === 'auto' && systemDark.value)

    // 移除旧主题类
    root.classList.remove('theme-light', 'theme-dark')

    // 添加新主题类（CSS 变量由 theme.scss 处理）
    root.classList.add(isDark ? 'theme-dark' : 'theme-light')
  }

  // 设置主题
  const setTheme = (theme: Theme) => {
    current.value = theme
    localStorage.setItem('lelamp-theme', theme)
    applyTheme(theme)
  }

  // 监听主题变化
  watch(current, (newTheme) => {
    applyTheme(newTheme)
  })

  return {
    current,
    systemDark,
    setTheme,
    initTheme
  }
})
