import { createRouter, createWebHistory } from 'vue-router'
import type { RouteRecordRaw } from 'vue-router'

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    redirect: '/connect'
  },
  {
    path: '/connect',
    name: 'connect',
    component: () => import('@/views/ConnectView.vue'),
    meta: { title: '连接设备' }
  },
  {
    path: '/setup',
    name: 'setup',
    component: () => import('@/views/SetupWizardView.vue'),
    meta: { title: '设备设置' }
  },
  {
    path: '/room',
    name: 'room',
    component: () => import('@/views/RoomView.vue'),
    meta: { requiresAuth: true, title: '控制台' }
  },
  {
    path: '/settings',
    name: 'settings',
    component: () => import('@/views/SettingsView.vue'),
    meta: { requiresAuth: true, title: '系统设置' }
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

router.beforeEach(async (to, _from, next) => {
  document.title = `${to.meta.title || 'LeLamp'} - LeLamp Web`

  // 检测是否需要进入设置模式
  // 如果访问 /connect 但设备未配置，重定向到 /setup
  if (to.path === '/connect') {
    try {
      const API_BASE = import.meta.env.VITE_API_BASE_URL || ''
      const response = await fetch(`${API_BASE}/api/setup/status`)
      if (response.ok) {
        const data = await response.json()
        if (data.needs_setup || data.is_ap_mode) {
          next({ path: '/setup', replace: true })
          return
        }
      }
    } catch (error) {
      // API 不可用时继续正常流程
      console.warn('Failed to check setup status:', error)
    }
  }

  next()
})

export default router
