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

  // 如果用户已连接 WebSocket（在 room 或 settings 页面），允许访问
  // 从 connectionStore 检查连接状态需要异步导入，这里简化处理
  const isConnected = sessionStorage.getItem('lelamp_connected') === 'true'

  // 如果用户正在连接页面或已连接，跳过设置检查
  if (to.path === '/connect' || to.path === '/room' || to.path === '/settings' || isConnected) {
    // 对于 room 和 settings，如果未连接则跳转到 connect
    if ((to.path === '/room' || to.path === '/settings') && !isConnected) {
      // 允许通过，让页面自己处理连接状态
    }
    next()
    return
  }

  // 检查设备配置状态（仅在访问首页时）
  try {
    const API_BASE = import.meta.env.VITE_API_BASE_URL || ''
    const response = await fetch(`${API_BASE}/api/system/setup/status`)

    if (response.ok) {
      const data = await response.json()

      // 如果设备在 AP 模式，跳转到设置页面
      if (data.is_ap_mode && to.path !== '/setup') {
        next({ path: '/setup', replace: true })
        return
      }
    }
  } catch (error) {
    // API 不可用时继续正常流程
    console.warn('Failed to check setup status:', error)
  }

  next()
})

export default router
