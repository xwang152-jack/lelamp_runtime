import { createRouter, createWebHistory } from 'vue-router'
import type { RouteRecordRaw } from 'vue-router'
import { useAuthStore } from '@/stores'

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
    meta: { title: '控制台' }
  },
  {
    path: '/settings',
    name: 'settings',
    component: () => import('@/views/SettingsView.vue'),
    meta: { title: '系统设置' }
  },
  {
    path: '/auth',
    name: 'auth',
    component: () => import('@/views/AuthView.vue'),
    meta: { title: '登录 / 注册' }
  },
  {
    path: '/profile',
    name: 'profile',
    component: () => import('@/views/ProfileView.vue'),
    meta: { requiresLogin: true, title: '个人中心' }
  },
  {
    path: '/devices',
    name: 'devices',
    component: () => import('@/views/DeviceManageView.vue'),
    meta: { requiresLogin: true, title: '设备管理' }
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

router.beforeEach(async (to, _from, next) => {
  document.title = `${to.meta.title || 'LeLamp'} - LeLamp Web`

  // 初始化认证状态
  const authStore = useAuthStore()
  if (!authStore.user && !authStore.loading) {
    authStore.restoreAuth()
  }

  // 需要登录的路由检查
  const requiresLogin = to.meta.requiresLogin === true

  if (requiresLogin && !authStore.isAuthenticated) {
    // 保存原始目标路径
    next({
      path: '/auth',
      query: { redirect: to.fullPath }
    })
    return
  }

  // AP 模式检查：所有非 /setup 路径重定向到配网页面
  if (to.path !== '/setup') {
    try {
      const API_BASE = import.meta.env.VITE_API_BASE_URL || ''
      const response = await fetch(`${API_BASE}/api/system/setup/status`)

      if (response.ok) {
        const data = await response.json()

        // 如果设备在 AP 模式，强制跳转到配网页面
        if (data.is_ap_mode) {
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
