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

  // 检查设备配置状态
  try {
    const API_BASE = import.meta.env.VITE_API_BASE_URL || ''
    const response = await fetch(`${API_BASE}/api/system/setup/status`)
    
    if (response.ok) {
      const data = await response.json()
      
      // 如果设备在 AP 模式或需要设置
      if (data.is_ap_mode || data.needs_setup) {
        // 如果当前不在设置页面,跳转到设置页面
        if (to.path !== '/setup') {
          next({ path: '/setup', replace: true })
          return
        }
      } 
      // 如果设备已配置且当前在根路径或连接页面
      else if (data.is_configured && (to.path === '/' || to.path === '/connect')) {
        // 跳过连接页面,直接进入控制台
        next({ path: '/room', replace: true })
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
