/**
 * 认证状态管理 Store
 */

import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { User, AuthState, StoredTokens } from '@/types/auth'
import * as authApi from '@/api/auth'
import { useSubscriptionStore } from './subscription'
import { TOKEN_ACCESS_EXPIRY_MS, TOKEN_REFRESH_THRESHOLD_MS, MAX_AUTH_RETRIES, STORAGE_KEYS } from '@/utils/auth-constants'

export const useAuthStore = defineStore('auth', () => {
  // State
  const user = ref<User | null>(null)
  const accessToken = ref<string | null>(null)
  const refreshToken = ref<string | null>(null)
  const loading = ref(false)
  const error = ref<string | null>(null)
  const tokenExpiresAt = ref<number>(0)

  // Computed
  const isAuthenticated = computed(() => {
    return !!user.value && !!accessToken.value && !isTokenExpired.value
  })

  const isTokenExpired = computed(() => {
    return tokenExpiresAt.value > 0 && Date.now() >= tokenExpiresAt.value
  })

  const shouldRefreshToken = computed(() => {
    return (
      !!accessToken.value &&
      tokenExpiresAt.value > 0 &&
      Date.now() >= (tokenExpiresAt.value - TOKEN_REFRESH_THRESHOLD_MS)
    )
  })

  const isAdmin = computed(() => user.value?.is_admin ?? false)

  // Actions

  /**
   * 从 localStorage 恢复认证状态
   */
  function restoreAuth() {
    try {
      const storedUser = localStorage.getItem(STORAGE_KEYS.USER)
      const storedTokens = localStorage.getItem(STORAGE_KEYS.TOKENS)

      if (storedUser) {
        user.value = JSON.parse(storedUser)
      }

      if (storedTokens) {
        const tokens: StoredTokens = JSON.parse(storedTokens)
        accessToken.value = tokens.access_token
        refreshToken.value = tokens.refresh_token
        tokenExpiresAt.value = tokens.expires_at

        // 检查是否需要刷新 token
        if (shouldRefreshToken.value) {
          refreshAccessToken().catch(() => {
            // 刷新失败，清除认证状态
            clearAuth()
          })
        }
      }
    } catch (e) {
      console.error('Failed to restore auth state:', e)
      clearAuth()
    }
  }

  /**
   * 保存认证状态到 localStorage
   */
  function saveAuthState(tokens: { access_token: string; refresh_token: string }) {
    const expiresAt = Date.now() + TOKEN_ACCESS_EXPIRY_MS

    accessToken.value = tokens.access_token
    refreshToken.value = tokens.refresh_token
    tokenExpiresAt.value = expiresAt

    const storedTokens: StoredTokens = {
      access_token: tokens.access_token,
      refresh_token: tokens.refresh_token,
      expires_at: expiresAt
    }

    localStorage.setItem(STORAGE_KEYS.TOKENS, JSON.stringify(storedTokens))
  }

  /**
   * 用户注册
   */
  async function register(username: string, email: string, password: string) {
    loading.value = true
    error.value = null

    try {
      const tokens = await authApi.registerUser({ username, email, password })
      saveAuthState(tokens)

      // 获取用户信息
      await fetchCurrentUser()

      return { success: true }
    } catch (e) {
      error.value = e instanceof Error ? e.message : '注册失败'
      return { success: false, error: error.value }
    } finally {
      loading.value = false
    }
  }

  /**
   * 用户登录
   */
  async function login(username: string, password: string) {
    loading.value = true
    error.value = null

    try {
      const tokens = await authApi.loginUser(username, password)
      saveAuthState(tokens)

      // 获取用户信息
      await fetchCurrentUser()

      return { success: true }
    } catch (e) {
      error.value = e instanceof Error ? e.message : '登录失败'
      return { success: false, error: error.value }
    } finally {
      loading.value = false
    }
  }

  /**
   * 退出登录
   */
  function logout() {
    user.value = null
    accessToken.value = null
    refreshToken.value = null
    tokenExpiresAt.value = 0
    error.value = null

    localStorage.removeItem(STORAGE_KEYS.TOKENS)
    localStorage.removeItem(STORAGE_KEYS.USER)

    // 重置订阅状态
    const subscriptionStore = useSubscriptionStore()
    subscriptionStore.resetToFree()
  }

  /**
   * 刷新访问令牌
   */
  async function refreshAccessToken() {
    if (!refreshToken.value) {
      throw new Error('No refresh token available')
    }

    try {
      const tokens = await authApi.refreshAccessToken(refreshToken.value)
      saveAuthState(tokens)
      return { success: true }
    } catch (e) {
      console.error('Failed to refresh token:', e)
      clearAuth()
      return { success: false, error: 'Token refresh failed' }
    }
  }

  /**
   * 获取当前用户信息
   */
  async function fetchCurrentUser(retryCount = 0) {
    if (!accessToken.value) {
      return
    }

    // 限制重试次数，防止无限递归
    if (retryCount > MAX_AUTH_RETRIES) {
      console.warn('Max retries reached for fetchCurrentUser')
      clearAuth()
      return
    }

    try {
      const userData = await authApi.getCurrentUser(accessToken.value)
      user.value = userData
      localStorage.setItem(STORAGE_KEYS.USER, JSON.stringify(userData))

      // 更新订阅状态
      const subscriptionStore = useSubscriptionStore()
      subscriptionStore.loadFromUser(userData.is_admin)
    } catch (e) {
      console.error('Failed to fetch current user:', e)
      // 如果是 401 错误，尝试刷新 token（只重试一次）
      if (e instanceof Error && e.message.includes('401') && retryCount < MAX_AUTH_RETRIES) {
        const result = await refreshAccessToken()
        if (result.success) {
          // 重试获取用户信息
          return fetchCurrentUser(retryCount + 1)
        }
      }
      // 刷新失败，清除认证状态
      clearAuth()
    }
  }

  /**
   * 绑定设备
   */
  async function bindDevice(deviceId: string, deviceSecret: string) {
    if (!accessToken.value) {
      return { success: false, error: '请先登录' }
    }

    loading.value = true
    error.value = null

    try {
      await authApi.bindDevice({ device_id: deviceId, device_secret: deviceSecret }, accessToken.value)
      return { success: true }
    } catch (e) {
      error.value = e instanceof Error ? e.message : '设备绑定失败'
      return { success: false, error: error.value }
    } finally {
      loading.value = false
    }
  }

  /**
   * 获取用户绑定的设备列表
   */
  async function getUserDevices() {
    if (!accessToken.value) {
      return []
    }

    try {
      return await authApi.getUserDevices(accessToken.value)
    } catch (e) {
      console.error('Failed to get user devices:', e)
      return []
    }
  }

  /**
   * 解绑设备
   */
  async function unbindDevice(deviceId: string) {
    if (!accessToken.value) {
      return { success: false, error: '请先登录' }
    }

    loading.value = true
    error.value = null

    try {
      await authApi.unbindDevice(deviceId, accessToken.value)
      return { success: true }
    } catch (e) {
      error.value = e instanceof Error ? e.message : '设备解绑失败'
      return { success: false, error: error.value }
    } finally {
      loading.value = false
    }
  }

  /**
   * 更新用户资料
   */
  async function updateProfile(data: { email?: string; current_password?: string; new_password?: string }) {
    if (!accessToken.value) {
      return { success: false, error: '请先登录' }
    }

    loading.value = true
    error.value = null

    try {
      const updatedUser = await authApi.updateProfile(data, accessToken.value)
      user.value = updatedUser
      localStorage.setItem(STORAGE_KEYS.USER, JSON.stringify(updatedUser))
      return { success: true }
    } catch (e) {
      error.value = e instanceof Error ? e.message : '更新失败'
      return { success: false, error: error.value }
    } finally {
      loading.value = false
    }
  }

  /**
   * 清除认证状态
   */
  function clearAuth() {
    logout()
  }

  return {
    // State
    user,
    accessToken,
    refreshToken,
    loading,
    error,
    tokenExpiresAt,

    // Computed
    isAuthenticated,
    isTokenExpired,
    shouldRefreshToken,
    isAdmin,

    // Actions
    restoreAuth,
    login,
    register,
    logout,
    refreshAccessToken,
    fetchCurrentUser,
    bindDevice,
    getUserDevices,
    unbindDevice,
    updateProfile,
    clearAuth
  }
})
