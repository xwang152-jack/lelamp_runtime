/**
 * 设置模式 Composable
 * 检测和管理设备的设置模式状态
 */
import { ref, computed } from 'vue'
import { setupApi } from '@/api/setup'

export interface SetupModeState {
  needsSetup: boolean
  isAPMode: boolean
  isConfigured: boolean
  configuredWifi: string | null
  currentMode: 'ap' | 'client' | 'unknown'
}

const state = ref<SetupModeState>({
  needsSetup: false,
  isAPMode: false,
  isConfigured: false,
  configuredWifi: null,
  currentMode: 'unknown'
})

const loading = ref(false)
const error = ref<string | null>(null)

export function useSetupMode() {
  /**
   * 检测设置模式
   */
  async function detectSetupMode(): Promise<boolean> {
    loading.value = true
    error.value = null

    try {
      const status = await setupApi.getStatus()

      state.value = {
        needsSetup: status.needs_setup,
        isAPMode: status.is_ap_mode,
        isConfigured: status.is_configured,
        configuredWifi: status.configured_wifi,
        currentMode: status.current_mode
      }

      return status.needs_setup || status.is_ap_mode
    } catch (e) {
      const message = e instanceof Error ? e.message : '检测设置模式失败'
      error.value = message
      console.error('Failed to detect setup mode:', e)
      return false
    } finally {
      loading.value = false
    }
  }

  /**
   * 启动 AP 模式
   */
  async function startAPMode(ssid?: string, password?: string): Promise<boolean> {
    loading.value = true
    error.value = null

    try {
      const result = await setupApi.startAPMode(ssid, password)

      if (result.success) {
        state.value.isAPMode = true
        state.value.currentMode = 'ap'
        return true
      }

      error.value = result.message
      return false
    } catch (e) {
      const message = e instanceof Error ? e.message : '启动 AP 模式失败'
      error.value = message
      return false
    } finally {
      loading.value = false
    }
  }

  /**
   * 停止 AP 模式
   */
  async function stopAPMode(): Promise<boolean> {
    loading.value = true
    error.value = null

    try {
      const result = await setupApi.stopAPMode()

      if (result.success) {
        state.value.isAPMode = false
        return true
      }

      error.value = result.message
      return false
    } catch (e) {
      const message = e instanceof Error ? e.message : '停止 AP 模式失败'
      error.value = message
      return false
    } finally {
      loading.value = false
    }
  }

  /**
   * 完成配置
   */
  async function completeSetup(wifiSsid: string, restartDelay = 5): Promise<boolean> {
    loading.value = true
    error.value = null

    try {
      const result = await setupApi.completeSetup(wifiSsid, restartDelay)

      if (result.success) {
        state.value.isConfigured = true
        state.value.configuredWifi = wifiSsid
        state.value.needsSetup = false
        return true
      }

      error.value = result.message
      return false
    } catch (e) {
      const message = e instanceof Error ? e.message : '完成配置失败'
      error.value = message
      return false
    } finally {
      loading.value = false
    }
  }

  /**
   * 重置配置
   */
  async function resetSetup(): Promise<boolean> {
    loading.value = true
    error.value = null

    try {
      const result = await setupApi.resetSetup()

      if (result.success) {
        state.value.isConfigured = false
        state.value.needsSetup = true
        state.value.configuredWifi = null
        return true
      }

      error.value = result.message
      return false
    } catch (e) {
      const message = e instanceof Error ? e.message : '重置配置失败'
      error.value = message
      return false
    } finally {
      loading.value = false
    }
  }

  /**
   * 刷新设置状态
   */
  async function refreshStatus(): Promise<void> {
    await detectSetupMode()
  }

  // 计算属性
  const isInSetupMode = computed(() => state.value.needsSetup || state.value.isAPMode)
  const canExitSetup = computed(() => state.value.isConfigured)
  const setupModeTitle = computed(() => {
    if (state.value.isAPMode) return '设置模式'
    if (state.value.needsSetup) return '需要配置'
    return '已配置'
  })

  return {
    // 状态
    state,
    loading,
    error,
    isInSetupMode,
    canExitSetup,
    setupModeTitle,

    // 方法
    detectSetupMode,
    startAPMode,
    stopAPMode,
    completeSetup,
    resetSetup,
    refreshStatus
  }
}
