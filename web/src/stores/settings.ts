/**
 * 设置状态管理
 */

import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type {
  AppSettings,
  WiFiNetwork,
  WiFiStatus,
  SettingsUpdate,
} from '@/types/settings'
import {
  getSettings,
  updateSettings,
  getWiFiStatus,
  scanWiFiNetworks,
  connectWiFi,
  disconnectWiFi,
} from '@/api/settings'

export const useSettingsStore = defineStore('settings', () => {
  // 状态
  const settings = ref<AppSettings | null>(null)
  const loading = ref(false)
  const error = ref<string | null>(null)

  // WiFi 状态
  const wifiStatus = ref<WiFiStatus | null>(null)
  const wifiNetworks = ref<WiFiNetwork[]>([])
  const wifiScanning = ref(false)

  // 当前设备 ID
  const currentLampId = ref<string>('lelamp')

  // 计算属性
  const hasPendingChanges = computed(() => {
    return settings.value?.requires_restart || false
  })

  const isConnectedToWiFi = computed(() => {
    return wifiStatus.value?.connected || false
  })

  // 获取设置
  async function fetchSettings(lampId?: string) {
    loading.value = true
    error.value = null

    try {
      const id = lampId || currentLampId.value
      const data = await getSettings(id)
      settings.value = data
      return data
    } catch (e) {
      error.value = e instanceof Error ? e.message : '获取设置失败'
      throw e
    } finally {
      loading.value = false
    }
  }

  // 更新设置
  async function saveSettings(updates: SettingsUpdate, lampId?: string) {
    loading.value = true
    error.value = null

    try {
      const id = lampId || currentLampId.value
      const data = await updateSettings(id, updates)
      settings.value = data
      return data
    } catch (e) {
      error.value = e instanceof Error ? e.message : '保存设置失败'
      throw e
    } finally {
      loading.value = false
    }
  }

  // 获取 WiFi 状态
  async function fetchWiFiStatus() {
    try {
      const data = await getWiFiStatus()
      wifiStatus.value = data
      return data
    } catch (e) {
      console.error('获取 WiFi 状态失败:', e)
      throw e
    }
  }

  // 扫描 WiFi 网络
  async function scanWiFi() {
    wifiScanning.value = true

    try {
      const data = await scanWiFiNetworks()
      wifiNetworks.value = data.networks
      return data.networks
    } catch (e) {
      console.error('WiFi 扫描失败:', e)
      throw e
    } finally {
      wifiScanning.value = false
    }
  }

  // 连接 WiFi
  async function connectToWiFi(ssid: string, password?: string) {
    try {
      const result = await connectWiFi({ ssid, password })
      if (result.success) {
        // 连接成功后更新状态
        await fetchWiFiStatus()
      }
      return result
    } catch (e) {
      console.error('WiFi 连接失败:', e)
      throw e
    }
  }

  // 断开 WiFi
  async function disconnectFromWiFi() {
    try {
      const result = await disconnectWiFi()
      if (result.success) {
        await fetchWiFiStatus()
      }
      return result
    } catch (e) {
      console.error('WiFi 断开失败:', e)
      throw e
    }
  }

  // 设置当前设备 ID
  function setLampId(lampId: string) {
    currentLampId.value = lampId
  }

  // 清除错误
  function clearError() {
    error.value = null
  }

  return {
    // 状态
    settings,
    loading,
    error,
    wifiStatus,
    wifiNetworks,
    wifiScanning,
    currentLampId,

    // 计算属性
    hasPendingChanges,
    isConnectedToWiFi,

    // 方法
    fetchSettings,
    saveSettings,
    fetchWiFiStatus,
    scanWiFi,
    connectToWiFi,
    disconnectFromWiFi,
    setLampId,
    clearError,
  }
})
