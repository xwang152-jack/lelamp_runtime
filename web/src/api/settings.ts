/**
 * 设置相关 API 调用
 */

import type {
  WiFiStatus,
  WiFiConnectRequest,
  WiFiScanResponse,
  AppSettings,
  SettingsUpdate,
  RestartRequest,
  RestartResponse,
  SystemInfo,
  SettingsFields,
} from '@/types/settings'

import { useConnectionStore } from '@/stores/connection'

/**
 * 获取当前的 API 基础 URL
 */
function getApiBase(): string {
  const store = useConnectionStore()
  let url = store.serverUrl || import.meta.env.VITE_API_BASE_URL || ''
  if (url.endsWith('/')) {
    url = url.slice(0, -1)
  }
  // 如果 url 以 /api 结尾，去掉它，因为下面的请求路径都带了 /api
  if (url.endsWith('/api')) {
    url = url.slice(0, -4)
  }
  return url
}

/**
 * 获取请求配置
 */
function getFetchConfig(
  method: string = 'GET',
  body?: any
): RequestInit {
  const config: RequestInit = {
    method,
    headers: {
      'Content-Type': 'application/json',
    },
    credentials: 'include',
  }

  if (body) {
    config.body = JSON.stringify(body)
  }

  return config
}

/**
 * 处理 API 响应
 */
async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const error = await response.json().catch(() => ({
      detail: response.statusText || '请求失败',
    }))
    throw new Error(error.detail || error.message || '请求失败')
  }

  return response.json() as Promise<T>
}

// ============================================================================
// WiFi API
// ============================================================================

/**
 * 获取 WiFi 状态
 */
export async function getWiFiStatus(): Promise<WiFiStatus> {
  const response = await fetch(`${getApiBase()}/api/system/wifi/status`, getFetchConfig())
  return handleResponse<WiFiStatus>(response)
}

/**
 * 扫描 WiFi 网络
 */
export async function scanWiFiNetworks(): Promise<WiFiScanResponse> {
  const response = await fetch(`${getApiBase()}/api/system/wifi/scan`, getFetchConfig())
  return handleResponse<WiFiScanResponse>(response)
}

/**
 * 连接 WiFi
 */
export async function connectWiFi(request: WiFiConnectRequest): Promise<{ success: boolean; message: string; ssid: string }> {
  const response = await fetch(`${getApiBase()}/api/system/wifi/connect`, getFetchConfig('POST', request))
  return handleResponse(response)
}

/**
 * 断开 WiFi
 */
export async function disconnectWiFi(): Promise<{ success: boolean; message: string }> {
  const response = await fetch(`${getApiBase()}/api/system/wifi/disconnect`, getFetchConfig('DELETE'))
  return handleResponse(response)
}

// ============================================================================
// 设置 API
// ============================================================================

/**
 * 获取应用设置
 */
export async function getSettings(lampId: string): Promise<AppSettings> {
  const response = await fetch(`${getApiBase()}/api/settings/?lamp_id=${encodeURIComponent(lampId)}`, getFetchConfig())
  return handleResponse<AppSettings>(response)
}

/**
 * 更新应用设置
 */
export async function updateSettings(lampId: string, updates: SettingsUpdate): Promise<AppSettings> {
  const response = await fetch(
    `${getApiBase()}/api/settings/?lamp_id=${encodeURIComponent(lampId)}`,
    getFetchConfig('PUT', updates)
  )
  return handleResponse<AppSettings>(response)
}

/**
 * 重置设置
 */
export async function resetSettings(lampId: string): Promise<{ success: boolean; message: string }> {
  const response = await fetch(
    `${getApiBase()}/api/settings/reset?lamp_id=${encodeURIComponent(lampId)}`,
    getFetchConfig('POST')
  )
  return handleResponse(response)
}

/**
 * 获取设置字段元数据
 */
export async function getSettingsFields(): Promise<SettingsFields> {
  const response = await fetch(`${getApiBase()}/api/settings/fields`, getFetchConfig())
  return handleResponse<SettingsFields>(response)
}

// ============================================================================
// 系统 API
// ============================================================================

/**
 * 获取系统信息
 */
export async function getSystemInfo(): Promise<SystemInfo> {
  const response = await fetch(`${getApiBase()}/api/system/info`, getFetchConfig())
  return handleResponse<SystemInfo>(response)
}

/**
 * 触发服务重启
 */
export async function triggerRestart(request: RestartRequest = {}): Promise<RestartResponse> {
  const response = await fetch(`${getApiBase()}/api/system/restart`, getFetchConfig('POST', request))
  return handleResponse<RestartResponse>(response)
}

/**
 * 取消计划的重启
 */
export async function cancelRestart(): Promise<{ success: boolean; message: string }> {
  const response = await fetch(`${getApiBase()}/api/system/restart/cancel`, getFetchConfig('POST'))
  return handleResponse(response)
}
