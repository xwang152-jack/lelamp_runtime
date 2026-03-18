/**
 * 设置模式 API 客户端
 * 提供设备首次配置相关的 API 接口
 */
import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_BASE_URL || ''

// 创建带默认配置的 axios 实例
const apiClient = axios.create({
  baseURL: API_BASE,
  timeout: 30000,  // 30 秒默认超时
  headers: {
    'Content-Type': 'application/json'
  }
})

// 添加响应拦截器统一处理错误
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    // 网络错误或超时
    if (error.code === 'ECONNABORTED') {
      error.message = '请求超时，请检查网络连接'
    } else if (error.response?.status === 401) {
      error.message = '认证失败，请重新进入设置模式'
    } else if (error.response?.status === 500) {
      error.message = '服务器错误，请稍后重试'
    }
    return Promise.reject(error)
  }
)

export interface SetupStatus {
  is_configured: boolean
  configured_wifi: string | null
  current_mode: 'ap' | 'client' | 'unknown'
  needs_setup: boolean
  can_exit_setup: boolean
  is_ap_mode: boolean
  connected_clients: Array<{ mac: string; ip: string }>
}

export interface APModeStartResponse {
  success: boolean
  message: string
  ssid: string
  password: string
  ip_address: string
}

export interface SetupCompleteResponse {
  success: boolean
  message: string
  restart_at: string
  delay_seconds: number
}

export interface APClient {
  mac: string
  ip: string
  connected_at: number
}

export interface APClientsResponse {
  clients: APClient[]
  total: number
}

/**
 * 设置模式 API
 */
export const setupApi = {
  /**
   * 获取设置状态
   */
  async getStatus(): Promise<SetupStatus> {
    const response = await apiClient.get<SetupStatus>('/api/setup/status')
    return response.data
  },

  /**
   * 启动 AP 模式
   */
  async startAPMode(ssid?: string, password?: string): Promise<APModeStartResponse> {
    const response = await apiClient.post<APModeStartResponse>('/api/setup/ap/start', {
      ssid,
      password
    })
    return response.data
  },

  /**
   * 停止 AP 模式
   */
  async stopAPMode(): Promise<{ success: boolean; message: string }> {
    const response = await apiClient.post<{ success: boolean; message: string }>('/api/setup/ap/stop')
    return response.data
  },

  /**
   * 完成配置并重启
   */
  async completeSetup(wifiSsid: string, restartDelay = 5): Promise<SetupCompleteResponse> {
    const response = await apiClient.post<SetupCompleteResponse>('/api/setup/complete', {
      wifi_ssid: wifiSsid,
      restart_delay: restartDelay
    })
    return response.data
  },

  /**
   * 重置配置
   */
  async resetSetup(): Promise<{ success: boolean; message: string }> {
    const response = await apiClient.post<{ success: boolean; message: string }>('/api/setup/reset')
    return response.data
  },

  /**
   * 获取 AP 模式下已连接的客户端
   */
  async getAPClients(): Promise<APClientsResponse> {
    const response = await apiClient.get<APClientsResponse>('/api/setup/ap/clients')
    return response.data
  }
}
