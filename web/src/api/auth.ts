/**
 * 认证相关 API 调用
 */

import type {
  User,
  TokenResponse,
  RegisterRequest,
  DeviceBindRequest,
  DeviceBindResponse,
  RefreshTokenRequest,
  ProfileUpdateRequest
} from '@/types/auth'

/**
 * 获取当前的 API 基础 URL
 */
function getApiBase(): string {
  const storedUrl = sessionStorage.getItem('lelamp_server_url')
  if (storedUrl) return storedUrl

  const url = import.meta.env.VITE_API_BASE_URL || ''
  if (url.endsWith('/')) {
    return url.slice(0, -1)
  }
  return url
}

/**
 * 获取请求配置（包含认证头）
 */
function getFetchConfig(
  method: string = 'GET',
  body?: any,
  token?: string
): RequestInit {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json'
  }

  if (token) {
    headers['Authorization'] = `Bearer ${token}`
  }

  const config: RequestInit = {
    method,
    headers,
    credentials: 'include'
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
    let errorMessage = `请求失败 (${response.status})`
    try {
      const error = await response.json()
      errorMessage = error.detail || error.message || errorMessage
    } catch {
      errorMessage = response.statusText || errorMessage
    }
    throw new Error(errorMessage)
  }

  try {
    return await response.json() as Promise<T>
  } catch {
    throw new Error('响应解析失败')
  }
}

// ============================================================================
// 认证 API
// ============================================================================

/**
 * 用户注册
 */
export async function registerUser(data: RegisterRequest): Promise<TokenResponse> {
  const response = await fetch(
    `${getApiBase()}/api/auth/register`,
    getFetchConfig('POST', data)
  )
  return handleResponse<TokenResponse>(response)
}

/**
 * 用户登录 (OAuth2 表单格式)
 */
export async function loginUser(username: string, password: string): Promise<TokenResponse> {
  // OAuth2 使用 form-data 格式
  const formData = new FormData()
  formData.append('username', username)
  formData.append('password', password)

  const response = await fetch(`${getApiBase()}/api/auth/login`, {
    method: 'POST',
    body: formData,
    credentials: 'include'
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: '登录失败' }))
    throw new Error(error.detail || '登录失败')
  }

  return response.json() as Promise<TokenResponse>
}

/**
 * 获取当前用户信息
 */
export async function getCurrentUser(token: string): Promise<User> {
  const response = await fetch(
    `${getApiBase()}/api/auth/me`,
    getFetchConfig('GET', undefined, token)
  )
  return handleResponse<User>(response)
}

/**
 * 刷新访问令牌
 */
export async function refreshAccessToken(refreshToken: string): Promise<TokenResponse> {
  const response = await fetch(
    `${getApiBase()}/api/auth/refresh-token`,
    getFetchConfig('POST', { refresh_token: refreshToken } as RefreshTokenRequest)
  )
  return handleResponse<TokenResponse>(response)
}

/**
 * 绑定设备
 */
export async function bindDevice(
  request: DeviceBindRequest,
  token: string
): Promise<DeviceBindResponse> {
  const response = await fetch(
    `${getApiBase()}/api/auth/bind-device`,
    getFetchConfig('POST', request, token)
  )
  return handleResponse<DeviceBindResponse>(response)
}

/**
 * 获取用户绑定的设备列表
 */
export async function getUserDevices(token: string): Promise<DeviceBindResponse[]> {
  const response = await fetch(
    `${getApiBase()}/api/auth/devices`,
    getFetchConfig('GET', undefined, token)
  )
  return handleResponse<DeviceBindResponse[]>(response)
}

/**
 * 解绑设备
 */
export async function unbindDevice(deviceId: string, token: string): Promise<{ success: boolean }> {
  const response = await fetch(
    `${getApiBase()}/api/auth/devices/${encodeURIComponent(deviceId)}`,
    getFetchConfig('DELETE', undefined, token)
  )
  return handleResponse<{ success: boolean }>(response)
}

/**
 * 更新用户资料
 */
export async function updateProfile(
  data: ProfileUpdateRequest,
  token: string
): Promise<User> {
  const response = await fetch(
    `${getApiBase()}/api/auth/profile`,
    getFetchConfig('PUT', data, token)
  )
  return handleResponse<User>(response)
}
