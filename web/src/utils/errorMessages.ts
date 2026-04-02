/**
 * 将 API 错误转换为用户友好的中文消息
 */

const ERROR_MAP: Record<string, string> = {
  'Incorrect username or password': '用户名或密码错误，请重新输入',
  'Username already exists': '该用户名已被注册，请换一个或直接登录',
  'Email already exists': '该邮箱已被注册，请直接登录',
  '设备密钥未配置，无法自动绑定': '设备绑定已跳过，后续可在设置页面完成',
  'Invalid device secret': '设备验证失败，请重新开始配置',
  'Internal server error': '设备遇到内部错误，请重启后再试',
}

const WIFI_FAILURE_HINT = '密码可能有误，或信号较弱，建议靠近路由器后重试'
const NETWORK_FAILURE_HINT = 'WiFi 已连接，但无法访问互联网，请检查路由器设置'
const GENERIC_ERROR = '操作失败，请重试'

export function formatApiError(error: unknown, context?: 'wifi' | 'network' | 'auth'): string {
  // axios error with response
  if (error && typeof error === 'object' && 'response' in error) {
    const axiosErr = error as { response?: { data?: { detail?: string } } }
    const detail = axiosErr.response?.data?.detail
    if (detail) {
      return ERROR_MAP[detail] ?? detail
    }
  }

  // Error with message
  if (error instanceof Error) {
    if (context === 'wifi') return WIFI_FAILURE_HINT
    if (context === 'network') return NETWORK_FAILURE_HINT
    return ERROR_MAP[error.message] ?? error.message ?? GENERIC_ERROR
  }

  if (typeof error === 'string') {
    return ERROR_MAP[error] ?? error
  }

  return GENERIC_ERROR
}

export { WIFI_FAILURE_HINT, NETWORK_FAILURE_HINT }
