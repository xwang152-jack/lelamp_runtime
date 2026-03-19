/**
 * 认证相关常量
 */

// Token 有效期（毫秒）
export const TOKEN_ACCESS_EXPIRY_MS = 30 * 60 * 1000 // 30 分钟

// Token 提前刷新时间（毫秒）- 在过期前多久刷新
export const TOKEN_REFRESH_THRESHOLD_MS = 25 * 60 * 1000 // 25 分钟

// 最大重试次数
export const MAX_AUTH_RETRIES = 1

// 存储键名
export const STORAGE_KEYS = {
  TOKENS: 'lelamp_tokens',
  USER: 'lelamp_user'
} as const

// WebSocket 连接超时（毫秒）
export const WS_CONNECTION_TIMEOUT = 10000
