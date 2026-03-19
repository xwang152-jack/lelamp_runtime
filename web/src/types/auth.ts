/**
 * 认证相关类型定义
 */

/**
 * 用户信息
 */
export interface User {
  id: number
  username: string
  email: string
  is_active: boolean
  is_admin: boolean
  created_at: string
}

/**
 * 登录请求 (OAuth2 表单格式)
 */
export interface LoginRequest {
  username: string
  password: string
}

/**
 * 注册请求
 */
export interface RegisterRequest {
  username: string
  email: string
  password: string
}

/**
 * Token 响应
 */
export interface TokenResponse {
  access_token: string
  refresh_token: string
}

/**
 * 刷新 Token 请求
 */
export interface RefreshTokenRequest {
  refresh_token: string
}

/**
 * 设备绑定请求
 */
export interface DeviceBindRequest {
  device_id: string
  device_secret: string
}

/**
 * 设备绑定响应
 */
export interface DeviceBindResponse {
  device_id: string
  permission_level: string
  bound_at: string
}

/**
 * 设备绑定信息
 */
export interface DeviceBinding {
  id: number
  device_id: string
  permission_level: string
  bound_at: string
  last_used?: string
}

/**
 * 用户资料更新请求
 */
export interface ProfileUpdateRequest {
  email?: string
  current_password?: string
  new_password?: string
}

/**
 * 认证状态
 */
export interface AuthState {
  user: User | null
  accessToken: string | null
  refreshToken: string | null
  isAuthenticated: boolean
  loading: boolean
  error: string | null
}

/**
 * 订阅层级
 */
export type SubscriptionTier = 'free' | 'basic' | 'pro'

/**
 * 功能标志
 */
export interface FeatureFlags {
  motorHealth: boolean
  advancedAI: boolean
  otaUpdates: boolean
  deviceBinding: boolean
  historyRecords: boolean
}

/**
 * 订阅状态
 */
export interface SubscriptionState {
  tier: SubscriptionTier
  isPremium: boolean
  features: FeatureFlags
}

/**
 * 存储在 localStorage 的 tokens
 */
export interface StoredTokens {
  access_token: string
  refresh_token: string
  expires_at: number
}
