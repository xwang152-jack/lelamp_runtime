/**
 * Token 存储工具
 *
 * 安全注意事项：
 * 1. Access token 存储在内存中（通过 Pinia store）
 * 2. Refresh token 存储在 localStorage（实际生产环境应使用 httpOnly cookie）
 * 3. 建议实施 CSP (Content Security Policy) 来减少 XSS 风险
 * 4. 考虑使用短期 token + 长期 refresh token 策略
 */

import type { StoredTokens } from '@/types/auth'
import { STORAGE_KEYS } from './auth-constants'

/**
 * 从 localStorage 保存 tokens
 * 注意：生产环境应考虑使用 httpOnly cookie 存储 refresh token
 */
export function saveTokens(tokens: StoredTokens): void {
  try {
    localStorage.setItem(STORAGE_KEYS.TOKENS, JSON.stringify(tokens))
  } catch (e) {
    console.error('Failed to save tokens:', e)
  }
}

/**
 * 从 localStorage 获取 tokens
 */
export function getTokens(): StoredTokens | null {
  try {
    const stored = localStorage.getItem(STORAGE_KEYS.TOKENS)
    if (stored) {
      return JSON.parse(stored)
    }
  } catch (e) {
    console.error('Failed to get tokens:', e)
  }
  return null
}

/**
 * 清除 tokens
 */
export function clearTokens(): void {
  try {
    localStorage.removeItem(STORAGE_KEYS.TOKENS)
  } catch (e) {
    console.error('Failed to clear tokens:', e)
  }
}

/**
 * 检查 token 是否即将过期
 */
export function isTokenExpiringSoon(expiresAt: number, thresholdMs: number): boolean {
  return Date.now() >= (expiresAt - thresholdMs)
}

/**
 * 检查 token 是否已过期
 */
export function isTokenExpired(expiresAt: number): boolean {
  return expiresAt > 0 && Date.now() >= expiresAt
}
