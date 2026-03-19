/**
 * 安全相关工具函数
 */

/**
 * CSP (Content Security Policy) 建议
 *
 * 在生产环境中，建议在服务器配置以下 CSP 头：
 *
 * Content-Security-Policy:
 *   default-src 'self';
 *   script-src 'self' 'unsafe-inline' 'unsafe-eval';
 *   style-src 'self' 'unsafe-inline';
 *   img-src 'self' data: https:;
 *   connect-src 'self' wss:// ws://;
 *   font-src 'self' data:;
 *   object-src 'none';
 *   base-uri 'self';
 *   form-action 'self';
 *   frame-ancestors 'none';
 *   block-all-mixed-content;
 */

/**
 * XSS 防护建议
 *
 * 1. Vue 模板默认会转义 HTML，这是第一层防护
 * 2. 避免使用 v-html，除非内容是可信的
 * 3. 如果必须使用 v-html，先使用 DOMPurify 清理
 * 4. 避免使用 dangerouslySetInnerHTML (React) 或类似功能
 */

/**
 * Token 存储安全建议
 *
 * 当前实现：
 * - Access Token: localStorage (容易受 XSS 攻击)
 * - Refresh Token: localStorage (容易受 XSS 攻击)
 *
 * 生产环境建议：
 * - Access Token: 内存存储 (Pinia store)
 * - Refresh Token: httpOnly cookie (服务器端设置)
 * - 实施 CSRF 保护
 * - 使用 SameSite=Strict cookie 属性
 */

/**
 * 密码安全建议
 *
 * 1. 前端验证：
 *    - 最小长度 6-8 字符
 *    - 建议包含大小写字母、数字、特殊字符
 *    - 实时强度提示
 *
 * 2. 后端验证（必须）：
 *    - 使用 bcrypt 或 argon2 哈希
 *    - 盐值随机生成
 *    - 限制登录尝试次数
 *    - 记录失败的登录尝试
 */

/**
 * API 安全建议
 *
 * 1. 所有敏感请求需要 CSRF token
 * 2. 实施 rate limiting
 * 3. 使用 HTTPS（生产环境必须）
 * 4. 设置适当的 CORS 头
 * 5. 敏感操作需要二次验证
 */

/**
 * WebSocket 安全建议
 *
 * 1. Token 通过子协议传递（而不是 URL 参数）
 * 2. 实施 origin 检查
 * 3. 限制消息大小和频率
 * 4. 实施 ping/pong 心跳检测
 * 5. 记录连接和断开事件
 */

/**
 * 获取安全的 CSP meta 标签内容
 * 可用于开发环境，生产环境应在服务器配置
 */
export const DEV_CSP_META = `default-src 'self' 'unsafe-inline' 'unsafe-eval'; connect-src 'self' wss:// ws://;`

/**
 * 清理用户输入以防止 XSS
 * 注意：Vue 模板默认转义，此函数仅用于特殊场景
 */
export function sanitizeInput(input: string): string {
  // 移除潜在的 HTML/JS 标签
  return input
    .replace(/<script[^>]*>.*?<\/script>/gi, '')
    .replace(/<[^>]*>/g, '')
    .trim()
}

/**
 * 验证密码强度
 */
export interface PasswordStrength {
  score: number // 0-4
  feedback: string
}

export function checkPasswordStrength(password: string): PasswordStrength {
  let score = 0

  if (password.length >= 8) score++
  if (password.length >= 12) score++
  if (/[a-z]/.test(password) && /[A-Z]/.test(password)) score++
  if (/\d/.test(password)) score++
  if (/[^a-zA-Z0-9]/.test(password)) score++

  const feedbacks = [
    '密码太弱',
    '密码较弱',
    '密码一般',
    '密码较强',
    '密码很强'
  ]

  return {
    score,
    feedback: feedbacks[score]
  }
}
