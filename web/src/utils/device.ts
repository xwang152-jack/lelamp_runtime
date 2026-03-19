/**
 * 设备相关工具函数
 */

/**
 * 获取权限等级的中文标签
 */
export function getPermissionLabel(level: string): string {
  const labels: Record<string, string> = {
    'owner': '所有者',
    'admin': '管理员',
    'user': '用户'
  }
  return labels[level] || level
}

/**
 * 格式化日期为中文格式
 */
export function formatDate(dateStr?: string): string {
  if (!dateStr) return '未知'
  const date = new Date(dateStr)
  return date.toLocaleDateString('zh-CN', {
    year: 'numeric',
    month: 'short',
    day: 'numeric'
  })
}

/**
 * 获取用户名首字母作为头像
 */
export function getUserInitial(username?: string): string {
  return username?.charAt(0).toUpperCase() || 'U'
}
