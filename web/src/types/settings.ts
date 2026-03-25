/**
 * 设置相关类型定义
 */

// ============================================================================
// WiFi 类型
// ============================================================================

export interface WiFiNetwork {
  ssid: string
  bssid: string
  signal_strength: number
  security: string
  frequency: string
  is_hidden: boolean
}

export interface WiFiStatus {
  connected: boolean
  ssid: string | null
  signal_strength: number | null
  ip_address: string | null
  gateway: string | null
  dns_servers: string[]
}

export interface WiFiConnectRequest {
  ssid: string
  password?: string
  hidden?: boolean
}

export interface WiFiConnectResponse {
  success: boolean
  message: string
  ssid: string
}

export interface WiFiScanResponse {
  networks: WiFiNetwork[]
  scan_time: string
  total: number
}

// ============================================================================
// 应用设置类型
// ============================================================================

export interface AppSettings {
  // UI
  theme: string
  language: string
  notifications_enabled: boolean
  brightness_level: number
  volume_level: number

  // LLM
  deepseek_model: string
  deepseek_base_url: string
  deepseek_api_key_configured: boolean
  deepseek_api_key_masked: string | null

  // Vision
  vision_enabled: boolean
  modelscope_model: string
  modelscope_api_key_configured: boolean
  modelscope_api_key_masked: string | null
  modelscope_timeout_s: number

  // Edge Vision 边缘视觉
  edge_vision_enabled: boolean
  edge_vision_prefer_local: boolean
  edge_vision_local_threshold: number

  // Camera
  camera_width: number
  camera_height: number
  camera_rotate_deg: number
  camera_flip: 'none' | 'horizontal' | 'vertical' | 'both'

  // Speech
  baidu_tts_per: number

  // Hardware
  led_brightness: number
  lamp_port: string
  lamp_id: string

  // Behavior
  greeting_text: string
  noise_cancellation: boolean
  motion_cooldown_s: number

  // Metadata
  requires_restart: boolean
  last_updated: string | null
}

export interface SettingsUpdate {
  // LLM
  deepseek_model?: string
  deepseek_base_url?: string
  deepseek_api_key?: string

  // Vision
  vision_enabled?: boolean
  modelscope_model?: string
  modelscope_api_key?: string
  modelscope_timeout_s?: number

  // Edge Vision 边缘视觉
  edge_vision_enabled?: boolean
  edge_vision_prefer_local?: boolean
  edge_vision_local_threshold?: number

  // Camera
  camera_width?: number
  camera_height?: number
  camera_rotate_deg?: number
  camera_flip?: 'none' | 'horizontal' | 'vertical' | 'both'

  // Speech
  baidu_tts_per?: number

  // Hardware
  led_brightness?: number
  lamp_port?: string
  lamp_id?: string

  // Behavior
  greeting_text?: string
  noise_cancellation?: boolean
  motion_cooldown_s?: number

  // UI
  theme?: string
  language?: string
  notifications_enabled?: boolean
  brightness_level?: number
  volume_level?: number
}

// ============================================================================
// 系统类型
// ============================================================================

export interface RestartRequest {
  delay_seconds?: number
  reason?: string
}

export interface RestartResponse {
  scheduled: boolean
  restart_at: string
  delay_seconds: number
  message: string
}

export interface SystemInfo {
  hostname: string
  uptime_seconds: number
  cpu_usage_percent: number
  memory_usage_percent: number
  disk_usage_percent: number
  wifi_available: boolean
}

// ============================================================================
// 设置字段元数据
// ============================================================================

export interface SettingField {
  name: string
  type: 'string' | 'password' | 'boolean' | 'integer' | 'float' | 'enum'
  default?: string | number | boolean
  min?: number
  max?: number
  options?: string[]
}

export interface SettingCategory {
  name: string
  fields: SettingField[]
}

export interface SettingsFields {
  categories: Record<string, SettingCategory>
}
