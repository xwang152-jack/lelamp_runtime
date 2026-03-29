export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'error'

export interface ConnectionState {
  isConnected: boolean
  ws: WebSocket | null
  serverUrl: string
  token: string
  connectionStatus: ConnectionStatus
}

export interface DataMessage {
  type: 'chat' | 'command' | 'camera_status' | 'vision_result' | 'connected' | 'state_update' | 'event' | 'log' | 'notification' | 'command_result' | 'subscription_confirmed' | 'error' | 'pong' | 'camera_frame'
  content?: string
  action?: string
  params?: Record<string, any>
  active?: boolean
  image_base64?: string
  lamp_id?: string
  server_time?: string
  message?: string
  data?: any
  success?: boolean
  result?: any
  error?: string
  timestamp?: string
  channels?: string[]
  code?: string
  conversation_state?: 'idle' | 'listening' | 'thinking' | 'speaking'
  // camera_frame 消息专用字段
  frame_b64?: string
  width?: number
  height?: number
  detections?: EdgeDetections
}

// 摄像头帧消息
export interface CameraFrameMessage {
  type: 'camera_frame'
  frame_b64: string
  width?: number
  height?: number
  timestamp: number
  detections?: EdgeDetections
}

// 边缘视觉检测结果
export interface EdgeDetections {
  faces?: Array<{
    x: number        // 归一化坐标 0-1
    y: number
    w: number        // 归一化宽高 0-1
    h: number
    confidence: number
  }>
  hands?: Array<{
    landmarks?: Array<{ x: number; y: number; z: number }>  // 21 个关键点
    gesture?: string
    handedness?: string
    confidence?: number
  }>
  presence?: boolean
  mode?: string
}
