export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'error'

export interface ConnectionState {
  isConnected: boolean
  ws: WebSocket | null
  serverUrl: string
  token: string
  connectionStatus: ConnectionStatus
}

export interface DataMessage {
  type: 'chat' | 'command' | 'camera_status' | 'vision_result' | 'connected' | 'state_update' | 'event' | 'log' | 'notification'
  content?: string
  action?: string
  params?: Record<string, any>
  active?: boolean
  image_base64?: string
  lamp_id?: string
  server_time?: string
  message?: string
  data?: any
}
