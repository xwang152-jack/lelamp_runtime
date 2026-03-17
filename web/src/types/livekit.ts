import type { Room as LiveKitRoom } from 'livekit-client'

export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'error'

export interface ConnectionState {
  isConnected: boolean
  room: LiveKitRoom | null
  serverUrl: string
  token: string
  connectionStatus: ConnectionStatus
}

export interface DataMessage {
  type: 'chat' | 'command' | 'camera_status' | 'vision_result'
  content?: string
  action?: string
  params?: Record<string, any>
  active?: boolean
  image_base64?: string
}
