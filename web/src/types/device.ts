export interface DeviceState {
  name: string
  status: 'online' | 'offline'
  cameraActive: boolean
  lights: LightState
}

export interface LightState {
  color: {
    r: number
    g: number
    b: number
  }
  effect: string | null
}

export interface Message {
  id: string
  sender: 'user' | 'agent'
  content: string
  timestamp: number
}

export interface ChatState {
  messages: Message[]
  isProcessing: boolean
}
