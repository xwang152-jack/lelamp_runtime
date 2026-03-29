import { useConnectionStore, useChatStore, useDeviceStore, useAuthStore } from '@/stores'
import type { DataMessage } from '@/types'
import { ElMessage } from 'element-plus'
import { WS_CONNECTION_TIMEOUT } from '@/utils/auth-constants'

// 摄像头帧回调函数类型
type Detections = {
  faces?: { center?: [number, number]; confidence?: number }[]
  hands?: { x?: number; y?: number; gesture?: string; handedness?: string; confidence?: number }[]
  presence?: boolean
  mode?: string
}

type CameraFrameCallback = (
  frameB64: string,
  info: { width?: number; height?: number; timestamp: number },
  detections?: Detections
) => void

// 全局摄像头帧回调
let cameraFrameCallback: CameraFrameCallback | null = null

export function useWebSocket() {
  const connectionStore = useConnectionStore()
  const chatStore = useChatStore()
  const deviceStore = useDeviceStore()
  const authStore = useAuthStore()

  // 注册摄像头帧回调
  function onCameraFrame(callback: CameraFrameCallback) {
    cameraFrameCallback = callback
  }

  // 取消注册摄像头帧回调
  function offCameraFrame() {
    cameraFrameCallback = null
  }

  async function connect(url: string, lampId: string = 'lelamp') {
    try {
      connectionStore.setConnectionStatus('connecting')
      connectionStore.setCredentials(url, '')

      const baseUrl = url.endsWith('/') ? url.slice(0, -1) : url

      // 构建 WebSocket URL，如果有 token 则添加到 query 参数
      let wsUrl = baseUrl.endsWith('/api')
        ? `${baseUrl.replace(/^http/, 'ws')}/ws/${lampId}`
        : `${baseUrl.replace(/^http/, 'ws')}/api/ws/${lampId}`

      // 添加 token 参数（如果已登录）
      if (authStore.accessToken) {
        wsUrl += `?token=${encodeURIComponent(authStore.accessToken)}`
      }

      const ws = new WebSocket(wsUrl)

      ws.onopen = () => {
        connectionStore.setWebSocket(ws)
        connectionStore.setConnectionStatus('connected')
        deviceStore.setStatus('online')
        // 设置连接标记，让路由守卫知道用户已连接
        sessionStorage.setItem('lelamp_connected', 'true')
        sessionStorage.setItem('lelamp_server_url', url)

        // 连接成功后，默认认为摄像头可用（API 模式下没有隐私保护）
        // 如果后端推送了状态，这里会被覆盖
        deviceStore.setCameraActive(true)
      }

      ws.onclose = () => {
        connectionStore.setConnectionStatus('disconnected')
        deviceStore.setStatus('offline')
        connectionStore.setWebSocket(null)
        // 清除连接标记
        sessionStorage.removeItem('lelamp_connected')
      }

      ws.onerror = (error) => {
        console.error('WebSocket error:', error)
        connectionStore.setConnectionStatus('error')
      }

      ws.onmessage = (event) => {
        handleDataReceived(event.data)
      }

      // Return a promise that resolves when connected
      return new Promise<void>((resolve, reject) => {
        const timeout = setTimeout(() => {
          reject(new Error('Connection timeout'))
        }, WS_CONNECTION_TIMEOUT)

        const checkConnection = () => {
          if (ws.readyState === WebSocket.OPEN) {
            clearTimeout(timeout)
            resolve()
          } else if (ws.readyState === WebSocket.CLOSED || ws.readyState === WebSocket.CLOSING) {
            clearTimeout(timeout)
            reject(new Error('Connection failed'))
          } else {
            setTimeout(checkConnection, 100)
          }
        }
        checkConnection()
      })
    } catch (error) {
      console.error('Connection setup failed:', error)
      connectionStore.setConnectionStatus('error')
      throw error
    }
  }

  function handleDataReceived(payload: string) {
    try {
      const data: DataMessage = JSON.parse(payload)

      switch (data.type) {
        case 'chat':
          if (data.content) {
            chatStore.addMessage('agent', data.content)
          }
          break
        case 'camera_status':
          if (data.active !== undefined) {
            deviceStore.setCameraActive(data.active)
          }
          break
        case 'camera_frame':
          // 处理摄像头帧消息
          if (data.frame_b64 && cameraFrameCallback) {
            const frameInfo = {
              width: data.width,
              height: data.height,
              timestamp: data.timestamp ? Date.parse(data.timestamp) : Date.now()
            }
            cameraFrameCallback(data.frame_b64, frameInfo, data.detections)
          }
          break
        case 'state_update':
          // Handle conversation state updates
          if (data.conversation_state) {
            deviceStore.setConversationState(data.conversation_state)
          }
          break
        case 'connected':
          console.log('Backend connection confirmed:', data)
          break
        case 'command_result':
          // 处理命令执行结果
          console.log('Command result:', data)
          if (data.success) {
            ElMessage.success({
              message: `命令执行成功: ${data.action}`,
              duration: 2000
            })
          } else {
            ElMessage.error({
              message: `命令执行失败: ${data.error || '未知错误'}`,
              duration: 3000
            })
          }
          break
      }
    } catch (error) {
      console.error('Failed to parse data:', error)
    }
  }

  async function sendCommand(action: string, params: Record<string, any> = {}) {
    if (!connectionStore.ws || connectionStore.ws.readyState !== WebSocket.OPEN) {
      console.warn('WebSocket not connected, cannot send command:', action, params)
      ElMessage.warning('设备未连接，请先连接设备')
      return false
    }

    const data = { type: 'command', action, params }
    console.log('Sending command:', data)
    connectionStore.ws.send(JSON.stringify(data))
    return true
  }

  async function sendChat(text: string) {
    if (!connectionStore.ws || connectionStore.ws.readyState !== WebSocket.OPEN) {
      console.warn('WebSocket not connected')
      return
    }

    const data = { type: 'command', action: 'chat', params: { text } }
    connectionStore.ws.send(JSON.stringify(data))
    chatStore.addMessage('user', text)
  }

  async function disconnect() {
    connectionStore.disconnect()
    deviceStore.setStatus('offline')
  }

  return {
    connect,
    disconnect,
    sendCommand,
    sendChat,
    onCameraFrame,
    offCameraFrame
  }
}
