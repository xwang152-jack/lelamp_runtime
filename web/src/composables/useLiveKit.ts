import { Room, RoomEvent } from 'livekit-client'
import { useConnectionStore, useChatStore, useDeviceStore } from '@/stores'
import type { DataMessage } from '@/types'

export function useLiveKit() {
  const connectionStore = useConnectionStore()
  const chatStore = useChatStore()
  const deviceStore = useDeviceStore()

  async function connect(url: string, token: string) {
    try {
      connectionStore.setConnectionStatus('connecting')
      connectionStore.setCredentials(url, token)

      const room = new Room({
        adaptiveStream: true,
        dynacast: true,
        audioCaptureDefaults: { autoGainControl: true },
        videoCaptureDefaults: { facingMode: 'environment' }
      })

      room.on(RoomEvent.Disconnected, () => {
        connectionStore.setConnectionStatus('disconnected')
        deviceStore.setStatus('offline')
      })

      ;(room.on(RoomEvent.TrackSubscribed, (track: any) => {
        console.log('Track subscribed:', track)
      }),
        room.on(RoomEvent.TrackUnsubscribed, (track: any) => {
          console.log('Track unsubscribed:', track)
        }),
        room.on(RoomEvent.DataReceived, (payload: Uint8Array) => {
          handleDataReceived(payload)
        }))

      await room.connect(url, token)
      connectionStore.setRoom(room)
      connectionStore.setConnectionStatus('connected')
      deviceStore.setStatus('online')
    } catch (error) {
      console.error('Connection failed:', error)
      connectionStore.setConnectionStatus('error')
      throw error
    }
  }

  function handleDataReceived(payload: Uint8Array) {
    try {
      const decoder = new TextDecoder()
      const data: DataMessage = JSON.parse(decoder.decode(payload))

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
      }
    } catch (error) {
      console.error('Failed to parse data:', error)
    }
  }

  async function sendCommand(action: string, params: Record<string, any>) {
    if (!connectionStore.room) {
      console.warn('Room not connected')
      return
    }

    const data: DataMessage = { type: 'command', action, params }
    const encoder = new TextEncoder()
    await connectionStore.room.localParticipant.publishData(encoder.encode(JSON.stringify(data)), {
      reliable: true
    })
  }

  async function sendChat(text: string) {
    if (!connectionStore.room) {
      console.warn('Room not connected')
      return
    }

    const data: DataMessage = { type: 'chat', content: text }
    const encoder = new TextEncoder()
    await connectionStore.room.localParticipant.publishData(encoder.encode(JSON.stringify(data)), {
      reliable: true
    })
    chatStore.addMessage('user', text)
  }

  async function disconnect() {
    if (connectionStore.room) {
      await connectionStore.room.disconnect()
    }
    connectionStore.disconnect()
    deviceStore.setStatus('offline')
  }

  return {
    connect,
    disconnect,
    sendCommand,
    sendChat
  }
}
