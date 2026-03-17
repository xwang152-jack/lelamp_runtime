import { defineStore } from 'pinia'
import type { Room } from 'livekit-client'
import type { ConnectionState, ConnectionStatus } from '@/types'

export const useConnectionStore = defineStore('connection', {
  state: (): ConnectionState => ({
    isConnected: false,
    room: null,
    serverUrl: '',
    token: '',
    connectionStatus: 'disconnected'
  }),

  actions: {
    setConnectionStatus(status: ConnectionStatus) {
      this.connectionStatus = status
      this.isConnected = status === 'connected'
    },

    setRoom(room: Room | null) {
      this.room = room
    },

    setCredentials(url: string, token: string) {
      this.serverUrl = url
      this.token = token
    },

    disconnect() {
      this.room = null
      this.isConnected = false
      this.connectionStatus = 'disconnected'
      this.serverUrl = ''
      this.token = ''
    }
  }
})
