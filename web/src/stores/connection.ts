import { defineStore } from 'pinia'
import type { ConnectionState, ConnectionStatus } from '@/types'

export const useConnectionStore = defineStore('connection', {
  state: (): ConnectionState => ({
    isConnected: false,
    ws: null,
    serverUrl: '',
    token: '',
    connectionStatus: 'disconnected'
  }),

  actions: {
    setConnectionStatus(status: ConnectionStatus) {
      this.connectionStatus = status
      this.isConnected = status === 'connected'
    },

    setWebSocket(ws: WebSocket | null) {
      this.ws = ws
    },

    setCredentials(url: string, token: string) {
      this.serverUrl = url
      this.token = token
    },

    disconnect() {
      if (this.ws) {
        this.ws.close()
        this.ws = null
      }
      this.isConnected = false
      this.connectionStatus = 'disconnected'
      this.serverUrl = ''
      this.token = ''
    }
  }
})
