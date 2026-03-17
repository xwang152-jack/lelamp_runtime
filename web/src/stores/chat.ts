import { defineStore } from 'pinia'
import type { ChatState, Message } from '@/types'

export const useChatStore = defineStore('chat', {
  state: (): ChatState => ({
    messages: [],
    isProcessing: false
  }),

  actions: {
    addMessage(sender: 'user' | 'agent', content: string) {
      const message: Message = {
        id: `msg-${Date.now()}-${Math.random()}`,
        sender,
        content,
        timestamp: Date.now()
      }
      this.messages.push(message)
    },

    clearMessages() {
      this.messages = []
    },

    setProcessing(processing: boolean) {
      this.isProcessing = processing
    }
  }
})
