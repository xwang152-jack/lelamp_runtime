import { defineStore } from 'pinia'
import type { DeviceState, LightState } from '@/types'

export const useDeviceStore = defineStore('device', {
  state: (): DeviceState => ({
    name: 'LeLamp',
    status: 'offline',
    cameraActive: false,
    lights: {
      color: { r: 255, g: 244, b: 229 },
      effect: null
    }
  }),

  actions: {
    setCameraActive(active: boolean) {
      this.cameraActive = active
    },

    setLightColor(r: number, g: number, b: number) {
      this.lights.color = { r, g, b }
      this.lights.effect = null
    },

    setLightEffect(effect: string) {
      this.lights.effect = effect
    },

    setStatus(status: 'online' | 'offline') {
      this.status = status
    }
  }
})
