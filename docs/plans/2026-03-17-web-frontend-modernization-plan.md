# Web Frontend Modernization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 使用 Vue 3 + TypeScript + Element Plus 重构 LeLamp Web 客户端，提升开发体验和用户体验，新增设置、健康监控、历史记录和多设备管理功能。

**Architecture:** 渐进式重构 - 新建 `web/` 目录与 `web_client/` 并存，使用 pnpm workspace 管理，LiveKit SDK 通过 composable 模式集成。

**Tech Stack:** Vue 3.4+, TypeScript 5.0+, Vite 5.0+, Pinia 2.1+, Element Plus 2.4+, livekit-client 2.0+, SCSS 1.7+, pnpm 8.0+

---

## Task 1: 初始化 pnpm Workspace

**Files:**
- Create: `pnpm-workspace.yaml`
- Create: `package.json`
- Create: `.npmrc`

**Step 1: 创建 pnpm workspace 配置**

```yaml
# pnpm-workspace.yaml
packages:
  - 'web'
```

**Step 2: 创建根 package.json**

```json
{
  "name": "lelamp-runtime-workspace",
  "private": true,
  "scripts": {
    "dev": "pnpm --filter web dev",
    "build": "pnpm --filter web build",
    "preview": "pnpm --filter web preview",
    "lint": "pnpm --filter web lint",
    "test": "pnpm --filter web test"
  }
}
```

**Step 3: 创建 .npmrc 确保 pnpm 版本**

```
shamefully-hoist=true
strict-peer-dependencies=false
```

**Step 4: 初始化 pnpm**

Run: `pnpm install`
Expected: 创建 node_modules 和 pnpm-lock.yaml

**Step 5: 提交**

```bash
git add pnpm-workspace.yaml package.json .npmrc pnpm-lock.yaml
git commit -m "chore: initialize pnpm workspace

- Add workspace configuration
- Add root package.json with convenience scripts
- Configure pnpm settings

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: 初始化 Vue 3 项目

**Files:**
- Create: `web/package.json`
- Create: `web/vite.config.ts`
- Create: `web/tsconfig.json`
- Create: `web/tsconfig.node.json`
- Create: `web/index.html`
- Create: `web/src/main.ts`
- Create: `web/src/App.vue`
- Create: `web/src/vite-env.d.ts`

**Step 1: 创建 web/package.json**

```json
{
  "name": "lelamp-web",
  "version": "3.0.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vue-tsc && vite build",
    "preview": "vite preview",
    "lint": "eslint . --ext .vue,.js,.jsx,.cjs,.mjs,.ts,.tsx,.cts,.mts --fix",
    "format": "prettier --write src/"
  },
  "dependencies": {
    "vue": "^3.4.21",
    "vue-router": "^4.3.0",
    "pinia": "^2.1.7",
    "element-plus": "^2.6.3",
    "@element-plus/icons-vue": "^2.3.1",
    "livekit-client": "^2.4.1",
    "axios": "^1.6.8",
    "echarts": "^5.5.0"
  },
  "devDependencies": {
    "@vitejs/plugin-vue": "^5.0.4",
    "@vue/tsconfig": "^0.5.1",
    "typescript": "^5.4.5",
    "vite": "^5.2.11",
    "vue-tsc": "^2.0.21",
    "sass": "^1.77.8",
    "unplugin-vue-components": "^0.26.0",
    "unplugin-auto-import": "^0.17.6",
    "@types/node": "^20.12.11",
    "eslint": "^8.57.0",
    "@typescript-eslint/eslint-plugin": "^7.16.0",
    "eslint-plugin-vue": "^9.26.0",
    "prettier": "^3.2.5"
  }
}
```

**Step 2: 创建 web/vite.config.ts**

```typescript
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { resolve } from 'path'
import AutoImport from 'unplugin-auto-import/vite'
import Components from 'unplugin-vue-components/vite'
import { ElementPlusResolver } from 'unplugin-vue-components/resolvers'

export default defineConfig({
  plugins: [
    vue(),
    AutoImport({
      resolvers: [ElementPlusResolver()],
      imports: ['vue', 'vue-router', 'pinia'],
      dts: 'src/auto-imports.d.ts',
    }),
    Components({
      resolvers: [ElementPlusResolver()],
      dts: 'src/components.d.ts',
    }),
  ],
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
    },
  },
  server: {
    port: 5173,
    open: false,
  },
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: false,
    rollupOptions: {
      output: {
        manualChunks: {
          'livekit': ['livekit-client'],
          'element-plus': ['element-plus'],
          'vue-vendor': ['vue', 'vue-router', 'pinia'],
          'echarts': ['echarts'],
        },
      },
    },
  },
})
```

**Step 3: 创建 web/tsconfig.json**

```json
{
  "extends": "@vue/tsconfig/tsconfig.dom.json",
  "compilerOptions": {
    "target": "ES2020",
    "module": "ESNext",
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "jsx": "preserve",
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"]
    },
    "types": ["vite/client"],
    "skipLibCheck": true
  },
  "include": ["src/**/*.ts", "src/**/*.d.ts", "src/**/*.tsx", "src/**/*.vue"],
  "exclude": ["dist", "node_modules"]
}
```

**Step 4: 创建 web/tsconfig.node.json**

```json
{
  "compilerOptions": {
    "composite": true,
    "skipLibCheck": true,
    "module": "ESNext",
    "moduleResolution": "bundler",
    "allowSyntheticDefaultImports": true
  },
  "include": ["vite.config.ts"]
}
```

**Step 5: 创建 web/index.html**

```html
<!DOCTYPE html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>LeLamp Web Client</title>
  </head>
  <body>
    <div id="app"></div>
    <script type="module" src="/src/main.ts"></script>
  </body>
</html>
```

**Step 6: 创建 web/src/main.ts**

```typescript
import { createApp } from 'vue'
import { createPinia } from 'pinia'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import zhCn from 'element-plus/dist/locale/zh-cn.mjs'
import App from './App.vue'
import router from './router'

const app = createApp(App)

app.use(createPinia())
app.use(router)
app.use(ElementPlus, { locale: zhCn })

app.mount('#app')
```

**Step 7: 创建 web/src/App.vue**

```vue
<template>
  <router-view />
</template>

<script setup lang="ts">
// 主应用组件
</script>

<style>
#app {
  width: 100%;
  height: 100vh;
  margin: 0;
  padding: 0;
}
</style>
```

**Step 8: 创建 web/src/vite-env.d.ts**

```typescript
/// <reference types="vite/client" />
```

**Step 9: 安装依赖并验证**

Run: `pnpm install`
Expected: 成功安装所有依赖

**Step 10: 提交**

```bash
git add web/
git commit -m "feat: initialize Vue 3 + TypeScript project

- Add Vite + Vue 3 + TypeScript configuration
- Configure Element Plus with auto-import
- Set up path aliases and build options
- Create basic app entry point

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: 创建路由配置

**Files:**
- Create: `web/src/router/index.ts`
- Create: `web/src/views/ConnectView.vue`
- Create: `web/src/views/LayoutView.vue`

**Step 1: 创建路由配置**

```typescript
// web/src/router/index.ts
import { createRouter, createWebHistory, type RouteRecordRaw } from 'vue-router'

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    component: () => import('@/views/LayoutView.vue'),
    redirect: '/connect',
    children: [
      {
        path: 'connect',
        name: 'connect',
        component: () => import('@/views/ConnectView.vue'),
        meta: { title: '连接设备' }
      },
      {
        path: 'room',
        name: 'room',
        component: () => import('@/views/RoomView.vue'),
        meta: { requiresAuth: true }
      },
      {
        path: 'settings',
        name: 'settings',
        component: () => import('@/views/SettingsView.vue'),
        meta: { title: '设置' }
      },
      {
        path: 'health',
        name: 'health',
        component: () => import('@/views/HealthView.vue'),
        meta: { title: '设备健康' }
      },
      {
        path: 'history',
        name: 'history',
        component: () => import('@/views/HistoryView.vue'),
        meta: { title: '历史记录' }
      },
      {
        path: 'devices',
        name: 'devices',
        component: () => import('@/views/DevicesView.vue'),
        meta: { title: '设备管理' }
      }
    ]
  }
]

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes
})

export default router
```

**Step 2: 创建布局视图**

```vue
<!-- web/src/views/LayoutView.vue -->
<template>
  <el-container class="layout-container">
    <el-header v-if="showHeader">
      <AppHeader @toggle-sidebar="drawerVisible = true" />
    </el-header>

    <el-container>
      <el-aside v-if="!isConnectPage" width="200px">
        <AppSidebar :collapsed="!showHeader" />
      </el-aside>

      <el-main>
        <router-view />
      </el-main>
    </el-container>

    <el-drawer v-model="drawerVisible" :with-header="false" size="280px">
      <AppSidebar @close="drawerVisible = false" />
    </el-drawer>
  </el-container>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useRoute } from 'vue-router'
import AppHeader from '@/components/common/AppHeader.vue'
import AppSidebar from '@/components/common/AppSidebar.vue'

const route = useRoute()
const drawerVisible = ref(false)

const isConnectPage = computed(() => route.name === 'connect')
const showHeader = computed(() => !isConnectPage.value)
</script>

<style scoped>
.layout-container {
  height: 100vh;
}

.el-header {
  background-color: var(--el-bg-color);
  border-bottom: 1px solid var(--el-border-color);
  padding: 0;
}

.el-aside {
  background-color: var(--el-bg-color-page);
  border-right: 1px solid var(--el-border-color);
}

.el-main {
  padding: 0;
  background-color: var(--el-bg-color-page);
  overflow: hidden;
}
</style>
```

**Step 3: 创建连接页面占位**

```vue
<!-- web/src/views/ConnectView.vue -->
<template>
  <div class="connect-view">
    <h1>连接设备</h1>
    <p>连接功能开发中...</p>
  </div>
</template>

<script setup lang="ts">
// 连接页面
</script>

<style scoped>
.connect-view {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
}
</style>
```

**Step 4: 提交**

```bash
git add web/src/router/ web/src/views/
git commit -m "feat: add router configuration and layout

- Create vue-router with route structure
- Add LayoutView with header and sidebar
- Add ConnectView placeholder
- Configure route meta for titles and auth

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: 创建 Pinia Stores

**Files:**
- Create: `web/src/stores/connection.ts`
- Create: `web/src/stores/device.ts`
- Create: `web/src/stores/chat.ts`
- Create: `web/src/stores/settings.ts`
- Create: `web/src/stores/history.ts`

**Step 1: 创建 connection store**

```typescript
// web/src/stores/connection.ts
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { Room } from 'livekit-client-sdk'

export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'error'

export const useConnectionStore = defineStore('connection', () => {
  const isConnected = ref(false)
  const room = ref<Room | null>(null)
  const serverUrl = ref('')
  const token = ref('')
  const status = ref<ConnectionStatus>('disconnected')
  const errorMessage = ref('')

  const isConnecting = computed(() => status.value === 'connecting')
  const canConnect = computed(() => status.value === 'disconnected' || status.value === 'error')

  function setServerUrl(url: string) {
    serverUrl.value = url
  }

  function setToken(newToken: string) {
    token.value = newToken
  }

  function setRoom(newRoom: Room | null) {
    room.value = newRoom
    isConnected.value = newRoom !== null
  }

  function setStatus(newStatus: ConnectionStatus) {
    status.value = newStatus
  }

  function setError(message: string) {
    errorMessage.value = message
    status.value = 'error'
  }

  function disconnect() {
    room.value?.disconnect()
    room.value = null
    isConnected.value = false
    status.value = 'disconnected'
  }

  return {
    isConnected,
    room,
    serverUrl,
    token,
    status,
    errorMessage,
    isConnecting,
    canConnect,
    setServerUrl,
    setToken,
    setRoom,
    setStatus,
    setError,
    disconnect
  }
})
```

**Step 2: 创建 device store**

```typescript
// web/src/stores/device.ts
import { defineStore } from 'pinia'
import { ref } from 'vue'

export interface MotorPosition {
  base_yaw: number
  base_pitch: number
  elbow_pitch: number
  wrist_roll: number
  wrist_pitch: number
}

export interface MotorHealth {
  temperature?: number
  voltage?: number
  load?: number
  status: 'healthy' | 'warning' | 'critical' | 'stalled'
}

export interface DeviceState {
  name: string
  status: 'online' | 'offline' | 'error'
  cameraActive: boolean
  motors: {
    positions: MotorPosition
    health: Record<string, MotorHealth>
  }
  lights: {
    color: { r: number; g: number; b: number }
    effect: string | null
  }
}

export const useDeviceStore = defineStore('device', () => {
  const state = ref<DeviceState>({
    name: 'LeLamp',
    status: 'offline',
    cameraActive: false,
    motors: {
      positions: {
        base_yaw: 0,
        base_pitch: 0,
        elbow_pitch: 0,
        wrist_roll: 0,
        wrist_pitch: 0
      },
      health: {}
    },
    lights: {
      color: { r: 255, g: 244, b: 229 },
      effect: null
    }
  })

  function setMotorPositions(positions: Partial<MotorPosition>) {
    Object.assign(state.value.motors.positions, positions)
  }

  function setMotorHealth(motor: string, health: MotorHealth) {
    if (!state.value.motors.health[motor]) {
      state.value.motors.health[motor] = {}
    }
    Object.assign(state.value.motors.health[motor], health)
  }

  function setCameraActive(active: boolean) {
    state.value.cameraActive = active
  }

  function setLightColor(r: number, g: number, b: number) {
    state.value.lights.color = { r, g, b }
  }

  function setLightEffect(effect: string | null) {
    state.value.lights.effect = effect
  }

  function setStatus(status: 'online' | 'offline' | 'error') {
    state.value.status = status
  }

  return {
    state,
    setMotorPositions,
    setMotorHealth,
    setCameraActive,
    setLightColor,
    setLightEffect,
    setStatus
  }
})
```

**Step 3: 创建 chat store**

```typescript
// web/src/stores/chat.ts
import { defineStore } from 'pinia'
import { ref } from 'vue'

export interface Message {
  id: string
  sender: 'user' | 'agent'
  content: string
  timestamp: number
}

export const useChatStore = defineStore('chat', () => {
  const messages = ref<Message[]>([])
  const isProcessing = ref(false)

  function addMessage(sender: 'user' | 'agent', content: string) {
    const message: Message = {
      id: Date.now().toString(),
      sender,
      content,
      timestamp: Date.now()
    }
    messages.value.push(message)
  }

  function clearMessages() {
    messages.value = []
  }

  function setProcessing(value: boolean) {
    isProcessing.value = value
  }

  return {
    messages,
    isProcessing,
    addMessage,
    clearMessages,
    setProcessing
  }
})
```

**Step 4: 创建 settings store**

```typescript
// web/src/stores/settings.ts
import { defineStore } from 'pinia'
import { ref } from 'vue'

export type Theme = 'light' | 'dark' | 'auto'
export type Language = 'zh' | 'en'

export interface ServerConfig {
  id: string
  name: string
  url: string
  token: string
}

export const useSettingsStore = defineStore('settings', () => {
  const theme = ref<Theme>('light')
  const language = ref<Language>('zh')
  const autoReconnect = ref(true)
  const notifications = ref(true)
  const savedServers = ref<ServerConfig[]>([])

  function setTheme(newTheme: Theme) {
    theme.value = newTheme
    applyTheme(newTheme)
  }

  function setLanguage(newLanguage: Language) {
    language.value = newLanguage
  }

  function addServer(config: Omit<ServerConfig, 'id'>) {
    const server: ServerConfig = {
      ...config,
      id: Date.now().toString()
    }
    savedServers.value.push(server)
    saveToStorage()
  }

  function removeServer(id: string) {
    const index = savedServers.value.findIndex(s => s.id === id)
    if (index !== -1) {
      savedServers.value.splice(index, 1)
      saveToStorage()
    }
  }

  function saveToStorage() {
    localStorage.setItem('lelamp_settings', JSON.stringify({
      theme: theme.value,
      language: language.value,
      autoReconnect: autoReconnect.value,
      notifications: notifications.value,
      savedServers: savedServers.value
    }))
  }

  function loadFromStorage() {
    const saved = localStorage.getItem('lelamp_settings')
    if (saved) {
      const data = JSON.parse(saved)
      theme.value = data.theme || 'light'
      language.value = data.language || 'zh'
      autoReconnect.value = data.autoReconnect ?? true
      notifications.value = data.notifications ?? true
      savedServers.value = data.savedServers || []
      applyTheme(theme.value)
    }
  }

  return {
    theme,
    language,
    autoReconnect,
    notifications,
    savedServers,
    setTheme,
    setLanguage,
    addServer,
    removeServer,
    saveToStorage,
    loadFromStorage
  }
})

function applyTheme(theme: Theme) {
  if (theme === 'dark') {
    document.documentElement.classList.add('dark')
  } else if (theme === 'light') {
    document.documentElement.classList.remove('dark')
  } else {
    // auto
    if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }
}
```

**Step 5: 创建 history store**

```typescript
// web/src/stores/history.ts
import { defineStore } from 'pinia'
import { ref } from 'vue'

export interface Conversation {
  id: string
  startTime: number
  endTime?: number
  messages: Array<{
    sender: 'user' | 'agent'
    content: string
    timestamp: number
  }>
}

export interface Operation {
  id: string
  timestamp: number
  type: 'command' | 'animation' | 'light'
  action: string
  params: Record<string, any>
  success: boolean
}

export const useHistoryStore = defineStore('history', () => {
  const conversations = ref<Conversation[]>([])
  const operations = ref<Operation[]>([])
  const currentConversation = ref<Conversation | null>(null)

  function startConversation() {
    currentConversation.value = {
      id: Date.now().toString(),
      startTime: Date.now(),
      messages: []
    }
  }

  function endConversation() {
    if (currentConversation.value) {
      currentConversation.value.endTime = Date.now()
      conversations.value.unshift({ ...currentConversation.value })
      currentConversation.value = null
      saveToStorage()
    }
  }

  function addConversationMessage(sender: 'user' | 'agent', content: string) {
    if (currentConversation.value) {
      currentConversation.value.messages.push({
        sender,
        content,
        timestamp: Date.now()
      })
    }
  }

  function addOperation(type: Operation['type'], action: string, params: Record<string, any>, success: boolean) {
    const operation: Operation = {
      id: Date.now().toString(),
      timestamp: Date.now(),
      type,
      action,
      params,
      success
    }
    operations.value.unshift(operation)
    saveToStorage()
  }

  function saveToStorage() {
    const data = {
      conversations: conversations.value.slice(0, 100), // 保留最近100条
      operations: operations.value.slice(0, 500) // 保留最近500条
    }
    localStorage.setItem('lelamp_history', JSON.stringify(data))
  }

  function loadFromStorage() {
    const saved = localStorage.getItem('lelamp_history')
    if (saved) {
      const data = JSON.parse(saved)
      conversations.value = data.conversations || []
      operations.value = data.operations || []
    }
  }

  function clearHistory() {
    conversations.value = []
    operations.value = []
    localStorage.removeItem('lelamp_history')
  }

  return {
    conversations,
    operations,
    currentConversation,
    startConversation,
    endConversation,
    addConversationMessage,
    addOperation,
    saveToStorage,
    loadFromStorage,
    clearHistory
  }
})
```

**Step 6: 提交**

```bash
git add web/src/stores/
git commit -m "feat: create Pinia stores for state management

- Add connection store for LiveKit room management
- Add device store for motor/light/camera state
- Add chat store for message handling
- Add settings store with persistence
- Add history store for conversations and operations

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: 创建 LiveKit Composable

**Files:**
- Create: `web/src/composables/useLiveKit.ts`
- Create: `web/src/types/livekit.ts`

**Step 1: 创建 LiveKit 类型定义**

```typescript
// web/src/types/livekit.ts
import type { Room } from 'livekit-client-sdk'

export interface LiveKitMessage {
  type: 'chat' | 'command' | 'vision_result' | 'camera_status' | 'health_update'
  content?: string
  image_base64?: string
  action?: string
  params?: Record<string, any>
  active?: boolean
  data?: Record<string, any>
}
```

**Step 2: 创建 useLiveKit composable**

```typescript
// web/src/composables/useLiveKit.ts
import { ref } from 'vue'
import type { Room, Track } from 'livekit-client-sdk'
import { RoomEvent, DataPacket_Kind } from 'livekit-client-sdk'
import { useConnectionStore } from '@/stores/connection'
import { useDeviceStore } from '@/stores/device'
import { useChatStore } from '@/stores/chat'
import { useHistoryStore } from '@/stores/history'
import type { LiveKitMessage } from '@/types/livekit'

export function useLiveKit() {
  const connectionStore = useConnectionStore()
  const deviceStore = useDeviceStore()
  const chatStore = useChatStore()
  const historyStore = useHistoryStore()

  const videoTracks = ref<Map<string, Track>>(new Map())

  async function connect(url: string, token: string) {
    connectionStore.setStatus('connecting')

    try {
      const room = new Room({
        adaptiveStream: true,
        dynacast: true,
        audioCaptureDefaults: {
          autoGainControl: true,
          echoCancellation: true,
          noiseSuppression: true,
        },
        videoCaptureDefaults: {
          facingMode: 'environment',
        },
      })

      // 监听事件
      room.on(RoomEvent.TrackSubscribed, handleTrackSubscribed)
      room.on(RoomEvent.TrackUnsubscribed, handleTrackUnsubscribed)
      room.on(RoomEvent.DataReceived, handleDataReceived)
      room.on(RoomEvent.Disconnected, handleDisconnected)
      room.on(RoomEvent.ConnectionStateChanged, handleConnectionStateChange)

      await room.connect(url, token)

      connectionStore.setRoom(room)
      connectionStore.setStatus('connected')
      deviceStore.setStatus('online')
      historyStore.startConversation()

      // 发布本地轨道（麦克风）
      try {
        await room.localParticipant.setMicrophoneEnabled(true)
      } catch (e) {
        console.warn('Could not enable microphone:', e)
      }

      return room
    } catch (error) {
      connectionStore.setError(`连接失败: ${error instanceof Error ? error.message : String(error)}`)
      throw error
    }
  }

  function handleTrackSubscribed(track: Track, publication: any, participant: any) {
    console.log('Track subscribed:', track.kind, track.sid)

    if (track.kind === 'video') {
      videoTracks.value.set(track.sid, track)
      deviceStore.setCameraActive(true)
    }
  }

  function handleTrackUnsubscribed(track: Track) {
    console.log('Track unsubscribed:', track.kind)
    videoTracks.value.delete(track.sid)

    if (track.kind === 'video' && videoTracks.value.size === 0) {
      deviceStore.setCameraActive(false)
    }
  }

  function handleDataReceived(payload: Uint8Array, participant: any) {
    try {
      const text = new TextDecoder().decode(payload)
      const data: LiveKitMessage = JSON.parse(text)

      switch (data.type) {
        case 'chat':
          chatStore.addMessage('agent', data.content || '')
          historyStore.addConversationMessage('agent', data.content || '')
          break

        case 'vision_result':
          // 视觉结果 - 通过事件通知
          deviceStore.setCameraActive(false)
          break

        case 'camera_status':
          deviceStore.setCameraActive(data.active || false)
          break

        case 'health_update':
          if (data.data) {
            Object.entries(data.data).forEach(([motor, health]: [string, any]) => {
              deviceStore.setMotorHealth(motor, health)
            })
          }
          break
      }
    } catch (e) {
      console.error('Failed to parse data message:', e)
    }
  }

  function handleDisconnected() {
    connectionStore.disconnect()
    deviceStore.setStatus('offline')
    deviceStore.setCameraActive(false)
    videoTracks.value.clear()
    historyStore.endConversation()
  }

  function handleConnectionStateChange(state: any) {
    console.log('Connection state changed:', state)
  }

  async function sendChat(text: string) {
    if (!connectionStore.room) return

    const message: LiveKitMessage = {
      type: 'chat',
      content: text
    }

    const encoder = new TextEncoder()
    const data = encoder.encode(JSON.stringify(message))

    try {
      await connectionStore.room.localParticipant.publishData(data, DataPacket_Kind.RELIABLE)
      chatStore.addMessage('user', text)
      historyStore.addConversationMessage('user', text)
    } catch (error) {
      console.error('Failed to send chat:', error)
    }
  }

  async function sendCommand(action: string, params: Record<string, any> = {}) {
    if (!connectionStore.room) return

    const message: LiveKitMessage = {
      type: 'command',
      action,
      params
    }

    const encoder = new TextEncoder()
    const data = encoder.encode(JSON.stringify(message))

    try {
      await connectionStore.room.localParticipant.publishData(data, DataPacket_Kind.RELIABLE)

      // 记录操作历史
      let type: 'command' | 'animation' | 'light' = 'command'
      if (action.startsWith('rgb_effect_') || action === 'set_rgb_solid') {
        type = 'light'
      } else if (action === 'play_recording') {
        type = 'animation'
      }
      historyStore.addOperation(type, action, params, true)
    } catch (error) {
      console.error('Failed to send command:', error)
      historyStore.addOperation('command', action, params, false)
    }
  }

  return {
    connect,
    videoTracks,
    sendChat,
    sendCommand
  }
}
```

**Step 3: 提交**

```bash
git add web/src/composables/ web/src/types/
git commit -m "feat: add LiveKit composable for WebRTC integration

- Create useLiveKit composable with connect/send functions
- Handle track subscription and data channel messages
- Integrate with connection, device, chat, and history stores
- Support chat, command, vision, camera_status, and health messages

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: 创建连接页面

**Files:**
- Modify: `web/src/views/ConnectView.vue`

**Step 1: 实现连接页面**

```vue
<!-- web/src/views/ConnectView.vue -->
<template>
  <div class="connect-view">
    <el-card class="connect-card">
      <template #header>
        <div class="card-header">
          <h1>🪔 LeLamp Web Client</h1>
          <p>智能台灯，陪伴成长</p>
        </div>
      </template>

      <el-form :model="form" label-position="top" @submit.prevent="handleConnect">
        <el-form-item label="选择服务器">
          <el-select
            v-model="selectedServerId"
            placeholder="选择已保存的服务器或输入新配置"
            style="width: 100%"
            clearable
            @change="onServerChange"
          >
            <el-option
              v-for="server in settingsStore.savedServers"
              :key="server.id"
              :label="server.name"
              :value="server.id"
            />
          </el-select>
        </el-form-item>

        <el-form-item label="LiveKit Server URL">
          <el-input
            v-model="form.serverUrl"
            placeholder="wss://your-project.livekit.cloud"
            clearable
          >
            <template #prepend>
              <el-icon><Link /></el-icon>
            </template>
          </el-input>
        </el-form-item>

        <el-form-item label="Access Token">
          <el-input
            v-model="form.token"
            type="textarea"
            :rows="3"
            placeholder="粘贴生成的 Token..."
            clearable
          />
          <template #append>
            <el-button @click="showTokenHelp = true">?</el-button>
          </template>
        </el-form-item>

        <el-form-item>
          <el-checkbox v-model="form.remember">记住配置</el-checkbox>
        </el-form-item>

        <el-form-item>
          <el-button
            type="primary"
            size="large"
            style="width: 100%"
            :loading="connectionStore.isConnecting"
            @click="handleConnect"
          >
            <el-icon class="btn-icon"><Connection /></el-icon>
            {{ connectionStore.isConnecting ? '连接中...' : '连接设备' }}
          </el-button>
        </el-form-item>

        <el-alert
          v-if="connectionStore.errorMessage"
          :title="connectionStore.errorMessage"
          type="error"
          :closable="false"
          show-icon
        />
      </el-form>

      <el-divider>或创建新服务器配置</el-divider>

      <el-button text @click="showAddServer = true">
        <el-icon><Plus /></el-icon>
        添加新服务器
      </el-button>
    </el-card>

    <!-- Token 帮助对话框 -->
    <el-dialog v-model="showTokenHelp" title="获取 Access Token" width="500px">
      <p>运行以下命令生成 Token：</p>
      <el-code>python3 scripts/generate_client_token.py</el-code>
      <p class="hint">Token 有效期 24 小时</p>
    </el-dialog>

    <!-- 添加服务器对话框 -->
    <el-dialog v-model="showAddServer" title="添加服务器配置" width="500px">
      <el-form :model="newServer" label-position="top">
        <el-form-item label="配置名称">
          <el-input v-model="newServer.name" placeholder="如：本地开发环境" />
        </el-form-item>
        <el-form-item label="Server URL">
          <el-input v-model="newServer.url" placeholder="wss://..." />
        </el-form-item>
        <el-form-item label="Token">
          <el-input v-model="newServer.token" type="textarea" :rows="3" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showAddServer = false">取消</el-button>
        <el-button type="primary" @click="addServer">确定</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { reactive, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { Link, Connection, Plus } from '@element-plus/icons-vue'
import { useConnectionStore } from '@/stores/connection'
import { useSettingsStore } from '@/stores/settings'

const router = useRouter()
const connectionStore = useConnectionStore()
const settingsStore = useSettingsStore()

const form = reactive({
  serverUrl: '',
  token: '',
  remember: false
})

const selectedServerId = ref('')
const showTokenHelp = ref(false)
const showAddServer = ref(false)

const newServer = reactive({
  name: '',
  url: '',
  token: ''
})

// 从 settings 加载选中的服务器
function onServerChange(serverId: string) {
  const server = settingsStore.savedServers.find(s => s.id === serverId)
  if (server) {
    form.serverUrl = server.url
    form.token = server.token
  }
}

async function handleConnect() {
  if (!form.serverUrl || !form.token) {
    ElMessage.warning('请输入 Server URL 和 Token')
    return
  }

  // 记住配置
  if (form.remember) {
    connectionStore.setServerUrl(form.serverUrl)
    connectionStore.setToken(form.token)
  }

  try {
    const { connect } = await import('@/composables/useLiveKit')
    const useLiveKit = (await import('@/composables/useLiveKit')).useLiveKit

    await connect(form.serverUrl, form.token)
    router.push('/room')
  } catch (error) {
    // 错误已在 store 中处理
  }
}

function addServer() {
  if (!newServer.name || !newServer.url || !newServer.token) {
    ElMessage.warning('请填写完整信息')
    return
  }

  settingsStore.addServer({
    name: newServer.name,
    url: newServer.url,
    token: newServer.token
  })

  selectedServerId.value = settingsStore.savedServers[settingsStore.savedServers.length - 1].id
  onServerChange(selectedServerId.value)

  showAddServer.value = false
  ElMessage.success('服务器配置已保存')
}

// 组件挂载时加载 settings
settingsStore.loadFromStorage()
</script>

<style scoped>
.connect-view {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 20px;
}

.connect-card {
  width: 100%;
  max-width: 500px;
}

.card-header {
  text-align: center;
}

.card-header h1 {
  margin: 0;
  font-size: 28px;
  color: var(--el-text-color-primary);
}

.card-header p {
  margin: 5px 0 0;
  color: var(--el-text-color-secondary);
}

.btn-icon {
  margin-right: 8px;
}

.hint {
  margin-top: 10px;
  color: var(--el-text-color-secondary);
  font-size: 12px;
}
</style>
```

**Step 2: 提交**

```bash
git add web/src/views/ConnectView.vue
git commit -m "feat: implement connect page with server management

- Add server URL and token input
- Support saved server configurations
- Add help dialog for token generation
- Implement connection with useLiveKit composable
- Add remember configuration option

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: 创建通用组件

**Files:**
- Create: `web/src/components/common/AppHeader.vue`
- Create: `web/src/components/common/AppSidebar.vue`
- Create: `web/src/components/common/StatusIndicator.vue`
- Create: `web/src/components/common/ToastNotification.vue`

**Step 1: 创建 AppHeader 组件**

```vue
<!-- web/src/components/common/AppHeader.vue -->
<template>
  <header class="app-header">
    <div class="header-left">
      <el-icon class="logo-icon" @click="router.push('/room')"><Monitor /></el-icon>
      <span class="app-title">LeLamp</span>
    </div>

    <div class="header-right">
      <StatusIndicator />
      <el-button
        v-if="connectionStore.isConnected"
        type="danger"
        text
        @click="handleDisconnect"
      >
        断开连接
      </el-button>
    </div>
  </header>
</template>

<script setup lang="ts">
import { useRouter } from 'vue-router'
import { Monitor } from '@element-plus/icons-vue'
import { useConnectionStore } from '@/stores/connection'
import StatusIndicator from './StatusIndicator.vue'
import { ElMessageBox } from 'element-plus'

const router = useRouter()
const connectionStore = useConnectionStore()

async function handleDisconnect() {
  try {
    await ElMessageBox.confirm('确定要断开连接吗？', '确认', {
      type: 'warning'
    })
    connectionStore.disconnect()
    router.push('/connect')
  } catch {
    // 用户取消
  }
}
</script>

<style scoped>
.app-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 20px;
  height: 60px;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 12px;
  cursor: pointer;
}

.logo-icon {
  font-size: 24px;
  color: var(--el-color-primary);
}

.app-title {
  font-size: 18px;
  font-weight: 600;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 16px;
}
</style>
```

**Step 2: 创建 AppSidebar 组件**

```vue
<!-- web/src/components/common/AppSidebar.vue -->
<template>
  <el-menu
    :default-active="activeMenu"
    class="sidebar-menu"
    @select="handleSelect"
  >
    <div class="sidebar-header">
      <span class="logo-text">LeLamp</span>
    </div>

    <el-menu-item index="/room">
      <el-icon><Monitor /></el-icon>
      <span>控制台</span>
    </el-menu-item>

    <el-menu-item index="/settings">
      <el-icon><Setting /></el-icon>
      <span>设置</span>
    </el-menu-item>

    <el-menu-item index="/health">
      <el-icon><TrendCharts /></el-icon>
      <span>设备健康</span>
    </el-menu-item>

    <el-menu-item index="/history">
      <el-icon><Clock /></el-icon>
      <span>历史记录</span>
    </el-menu-item>

    <el-menu-item index="/devices">
      <el-icon><Files /></el-icon>
      <span>设备管理</span>
    </el-menu-item>
  </el-menu>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { Monitor, Setting, TrendCharts, Clock, Files } from '@element-plus/icons-vue'

const router = useRouter()
const route = useRoute()

const activeMenu = computed(() => route.path)

function handleSelect(index: string) {
  router.push(index)
}
</script>

<style scoped>
.sidebar-menu {
  height: 100%;
  border-right: none;
}

.sidebar-header {
  padding: 20px;
  text-align: center;
  border-bottom: 1px solid var(--el-border-color);
}

.logo-text {
  font-size: 20px;
  font-weight: 600;
  color: var(--el-color-primary);
}

.el-menu-item {
  padding: 0 20px;
}
</style>
```

**Step 3: 创建 StatusIndicator 组件**

```vue
<!-- web/src/components/common/StatusIndicator.vue -->
<template>
  <div class="status-indicator">
    <div
      class="status-dot"
      :class="statusClass"
    />
    <span class="status-text">{{ statusText }}</span>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useConnectionStore } from '@/stores/connection'

const connectionStore = useConnectionStore()

const statusClass = computed(() => {
  switch (connectionStore.status) {
    case 'connected': return 'connected'
    case 'connecting': return 'connecting'
    case 'error': return 'error'
    default: return 'disconnected'
  }
})

const statusText = computed(() => {
  switch (connectionStore.status) {
    case 'connected': return '已连接'
    case 'connecting': return '连接中...'
    case 'error': return '连接错误'
    default: return '未连接'
  }
})
</script>

<style scoped>
.status-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
}

.status-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  transition: all 0.3s;
}

.status-dot.disconnected {
  background-color: var(--el-color-info);
}

.status-dot.connecting {
  background-color: var(--el-color-warning);
  animation: pulse 1s infinite;
}

.status-dot.connected {
  background-color: var(--el-color-success);
}

.status-dot.error {
  background-color: var(--el-color-danger);
}

@keyframes pulse {
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.5;
  }
}

.status-text {
  font-size: 12px;
  color: var(--el-text-color-secondary);
}
</style>
```

**Step 4: 创建 ToastNotification 组件**

```vue
<!-- web/src/components/common/ToastNotification.vue -->
<template>
  <teleport to="body">
    <transition-group name="toast" tag="div" class="toast-container">
      <div
        v-for="toast in toasts"
        :key="toast.id"
        class="toast"
        :class="`toast-${toast.type}`"
      >
        <span class="toast-icon">{{ toastIcons[toast.type] }}</span>
        <span class="toast-message">{{ toast.message }}</span>
      </div>
    </transition-group>
  </teleport>
</template>

<script setup lang="ts">
import { ref, h } from 'vue'
import { ElNotification } from 'element-plus'

export interface Toast {
  id: string
  type: 'success' | 'warning' | 'error' | 'info'
  message: string
  duration?: number
}

const toasts = ref<Toast[]>([])

const toastIcons = {
  success: '✅',
  warning: '⚠️',
  error: '❌',
  info: 'ℹ️'
}

function showToast(options: Omit<Toast, 'id'>) {
  const toast: Toast = {
    id: Date.now().toString(),
    duration: 3000,
    ...options
  }

  toasts.value.push(toast)

  setTimeout(() => {
    removeToast(toast.id)
  }, toast.duration)
}

function removeToast(id: string) {
  const index = toasts.value.findIndex(t => t.id === id)
  if (index !== -1) {
    toasts.value.splice(index, 1)
  }
}

// 导出为全局函数
declare global {
  showToast: typeof showToast
}

window.showToast = showToast

defineExpose({
  showToast
})
</script>

<style scoped>
.toast-container {
  position: fixed;
  top: 80px;
  right: 20px;
  z-index: 9999;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.toast {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 12px 16px;
  border-radius: 8px;
  background: white;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  min-width: 200px;
}

.toast-success {
  border-left: 4px solid var(--el-color-success);
}

.toast-warning {
  border-left: 4px solid var(--el-color-warning);
}

.toast-error {
  border-left: 4px solid var(--el-color-danger);
}

.toast-info {
  border-left: 4px solid var(--el-color-info);
}

.toast-enter-active,
.toast-leave-active {
  transition: all 0.3s ease;
}

.toast-enter-from {
  opacity: 0;
  transform: translateX(30px);
}

.toast-leave-to {
  opacity: 0;
  transform: translateX(30px);
}
</style>
```

**Step 5: 提交**

```bash
git add web/src/components/common/
git commit -m "feat: add common UI components

- Add AppHeader with logo and disconnect button
- Add AppSidebar with navigation menu
- Add StatusIndicator with connection state
- Add ToastNotification for global notifications

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: 创建主控制台页面 (RoomView)

**Files:**
- Create: `web/src/views/RoomView.vue`

**Step 1: 实现 RoomView 基础布局**

```vue
<!-- web/src/views/RoomView.vue -->
<template>
  <div class="room-view">
    <!-- 视频区域 -->
    <div class="video-section">
      <div class="video-container" ref="videoContainer">
        <div v-if="!hasVideo" class="video-placeholder">
          <el-icon class="placeholder-icon"><VideoCamera /></el-icon>
          <p>等待摄像头画面...</p>
        </div>
      </div>

      <!-- 隐私指示器 -->
      <PrivacyIndicator :active="deviceStore.state.cameraActive" />
    </div>

    <!-- 控制面板 -->
    <div class="control-section">
      <el-tabs v-model="activeTab" class="control-tabs">
        <el-tab-pane label="⚡ 快捷操作" name="quick">
          <QuickActions @send-chat="sendChat" />
        </el-tab-pane>

        <el-tab-pane label="📸 视觉助手" name="vision">
          <VisionPanel @send-command="sendCommand" />
        </el-tab-pane>

        <el-tab-pane label="🎭 动作表情" name="animation">
          <AnimationPanel @send-command="sendCommand" />
        </el-tab-pane>

        <el-tab-pane label="💡 灯光魔法" name="light">
          <LightPanel @set-color="setRgbColor" @set-effect="setRgbEffect" />
        </el-tab-pane>
      </el-tabs>

      <!-- 聊天框 -->
      <ChatBox
        :messages="chatStore.messages"
        :is-processing="chatStore.isProcessing"
        @send="sendChat"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { VideoCamera } from '@element-plus/icons-vue'
import { useConnectionStore } from '@/stores/connection'
import { useDeviceStore } from '@/stores/device'
import { useChatStore } from '@/stores/chat'
import { useLiveKit } from '@/composables/useLiveKit'
import PrivacyIndicator from '@/components/common/PrivacyIndicator.vue'
import QuickActions from '@/components/room/QuickActions.vue'
import VisionPanel from '@/components/room/VisionPanel.vue'
import AnimationPanel from '@/components/room/AnimationPanel.vue'
import LightPanel from '@/components/room/LightPanel.vue'
import ChatBox from '@/components/room/ChatBox.vue'

const connectionStore = useConnectionStore()
const deviceStore = useDeviceStore()
const chatStore = useChatStore()
const { videoTracks, sendChat, sendCommand } = useLiveKit()

const activeTab = ref('quick')
const videoContainer = ref<HTMLElement>()

const hasVideo = computed(() => videoTracks.value.size > 0)

async function setRgbColor(r: number, g: number, b: number) {
  await sendCommand('set_rgb_solid', { r, g, b })
}

async function setRgbEffect(effect: string) {
  await sendCommand(`rgb_effect_${effect}`, {})
}
</script>

<style scoped>
.room-view {
  display: flex;
  gap: 20px;
  padding: 20px;
  height: calc(100vh - 60px);
}

.video-section {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 10px;
  min-width: 0;
}

.video-container {
  flex: 1;
  background: #000;
  border-radius: 12px;
  overflow: hidden;
  position: relative;
}

.video-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: #666;
}

.placeholder-icon {
  font-size: 48px;
  margin-bottom: 10px;
}

.control-section {
  width: 400px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.control-tabs {
  flex: 1;
  display: flex;
  flex-direction: column;
}

.control-tabs :deep(.el-tabs__content) {
  flex: 1;
  overflow-y: auto;
}

.control-tabs :deep(.el-tab-pane) {
  height: 100%;
}
</style>
```

**Step 2: 提交**

```bash
git add web/src/views/RoomView.vue
git commit -m "feat: implement room view with video and controls

- Add video player container with LiveKit tracks
- Add tab-based control panel
- Integrate QuickActions, VisionPanel, AnimationPanel, LightPanel, ChatBox
- Add PrivacyIndicator for camera status

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: 创建房间组件

**Files:**
- Create: `web/src/components/room/QuickActions.vue`
- Create: `web/src/components/room/VisionPanel.vue`
- Create: `web/src/components/room/AnimationPanel.vue`
- Create: `web/src/components/room/LightPanel.vue`
- Create: `web/src/components/room/ChatBox.vue`
- Create: `web/src/components/room/PrivacyIndicator.vue`

**Step 1: 创建 QuickActions 组件**

```vue
<!-- web/src/components/room/QuickActions.vue -->
<template>
  <div class="quick-actions">
    <p class="panel-desc">快速执行常用操作</p>
    <div class="action-grid">
      <el-button
        v-for="action in actions"
        :key="action.key"
        size="large"
        @click="handleAction(action)"
      >
        <span class="action-icon">{{ action.icon }}</span>
        <span>{{ action.label }}</span>
      </el-button>
    </div>
  </div>
</template>

<script setup lang="ts">
interface Action {
  key: string
  icon: string
  label: string
  message: string
}

const actions: Action[] = [
  { key: 'hello', icon: '👋', label: '打招呼', message: '你好' },
  { key: 'time', icon: '⏰', label: '查看时间', message: '现在几点了' },
  { key: 'joke', icon: '😄', label: '讲笑话', message: '讲个笑话' },
  { key: 'sing', icon: '🎵', label: '唱歌', message: '唱首歌' }
]

const emit = defineEmits<{
  sendChat: [message: string]
}>()

function handleAction(action: Action) {
  emit('sendChat', action.message)
}
</script>

<style scoped>
.quick-actions {
  padding: 16px;
}

.panel-desc {
  margin: 0 0 16px;
  color: var(--el-text-color-secondary);
}

.action-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 12px;
}

.el-button {
  height: 80px;
  flex-direction: column;
  gap: 8px;
}

.action-icon {
  font-size: 24px;
}
</style>
```

**Step 2: 创建 VisionPanel 组件**

```vue
<!-- web/src/components/room/VisionPanel.vue -->
<template>
  <div class="vision-panel">
    <p class="panel-desc">使用摄像头识别物体、检查作业</p>

    <div class="vision-buttons">
      <el-button
        size="large"
        type="primary"
        @click="handleCapture"
      >
        <el-icon><Camera /></el-icon>
        拍照识别
      </el-button>

      <el-button
        size="large"
        type="success"
        @click="handleHomework"
      >
        <el-icon><Notebook /></el-icon>
        检查作业
      </el-button>

      <el-button
        size="large"
        @click="handleFeishu"
      >
        <el-icon><Promotion /></el-icon>
        推送飞书
      </el-button>
    </div>

    <!-- 结果展示 -->
    <el-dialog v-model="showResult" title="识别结果" width="500px">
      <div v-if="resultImage" class="result-image">
        <img :src="`data:image/jpeg;base64,${resultImage}`" alt="拍摄画面" />
      </div>
      <p class="result-text">{{ resultText }}</p>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { Camera, Notebook, Promotion } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'

const emit = defineEmits<{
  sendCommand: [action: string, params: Record<string, any>]
  sendChat: [message: string]
}>()

const showResult = ref(false)
const resultImage = ref('')
const resultText = ref('')

async function handleCapture() {
  ElMessage.info('正在拍照识别...')
  await emit('sendChat', '这是什么')
}

async function handleHomework() {
  ElMessage.info('正在检查作业，请稍候...')
  await emit('sendChat', '帮我检查作业')
}

async function handleFeishu() {
  ElMessage.info('正在推送到飞书...')
  await emit('sendCommand', 'capture_to_feishu', {})
}

// 监听视觉结果
// 在实际使用中，这应该从 store 获取
</script>

<style scoped>
.vision-panel {
  padding: 16px;
}

.panel-desc {
  margin: 0 0 16px;
  color: var(--el-text-color-secondary);
}

.vision-buttons {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.result-image img {
  width: 100%;
  border-radius: 8px;
  margin-bottom: 16px;
}
</style>
```

**Step 3: 创建 AnimationPanel 组件**

```vue
<!-- web/src/components/room/AnimationPanel.vue -->
<template>
  <div class="animation-panel">
    <p class="panel-desc">让台灯做出有趣的动作</p>

    <div class="animation-grid">
      <el-button
        v-for="anim in animations"
        :key="anim.key"
        size="large"
        @click="handleAnimation(anim)"
      >
        <span class="anim-icon">{{ anim.icon }}</span>
        <span>{{ anim.label }}</span>
      </el-button>
    </div>
  </div>
</template>

<script setup lang="ts">
interface Animation {
  key: string
  icon: string
  label: string
}

const animations: Animation[] = [
  { key: 'nod', icon: '👍', label: '点头' },
  { key: 'shake', icon: '👎', label: '摇头' },
  { key: 'excited', icon: '🎉', label: '兴奋' },
  { key: 'sleep', icon: '😴', label: '睡觉' },
  { key: 'dance', icon: '💃', label: '跳舞' },
  { key: 'think', icon: '🤔', label: '思考' }
]

const emit = defineEmits<{
  sendCommand: [action: string, params: Record<string, any>]
}>()

function handleAnimation(anim: Animation) {
  emit('sendCommand', 'play_recording', { recording_name: anim.key })
}
</script>

<style scoped>
.animation-panel {
  padding: 16px;
}

.panel-desc {
  margin: 0 0 16px;
  color: var(--el-text-color-secondary);
}

.animation-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 12px;
}

.el-button {
  height: 80px;
  flex-direction: column;
  gap: 4px;
}

.anim-icon {
  font-size: 24px;
}
</style>
```

**Step 4: 创建 LightPanel 组件**

```vue
<!-- web/src/components/room/LightPanel.vue -->
<template>
  <div class="light-panel">
    <p class="panel-desc">调整灯光颜色和效果</p>

    <!-- 颜色选择器 -->
    <div class="color-section">
      <h4>快速颜色</h4>
      <div class="color-swatches">
        <div
          v-for="color in presetColors"
          :key="color.name"
          class="color-swatch"
          :style="{ background: color.hex }"
          :title="color.name"
          @click="setColor(color.r, color.g, color.b)"
        />
      </div>
    </div>

    <!-- 效果 -->
    <div class="effect-section">
      <h4>特效动画</h4>
      <div class="effect-grid">
        <el-button
          v-for="effect in effects"
          :key="effect.key"
          @click="setEffect(effect.key)"
        >
          {{ effect.label }}
        </el-button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
const emit = defineEmits<{
  sendCommand: [action: string, params: Record<string, any>]
  setColor: [r: number, g: number, b: number]
  setEffect: [effect: string]
}>()

const presetColors = [
  { name: '暖红', hex: '#FF6B6B', r: 255, g: 107, b: 107 },
  { name: '粉红', hex: '#FFB6C1', r: 255, g: 182, b: 193 },
  { name: '橙色', hex: '#FFA500', r: 255, g: 165, b: 0 },
  { name: '金黄', hex: '#FFD700', r: 255, g: 215, b: 0 },
  { name: '浅绿', hex: '#90EE90', r: 144, g: 238, b: 144 },
  { name: '天蓝', hex: '#87CEEB', r: 135, g: 206, b: 235 },
  { name: '紫色', hex: '#9370DB', r: 147, g: 112, b: 219 },
  { name: '暖白', hex: 'RGB(255, 244, 229)', r: 255, g: 244, b: 229 }
]

const effects = [
  { key: 'breathing', label: '💗 呼吸灯' },
  { key: 'rainbow', label: '🌈 彩虹' },
  { key: 'wave', label: '🌊 波浪' },
  { key: 'fire', label: '🔥 火焰' },
  { key: 'fireworks', label: '🎆 烟花' },
  { key: 'starry', label: '⭐ 星空' }
]

function setColor(r: number, g: number, b: number) {
  emit('setColor', r, g, b)
}

function setEffect(effect: string) {
  emit('setEffect', effect)
}
</script>

<style scoped>
.light-panel {
  padding: 16px;
}

.panel-desc {
  margin: 0 0 16px;
  color: var(--el-text-color-secondary);
}

.color-section h4 {
  margin: 0 0 12px;
  font-size: 14px;
}

.color-swatches {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-bottom: 20px;
}

.color-swatch {
  width: 40px;
  height: 40px;
  border-radius: 8px;
  cursor: pointer;
  border: 2px solid transparent;
  transition: all 0.2s;
}

.color-swatch:hover {
  transform: scale(1.1);
  border-color: var(--el-color-primary);
}

.effect-section h4 {
  margin: 0 0 12px;
  font-size: 14px;
}

.effect-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 8px;
}
</style>
```

**Step 5: 创建 ChatBox 组件**

```vue
<!-- web/src/components/room/ChatBox.vue -->
<template>
  <div class="chat-box">
    <div class="chat-header">
      <h3>💬 实时对话</h3>
      <el-button
        size="small"
        text
        @click="handleClear"
      >
        清空
      </el-button>
    </div>

    <div class="chat-messages" ref="messagesContainer">
      <div
        v-for="msg in messages"
        :key="msg.id"
        :class="['message', msg.sender]"
      >
        {{ msg.content }}
      </div>
    </div>

    <div class="chat-input">
      <el-input
        v-model="inputText"
        placeholder="输入消息..."
        @keyup.enter="handleSend"
      />
      <el-button
        type="primary"
        :disabled="!inputText.trim()"
        @click="handleSend"
      >
        发送
      </el-button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, nextTick } from 'vue'
import type { Message } from '@/stores/chat'

interface Props {
  messages: Message[]
  isProcessing: boolean
}

const props = defineProps<Props>()

const emit = defineEmits<{
  send: [message: string]
}>()

const inputText = ref('')
const messagesContainer = ref<HTMLElement>()

function handleSend() {
  const text = inputText.value.trim()
  if (text) {
    emit('send', text)
    inputText.value = ''
  }
}

function handleClear() {
  // 通过 emit 通知父组件清空
  messages.value = []
}

watch(() => props.messages.length, () => {
  nextTick(() => {
    if (messagesContainer.value) {
      messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
    }
  })
})
</script>

<style scoped>
.chat-box {
  display: flex;
  flex-direction: column;
  height: 300px;
}

.chat-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px;
  border-bottom: 1px solid var(--el-border-color);
}

.chat-header h3 {
  margin: 0;
  font-size: 16px;
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 12px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.message {
  padding: 8px 12px;
  border-radius: 12px;
  max-width: 80%;
  word-break: break-all;
}

.message.user {
  align-self: flex-end;
  background: var(--el-color-primary);
  color: white;
}

.message.agent {
  align-self: flex-start;
  background: var(--el-fill-color-light);
}

.chat-input {
  display: flex;
  gap: 8px;
  padding: 12px;
  border-top: 1px solid var(--el-border-color);
}

.chat-input .el-input {
  flex: 1;
}
</style>
```

**Step 6: 创建 PrivacyIndicator 组件**

```vue
<!-- web/src/components/room/PrivacyIndicator.vue -->
<template>
  <div class="privacy-indicator">
    <span class="privacy-icon">🔒</span>
    <span class="privacy-text">
      {{ active ? '摄像头已激活' : '摄像头已关闭' }}
    </span>
    <span
      class="privacy-led"
      :class="{ active }"
    />
  </div>
</template>

<script setup lang="ts">
interface Props {
  active: boolean
}

defineProps<Props>()
</script>

<style scoped>
.privacy-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  background: var(--el-fill-color-blank);
  border-radius: 6px;
}

.privacy-led {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: #ccc;
  transition: all 0.3s;
}

.privacy-led.active {
  background: #f56c6c;
  animation: blink 1.5s ease-in-out infinite;
}

@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.3; }
}
</style>
```

**Step 7: 提交**

```bash
git add web/src/components/room/
git commit -m "feat: add room control panel components

- Add QuickActions with preset action buttons
- Add VisionPanel with capture, homework, and Feishu features
- Add AnimationPanel with motor animation controls
- Add LightPanel with color picker and effect buttons
- Add ChatBox with message display and input
- Add PrivacyIndicator for camera status display

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## 任务进度概览

以上是 Phase 3 前端现代化的主要任务。全部任务数量约 30+ 个，由于篇幅限制，以上展示了核心任务。

**完成顺序**:
1. 初始化 pnpm Workspace
2. 初始化 Vue 3 项目
3. 创建路由配置
4. 创建 Pinia Stores
5. 创建 LiveKit Composable
6. 创建连接页面
7. 创建通用组件
8. 创建主控制台页面
9. 创建房间组件

**后续任务** (简要):
- Task 10: 创建设置页面
- Task 11: 创建设备健康页面
- Task 12: 创建历史记录页面
- Task 13: 创建设备管理页面
- Task 14: 添加国际化支持
- Task 15: 添加主题切换功能
- Task 16: 添加图表组件（健康数据可视化）
- Task 17: 配置构建和部署
- Task 18: 集成测试
- Task 19: 性能优化
- Task 20: 文档更新

**预计总工作量**: 约 15-20 小时

---

**下一步**: 选择执行方式（Subagent-Driven 或 Parallel Session）
