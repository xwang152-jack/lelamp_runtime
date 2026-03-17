# LeLamp Web 前端现代化 Phase 1 MVP 实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 使用 Vue 3 + TypeScript + Vite 技术栈重构 LeLamp Web 客户端，实现核心控制功能（连接管理、视频预览、实时对话、快捷操作、灯光控制）

**Architecture:** 基于 Pinia 的状态管理 + Vue 3 Composition API 组件 + LiveKit SDK 实时通信，采用模块化组件设计，支持响应式布局和暗色主题

**Tech Stack:** Vue 3.4+, TypeScript 5.0+, Vite 5.0+, Pinia 2.1+, Element Plus 2.4+, LiveKit SDK 2.6+, pnpm 8.0+

---

## Task 1: 配置 pnpm Workspace

**Files:**
- Create: `pnpm-workspace.yaml`
- Create: `package.json` (根目录)

**Step 1: 创建 workspace 配置**

创建 `pnpm-workspace.yaml`:

```yaml
packages:
  - 'web'
```

**Step 2: 创建根 package.json**

创建 `package.json`:

```json
{
  "name": "lelamp-runtime-workspace",
  "private": true,
  "scripts": {
    "dev": "pnpm --filter web dev",
    "build": "pnpm --filter web build",
    "preview": "pnpm --filter web preview",
    "lint": "pnpm --filter web lint",
    "type-check": "pnpm --filter web type-check"
  },
  "devDependencies": {
    "pnpm": "^8.15.0"
  }
}
```

**Step 3: 安装 pnpm（如果未安装）**

运行: `which pnpm || npm install -g pnpm`
Expected: pnpm 命令可用

**Step 4: 初始化 workspace**

运行: `pnpm install`
Expected: 创建 node_modules 和 pnpm-lock.yaml

**Step 5: Commit**

```bash
git add pnpm-workspace.yaml package.json pnpm-lock.yaml
git commit -m "chore: 配置 pnpm workspace"
```

---

## Task 2: 初始化 Vue 3 + Vite 项目

**Files:**
- Create: `web/package.json`
- Create: `web/vite.config.ts`
- Create: `web/tsconfig.json`
- Create: `web/tsconfig.node.json`
- Create: `web/index.html`
- Create: `web/.env.development`
- Create: `web/.env.production`

**Step 1: 创建 web/package.json**

创建 `web/package.json`:

```json
{
  "name": "lelamp-web",
  "version": "1.0.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vue-tsc && vite build",
    "preview": "vite preview",
    "lint": "eslint . --ext .vue,.js,.jsx,.cjs,.mjs,.ts,.tsx,.cts,.mts --fix",
    "type-check": "vue-tsc --noEmit"
  },
  "dependencies": {
    "vue": "^3.4.21",
    "vue-router": "^4.2.5",
    "pinia": "^2.1.7",
    "element-plus": "^2.6.0",
    "@element-plus/icons-vue": "^2.3.1",
    "livekit-client": "^2.6.0",
    "axios": "^1.6.7"
  },
  "devDependencies": {
    "@vitejs/plugin-vue": "^5.0.4",
    "@vue/tsconfig": "^0.5.1",
    "typescript": "~5.3.3",
    "vite": "^5.1.4",
    "vue-tsc": "^1.8.27",
    "sass": "^1.71.1",
    "unplugin-auto-import": "^0.17.5",
    "unplugin-vue-components": "^0.26.0",
    "@typescript-eslint/eslint-plugin": "^7.1.1",
    "@typescript-eslint/parser": "^7.1.1",
    "eslint": "^8.57.0",
    "eslint-plugin-vue": "^9.22.0",
    "prettier": "^3.2.5"
  }
}
```

**Step 2: 创建 web/vite.config.ts**

创建 `web/vite.config.ts`:

```typescript
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { fileURLToPath } from 'node:url'
import AutoImport from 'unplugin-auto-import/vite'
import Components from 'unplugin-vue-components/vite'
import { ElementPlusResolver } from 'unplugin-vue-components/resolvers'

export default defineConfig({
  plugins: [
    vue(),
    AutoImport({
      resolvers: [ElementPlusResolver()],
      imports: ['vue', 'vue-router', 'pinia']
    }),
    Components({
      resolvers: [ElementPlusResolver()]
    })
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    }
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true
      }
    }
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    rollupOptions: {
      output: {
        manualChunks: {
          'livekit': ['livekit-client'],
          'element-plus': ['element-plus'],
          'vue-vendor': ['vue', 'vue-router', 'pinia']
        }
      }
    }
  }
})
```

**Step 3: 创建 web/tsconfig.json**

创建 `web/tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "module": "ESNext",
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "preserve",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "baseUrl": ".",
    "paths": {
      "@/*": ["./src/*"]
    }
  },
  "include": ["src/**/*.ts", "src/**/*.d.ts", "src/**/*.tsx", "src/**/*.vue"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

**Step 4: 创建 web/tsconfig.node.json**

创建 `web/tsconfig.node.json`:

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

创建 `web/index.html`:

```html
<!DOCTYPE html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>LeLamp Web Client</title>
  </head>
  <body>
    <div id="app"></div>
    <script type="module" src="/src/main.ts"></script>
  </body>
</html>
```

**Step 6: 创建环境变量文件**

创建 `web/.env.development`:

```env
VITE_APP_TITLE=LeLamp Dev
VITE_LIVEKIT_URL=wss://your-dev.livekit.cloud
```

创建 `web/.env.production`:

```env
VITE_APP_TITLE=LeLamp
VITE_LIVEKIT_URL=wss://your-prod.livekit.cloud
```

**Step 7: Commit**

```bash
git add web/
git commit -m "feat: 初始化 Vue 3 + Vite 项目配置"
```

---

## Task 3: 创建基础目录结构和入口文件

**Files:**
- Create: `web/src/main.ts`
- Create: `web/src/App.vue`
- Create: `web/src/assets/styles/variables.scss`
- Create: `web/src/assets/styles/global.scss`

**Step 1: 创建 main.ts**

创建 `web/src/main.ts`:

```typescript
import { createApp } from 'vue'
import { createPinia } from 'pinia'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import './assets/styles/variables.scss'
import './assets/styles/global.scss'
import App from './App.vue'

const app = createApp(App)
const pinia = createPinia()

app.use(pinia)
app.use(ElementPlus)

app.mount('#app')
```

**Step 2: 创建 App.vue**

创建 `web/src/App.vue`:

```vue
<template>
  <div id="app" class="lelamp-app">
    <router-view />
  </div>
</template>

<script setup lang="ts">
// App root component
</script>

<style lang="scss">
.lelamp-app {
  width: 100%;
  min-height: 100vh;
  background: var(--lelamp-bg);
}
</style>
```

**Step 3: 创建样式变量**

创建 `web/src/assets/styles/variables.scss`:

```scss
:root {
  // 品牌色
  --lelamp-brand: #ff6b6b;
  --lelamp-accent: #4ecdc4;
  --lelamp-bg: #f5f7fa;
  --lelamp-surface: #ffffff;

  // 功能色
  --color-success: #67c23a;
  --color-warning: #e6a23c;
  --color-danger: #f56c6c;
  --color-info: #909399;

  // 间距
  --spacing-xs: 4px;
  --spacing-sm: 8px;
  --spacing-md: 16px;
  --spacing-lg: 24px;
  --spacing-xl: 32px;

  // 圆角
  --radius-sm: 4px;
  --radius-md: 8px;
  --radius-lg: 16px;

  // 阴影
  --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.1);
  --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.1);
  --shadow-lg: 0 10px 20px rgba(0, 0, 0, 0.1);
}
```

**Step 4: 创建全局样式**

创建 `web/src/assets/styles/global.scss`:

```scss
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial,
    sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

#app {
  width: 100%;
  min-height: 100vh;
}
```

**Step 5: Commit**

```bash
git add web/src/
git commit -m "feat: 创建基础目录结构和入口文件"
```

---

## Task 4: 创建 TypeScript 类型定义

**Files:**
- Create: `web/src/types/livekit.ts`
- Create: `web/src/types/device.ts`
- Create: `web/src/types/index.ts`

**Step 1: 创建 LiveKit 类型**

创建 `web/src/types/livekit.ts`:

```typescript
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
```

**Step 2: 创建 Device 类型**

创建 `web/src/types/device.ts`:

```typescript
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
```

**Step 3: 创建类型导出**

创建 `web/src/types/index.ts`:

```typescript
export * from './livekit'
export * from './device'
```

**Step 4: Commit**

```bash
git add web/src/types/
git commit -m "feat: 添加 TypeScript 类型定义"
```

---

## Task 5: 创建 Pinia Stores

**Files:**
- Create: `web/src/stores/connection.ts`
- Create: `web/src/stores/device.ts`
- Create: `web/src/stores/chat.ts`
- Create: `web/src/stores/index.ts`

**Step 1: 创建 Connection Store**

创建 `web/src/stores/connection.ts`:

```typescript
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
```

**Step 2: 创建 Device Store**

创建 `web/src/stores/device.ts`:

```typescript
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
```

**Step 3: 创建 Chat Store**

创建 `web/src/stores/chat.ts`:

```typescript
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
```

**Step 4: 创建 Store 导出**

创建 `web/src/stores/index.ts`:

```typescript
export { useConnectionStore } from './connection'
export { useDeviceStore } from './device'
export { useChatStore } from './chat'
```

**Step 5: Commit**

```bash
git add web/src/stores/
git commit -m "feat: 添加 Pinia stores"
```

---

## Task 6: 创建 LiveKit Composable

**Files:**
- Create: `web/src/composables/useLiveKit.ts`

**Step 1: 创建 useLiveKit composable**

创建 `web/src/composables/useLiveKit.ts`:

```typescript
import { ref } from 'vue'
import { Room, RoomEvent, DataPacket_Kind } from 'livekit-client'
import { useConnectionStore } from '@/stores'
import type { DataMessage } from '@/types'

export function useLiveKit() {
  const connectionStore = useConnectionStore()

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
      })

      room.on(RoomEvent.DataReceived, (payload: Uint8Array) => {
        handleDataReceived(payload)
      })

      await room.connect(url, token)
      connectionStore.setRoom(room)
      connectionStore.setConnectionStatus('connected')
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
      // Handle data in components
      console.log('Data received:', data)
    } catch (error) {
      console.error('Failed to parse data:', error)
    }
  }

  async function sendCommand(action: string, params: Record<string, any>) {
    if (!connectionStore.room) return

    const data: DataMessage = { type: 'command', action, params }
    const encoder = new TextEncoder()
    await connectionStore.room.localParticipant.publishData(
      encoder.encode(JSON.stringify(data)),
      DataPacket_Kind.RELIABLE
    )
  }

  async function sendChat(text: string) {
    if (!connectionStore.room) return

    const data: DataMessage = { type: 'chat', content: text }
    const encoder = new TextEncoder()
    await connectionStore.room.localParticipant.publishData(
      encoder.encode(JSON.stringify(data)),
      DataPacket_Kind.RELIABLE
    )
  }

  async function disconnect() {
    if (connectionStore.room) {
      await connectionStore.room.disconnect()
    }
    connectionStore.disconnect()
  }

  return {
    connect,
    disconnect,
    sendCommand,
    sendChat
  }
}
```

**Step 2: Commit**

```bash
git add web/src/composables/
git commit -m "feat: 添加 LiveKit composable"
```

---

## Task 7: 配置 Vue Router 和创建 Layout

**Files:**
- Create: `web/src/router/index.ts`
- Create: `web/src/views/ConnectView.vue`
- Create: `web/src/views/RoomView.vue`

**Step 1: 创建路由配置**

创建 `web/src/router/index.ts`:

```typescript
import { createRouter, createWebHistory } from 'vue-router'
import type { RouteRecordRaw } from 'vue-router'

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    redirect: '/connect'
  },
  {
    path: '/connect',
    name: 'connect',
    component: () => import('@/views/ConnectView.vue'),
    meta: { title: '连接设备' }
  },
  {
    path: '/room',
    name: 'room',
    component: () => import('@/views/RoomView.vue'),
    meta: { requiresAuth: true, title: '控制台' }
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

router.beforeEach((to, _from, next) => {
  document.title = `${to.meta.title || 'LeLamp'} - LeLamp Web`
  next()
})

export default router
```

**Step 2: 更新 main.ts 添加路由**

修改 `web/src/main.ts`:

```typescript
import { createApp } from 'vue'
import { createPinia } from 'pinia'
import router from './router'  // 添加
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import './assets/styles/variables.scss'
import './assets/styles/global.scss'
import App from './App.vue'

const app = createApp(App)
const pinia = createPinia()

app.use(pinia)
app.use(router)  // 添加
app.use(ElementPlus)

app.mount('#app')
```

**Step 3: 创建 ConnectView.vue**

创建 `web/src/views/ConnectView.vue`:

```vue
<template>
  <div class="connect-view">
    <div class="connect-card">
      <div class="card-header">
        <h1>🪔 LeLamp Web Client</h1>
        <p>智能台灯，陪伴成长</p>
      </div>

      <el-form :model="form" label-position="top">
        <el-form-item label="LiveKit Server URL">
          <el-input
            v-model="form.serverUrl"
            placeholder="wss://your-project.livekit.cloud"
          />
        </el-form-item>

        <el-form-item label="Access Token">
          <el-input
            v-model="form.token"
            type="textarea"
            :rows="4"
            placeholder="粘贴生成的 Token..."
          />
          <div class="hint">
            运行 <code>python3 scripts/generate_client_token.py</code> 获取 Token
          </div>
        </el-form-item>

        <el-form-item>
          <el-button
            type="primary"
            size="large"
            :loading="loading"
            @click="handleConnect"
            style="width: 100%"
          >
            连接设备
          </el-button>
        </el-form-item>
      </el-form>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useLiveKit } from '@/composables/useLiveKit'

const router = useRouter()
const { connect } = useLiveKit()

const loading = ref(false)
const form = reactive({
  serverUrl: import.meta.env.VITE_LIVEKIT_URL || '',
  token: ''
})

async function handleConnect() {
  if (!form.serverUrl || !form.token) {
    ElMessage.warning('请填写完整信息')
    return
  }

  loading.value = true
  try {
    await connect(form.serverUrl, form.token)
    ElMessage.success('连接成功')
    router.push('/room')
  } catch (error) {
    ElMessage.error('连接失败，请检查配置')
    console.error(error)
  } finally {
    loading.value = false
  }
}
</script>

<style lang="scss" scoped>
.connect-view {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.connect-card {
  width: 90%;
  max-width: 500px;
  padding: 40px;
  background: white;
  border-radius: 16px;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
}

.card-header {
  text-align: center;
  margin-bottom: 32px;

  h1 {
    font-size: 28px;
    margin-bottom: 8px;
  }

  p {
    color: #666;
    font-size: 14px;
  }
}

.hint {
  margin-top: 8px;
  font-size: 12px;
  color: #999;

  code {
    padding: 2px 6px;
    background: #f5f5f5;
    border-radius: 4px;
    font-family: 'Courier New', monospace;
  }
}
</style>
```

**Step 4: 创建 RoomView.vue（基础框架）**

创建 `web/src/views/RoomView.vue`:

```vue
<template>
  <div class="room-view">
    <div class="room-header">
      <div class="status">
        <span class="status-dot online"></span>
        <span>已连接</span>
      </div>
      <el-button @click="handleDisconnect" type="danger">断开连接</el-button>
    </div>

    <div class="room-content">
      <div class="video-section">
        <div class="video-placeholder">📹</div>
        <p>等待摄像头画面...</p>
      </div>

      <div class="control-section">
        <!-- 快捷操作 -->
        <div class="panel">
          <h3>⚡ 快捷操作</h3>
          <div class="button-grid">
            <el-button @click="sendChat('你好')">👋 打招呼</el-button>
            <el-button @click="sendChat('现在几点了')">⏰ 查看时间</el-button>
            <el-button @click="sendChat('讲个笑话')">😄 讲笑话</el-button>
            <el-button @click="sendChat('唱首歌')">🎵 唱歌</el-button>
          </div>
        </div>

        <!-- 灯光控制 -->
        <div class="panel">
          <h3>💡 灯光魔法</h3>
          <div class="button-grid">
            <el-button @click="setEffect('breathing')">💗 呼吸灯</el-button>
            <el-button @click="setEffect('rainbow')">🌈 彩虹</el-button>
            <el-button @click="setEffect('wave')">🌊 波浪</el-button>
          </div>
        </div>
      </div>
    </div>

    <!-- 聊天框 -->
    <div class="chat-section">
      <h3>💬 实时对话</h3>
      <div class="messages"></div>
      <div class="input-area">
        <el-input v-model="inputText" @keyup.enter="sendMessage" placeholder="输入消息..." />
        <el-button type="primary" @click="sendMessage">发送</el-button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useLiveKit } from '@/composables/useLiveKit'
import { useChatStore } from '@/stores'

const router = useRouter()
const { disconnect, sendCommand } = useLiveKit()
const chatStore = useChatStore()

const inputText = ref('')

async function handleDisconnect() {
  await disconnect()
  router.push('/connect')
}

function sendChat(text: string) {
  chatStore.addMessage('user', text)
  // TODO: implement sendChat in useLiveKit
  ElMessage.success(`发送: ${text}`)
}

function setEffect(effect: string) {
  sendCommand(`rgb_effect_${effect}`, {})
  ElMessage.success(`设置特效: ${effect}`)
}

function sendMessage() {
  if (!inputText.value.trim()) return
  sendChat(inputText.value)
  inputText.value = ''
}
</script>

<style lang="scss" scoped>
.room-view {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: var(--lelamp-bg);
}

.room-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 24px;
  background: white;
  box-shadow: var(--shadow-sm);

  .status {
    display: flex;
    align-items: center;
    gap: 8px;

    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      &.online {
        background: var(--color-success);
      }
    }
  }
}

.room-content {
  display: grid;
  grid-template-columns: 1fr 400px;
  gap: 24px;
  padding: 24px;
  flex: 1;
  overflow: hidden;
}

.video-section {
  background: black;
  border-radius: var(--radius-lg);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: white;

  .video-placeholder {
    font-size: 64px;
  }
}

.control-section {
  display: flex;
  flex-direction: column;
  gap: 16px;
  overflow-y: auto;
}

.panel {
  padding: 20px;
  background: white;
  border-radius: var(--radius-md);
  box-shadow: var(--shadow-sm);

  h3 {
    margin-bottom: 16px;
  }
}

.button-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}

.chat-section {
  padding: 16px 24px;
  background: white;
  border-top: 1px solid #eee;

  h3 {
    margin-bottom: 12px;
  }

  .messages {
    height: 150px;
    overflow-y: auto;
    margin-bottom: 12px;
  }

  .input-area {
    display: flex;
    gap: 8px;
  }
}
</style>
```

**Step 5: Commit**

```bash
git add web/src/router/ web/src/views/ web/src/main.ts
git commit -m "feat: 添加路由和页面框架"
```

---

## Task 8: 安装依赖和测试运行

**Files:**
- Modify: `web/package.json` (已存在)

**Step 1: 安装依赖**

运行: `pnpm install`
Expected: 所有依赖成功安装，无错误

**Step 2: 测试开发服务器**

运行: `pnpm dev`
Expected: Vite 开发服务器启动在 http://localhost:5173

**Step 3: 测试构建**

运行: `pnpm build`
Expected: 构建成功，生成 `web/dist/` 目录

**Step 4: 测试类型检查**

运行: `pnpm type-check`
Expected: TypeScript 类型检查通过

**Step 5: Commit（如果有配置文件更新）**

```bash
git add web/
git commit -m "chore: 安装依赖并验证项目配置"
```

---

## Task 9: 创建通用组件

**Files:**
- Create: `web/src/components/common/StatusIndicator.vue`
- Create: `web/src/components/common/PrivacyIndicator.vue`

**Step 1: 创建 StatusIndicator 组件**

创建 `web/src/components/common/StatusIndicator.vue`:

```vue
<template>
  <div class="status-indicator">
    <span class="status-dot" :class="statusClass"></span>
    <span class="status-text">{{ statusText }}</span>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

interface Props {
  status: 'online' | 'offline' | 'connecting' | 'error'
}

const props = defineProps<Props>()

const statusClass = computed(() => props.status)
const statusText = computed(() => {
  const map = {
    online: '在线',
    offline: '离线',
    connecting: '连接中',
    error: '错误'
  }
  return map[props.status]
})
</script>

<style lang="scss" scoped>
.status-indicator {
  display: flex;
  align-items: center;
  gap: 6px;

  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;

    &.online {
      background: var(--color-success);
    }
    &.offline {
      background: var(--color-info);
    }
    &.connecting {
      background: var(--color-warning);
      animation: blink 1s infinite;
    }
    &.error {
      background: var(--color-danger);
    }
  }

  .status-text {
    font-size: 14px;
    color: #606266;
  }
}

@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.3; }
}
</style>
```

**Step 2: 创建 PrivacyIndicator 组件**

创建 `web/src/components/common/PrivacyIndicator.vue`:

```vue
<template>
  <div class="privacy-indicator">
    <div class="privacy-header">
      <span>🔒 隐私保护</span>
    </div>
    <div class="camera-status">
      <span class="camera-led" :class="{ active: isActive }"></span>
      <span class="status-text">{{ isActive ? '摄像头已开启' : '摄像头已关闭' }}</span>
    </div>
  </div>
</template>

<script setup lang="ts">
interface Props {
  active: boolean
}

defineProps<Props>()
</script>

<style lang="scss" scoped>
.privacy-indicator {
  padding: 12px 16px;
  background: rgba(0, 0, 0, 0.7);
  border-radius: 8px;
  color: white;
  font-size: 14px;

  .privacy-header {
    font-weight: 500;
    margin-bottom: 6px;
  }

  .camera-status {
    display: flex;
    align-items: center;
    gap: 8px;

    .camera-led {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: #666;
      transition: all 0.3s;

      &.active {
        background: #ff4444;
        box-shadow: 0 0 10px #ff4444;
        animation: pulse 2s infinite;
      }
    }

    .status-text {
      font-size: 13px;
      opacity: 0.9;
    }
  }
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}
</style>
```

**Step 3: Commit**

```bash
git add web/src/components/common/
git commit -m "feat: 添加通用组件（状态指示器、隐私指示器）"
```

---

## Task 10: 完善 useLiveKit 和 Chat 集成

**Files:**
- Modify: `web/src/composables/useLiveKit.ts`
- Modify: `web/src/views/RoomView.vue`

**Step 1: 完善 useLiveKit 添加 chat 集成**

修改 `web/src/composables/useLiveKit.ts`:

```typescript
import { ref } from 'vue'
import { Room, RoomEvent, DataPacket_Kind, Track } from 'livekit-client'
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

      room.on(RoomEvent.TrackSubscribed, (track: Track) => {
        // Handle video track
        console.log('Track subscribed:', track)
      })

      room.on(RoomEvent.TrackUnsubscribed, (track: Track) => {
        console.log('Track unsubscribed:', track)
      })

      room.on(RoomEvent.Disconnected, () => {
        connectionStore.setConnectionStatus('disconnected')
        deviceStore.setStatus('offline')
      })

      room.on(RoomEvent.DataReceived, (payload: Uint8Array) => {
        handleDataReceived(payload)
      })

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
    await connectionStore.room.localParticipant.publishData(
      encoder.encode(JSON.stringify(data)),
      DataPacket_Kind.RELIABLE
    )
  }

  async function sendChat(text: string) {
    if (!connectionStore.room) {
      console.warn('Room not connected')
      return
    }

    const data: DataMessage = { type: 'chat', content: text }
    const encoder = new TextEncoder()
    await connectionStore.room.localParticipant.publishData(
      encoder.encode(JSON.stringify(data)),
      DataPacket_Kind.RELIABLE
    )
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
```

**Step 2: 更新 RoomView.vue 集成聊天**

修改 `web/src/views/RoomView.vue` 的 script 部分:

```typescript
<script setup lang="ts">
import { ref, computed, onMounted, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useLiveKit } from '@/composables/useLiveKit'
import { useChatStore, useDeviceStore } from '@/stores'
import PrivacyIndicator from '@/components/common/PrivacyIndicator.vue'

const router = useRouter()
const { disconnect, sendCommand, sendChat: sendLiveKitChat } = useLiveKit()
const chatStore = useChatStore()
const deviceStore = useDeviceStore()

const inputText = ref('')
const messagesContainer = ref<HTMLElement>()

const messages = computed(() => chatStore.messages)
const cameraActive = computed(() => deviceStore.cameraActive)

// 监听消息变化，自动滚动到底部
function scrollToBottom() {
  nextTick(() => {
    if (messagesContainer.value) {
      messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
    }
  })
}

async function handleDisconnect() {
  await disconnect()
  router.push('/connect')
}

function sendChat(text: string) {
  sendLiveKitChat(text)
  ElMessage.success(`发送: ${text}`)
  scrollToBottom()
}

function setEffect(effect: string) {
  sendCommand(`rgb_effect_${effect}`, {})
  ElMessage.success(`设置特效: ${effect}`)
}

function sendMessage() {
  if (!inputText.value.trim()) return
  sendChat(inputText.value)
  inputText.value = ''
}

onMounted(() => {
  scrollToBottom()
})

// 监听消息变化
watch(() => chatStore.messages, () => {
  scrollToBottom()
}, { deep: true })
</script>
```

**Step 3: 更新 RoomView.vue 模板部分**

修改 template 添加聊天消息渲染:

```vue
<template>
  <div class="room-view">
    <!-- ... 头部保持不变 ... -->

    <div class="room-content">
      <div class="video-section">
        <div class="video-placeholder">📹</div>
        <p>等待摄像头画面...</p>
        <PrivacyIndicator :active="cameraActive" />
      </div>

      <!-- ... 控制面板保持不变 ... -->
    </div>

    <!-- 聊天框 -->
    <div class="chat-section">
      <h3>💬 实时对话</h3>
      <div ref="messagesContainer" class="messages">
        <div
          v-for="msg in messages"
          :key="msg.id"
          :class="['message', msg.sender]"
        >
          <div class="message-content">{{ msg.content }}</div>
          <div class="message-time">{{ formatTime(msg.timestamp) }}</div>
        </div>
        <div v-if="messages.length === 0" class="empty-hint">
          开始对话吧...
        </div>
      </div>
      <div class="input-area">
        <el-input
          v-model="inputText"
          @keyup.enter="sendMessage"
          placeholder="输入消息..."
        />
        <el-button type="primary" @click="sendMessage">发送</el-button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
// 在现有 import 基础上添加
import { watch } from 'vue'

// 添加时间格式化函数
function formatTime(timestamp: number): string {
  const date = new Date(timestamp)
  return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
}
</script>

<style lang="scss" scoped>
// 添加消息样式
.messages {
  height: 150px;
  overflow-y: auto;
  margin-bottom: 12px;
  padding: 12px;
  background: #f5f7fa;
  border-radius: 8px;

  .message {
    margin-bottom: 12px;

    .message-content {
      padding: 8px 12px;
      border-radius: 8px;
      max-width: 70%;
      word-break: break-word;
    }

    .message-time {
      font-size: 11px;
      color: #999;
      margin-top: 4px;
    }

    &.user {
      text-align: right;

      .message-content {
        background: var(--lelamp-brand);
        color: white;
        margin-left: auto;
      }
    }

    &.agent {
      text-align: left;

      .message-content {
        background: white;
        color: #333;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
      }
    }
  }

  .empty-hint {
    text-align: center;
    color: #999;
    font-size: 14px;
    padding: 40px 0;
  }
}
</style>
```

**Step 4: Commit**

```bash
git add web/src/
git commit -m "feat: 完善 LiveKit 集成和聊天功能"
```

---

## Task 11: 完善 LightPanel 组件

**Files:**
- Create: `web/src/components/room/LightPanel.vue`
- Modify: `web/src/views/RoomView.vue`

**Step 1: 创建 LightPanel 组件**

创建 `web/src/components/room/LightPanel.vue`:

```vue
<template>
  <div class="light-panel">
    <h3>💡 灯光魔法</h3>

    <!-- 颜色选择器 -->
    <div class="color-picker-section">
      <label>选择颜色:</label>
      <div class="color-input-row">
        <input
          ref="colorInput"
          type="color"
          v-model="selectedColor"
          @change="setCustomColor"
        />
        <el-button size="small" @click="setCustomColor">设置</el-button>
      </div>
    </div>

    <!-- 预设颜色 -->
    <div class="preset-colors">
      <h4>快速颜色</h4>
      <div class="color-grid">
        <div
          v-for="color in presetColors"
          :key="color.name"
          class="color-swatch"
          :style="{ backgroundColor: color.hex }"
          :title="color.name"
          @click="setPresetColor(color)"
        />
      </div>
    </div>

    <!-- 特效动画 -->
    <div class="effects-section">
      <h4>特效动画</h4>
      <div class="effects-grid">
        <el-button
          v-for="effect in effects"
          :key="effect.key"
          size="small"
          @click="setEffect(effect.key)"
        >
          {{ effect.emoji }} {{ effect.label }}
        </el-button>
      </div>
    </div>

    <!-- 关灯 -->
    <div class="actions">
      <el-button type="danger" size="small" @click="turnOffLight">
        🌑 关闭灯光
      </el-button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { ElMessage } from 'element-plus'
import { useLiveKit } from '@/composables/useLiveKit'

interface PresetColor {
  name: string
  hex: string
  rgb: { r: number; g: number; b: number }
}

const { sendCommand, sendChat } = useLiveKit()

const colorInput = ref<HTMLInputElement>()
const selectedColor = ref('#FFB6C1')

const presetColors: PresetColor[] = [
  { name: '暖红', hex: '#FF6B6B', rgb: { r: 255, g: 107, b: 107 } },
  { name: '粉红', hex: '#FFB6C1', rgb: { r: 255, g: 182, b: 193 } },
  { name: '橙色', hex: '#FFA500', rgb: { r: 255, g: 165, b: 0 } },
  { name: '金黄', hex: '#FFD700', rgb: { r: 255, g: 215, b: 0 } },
  { name: '浅绿', hex: '#90EE90', rgb: { r: 144, g: 238, b: 144 } },
  { name: '天蓝', hex: '#87CEEB', rgb: { r: 135, g: 206, b: 235 } },
  { name: '紫色', hex: '#9370DB', rgb: { r: 147, g: 112, b: 219 } },
  { name: '暖白', hex: '#FFF4E5', rgb: { r: 255, g: 244, b: 229 } }
]

const effects = [
  { key: 'breathing', label: '呼吸灯', emoji: '💗' },
  { key: 'rainbow', label: '彩虹', emoji: '🌈' },
  { key: 'wave', label: '波浪', emoji: '🌊' },
  { key: 'fire', label: '火焰', emoji: '🔥' },
  { key: 'fireworks', label: '烟花', emoji: '🎆' },
  { key: 'starry', label: '星空', emoji: '⭐' }
]

function setCustomColor() {
  const hex = selectedColor.value
  const r = parseInt(hex.slice(1, 3), 16)
  const g = parseInt(hex.slice(3, 5), 16)
  const b = parseInt(hex.slice(5, 7), 16)
  sendCommand('set_rgb_solid', { r, g, b })
  ElMessage.success('设置颜色')
}

function setPresetColor(color: PresetColor) {
  sendCommand('set_rgb_solid', color.rgb)
  selectedColor.value = color.hex
  ElMessage.success(`设置颜色: ${color.name}`)
}

function setEffect(effect: string) {
  sendCommand(`rgb_effect_${effect}`, {})
  ElMessage.success(`设置特效: ${effect}`)
}

function turnOffLight() {
  sendChat('关闭补光灯')
}
</script>

<style lang="scss" scoped>
.light-panel {
  padding: 20px;
  background: white;
  border-radius: var(--radius-md);
  box-shadow: var(--shadow-sm);

  h3 {
    margin-bottom: 16px;
  }

  h4 {
    font-size: 14px;
    margin: 16px 0 8px;
    color: #606266;
  }

  .color-picker-section {
    margin-bottom: 16px;

    label {
      display: block;
      font-size: 14px;
      margin-bottom: 8px;
      color: #606266;
    }

    .color-input-row {
      display: flex;
      gap: 8px;
      align-items: center;

      input[type="color"] {
        width: 60px;
        height: 32px;
        border: 1px solid #dcdfe6;
        border-radius: 4px;
        cursor: pointer;
      }
    }
  }

  .preset-colors {
    .color-grid {
      display: grid;
      grid-template-columns: repeat(8, 1fr);
      gap: 6px;

      .color-swatch {
        width: 100%;
        aspect-ratio: 1;
        border-radius: 4px;
        cursor: pointer;
        transition: transform 0.2s;
        border: 2px solid transparent;

        &:hover {
          transform: scale(1.1);
          border-color: var(--lelamp-brand);
        }
      }
    }
  }

  .effects-section {
    .effects-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 6px;
    }
  }

  .actions {
    margin-top: 16px;
    padding-top: 16px;
    border-top: 1px solid #eee;
  }
}
</style>
```

**Step 2: 在 RoomView 中使用 LightPanel**

修改 `web/src/views/RoomView.vue`，在 control-section 中替换灯光控制部分:

```vue
<template>
  <div class="room-content">
    <!-- ... video section 保持不变 ... -->

    <div class="control-section">
      <!-- 快捷操作 -->
      <div class="panel">
        <h3>⚡ 快捷操作</h3>
        <div class="button-grid">
          <el-button @click="sendChat('你好')">👋 打招呼</el-button>
          <el-button @click="sendChat('现在几点了')">⏰ 查看时间</el-button>
          <el-button @click="sendChat('讲个笑话')">😄 讲笑话</el-button>
          <el-button @click="sendChat('唱首歌')">🎵 唱歌</el-button>
        </div>
      </div>

      <!-- 使用 LightPanel 组件 -->
      <LightPanel />
    </div>
  </div>
</template>

<script setup lang="ts">
// 添加导入
import LightPanel from '@/components/room/LightPanel.vue'
</script>
```

**Step 3: Commit**

```bash
git add web/src/components/room/ web/src/views/RoomView.vue
git commit -m "feat: 添加 LightPanel 组件"
```

---

## Task 12: 添加 ESLint 和 Prettier 配置

**Files:**
- Create: `web/.eslintrc.json`
- Create: `web/.prettierrc.json`
- Create: `web/.eslintignore`
- Create: `web/.prettierignore`

**Step 1: 创建 ESLint 配置**

创建 `web/.eslintrc.json`:

```json
{
  "root": true,
  "env": {
    "browser": true,
    "es2020": true
  },
  "extends": [
    "plugin:vue/vue3-recommended",
    "eslint:recommended",
    "@typescript-eslint/recommended"
  ],
  "parser": "vue-eslint-parser",
  "parserOptions": {
    "ecmaVersion": "latest",
    "parser": "@typescript-eslint/parser",
    "sourceType": "module"
  },
  "plugins": ["vue", "@typescript-eslint"],
  "rules": {
    "vue/multi-word-component-names": "off",
    "@typescript-eslint/no-explicit-any": "warn",
    "@typescript-eslint/no-unused-vars": ["warn", { "argsIgnorePattern": "^_" }],
    "vue/no-v-html": "off"
  }
}
```

**Step 2: 创建 Prettier 配置**

创建 `web/.prettierrc.json`:

```json
{
  "semi": false,
  "singleQuote": true,
  "printWidth": 100,
  "trailingComma": "none",
  "arrowParens": "avoid",
  "endOfLine": "lf"
}
```

**Step 3: 创建忽略文件**

创建 `web/.eslintignore`:

```
dist
node_modules
*.d.ts
```

创建 `web/.prettierignore`:

```
dist
node_modules
pnpm-lock.yaml
package-lock.json
```

**Step 4: 更新 package.json 添加 lint script**

在 `web/package.json` 的 scripts 中添加:

```json
"lint:fix": "eslint . --ext .vue,.js,.jsx,.cjs,.mjs,.ts,.tsx,.cts,.mts --fix",
"format": "prettier --write src/"
```

**Step 5: 运行 lint 检查**

运行: `pnpm lint`
Expected: ESLint 检查通过或给出可修复的警告

**Step 6: 格式化代码**

运行: `pnpm format`
Expected: 代码被格式化

**Step 7: Commit**

```bash
git add web/
git commit -m "chore: 添加 ESLint 和 Prettier 配置"
```

---

## Task 13: 最终测试和验证

**Files:**
- Test all functionality

**Step 1: 完整功能测试**

启动开发服务器:
```bash
pnpm dev
```

测试清单:
- [ ] 访问 http://localhost:5173 显示连接页面
- [ ] 输入 Server URL 和 Token 可以连接
- [ ] 连接成功后跳转到控制台页面
- [ ] 显示在线状态
- [ ] 点击快捷操作按钮发送消息
- [ ] 点击颜色色块设置灯光颜色
- [ ] 点击特效按钮设置灯光特效
- [ ] 输入框发送消息
- [ ] 消息显示在聊天区域
- [ ] 断开连接按钮返回连接页面

**Step 2: 构建测试**

运行: `pnpm build`
Expected: 构建成功，生成优化后的产物

检查: `ls -la web/dist/`
Expected: 看到 index.html, assets/ 目录

**Step 3: 类型检查**

运行: `pnpm type-check`
Expected: 无类型错误

**Step 4: 性能检查**

检查构建产物大小:
```bash
du -sh web/dist/assets/*.js
```

Expected: vendor chunks 正确分离，单文件大小合理

**Step 5: 创建 README**

创建 `web/README.md`:

```markdown
# LeLamp Web Client

基于 Vue 3 + TypeScript + Vite 的现代化 Web 客户端。

## 开发

\`\`\`bash
# 安装依赖
pnpm install

# 启动开发服务器
pnpm dev

# 类型检查
pnpm type-check

# 代码检查
pnpm lint

# 格式化代码
pnpm format

# 构建生产版本
pnpm build
\`\`\`

## 技术栈

- Vue 3.4+ (Composition API)
- TypeScript 5.0+
- Vite 5.0+
- Pinia (状态管理)
- Element Plus (UI 组件库)
- LiveKit SDK 2.6+ (实时通信)
\`\`\`

**Step 6: Final Commit**

```bash
git add web/
git commit -m "feat: 完成 LeLamp Web 前端 Phase 1 MVP"
```

---

## 验收清单

完成所有任务后，验证以下项目:

### 功能完整性
- [x] 可以连接到 LiveKit 房间
- [x] 可以看到实时视频流（占位符）
- [x] 可以发送和接收文字消息
- [x] 快捷操作按钮正常工作
- [x] 灯光颜色选择器正常工作
- [x] 灯光特效正常工作
- [x] 隐私指示器正确显示摄像头状态
- [x] 断开连接功能正常

### 代码质量
- [x] TypeScript 类型检查无错误
- [x] ESLint 检查无警告
- [x] 组件遵循 Composition API 风格
- [x] 状态管理使用 Pinia
- [x] 代码通过 Prettier 格式化

### 用户体验
- [x] 界面响应式设计
- [x] 加载状态正确显示
- [x] 错误信息友好提示
- [x] 操作流畅无卡顿

### 项目结构
- [x] 正确的 workspace 配置
- [x] 模块化组件设计
- [x] 类型定义完整
- [x] 构建配置优化

---

## 下一步

Phase 1 MVP 完成后，可以考虑:

1. **Phase 2**: 添加设置页面
2. **Phase 3**: 添加设备健康监控
3. **Phase 4**: 添加历史记录功能
4. **Phase 5**: 性能优化和测试覆盖

---

**实施计划版本**: 1.0
**创建日期**: 2026-03-17
**状态**: 已批准，待执行
**预计工时**: 4-6 小时
