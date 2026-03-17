# LeLamp Web 前端现代化实施方案（Phase 1 - MVP）

**日期**: 2026-03-17
**版本**: 1.0
**状态**: 已批准
**基于**: `2026-03-17-web-frontend-modernization-design.md`

---

## 1. 项目概述

### 1.1 目标

基于已有的完整设计文档，实施**第一阶段核心功能（MVP）**，建立稳固的前端架构基础，为后续功能扩展做好准备。

### 1.2 实施范围

**包含功能** ✅：
- 连接管理（LiveKit 房间连接/断开）
- 视频预览（WebRTC 实时视频流）
- 实时对话（文字聊天）
- 快捷操作（打招呼、时间、笑话、唱歌）
- 基础灯光控制（颜色选择器、预设颜色、6 种特效）

**暂不包含** ❌（后续迭代）：
- 设置页面
- 健康监控
- 历史记录
- 多设备管理

### 1.3 技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| Vue | 3.4+ | 渐进式框架，Composition API |
| TypeScript | 5.0+ | 类型安全 |
| Vite | 5.0+ | 快速构建工具 |
| Pinia | 2.1+ | 状态管理 |
| Vue Router | 4.2+ | 路由管理 |
| Element Plus | 2.4+ | UI 组件库 |
| LiveKit SDK | 2.6+ | 实时通信 |
| axios | 1.6+ | HTTP 客户端 |
| SCSS | 1.7+ | 样式预处理 |
| pnpm | 8.0+ | 包管理器 |

---

## 2. 项目结构

```
lelamp_runtime/
├── package.json                 # 根 workspace 配置
├── pnpm-workspace.yaml          # workspace 定义
├── web/                         # Vue 3 前端项目
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── index.html
│   ├── .env.development
│   ├── .env.production
│   ├── src/
│   │   ├── main.ts              # 入口文件
│   │   ├── App.vue              # 根组件
│   │   ├── assets/              # 静态资源
│   │   │   └── styles/          # 全局样式
│   │   │       ├── variables.scss    # SCSS 变量
│   │   │       └── global.scss       # 全局样式
│   │   ├── components/          # 组件
│   │   │   ├── common/          # 通用组件
│   │   │   │   ├── AppHeader.vue        # 应用头部
│   │   │   │   └── StatusIndicator.vue  # 状态指示器
│   │   │   └── room/            # 控制台组件
│   │   │       ├── VideoPlayer.vue      # 视频播放器
│   │   │       ├── QuickActions.vue     # 快捷操作面板
│   │   │       ├── LightPanel.vue       # 灯光控制面板
│   │   │       ├── ChatBox.vue          # 聊天对话框
│   │   │       └── PrivacyIndicator.vue # 隐私指示器
│   │   ├── views/               # 页面视图
│   │   │   ├── ConnectView.vue         # 连接页面
│   │   │   └── RoomView.vue            # 控制台页面
│   │   ├── stores/              # Pinia stores
│   │   │   ├── connection.ts           # 连接状态管理
│   │   │   ├── device.ts               # 设备状态管理
│   │   │   └── chat.ts                 # 聊天状态管理
│   │   ├── composables/         # Vue composables
│   │   │   └── useLiveKit.ts           # LiveKit 集成
│   │   ├── types/               # TypeScript 类型定义
│   │   │   ├── livekit.ts              # LiveKit 类型
│   │   │   └── device.ts               # 设备类型
│   │   └── utils/               # 工具函数
│   │       └── format.ts               # 格式化工具
│   └── dist/                    # 构建产物
└── web_client/                  # 旧的 HTML/JS（保留作为对比）
```

---

## 3. 核心架构设计

### 3.1 状态管理（Pinia）

#### Connection Store

```typescript
// stores/connection.ts
interface ConnectionState {
  isConnected: boolean
  room: Room | null
  serverUrl: string
  token: string
  connectionStatus: 'disconnected' | 'connecting' | 'connected' | 'error'
}

interface ConnectionActions {
  connect(url: string, token: string): Promise<void>
  disconnect(): void
  setConnectionStatus(status: ConnectionStatus): void
  setRoom(room: Room): void
}
```

#### Device Store

```typescript
// stores/device.ts
interface DeviceState {
  name: string
  status: 'online' | 'offline'
  cameraActive: boolean
  lights: {
    color: { r: number; g: number; b: number }
    effect: string | null
  }
}

interface DeviceActions {
  setCameraActive(active: boolean): void
  setLightColor(r: number, g: number, b: number): void
  setLightEffect(effect: string): void
}
```

#### Chat Store

```typescript
// stores/chat.ts
interface ChatState {
  messages: Message[]
  isProcessing: boolean
}

interface Message {
  id: string
  sender: 'user' | 'agent'
  content: string
  timestamp: number
}

interface ChatActions {
  addMessage(sender: 'user' | 'agent', content: string): void
  clearMessages(): void
  setProcessing(processing: boolean): void
}
```

### 3.2 LiveKit 集成

```typescript
// composables/useLiveKit.ts
export function useLiveKit() {
  const connectionStore = useConnectionStore()
  const deviceStore = useDeviceStore()
  const chatStore = useChatStore()

  async function connect(url: string, token: string) {
    const room = new Room({
      adaptiveStream: true,
      dynacast: true,
      audioCaptureDefaults: { autoGainControl: true },
      videoCaptureDefaults: { facingMode: 'environment' },
    })

    // 事件监听
    room.on(RoomEvent.TrackSubscribed, handleTrackSubscribed)
    room.on(RoomEvent.TrackUnsubscribed, handleTrackUnsubscribed)
    room.on(RoomEvent.DataReceived, handleDataReceived)
    room.on(RoomEvent.Disconnected, handleDisconnected)

    await room.connect(url, token)
    connectionStore.setRoom(room)
    connectionStore.setConnectionStatus('connected')
  }

  function handleDataReceived(payload: Uint8Array) {
    const data = JSON.parse(new TextDecoder().decode(payload))

    switch (data.type) {
      case 'chat':
        chatStore.addMessage('agent', data.content)
        break
      case 'camera_status':
        deviceStore.setCameraActive(data.active)
        break
    }
  }

  async function sendCommand(action: string, params: Record<string, any>) {
    const data = JSON.stringify({ type: 'command', action, params })
    const encoder = new TextEncoder()
    await connectionStore.room?.localParticipant.publishData(
      encoder.encode(data),
      DataPacket_Kind.RELIABLE
    )
  }

  async function sendChat(text: string) {
    const data = JSON.stringify({ type: 'chat', content: text })
    const encoder = new TextEncoder()
    await connectionStore.room?.localParticipant.publishData(
      encoder.encode(data),
      DataPacket_Kind.RELIABLE
    )
    chatStore.addMessage('user', text)
  }

  async function disconnect() {
    await connectionStore.room?.disconnect()
    connectionStore.setRoom(null)
    connectionStore.setConnectionStatus('disconnected')
  }

  return { connect, disconnect, sendCommand, sendChat }
}
```

### 3.3 数据流设计

```
用户操作 → sendCommand/sendChat → LiveKit Room → 后端 Agent
                                              ↓
设备状态更新 ← handleDataReceived ← LiveKit Room ← 后端 Agent
     ↓
Pinia Store 更新 → 组件重新渲染
```

---

## 4. 路由设计

```typescript
// router/index.ts
const routes = [
  {
    path: '/',
    component: LayoutView,
    redirect: '/connect',
    children: [
      {
        path: 'connect',
        name: 'connect',
        component: ConnectView,
        meta: { title: '连接设备' }
      },
      {
        path: 'room',
        name: 'room',
        component: RoomView,
        meta: { requiresAuth: true, title: '控制台' }
      }
    ]
  }
]
```

---

## 5. 页面与组件设计

### 5.1 连接页面（ConnectView）

**功能**：
- Server URL 输入框（支持历史记录）
- Token 多行输入框
- 连接按钮（带加载状态）
- 错误提示

**UI 布局**：
- 居中卡片式设计
- 简洁的表单布局
- 响应式设计

### 5.2 控制台页面（RoomView）

**布局结构**：
```
┌─────────────────────────────────────────┐
│  Header（连接状态 + 断开按钮）           │
├─────────────────┬───────────────────────┤
│  Video Player   │  Tab Panel            │
│  16:9           │  - 快捷操作           │
│  + Privacy LED  │  - 灯光控制           │
├─────────────────┴───────────────────────┤
│  Chat Box                              │
│  - 消息列表                             │
│  - 输入框 + 发送按钮                    │
└─────────────────────────────────────────┘
```

**组件职责**：
- `VideoPlayer.vue`：渲染视频轨道，处理视频流
- `PrivacyIndicator.vue`：显示摄像头状态（LED 指示器）
- `QuickActions.vue`：快捷操作按钮网格
- `LightPanel.vue`：颜色选择器 + 特效按钮
- `ChatBox.vue`：消息列表 + 输入框

### 5.3 核心组件实现要点

#### VideoPlayer.vue

```vue
<template>
  <div class="video-player">
    <div ref="videoContainer" class="video-container">
      <div v-if="!hasVideo" class="placeholder">
        <div class="placeholder-icon">📹</div>
        <p>等待摄像头画面...</p>
      </div>
    </div>
    <PrivacyIndicator :active="deviceStore.cameraActive" />
  </div>
</template>

<script setup lang="ts">
const videoContainer = ref<HTMLDivElement>()
const hasVideo = ref(false)

onMounted(() => {
  // 监听 TrackSubscribed 事件，渲染视频
})
</script>
```

#### QuickActions.vue

```vue
<template>
  <div class="quick-actions">
    <h3>⚡ 快捷操作</h3>
    <div class="action-grid">
      <el-button
        v-for="action in actions"
        :key="action.key"
        @click="handleAction(action)"
      >
        {{ action.emoji }} {{ action.label }}
      </el-button>
    </div>
  </div>
</template>

<script setup lang="ts">
const actions = [
  { key: 'greet', label: '打招呼', emoji: '👋', chat: '你好' },
  { key: 'time', label: '查看时间', emoji: '⏰', chat: '现在几点了' },
  { key: 'joke', label: '讲笑话', emoji: '😄', chat: '讲个笑话' },
  { key: 'sing', label: '唱歌', emoji: '🎵', chat: '唱首歌' },
]

const { sendChat } = useLiveKit()

function handleAction(action: Action) {
  sendChat(action.chat)
}
</script>
```

#### LightPanel.vue

```vue
<template>
  <div class="light-panel">
    <h3>💡 灯光魔法</h3>

    <!-- 颜色选择器 -->
    <div class="color-picker">
      <input type="color" v-model="selectedColor" @change="setCustomColor" />
      <el-button @click="setCustomColor">设置</el-button>
    </div>

    <!-- 预设颜色 -->
    <div class="preset-colors">
      <div
        v-for="color in presetColors"
        :key="color.name"
        class="color-swatch"
        :style="{ backgroundColor: color.hex }"
        @click="setPresetColor(color)"
      />
    </div>

    <!-- 特效动画 -->
    <div class="effects">
      <el-button
        v-for="effect in effects"
        :key="effect.key"
        @click="setEffect(effect.key)"
      >
        {{ effect.emoji }} {{ effect.label }}
      </el-button>
    </div>
  </div>
</template>
```

#### ChatBox.vue

```vue
<template>
  <div class="chat-box">
    <div class="chat-header">
      <h3>💬 实时对话</h3>
      <el-button @click="clearMessages" size="small">清空</el-button>
    </div>

    <div class="messages">
      <div
        v-for="msg in chatStore.messages"
        :key="msg.id"
        :class="['message', msg.sender]"
      >
        <div class="message-content">{{ msg.content }}</div>
        <div class="message-time">{{ formatTime(msg.timestamp) }}</div>
      </div>
    </div>

    <div class="input-area">
      <el-input
        v-model="inputText"
        placeholder="输入消息..."
        @keyup.enter="sendMessage"
      />
      <el-button type="primary" @click="sendMessage">发送</el-button>
    </div>
  </div>
</template>
```

---

## 6. 主题与样式

### 6.1 设计变量

```scss
// styles/variables.scss
:root {
  // 品牌色
  --lelamp-brand: #FF6B6B;
  --lelamp-accent: #4ECDC4;
  --lelamp-bg: #F5F7FA;
  --lelamp-surface: #FFFFFF;

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

### 6.2 Element Plus 主题定制

```typescript
// styles/element-variables.scss
@forward 'element-plus/theme-chalk/src/common/var.scss' with (
  $colors: (
    'primary': (
      'base': #FF6B6B,
    ),
  )
);
```

---

## 7. 开发工具配置

### 7.1 Vite 配置

```typescript
// vite.config.ts
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { fileURLToPath } from 'node:url'

export default defineConfig({
  plugins: [vue()],
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
        changeOrigin: true,
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
          'vue-vendor': ['vue', 'vue-router', 'pinia'],
        }
      }
    }
  }
})
```

### 7.2 TypeScript 配置

```json
// tsconfig.json
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

### 7.3 ESLint 配置

```json
// .eslintrc.json
{
  "extends": [
    "plugin:vue/vue3-recommended",
    "plugin:@typescript-eslint/recommended"
  ],
  "rules": {
    "vue/multi-word-component-names": "off",
    "@typescript-eslint/no-explicit-any": "warn"
  }
}
```

### 7.4 Prettier 配置

```json
// .prettierrc.json
{
  "semi": false,
  "singleQuote": true,
  "printWidth": 100,
  "trailingComma": "none"
}
```

---

## 8. 构建与部署

### 8.1 Workspace 配置

```yaml
# pnpm-workspace.yaml（根目录）
packages:
  - 'web'
```

```json
// package.json（根目录）
{
  "name": "lelamp-runtime-workspace",
  "private": true,
  "scripts": {
    "dev": "pnpm --filter web dev",
    "build": "pnpm --filter web build",
    "preview": "pnpm --filter web preview"
  }
}
```

### 8.2 开发模式

```bash
# 终端 1: 后端
uv run main.py dev

# 终端 2: 前端
cd web && pnpm dev
# 访问 http://localhost:5173
```

### 8.3 生产构建

```bash
cd web && pnpm build
# 构建产物到 web/dist/
```

### 8.4 部署策略

**选项 A：集成部署**
- 后端 FastAPI 服务 `web/dist/` 静态文件
- 单一服务器部署

**选项 B：独立部署**
- `web/dist/` 部署到 Vercel/Netlify
- 通过 LiveKit Cloud URL 连接
- 前后端分离部署

---

## 9. 验收标准

### 9.1 功能完整性

- [ ] 可以连接到 LiveKit 房间
- [ ] 可以看到实时视频流
- [ ] 可以发送和接收文字消息
- [ ] 快捷操作按钮正常工作（打招呼、时间、笑话、唱歌）
- [ ] 灯光颜色选择器正常工作
- [ ] 灯光特效正常工作（呼吸、彩虹、波浪、火焰、烟花、星空）
- [ ] 隐私指示器正确显示摄像头状态
- [ ] 断开连接功能正常

### 9.2 代码质量

- [ ] TypeScript 类型检查无错误（`pnpm type-check`）
- [ ] ESLint 检查无警告（`pnpm lint`）
- [ ] 组件遵循 Composition API 风格
- [ ] 状态管理使用 Pinia
- [ ] 代码通过 Prettier 格式化

### 9.3 用户体验

- [ ] 界面响应式设计（支持移动端）
- [ ] 加载状态正确显示
- [ ] 错误信息友好提示
- [ ] 操作流畅无卡顿
- [ ] 页面切换动画自然

### 9.4 性能指标

- [ ] 首屏加载时间 < 2s
- [ ] 构建产物大小合理（vendor chunk 分离）
- [ ] 无内存泄漏（正确清理事件监听器）

---

## 10. 实施计划概要

本设计文档将转入详细实施计划阶段。后续将创建以下任务：

### 阶段 1：项目初始化
- 配置 pnpm workspace
- 初始化 Vue 3 + Vite 项目
- 配置 TypeScript、ESLint、Prettier
- 创建基础目录结构

### 阶段 2：核心功能实现
- 实现 Pinia stores（connection, device, chat）
- 实现 useLiveKit composable
- 创建连接页面
- 创建控制台页面布局

### 阶段 3：组件开发
- VideoPlayer 组件
- QuickActions 组件
- LightPanel 组件
- ChatBox 组件
- PrivacyIndicator 组件

### 阶段 4：样式与主题
- 实现 SCSS 变量系统
- Element Plus 主题定制
- 响应式布局适配

### 阶段 5：测试与优化
- 功能测试
- 性能优化
- 兼容性测试

---

**文档版本**: 1.0
**创建日期**: 2026-03-17
**状态**: 已批准，等待实施计划
**下一步**: 调用 `writing-plans` skill 创建详细实施计划
