# LeLamp Web 前端现代化设计文档

**日期**: 2026-03-17
**版本**: 1.0
**状态**: 设计阶段

---

## 1. 项目概述

### 1.1 目标

全面重构 LeLamp Web 客户端，使用现代化技术栈提升：
- **开发体验**: TypeScript 类型安全，组件化开发，易于维护
- **用户体验**: 现代化 UI 设计，流畅的交互反馈
- **扩展性**: 为新功能（设置、健康监控、历史记录、多设备）打好基础

### 1.2 现状分析

**当前实现** (`web_client/`):
- 原生 HTML + JavaScript
- 约 600 行代码，单一文件结构
- 使用 LiveKit JS SDK 进行 WebRTC 通信
- 功能完整但 UI 简陋，代码可维护性差

### 1.3 技术栈

| 类别 | 技术选型 | 版本 |
|------|----------|------|
| 框架 | Vue 3 | ^3.4+ |
| 语言 | TypeScript | ^5.0+ |
| 构建工具 | Vite | ^5.0+ |
| 状态管理 | Pinia | ^2.1+ |
| 路由 | Vue Router | ^4.2+ |
| UI 组件 | Element Plus | ^2.4+ |
| 实时通信 | livekit-client-sdk | ^2.0+ |
| HTTP 客户端 | axios | ^1.6+ |
| 样式 | SCSS | ^1.7+ |
| 代码规范 | ESLint + Prettier | latest |
| 包管理器 | pnpm | ^8.0+ |

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
│   │   ├── main.ts
│   │   ├── App.vue
│   │   ├── assets/              # 静态资源
│   │   │   ├── images/
│   │   │   └── styles/
│   │   ├── components/          # 通用组件
│   │   │   ├── common/
│   │   │   │   ├── AppHeader.vue
│   │   │   │   ├── AppSidebar.vue
│   │   │   │   ├── StatusIndicator.vue
│   │   │   │   └── ToastNotification.vue
│   │   │   ├── room/
│   │   │   │   ├── VideoPlayer.vue
│   │   │   │   ├── QuickActions.vue
│   │   │   │   ├── VisionPanel.vue
│   │   │   │   ├── AnimationPanel.vue
│   │   │   │   ├── LightPanel.vue
│   │   │   │   ├── ChatBox.vue
│   │   │   │   └── PrivacyIndicator.vue
│   │   │   ├── settings/
│   │   │   │   ├── ThemeSwitcher.vue
│   │   │   │   ├── LanguageSelector.vue
│   │   │   │   ├── ServerManager.vue
│   │   │   │   └── GeneralSettings.vue
│   │   │   ├── health/
│   │   │   │   ├── HealthOverview.vue
│   │   │   │   ├── MotorHealthCard.vue
│   │   │   │   └── HealthChart.vue
│   │   │   └── history/
│   │   │       ├── ConversationList.vue
│   │   │       └── OperationLog.vue
│   │   ├── views/               # 页面视图
│   │   │   ├── ConnectView.vue
│   │   │   ├── RoomView.vue
│   │   │   ├── SettingsView.vue
│   │   │   ├── HealthView.vue
│   │   │   ├── HistoryView.vue
│   │   │   └── DevicesView.vue
│   │   ├── stores/              # Pinia stores
│   │   │   ├── connection.ts
│   │   │   ├── device.ts
│   │   │   ├── chat.ts
│   │   │   ├── settings.ts
│   │   │   └── history.ts
│   │   ├── composables/         # Vue composables
│   │   │   ├── useLiveKit.ts
│   │   │   ├── useTheme.ts
│   │   │   ├── useI18n.ts
│   │   │   └── useNotification.ts
│   │   ├── types/               # TypeScript 类型
│   │   │   ├── livekit.ts
│   │   │   ├── device.ts
│   │   │   └── api.ts
│   │   ├── utils/               # 工具函数
│   │   │   ├── storage.ts
│   │   │   ├── format.ts
│   │   │   └── validation.ts
│   │   ├── i18n/                # 国际化
│   │   │   ├── index.ts
│   │   │   ├── zh-CN.ts
│   │   │   └── en-US.ts
│   │   └── styles/              # 全局样式
│   │       ├── variables.scss
│   │       ├── mixins.scss
│   │       └── global.scss
│   └── dist/                    # 构建产物
├── web_client/                  # 旧的 HTML/JS（保留用于对比）
└── ...（后端代码）
```

---

## 3. 路由设计

```typescript
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
      },
      {
        path: 'settings',
        name: 'settings',
        component: SettingsView,
        meta: { title: '设置' }
      },
      {
        path: 'health',
        name: 'health',
        component: HealthView,
        meta: { title: '设备健康' }
      },
      {
        path: 'history',
        name: 'history',
        component: HistoryView,
        meta: { title: '历史记录' }
      },
      {
        path: 'devices',
        name: 'devices',
        component: DevicesView,
        meta: { title: '设备管理' }
      }
    ]
  }
]
```

---

## 4. 状态管理 (Pinia Stores)

### 4.1 Connection Store

```typescript
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
}
```

### 4.2 Device Store

```typescript
interface DeviceState {
  name: string
  status: 'online' | 'offline' | 'error'
  cameraActive: boolean
  motors: {
    positions: Record<string, number>
    health: Record<string, MotorHealth>
  }
  lights: {
    color: { r: number; g: number; b: number }
    effect: string | null
  }
}

interface MotorHealth {
  temperature: number
  voltage: number
  load: number
  status: 'healthy' | 'warning' | 'critical' | 'stalled'
}
```

### 4.3 Chat Store

```typescript
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
```

### 4.4 Settings Store

```typescript
interface SettingsState {
  theme: 'light' | 'dark' | 'auto'
  language: 'zh' | 'en'
  autoReconnect: boolean
  notifications: boolean
  savedServers: ServerConfig[]
}

interface ServerConfig {
  name: string
  url: string
  token: string
}
```

### 4.5 History Store

```typescript
interface HistoryState {
  conversations: Conversation[]
  operations: Operation[]
}

interface Conversation {
  id: string
  timestamp: number
  messages: Message[]
}

interface Operation {
  id: string
  timestamp: number
  type: 'command' | 'animation' | 'light'
  action: string
  params: Record<string, any>
  success: boolean
}
```

---

## 5. LiveKit 集成

### 5.1 Composable: useLiveKit

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
      videoCaptureDefaults: {
        facingMode: 'environment',
      },
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
      case 'vision_result':
        deviceStore.setVisionResult(data.content, data.image_base64)
        break
      case 'camera_status':
        deviceStore.setCameraActive(data.active)
        break
      case 'health_update':
        deviceStore.updateHealth(data)
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

  return { connect, disconnect, sendCommand, sendChat }
}
```

---

## 6. 页面设计

### 6.1 连接页面 (ConnectView)

- Server URL 输入框
- Token 输入框（多行）
- 服务器配置下拉（已保存的服务器）
- 连接按钮
- 记住凭证选项

### 6.2 主控制台 (RoomView)

**布局**:
- 左侧：视频预览区 + 隐私指示器
- 右侧：Tab 面板（快捷操作、视觉助手、动作表情、灯光魔法）
- 底部：实时对话框

**快捷操作**:
- 打招呼、查看时间、讲笑话、唱歌

**视觉助手**:
- 拍照识别、检查作业、推送飞书
- 结果展示（图片 + 文字）

**动作表情**:
- 点头、摇头、兴奋、睡觉、跳舞、思考

**灯光魔法**:
- 颜色选择器
- 预设颜色（8 个色块）
- 特效动画（呼吸、彩虹、波浪、火焰、烟花、星空）

### 6.3 设置页面 (SettingsView)

**主题设置**:
- 亮色/暗色/自动切换
- 预览效果

**语言设置**:
- 中文/英文切换

**服务器管理**:
- 已保存服务器列表
- 添加/编辑/删除服务器配置

**通用设置**:
- 自动重连
- 通知开关
- 调试模式

### 6.4 设备健康页面 (HealthView)

**健康概览**:
- 整体状态指示
- 最后检查时间

**电机健康卡片**:
- 5 个电机的温度、电压、负载
- 状态图标（健康/警告/危险/堵转）
- 统计信息（警告次数、危险次数、堵转次数）

### 6.5 历史记录页面 (HistoryView)

**对话记录**:
- 按时间分组的对话列表
- 搜索功能

**操作日志**:
- 时间线视图
- 筛选功能（按类型）

### 6.6 设备管理页面 (DevicesView)

**多设备支持**:
- 设备列表
- 添加新设备
- 快速切换设备

---

## 7. 主题与样式

### 7.1 设计变量

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
  --radius-full: 9999px;

  // 阴影
  --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.1);
  --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.1);
  --shadow-lg: 0 10px 20px rgba(0, 0, 0, 0.1);
}

// 暗色主题
.dark {
  --lelamp-bg: #141414;
  --lelamp-surface: #1d1d1d;
  --el-bg-color: #1d1d1d;
  --el-text-color-primary: #e5eaf3;
}
```

### 7.2 Element Plus 主题定制

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

## 8. 国际化

### 8.1 支持语言

- 中文 (zh-CN)
- 英文 (en-US)

### 8.2 翻译结构

```typescript
// i18n/zh-CN.ts
export default {
  common: {
    connect: '连接设备',
    disconnect: '断开连接',
    settings: '设置',
    health: '设备健康',
    history: '历史记录',
    devices: '设备管理',
    save: '保存',
    cancel: '取消',
  },
  connect: {
    title: '连接 LeLamp',
    serverUrl: 'LiveKit Server URL',
    token: 'Access Token',
    connectBtn: '连接',
    savedServers: '已保存的服务器',
  },
  room: {
    quickActions: '快捷操作',
    vision: '视觉助手',
    animation: '动作表情',
    light: '灯光魔法',
    chat: '实时对话',
  },
  // ...
}
```

---

## 9. 构建配置

### 9.1 Vite 配置

```typescript
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { ElementPlusResolver } from 'unplugin-vue-components/resolvers'
import { fileURLToPath } from 'node:url'

export default defineConfig({
  plugins: [
    vue(),
    ElementPlusResolver(),
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
        changeOrigin: true,
      }
    }
  },
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: false,
    minify: 'terser',
    rollupOptions: {
      output: {
        manualChunks: {
          'livekit': ['livekit-client'],
          'element-plus': ['element-plus'],
          'vue-vendor': ['vue', 'vue-router', 'pinia'],
          'echarts': ['echarts'],
        }
      }
    }
  }
})
```

### 9.2 pnpm Workspace

```yaml
# pnpm-workspace.yaml
packages:
  - 'web'
```

```json
// package.json (根目录)
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

---

## 10. 部署策略

### 10.1 开发模式

```bash
# 终端 1: 后端
uv run main.py dev

# 终端 2: 前端
cd web && pnpm dev
```

### 10.2 生产集成模式

```bash
cd web && pnpm build
# 构建产物到 web/dist/
# 后端服务静态文件目录
```

### 10.3 独立部署模式

```bash
cd web && pnpm build
# 部署 dist/ 到 Vercel/Netlify
# 通过 LiveKit Cloud URL 连接
```

---

## 11. 开发规范

### 11.1 代码风格

- 使用 ESLint + Prettier
- 遵循 Vue 3 Composition API 风格
- 使用 `<script setup>` 语法
- 组件命名使用 PascalCase
- Composable 函数命名使用 `use` 前缀

### 11.2 Git 提交规范

```
feat: 添加新功能
fix: 修复问题
refactor: 重构代码
style: 样式调整
docs: 文档更新
chore: 构建/工具更新
```

---

## 12. 迁移策略

### 12.1 渐进式迁移

1. **阶段 1**: 基础框架搭建
   - 初始化项目
   - 配置构建工具
   - 建立基础组件

2. **阶段 2**: 核心功能迁移
   - 连接管理
   - 视频预览
   - 实时对话

3. **阶段 3**: 控制面板迁移
   - 快捷操作
   - 视觉助手
   - 动作表情
   - 灯光控制

4. **阶段 4**: 新功能开发
   - 设置页面
   - 设备健康
   - 历史记录

5. **阶段 5**: 优化与部署
   - 性能优化
   - 测试覆盖
   - 生产部署

### 12.2 回退策略

- 保留 `web_client/` 作为备份
- 可以随时切换回旧版本
- 新版本独立运行，互不影响

---

**文档版本**: 1.0
**最后更新**: 2026-03-17
**状态**: 待审批
