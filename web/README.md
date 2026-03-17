# LeLamp Web Client

基于 Vue 3 + TypeScript + Vite 的现代化 Web 客户端。

## 开发

```bash
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
```

## 技术栈

- Vue 3.4+ (Composition API)
- TypeScript 5.0+
- Vite 5.0+
- Pinia (状态管理)
- Element Plus (UI 组件库)
- LiveKit SDK 2.6+ (实时通信)

## 功能

- ✅ 连接管理（LiveKit 房间连接/断开）
- ✅ 视频预览（WebRTC 实时视频流）
- ✅ 实时对话（文字聊天）
- ✅ 快捷操作（打招呼、时间、笑话、唱歌）
- ✅ 基础灯光控制（颜色、特效）

## 项目结构

```
src/
├── assets/          # 静态资源
├── components/      # Vue 组件
│   ├── common/      # 通用组件
│   ├── connect/     # 连接相关组件
│   └── room/        # 房间相关组件
├── composables/     # Composition API 函数
├── router/          # 路由配置
├── stores/          # Pinia 状态管理
├── types/           # TypeScript 类型定义
├── utils/           # 工具函数
├── views/           # 页面视图
└── main.ts          # 应用入口
```

## 环境变量

创建 `.env.local` 文件：

```env
VITE_LIVEKIT_URL=https://your-livekit-server.com
VITE_LIVEKIT_TOKEN=your-access-token
```

## 开发规范

- 使用 Composition API 风格
- 组件命名采用 PascalCase
- 使用 `<script setup>` 语法
- 遵循 TypeScript 严格模式
- 代码格式化使用 Prettier
- 代码检查使用 ESLint

## 构建

生产构建会自动进行以下优化：

- 代码压缩和混淆
- Vendor chunks 分离（Vue、Element Plus、LiveKit）
- CSS 压缩和合并
- 资源哈希命名
- Tree-shaking 优化

构建产物位于 `dist/` 目录，可直接部署到静态服务器。
