# LeLamp 用户端 APP 产品设计评审

**评审时间**: 2026-03-16
**评审角色**: 产品经理视角
**评审对象**: LeLamp Web Client (MVP) 及移动端 App 规划

---

## 📊 执行摘要

### 总体评价
当前 Web Client 完成了基础的 MVP 验证，具备了核心技术可行性，但从**商业化产品**的角度来看，**还有大量用户体验、安全性和商业价值的工作需要补齐**。

**评分**: 🟡 **6.5/10** (技术原型合格，但商业就绪度不足)

---

## ✅ 当前已完成的亮点

### 1. 技术架构扎实
- ✅ **LiveKit WebRTC 集成**: 实现了低延迟实时音视频通信
- ✅ **双向数据通道**: 可以通过 Data Channel 发送控制指令
- ✅ **Token 认证机制**: 基于 LiveKit AccessToken 的安全连接
- ✅ **后端能力丰富**:
  - 20+ function tools (运动控制、RGB 灯效、视觉 AI、作业检查、飞书推送、联网搜索、OTA 更新)
  - 完整的语音对话系统 (DeepSeek LLM + Baidu Speech)

### 2. 用户体验友好的功能
- ✅ **快捷操作按钮**: 打招呼、检查作业、开灯/关灯等高频场景直达
- ✅ **实时视频预览**: 直观看到台灯的摄像头画面
- ✅ **聊天对话历史**: 展示用户与 Agent 的交互记录

---

## 🚨 核心问题与改进建议

### 问题 1: 用户身份与设备管理缺失 ⭐⭐⭐⭐⭐ (最高优先级)

**当前状态**:
- ❌ **没有用户账号体系**: 任何人拿到 Token 都能连接设备
- ❌ **没有设备绑定机制**: 无法验证用户是否是设备的合法拥有者
- ❌ **Token 生成是手动的**: 需要在终端运行 Python 脚本生成 Token，技术门槛太高

**风险**:
- 家庭隐私严重泄露 (视频流可被任何知道房间号的人观看)
- 无法支持多用户场景 (父母、孩子、祖父母分别使用)
- 无法实现商业化付费订阅 (没有账号体系就没有计费载体)

**改进方案**:

#### Phase 1 (MVP 最小安全闭环):
```plaintext
┌─────────────────────────────────────────────────────────────┐
│ 1. 设备首次启动时生成唯一 Device ID (基于 CPU 序列号)      │
│ 2. 用户扫描设备底部二维码 (包含 Device ID + 出厂 Secret)    │
│ 3. App 调用后端 API: POST /api/device/bind                  │
│    请求体: { "device_id": "xxx", "secret": "yyy" }          │
│ 4. 后端验证 Secret 合法性，创建 User <-> Device 绑定关系    │
│ 5. 返回长期有效的 Device Token (存储在 App 本地)            │
│ 6. 后续连接时，App 用 Device Token 换取临时 LiveKit Token   │
└─────────────────────────────────────────────────────────────┘
```

#### Phase 2 (多用户支持):
- 主账号可以邀请家庭成员 (通过二维码分享)
- 支持权限分级:
  - **管理员**: 可以解绑设备、修改设置、查看所有记录
  - **家庭成员**: 可以使用日常功能 (对话、看视频、开关灯)
  - **临时访客**: 仅能查看实时视频 (适合保姆、维修人员)

**参考实现**:
- 家用摄像头产品 (小米摄像头、萤石云)
- 智能门锁 App (Aqara、鹿客)

---

### 问题 2: 视觉功能未在 Web Client 中体现 ⭐⭐⭐⭐

**当前状态**:
- ✅ 后端已实现视觉 AI 功能:
  - `vision_answer()`: 通用视觉问答
  - `check_homework()`: 作业检查
  - `capture_to_feishu()`: 拍照推送飞书
- ❌ **Web Client 没有暴露这些功能的入口**
- ❌ **用户不知道可以使用摄像头进行作业检查**

**用户故事**:
> "作为一位家长，我希望点击'检查作业'按钮时，可以看到 AI 的批改结果显示在聊天框中，而不是只听语音反馈。这样我可以保存截图分享给老师。"

**改进方案**:

#### 新增功能区 (视觉助手):
```html
<!-- 在 Web Client 中新增一个折叠面板 -->
<div class="vision-panel">
    <h3>📸 视觉助手</h3>
    <button onclick="captureAndAsk('这是什么')">📷 拍照识别</button>
    <button onclick="checkHomework()">📚 检查作业</button>
    <button onclick="sendToFeishu()">✈️ 推送飞书</button>

    <!-- 展示最新的视觉结果 -->
    <div id="vision-result">
        <img id="captured-image" src="" alt="最新拍摄">
        <p id="vision-text"></p>
    </div>
</div>
```

#### 交互流程优化:
1. 用户点击"检查作业" → 发送文本指令到 Agent
2. Agent 调用 `check_homework()` → 返回文本结果
3. Web Client 通过 Data Channel 接收结果 → **同时在聊天框和视觉结果区展示**
4. 可选：展示拍摄的照片缩略图 (Base64 编码或 LiveKit 截图)

---

### 问题 3: 运动控制功能用户体验差 ⭐⭐⭐⭐

**当前状态**:
- ✅ 后端支持 `play_recording()` 和 `move_joint()` 控制运动
- ❌ **Web Client 只能通过文字输入"播放动画 xxx"**
- ❌ **没有可视化的动画列表**
- ❌ **没有关节控制面板 (摇杆、滑块)**

**用户故事**:
> "我的孩子只有 5 岁，他只会点按钮，不会打字。我希望 App 上有'摇头'、'点头'、'跳舞'这样的动画按钮，孩子一按就能让台灯动起来。"

**改进方案**:

#### 新增动画库面板:
```html
<div class="animation-panel">
    <h3>🎭 动作表情</h3>
    <div class="animation-grid">
        <button onclick="playAnim('nod')">👍 点头</button>
        <button onclick="playAnim('shake')">👎 摇头</button>
        <button onclick="playAnim('excited')">🎉 兴奋</button>
        <button onclick="playAnim('sleep')">😴 睡觉</button>
        <button onclick="playAnim('dance')">💃 跳舞</button>
        <button onclick="playAnim('think')">🤔 思考</button>
    </div>
</div>
```

#### 关节手动控制 (高级用户):
```html
<details>
    <summary>🎮 专家模式 - 关节控制</summary>
    <div class="joint-control">
        <label>底座旋转 (base_yaw):
            <input type="range" min="-180" max="180" value="0"
                   oninput="moveJoint('base_yaw', this.value)">
            <span id="yaw-value">0°</span>
        </label>
        <!-- 其他 4 个关节的滑块... -->
    </div>
</details>
```

**优先级**: 动画按钮 > 关节滑块 (儿童用户为主)

---

### 问题 4: RGB 灯光效果未可视化 ⭐⭐⭐

**当前状态**:
- ✅ 后端支持 17 种 RGB 灯效:
  - 纯色、呼吸灯、彩虹、波浪、火焰、烟花、星空...
- ❌ **Web Client 只有"开灯/关灯"按钮**
- ❌ **无法让用户探索台灯的灯光能力**

**改进方案**:

#### 灯光调色板:
```html
<div class="rgb-panel">
    <h3>💡 灯光魔法</h3>

    <!-- 颜色选择器 -->
    <div class="color-picker">
        <input type="color" id="light-color" value="#FFB6C1">
        <button onclick="setRGB()">设置纯色</button>
    </div>

    <!-- 预设效果 -->
    <div class="effect-grid">
        <button onclick="rgbEffect('breathing')">💗 呼吸灯</button>
        <button onclick="rgbEffect('rainbow')">🌈 彩虹</button>
        <button onclick="rgbEffect('wave')">🌊 波浪</button>
        <button onclick="rgbEffect('fire')">🔥 火焰</button>
        <button onclick="rgbEffect('fireworks')">🎆 烟花</button>
        <button onclick="rgbEffect('starry')">⭐ 星空</button>
    </div>
</div>
```

**商业价值**:
- 增加用户粘性 (每天换一种灯效)
- 可作为订阅会员的增值功能 (基础版 3 种灯效，高级版解锁全部)

---

### 问题 5: 缺少离线降级策略 ⭐⭐⭐

**当前状态**:
- ❌ **完全依赖 LiveKit 云服务**
- ❌ **如果网络断开或 LiveKit 服务故障，App 完全不可用**

**改进方案**:

#### 局域网直连模式 (Fallback):
```plaintext
┌────────────────────────────────────────────────────┐
│ 1. App 和台灯在同一 Wi-Fi 下                        │
│ 2. 台灯启动本地 WebSocket 服务器 (端口 8765)        │
│ 3. App 尝试连接 LiveKit 失败后，自动降级到 WS 模式  │
│ 4. 通过 WS 发送控制指令 (开灯/关灯/播放动画)        │
│ 5. 功能受限：无视频流、无语音，仅基础控制            │
└────────────────────────────────────────────────────┘
```

**参考案例**:
- 小米智能家居 (局域网优先 + 云端备用)
- Tesla App (蓝牙直连 + 互联网云控)

---

### 问题 6: 隐私保护需要产品化 ⭐⭐⭐⭐⭐

**当前状态**:
- ✅ 后端已实现 `CameraPrivacyManager` 类:
  - LED 指示灯 (红灯呼吸 = 摄像头激活)
  - 用户同意机制 (consent timeout)
- ❌ **用户在 App 上看不到隐私状态**
- ❌ **无法一键禁用摄像头**

**改进方案**:

#### 隐私仪表盘:
```html
<div class="privacy-dashboard">
    <h3>🔒 隐私控制</h3>

    <!-- 摄像头状态指示 -->
    <div class="camera-status">
        <span class="status-indicator" id="camera-led"></span>
        <span id="camera-status-text">摄像头已关闭</span>
    </div>

    <!-- 快速开关 -->
    <label class="switch">
        <input type="checkbox" id="camera-toggle" onchange="toggleCamera()">
        <span class="slider">允许使用摄像头</span>
    </label>

    <!-- 使用统计 -->
    <details>
        <summary>本月使用统计</summary>
        <ul>
            <li>启用次数: 23 次</li>
            <li>累计时长: 1 小时 47 分钟</li>
            <li>最近使用: 2026-03-16 09:30</li>
        </ul>
    </details>
</div>
```

**法律合规**:
- 符合 GDPR / CCPA 要求 (用户知情权 + 撤回权)
- 产品说明书中必须明示摄像头用途
- App Store / Google Play 上架时必须在隐私政策中声明

---

### 问题 7: 缺少错误处理和用户反馈 ⭐⭐⭐

**当前状态**:
- ❌ 连接失败时只有 `alert('连接失败')`
- ❌ 功能执行失败时用户不知道发生了什么
- ❌ 没有加载状态指示 (Spinner/Progress Bar)

**改进方案**:

#### 全局 Toast 通知系统:
```javascript
function showToast(message, type = 'info') {
    // type: 'success' | 'error' | 'warning' | 'info'
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
}

// 使用示例:
showToast('✅ 灯光已设置为彩虹模式', 'success');
showToast('❌ 动作执行失败：电机过热，请稍后再试', 'error');
```

#### 加载状态:
- 连接 LiveKit 时显示 Spinner
- 检查作业时显示"AI 批改中..."
- OTA 更新时显示进度条 (0% → 100%)

---

## 📱 移动端 App 特有需求

### 1. 推送通知 (Push Notification)
**场景**:
- 孩子完成作业后 → 推送"小明已完成数学作业，正确率 85%"
- 台灯电量低于 20% → 推送"台灯电量不足，请充电"
- 固件更新可用 → 推送"LeLamp 1.2.0 版本发布"

**技术实现**:
- iOS: Apple Push Notification Service (APNs)
- Android: Firebase Cloud Messaging (FCM)
- 后端: Supabase Realtime + Edge Functions

### 2. 本地相册集成
**场景**:
> "我想让台灯帮我看看相册里的这张照片是什么花。"

**实现**:
- 允许用户从相册选择照片 → 上传到台灯 → 调用 `vision_answer()`
- 类似微信"从相册选择"功能

### 3. 语音唤醒 (Voice Wake-up)
**场景**:
> "在 App 打开的情况下,用户说'嘿,小台灯'就能开始对话,而不需要点按钮。"

**技术**:
- iOS: Speech Framework
- Android: SpeechRecognizer API
- 关键词检测 (wake word detection): "嘿小台灯" / "LeLamp"

### 4. 家长控制 (Parental Control)
**场景**:
- 设置"勿扰时间": 晚上 10 点到早上 7 点台灯自动静音
- 限制使用时长: 每天最多使用 2 小时
- 内容过滤: 禁止访问某些话题 (通过 LLM System Prompt 控制)

---

## 💰 商业化价值提升建议

### 1. 订阅制会员体系
| 功能 | 免费版 | 高级版 (¥19.9/月) |
|------|--------|-------------------|
| 局域网控制 | ✅ | ✅ |
| 远程视频查看 | ❌ | ✅ |
| AI 作业检查 | 每天 3 次 | 无限次 |
| 云端录制回放 | ❌ | ✅ (保存 7 天) |
| 高级灯效 | 3 种 | 17 种全解锁 |
| 自定义动画 | ❌ | ✅ (上传自己的动作) |

### 2. 增值服务
- **学习报告**: 每周生成孩子的学习数据分析 (作业完成率、知识点薄弱项)
- **家庭相册**: 自动整理台灯拍摄的精彩瞬间 (孩子笑脸、获奖证书)
- **多设备联动**: 一个账号管理多台台灯 (客厅 + 卧室 + 书房)

### 3. 硬件捆绑销售
- **台灯 + 会员套餐**: 购买台灯赠送 3 个月高级会员
- **家庭套装**: 2 台台灯 + 1 年会员 (优惠 15%)

---

## 🚀 开发优先级建议 (OKR 模式)

### Q2 2026 (当前季度)
**目标**: 完成 MVP → Alpha 测试版本

| 优先级 | 任务 | 预计工时 | 负责人 |
|--------|------|---------|--------|
| P0 | 实现用户账号体系 (手机号注册 + 验证码登录) | 5 天 | 后端 |
| P0 | 设备绑定流程 (扫码 + 云端验证) | 3 天 | 全栈 |
| P0 | Token 自动生成 API | 2 天 | 后端 |
| P1 | Web Client 新增视觉功能区 | 2 天 | 前端 |
| P1 | Web Client 新增动画按钮面板 | 1 天 | 前端 |
| P1 | 错误处理 + Toast 通知系统 | 1 天 | 前端 |
| P2 | 隐私仪表盘界面 | 1 天 | 前端 |

### Q3 2026
**目标**: 移动端 App Beta 版 + 商业化准备

- Flutter 工程搭建
- iOS/Android 双端适配
- 推送通知集成
- 支付系统对接 (微信支付 / Apple Pay)
- 应用商店上架准备 (资质、隐私政策、审核)

---

## 📝 总结与行动项

### 🎯 核心问题
1. **安全性**: 缺少用户身份验证,存在严重隐私风险 (P0 必须解决)
2. **可用性**: 功能强大但入口隐藏,用户不知道可以做什么 (P1 优先优化)
3. **商业化**: 没有付费订阅能力,无法变现 (P1 规划路径)

### ✅ 下一步行动
1. **本周**: 召开产品评审会,确认优先级和排期
2. **下周**: 启动 P0 任务 (用户体系 + 设备绑定)
3. **2 周后**: 发布 Web Client v2.0 (新增视觉功能 + 动画面板)
4. **1 个月后**: 启动 Flutter App 开发

---

## 附录: 竞品参考

### 类似产品
- **小米智能台灯 Pro**: App 功能简单但稳定 (色温调节 + 定时开关)
- **Amazon Echo Show**: 视频通话 + 智能助手 (参考其隐私设计)
- **Anki Vector**: 机器人台灯,有丰富的表情动画库

### 学习要点
- **小米**: 极致的稳定性和快速响应 (局域网优先策略)
- **Amazon**: 清晰的隐私指示 (摄像头物理开关 + LED 常亮)
- **Anki**: 趣味性动画设计 (让机器人有生命力)

---

> **文档版本**: v1.0
> **作者**: Claude (Product Manager Mode)
> **最后更新**: 2026-03-16
