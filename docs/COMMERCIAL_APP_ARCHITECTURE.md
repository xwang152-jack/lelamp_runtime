# LeLamp 商业化 App 架构建议书

本文档旨在规划 LeLamp 配套用户端 App 的技术架构与功能路线，确保产品具备商业化交付能力。

## 1. 技术选型建议

### 跨平台移动端框架
- **首选方案**: **Flutter** (Google)
  - **理由**: 
    - 性能接近原生，UI 渲染一致性好。
    - LiveKit 官方提供高质量的 [flutter_livekit](https://pub.dev/packages/livekit_client) SDK。
    - 适合重交互、即时通讯类应用。
- **备选方案**: **React Native** (Meta)
  - **理由**: 
    - 如果团队熟悉 JavaScript/React，上手更快。
    - 生态极其丰富，但原生模块（如蓝牙、后台保活）调试成本略高。

### 后端服务 (BaaS)
- **Supabase / Firebase**: 提供用户认证 (Auth)、数据库 (PostgreSQL/NoSQL)、实时订阅功能。
- **LiveKit Cloud / Self-hosted**: 处理音视频流的核心服务。

---

## 2. 核心功能模块

### 2.1 设备配网 (Provisioning)
用户首次拿到设备时，需要将家里的 Wi-Fi 账号密码告诉台灯。
- **流程**:
  1. 台灯上电，长按配网键进入 **BLE 广播模式**。
  2. App 扫描附近的 LeLamp 设备。
  3. App 通过蓝牙 (GATT) 发送 Wi-Fi SSID 和 Password。
  4. 台灯连接 Wi-Fi，成功后向 App 反馈 IP 地址和 Device ID。
- **技术点**: Flutter Blue Plus / React Native BLE PLX。

### 2.2 用户与设备绑定
- **机制**:
  - 每台设备底部张贴 **二维码** (包含 Device ID 和出厂 Secret)。
  - App 扫描二维码，向云端服务器注册绑定关系 (User <-> Device)。
  - 云端验证 Secret 合法性，防止伪造设备接入。

### 2.3 远程控制与监控 (核心体验)
- **技术**: 基于 **LiveKit (WebRTC)**。
- **功能**:
  - **实时视频**: 查看台灯摄像头画面 (H.264/VP8)。
  - **双向语音**: 类似微信视频通话，用户可直接对孩子/家人说话。
  - **数据信道 (Data Channel)**: 发送控制指令（开灯/关灯/摇头/检查作业）。
    - 优势：低延迟 (<100ms)，比传统 MQTT 更快且不需要建立两条连接。

### 2.4 OTA 固件升级
- **流程**:
  1. App 查询云端是否有新固件版本。
  2. App 发送指令 `{"type": "ota_update", "version": "1.0.1"}` 给台灯。
  3. 台灯执行下载与重启（复用现有的 `ota.py` 逻辑）。
  4. App 接收台灯的升级进度推送。

---

## 3. 安全与隐私 (商业化必读)

### 3.1 视频流加密
- LiveKit 默认开启 **DTLS/SRTP** 加密，确保视频流在传输过程中无法被窃听。
- 建议开启 **E2EE (端到端加密)**:
  - 仅 App 和台灯持有密钥，甚至连 LiveKit 服务器都无法解码视频内容。
  - 这是家庭隐私产品的核心卖点。

### 3.2 访问控制
- **Token 机制**: 
  - 台灯和 App 都不保存永久凭证。
  - 每次连接前，向您的业务服务器请求一次性 Access Token。
  - 业务服务器校验用户权限（是否是设备的主人）后签发 Token。

---

## 4. 商业模式扩展 (SaaS)
- **基础版 (免费)**: 
  - 局域网控制、基础语音对话。
- **高级订阅版 (订阅制)**:
  - **远程看家**: 允许公网远程查看视频。
  - **AI 辅导**: 接入更高级的 DeepSeek/GPT-4 模型进行作业检查。
  - **云存储**: 录制精彩时刻（如孩子解出难题的瞬间）上传云端。

---

## 5. 开发路线图 (Roadmap)

| 阶段 | 目标 | 关键产出 |
| :--- | :--- | :--- |
| **Phase 1 (MVP)** | 跑通核心链路 | Web 客户端 (已完成原型), 局域网 Token 生成 |
| **Phase 2 (Alpha)** | 移动端 Demo | Flutter App 工程搭建, 集成 LiveKit SDK, 实现视频通话 |
| **Phase 3 (Beta)** | 设备管理 | 蓝牙配网功能, 扫码绑定, 用户账户体系 |
| **Phase 4 (Release)** | 商业化交付 | 支付系统集成, OTA 完整闭环, 应用商店上架 |

---

> 此文档生成于 LeLamp Runtime 项目开发过程。
> 关联代码: `web/`, `scripts/generate_client_token.py`, `lelamp/utils/ota.py`
