# LeLamp 用户快速使用指南

**最后更新**: 2026-04-01
**适用版本**: LeLamp Runtime v0.1.0

> 💡 **新用户?** 如果您是第一次使用，请先阅读 [完整设置指南](SETUP_GUIDE.md)

---

## 🎯 快速开始（已配置设备）

如果您的 LeLamp 已经完成配置，直接开始使用：

### 方式一：Web 界面控制（推荐）

1. **打开浏览器**，访问：
   - mDNS: `http://lelamp.local:8000`（推荐，无需知道 IP）
   - 局域网: `http://<台灯IP>:8000`
   - 开发模式: `http://<台灯IP>:5173`
2. **点击"连接设备"**
3. **开始使用**：
   - 💬 文字聊天
   - 🎭 动作控制
   - 💡 灯光调节
   - 👁️ 拍照识别

### 方式二：语音对话

1. **确保语音服务已启动**
2. **直接对台灯说话**
3. **LED 状态指示**：
   - 🤍 白色：空闲
   - 🔵 蓝色：倾听中
   - 🟣 紫色：思考中
   - 🌈 彩色：说话中

---

## 📖 核心功能

### 1. 语音对话

**唤醒方式**：直接说话，无需特定唤醒词

**示例对话**：
```
你: "你好"
台灯: "你好呀，小主人！本灯又来陪你啦~"

你: "现在几点了"
台灯: "现在是下午 3:45，怎么，还没写完作业就想着玩了？"

你: "讲个笑话"
台灯: "好吧...为什么程序员总是搞混万圣节和圣诞节？因为 Oct 31 == Dec 25！"
```

### 2. 动作表情

**6 个预设动作**：
- 👍 **点头** (nod): 表示同意
- 👎 **摇头** (shake): 表示否定
- 🎉 **兴奋** (excited): 快速摆动
- 😴 **睡觉** (sleep): 缓慢低头
- 💃 **跳舞** (dance): 有节奏摆动
- 🤔 **思考** (think): 缓慢转动

**触发方式**：
- 语音指令："点个头"、"跳个舞"
- Web 界面按钮
- 对话自动触发

### 3. 灯光效果

**纯色灯光**：🔴 暖红 | 💖 粉红 | 🟠 橙色 | 🟡 金黄 | 🟢 浅绿 | 🔵 天蓝 | 🟣 紫色 | ⚪ 暖白

**灯效动画**：
- 💗 **呼吸灯**: 缓慢呼吸
- 🌈 **彩虹**: 彩虹流动
- 🌊 **波浪**: 波浪起伏
- 🔥 **火焰**: 火焰跳动
- 🎆 **烟花**: 烟花绽放
- ⭐ **星空**: 星空闪烁

**触发方式**：
- 语音指令："打开红色灯光"、"来个彩虹灯效"
- Web 界面调色盘

### 4. 视觉识别

**拍照识别**：
```
你: "这是什么？" (举起物体)
台灯: "这是一个红苹果，看起来很新鲜。"
```

**检查作业**：
```
你: "帮我检查作业"
台灯: "好的，让我看看..."

✅ 第1题: 5 + 3 = 8 (正确)
✅ 第2题: 12 - 7 = 5 (正确)
❌ 第3题: 6 × 4 = 32 (错误，应该是 24)

正确率: 67% (2/3)
```

**推送飞书**：
```
你: "拍照发送到飞书"
台灯: "好的，正在拍照..."
```

### 5. 系统管理

**音量控制**：
```
你: "把音量调到 80"
台灯: "好的，已设置音量为 80%"
```

**联网搜索**：
```
你: "今天天气怎么样？"
台灯: "今天北京晴，最高温度 15℃，最低 5℃。"
```

**OTA 更新**：
```
你: "检查更新"
台灯: "发现新版本 1.2.0！是否现在更新？"

你: "确认更新"
台灯: "更新成功！服务将在 5 秒后重启。"
```

---

## 🛠️ 服务管理

### 查看服务状态

```bash
# 查看所有服务
systemctl status lelamp-*

# 查看特定服务
systemctl status lelamp-api       # Web API 服务
systemctl status lelamp-livekit   # 语音服务
systemctl status lelamp-frontend  # Web 界面
```

### 重启服务

```bash
# 重启所有服务
systemctl restart lelamp-{api,livekit,frontend}

# 重启单个服务
systemctl restart lelamp-api
```

### 查看日志

```bash
# 实时日志
journalctl -u lelamp-api -f

# 最近 50 行
journalctl -u lelamp-api -n 50
```

---

## 🔧 常见问题

### Q1: Web 界面无法访问？

**检查清单**：
1. 确认台灯已开机（等待 30 秒启动）
2. 检查网络连接
3. 尝试 mDNS 访问：`http://lelamp.local:8000`
4. 或直接访问：`http://<台灯IP>:8000`
5. 如果仍无法访问，重启服务：
   ```bash
   systemctl restart lelamp-api
   ```

### Q2: 语音不工作？

**解决方案**：
1. 检查麦克风是否连接
2. 确认语音服务已启动：
   ```bash
   systemctl status lelamp-livekit
   ```
3. 重启语音服务：
   ```bash
   systemctl restart lelamp-livekit
   ```

### Q3: LED 灯不亮？

**可能原因**：
- GPIO 权限不足
- LED 硬件未连接

**解决方案**：
```bash
# 测试 LED（需要 sudo）
sudo -u pi uv run -m tests.test_rgb
```

### Q4: 如何获取台灯 IP 地址？

**方法 1: 路由器管理页面**
- 登录路由器，查找名为 `lelamp-local` 或 `raspberrypi` 的设备

**方法 2: 网络扫描工具**
- 使用 Fing App 扫描局域网

**方法 3: 直接查看**
- 连接显示器和键盘，执行：`hostname -I`

### Q5: 忘记 WiFi 密码？

**重新配置 WiFi**：
```bash
# 方法 1: Web 界面
访问 http://lelamp.local:8000/settings → WiFi 设置

# 方法 2: 命令行
nmcli device wifi connect "SSID" password "密码"

# 方法 3: 重新进入设置模式
sudo rm /var/lib/lelamp/setup_status.json
sudo reboot
```

---

## 📱 首次开箱配置（Captive Portal）

如果您是第一次使用 LeLamp，设备会自动进入配置模式：

### 配置流程

1. **连接热点**
   - WiFi 名称：`LeLamp-Setup`
   - 密码：每次随机生成（显示在 Captive Portal 欢迎页面）
   - LED 呈蓝色呼吸效果

2. **打开浏览器**
   - 访问任意网址（自动跳转）
   - 或访问：`http://192.168.4.1:8080`

3. **配置 WiFi**
   - 选择您的 WiFi 网络
   - 输入密码
   - 点击"连接"

4. **完成设置**
   - 等待连接成功
   - 记录显示的 IP 地址
   - 开始使用！

### 重新进入设置模式

```bash
# 在台灯上执行
sudo rm /var/lib/lelamp/setup_status.json
sudo reboot
```

---

## 🎨 高级功能

### 自定义动作

录制您自己的动作：

```bash
# 录制新动作
uv run -m lelamp.record --id lelamp --port /dev/ttyACM0 --name my_action

# 回放动作
uv run -m lelamp.replay --id lelamp --port /dev/ttyACM0 --name my_action
```

### 边缘视觉（本地 AI）

启用本地视觉识别（无需云端 API）：

```bash
# 在 .env 中配置
LELAMP_EDGE_VISION_ENABLED=true
LELAMP_EDGE_VISION_MODEL_DIR=/home/pi/lelamp_runtime/models
```

**⚠️ 平台支持**：
- ✅ macOS (Apple Silicon)
- ✅ Linux x86_64
- ✅ Windows AMD64
- ❌ Raspberry Pi ARM (需要手动编译)

**Raspberry Pi 用户**：
- 边缘视觉官方不支持 ARM 架构
- 可尝试手动编译（参考 [FEATURES.md](FEATURES.md)）
- 或使用云端视觉识别（自动降级）

**支持功能**（在支持平台上）：
- 人脸检测（< 50ms）
- 手势追踪（< 100ms）
- 物体识别（< 300ms）

### Web 设置页面

访问 Web 设置页面进行系统配置：

```
http://lelamp.local:8000/settings
```

或开发模式下：
```
http://<台灯IP>:5173/settings
```

**配置分类**：
- 📶 WiFi 网络
- 🤖 LLM 模型
- 👁️ 视觉识别
- 📹 摄像头设置
- 🎤 语音配置
- ⚙️ 硬件配置
- 🎭 行为配置

---

## 📊 性能指标

| 指标 | 数值 | 说明 |
|------|------|------|
| 视频延迟 | < 200ms | LiveKit WebRTC |
| 语音识别延迟 | < 500ms | Baidu Speech |
| 视觉识别延迟 | 3-8 秒 | Qwen VL API |
| 动作响应时间 | < 100ms | 本地电机控制 |
| 灯光响应时间 | < 50ms | 本地 LED 控制 |
| CPU 占用率 | 30-50% | Raspberry Pi 4B |
| 内存占用 | 800MB - 1.2GB | Python + Services |

---

## 🔗 相关链接

- 📘 [完整设置指南](SETUP_GUIDE.md) - 详细配置教程
- 🔧 [开发者指南](../CLAUDE.md) - 代码架构和开发
- 📖 [Captive Portal 指南](CAPTIVE_PORTAL_GUIDE.md) - 开箱配置
- 🐛 [问题反馈](https://github.com/xwang152-jack/lelamp_runtime/issues)
- 🌐 [项目主页](https://github.com/xwang152-jack/lelamp_runtime)

---

**文档版本**: v0.1.0
**最后更新**: 2026-04-01
**作者**: LeLamp 开发团队
**许可证**: 参见主项目许可证
