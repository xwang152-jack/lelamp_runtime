# LeLamp 项目从零配置教学指南

**最后更新**: 2026-03-18
**适用人群**: 首次接触 LeLamp 项目的开发者
**预计完成时间**: 45-60 分钟（含树莓派初始设置）

---

## 目录

1. [配置前置检查清单](#配置前置检查清单)
2. [阶段零：树莓派初始设置](#阶段零树莓派初始设置)
3. [阶段一：环境准备](#阶段一环境准备)
4. [阶段二：获取 API 密钥](#阶段二获取-api-密钥)
5. [阶段三：环境变量配置](#阶段三环境变量配置)
6. [阶段四：硬件配置](#阶段四硬件配置首次使用)
7. [阶段五：启动服务](#阶段五启动服务)
8. [阶段六：验证测试](#阶段六验证测试)
9. [常见配置问题](#常见配置问题)
10. [验证清单](#验证清单)

---

## 配置前置检查清单

在开始配置之前，请确认您已准备好以下硬件和账号：

### 硬件要求

| 项目 | 要求说明 | 备注 |
|------|---------|------|
| [ ] **树莓派** | Raspberry Pi 4B+ (4GB RAM 推荐) | Pi 3B+ 也可使用，但性能较低 |
| [ ] **LeLamp 硬件** | 5轴电机套件 + LED矩阵 + 摄像头 | 确保所有硬件已正确连接 |
| [ ] **网络连接** | WiFi 或有线网络 | 需要稳定的互联网连接 |
| [ ] **电源** | 5V 3A USB-C 电源适配器 | 供电不足会导致系统不稳定 |

### 账号和服务准备

您需要注册以下服务并获取 API 密钥：

| 服务 | 用途 | 费用 | 注册链接 |
|------|------|------|---------|
| [ ] **DeepSeek** | LLM 对话引擎 | 按量付费 (新用户有免费额度) | https://platform.deepseek.com |
| [ ] **Baidu Speech** | 语音识别和合成 | 按量付费 (有免费额度) | https://cloud.baidu.com/product/speech/tts |
| [ ] **LiveKit** | 实时音视频通信 | 免费额度 50GB/月 | https://cloud.livekit.org |
| [ ] **ModelScope** | 视觉识别 (可选) | 按量付费 | https://modelscope.cn |

### 软件工具

请确保您的开发机和树莓派已安装以下软件：

| 工具 | 版本要求 | 检查命令 | 安装链接 |
|------|---------|---------|---------|
| [ ] **Python** | 3.12+ | `python3 --version` | https://www.python.org |
| [ ] **UV** | 最新版 | `uv --version` | https://astral.sh/uv |
| [ ] **Git** | 任意版本 | `git --version` | `sudo apt install git` |
| [ ] **Node.js** | 18+ (前端开发) | `node --version` | https://nodejs.org |

---

## 阶段零：树莓派初始设置

> **适用场景**：首次使用全新或已重置的树莓派
> **预计时间**：15-20 分钟

本阶段将指导您完成树莓派的初始设置，包括操作系统安装、首次启动、获取 IP 地址和 WiFi 配置。这些是所有后续步骤的前提条件。

### 0.1 烧录树莓派操作系统

**所需材料**:
- Micro SD 卡（16GB 以上，Class 10 推荐）
- SD 卡读卡器
- 电脑（Windows/Mac/Linux）

**步骤**:

#### 步骤 1：下载 Raspberry Pi Imager

访问官网下载: https://www.raspberrypi.com/software/

支持 Windows、macOS 和 Ubuntu。

#### 步骤 2：选择操作系统

1. 插入 SD 卡到电脑
2. 打开 Raspberry Pi Imager
3. 点击 "选择操作系统" (Choose OS)
4. 推荐选择: **Raspberry Pi OS (64-bit)** - Bookworm 版本
5. 如果不需要桌面环境，可选择 **Raspberry Pi OS Lite**（更轻量）

#### 步骤 3：选择存储设备

点击 "选择存储设备" (Choose Storage)，然后选择您的 SD 卡。

#### 步骤 4：高级设置（重要！）

在开始烧录前，点击**齿轮图标**（或按 `Ctrl+Shift+X`）进行预配置：

```
设置主机名:     lelamp-local
启用 SSH:       使用密码认证
用户名:         pi（或自定义）
密码:           自行设置并记住
配置 WiFi:      输入 SSID 和密码（如果已知）
WiFi 国家:      CN China
时区设置:       Asia/Shanghai
键盘布局:       us
```

**为什么要配置这些？**
- **主机名**：方便在路由器中识别设备
- **SSH**：允许远程登录，无需连接显示器和键盘
- **WiFi**：开机即联网，省去后续配置

#### 步骤 5：开始烧录

点击 "写入" (Write) 开始烧录。烧录过程可能需要 5-15 分钟，取决于 SD 卡速度。

烧录完成后，取出 SD 卡并插入树莓派。

### 0.2 首次启动树莓派

#### 方式 A：带显示器和键盘（推荐新手）

1. 将 SD 卡插入树莓派
2. 连接 HDMI 显示器、USB 键盘
3. 插入电源启动（USB-C 电源适配器）
4. 首次启动会进入配置向导
5. 完成基本设置后进入桌面或命令行

#### 方式 B：无头模式（Headless，通过 SSH）

1. 插入 SD 卡和电源
2. 等待约 1-2 分钟启动
3. 跳转到 [0.3 获取 IP 地址](#03-获取树莓派-ip-地址)

### 0.3 获取树莓派 IP 地址

获取 IP 地址是后续 SSH 连接和访问 Web 界面的关键步骤。以下提供多种方法：

#### 方法 1：路由器管理页面（最简单）

1. 登录路由器管理页面
   - 常见地址: `192.168.1.1` 或 `192.168.0.1` 或 `192.168.31.1`
   - 查看路由器背面的标签获取默认地址
2. 找到"已连接设备"、"DHCP客户端列表"或"终端设备"
3. 查找名为 `lelamp-local` 或 `raspberrypi` 的设备
4. 记录其 IP 地址（如 `192.168.1.100`）

**优点**：无需额外工具，最直观
**缺点**：需要登录路由器

#### 方法 2：网络扫描工具

在您的电脑上运行以下命令：

```bash
# 使用 nmap 扫描局域网（需先安装: brew install nmap 或 apt install nmap）
nmap -sn 192.168.1.0/24

# 或使用 arp-scan（Linux）
sudo arp-scan --localnet

# 或使用特定工具（如 Fing App、Angry IP Scanner）
```

查找包含 "Raspberry Pi" 或 MAC 地址以 `b8:27:eb` 开头的设备。

**优点**：技术友好，信息详细
**缺点**：需要安装额外工具

#### 方法 3：HDMI 显示器直接查看

1. 连接显示器和键盘到树莓派
2. 登录系统（用户名 `pi`，密码为您设置的密码）
3. 在终端输入：

```bash
# 方法 1：显示所有 IP 地址
ip addr show

# 方法 2：仅显示 IP 地址
hostname -I

# 方法 3：查看特定网络接口
ip addr show wlan0   # WiFi
ip addr show eth0    # 有线网络
```

**输出示例**:
```
pi@lelamp-local:~$ hostname -I
192.168.1.100 fe80::1
```

**优点**：最直接，无需猜测
**缺点**：需要显示器和键盘

#### 方法 4：串口调试（高级用户）

使用 USB-TTL 串口模块连接树莓派 GPIO 的 UART 引脚：

**引脚连接**:
- USB-TTL GND → GPIO Pin 6 (GND)
- USB-TTL TXD → GPIO Pin 10 (RXD)
- USB-TTL RXD → GPIO Pin 8 (TXD)

使用 PuTTY（Windows）或 screen/minicom（Mac/Linux）连接：

```bash
# Mac/Linux
screen /dev/tty.usbserial-xxx 115200

# 或使用 minicom
minicom -D /dev/tty.usbserial-xxx -b 115200
```

**优点**：即使网络配置错误也能访问
**缺点**：需要额外硬件，接线复杂

### 0.4 SSH 连接到树莓派

获取 IP 地址后，即可通过 SSH 远程连接树莓派。

#### 从 Mac/Linux 终端

```bash
ssh pi@192.168.1.100  # 替换为实际 IP
```

#### 从 Windows (PowerShell 或 CMD)

```powershell
ssh pi@192.168.1.100
```

#### 从 Windows (使用 PuTTY)

1. 下载 PuTTY: https://www.putty.org/
2. Host Name 输入: `192.168.1.100`
3. Port: `22`
4. 点击 "Open"

#### 首次连接确认

第一次连接时会看到以下提示：

```
The authenticity of host '192.168.1.100' can't be established.
ECDSA key fingerprint is SHA256:xxx...
Are you sure you want to continue connecting (yes/no)?
```

输入 `yes` 并按回车确认。

然后输入密码（如果您在烧录时设置了密码，使用您设置的密码；默认是 `raspberry`）。

**连接成功的标志**:
```
pi@lelamp-local:~$
```

### 0.5 配置 WiFi（如果烧录时未配置）

如果您在烧录时没有配置 WiFi，或者需要更换网络，可以使用以下方法：

#### 命令行方式 (nmcli，推荐)

```bash
# 1. 扫描可用网络
nmcli device wifi list

# 2. 连接到 WiFi（替换 SSID 和密码）
sudo nmcli device wifi connect "YourWiFiSSID" password "YourPassword"

# 3. 验证连接
nmcli connection show
ip addr show wlan0
```

**输出示例**:
```
pi@lelamp-local:~$ nmcli connection show
NAME    UUID                                  TYPE      DEVICE
MyWiFi  xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx  wifi      wlan0
```

#### 配置文件方式 (wpa_supplicant)

```bash
# 1. 编辑 wpa_supplicant 配置
sudo nano /etc/wpa_supplicant/wpa_supplicant.conf

# 2. 在文件末尾添加网络配置
network={
    ssid="YourWiFiSSID"
    psk="YourPassword"
    key_mgmt=WPA-PSK
}

# 3. 保存（Ctrl+X，然后 Y，然后回车）

# 4. 重启网络服务
sudo systemctl restart networking
# 或重启树莓派
sudo reboot
```

#### 重启后验证

```bash
# 查看无线网络状态
iwconfig wlan0

# 查看 IP 地址
ip addr show wlan0

# 测试网络连通性
ping -c 4 8.8.8.8

# 测试 DNS 解析
ping -c 4 baidu.com
```

**预期输出**:
```
pi@lelamp-local:~$ ping -c 4 8.8.8.8
PING 8.8.8.8 (8.8.8.8) 56(84) bytes of data.
64 bytes from 8.8.8.8: icmp_seq=1 ttl=117 time=12.3 ms
64 bytes from 8.8.8.8: icmp_seq=2 ttl=117 time=11.8 ms
...
```

### 0.6 更新系统（推荐）

在继续之前，建议更新系统到最新版本：

```bash
# 更新软件包列表
sudo apt update

# 升级已安装的软件包
sudo apt upgrade -y

# 安装必要工具
sudo apt install -y git curl vim networkmanager

# 清理不需要的包
sudo apt autoremove -y
```

**预计时间**: 5-10 分钟（取决于网络速度和更新数量）

### 0.7 配置静态 IP（可选）

如果您希望树莓派使用固定的 IP 地址，方便后续访问：

#### 使用 nmcli 配置静态 IP

```bash
# 获取当前连接名称
nmcli connection show

# 修改连接为静态 IP（替换 YourWiFiSSID 和实际 IP）
sudo nmcli connection modify "YourWiFiSSID" \
    ipv4.addresses 192.168.1.100/24 \
    ipv4.gateway 192.168.1.1 \
    ipv4.dns "8.8.8.8 8.8.4.4" \
    ipv4.method manual

# 重启网络
sudo systemctl restart NetworkManager
# 或重启树莓派
sudo reboot
```

**注意事项**:
- 确保选择的 IP 地址没有被其他设备使用
- 确保网关地址与您的路由器地址匹配
- 如果使用有线网络，将连接名称替换为 `Wired connection 1` 或类似名称

### 阶段零完成检查

确认以下项目已完成后再进入下一阶段：

- [ ] SD 卡已烧录 Raspberry Pi OS
- [ ] 树莓派成功启动
- [ ] 已获取树莓派 IP 地址
- [ ] SSH 连接成功
- [ ] WiFi 已配置并连接
- [ ] 系统已更新到最新版本
- [ ] （可选）静态 IP 已配置

**完成！** 现在您可以继续执行 [阶段一：环境准备](#阶段一环境准备)

---

## 阶段一：环境准备

### 1.1 克隆项目

**在树莓派上执行**:

```bash
# 进入用户主目录
cd ~

# 克隆项目仓库
git clone https://github.com/xwang152-jack/lelamp_runtime.git

# 进入项目目录
cd lelamp_runtime

# 查看项目结构
ls -la
```

**预期输出**:
```
total 120
drwxr-xr-x  38 pi   staff   1216 Mar 18 11:15 .
drwxr-xr-x   5 pi   staff    160 Mar 18 10:00 ..
-rw-r--r--   1 pi   staff   8234 Mar 18 10:00 CLAUDE.md
-rw-r--r--   1 pi   staff   2341 Mar 18 10:00 README.md
-rw-r--r--   1 pi   staff   5678 Mar 18 10:00 pyproject.toml
-rw-r--r--   1 pi   staff   7890 Mar 18 10:00 .env.example
drwxr-xr-x   5 pi   staff    160 Mar 18 10:00 lelamp/
drwxr-xr-x   5 pi   staff    160 Mar 18 10:00 web/
drwxr-xr-x   5 pi   staff    160 Mar 18 10:00 scripts/
...
```

### 1.2 安装 UV 包管理器

UV 是一个快速的 Python 包管理器，可以大大加快依赖安装速度。

```bash
# 安装 UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# 重新加载 shell 配置
source ~/.bashrc
# 或
source ~/.zshrc

# 验证安装
uv --version
```

**预期输出**:
```
uv 0.1.20 (或更高版本)
```

### 1.3 安装 Python 依赖

LeLamp 项目支持两种安装模式：

#### 在树莓派上（包含硬件依赖）

```bash
cd ~/lelamp_runtime

# 安装所有依赖（包括硬件相关的库）
uv sync --extra hardware
```

#### 在开发机上（仅电机控制，无硬件）

```bash
cd ~/lelamp_runtime

# 仅安装软件依赖
uv sync
```

**注意**: 如果遇到 Git LFS 相关问题，使用以下命令：

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

### 1.4 验证安装

```bash
# 检查 Python 版本
uv run python --version
```

**预期输出**:
```
Python 3.12.x
```

---

## 阶段二：获取 API 密钥

### 2.1 DeepSeek API Key

DeepSeek 提供强大的 LLM 对话能力。

**注册步骤**:

1. 访问 https://platform.deepseek.com
2. 点击右上角 "注册" 按钮
3. 使用邮箱或手机号完成注册
4. 登录后进入控制台
5. 点击左侧菜单 "API Keys"
6. 点击 "创建新的 API Key"
7. 复制生成的 API Key (格式: `sk-xxxxxx`)
8. **妥善保管** - 离开页面后将无法再次查看完整密钥

**API Key 示例**:
```
sk-1234567890abcdef1234567890abcdef
```

### 2.2 Baidu Speech API Key

百度语音提供语音识别 (STT) 和语音合成 (TTS) 服务。

**注册步骤**:

1. 访问 https://cloud.baidu.com/product/speech/tts
2. 点击 "立即使用"
3. 登录或注册百度智能云账号
4. 进入 "语音技术" 控制台
5. 开通 "语音识别" 和 "语音合成" 服务
6. 创建应用，获取以下信息：
   - **API Key**
   - **Secret Key**

**保存以下信息**:
```
API Key:    24 characters (字母+数字)
Secret Key: 32 characters (字母+数字)
```

### 2.3 LiveKit 配置

LiveKit 提供实时音视频通信能力。

**注册步骤**:

1. 访问 https://cloud.livekit.org
2. 点击 "Start Building" 或 "Sign Up"
3. 使用 GitHub 或邮箱注册
4. 创建新项目 (Project)
5. 在项目设置页面获取：
   - **WebSocket URL** (格式: `wss://xxx.livekit.cloud`)
   - **API Key**
   - **API Secret**

**保存以下信息**:
```
WebSocket URL: wss://xxxxxxxx-xxxx-xxxx.livekit.cloud
API Key:       APIxxxxxxxxxxxx (以 API 开头)
API Secret:    32 characters (字母+数字)
```

### 2.4 ModelScope API Key (可选)

ModelScope 提供视觉识别能力，用于拍照识别和检查作业功能。

**注册步骤**:

1. 访问 https://modelscope.cn
2. 点击右上角 "注册"
3. 完成注册并登录
4. 进入 "个人中心" → "访问令牌"
5. 创建新的访问令牌
6. 复制生成的 API Key

**API Key 示例**:
```
xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

---

## 阶段三：环境变量配置

### 3.1 创建 .env 文件

```bash
cd ~/lelamp_runtime

# 复制配置模板
cp .env.example .env

# 编辑配置文件
nano .env
```

### 3.2 必填配置（最小可运行版本）

以下是让 LeLamp 正常运行的最小配置：

```bash
# ============================================================================
# 必填配置 - 请填入您在阶段二获取的实际值
# ============================================================================

# LiveKit - 实时通信
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=dev_xxx
LIVEKIT_API_SECRET=xxx

# DeepSeek - LLM 对话
DEEPSEEK_API_KEY=sk-xxx

# Baidu Speech - 语音服务
BAIDU_SPEECH_API_KEY=xxx
BAIDU_SPEECH_SECRET_KEY=xxx

# 开发模式（跳过授权检查）
LELAMP_DEV_MODE=1
```

### 3.3 可选配置（增强功能）

在完成必填配置后，您可以添加以下可选配置：

#### 视觉识别功能

```bash
# ModelScope - 视觉识别
MODELSCOPE_API_KEY=xxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

#### 摄像头参数调整

```bash
# 摄像头分辨率
LELAMP_CAMERA_WIDTH=1024
LELAMP_CAMERA_HEIGHT=768

# 摄像头旋转（如果图像倒置）
LELAMP_CAMERA_ROTATE_DEG=180
```

#### LED 亮度调整

```bash
# LED 亮度 (0-255，默认 25)
LELAMP_LED_BRIGHTNESS=50
```

#### 联网搜索功能

```bash
# Bocha API - 网络搜索
BOCHA_API_KEY=your_bocha_api_key
```

### 3.4 保存并验证配置

```bash
# 按 Ctrl+X 保存并退出 nano 编辑器

# 验证配置文件已创建
ls -la .env

# 查看配置内容（确保没有多余空格或引号）
cat .env
```

**注意事项**:
- ✅ API Key 值**不要加引号**
- ✅ 确保 API Key **前后没有空格**
- ✅ 注释行以 `#` 开头
- ❌ **切勿提交 .env 文件到 Git**

---

## 阶段四：硬件配置（首次使用）

### 4.1 查找串口设备

LeLamp 通过 USB 串口控制电机。首先确认串口设备：

```bash
cd ~/lelamp_runtime

# 查找串口设备
uv run lerobot-find-port
```

**预期输出**:
```
Searching for Feetech servos...
Found port: /dev/ttyACM0
Servo count: 5
```

**如果找不到设备**:
1. 检查 USB 连接是否稳固
2. 确认电机驱动板已通电
3. 尝试重新插拔 USB 线

### 4.2 添加串口权限

默认情况下，普通用户无法访问串口设备。需要添加用户到 `dialout` 组：

```bash
# 将当前用户添加到 dialout 组
sudo usermod -a -G dialout $USER

# 添加用户到 video 组（摄像头权限）
sudo usermod -a -G video $USER

# 查看当前用户所属组
groups
```

**重要**: 修改组权限后，需要**重新登录**才能生效。

```bash
# 重新登录或重启
sudo reboot
```

### 4.3 设置电机 ID

每个 LeLamp 设备需要设置唯一的设备 ID：

```bash
# 设置电机 ID 为 "lelamp"
uv run -m lelamp.setup_motors --id lelamp --port /dev/ttyACM0
```

**预期输出**:
```
Setting up motors for lamp_id=lelamp on port=/dev/ttyACM0
Motor 1 (base_yaw): ID=1
Motor 2 (base_pitch): ID=2
Motor 3 (elbow_pitch): ID=3
Motor 4 (wrist_roll): ID=4
Motor 5 (wrist_pitch): ID=5
Setup complete!
```

### 4.4 校准电机

首次使用前，需要对电机进行校准，确保所有关节回到原点：

```bash
# 校准电机（需要 sudo 权限）
sudo uv run -m lelamp.calibrate --id lelamp --port /dev/ttyACM0
```

**预期输出**:
```
Calibrating motors for lamp_id=lelamp...
Motor 1: Moving to 0°... OK
Motor 2: Moving to 0°... OK
Motor 3: Moving to 0°... OK
Motor 4: Moving to 0°... OK
Motor 5: Moving to 0°... OK
Calibration complete!
```

**注意**: 校准过程中电机会产生动作，请确保台灯周围有足够空间。

---

## 阶段五：启动服务

### 5.1 启动 LeLamp Agent

LeLamp Agent 是主程序，负责处理语音对话、视觉识别和控制硬件。

```bash
cd ~/lelamp_runtime

# 启动 Agent（需要 sudo 权限用于 LED 控制）
sudo uv run main.py console
```

**预期输出**:
```
INFO:root:config ready: lamp_id=lelamp port=/dev/ttyACM0 vision=True camera=0
INFO:lelamp.service.motors:MotorsService started
INFO:lelamp.service.rgb:RGBService started
INFO:lelamp.service.vision:VisionService started
INFO:livekit:Connected to LiveKit
INFO:root:LeLamp agent is running...
```

**启动成功的标志**:
- ✅ 所有服务启动成功
- ✅ LED 显示白色（空闲状态）
- ✅ "Connected to LiveKit" 消息出现

**如果启动失败**:
1. 检查 .env 文件配置是否正确
2. 确认网络连接正常
3. 查看错误信息并参考 [常见配置问题](#常见配置问题)

### 5.2 生成 Web 访问 Token

打开另一个终端窗口，生成访问 Token：

```bash
cd ~/lelamp_runtime

# 生成 Token（自定义房间名和用户名）
uv run python scripts/generate_client_token.py --room lelamp-room --user your_name
```

**预期输出**:
```
========================================
LiveKit Client Token Generator
========================================
Room:    lelamp-room
User:    your_name
TTL:     1 hour
----------------------------------------
Token:
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJh...
----------------------------------------
Copy the token above and paste it in the web client.
========================================
```

**复制并保存这个 Token**，在下一步连接 Web 客户端时需要用到。

### 5.3 启动 API 服务器（可选）

API 服务器提供 Web 设置页面所需的后端接口，包括 WiFi 配置和系统设置管理。

```bash
# 在第三个终端窗口
cd ~/lelamp_runtime

# 启动 API 服务器
uv run python -m uvicorn lelamp.api.app:app --host 0.0.0.0 --port 8000
```

**预期输出**:
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### 5.4 启动前端开发服务器

在第四个终端窗口（或您的开发机上）启动前端：

```bash
cd ~/lelamp_runtime/web

# 安装前端依赖（首次运行）
npm install

# 启动开发服务器
npm run dev
```

**预期输出**:
```
  VITE v5.x.x  ready in xxx ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

### 5.5 连接 Web 客户端

1. 在浏览器中打开 `http://localhost:5173`（或您的树莓派 IP 地址）
2. 输入 LiveKit URL（与 .env 中的 `LIVEKIT_URL` 相同）
3. 粘贴之前生成的 Token
4. 点击 "连接设备" 按钮

**连接成功的标志**:
- ✅ 看到摄像头视频画面
- ✅ LED 显示蓝色（正在倾听）
- ✅ 可以听到台灯的问候语

---

## 阶段六：验证测试

### 6.1 测试 RGB LED

验证 LED 矩阵是否正常工作：

```bash
cd ~/lelamp_runtime

# 运行 LED 测试（需要 sudo）
sudo uv run -m lelamp.test.test_rgb
```

**预期结果**: LED 矩阵显示彩色动画（红色、绿色、蓝色依次切换）

**测试清单**:
- [ ] 红色 LED 亮起
- [ ] 绿色 LED 亮起
- [ ] 蓝色 LED 亮起
- [ ] 彩色动画流畅

### 6.2 测试音频系统

验证麦克风和扬声器是否正常工作：

```bash
cd ~/lelamp_runtime

# 运行音频测试
uv run -m lelamp.test.test_audio
```

**预期结果**:
1. 录音 5 秒（对着麦克风说话）
2. 自动播放录音

**测试清单**:
- [ ] 麦克风录音正常
- [ ] 扬声器播放正常
- [ ] 音质清晰无杂音

### 6.3 测试电机

验证电机控制是否正常：

```bash
cd ~/lelamp_runtime

# 运行电机测试
uv run -m lelamp.test.test_motors --id lelamp --port /dev/ttyACM0
```

**预期结果**: 电机执行一系列测试动作

**测试清单**:
- [ ] 底座旋转 (base_yaw)
- [ ] 底座俯仰 (base_pitch)
- [ ] 肘部运动 (elbow_pitch)
- [ ] 腕部旋转 (wrist_roll)
- [ ] 灯头俯仰 (wrist_pitch)

### 6.4 Web 界面测试

通过 Web 客户端验证各项功能：

**语音对话测试**:
- [ ] 点击麦克风按钮说话
- [ ] 台灯正确识别语音
- [ ] 台灯做出适当回应
- [ ] 状态指示灯正确变化（蓝→紫→彩色）

**动作控制测试**:
- [ ] 点击 "点头" 按钮，台灯点头
- [ ] 点击 "摇头" 按钮，台灯摇头
- [ ] 点击 "跳舞" 按钮，台灯跳舞

**灯光控制测试**:
- [ ] 选择红色，LED 变红
- [ ] 选择蓝色，LED 变蓝
- [ ] 启动 "呼吸" 灯效，LED 呼吸闪烁

**视觉识别测试** (需要 ModelScope API Key):
- [ ] 点击 "拍照识别" 按钮
- [ ] 台灯拍照并识别物体
- [ ] 返回正确的识别结果

---

## 常见配置问题

### 问题 1: 找不到树莓派 IP 地址

**现象**: 无法通过 SSH 连接到树莓派，不知道 IP 地址是多少。

**解决方案**:

**方法 1: 检查路由器设备列表**
```bash
# 登录路由器管理页面，查找以下名称的设备:
# - lelamp-local (如果您设置了主机名)
# - raspberrypi (默认主机名)
# - Raspberry Pi Foundation (设备制造商)
```

**方法 2: 使用网络扫描工具**
```bash
# Mac/Linux
nmap -sn 192.168.1.0/24  # 根据您的网络段调整

# 或使用 arp-scan
sudo arp-scan --localnet
```

**方法 3: 连接显示器直接查看**
```bash
# 连接 HDMI 显示器和 USB 键盘，登录后执行:
hostname -I
ip addr show
```

**预防措施**: 在烧录系统时设置主机名为 `lelamp-local`，方便在路由器中识别。

---

### 问题 2: SSH 连接被拒绝

**错误信息**:
```
ssh: connect to host 192.168.1.100 port 22: Connection refused
```

**解决方案**:

1. **确认 SSH 已启用**
```bash
# 如果有显示器连接，在树莓派上执行:
sudo raspi-config
# 选择: Interface Options → SSH → Enable
```

2. **检查 IP 地址是否正确**
```bash
# 在您的电脑上 ping 树莓派
ping -c 4 192.168.1.100
```

3. **检查防火墙设置**
```bash
# 在树莓派上检查防火墙
sudo ufw status
# 如果需要启用 SSH
sudo ufw allow ssh
```

---

### 问题 3: WiFi 连接失败

**错误信息**:
```
Error: Connection activation failed
```

**解决方案**:

1. **检查 WiFi 密码是否正确**
```bash
# 重新连接，注意密码大小写
sudo nmcli device wifi connect "SSID" password "Password"
```

2. **检查 WiFi 频段支持**
```bash
# 树莓派 3B+ 及以下不支持 5GHz WiFi
# 如果使用 5GHz，尝试切换到 2.4GHz
```

3. **检查 WiFi 国家设置**
```bash
# 设置正确的国家代码
sudo raspi-config
# Localisation Options → WLAN Country → CN China
```

4. **重启网络服务**
```bash
sudo systemctl restart NetworkManager
# 或完全重启
sudo reboot
```

---

### 问题 4: 树莓派无法联网

**错误信息**:
```
ping: google.com: Temporary failure in name resolution
```

**解决方案**:

1. **检查 WiFi 是否连接**
```bash
nmcli connection show
iwconfig wlan0
```

2. **检查 DNS 配置**
```bash
# 查看 DNS 服务器
cat /etc/resolv.conf

# 手动设置 DNS
sudo nmcli connection modify "WiFi名称" ipv4.dns "8.8.8.8 8.8.4.4"
sudo systemctl restart NetworkManager
```

3. **测试网关连通性**
```bash
# 查看默认网关
ip route | grep default

# Ping 网关
ping -c 4 192.168.1.1  # 替换为实际网关地址
```

4. **重启路由器和树莓派**
```bash
sudo reboot
```

---

### 问题 5: SD 卡烧录失败

**错误信息**:
```
Error writing to device
```

**解决方案**:

1. **检查 SD 卡是否被写保护**
   - 确认 SD 卡的写保护开关已关闭
   - 尝试使用其他 SD 卡读卡器

2. **使用其他烧录工具**
   - **balenaEtcher**: https://etcher.balena.io/
   - **dd 命令** (Mac/Linux):
     ```bash
     diskutil list                          # 找到 SD 卡磁盘号
     sudo diskutil unmountDisk /dev/disk2   # 卸载 SD 卡
     sudo dd if=path/to/image.img of=/dev/rdisk2 bs=1m  # 烧录
     ```

3. **检查 SD 卡健康状态**
   - 使用 SD 卡检测工具检查坏块
   - 更换新的 SD 卡

---

### 问题 7: 串口权限不足

**错误信息**:
```
PermissionError: [Errno 13] Permission denied: '/dev/ttyACM0'
```

**解决方案**:

```bash
# 将用户添加到 dialout 组
sudo usermod -a -G dialout $USER

# 重新登录生效
# 或
sudo reboot
```

### 问题 8: 摄像头权限

**错误信息**:
```
PermissionError: [Errno 13] Permission denied: '/dev/video0'
```

**解决方案**:

```bash
# 将用户添加到 video 组
sudo usermod -a -G video $USER

# 重新登录生效
```

### 问题 9: GPIO 权限（LED）

**错误信息**:
```
RuntimeError: GPIO access not permitted. You need sudo access.
```

**解决方案**:

```bash
# LED 测试需要 sudo 权限
sudo uv run -m lelamp.test.test_rgb
```

### 问题 10: API Key 无效

**错误信息**:
```
AuthenticationError: Invalid API key
```

**解决方案**:

1. 检查 .env 文件中 API Key 是否正确复制
2. 确认 API Key 没有额外的空格或引号
3. 验证 API Key 是否已激活/有效
4. 检查 API 服务是否正常运行

```bash
# 查看当前配置（确认没有多余空格）
cat .env | grep API_KEY
```

### 问题 11: 网络连接超时

**错误信息**:
```
TimeoutError: Connection timeout
```

**解决方案**:

1. 检查网络连接是否正常
```bash
ping -c 4 api.deepseek.com
ping -c 4 vop.baidu.com
```

2. 检查防火墙设置
```bash
sudo ufw status
```

3. 如果使用代理，配置代理环境变量
```bash
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port
```

### 问题 12: 依赖安装失败

**错误信息**:
```
Error: Failed to download wheel
```

**解决方案**:

```bash
# 清理缓存重新安装
uv cache clean
uv sync --extra hardware

# 如果是 LFS 问题
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

---

## 验证清单

配置完成后，请逐项验证以下内容：

### 阶段零：树莓派初始设置

- [ ] SD 卡已烧录 Raspberry Pi OS
- [ ] 树莓派成功启动并运行
- [ ] 已成功获取树莓派 IP 地址
- [ ] SSH 连接成功建立
- [ ] WiFi 已配置并能正常连接
- [ ] 网络连通性正常（可访问互联网）
- [ ] 系统已更新到最新版本
- [ ] （可选）静态 IP 已配置

### 环境准备

- [ ] Python 3.12+ 已安装
- [ ] UV 包管理器已安装
- [ ] 项目已克隆到本地
- [ ] Python 依赖安装成功

### 账号配置

- [ ] DeepSeek API Key 已获取
- [ ] Baidu Speech API Key 已获取
- [ ] LiveKit 账号已创建
- [ ] ModelScope API Key 已获取（可选）

### 环境变量

- [ ] .env 文件已创建
- [ ] LIVEKIT_URL 已配置
- [ ] LIVEKIT_API_KEY 已配置
- [ ] LIVEKIT_API_SECRET 已配置
- [ ] DEEPSEEK_API_KEY 已配置
- [ ] BAIDU_SPEECH_API_KEY 已配置
- [ ] BAIDU_SPEECH_SECRET_KEY 已配置
- [ ] LELAMP_DEV_MODE 已设置为 1

### 硬件配置

- [ ] 串口设备已找到 (/dev/ttyACM0 或类似)
- [ ] 用户已添加到 dialout 组
- [ ] 用户已添加到 video 组
- [ ] 电机 ID 已设置
- [ ] 电机已校准

### 功能测试

- [ ] LED 测试通过
- [ ] 音频测试通过
- [ ] 电机测试通过
- [ ] Agent 启动成功
- [ ] LiveKit 连接成功
- [ ] Web 界面可访问
- [ ] 语音对话正常
- [ ] 动作控制正常
- [ ] 灯光效果正常
- [ ] 视觉识别正常（可选）

---

## 配置文件位置

| 文件 | 位置 | 说明 |
|------|------|------|
| `.env` | `~/lelamp_runtime/.env` | 主环境变量配置 |
| `pyproject.toml` | `~/lelamp_runtime/pyproject.toml` | Python 依赖定义 |
| `web/.env.development` | `~/lelamp_runtime/web/.env.development` | 前端环境变量 |
| `~/.config/uv/uv.toml` | 用户目录 | UV 全局配置 |

---

## 下一步

恭喜！您已成功配置 LeLamp 项目。接下来您可以：

1. **阅读完整用户指南**: [docs/USER_GUIDE.md](USER_GUIDE.md)
2. **自定义台灯人格**: 编辑 `main.py` 中的 `instructions`
3. **录制新动作**: 使用 `lelamp.record` 模块
4. **部署到生产**: 配置 systemd 服务
5. **贡献代码**: 提交 Pull Request

---

## 获取帮助

如果配置过程中遇到问题：

1. 查看 [docs/USER_GUIDE.md](USER_GUIDE.md) 的故障排查章节
2. 搜索项目 Issues: https://github.com/xwang152-jack/lelamp_runtime/issues
3. 提交新的 Issue，附上错误日志

---

**文档版本**: v1.0
**最后更新**: 2026-03-18
**作者**: LeLamp 开发团队
**许可证**: 参见主项目许可证
