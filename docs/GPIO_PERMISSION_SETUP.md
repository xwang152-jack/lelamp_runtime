# GPIO 权限配置指南

## 问题描述

LeLamp 的 RGB LED 灯光控制需要访问 Raspberry Pi 的 GPIO 内存,这需要特殊权限。如果没有正确配置:

- ❌ 前端灯光控制指令无响应
- ❌ RGBService 初始化失败或降级到 NoOpRGBService
- ❌ WebSocket 命令返回"执行失败"

## 根本原因

`rpi_ws281x` 库需要直接访问 `/dev/gpiomem` 或 `/dev/mem` 来控制 WS2812 LED。这些设备通常需要 root 权限。

## 解决方案

### 方案 1: 使用 sudo 启动 API 服务器(推荐)

这是最简单可靠的方法:

```bash
cd /home/pi/lelamp_runtime
sudo bash scripts/start_api_with_gpio.sh
```

此脚本会:
- ✅ 检查 GPIO 权限
- ✅ 激活虚拟环境
- ✅ 使用 `sudo -E uv run` 保留环境变量
- ✅ 启动 API 服务器

### 方案 2: 配置 GPIO 权限(可选)

如果不想每次都使用 sudo,可以配置 udev 规则:

```bash
# 配置 GPIO 访问权限
sudo bash scripts/setup_gpio_permissions.sh

# 注销并重新登录以使组权限生效
# 或运行: newgrp gpio
```

**注意**: 即使配置了 udev 规则,某些情况下仍可能需要 sudo。

### 方案 3: 配置 sudo 免密(可选)

如果希望免密启动 API 服务器:

```bash
sudo bash scripts/setup_sudoers.sh
```

这会配置 sudoers,允许 `pi` 用户免密运行 `uv run uvicorn`。

## 验证配置

运行诊断脚本检查配置:

```bash
bash scripts/diagnose_gpio.sh
```

该脚本会检查:
- GPIO 设备是否存在
- 用户组权限
- udev 规则
- Python 库安装
- API 服务状态

## 常见问题

### Q1: 为什么需要 sudo?

**A**: WS2812 LED 需要精确的时序控制,`rpi_ws281x` 库通过直接访问 GPIO 内存实现。这需要 root 权限。

### Q2: 配置了 udev 规则还是不行?

**A**: 某些版本的 `rpi_ws281x` 无论如何都需要 root 权限。建议使用 `sudo` 启动。

### Q3: 前端显示"命令执行成功"但灯光不亮?

**A**:
1. 检查 API 服务是否用 sudo 启动
2. 运行 `bash scripts/diagnose_gpio.sh` 诊断
3. 查看服务器日志是否有错误

### Q4: 如何在开发环境(非树莓派)测试?

**A**: 开发环境会自动使用 `NoOpRGBService`(模拟服务),灯光控制会返回成功但不会实际点亮。

## 启动命令总结

### 生产环境(树莓派)

```bash
# 推荐:使用sudo启动
sudo bash scripts/start_api_with_gpio.sh
```

### 开发环境

```bash
# 直接启动(GPI O会被模拟)
uv run uvicorn lelamp.api.app:app --host 0.0.0.0 --port 8000
```

### 调试模式

```bash
# 启用详细日志
sudo LOG_LEVEL=DEBUG uv run uvicorn lelamp.api.app:app --host 0.0.0.0 --port 8000 --log-level debug
```

## 相关文件

- `scripts/start_api_with_gpio.sh` - 推荐的启动脚本
- `scripts/setup_gpio_permissions.sh` - GPIO权限配置
- `scripts/setup_sudoers.sh` - sudo免密配置
- `scripts/diagnose_gpio.sh` - 诊断工具
- `lelamp/service/rgb/rgb_service.py` - RGB服务实现
- `lelamp/api/routes/websocket.py` - WebSocket命令处理

## 技术细节

### RGB LED 硬件
- 型号: WS2812B
- 数量: 64颗 (8x8 矩阵)
- 控制: GPIO 12 (PWM0)
- 库: rpi_ws281x

### 服务降级机制
当 RGB 服务初始化失败时,会自动降级到 `NoOpRGBService`:
```python
try:
    rgb_service = RGBService()
    rgb_service.start()
except Exception as e:
    logger.error(f"RGBService start failed: {e}")
    from lelamp.service.rgb.noop_rgb_service import NoOpRGBService
    rgb_service = NoOpRGBService()  # 模拟服务
```

### 检测服务状态
查看日志确认 RGB 服务是否正常:
```bash
# 应该看到 "RGBService started"
# 而不是 "RGBService start failed"
```
