# 前端灯光控制问题 - 解决方案总结

## 问题症状

- ❌ 前端发送灯光控制指令无响应
- ❌ 点击颜色按钮或特效按钮没有效果
- ❌ WebSocket 命令执行失败

## 根本原因

**RGB LED 控制 (WS2812) 需要 GPIO 内存访问权限**

`rpi_ws281x` 库需要 root 权限才能访问 `/dev/gpiomem` 来控制 WS2812 LED 灯带。如果没有正确配置:

1. API 服务器虽然启动,但 RGB 服务初始化失败
2. 系统自动降级到 `NoOpRGBService`(无操作服务)
3. 所有灯光指令返回"成功"但实际不执行
4. 前端显示"命令执行成功"但灯光不亮

## 快速修复

### 方法 1: 一键修复(推荐)

```bash
cd /home/pi/lelamp_runtime
sudo bash scripts/fix_rgb_control.sh
```

此脚本会自动:
1. 配置 GPIO 权限
2. 检查并安装依赖
3. 停止旧的服务进程
4. 使用 sudo 启动 API 服务器

### 方法 2: 手动启动

```bash
cd /home/pi/lelamp_runtime
sudo bash scripts/start_api_with_gpio.sh
```

### 方法 3: 诊断问题

```bash
bash scripts/diagnose_gpio.sh
```

## 技术细节

### 代码修改

1. **WebSocket 路由修复** (`lelamp/api/routes/websocket.py`):
   - 修复了命令执行结果判断逻辑
   - 现在正确检测失败并返回错误信息

2. **启动脚本优化** (`scripts/start_api_with_gpio.sh`):
   - 添加了 GPIO 权限检查
   - 使用 `sudo -E uv run` 保留环境变量
   - 添加了详细的错误提示

### 服务降级机制

当 RGB 服务初始化失败时的自动降级:

```python
# lelamp/api/app.py (第 143-154 行)
try:
    from lelamp.service.rgb.rgb_service import RGBService
    rgb_service = RGBService()
    rgb_service.start()
    app.state.rgb_service = rgb_service
    logger.info("RGBService started")
except Exception as e:
    logger.error(f"RGBService start failed: {e}")
    from lelamp.service.rgb.noop_rgb_service import NoOpRGBService
    rgb_service = NoOpRGBService()  # 降级到模拟服务
    rgb_service.start()
    app.state.rgb_service = rgb_service
```

### WebSocket 命令执行流程

```
前端 → WebSocket → API服务器 → Agent._execute_command() → RGBTools
                                      ↓
                               返回结果字符串
                                      ↓
                         检查是否包含"失败"/"错误"
                                      ↓
                    发送 command_result 消息回前端
```

## 验证修复

运行以下命令检查:

```bash
# 1. 检查 API 服务是否用 sudo 启动
ps aux | grep uvicorn

# 应该看到:
# root ... sudo -E uv run uvicorn lelamp.api.app:app

# 2. 检查 GPIO 权限
ls -l /dev/gpiomem

# 应该看到:
# crw-rw---- 1 root video ...

# 3. 检查服务日志
# 应该看到 "RGBService started"
# 而不是 "RGBService start failed"

# 4. 测试灯光控制
# 在前端点击颜色按钮,应该看到灯光变化
```

## 常见问题

### Q: 为什么配置了 udev 规则还是不行?

A: 某些版本的 `rpi_ws281x` 库无论如何都需要 root 权限。最可靠的方法是使用 sudo 启动 API 服务器。

### Q: 前端显示"命令执行成功"但灯光不亮?

A: 这是因为 `NoOpRGBService` 也会返回成功消息。检查:
1. API 服务是否用 sudo 启动
2. 日志中是否有 "RGBService started"
3. 运行 `bash scripts/diagnose_gpio.sh` 诊断

### Q: 如何在开发环境测试?

A: 开发环境会自动使用 `NoOpRGBService`,这是正常的。实际硬件测试需要在树莓派上进行。

## 相关文件

### 新增文件
- `scripts/diagnose_gpio.sh` - GPIO 诊断工具
- `scripts/fix_rgb_control.sh` - 一键修复脚本
- `docs/GPIO_PERMISSION_SETUP.md` - GPIO 权限配置详细文档

### 修改文件
- `lelamp/api/routes/websocket.py` - 修复命令执行结果判断
- `scripts/start_api_with_gpio.sh` - 改进启动脚本,添加检查
- `scripts/setup_gpio_permissions.sh` - 完善权限配置

## 下一步

1. **立即修复**: 运行 `sudo bash scripts/fix_rgb_control.sh`

2. **永久配置**:
   ```bash
   # 配置 GPIO 权限
   sudo bash scripts/setup_gpio_permissions.sh

   # 配置 sudo 免密(可选)
   sudo bash scripts/setup_sudoers.sh
   ```

3. **验证修复**:
   ```bash
   # 运行诊断
   bash scripts/diagnose_gpio.sh

   # 测试灯光控制
   # 在前端界面点击颜色按钮
   ```

## 联系支持

如果问题仍然存在:
1. 运行 `bash scripts/diagnose_gpio.sh` 并保存输出
2. 查看 API 服务器日志(`LOG_LEVEL=DEBUG`)
3. 提供诊断结果和日志以获取支持
