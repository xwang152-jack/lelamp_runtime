# 2026-03-19 前端灯光控制修复

## 问题

前端发送灯光控制命令返回成功，但 LeLamp 设备的 LED 实际上没有变化。

## 根本原因

`ws281x` LED 驱动库需要访问 `/dev/mem` 设备来控制 GPIO，但 `pi` 用户没有权限访问该设备。

错误信息：
```
RGBService start failed: ws2811_init failed with code -5 (mmap() failed), fallback to NoOpRGBService
```

RGB 服务启动失败后自动降级到 `NoOpRGBService`（无操作服务），所有命令执行成功但实际硬件不响应。

## 解决方案

### 核心修复
使用 `sudo` 运行 API 服务器以获取必要的硬件访问权限。

### 具体实现

1. **WebSocket 路由修复** (`lelamp/api/routes/websocket.py`)
   - 修复 Priority 参数类型错误：`int` → `Priority` 枚举
   - 重写命令执行逻辑，直接使用硬件服务
   - 添加更多灯光控制命令支持

2. **数据库初始化**
   - 创建缺失的 `device_states` 等数据库表
   - 修复状态查询功能

3. **系统配置**
   - 创建启动脚本：`/usr/local/bin/start-lelamp-api`
   - 配置 systemd 服务：`lelamp-api.service`
   - 设置开机自启动

## 验证结果

### 功能测试
- ✅ WebSocket 连接正常
- ✅ 基础 RGB 颜色控制（红、绿、蓝、白、暖白、紫）
- ✅ RGB 效果控制（彩虹、呼吸效果）
- ✅ 亮度控制（10%-100%）
- ✅ 停止效果命令

### 测试命令
```bash
# 测试灯光控制
cd /home/pi/lelamp_runtime && .venv/bin/python /tmp/test_multi.py

# 查看日志
tail -f /tmp/uvicorn.log | grep -E 'RGB|rgb'
```

## 使用方法

### 手动启动
```bash
sudo .venv/bin/python -m uvicorn lelamp.api.app:app --host 0.0.0.0 --port 8000
```

### 使用启动脚本
```bash
start-lelamp-api
```

### 使用 systemd 服务（推荐）
```bash
sudo systemctl start lelamp-api    # 启动服务
sudo systemctl stop lelamp-api     # 停止服务
sudo systemctl status lelamp-api   # 查看状态
sudo journalctl -u lelamp-api -f   # 查看日志
```

## 相关文件

- **代码修改**: `lelamp/api/routes/websocket.py`
- **设置指南**: `docs/setup/api-server-setup.md`
- **启动脚本**: `/usr/local/bin/start-lelamp-api`
- **系统服务**: `/etc/systemd/system/lelamp-api.service`

## 更新仓库

- ✅ GitHub: 已同步
- ✅ Raspberry Pi: 已同步

## 后续建议

1. **安全加固**: 考虑使用 capabilities 而不是完整的 sudo 权限
2. **权限管理**: 研究 `/dev/gpiomem` 替代方案
3. **监控**: 添加 RGB 服务健康检查
4. **文档**: 用户手册中添加权限说明
