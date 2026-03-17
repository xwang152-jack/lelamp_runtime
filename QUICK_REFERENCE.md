# LeLamp 快速参考

## 🚀 常用命令

### 前端开发（Vue 3 + TypeScript + Vite）
```bash
cd web
pnpm dev
```
访问: http://localhost:5173

**技术栈**: Vue 3 + TypeScript + Vite + Pinia + Element Plus

### 生成 Token
```bash
./quick_start.sh
```
Token 会自动复制到剪贴板

### 端到端测试（需要 Raspberry Pi）
```bash
./test_e2e.sh
```

### 查找电机端口
```bash
uv run lerobot-find-port
```

### 测试 RGB LED（需要 Raspberry Pi + sudo）
```bash
sudo uv run -m lelamp.test.test_rgb
```

### 测试音频系统
```bash
uv run -m lelamp.test.test_audio
```

### 测试电机（需要 Raspberry Pi）
```bash
uv run -m lelamp.test.test_motors --id <lamp_id> --port <port>
```

### 录制电机动作（需要 Raspberry Pi）
```bash
uv run -m lelamp.record --id <lamp_id> --port <port> --name <recording_name>
```

### 回放电机动作（需要 Raspberry Pi）
```bash
uv run -m lelamp.replay --id <lamp_id> --port <port> --name <recording_name>
```

### 启动主程序（需要 Raspberry Pi + sudo）
```bash
sudo uv run main.py console
```

## 📁 重要文件

### 配置
- `.env` - 环境配置（不要提交到 Git）
- `CLAUDE.md` - Claude Code 项目指南
- `README.md` - 项目说明

### 文档
- `docs/USER_GUIDE.md` - 用户指南
- `docs/TESTING_CHECKLIST.md` - 测试清单

### 前端
- `web/` - Vue 3 前端应用

### 脚本
- `quick_start.sh` - Token 生成工具
- `test_e2e.sh` - 端到端测试
- `cleanup_test_files.sh` - 清理测试文件
- `cleanup_md_docs.sh` - 清理 Markdown 文档
- `remove_web_client.sh` - 删除旧版 web_client

## 🔧 故障排查

### 后端无法启动（macOS）
**原因**: macOS 不支持硬件依赖
**解决**: 只能测试前端，完整功能需要 Raspberry Pi

### Token 生成失败
**检查**:
1. `.env` 文件是否存在
2. LiveKit 配置是否正确
3. 运行 `./quick_start.sh` 查看错误信息

### 前端构建失败
**解决**:
```bash
cd web
rm -rf node_modules pnpm-lock.yaml
pnpm install
```

### 旧版 web_client 访问失败
**原因**: 旧版 web_client 已删除，项目统一使用 web/
**解决**: 访问 http://localhost:5173（新前端）

## 📋 平台兼容性

| 功能 | macOS | Raspberry Pi |
|------|-------|--------------|
| 前端开发 | ✅ | ✅ |
| 后端运行 | ❌ | ✅ |
| 摄像头 | ❌ | ✅ |
| RGB LED | ❌ | ✅ |
| 电机控制 | ❌ | ✅ |

## 🎯 开发工作流

### macOS（前端开发）
1. `cd web && pnpm dev`
2. 访问 http://localhost:5173
3. 测试 UI 组件和交互
4. 使用 Vue 3 + TypeScript 开发
5. 无法连接真实后端（需要 Raspberry Pi）

### Raspberry Pi（完整系统）
1. 准备硬件（摄像头、电机、LED）
2. 配置 `.env` 文件
3. `sudo uv run main.py console`
4. 前端连接到后端
5. 完整功能测试

## 💡 提示

- **前端开发**: 在 macOS 上进行
- **完整测试**: 需要 Raspberry Pi 硬件
- **Token 安全**: 不要分享或提交到 Git
- **调试模式**: 设置 `LOG_LEVEL=DEBUG`
- **开发模式**: 确保 `LELAMP_DEV_MODE=1`（配置已设置）

---

**详细文档**: 查看 `docs/` 目录
**问题反馈**: 查看归档的测试文档或提交 Issue
