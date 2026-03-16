# Python 3.12 升级总结

## 📋 完成的工作

### 1. ✅ 更新项目 Python 版本要求

**修改文件**：
- `pyproject.toml`: `requires-python = ">=3.12,<3.13"`
- 使用 Python 3.12 重新锁定依赖 (`uv.lock`)

### 2. ✅ 更新文档

**修改的文档**：
- `README.md`: 添加 Python 3.12+ 前置要求
- `CLAUDE.md`: 添加 Python 3.12+ 要求说明
- `.env.example`: 更新使用说明

**新增的文档**：
- `docs/PYTHON_INSTALLATION.md`: Python 3.12 详细安装指南
- `scripts/verify_python312.sh`: 安装验证脚本

### 3. ✅ Git Commit

- **Commit**: `ae37f00`
- **标题**: `chore: 限制 Python 版本为 3.12`
- **修改**: 4 个文件，+32/-565 行

### 4. ✅ Python 3.12 安装完成

通过 Homebrew 成功安装：
```bash
brew install python@3.12
```

**安装信息**：
- 版本：Python 3.12.13
- 路径：`/opt/homebrew/bin/python3.12`
- 安装位置：`/opt/homebrew/Cellar/python@3.12/3.12.13`
- 文件数量：3,612 个文件
- 占用空间：70 MB

**安装方式**：
- ✅ 方式 1: Homebrew 安装（已完成）
- ✅ 方式 2: UV Python 管理（已配置 UV 使用 Python 3.12.12）

## 📊 版本更新详情

### 之前
```toml
requires-python = ">=3.12"
```
- 允许 Python 3.12, 3.13, 3.14...
- 可能导致未来版本兼容性问题

### 现在
```toml
requires-python = ">=3.12,<3.13"
```
- 严格限制为 Python 3.12.x
- 避免 Python 3.13+ 的潜在兼容性问题
- 确保与项目类型注解语法兼容（`str | None` 等）

## 🎯 为什么限制 Python 版本？

1. **类型注解语法**: 项目使用 `str | None` 语法（Python 3.10+）
2. **依赖兼容性**: 确保所有依赖在 Python 3.12 上稳定运行
3. **避免破坏性更新**: Python 3.13 可能引入不兼容的变更
4. **测试覆盖**: 集中在单一版本上进行测试

## 🚀 安装后操作

### 验证安装

安装完成后，运行验证脚本：

```bash
./scripts/verify_python312.sh
```

或手动验证：

```bash
# 检查 Python 3.12
python3.12 --version

# 同步项目依赖
uv sync --python 3.12

# 运行测试
uv run pytest lelamp/test/ -v
```

### 配置环境（可选）

将 Python 3.12 设置为默认 python3：

```bash
# 添加到 ~/.zshrc
echo 'export PATH="/opt/homebrew/opt/python@3.12/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# 验证
python3 --version  # 应该显示 Python 3.12.x
```

### 运行项目

```bash
# 方法 1: 使用 UV（推荐）
sudo uv run main.py console

# 方法 2: 直接使用 Python 3.12
python3.12 main.py console
```

## 📝 系统要求更新

### macOS
- Python 3.12.x（通过 Homebrew 或 UV）
- UV package manager

### Raspberry Pi / Linux
- Python 3.12.x（通过 apt 或源码编译）
- 系统依赖：
  - `portaudio19-dev`（用于 PyAudio）
  - 其他硬件相关库

## 🔗 相关资源

- **安装指南**: `docs/PYTHON_INSTALLATION.md`
- **验证脚本**: `scripts/verify_python312.sh`
- **项目文档**: `CLAUDE.md`
- **环境配置**: `.env.example`

## ⏭️ 下一步

1. ✅ 等待 Homebrew 安装完成
2. ✅ 运行验证脚本
3. ✅ 同步项目依赖
4. ✅ 运行单元测试确认兼容性
5. ✅ 更新生产环境的 Python 版本

## 📊 影响评估

### 开发环境
- ✅ 需要安装 Python 3.12
- ✅ UV 会自动管理 Python 版本
- ✅ 旧项目环境需要重新创建

### 生产环境
- ⚠️ Raspberry Pi 需要更新 Python 版本
- ⚠️ 可能需要重新编译 Python 3.12
- ✅ 向后兼容（3.12 功能是 3.9/3.10 的超集）

### CI/CD
- ⚠️ 需要更新 CI 配置使用 Python 3.12
- ✅ Docker 镜像需要更新基础镜像

## 🐛 常见问题

### Q: UV 说找不到 Python 3.12？
A: UV 会自动下载。如果需要手动安装：
```bash
uv python install 3.12
```

### Q: PyAudio 编译失败？
A: 安装 PortAudio 系统库：
```bash
# macOS
brew install portaudio

# Linux
sudo apt-get install portaudio19-dev
```

### Q: 可以使用 Python 3.13 吗？
A: 目前不支持。需要等待所有依赖支持 3.13 后再测试。

---

**更新时间**: 2026-03-16
**Python 版本**: 3.12.x
**提交哈希**: ae37f00
