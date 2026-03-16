# Python 3.12 安装指南

## macOS 系统安装

### 方法 1: 使用 Homebrew（推荐）

```bash
# 安装 Python 3.12
brew install python@3.12

# 验证安装
python3.12 --version

# 将 Python 3.12 添加到 PATH（可选）
echo 'export PATH="/opt/homebrew/opt/python@3.12/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### 方法 2: 使用 UV 自动管理

UV 会自动下载和管理 Python 版本，无需手动安装：

```bash
# UV 将自动使用 Python 3.12
uv sync

# 或明确指定 Python 版本
uv sync --python 3.12
```

## Linux (Raspberry Pi) 系统安装

### Ubuntu/Debian 系统

```bash
# 添加 deadsnakes PPA（如果 Python 3.12 不在默认仓库）
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

# 安装 Python 3.12
sudo apt install python3.12 python3.12-venv python3.12-dev

# 验证安装
python3.12 --version
```

### Raspberry Pi OS

```bash
# 从源码编译（如果软件源没有 3.12）
sudo apt update
sudo apt install -y build-essential zlib1g-dev libncurses5-dev \
    libgdbm-dev libnss3-dev libssl-dev libreadline-dev \
    libffi-dev libsqlite3-dev wget libbz2-dev

# 下载 Python 3.12
cd /tmp
wget https://www.python.org/ftp/python/3.12.7/Python-3.12.7.tgz
tar -xf Python-3.12.7.tgz
cd Python-3.12.7

# 编译安装
./configure --enable-optimizations
make -j $(nproc)
sudo make altinstall

# 验证安装
python3.12 --version
```

## 验证安装

运行验证脚本：

```bash
./scripts/verify_python312.sh
```

或手动验证：

```bash
# 检查 Python 版本
python3.12 --version

# 检查 pip
pip3.12 --version

# 验证项目依赖
cd /path/to/lelamp_runtime
uv sync --python 3.12
```

## 常见问题

### Q: 为什么需要 Python 3.12？

A: 项目使用了 Python 3.10+ 的类型注解语法（如 `str | None`），并限制在 3.12.x 以确保兼容性。

### Q: 可以使用 Python 3.13 吗？

A: 目前项目限制为 `>=3.12,<3.13`，以避免潜在的兼容性问题。未来版本可能会支持 3.13+。

### Q: UV 无法找到 Python 3.12？

A: UV 会自动下载所需的 Python 版本。如果遇到问题，可以手动指定：

```bash
uv python install 3.12
uv sync --python 3.12
```

### Q: Homebrew 安装后找不到 python3.12？

A: 将 Homebrew 的 Python 路径添加到 PATH：

```bash
export PATH="/opt/homebrew/opt/python@3.12/bin:$PATH"
```

或使用完整路径：

```bash
/opt/homebrew/opt/python@3.12/bin/python3.12 --version
```

## 下一步

安装完成后：

1. **同步依赖**：
   ```bash
   uv sync
   ```

2. **配置环境变量**：
   ```bash
   cp .env.example .env
   # 编辑 .env 文件填入实际配置
   ```

3. **运行项目**：
   ```bash
   sudo uv run main.py console
   ```

## 参考资源

- [Python 官方下载](https://www.python.org/downloads/)
- [UV 文档](https://github.com/astral-sh/uv)
- [Homebrew Python](https://docs.brew.sh/Homebrew-and-Python)
