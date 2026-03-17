# LeLamp Phase 3 快速命令参考

**常用命令速查卡** - 保存此文件以备快速查阅

---

## 🚀 快速测试

```bash
# 快速验证（推荐）
./quick_test.sh

# 完整测试
./test_phase3.sh

# 手动测试
uv run pytest lelamp/test/integration/ -v
```

---

## 📡 API 服务器

### 启动服务器

```bash
# 开发模式（热重载）
uv run uvicorn lelamp.api.app:app --host 0.0.0.0 --port 8000 --reload

# 生产模式
uv run uvicorn lelamp.api.app:app --host 0.0.0.0 --port 8000 --workers 4

# 使用启动脚本
uv run python scripts/run_api_server.py
```

### 访问文档

```bash
# Swagger UI
open http://localhost:8000/docs

# ReDoc
open http://localhost:8000/redoc
```

### 停止服务器

```bash
# 查找进程
ps aux | grep uvicorn

# 杀死进程
pkill -f uvicorn

# 或使用 PID
kill <PID>
```

---

## 🧪 测试命令

### 运行测试

```bash
# 所有测试
uv run pytest

# Phase 3 测试
uv run pytest lelamp/test/integration/test_database.py
uv run pytest lelamp/test/integration/test_api.py
uv run pytest lelamp/test/integration/test_websocket.py
uv run pytest lelamp/test/integration/test_e2e.py

# 特定测试
uv run pytest lelamp/test/integration/test_api.py::test_get_device_state

# 匹配模式
uv run pytest -k "test_device"

# 只运行失败的测试
uv run pytest --lf

# 遇到第一个失败时停止
uv run pytest -x
```

### 测试选项

```bash
# 详细输出
uv run pytest -v

# 超详细输出
uv run pytest -vv

# 显示 print 输出
uv run pytest -s

# 简短 traceback
uv run pytest --tb=line

# 无 traceback
uv run pytest --tb=no
```

### 覆盖率

```bash
# 终端覆盖率
uv run pytest --cov=lelamp.api --cov=lelamp.database --cov-report=term

# HTML 报告
uv run pytest --cov=lelamp.api --cov=lelamp.database --cov-report=html

# 两种都有
uv run pytest --cov=lelamp.api --cov=lelamp.database --cov-report=html --cov-report=term

# 查看报告
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

---

## 🗄️ 数据库操作

### 初始化数据库

```bash
# 创建所有表
uv run python -c "from lelamp.database import Base, engine; Base.metadata.create_all(bind=engine); print('Database initialized!')"

# 使用初始化脚本
uv run python lelamp/database/init_db.py
```

### 数据库位置

```bash
# 默认位置
./lelamp.db

# 测试数据库（内存）
sqlite:///:memory:

# 自定义位置
export LELAMP_DATABASE_URL="sqlite:///path/to/database.db"
```

### 查看数据库

```bash
# 使用 sqlite3 命令行
sqlite3 lelamp.db

# 常用 SQL
.tables
.schema conversations
SELECT * FROM conversations LIMIT 10;
```

---

## 🌐 API 测试（curl）

### 基础测试

```bash
# 健康检查
curl http://localhost:8000/health | jq '.'

# 设备列表
curl http://localhost:8000/api/devices | jq '.'

# 设备状态
curl http://localhost:8000/api/devices/lelamp_001/state | jq '.'

# 发送命令
curl -X POST http://localhost:8000/api/devices/lelamp_001/command \
  -H "Content-Type: application/json" \
  -d '{"type":"motor_move","action":"move_joint","params":{"joint_name":"base_yaw","position":45.0}}' | jq '.'

# 操作日志
curl http://localhost:8000/api/devices/lelamp_001/operations | jq '.'

# 对话记录
curl http://localhost:8000/api/devices/lelamp_001/conversations | jq '.'

# 统计数据
curl http://localhost:8000/api/devices/lelamp_001/statistics | jq '.'
```

### 带参数的查询

```bash
# 分页查询
curl "http://localhost:8000/api/devices/lelamp_001/operations?skip=0&limit=10" | jq '.'

# 时间过滤
curl "http://localhost:8000/api/devices/lelamp_001/operations?hours=24" | jq '.'

# 组合参数
curl "http://localhost:8000/api/devices/lelamp_001/conversations?skip=0&limit=20" | jq '.'
```

---

## 🔌 WebSocket 测试

### Python 客户端

```bash
# 创建测试文件
cat > test_ws.py << 'EOF'
import asyncio
import websockets
import json

async def test():
    async with websockets.connect("ws://localhost:8000/api/ws/lelamp_001") as ws:
        msg = await ws.recv()
        print(f"收到: {msg}")

        await ws.send(json.dumps({"type": "ping"}))
        pong = await ws.recv()
        print(f"收到: {pong}")

asyncio.run(test())
EOF

# 运行测试
uv run python test_ws.py
```

### JavaScript 客户端

```javascript
// 在浏览器控制台运行
const ws = new WebSocket('ws://localhost:8000/api/ws/lelamp_001');

ws.onmessage = (event) => {
    console.log('收到:', JSON.parse(event.data));
};

ws.onopen = () => {
    console.log('已连接');
    ws.send(JSON.stringify({type: 'ping'}));
};
```

---

## 📝 代码质量

### 类型检查

```bash
# Mypy 类型检查
uv run mypy lelamp/api/

# 或使用 ruff
uv run ruff check lelamp/api/
```

### 代码格式化

```bash
# 使用 ruff 格式化
uv run ruff format lelamp/api/

# 检查格式
uv run ruff format --check lelamp/api/
```

### Linting

```bash
# Pylint
uv run pylint lelamp/api/

# Ruff lint
uv run ruff check lelamp/api/
```

---

## 📚 文档命令

### 查看文档

```bash
# 测试指南
cat TEST_GUIDE.md

# 快速测试指南
cat TESTING_QUICKSTART.md

# API 文档
cat lelamp/api/API_DOCUMENTATION.md

# 部署指南
cat lelamp/api/DEPLOYMENT_GUIDE.md

# 使用示例
cat lelamp/api/EXAMPLES.md

# 测试结果
cat TEST_RESULTS_PHASE3.md

# Phase 3 总结
cat PHASE3_FINAL_SUMMARY.md
```

### 生成文档

```bash
# API 文档自动生成
# 启动服务器后访问:
open http://localhost:8000/docs
```

---

## 🔧 依赖管理

### 安装依赖

```bash
# 所有依赖
uv sync

# API 依赖
uv sync --extra api

# 开发依赖
uv sync --extra dev

# API + 开发依赖
uv sync --extra api --extra dev

# 所有可选依赖
uv sync --all-extras
```

### 更新依赖

```bash
# 更新所有依赖
uv sync --upgrade

# 更新特定包
uv add fastapi --upgrade
```

### 查看依赖

```bash
# 列出所有依赖
uv pip list

# 查看依赖树
uv pip tree
```

---

## 🐛 调试命令

### 查看日志

```bash
# API 服务器日志（如果使用启动脚本）
tail -f /tmp/lelamp_api.log

# 查看最后 20 行
tail -n 20 /tmp/lelamp_api.log

# 搜索错误
grep -i error /tmp/lelamp_api.log
```

### 调试模式

```bash
# 启动调试服务器
uv run uvicorn lelamp.api.app:app --host 0.0.0.0 --port 8000 --reload --log-level debug

# 测试时查看详细输出
uv run pytest -vv -s lelamp/test/integration/test_api.py::test_get_device_state
```

---

## 🎯 常用操作

### 检查端口占用

```bash
# 检查 8000 端口
lsof -ti:8000

# 杀死占用端口的进程
kill -9 $(lsof -ti:8000)
```

### 查找进程

```bash
# 查找 uvicorn 进程
ps aux | grep uvicorn

# 查找 Python 进程
ps aux | grep python

# 查看特定进程
ps aux | grep -E "lelamp|api"
```

### 清理文件

```bash
# 清理测试数据库
rm -f lelamp.db lelamp-test.db

# 清理覆盖率报告
rm -rf htmlcov/

# 清理 Python 缓存
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
```

---

## 📊 统计命令

### 代码统计

```bash
# 统计代码行数
find lelamp/api lelamp/database -name "*.py" | xargs wc -l

# 统计测试代码
find lelamp/test/integration -name "test_*.py" | xargs wc -l

# 统计所有 Python 文件
find . -name "*.py" | xargs wc -l
```

### Git 统计

```bash
# 查看 Phase 3 提交
git log --oneline --grep="Phase 3\|feat:"

# 查看提交统计
git shortlog -sn --grep="Phase 3\|feat:"

# 查看文件变更
git diff --stat HEAD~5 HEAD
```

---

## 🎉 快速验收

### 一键验证

```bash
# 运行快速测试
./quick_test.sh

# 如果看到 79 passed，说明测试通过！
# 预期输出: ====================== 79 passed in 8.74s =======================
```

### 手动验证

```bash
# 1. 启动服务器
uv run uvicorn lelamp.api.app:app --host 0.0.0.0 --port 8000 &

# 2. 测试健康检查
curl http://localhost:8000/health | jq '.'

# 3. 运行测试
uv run pytest lelamp/test/integration/ -q

# 4. 停止服务器
pkill -f uvicorn
```

---

## 💡 提示

1. **快速测试**: 使用 `./quick_test.sh`
2. **查看日志**: `tail -f /tmp/lelamp_api.log`
3. **调试测试**: `uv run pytest -vv -s`
4. **API 文档**: 访问 http://localhost:8000/docs
5. **停止服务器**: `pkill -f uvicorn`

---

**保存此文件** 以备快速查阅！

**更新日期**: 2026-03-17
