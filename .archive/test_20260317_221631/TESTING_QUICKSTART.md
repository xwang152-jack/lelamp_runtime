# LeLamp Phase 3 快速测试指南

本指南将帮助你快速测试 LeLamp Phase 3 后端功能。

## 🚀 快速开始

### 方法 1: 使用自动化测试脚本（推荐）

```bash
# 运行完整的测试套件
./test_phase3.sh
```

这个脚本会自动执行所有测试步骤，包括：
1. 环境准备
2. 启动 API 服务器
3. 运行单元测试
4. 运行集成测试
5. 运行 E2E 测试
6. 生成覆盖率报告
7. 手动 API 测试
8. WebSocket 测试

### 方法 2: 手动逐步测试

按照下面的步骤手动进行测试。

---

## 📋 测试步骤

### 步骤 1: 安装依赖

```bash
# 安装 API 相关依赖
uv sync --extra api
```

### 步骤 2: 启动 API 服务器

```bash
# 启动 FastAPI 服务器
uv run uvicorn lelamp.api.app:app --host 0.0.0.0 --port 8000 --reload
```

服务器启动后，访问：
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 步骤 3: 运行测试

#### 3.1 数据库测试

```bash
uv run pytest lelamp/test/integration/test_database.py -v
```

预期结果: 19 个测试全部通过 ✅

#### 3.2 API 测试

```bash
uv run pytest lelamp/test/integration/test_api.py -v
```

预期结果: 27 个测试全部通过 ✅

#### 3.3 WebSocket 测试

```bash
uv run pytest lelamp/test/integration/test_websocket.py -v
```

预期结果: 22 个测试全部通过 ✅

#### 3.4 E2E 测试

```bash
uv run pytest lelamp/test/integration/test_e2e.py -v
```

预期结果: 11 个测试全部通过 ✅

### 步骤 4: 生成覆盖率报告

```bash
# 生成 HTML 覆盖率报告
uv run pytest --cov=lelamp.api --cov=lelamp.database --cov-report=html

# 查看报告
open htmlcov/index.html  # macOS
# 或
xdg-open htmlcov/index.html  # Linux
```

目标覆盖率: > 60%

### 步骤 5: 手动 API 测试

使用 curl 测试主要端点：

```bash
# 1. 健康检查
curl http://localhost:8000/health | jq '.'

# 2. 获取设备列表
curl http://localhost:8000/api/devices | jq '.'

# 3. 获取设备状态
curl http://localhost:8000/api/devices/lelamp_001/state | jq '.'

# 4. 发送命令
curl -X POST http://localhost:8000/api/devices/lelamp_001/command \
  -H 'Content-Type: application/json' \
  -d '{
    "type": "motor_move",
    "action": "move_joint",
    "params": {"joint_name": "base_yaw", "position": 45.0}
  }' | jq '.'

# 5. 获取操作日志
curl http://localhost:8000/api/devices/lelamp_001/operations | jq '.'

# 6. 获取对话记录
curl http://localhost:8000/api/devices/lelamp_001/conversations | jq '.'

# 7. 获取统计数据
curl http://localhost:8000/api/devices/lelamp_001/statistics | jq '.'

# 8. 获取健康状态
curl http://localhost:8000/api/devices/lelamp_001/health | jq '.'
```

### 步骤 6: WebSocket 测试

创建测试脚本 `test_websocket.py`:

```python
import asyncio
import websockets
import json

async def test_websocket():
    uri = "ws://localhost:8000/api/ws/lelamp_001"
    async with websockets.connect(uri) as websocket:
        # 接收连接确认
        message = await websocket.recv()
        print(f"收到: {message}")

        # 发送 ping
        await websocket.send(json.dumps({"type": "ping"}))

        # 接收 pong
        pong = await websocket.recv()
        print(f"收到: {pong}")

        # 订阅频道
        await websocket.send(json.dumps({
            "type": "subscribe",
            "channels": ["state", "events"]
        }))

        # 接收订阅确认
        confirm = await websocket.recv()
        print(f"收到: {confirm}")

        print("✓ WebSocket 测试通过!")

asyncio.run(test_websocket())
```

运行测试：

```bash
# 安装 websockets 客户端（如果需要）
uv add --dev websockets

# 运行测试
uv run python test_websocket.py
```

---

## ✅ 测试检查清单

### 功能测试

- [ ] API 服务器正常启动
- [ ] 健康检查端点响应正常
- [ ] 设备列表查询正常
- [ ] 设备状态查询正常
- [ ] 命令发送成功
- [ ] 操作日志记录正常
- [ ] 对话记录查询正常
- [ ] 统计数据计算正确
- [ ] WebSocket 连接成功
- [ ] WebSocket 消息收发正常
- [ ] 数据库 CRUD 操作正常
- [ ] 数据清理功能正常

### 测试覆盖率

- [ ] 数据库测试: 19/19 通过
- [ ] API 测试: 27/27 通过
- [ ] WebSocket 测试: 22/22 通过
- [ ] E2E 测试: 11/11 通过
- [ ] 总覆盖率 > 60%

### 性能测试（可选）

- [ ] 并发请求处理正常
- [ ] 响应时间合理（< 100ms）
- [ ] 无内存泄漏
- [ ] 服务器稳定性良好

---

## 🐛 故障排除

### 问题 1: 端口 8000 被占用

```bash
# 查找占用端口的进程
lsof -ti:8000

# 杀死进程
kill -9 $(lsof -ti:8000)
```

### 问题 2: 数据库锁定错误

```bash
# 删除测试数据库
rm -f lelamp.db lelamp-test.db

# 使用内存数据库运行测试
export LELAMP_DATABASE_URL="sqlite:///:memory:"
```

### 问题 3: 依赖缺失

```bash
# 重新安装依赖
uv sync --extra api --reinstall
```

### 问题 4: WebSocket 连接失败

```bash
# 检查服务器日志
tail -f /tmp/lelamp_api.log

# 确认 WebSocket 端点可用
curl -i -N \
  -H "Connection: Upgrade" \
  -H "Upgrade: websocket" \
  http://localhost:8000/api/ws/lelamp_001
```

### 问题 5: 测试失败

```bash
# 运行特定测试并查看详细输出
uv run pytest lelamp/test/integration/test_api.py::test_get_device_state -vv -s

# 查看详细错误信息
uv run pytest lelamp/test/integration/ --tb=long
```

---

## 📊 测试结果示例

### 成功的测试输出

```
============================= test session starts ==============================
collected 79 items

test_database.py ...................                                 [ 27%]
test_api.py ...........................                           [ 61%]
test_websocket.py ......................                          [ 89%]
test_e2e.py ...........                                           [100%]

============================== 79 passed in 2.34s ===============================

---------- coverage: platform darwin, python 3.12.5 -----------
Name                             Stmts   Miss  Cover   Missing
--------------------------------------------------------------
lelamp/api/app.py                   89      5    94%   23-27
lelamp/api/routes/devices.py       234     12    95%   145-156
lelamp/database/crud.py            312     45    86%   78-92
lelamp/database/models.py          156      8    95%   123-134
--------------------------------------------------------------
TOTAL                             791     70    91%
```

---

## 📚 相关文档

- **API 文档**: `lelamp/api/API_DOCUMENTATION.md`
- **部署指南**: `lelamp/api/DEPLOYMENT_GUIDE.md`
- **使用示例**: `lelamp/api/EXAMPLES.md`
- **完整测试文档**: `lelamp/api/TESTING.md`
- **Phase 3 完成报告**: `docs/PHASE3_COMPLETION_REPORT.md`

---

## 🎯 下一步

测试通过后，你可以：

1. **集成到主应用**: 将 API 与 LiveKit Agent 集成
2. **前端对接**: 让 Web 前端调用新的 REST API
3. **部署到生产**: 参考 DEPLOYMENT_GUIDE.md
4. **性能优化**: 添加缓存、索引等优化
5. **监控告警**: 集成 Prometheus + Grafana

---

## 💡 提示

- 使用 `-v` 标志查看详细测试输出
- 使用 `-s` 标志查看 print 语句输出
- 使用 `--lf` 只运行上次失败的测试
- 使用 `--cov` 生成覆盖率报告
- 使用 `--tb=short` 减少错误回溯信息

测试愉快！ 🎉
