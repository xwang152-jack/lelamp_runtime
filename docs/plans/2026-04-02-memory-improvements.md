# Memory System Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现 5 项记忆系统改进：last_consolidated 偏移量、token-budget 触发、矛盾检测、失败降级保留、对话摘要注入。

**Architecture:**
- `consolidator.py` 接收改动最多：token-budget 判断、矛盾检测 prompt、失败计数+降级存储
- `agent/lelamp_agent.py` 增加 `_consolidation_offset` 偏移量追踪、token-budget 触发路径、摘要注入
- `memory/config.py` 新增三个可选配置项

**Tech Stack:** Python asyncio, SQLAlchemy, httpx, pytest, SQLite

---

## Task 1: last_consolidated 偏移量

### 涉及文件
- Modify: `lelamp/agent/lelamp_agent.py:348-549`（记忆系统初始化和整合相关方法）
- Modify: `lelamp/memory/consolidator.py:108-130`（移除硬编码的 `[-20:]` 截取）
- Test: `tests/test_memory.py`

### 背景
当前 `_run_consolidation()` 每次都传 `conversation_turns[-20:]`，即使前面的轮次已经整合过。这会触发重复 LLM 调用，产生噪音（尽管 bigram 去重能拦截）。

应该只传 `turns[offset:]`（未整合的新轮次），整合后把 offset 推进到当前长度。

### Step 1: 在 `lelamp_agent.py` 的 `__init__` 中初始化偏移量

在第 349 行附近（`self._conversation_turns: list[dict] = []` 这行后面）添加：

```python
self._conversation_turns: list[dict] = []
self._consolidation_offset: int = 0  # 已整合到哪一轮（turns[offset:] 是待整合的新轮次）
```

### Step 2: 修改 `_run_consolidation()` 只传新轮次

将 `lelamp_agent.py:532-548` 的 `_run_consolidation` 方法改为：

```python
async def _run_consolidation(self) -> None:
    """后台执行记忆整合（非阻塞）"""
    try:
        # 只整合 offset 之后的新轮次
        new_turns = list(self._conversation_turns[self._consolidation_offset:])
        if not new_turns:
            return

        result = await self._memory_consolidator.consolidate(
            lamp_id=self._lamp_id,
            session_id=self._session_id,
            conversation_turns=new_turns,
        )
        if result:
            # 整合成功：推进偏移量到当前总轮数
            self._consolidation_offset = len(self._conversation_turns)
            if result.new_memories_count > 0:
                new_instructions = self._build_dynamic_instructions()
                await self.update_instructions(new_instructions)
                logger.info(f"Memory consolidated: {result.new_memories_count} new memories")
    except Exception as e:
        logger.warning(f"Background consolidation failed (non-critical): {e}")
```

注意：同时**删除**原来最后一行 `self._conversation_turns = turns[-5:]`，不再截断历史（offset 机制已经避免重复处理）。

### Step 3: 修改 `consolidator.py` 的 `consolidate()` 移除内部截取

将第 127 行：
```python
recent_turns = conversation_turns[-20:]
```
改为：
```python
# 调用方负责传入未整合的轮次；内部最多保留 40 条防止 prompt 过大
recent_turns = conversation_turns[-40:]
```

### Step 4: 在现有测试文件里补充 offset 相关测试

在 `tests/test_memory.py` 末尾追加：

```python
# ==================== last_consolidated offset 测试 ====================

@pytest.mark.asyncio
async def test_consolidation_offset_only_new_turns(store):
    """整合后 offset 推进，下次整合只处理新轮次"""
    from unittest.mock import AsyncMock, patch
    from lelamp.memory.consolidator import MemoryConsolidator, ConsolidationResult

    consolidator = MemoryConsolidator(
        base_url="http://localhost",
        api_key="test",
        model="test-model",
        memory_store=store,
    )

    # 模拟整合返回结果
    mock_result = ConsolidationResult(new_memories_count=1, summary_saved=True)

    with patch.object(consolidator, "consolidate", new_callable=AsyncMock) as mock_consolidate:
        mock_consolidate.return_value = mock_result

        turns = [{"role": "user", "content": f"msg{i}"} for i in range(12)]
        offset = 0

        # 第一次整合：传 turns[0:]
        new_turns = turns[offset:]
        result = await consolidator.consolidate("lamp1", "sess1", new_turns)
        offset = len(turns)  # offset 推进到 12

        # 继续对话，新增 5 条
        turns += [{"role": "user", "content": f"new{i}"} for i in range(5)]

        # 第二次整合：只传 turns[12:] — 即 5 条新轮次
        new_turns = turns[offset:]
        assert len(new_turns) == 5, f"应只传 5 条新轮次，实际 {len(new_turns)}"
```

### Step 5: 运行测试

```bash
uv run pytest tests/test_memory.py -v -k "offset"
```

期望：PASS（新测试通过，现有测试不受影响）

### Step 6: 提交

```bash
git add lelamp/agent/lelamp_agent.py lelamp/memory/consolidator.py tests/test_memory.py
git commit -m "feat(memory): 添加 last_consolidated offset 偏移量追踪，避免重复整合已处理轮次"
```

---

## Task 2: Token-budget 触发策略

### 涉及文件
- Modify: `lelamp/memory/config.py`
- Modify: `lelamp/memory/consolidator.py`
- Modify: `lelamp/agent/lelamp_agent.py`（`_check_consolidation`）
- Test: `tests/test_memory.py`

### 背景
固定轮数触发（10 轮）对语音对话不适用：语音单轮可能极短（"好的"），或极长（用户说了一大段话）。
应该在超过 context token 预算的 70% 时触发整合，无论轮数多少。

### Step 1: 在 `memory/config.py` 新增配置项

将整个 `MemoryConfig` dataclass 替换为：

```python
@dataclass(frozen=True)
class MemoryConfig:
    """记忆系统配置"""

    enabled: bool = True
    token_budget: int = 400
    consolidation_min_turns: int = 10
    consolidation_cooldown_s: float = 300.0
    max_content_length: int = 500
    # token-budget 触发配置
    context_token_limit: int = 8000     # 整体 context 窗口估算上限
    context_budget_ratio: float = 0.7   # 超过此比例时触发整合
    summary_token_budget: int = 200     # 摘要注入 token 预算（Task 5 使用）
```

同时在 `load_memory_config()` 中补充加载：

```python
def load_memory_config() -> MemoryConfig:
    return MemoryConfig(
        enabled=_get_env_bool("LELAMP_MEMORY_ENABLED", True),
        token_budget=_get_env_int("LELAMP_MEMORY_TOKEN_BUDGET", 400),
        consolidation_min_turns=_get_env_int(
            "LELAMP_MEMORY_CONSOLIDATION_MIN_TURNS", 10
        ),
        consolidation_cooldown_s=_get_env_float(
            "LELAMP_MEMORY_CONSOLIDATION_COOLDOWN_S", 300.0
        ),
        max_content_length=_get_env_int("LELAMP_MEMORY_MAX_CONTENT_LENGTH", 500),
        context_token_limit=_get_env_int("LELAMP_MEMORY_CONTEXT_TOKEN_LIMIT", 8000),
        context_budget_ratio=_get_env_float("LELAMP_MEMORY_CONTEXT_BUDGET_RATIO", 0.7),
        summary_token_budget=_get_env_int("LELAMP_MEMORY_SUMMARY_TOKEN_BUDGET", 200),
    )
```

### Step 2: 在 `consolidator.py` 添加 `should_consolidate_by_tokens()` 方法

在 `should_consolidate()` 方法后面（约第 107 行）插入：

```python
def should_consolidate_by_tokens(
    self,
    conversation_turns: list[dict],
) -> bool:
    """
    基于 token 估算判断是否应触发整合。

    当未整合轮次的估算 token 数超过 context 预算的阈值时触发，
    与 should_consolidate（轮数）互补，两者满足其一即整合。
    """
    if not conversation_turns:
        return False

    # 中文 ~0.67 chars/token 的粗略估算
    estimated = sum(len(t.get("content", "")) / 0.67 for t in conversation_turns)
    threshold = self._context_token_limit * self._context_budget_ratio

    return estimated > threshold
```

同时在 `__init__` 中保存这两个配置：

```python
# 在 __init__ 末尾补充（load_memory_config 已经在调用）：
config = load_memory_config()
self._min_turns = config.consolidation_min_turns
self._cooldown_s = config.consolidation_cooldown_s
self._context_token_limit = config.context_token_limit          # 新增
self._context_budget_ratio = config.context_budget_ratio        # 新增
```

（注意：`config = load_memory_config()` 在 `__init__` 末尾已经有一次调用，只需在该调用后补充两行赋值即可，不要重复调用。）

### Step 3: 修改 `lelamp_agent.py` 的 `_check_consolidation()`

将当前实现替换为（满足轮数条件 **或** token 预算条件，均触发）：

```python
async def _check_consolidation(self) -> None:
    """检查是否需要触发记忆整合（轮数触发 or token-budget 触发）"""
    if (
        not self._memory_initialized
        or not self._memory_consolidator
        or not self._session_id
    ):
        return

    # 只看未整合的新轮次
    new_turns = self._conversation_turns[self._consolidation_offset:]
    if not new_turns:
        return

    should = (
        self._memory_consolidator.should_consolidate(new_turns)
        or self._memory_consolidator.should_consolidate_by_tokens(new_turns)
    )
    if should:
        asyncio.create_task(self._run_consolidation())
```

### Step 4: 补充测试

在 `tests/test_memory.py` 末尾追加：

```python
# ==================== token-budget 触发测试 ====================

def test_should_consolidate_by_tokens_below_threshold(store):
    """token 数不足时不触发"""
    from lelamp.memory.consolidator import MemoryConsolidator

    consolidator = MemoryConsolidator(
        base_url="http://localhost",
        api_key="test",
        model="test-model",
        memory_store=store,
    )
    turns = [{"role": "user", "content": "好的"}] * 5  # 极短，估算很小
    assert consolidator.should_consolidate_by_tokens(turns) is False


def test_should_consolidate_by_tokens_above_threshold(store):
    """token 数超过阈值时触发"""
    from lelamp.memory.consolidator import MemoryConsolidator

    consolidator = MemoryConsolidator(
        base_url="http://localhost",
        api_key="test",
        model="test-model",
        memory_store=store,
    )
    # context_token_limit=8000, ratio=0.7 → 阈值 5600 tokens
    # 每条 ~3760 chars → ~5612 tokens，超过阈值
    long_content = "这是一段很长的对话内容。" * 300  # ~3600 chars → ~5373 tokens
    turns = [{"role": "user", "content": long_content}, {"role": "assistant", "content": long_content}]
    assert consolidator.should_consolidate_by_tokens(turns) is True


def test_should_consolidate_by_tokens_empty(store):
    """空列表不触发"""
    from lelamp.memory.consolidator import MemoryConsolidator

    consolidator = MemoryConsolidator(
        base_url="http://localhost",
        api_key="test",
        model="test-model",
        memory_store=store,
    )
    assert consolidator.should_consolidate_by_tokens([]) is False
```

### Step 5: 运行测试

```bash
uv run pytest tests/test_memory.py -v -k "token"
```

期望：3 个 PASS

### Step 6: 提交

```bash
git add lelamp/memory/config.py lelamp/memory/consolidator.py lelamp/agent/lelamp_agent.py tests/test_memory.py
git commit -m "feat(memory): 添加 token-budget 触发策略，补充 context_token_limit/ratio 配置项"
```

---

## Task 3: 矛盾检测（Prompt 层）

### 涉及文件
- Modify: `lelamp/memory/consolidator.py`（prompt + 响应处理）
- Test: `tests/test_memory.py`

### 背景
当前 bigram 去重只能识别"内容相近"，无法识别"内容矛盾"。
例如用户说"我今天想换冷色调"，而已有记忆是"用户偏好暖色调灯光"——两者语义矛盾但词汇不重叠，会共存于记忆库中。

### Step 1: 更新 `_CONSOLIDATION_SYSTEM` prompt

将 `consolidator.py` 第 28-49 行的 `_CONSOLIDATION_SYSTEM` 替换为：

```python
_CONSOLIDATION_SYSTEM = """你是一个记忆管理系统。你的任务是从对话记录中提取值得长期记住的信息。

规则：
1. 只提取用户明确表达的偏好、事实、关系和重要上下文
2. 不要提取临时性信息（如"今天天气不错"）
3. 不要提取已经很明显的信息
4. 如果新信息与旧信息矛盾（例如偏好改变），在 memories 中包含新信息，并在 obsolete_ids 中列出应该失效的旧记忆 id
5. 每条记忆控制在 100 字以内

请返回 JSON 格式：
{
  "memories": [
    {"content": "用户偏好冷色调灯光", "category": "preference", "importance": 7}
  ],
  "obsolete_ids": [123],
  "summary": "本次对话的简短摘要（2-3句话）",
  "topics": ["关键词1", "关键词2"]
}

category 可选值: preference(偏好), fact(事实), relationship(关系), context(上下文), general(通用)
importance 范围: 1-10，越高越重要
obsolete_ids: 应该失效的旧记忆 id 列表（整数），没有则返回空数组 []

如果没有值得记住的信息，返回空的 memories 数组。"""
```

### Step 2: 在 `consolidate()` 中传递记忆 id，并处理 `obsolete_ids`

将 `consolidator.py` 中构建 `existing_text` 的部分（约第 132-138 行）改为包含 id：

```python
existing_text = ""
if existing:
    lines = ["已有记忆（避免重复，如有矛盾请列入 obsolete_ids）:"]
    for m in existing:
        lines.append(f"- [id={m.id}] [{m.category}] {m.content}")
    existing_text = "\n".join(lines) + "\n\n"
```

### Step 3: 在 `consolidate()` 的响应处理中处理 `obsolete_ids`

在 `# 提取记忆` 注释（约第 179 行）之后，在调用 `_deduplicate_memories` 之前，插入：

```python
# 处理矛盾记忆失效
obsolete_ids = data.get("obsolete_ids", [])
if obsolete_ids:
    for oid in obsolete_ids:
        try:
            self._store.deactivate_memory(int(oid))
        except (ValueError, TypeError, Exception) as e:
            logger.warning(f"Failed to deactivate obsolete memory id={oid}: {e}")
    logger.info(f"Deactivated {len(obsolete_ids)} obsolete memories due to contradiction")
```

### Step 4: 补充测试

在 `tests/test_memory.py` 末尾追加：

```python
# ==================== 矛盾检测测试 ====================

@pytest.mark.asyncio
async def test_consolidation_deactivates_obsolete_memories(store):
    """整合时 obsolete_ids 中的旧记忆被软删除"""
    import json
    from unittest.mock import patch, AsyncMock
    from lelamp.memory.consolidator import MemoryConsolidator

    # 预先创建一条旧记忆
    old_mem = store.add_memory("lamp1", "用户偏好暖色调灯光", category="preference", importance=7)
    old_id = old_mem.id

    consolidator = MemoryConsolidator(
        base_url="http://localhost",
        api_key="test",
        model="test-model",
        memory_store=store,
    )

    # LLM 返回矛盾检测结果，列出旧记忆 id 为废弃
    mock_response = json.dumps({
        "memories": [{"content": "用户偏好冷色调灯光", "category": "preference", "importance": 7}],
        "obsolete_ids": [old_id],
        "summary": "用户改变了灯光偏好",
        "topics": ["灯光", "偏好"],
    })

    with patch.object(consolidator, "_call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = mock_response
        turns = [{"role": "user", "content": "我改成冷色调了"}]
        result = await consolidator.consolidate("lamp1", "sess1", turns)

    # 旧记忆应已软删除
    active = store.get_all_active_memories("lamp1")
    active_ids = [m.id for m in active]
    assert old_id not in active_ids, "旧的矛盾记忆应已被 deactivate"
    # 新记忆应已存入
    assert any("冷色调" in m.content for m in active), "新记忆应已保存"
```

### Step 5: 运行测试

```bash
uv run pytest tests/test_memory.py -v -k "obsolete"
```

期望：PASS

### Step 6: 提交

```bash
git add lelamp/memory/consolidator.py tests/test_memory.py
git commit -m "feat(memory): 添加矛盾检测，整合时通过 obsolete_ids 软删除已过时的旧记忆"
```

---

## Task 4: 整合失败降级保留

### 涉及文件
- Modify: `lelamp/memory/consolidator.py`

### 背景
Raspberry Pi 上网络不稳定，LLM 调用可能连续失败。目前失败时直接返回 None，本轮对话信息完全丢失。
应在失败 3 次后自动降级：直接把对话文本拼接为摘要存入 `ConversationSummary`，不依赖 LLM。

### Step 1: 在 `__init__` 中添加失败计数器

在 `MemoryConsolidator.__init__` 末尾（`self._cooldown_s = ...` 之后）添加：

```python
self._llm_failure_count: int = 0        # 连续 LLM 调用失败次数
self._max_llm_failures: int = 3         # 超过此数触发降级
```

### Step 2: 在 `consolidate()` 中包裹失败降级逻辑

将 `consolidate()` 方法中调用 `_call_llm` 之后的处理逻辑，改为：

```python
# 调用 LLM
llm_response = await self._call_llm(_CONSOLIDATION_SYSTEM, user_prompt)
if not llm_response:
    self._llm_failure_count += 1
    if self._llm_failure_count >= self._max_llm_failures:
        # 降级：直接 raw-archive，重置失败计数
        self._llm_failure_count = 0
        logger.warning(
            f"LLM consolidation failed {self._max_llm_failures} times, "
            "falling back to raw archive"
        )
        return self._raw_archive(lamp_id, session_id, recent_turns)
    return None

# LLM 成功：重置失败计数
self._llm_failure_count = 0
```

### Step 3: 添加 `_raw_archive()` 方法

在 `_call_llm()` 方法之后添加：

```python
def _raw_archive(
    self,
    lamp_id: str,
    session_id: str,
    turns: list[dict],
) -> ConsolidationResult:
    """
    降级存储：LLM 多次失败时直接把对话拼接为摘要存入 ConversationSummary。
    不提取长期记忆，仅保留对话历史以防数据丢失。
    """
    user_lines = [
        t["content"][:80]
        for t in turns
        if t.get("role") == "user" and t.get("content")
    ]
    fallback_summary = "【自动降级摘要】" + " | ".join(user_lines[:5])

    try:
        now = datetime.utcnow()
        self._store.save_summary(
            lamp_id=lamp_id,
            session_id=session_id,
            summary=fallback_summary[:500],
            key_topics=[],
            message_count=len(turns),
            started_at=now,
            ended_at=now,
        )
        logger.info(f"Raw archive saved for session {session_id}")
        return ConsolidationResult(new_memories_count=0, summary_saved=True)
    except Exception as e:
        logger.error(f"Raw archive failed: {e}")
        return ConsolidationResult(new_memories_count=0, summary_saved=False)
```

### Step 4: 补充测试

在 `tests/test_memory.py` 末尾追加：

```python
# ==================== 失败降级测试 ====================

@pytest.mark.asyncio
async def test_consolidation_raw_archive_on_repeated_failure(store):
    """LLM 连续失败 3 次后触发 raw archive"""
    from unittest.mock import patch, AsyncMock
    from lelamp.memory.consolidator import MemoryConsolidator

    consolidator = MemoryConsolidator(
        base_url="http://localhost",
        api_key="test",
        model="test-model",
        memory_store=store,
    )

    turns = [
        {"role": "user", "content": "我喜欢安静的音乐"},
        {"role": "assistant", "content": "好的"},
    ]

    with patch.object(consolidator, "_call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = None  # 持续失败

        # 前 2 次失败不降级
        result1 = await consolidator.consolidate("lamp1", "sess_fail", turns)
        assert result1 is None

        result2 = await consolidator.consolidate("lamp1", "sess_fail", turns)
        assert result2 is None

        # 第 3 次触发降级
        result3 = await consolidator.consolidate("lamp1", "sess_fail", turns)
        assert result3 is not None
        assert result3.summary_saved is True
        assert result3.new_memories_count == 0

    # 检查 raw archive 已存入摘要
    summaries = store.get_recent_summaries("lamp1", hours=1, limit=5)
    assert any("自动降级摘要" in s.summary for s in summaries)

    # 失败计数应已重置
    assert consolidator._llm_failure_count == 0
```

### Step 5: 运行测试

```bash
uv run pytest tests/test_memory.py -v -k "raw_archive"
```

期望：PASS

### Step 6: 提交

```bash
git add lelamp/memory/consolidator.py tests/test_memory.py
git commit -m "feat(memory): LLM 连续失败 3 次后降级 raw archive，避免对话数据丢失"
```

---

## Task 5: ConversationSummary 注入

### 涉及文件
- Modify: `lelamp/agent/lelamp_agent.py`（`_build_dynamic_instructions()`）
- Test: `tests/test_memory.py`

### 背景
目前只有长期记忆（`Memory`）被注入 system prompt。`ConversationSummary` 只有通过 `recall_memory` tool 才能查询，Agent 不会主动回顾历史。
注入最近 2 条对话摘要，能让 Agent 对上次聊到的内容有感知（"上次你说过..."）。

### Step 1: 修改 `_build_dynamic_instructions()` 加入摘要注入

将整个方法替换为：

```python
def _build_dynamic_instructions(self) -> str:
    """构建带记忆上下文和近期摘要的动态 system prompt"""
    base = self._INSTRUCTIONS

    if not self._memory_initialized or not self._memory_store:
        return base

    sections: list[str] = []

    # --- 长期记忆（按 importance 排序，400 token 预算）---
    try:
        token_budget = int(os.getenv("LELAMP_MEMORY_TOKEN_BUDGET", "400"))
        memories = self._memory_store.get_active_memories(
            lamp_id=self._lamp_id,
            max_tokens=token_budget,
        )
        if memories:
            lines = ["\n\n# Memory", "You remember the following about this user:"]
            for m in memories:
                lines.append(f"- [{m.category}] {m.content}")
            lines.append(
                "\nUse these memories naturally in conversation. "
                "Use save_memory() to remember new important things."
            )
            sections.append("\n".join(lines))
    except Exception as e:
        logger.warning(f"Failed to load memories for prompt: {e}")

    # --- 近期对话摘要（最近 72h，最多 2 条，200 token 预算）---
    try:
        summary_token_budget = int(os.getenv("LELAMP_MEMORY_SUMMARY_TOKEN_BUDGET", "200"))
        summaries = self._memory_store.get_recent_summaries(
            lamp_id=self._lamp_id,
            hours=72,
            limit=2,
        )
        if summaries:
            used = 0
            lines = ["\n\n# Recent Conversations", "Summary of recent sessions with this user:"]
            for s in summaries:
                entry = f"- {s.summary}"
                est = len(entry) / 0.67 + 5
                if used + est > summary_token_budget:
                    break
                lines.append(entry)
                used += est
            if len(lines) > 2:  # 有实际摘要内容
                sections.append("\n".join(lines))
    except Exception as e:
        logger.warning(f"Failed to load summaries for prompt: {e}")

    if not sections:
        return base

    return base + "".join(sections)
```

### Step 2: 补充测试

在 `tests/test_memory.py` 末尾追加：

```python
# ==================== 摘要注入测试 ====================

def test_build_dynamic_instructions_includes_summary(store):
    """_build_dynamic_instructions 注入近期对话摘要"""
    from datetime import datetime, timedelta
    from unittest.mock import patch, MagicMock

    # 存入一条近期摘要
    now = datetime.utcnow()
    store.save_summary(
        lamp_id="lamp_test",
        session_id="sess_inject",
        summary="用户最近在练习钢琴，每天练习 30 分钟",
        key_topics=["钢琴", "练习"],
        message_count=5,
        started_at=now - timedelta(hours=1),
        ended_at=now,
    )

    # 构造最小化的 agent mock 来调用 _build_dynamic_instructions
    import os
    import sys
    sys.path.insert(0, ".")

    # 用 MagicMock 模拟 agent 的必要属性
    from lelamp.agent.lelamp_agent import LeLamp

    mock_agent = MagicMock(spec=LeLamp)
    mock_agent._memory_initialized = True
    mock_agent._memory_store = store
    mock_agent._lamp_id = "lamp_test"
    mock_agent._INSTRUCTIONS = "BASE"

    result = LeLamp._build_dynamic_instructions(mock_agent)

    assert "Recent Conversations" in result, "应包含摘要区块"
    assert "钢琴" in result, "应包含摘要内容"
```

### Step 3: 运行测试

```bash
uv run pytest tests/test_memory.py -v -k "summary"
```

期望：PASS

### Step 4: 运行完整记忆测试套件

```bash
uv run pytest tests/test_memory.py -v
```

期望：全部 PASS

### Step 5: 提交

```bash
git add lelamp/agent/lelamp_agent.py tests/test_memory.py
git commit -m "feat(memory): 注入近期对话摘要到 system prompt，提升跨 session 对话连贯性"
```

---

## Task 6: 最终验收

### Step 1: 运行完整测试套件

```bash
uv run pytest tests/ -v
```

期望：全部 PASS，无 regression

### Step 2: Lint 检查

```bash
uv run ruff check lelamp/memory/ lelamp/agent/lelamp_agent.py
```

期望：无错误（或只有预期的警告）

### Step 3: 验证配置读取

```bash
uv run python -c "
from lelamp.memory.config import load_memory_config
c = load_memory_config()
print('config ok:', c)
assert c.context_token_limit == 8000
assert c.context_budget_ratio == 0.7
assert c.summary_token_budget == 200
print('all assertions passed')
"
```

期望：打印 `all assertions passed`
