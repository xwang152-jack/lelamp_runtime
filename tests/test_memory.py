"""
记忆系统单元测试

测试 Memory 模型、MemoryStore CRUD、MemoryConfig 和 MemoryTools。
MemoryConsolidator 的 LLM 调用通过 mock 测试。
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


# ==================== Fixtures ====================

@pytest.fixture(scope="module")
def memory_engine():
    """创建内存 SQLite 引擎（记忆系统专用）"""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    from lelamp.database.base import Base
    from lelamp.memory.models import Memory, ConversationSummary  # noqa: F401
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def memory_db(memory_engine):
    """每个测试独立的 session"""
    SessionLocal = sessionmaker(bind=memory_engine, expire_on_commit=False)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
def store(memory_engine):
    """创建使用内存数据库的 MemoryStore，每个测试后清理数据"""
    from lelamp.memory.models import Memory, ConversationSummary

    SessionLocal = sessionmaker(bind=memory_engine, expire_on_commit=False)

    with patch("lelamp.memory.store.SessionLocal", SessionLocal):
        from lelamp.memory.store import MemoryStore
        s = MemoryStore()
        yield s
        # 清理所有数据
        db = SessionLocal()
        try:
            db.query(Memory).delete()
            db.query(ConversationSummary).delete()
            db.commit()
        finally:
            db.close()


# ==================== 模型测试 ====================

@pytest.mark.unit
class TestMemoryModel:
    """测试 Memory 模型"""

    def test_create_memory(self, memory_db):
        from lelamp.memory.models import Memory

        m = Memory(
            lamp_id="test",
            content="用户喜欢蓝色灯光",
            category="preference",
            importance=7,
        )
        memory_db.add(m)
        memory_db.commit()

        assert m.id is not None
        assert m.lamp_id == "test"
        assert m.content == "用户喜欢蓝色灯光"
        assert m.category == "preference"
        assert m.importance == 7
        assert m.is_active is True
        assert m.access_count == 0

    def test_create_conversation_summary(self, memory_db):
        from lelamp.memory.models import ConversationSummary

        now = datetime.utcnow()
        s = ConversationSummary(
            lamp_id="test",
            session_id="sess_123",
            summary="用户问了关于作业的问题",
            key_topics=["作业", "数学"],
            message_count=5,
            started_at=now,
            ended_at=now,
        )
        memory_db.add(s)
        memory_db.commit()

        assert s.id is not None
        assert s.session_id == "sess_123"
        assert s.key_topics == ["作业", "数学"]


# ==================== MemoryStore 测试 ====================

@pytest.mark.unit
class TestMemoryStore:
    """测试 MemoryStore CRUD 操作"""

    def test_add_and_get_memory(self, store):
        store.add_memory("lamp_1", "用户喜欢暖色调", category="preference", importance=8)
        store.add_memory("lamp_1", "用户是个小学生", category="fact", importance=6)

        memories = store.get_active_memories("lamp_1", max_tokens=1000)
        assert len(memories) == 2
        # 高 importance 排在前面
        assert memories[0].importance >= memories[1].importance

    def test_add_memory_clamps_importance(self, store):
        store.add_memory("lamp_1", "测试高", importance=100)
        store.add_memory("lamp_1", "测试低", importance=-5)

        memories = store.get_all_active_memories("lamp_1")
        assert memories[0].importance == 10
        assert memories[1].importance == 1

    def test_token_budget(self, store):
        """验证 token 预算控制"""
        for i in range(20):
            store.add_memory("lamp_1", f"记忆内容{i}：" + "测试" * 50, importance=5 + i)

        memories = store.get_active_memories("lamp_1", max_tokens=100)
        # 预算很小，只应该返回少量记忆
        assert len(memories) < 10

    def test_search_memories(self, store):
        store.add_memory("lamp_1", "用户喜欢蓝色灯光", category="preference")
        store.add_memory("lamp_1", "用户喜欢吃苹果", category="fact")
        store.add_memory("lamp_1", "今天天气不错", category="general")

        results = store.search_memories("lamp_1", "灯光")
        assert len(results) == 1
        assert "蓝色" in results[0].content

        results = store.search_memories("lamp_1", "用户")
        assert len(results) == 2

    def test_deactivate_memory(self, store):
        m = store.add_memory("lamp_1", "临时信息")
        assert store.deactivate_memory(m.id) is True

        memories = store.get_active_memories("lamp_1")
        assert len(memories) == 0

    def test_deactivate_by_content_hint(self, store):
        store.add_memory("lamp_1", "用户喜欢蓝色灯光")
        store.add_memory("lamp_1", "蓝色是天空的颜色")
        store.add_memory("lamp_1", "用户喜欢吃苹果")

        count = store.deactivate_by_content_hint("lamp_1", "蓝色")
        assert count == 2

        memories = store.get_active_memories("lamp_1")
        assert len(memories) == 1

    def test_update_memory(self, store):
        m = store.add_memory("lamp_1", "用户喜欢蓝色", importance=5)
        updated = store.update_memory(m.id, content="用户现在喜欢绿色", importance=9)
        assert updated is not None
        assert updated.content == "用户现在喜欢绿色"
        assert updated.importance == 9

    def test_save_and_get_summary(self, store):
        now = datetime.utcnow()
        store.save_summary(
            lamp_id="lamp_1",
            session_id="sess_1",
            summary="讨论了数学作业",
            key_topics=["数学", "作业"],
            message_count=10,
            started_at=now,
            ended_at=now,
        )

        summaries = store.get_recent_summaries("lamp_1")
        assert len(summaries) == 1
        assert "数学" in summaries[0].summary

    def test_get_recent_summaries_hours_filter(self, store):
        now = datetime.utcnow()
        old = now - timedelta(hours=200)

        store.save_summary(
            "lamp_1", "sess_old", "旧对话", [], 5, old, old,
        )
        store.save_summary(
            "lamp_1", "sess_new", "新对话", [], 3, now, now,
        )

        summaries = store.get_recent_summaries("lamp_1", hours=168)
        assert len(summaries) == 1
        assert summaries[0].session_id == "sess_new"

    def test_isolation_by_lamp_id(self, store):
        store.add_memory("lamp_a", "A 的记忆")
        store.add_memory("lamp_b", "B 的记忆")

        assert len(store.get_active_memories("lamp_a")) == 1
        assert len(store.get_active_memories("lamp_b")) == 1

    def test_nonexistent_lamp_id(self, store):
        assert store.get_active_memories("nonexistent") == []
        assert store.search_memories("nonexistent", "anything") == []
        assert store.get_recent_summaries("nonexistent") == []


# ==================== MemoryConfig 测试 ====================

@pytest.mark.unit
class TestMemoryConfig:
    """测试记忆配置"""

    def test_default_config(self):
        from lelamp.memory.config import MemoryConfig

        config = MemoryConfig()
        assert config.enabled is True
        assert config.token_budget == 400
        assert config.consolidation_min_turns == 10
        assert config.consolidation_cooldown_s == 300.0

    def test_load_config_from_env(self):
        import os
        os.environ["LELAMP_MEMORY_ENABLED"] = "0"
        os.environ["LELAMP_MEMORY_TOKEN_BUDGET"] = "200"
        try:
            from lelamp.memory.config import load_memory_config
            from importlib import reload
            import lelamp.memory.config as cfg_mod
            reload(cfg_mod)
            config = cfg_mod.load_memory_config()
            assert config.enabled is False
            assert config.token_budget == 200
        finally:
            os.environ.pop("LELAMP_MEMORY_ENABLED", None)
            os.environ.pop("LELAMP_MEMORY_TOKEN_BUDGET", None)

    def test_frozen_config(self):
        from lelamp.memory.config import MemoryConfig

        config = MemoryConfig()
        with pytest.raises(AttributeError):
            config.enabled = False


# ==================== MemoryTools 测试 ====================

@pytest.mark.unit
class TestMemoryTools:
    """测试记忆工具"""

    @pytest.fixture
    def tools(self, store):
        from lelamp.agent.tools.memory_tools import MemoryTools
        return MemoryTools(memory_store=store, lamp_id="test_lamp")

    @pytest.mark.asyncio
    async def test_save_memory(self, tools):
        result = await tools.save_memory("用户喜欢暖色调灯光", "preference")
        assert "已记住" in result

        memories = tools._store.get_active_memories("test_lamp")
        assert len(memories) == 1
        assert "暖色调" in memories[0].content

    @pytest.mark.asyncio
    async def test_save_memory_empty(self, tools):
        result = await tools.save_memory("", "general")
        assert "为空" in result

    @pytest.mark.asyncio
    async def test_save_memory_invalid_category(self, tools):
        result = await tools.save_memory("测试内容", "invalid_cat")
        assert "已记住" in result
        memories = tools._store.get_active_memories("test_lamp")
        assert memories[0].category == "general"

    @pytest.mark.asyncio
    async def test_recall_memory(self, tools):
        tools._store.add_memory("test_lamp", "用户喜欢蓝色灯光")
        tools._store.add_memory("test_lamp", "用户是个小学生")

        result = await tools.recall_memory("灯光")
        assert "1 条" in result
        assert "蓝色" in result

    @pytest.mark.asyncio
    async def test_recall_memory_empty(self, tools):
        result = await tools.recall_memory("")
        assert "为空" in result

    @pytest.mark.asyncio
    async def test_recall_memory_not_found(self, tools):
        result = await tools.recall_memory("不存在的内容")
        assert "没有找到" in result

    @pytest.mark.asyncio
    async def test_forget_memory(self, tools):
        tools._store.add_memory("test_lamp", "过时的信息")
        result = await tools.forget_memory("过时")
        assert "已删除 1 条" in result

        memories = tools._store.get_active_memories("test_lamp")
        assert len(memories) == 0

    @pytest.mark.asyncio
    async def test_forget_memory_not_found(self, tools):
        result = await tools.forget_memory("不存在")
        assert "没有找到" in result


# ==================== MemoryConsolidator 测试 ====================

@pytest.mark.unit
class TestMemoryConsolidator:
    """测试记忆整合器"""

    @pytest.fixture
    def consolidator(self, store):
        from lelamp.memory.consolidator import MemoryConsolidator
        return MemoryConsolidator(
            base_url="https://api.deepseek.com",
            api_key="test_key",
            model="deepseek-chat",
            memory_store=store,
        )

    def test_should_consolidate_enough_turns(self, consolidator):
        turns = [{"role": "user", "content": f"消息{i}"} for i in range(15)]
        assert consolidator.should_consolidate(turns) is True

    def test_should_not_consolidate_few_turns(self, consolidator):
        turns = [{"role": "user", "content": f"消息{i}"} for i in range(3)]
        assert consolidator.should_consolidate(turns) is False

    def test_should_not_consolidate_cooldown(self, consolidator):
        turns = [{"role": "user", "content": f"消息{i}"} for i in range(15)]
        # 第一次通过（原子设置时间戳）
        assert consolidator.should_consolidate(turns) is True
        # 第二次被冷却阻止
        assert consolidator.should_consolidate(turns) is False

    def test_deduplicate_memories(self, consolidator):
        existing = [MagicMock(content="用户喜欢蓝色灯光", importance=5, id=1)]
        new_memories = [
            {"content": "用户喜欢蓝色灯光", "category": "preference", "importance": 5},
            {"content": "用户是个小学生", "category": "fact", "importance": 6},
        ]
        deduped = consolidator._deduplicate_memories(new_memories, existing)
        assert len(deduped) == 1
        assert "小学生" in deduped[0]["content"]

    def test_deduplicate_different_content(self, consolidator):
        """内容不同的记忆不应该被去重"""
        existing = [MagicMock(content="用户喜欢蓝色灯光", importance=5, id=1)]
        new_memories = [
            {"content": "用户喜欢吃苹果", "category": "fact", "importance": 5},
        ]
        deduped = consolidator._deduplicate_memories(new_memories, existing)
        assert len(deduped) == 1

    def test_content_overlap(self):
        from lelamp.memory.consolidator import MemoryConsolidator

        # 完全相同
        assert MemoryConsolidator._content_overlap("abc", "abc") == 1.0
        # 完全不同
        overlap = MemoryConsolidator._content_overlap("abc", "xyz")
        assert overlap == 0.0
        # 部分重叠
        overlap = MemoryConsolidator._content_overlap("abcdef", "abcxyz")
        assert 0 < overlap < 1
        # 中文重叠
        overlap = MemoryConsolidator._content_overlap("用户喜欢蓝色灯光", "用户喜欢蓝色灯光")
        assert overlap == 1.0
        # 中文不同内容
        overlap = MemoryConsolidator._content_overlap("用户喜欢蓝色灯光", "用户喜欢吃苹果")
        # 只共享 "用户" 和 "喜欢" 的 bigram
        assert 0 < overlap < 1
        # 空字符串
        assert MemoryConsolidator._content_overlap("", "abc") == 0.0

    @pytest.mark.asyncio
    async def test_consolidate_with_mock_llm(self, consolidator, store):
        """mock LLM 调用测试整合流程"""
        import json

        mock_response = json.dumps({
            "memories": [
                {"content": "用户偏好暖色调", "category": "preference", "importance": 7},
            ],
            "summary": "用户表达了灯光颜色偏好",
            "topics": ["灯光", "颜色"],
        }, ensure_ascii=False)

        with patch.object(
            consolidator, "_call_llm", return_value=mock_response
        ):
            turns = [{"role": "user", "content": "我喜欢暖色调的灯光"}] * 12
            result = await consolidator.consolidate("lamp_1", "sess_1", turns)

            assert result is not None
            assert result.new_memories_count == 1
            assert result.summary_saved is True

    @pytest.mark.asyncio
    async def test_consolidate_llm_failure(self, consolidator):
        """LLM 调用失败时返回 None"""
        with patch.object(consolidator, "_call_llm", return_value=None):
            turns = [{"role": "user", "content": "test"}] * 12
            result = await consolidator.consolidate("lamp_1", "sess_1", turns)
            assert result is None

    @pytest.mark.asyncio
    async def test_consolidate_json_parse_error(self, consolidator):
        """JSON 解析失败时返回 None"""
        with patch.object(consolidator, "_call_llm", return_value="not json at all"):
            turns = [{"role": "user", "content": "test"}] * 12
            result = await consolidator.consolidate("lamp_1", "sess_1", turns)
            assert result is None

    @pytest.mark.asyncio
    async def test_consolidate_json_in_code_block(self, consolidator, store):
        """测试从 markdown code block 中提取 JSON"""
        import json

        mock_response = '```json\n' + json.dumps({
            "memories": [{"content": "测试记忆", "category": "fact", "importance": 5}],
            "summary": "测试摘要",
            "topics": ["测试"],
        }, ensure_ascii=False) + '\n```'

        with patch.object(consolidator, "_call_llm", return_value=mock_response):
            turns = [{"role": "user", "content": "test"}] * 12
            result = await consolidator.consolidate("lamp_1", "sess_1", turns)
            assert result is not None
            assert result.new_memories_count == 1

    @pytest.mark.asyncio
    async def test_consolidate_empty_memories(self, consolidator, store):
        """LLM 返回空 memories 时正常处理"""
        import json

        mock_response = json.dumps({
            "memories": [],
            "summary": "闲聊",
            "topics": [],
        }, ensure_ascii=False)

        with patch.object(consolidator, "_call_llm", return_value=mock_response):
            turns = [{"role": "user", "content": "你好"}] * 12
            result = await consolidator.consolidate("lamp_1", "sess_1", turns)

            assert result is not None
            assert result.new_memories_count == 0
            assert result.summary_saved is True


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
    # 每条 ~3900 chars / 0.67 ≈ 5820 tokens，两条合计 ~11640 tokens，超过阈值
    long_content = "这是一段很长的对话内容。" * 300
    turns = [
        {"role": "user", "content": long_content},
        {"role": "assistant", "content": long_content},
    ]
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
