"""
记忆存储 — 线程安全的 CRUD 操作

使用 SQLAlchemy 操作 SQLite，遵循 threading.Lock 模式保证
asyncio 上下文和线程上下文的并发安全。
"""
import logging
import threading
from datetime import datetime, timedelta

from sqlalchemy import desc
from sqlalchemy.orm import Session

from lelamp.database.base import SessionLocal
from lelamp.memory.models import Memory, ConversationSummary

logger = logging.getLogger("lelamp.memory.store")

# 中文字符约 1.5 token/字符的粗略估算
_CHARS_PER_TOKEN = 0.67


def _make_session() -> Session:
    """创建 session，确保 expire_on_commit=False 防止 detached error"""
    return SessionLocal()


class MemoryStore:
    """线程安全的记忆 CRUD 操作"""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    # ==================== 长期记忆 ====================

    def add_memory(
        self,
        lamp_id: str,
        content: str,
        *,
        category: str = "general",
        importance: int = 5,
        source: str = "explicit",
    ) -> Memory:
        """新增一条记忆"""
        with self._lock:
            db = _make_session()
            try:
                memory = Memory(
                    lamp_id=lamp_id,
                    content=content[:1000],
                    category=category,
                    importance=max(1, min(10, importance)),
                    source=source,
                )
                db.add(memory)
                db.commit()
                return memory
            except Exception as e:
                db.rollback()
                logger.error(f"add_memory failed: {e}")
                raise
            finally:
                db.close()

    def get_active_memories(
        self, lamp_id: str, *, max_tokens: int = 400
    ) -> list[Memory]:
        """
        获取活跃记忆，按 importance 降序排列，受 token 预算控制。

        优先注入高 importance 的记忆，预算用尽时停止。
        """
        with self._lock:
            db = _make_session()
            try:
                rows = (
                    db.query(Memory)
                    .filter(
                        Memory.lamp_id == lamp_id,
                        Memory.is_active.is_(True),
                    )
                    .order_by(desc(Memory.importance))
                    .all()
                )

                result: list[Memory] = []
                used_tokens = 0
                for m in rows:
                    est = len(m.content) / _CHARS_PER_TOKEN + 10
                    if used_tokens + est > max_tokens:
                        break
                    result.append(m)
                    used_tokens += est

                # 更新访问时间和计数
                for m in result:
                    m.access_count += 1
                    m.last_accessed = datetime.utcnow()
                if result:
                    db.commit()

                # 使用 expire_on_commit=False 后属性可安全访问
                # expunge 使对象在 session 关闭后仍可访问
                for row in result:
                    db.expunge(row)
                return result
            except Exception as e:
                db.rollback()
                logger.error(f"get_active_memories failed: {e}")
                return []
            finally:
                db.close()

    def search_memories(
        self, lamp_id: str, query: str, *, limit: int = 10
    ) -> list[Memory]:
        """关键词搜索记忆（LIKE 查询，无向量数据库）"""
        with self._lock:
            db = _make_session()
            try:
                pattern = f"%{query}%"
                rows = (
                    db.query(Memory)
                    .filter(
                        Memory.lamp_id == lamp_id,
                        Memory.is_active.is_(True),
                        Memory.content.ilike(pattern),
                    )
                    .order_by(desc(Memory.importance))
                    .limit(limit)
                    .all()
                )
                for row in rows:
                    db.expunge(row)
                return rows
            except Exception as e:
                logger.error(f"search_memories failed: {e}")
                return []
            finally:
                db.close()

    def update_memory(
        self,
        memory_id: int,
        *,
        content: str | None = None,
        importance: int | None = None,
    ) -> Memory | None:
        """更新已有记忆"""
        with self._lock:
            db = _make_session()
            try:
                memory = db.query(Memory).filter(Memory.id == memory_id).first()
                if memory is None:
                    return None
                if content is not None:
                    memory.content = content[:1000]
                if importance is not None:
                    memory.importance = max(1, min(10, importance))
                db.commit()
                return memory
            except Exception as e:
                db.rollback()
                logger.error(f"update_memory failed: {e}")
                return None
            finally:
                db.close()

    def deactivate_memory(self, memory_id: int) -> bool:
        """软删除一条记忆"""
        with self._lock:
            db = _make_session()
            try:
                memory = db.query(Memory).filter(Memory.id == memory_id).first()
                if memory is None:
                    return False
                memory.is_active = False
                db.commit()
                return True
            except Exception as e:
                db.rollback()
                logger.error(f"deactivate_memory failed: {e}")
                return False
            finally:
                db.close()

    def deactivate_by_content_hint(
        self, lamp_id: str, content_hint: str
    ) -> int:
        """按内容关键词软删除记忆，返回删除数量"""
        with self._lock:
            db = _make_session()
            try:
                pattern = f"%{content_hint}%"
                count = (
                    db.query(Memory)
                    .filter(
                        Memory.lamp_id == lamp_id,
                        Memory.is_active.is_(True),
                        Memory.content.ilike(pattern),
                    )
                    .update({"is_active": False}, synchronize_session="fetch")
                )
                db.commit()
                return count
            except Exception as e:
                db.rollback()
                logger.error(f"deactivate_by_content_hint failed: {e}")
                return 0
            finally:
                db.close()

    def get_all_active_memories(self, lamp_id: str) -> list[Memory]:
        """获取所有活跃记忆（用于去重判断）"""
        with self._lock:
            db = _make_session()
            try:
                rows = (
                    db.query(Memory)
                    .filter(
                        Memory.lamp_id == lamp_id,
                        Memory.is_active.is_(True),
                    )
                    .all()
                )
                # expunge 使对象在 session 关闭后仍可访问属性
                for row in rows:
                    db.expunge(row)
                return rows
            except Exception as e:
                logger.error(f"get_all_active_memories failed: {e}")
                return []
            finally:
                db.close()

    # ==================== 对话摘要 ====================

    def save_summary(
        self,
        lamp_id: str,
        session_id: str,
        summary: str,
        key_topics: list[str],
        message_count: int,
        started_at: datetime,
        ended_at: datetime,
    ) -> ConversationSummary:
        """保存一条对话摘要"""
        with self._lock:
            db = _make_session()
            try:
                record = ConversationSummary(
                    lamp_id=lamp_id,
                    session_id=session_id,
                    summary=summary[:5000],
                    key_topics=key_topics,
                    message_count=message_count,
                    started_at=started_at,
                    ended_at=ended_at,
                )
                db.add(record)
                db.commit()
                return record
            except Exception as e:
                db.rollback()
                logger.error(f"save_summary failed: {e}")
                raise
            finally:
                db.close()

    def get_recent_summaries(
        self,
        lamp_id: str,
        *,
        hours: int = 168,
        limit: int = 10,
    ) -> list[ConversationSummary]:
        """获取近期对话摘要"""
        with self._lock:
            db = _make_session()
            try:
                since = datetime.utcnow() - timedelta(hours=hours)
                rows = (
                    db.query(ConversationSummary)
                    .filter(
                        ConversationSummary.lamp_id == lamp_id,
                        ConversationSummary.ended_at >= since,
                    )
                    .order_by(desc(ConversationSummary.ended_at))
                    .limit(limit)
                    .all()
                )
                for row in rows:
                    db.expunge(row)
                return rows
            except Exception as e:
                logger.error(f"get_recent_summaries failed: {e}")
                return []
            finally:
                db.close()

    def search_summaries(
        self, lamp_id: str, query: str, *, limit: int = 5
    ) -> list[ConversationSummary]:
        """搜索对话摘要（关键词匹配 summary 内容）"""
        with self._lock:
            db = _make_session()
            try:
                pattern = f"%{query}%"
                rows = (
                    db.query(ConversationSummary)
                    .filter(
                        ConversationSummary.lamp_id == lamp_id,
                        ConversationSummary.summary.ilike(pattern),
                    )
                    .order_by(desc(ConversationSummary.ended_at))
                    .limit(limit)
                    .all()
                )
                for row in rows:
                    db.expunge(row)
                return rows
            except Exception as e:
                logger.error(f"search_summaries failed: {e}")
                return []
            finally:
                db.close()
