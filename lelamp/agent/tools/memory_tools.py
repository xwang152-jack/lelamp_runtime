"""
记忆管理工具 — Agent 的记忆能力接口

提供 save_memory / recall_memory / forget_memory 三个方法，
由 LeLamp Agent 类通过 @function_tool 委托调用。
"""

import logging

from lelamp.memory.store import MemoryStore

logger = logging.getLogger("lelamp.agent.tools.memory")


class MemoryTools:
    """记忆管理工具（由 Agent 的 @function_tool 方法委托调用）"""

    CONTENT_MAX_LENGTH = 500

    def __init__(self, memory_store: MemoryStore, *, lamp_id: str = "") -> None:
        self._store = memory_store
        self._lamp_id = lamp_id

    async def save_memory(self, content: str, category: str = "general") -> str:
        """
        记住一个重要信息（用户偏好、事实、上下文等）。

        Args:
            content: 要记住的内容 (max 500 chars)
            category: 分类 - preference(偏好)/fact(事实)/relationship(关系)/context(上下文)/general(通用)
        """
        if not content or not content.strip():
            return "内容为空，无法保存"

        content = content.strip()[: self.CONTENT_MAX_LENGTH]

        valid_categories = {"preference", "fact", "relationship", "context", "general"}
        if category not in valid_categories:
            category = "general"

        try:
            self._store.add_memory(
                lamp_id=self._lamp_id,
                content=content,
                category=category,
                importance=6,
                source="explicit",
            )
            return f"已记住: {content}"
        except Exception as e:
            logger.error(f"save_memory failed: {e}")
            return "记忆保存失败，请稍后再试"

    async def recall_memory(self, query: str) -> str:
        """
        搜索你的记忆，查找相关信息。

        Args:
            query: 搜索关键词
        """
        if not query or not query.strip():
            return "搜索关键词为空"

        try:
            memories = self._store.search_memories(
                lamp_id=self._lamp_id,
                query=query.strip(),
                limit=10,
            )
            if not memories:
                return "没有找到相关记忆"

            lines = [f"找到 {len(memories)} 条相关记忆:"]
            for m in memories:
                lines.append(f"- [{m.category}] {m.content}")
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"recall_memory failed: {e}")
            return "记忆搜索失败，请稍后再试"

    async def forget_memory(self, content_hint: str) -> str:
        """
        删除一条记忆（当信息过时或不再相关时使用）。

        Args:
            content_hint: 要删除的记忆的关键词
        """
        if not content_hint or not content_hint.strip():
            return "关键词为空，无法删除"

        try:
            count = self._store.deactivate_by_content_hint(
                lamp_id=self._lamp_id,
                content_hint=content_hint.strip(),
            )
            if count == 0:
                return f"没有找到包含「{content_hint}」的记忆"
            return f"已删除 {count} 条记忆"
        except Exception as e:
            logger.error(f"forget_memory failed: {e}")
            return "记忆删除失败，请稍后再试"
