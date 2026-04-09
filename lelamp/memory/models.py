"""
记忆系统数据库模型

双层记忆架构（参考 nanobot）：
- Memory: 长期记忆，每次对话自动注入 system prompt
- ConversationSummary: 对话摘要，按需搜索查询
"""
from datetime import datetime, UTC

from sqlalchemy import (
    Boolean,
    Index,
    Integer,
    String,
    DateTime,
    JSON,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from lelamp.database.base import Base


class Memory(Base):
    """
    长期记忆存储（对应 nanobot 的 MEMORY.md）

    存储: 用户偏好、事实、关系、上下文等持久化信息。
    每次对话时按 importance 排序注入 system prompt，受 token 预算控制。
    """

    __tablename__ = "memories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    lamp_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    category: Mapped[str] = mapped_column(
        String(50), nullable=False, default="general"
    )
    # 分类: preference / fact / relationship / context / general
    content: Mapped[str] = mapped_column(String(1000), nullable=False)
    importance: Mapped[int] = mapped_column(
        Integer, nullable=False, default=5
    )
    # importance: 1-10，越高越重要，用于 token 预算分配
    source: Mapped[str] = mapped_column(
        String(50), nullable=False, default="explicit"
    )
    # source: explicit(LLM 工具调用) / auto(自动整合) / system
    access_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    last_accessed: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC), nullable=False
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, default=True, nullable=False
    )

    __table_args__ = (
        Index("ix_memories_lamp_id_category", "lamp_id", "category"),
        Index("ix_memories_lamp_id_active", "lamp_id", "is_active"),
        Index("ix_memories_lamp_id_importance", "lamp_id", "importance"),
    )

    def __repr__(self) -> str:
        return f"<Memory(id={self.id}, lamp_id={self.lamp_id}, category={self.category}, content={self.content[:30]}...)>"


class ConversationSummary(Base):
    """
    对话摘要（对应 nanobot 的 HISTORY.md）

    存储: 对话的摘要和关键话题标签，支持按需搜索。
    """

    __tablename__ = "conversation_summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    lamp_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    session_id: Mapped[str] = mapped_column(String(100), nullable=False)
    summary: Mapped[str] = mapped_column(Text(5000), nullable=False)
    key_topics: Mapped[list] = mapped_column(
        JSON, nullable=False, default=list
    )
    # key_topics: ["作业", "数学", "鼓励"] — 用于搜索的标签
    message_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    ended_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), nullable=False
    )

    __table_args__ = (
        Index("ix_conv_summaries_lamp_id_timestamp", "lamp_id", "ended_at"),
    )

    def __repr__(self) -> str:
        return f"<ConversationSummary(id={self.id}, lamp_id={self.lamp_id}, session_id={self.session_id})>"
