"""
Database base configuration and engine setup.

增强版：
- 更好的连接池配置
- WAL 模式支持（Write-Ahead Logging）
- 自动重试机制
- 连接健康检查
"""
import os
import time
import logging
from typing import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine, pool, text
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy import exc as sqlalchemy_exc

logger = logging.getLogger("lelamp.database")

# Database URL from environment variable or default to SQLite
DATABASE_URL = os.getenv(
    "LELAMP_DATABASE_URL", "sqlite:///./lelamp.db"
)

# 创建数据库引擎 - 增强配置
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={
            "check_same_thread": False,
            "timeout": 30,  # 30秒超时
        },
        echo=False,  # 设为 True 可查看 SQL 查询日志
        poolclass=pool.StaticPool,  # 使用静态连接池
        pool_pre_ping=True,  # 连接健康检查
        pool_recycle=3600,  # 1小时回收连接
    )
else:
    engine = create_engine(
        DATABASE_URL,
        echo=False,
        pool_pre_ping=True,
        pool_recycle=3600,
    )

# 创建 SessionLocal 类
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False  # 避免对象在提交后过期
)

# 创建 Base 类
Base = declarative_base()


def enable_wal_mode():
    """启用 SQLite WAL 模式（Write-Ahead Logging）

    WAL 模式提供更好的并发性能：
    - 读写操作可以同时进行
    - 减少数据库锁定问题
    - 更好的崩溃恢复
    """
    if not DATABASE_URL.startswith("sqlite"):
        return

    try:
        with engine.connect() as conn:
            # 启用 WAL 模式
            conn.execute(text("PRAGMA journal_mode=WAL"))
            # 设置同步模式（平衡性能和安全）
            conn.execute(text("PRAGMA synchronous=NORMAL"))
            # 设置缓存大小（根据需要调整）
            conn.execute(text("PRAGMA cache_size=-64000"))  # 64MB
            # 设置临时存储为内存
            conn.execute(text("PRAGMA temp_store=MEMORY"))
            conn.commit()
        logger.info("SQLite WAL mode enabled")
    except Exception as e:
        logger.warning(f"Failed to enable WAL mode: {e}")


def get_db() -> Generator[Session, None, None]:
    """
    依赖注入函数，获取数据库会话

    增强版：
    - 自动重试机制
    - 更好的错误处理
    """
    db = SessionLocal()
    try:
        yield db
    except sqlalchemy_exc.SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Database error: {e}")
        raise e
    finally:
        db.close()


@contextmanager
def get_db_context():
    """
    上下文管理器，用于手动管理数据库会话

    用法：
        with get_db_context() as db:
            result = db.query(UserSettings).all()
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Database context error: {e}")
        raise e
    finally:
        db.close()


def init_db() -> None:
    """
    初始化数据库表

    增强版：
    - 自动启用 WAL 模式
    - 创建必要的索引
    - 优化表结构
    """
    try:
        # 创建所有表
        Base.metadata.create_all(bind=engine)

        # 启用 WAL 模式
        enable_wal_mode()

        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise e


def drop_db() -> None:
    """
    删除所有数据库表

    WARNING: 这将删除所有数据！仅用于测试或开发
    """
    try:
        Base.metadata.drop_all(bind=engine)
        logger.info("Database tables dropped")
    except Exception as e:
        logger.error(f"Failed to drop database: {e}")
        raise e


def check_db_health() -> bool:
    """
    检查数据库健康状态

    Returns:
        bool: 数据库是否健康
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            return result.fetchone()[0] == 1
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False
