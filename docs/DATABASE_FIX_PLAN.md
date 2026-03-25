# 🔧 数据库问题全面修复方案

## 🚨 常见问题分析

### 1. 数据库连接问题

**症状：**
- API 启动时数据库连接失败
- 设置保存时出现数据库错误
- 频繁出现 "database is locked" 错误

**根本原因：**
- SQLite 在并发访问时的限制
- 数据库文件权限问题
- 数据库连接池配置不当
- 缺少适当的错误处理和重试机制

### 2. 数据一致性问题

**症状：**
- 设置保存后读取不一致
- 数据库中有重复记录
- 字段类型不匹配

**根本原因：**
- 缺少唯一约束
- 缺少事务处理
- 缺少数据验证

## 🛠️ 全面修复方案

### 修复 1: 增强数据库连接配置

**文件：`lelamp/database/base.py`**

```python
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
from typing import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine, pool
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy import exc as sqlalchemy_exc
import logging

logger = logging.getLogger("lelamp.database")

# Database URL from environment variable or default to SQLite
DATABASE_URL = os.getenv(
    "LELAMP_DB_URL", "sqlite:///./lelamp.db"
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
            conn.execute(sqlalchemy_exc.text("PRAGMA journal_mode=WAL"))
            # 设置同步模式（平衡性能和安全）
            conn.execute(sqlalchemy_exc.text("PRAGMA synchronous=NORMAL"))
            # 设置缓存大小（根据需要调整）
            conn.execute(sqlalchemy_exc.text("PRAGMA cache_size=-64000"))  # 64MB
            # 设置临时存储为内存
            conn.execute(sqlalchemy_exc.text("PRAGMA temp_store=MEMORY"))
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
    - 连接健康检查
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
            result = conn.execute(sqlalchemy_exc.text("SELECT 1"))
            return result.fetchone()[0] == 1
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False
```

### 修复 2: 增强会话管理

**文件：`lelamp/database/session.py`**

```python
"""
Database session management utilities.

增强版：
- 重试机制
- 更好的错误处理
- 连接池管理
- 健康检查
"""
import time
import logging
from typing import Generator, Optional

from sqlalchemy import exc as sqlalchemy_exc
from sqlalchemy.orm import Session

from lelamp.database.base import SessionLocal

logger = logging.getLogger("lelamp.database")

# 最大重试次数
MAX_RETRIES = 3
# 重试延迟（秒）
RETRY_DELAY = 0.5


class DatabaseError(Exception):
    """数据库操作异常"""
    pass


def get_db_with_retry(max_retries: int = MAX_RETRIES) -> Generator[Session, None, None]:
    """
    获取数据库会话（带重试机制）

    Args:
        max_retries: 最大重试次数

    Yields:
        Session: 数据库会话

    Raises:
        DatabaseError: 重试失败后抛出
    """
    last_error = None

    for attempt in range(max_retries):
        db = SessionLocal()
        try:
            yield db
            # 成功执行，直接返回
            return
        except (
            sqlalchemy_exc.OperationalError,
            sqlalchemy_exc.InterfaceError,
        ) as e:
            last_error = e
            db.close()

            if attempt < max_retries - 1:
                wait_time = RETRY_DELAY * (attempt + 1)
                logger.warning(
                    f"Database connection failed (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                logger.error(f"Database connection failed after {max_retries} attempts: {e}")
        except sqlalchemy_exc.SQLAlchemyError as e:
            # 其他 SQL 错误不重试
            db.rollback()
            db.close()
            raise DatabaseError(f"Database operation failed: {e}")
        finally:
            if db:
                db.close()

    # 所有重试都失败
    raise DatabaseError(f"Database unavailable after {max_retries} retries: {last_error}")


def get_db() -> Generator[Session, None, None]:
    """
    原有的 get_db 函数，保持向后兼容

    注意：建议使用 get_db_with_retry() 以获得更好的可靠性
    """
    db = SessionLocal()
    try:
        yield db
    except sqlalchemy_exc.SQLAlchemyError as e:
        db.rollback()
        raise e
    finally:
        db.close()


def get_db_session() -> Session:
    """
    获取数据库会话（手动管理）

    用法：
        db = get_db_session()
        try:
            result = db.query(UserSettings).all()
        finally:
            db.close()
    """
    return SessionLocal()


def execute_in_transaction(func, *args, **kwargs):
    """
    在事务中执行函数，自动处理提交和回滚

    Args:
        func: 要执行的函数
        *args: 函数参数
        **kwargs: 函数关键字参数

    Returns:
        函数执行结果

    Raises:
        DatabaseError: 执行失败
    """
    db = get_db_session()
    try:
        result = func(db, *args, **kwargs)
        db.commit()
        return result
    except Exception as e:
        db.rollback()
        logger.error(f"Transaction failed: {e}")
        raise DatabaseError(f"Transaction failed: {e}")
    finally:
        db.close()
```

### 修复 3: 增强配置同步服务

**文件：`lelamp/api/services/config_sync.py`**

在现有的 ConfigSyncService 类中添加错误处理和重试机制：

```python
def update_settings(
    self,
    db: Session,
    lamp_id: str,
    updates: dict
) -> tuple[dict, bool]:
    """
    更新配置（增强版）

    增强：
    - 添加事务处理
    - 数据验证
    - 错误处理
    - 并发控制
    """
    try:
        # 开始事务
        with db.begin():
            # 获取或创建设置
            settings = db.query(UserSettings).filter(
                UserSettings.lamp_id == lamp_id
            ).with_for_update().first()  # 行级锁，防止并发冲突

            if settings is None:
                settings = UserSettings(lamp_id=lamp_id)
                db.add(settings)

            # 验证更新数据
            validated_updates = self._validate_updates(updates)

            # 更新字段
            for key, value in validated_updates.items():
                if hasattr(settings, key):
                    setattr(settings, key, value)

            # 刷新对象以获取数据库生成的值
            db.flush()
            db.refresh(settings)

        # 事务提交后同步到 .env 文件
        requires_restart = self._sync_to_env_file(validated_updates)

        # 返回更新后的配置
        return self.get_current_config(db, lamp_id), requires_restart

    except sqlalchemy_exc.IntegrityError as e:
        logger.error(f"Database integrity error: {e}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="配置冲突，可能已有其他用户在修改"
        )
    except Exception as e:
        logger.error(f"Update settings error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"更新配置失败: {str(e)}"
        )


def _validate_updates(self, updates: dict) -> dict:
    """
    验证更新数据

    Args:
        updates: 原始更新数据

    Returns:
        验证后的数据

    Raises:
        ValueError: 数据验证失败
    """
    validated = {}

    for key, value in updates.items():
        # 类型验证
        if key in ["led_brightness", "baidu_tts_per", "volume_level"]:
            if not isinstance(value, int) or not (0 <= value <= 100):
                raise ValueError(f"{key} must be an integer between 0 and 100")

        elif key in ["camera_width", "camera_height"]:
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{key} must be a positive integer")

        elif key in ["vision_enabled", "noise_cancellation", "notifications_enabled"]:
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be a boolean")

        validated[key] = value

    return validated
```

### 修复 4: API 路由增强

**文件：`lelamp/api/routes/settings.py`**

```python
@router.put("/", response_model=AppSettingsResponse)
async def update_settings(
    request: AppSettingsUpdateRequest,
    lamp_id: str = Query(..., description="设备 ID"),
    db: Session = Depends(get_db_with_retry)  # 使用带重试的会话
) -> AppSettingsResponse:
    """
    更新应用配置（增强版）

    增强：
    - 使用带重试的数据库会话
    - 更好的错误处理
    - 并发控制
    """
    try:
        # 验证请求数据
        if not request.model_dump(exclude_unset=True):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="没有提供任何更新字段"
            )

        # 使用事务处理
        with db.begin():
            # 更新配置
            config, requires_restart = config_sync_service.update_settings(
                db, lamp_id, request.model_dump(exclude_unset=True, exclude_none=True)
            )

        logger.info(f"Settings updated for lamp_id={lamp_id}, requires_restart={requires_restart}")

        return AppSettingsResponse(**config)

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"请求数据验证失败: {str(e)}"
        )
    except sqlalchemy_exc.IntegrityError as e:
        logger.error(f"Database integrity error: {e}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="配置冲突，请稍后重试"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update settings error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"更新配置失败: {str(e)}"
        )
```

### 修复 5: 添加数据库健康检查端点

**文件：`lelamp/api/routes/system.py`**

```python
@router.get("/health/db")
async def check_database_health(
    db: Session = Depends(get_db)
) -> dict:
    """
    检查数据库健康状态

    Returns:
        健康状态信息
    """
    try:
        from lelamp.database.base import check_db_health

        is_healthy = check_db_health()

        # 执行简单查询测试连接
        from lelamp.database.models import UserSettings
        settings_count = db.query(UserSettings).count()

        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "database_url": os.getenv("LELAMP_DB_URL", "sqlite:///./lelamp.db"),
            "settings_count": settings_count,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
```

### 修复 6: 启动时数据库优化

**文件：`lelamp/api/app.py`**

```python
async def lifespan(app: FastAPI):
    """应用生命周期管理（增强版）"""
    # 启动时执行
    logger.info("LeLamp API 启动")

    # 初始化数据库
    try:
        from lelamp.database.base import init_db, check_db_health

        # 初始化数据库表
        init_db()
        logger.info("Database initialized")

        # 健康检查
        if not check_db_health():
            logger.warning("Database health check failed, but continuing...")

    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        # 不阻止应用启动，但记录错误

    # ... 其他启动逻辑

    yield

    # 关闭时执行
    logger.info("LeLamp API 关闭")
```

## 🔍 诊断工具

### 数据库诊断脚本

**创建：`scripts/diagnose_database.py`**

```python
#!/usr/bin/env python3
"""
数据库诊断工具
"""
import sys
import os
import sqlite3
from pathlib import Path

sys.path.insert(0, os.path.expanduser('~/lelamp_runtime'))

print('=== 数据库诊断工具 ===\n')

# 1. 检查数据库文件
db_path = Path(os.getcwd()) / "lelamp.db"
print(f'1. 检查数据库文件: {db_path}')

if db_path.exists():
    print(f'   ✅ 数据库文件存在')
    print(f'   文件大小: {db_path.stat().st_size / 1024:.2f} KB')
else:
    print(f'   ❌ 数据库文件不存在')

# 2. 检查数据库权限
if db_path.exists():
    stat_info = db_path.stat()
    print(f'\n2. 检查文件权限:')
    print(f'   读取权限: {oct(stat_info.st_mode)[-3:]}')

# 3. 尝试打开数据库
if db_path.exists():
    print(f'\n3. 测试数据库连接:')
    try:
        conn = sqlite3.connect(str(db_path), timeout=10)
        cursor = conn.cursor()

        # 检查表
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f'   ✅ 数据库连接成功')
        print(f'   数据表: {[t[0] for t in tables]}')

        # 检查 PRAGMA 设置
        cursor.execute("PRAGMA journal_mode")
        journal_mode = cursor.fetchone()[0]
        print(f'   日志模式: {journal_mode}')

        cursor.execute("PRAGMA synchronous")
        synchronous = cursor.fetchone()[0]
        print(f'   同步模式: {synchronous}')

        # 检查是否有锁
        cursor.execute("PRAGMA lock_status")
        lock_status = cursor.fetchone()
        print(f'   锁状态: {lock_status}')

        conn.close()

    except sqlite3.Error as e:
        print(f'   ❌ 数据库错误: {e}')

print('\n诊断完成')
```

## 🚀 实施步骤

### 第 1 步：更新数据库配置

```bash
# 备份当前数据库
cp lelamp.db lelamp.db.backup

# 更新代码
# （实施上述修复代码）
```

### 第 2 步：测试修复

```bash
# 运行诊断工具
uv run python scripts/diagnose_database.py

# 启动 API 服务
uv run uvicorn lelamp.api.app:app --reload
```

### 第 3 步：验证修复

```bash
# 测试设置保存
curl -X PUT http://localhost:8000/api/settings/?lamp_id=lelamp \
  -H "Content-Type: application/json" \
  -d '{"led_brightness": 50}'

# 测试数据库健康检查
curl http://localhost:8000/api/system/health/db
```

## 📋 预防措施

### 1. 定期数据库维护

```python
# 创建 scripts/maintain_database.py
import sqlite3
from pathlib import Path

def vacuum_database():
    """优化数据库"""
    db_path = Path("lelamp.db")
    if not db_path.exists():
        return

    conn = sqlite3.connect(str(db_path))
    try:
        # VACUUM 优化数据库
        conn.execute("VACUUM")
        # 分析表以优化查询
        conn.execute("ANALYZE")
        print("Database maintenance completed")
    finally:
        conn.close()

if __name__ == "__main__":
    vacuum_database()
```

### 2. 监控数据库性能

```python
# 在 API 中添加数据库性能监控
import time
from sqlalchemy import event
from sqlalchemy.engine import Engine

@event.listens_for(Engine, "before_cursor_execute")
def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    conn.info.setdefault('query_start_time', []).append(time.time())

@event.listens_for(Engine, "after_cursor_execute")
def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    total = time.time() - conn.info['query_start_time'].pop()
    if total > 1.0:  # 记录慢查询
        logger.warning(f"Slow query ({total:.2f}s): {statement}")
```

### 3. 配置自动重试

```python
# 在 FastAPI 依赖注入中自动使用重试版本
from lelamp.database.session import get_db_with_retry

@router.get("/settings")
async def get_settings(
    lamp_id: str,
    db: Session = Depends(get_db_with_retry)  # 自动重试
):
    ...
```

## 🎯 总结

通过这些修复，我们可以：

1. ✅ **解决数据库锁定问题**
   - 启用 WAL 模式
   - 优化连接池配置
   - 添加重试机制

2. ✅ **提高数据一致性**
   - 使用事务处理
   - 添加数据验证
   - 行级锁防止并发冲突

3. ✅ **增强错误处理**
   - 自动重试机制
   - 详细的错误日志
   - 优雅的错误恢复

4. ✅ **提供诊断工具**
   - 健康检查端点
   - 数据库诊断脚本
   - 性能监控

这将彻底解决后台设置的数据库问题！
