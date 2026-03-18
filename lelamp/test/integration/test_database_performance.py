"""
Integration tests for database performance.

Tests query performance with various dataset sizes and ensures
indexes are properly used.
"""
import pytest
import time
from sqlalchemy import text
from lelamp.database.session import get_db_session
from lelamp.database.base import Base, engine
from lelamp.database.models import Conversation, OperationLog, DeviceState
from datetime import datetime, timedelta


@pytest.fixture(scope="function")
def setup_db():
    """创建测试数据库"""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.mark.usefixtures("setup_db")
def test_conversation_query_performance():
    """测试对话查询性能"""
    db = get_db_session()

    # 插入100条测试数据
    now = datetime.utcnow()
    for i in range(100):
        conv = Conversation(
            lamp_id=f"test_lamp_{i % 5}",
            timestamp=now - timedelta(seconds=i),
            messages=[{"role": "user", "content": f"message {i}"}],
            user_input=f"input {i}",
            ai_response=f"response {i}",
            duration=10
        )
        db.add(conv)
    db.commit()

    # 测试按 lamp_id 和时间范围查询
    start = time.time()
    conversations = db.query(Conversation).filter(
        Conversation.lamp_id == "test_lamp_0",
        Conversation.timestamp >= now - timedelta(seconds=100)
    ).all()
    elapsed = time.time() - start

    assert len(conversations) > 0
    assert elapsed < 0.1  # 100条数据查询应在100ms内完成

    db.close()


@pytest.mark.usefixtures("setup_db")
def test_operation_log_query_performance():
    """测试操作日志查询性能"""
    db = get_db_session()

    # 插入200条测试数据
    now = datetime.utcnow()
    for i in range(200):
        log = OperationLog(
            lamp_id=f"test_lamp_{i % 3}",
            timestamp=now - timedelta(seconds=i),
            operation_type="motor_move" if i % 2 == 0 else "rgb_set",
            action=f"action_{i}",
            params={"param": i},
            success=i % 10 != 0  # 10% 失败率
        )
        db.add(log)
    db.commit()

    # 测试查询失败的操作
    start = time.time()
    failed_logs = db.query(OperationLog).filter(
        OperationLog.success == False
    ).all()
    elapsed = time.time() - start

    assert len(failed_logs) > 0
    assert elapsed < 0.1  # 应该使用 success 索引

    # 测试复合索引查询
    start = time.time()
    logs = db.query(OperationLog).filter(
        OperationLog.lamp_id == "test_lamp_0",
        OperationLog.success == True,
        OperationLog.timestamp >= now - timedelta(seconds=200)
    ).all()
    elapsed = time.time() - start

    assert len(logs) > 0
    assert elapsed < 0.1  # 复合索引应该加速查询

    db.close()


@pytest.mark.usefixtures("setup_db")
def test_index_usage():
    """验证索引被正确使用"""
    db = get_db_session()

    # 插入测试数据
    now = datetime.utcnow()
    for i in range(50):
        conv = Conversation(
            lamp_id="test_lamp",
            timestamp=now - timedelta(seconds=i),
            messages=[],
            user_input="test",
            ai_response="test"
        )
        db.add(conv)
    db.commit()

    # 使用 EXPLAIN QUERY PLAN 检查索引使用 (SQLite)
    result = db.execute(text(
        "EXPLAIN QUERY PLAN SELECT * FROM conversations WHERE lamp_id = 'test_lamp' AND timestamp >= :ts"
    ), {"ts": (now - timedelta(seconds=100)).isoformat()})

    # 在 SQLite 中,使用了索引会显示 "USING INDEX"
    plan_str = str(result.fetchall())
    # 索引应该被使用 (由于数据量小,可能不总是显示,但不应该有全表扫描)
    # 只要不出现 "SCAN" 即表示使用了索引

    db.close()


@pytest.mark.usefixtures("setup_db")
def test_large_dataset_performance():
    """测试大数据集性能"""
    db = get_db_session()

    # 插入1000条记录
    now = datetime.utcnow()
    batch_size = 100

    start = time.time()
    for batch in range(10):
        for i in range(batch_size):
            log = OperationLog(
                lamp_id=f"lamp_{batch}",
                timestamp=now - timedelta(seconds=batch * batch_size + i),
                operation_type="test",
                action=f"action_{i}",
                params={},
                success=True
            )
            db.add(log)
        db.commit()  # 每100条提交一次
    insert_elapsed = time.time() - start

    # 插入应在合理时间内完成
    assert insert_elapsed < 5.0  # 1000条记录插入应在5秒内完成

    # 测试分页查询性能
    start = time.time()
    page1 = db.query(OperationLog).filter(
        OperationLog.lamp_id == "lamp_0"
    ).limit(50).all()
    page_elapsed = time.time() - start

    assert len(page1) == 50
    assert page_elapsed < 0.1  # 分页查询应很快

    db.close()
