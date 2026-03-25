#!/usr/bin/env python3
"""
数据库诊断工具

检查数据库文件、权限、连接和配置
"""
import sys
import os
import sqlite3
import time
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.expanduser('~/lelamp_runtime'))

print('=== 数据库诊断工具 ===\n')

# 1. 检查数据库文件
db_path = Path(os.getcwd()) / "lelamp.db"
print(f'1. 检查数据库文件: {db_path}')

if db_path.exists():
    print(f'   ✅ 数据库文件存在')
    size_kb = db_path.stat().st_size / 1024
    print(f'   文件大小: {size_kb:.2f} KB')

    # 检查文件权限
    stat_info = db_path.stat()
    mode = oct(stat_info.st_mode)[-3:]
    print(f'   文件权限: {mode}')

    # 检查是否可写
    if os.access(db_path, os.R_OK | os.W_OK):
        print(f'   ✅ 可读写')
    else:
        print(f'   ❌ 读写权限不足')
else:
    print(f'   ❌ 数据库文件不存在')
    print(f'   将在首次运行时自动创建')

# 2. 测试数据库连接
if db_path.exists():
    print(f'\n2. 测试数据库连接:')
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
        if journal_mode.upper() == 'WAL':
            print(f'   ✅ WAL 模式已启用（推荐）')
        else:
            print(f'   ⚠️  建议启用 WAL 模式以提升并发性能')

        cursor.execute("PRAGMA synchronous")
        synchronous = cursor.fetchone()[0]
        print(f'   同步模式: {synchronous}')

        cursor.execute("PRAGMA cache_size")
        cache_size = cursor.fetchone()[0]
        print(f'   缓存大小: {cache_size}')

        # 检查锁状态
        try:
            cursor.execute("PRAGMA lock_status")
            lock_status = cursor.fetchone()
            print(f'   锁状态: {lock_status}')
        except:
            # 某些 SQLite 版本不支持 lock_status
            print(f'   锁状态: 不支持查询')

        # 检查数据库完整性
        print(f'\n3. 检查数据库完整性:')
        cursor.execute("PRAGMA integrity_check")
        integrity = cursor.fetchall()
        if integrity[0][0] == 'ok':
            print(f'   ✅ 数据库完整性检查通过')
        else:
            print(f'   ❌ 数据库完整性问题: {integrity}')

        # 检查表结构
        print(f'\n4. 检查表结构:')
        for table_name in ['conversations', 'operation_logs', 'device_states', 'user_settings']:
            cursor.execute(f"SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='{table_name}'")
            exists = cursor.fetchone()[0]
            if exists:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                print(f'   ✅ {table_name}: {count} 条记录')
            else:
                print(f'   ⚠️  {table_name}: 表不存在')

        conn.close()

    except sqlite3.Error as e:
        print(f'   ❌ 数据库错误: {e}')
        print(f'   错误类型: {type(e).__name__}')
    except Exception as e:
        print(f'   ❌ 未知错误: {e}')

# 5. 测试并发性能
if db_path.exists():
    print(f'\n5. 测试并发性能:')
    try:
        # 模拟多个并发连接
        start_time = time.time()
        connections = []

        for i in range(5):
            conn = sqlite3.connect(str(db_path), timeout=5)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM user_settings")
            count = cursor.fetchone()[0]
            connections.append(conn)

        elapsed = time.time() - start_time
        print(f'   5个并发连接耗时: {elapsed:.3f}秒')

        for conn in connections:
            conn.close()

        if elapsed < 1.0:
            print(f'   ✅ 并发性能良好')
        else:
            print(f'   ⚠️  并发性能较慢，建议启用 WAL 模式')

    except Exception as e:
        print(f'   ❌ 并发测试失败: {e}')

# 6. 环境变量检查
print(f'\n6. 检查环境变量:')
db_url = os.getenv('LELAMP_DATABASE_URL')
if db_url:
    print(f'   LELAMP_DATABASE_URL: {db_url}')
else:
    print(f'   使用默认: sqlite:///./lelamp.db')

# 7. 建议和总结
print(f'\n=== 诊断建议 ===\n')

recommendations = []

if not db_path.exists():
    recommendations.append('数据库文件不存在，首次运行将自动创建')
else:
    # 检查 WAL 模式
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode")
        journal_mode = cursor.fetchone()[0]
        conn.close()

        if journal_mode.upper() != 'WAL':
            recommendations.append('建议启用 WAL 模式以提升并发性能')
            recommendations.append('可以在代码中调用 enable_wal_mode() 函数')
    except:
        pass

    # 检查文件大小
    if size_kb > 10240:  # 10MB
        recommendations.append('数据库文件较大，建议定期清理或优化')

if recommendations:
    print('建议:')
    for i, rec in enumerate(recommendations, 1):
        print(f'  {i}. {rec}')
else:
    print('✅ 数据库状态良好，无需优化')

print('\n诊断完成')
