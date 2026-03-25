#!/usr/bin/env python3
"""
数据库维护工具

定期运行以优化数据库性能
"""
import sys
import os
import sqlite3
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.expanduser('~/lelamp_runtime'))

print('=== 数据库维护工具 ===')
print(f'开始时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')

db_path = Path(os.getcwd()) / "lelamp.db"

if not db_path.exists():
    print('❌ 数据库文件不存在')
    sys.exit(1)

try:
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # 1. 获取当前统计信息
    print('1. 当前统计信息:')

    tables = ['conversations', 'operation_logs', 'device_states', 'user_settings']
    total_records = 0

    for table_name in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        total_records += count
        print(f'   {table_name}: {count} 条记录')

    print(f'   总计: {total_records} 条记录')

    # 2. 数据库优化
    print('\n2. 执行数据库优化:')

    # VACUUM - 重建数据库文件，减少碎片
    print('   执行 VACUUM（重建数据库）...')
    start_time = datetime.now()
    cursor.execute("VACUUM")
    vacuum_time = (datetime.now() - start_time).total_seconds()
    print(f'   ✅ VACUUM 完成 (耗时: {vacuum_time:.2f}秒)')

    # ANALYZE - 更新查询优化器的统计信息
    print('   执行 ANALYZE（更新统计信息）...')
    cursor.execute("ANALYZE")
    print('   ✅ ANALYZE 完成')

    # 3. 清理过期数据（可选）
    print('\n3. 清理过期数据:')

    # 清理 30 天前的对话记录
    cutoff_date = (datetime.now() - datetime.timedelta(days=30)).isoformat()
    cursor.execute(
        "DELETE FROM conversations WHERE timestamp < ?",
        (cutoff_date,)
    )
    deleted_conversations = cursor.rowcount
    print(f'   删除 {deleted_conversations} 条过期对话记录')

    # 清理 30 天前的操作日志
    cursor.execute(
        "DELETE FROM operation_logs WHERE timestamp < ?",
        (cutoff_date,)
    )
    deleted_logs = cursor.rowcount
    print(f'   删除 {deleted_logs} 条过期操作日志')

    # 提交更改
    conn.commit()

    # 4. 重建索引
    print('\n4. 重建索引:')
    cursor.execute("REINDEX")
    print('   ✅ 索引重建完成')

    # 5. 获取优化后的统计信息
    print('\n5. 优化后统计信息:')
    total_records_after = 0
    for table_name in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        total_records_after += count

    print(f'   总计: {total_records_after} 条记录')
    print(f'   清理了: {total_records - total_records_after} 条记录')

    # 6. 检查数据库文件大小
    file_size_before = db_path.stat().st_size / 1024
    print(f'\n6. 数据库文件大小: {file_size_before:.2f} KB')

    conn.close()

    print(f'\n✅ 数据库维护完成')
    print(f'完成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

except Exception as e:
    print(f'\n❌ 维护失败: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

print('\n提示: 建议定期运行此脚本（如每周一次）以保持数据库性能')
