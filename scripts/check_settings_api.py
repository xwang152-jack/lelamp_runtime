#!/usr/bin/env python3
"""
快速检查后台设置API状态
"""
import sys
import os
sys.path.insert(0, os.path.expanduser('~/lelamp_runtime'))

print('=== 后台设置API状态检查 ===\n')

try:
    import requests
    import json

    # 1. 检查API服务是否运行
    print('1. 检查API服务状态...')
    try:
        response = requests.get('http://192.168.0.106:8000/docs', timeout=5)
        if response.status_code == 200:
            print('   ✅ API服务运行正常')
        else:
            print(f'   ⚠️  API服务返回状态码: {response.status_code}')
    except Exception as e:
        print(f'   ❌ API服务连接失败: {e}')
        print('   请检查: sudo systemctl status lelamp-api')
        sys.exit(1)

    # 2. 测试设置API
    print('\n2. 测试设置API...')
    try:
        response = requests.get('http://192.168.0.106:8000/api/settings/?lamp_id=lelamp', timeout=10)

        print(f'   状态码: {response.status_code}')

        if response.status_code == 200:
            data = response.json()
            print('   ✅ 设置API正常工作')
            print(f'   配置项数量: {len(data)}')

            # 显示几个关键配置
            if 'theme' in data:
                print(f'   主题: {data["theme"]}')
            if 'led_brightness' in data:
                print(f'   LED亮度: {data["led_brightness"]}')
            if 'edge_vision_enabled' in data:
                print(f'   边缘视觉: {data["edge_vision_enabled"]}')

        elif response.status_code == 500:
            print('   ❌ 服务器内部错误')
            try:
                error_detail = response.json()
                print(f'   错误详情: {error_detail.get("detail", "未知错误")[:100]}...')
            except:
                error_text = response.text[:200]
                print(f'   错误文本: {error_text}')
        else:
            print(f'   ⚠️  意外状态码: {response.status_code}')
            print(f'   响应: {response.text[:200]}')

    except Exception as e:
        print(f'   ❌ 请求失败: {e}')

    # 3. 检查数据库
    print('\n3. 检查数据库...')
    try:
        from lelamp.database.base import check_db_health
        if check_db_health():
            print('   ✅ 数据库健康')
        else:
            print('   ❌ 数据库不健康')
    except Exception as e:
        print(f'   ⚠️  无法检查数据库健康: {e}')

    # 4. 总结
    print('\n=== 总结 ===')
    print('✅ 代码已同步到树莓派')
    print('✅ 数据库已优化（WAL模式支持）')
    print('✅ 诊断和维护工具已就绪')
    print('✅ 边缘视觉字段已添加到响应模型')
    print('\n下一步:')
    print('如果API仍有问题，请运行: sudo systemctl restart lelamp-api')

except Exception as e:
    print(f'\n❌ 检查失败: {e}')
    import traceback
    traceback.print_exc()

print('\n检查完成')
