#!/usr/bin/env python3
"""
测试设置 API 端点
"""
import requests
import json

# API 基础 URL
API_BASE = "http://localhost:8000/api"

def test_get_settings():
    """测试获取设置"""
    lamp_id = "lelamp"
    url = f"{API_BASE}/settings/?lamp_id={lamp_id}"

    print(f"GET {url}")
    response = requests.get(url)

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("Settings retrieved successfully:")
        print(f"  - deepseek_model: {data.get('deepseek_model')}")
        print(f"  - deepseek_base_url: {data.get('deepseek_base_url')}")
        print(f"  - requires_restart: {data.get('requires_restart')}")
        return True
    else:
        print(f"Error: {response.text}")
        return False

def test_update_settings():
    """测试更新设置"""
    lamp_id = "lelamp"
    url = f"{API_BASE}/settings/?lamp_id={lamp_id}"

    # 测试数据
    updates = {
        "deepseek_model": "deepseek-chat",
        "deepseek_base_url": "https://api.deepseek.com"
    }

    print(f"\nPUT {url}")
    print(f"Payload: {json.dumps(updates, indent=2)}")

    response = requests.put(
        url,
        json=updates,
        headers={"Content-Type": "application/json"}
    )

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("Settings updated successfully:")
        print(f"  - deepseek_model: {data.get('deepseek_model')}")
        print(f"  - deepseek_base_url: {data.get('deepseek_base_url')}")
        print(f"  - requires_restart: {data.get('requires_restart')}")
        return True
    else:
        print(f"Error: {response.text}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("测试设置 API")
    print("=" * 60)

    # 测试获取设置
    if not test_get_settings():
        print("\n❌ 获取设置失败")
        exit(1)

    # 测试更新设置
    if not test_update_settings():
        print("\n❌ 更新设置失败")
        exit(1)

    print("\n✅ 所有测试通过")
