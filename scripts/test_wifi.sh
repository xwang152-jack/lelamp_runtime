#!/bin/bash
# 测试WiFi状态和扫描功能

echo "=== 测试WiFi连接状态 ==="
echo "1. nmcli连接状态:"
ssh pi@192.168.0.104 "nmcli -t -f ACTIVE,NAME,DEVICE connection show"

echo -e "\n2. wlan0设备状态:"
ssh pi@192.168.0.104 "nmcli -t device show wlan0"

echo -e "\n3. wlan0 IP地址:"
ssh pi@192.168.0.104 "nmcli -t -f IP4.ADDRESS device show wlan0"

echo -e "\n4. API WiFi状态端点:"
ssh pi@192.168.0.104 "curl -s 'http://localhost:8000/api/system/wifi/status'"

echo -e "\n=== 测试WiFi扫描 ==="
echo "5. nmcli扫描结果:"
ssh pi@192.168.0.104 "sudo nmcli -t -f SSID,BSSID,SECURITY,FREQ device wifi list | head -3"

echo -e "\n6. API WiFi扫描端点:"
ssh pi@192.168.0.104 "curl -s 'http://localhost:8000/api/system/wifi/scan' | python -m json.tool | head -20"
