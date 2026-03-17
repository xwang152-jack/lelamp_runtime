#!/bin/bash

# LeLamp .env 配置修复脚本
# 修复 .env 文件中的配置问题

echo "================================"
echo "LeLamp 配置修复"
echo "================================"
echo ""

cd /Users/jackwang/lelamp_runtime

echo "1. 检查 .env 文件..."
if [ ! -f ".env" ]; then
    echo "错误: .env 文件不存在"
    exit 1
fi
echo "✓ .env 文件存在"

echo ""
echo "2. 备份原文件..."
cp .env .env.backup
echo "✓ 备份已保存到 .env.backup"

echo ""
echo "3. 修复配置问题..."

# 修复 LIVEKIT_API_SECRET（移除追加的 DEV_MODE）
sed -i.bak 's/^LIVEKIT_API_SECRET=.*LELAMP_DEV_MODE=1$/LIVEKIT_API_SECRET=J9Pgnz3OdweCXJUiVZadjxUKWRljdIAWBWTFsMbBTY/' .env

# 确保 LELAMP_DEV_MODE 在独立行
if ! grep -q "^LELAMP_DEV_MODE=1$" .env; then
    # 移除错误追加的 DEV_MODE
    sed -i.bak '/^LIVEKIT_API_SECRET=.*LELAMP_DEV_MODE=1$/d' .env
    # 添加独立的 DEV_MODE 行
    echo "LELAMP_DEV_MODE=1" >> .env
fi

echo "✓ 配置已修复"

echo ""
echo "4. 验证配置..."
echo ""
echo "当前配置:"
grep "^LELAMP_DEV_MODE\|^LIVEKIT_API_SECRET" .env | sed 's/=.*/= ***/'

echo ""
echo "================================"
echo "✓ 配置修复完成！"
echo "================================"
echo ""
echo "现在可以启动后端服务了："
echo ""
echo "  sudo uv run main.py console"
echo ""
echo "如果还有问题，查看备份："
echo "  cat .env.backup"
echo ""
