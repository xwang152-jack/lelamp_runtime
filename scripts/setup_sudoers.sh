#!/bin/bash
#
# 配置sudoers允许pi用户免密运行uvicorn
#

echo "配置sudoers规则..."

# 创建sudoers规则文件
sudo tee /etc/sudoers.d/99-lelamp-api > /dev/null <<EOF
# 允许pi用户免密运行API服务器
pi ALL=(ALL) NOPASSWD: /home/pi/lelamp_runtime/.venv/bin/python -m uvicorn *
EOF

# 设置正确的权限
sudo chmod 440 /etc/sudoers.d/99-lelamp-api

echo "sudoers配置完成！现在pi用户可以免密运行API服务器"
