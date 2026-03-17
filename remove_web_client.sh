#!/bin/bash

# LeLamp 删除 web_client 目录并更新相关文档
# web_client 是旧版本，项目现在使用 web/ 目录

echo "================================"
echo "LeLamp 删除 web_client 目录"
echo "================================"
echo ""

cd /Users/jackwang/lelamp_runtime

# 1. 备份 web_client
echo "📦 备份 web_client 目录..."
BACKUP_DIR=".archive/web_client_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp -r web_client "$BACKUP_DIR/"
echo "  ✓ 备份到: $BACKUP_DIR"
echo ""

# 2. 删除 web_client 目录
echo "🗑️  删除 web_client 目录..."
rm -rf web_client
echo "  ✓ 已删除"
echo ""

# 3. 更新 CLAUDE.md
echo "📝 更新 CLAUDE.md..."
sed -i.bak 's/- `web_client\/`: Web-based user client for remote control and monitoring/- `web\/`: Vue 3 + TypeScript + Vite 前端应用/g' CLAUDE.md
sed -i.bak '/\*\*Web Client\*\* (\`web_client\/\`):/,/^$/d' CLAUDE.md
echo "  ✓ 已更新 CLAUDE.md"
echo ""

# 4. 更新 README.md
echo "📝 更新 README.md..."
sed -i.bak 's|http://localhost:8000/web_client/|http://localhost:5173|g' README.md
sed -i.bak 's|- 🌐 \[Web Client 文档\](./web_client/README.md) - Web 界面使用说明|- 🌐 [Web 前端](./web/) - Vue 3 前端应用|g' README.md
sed -i.bak 's/├── web_client\/                 # Web 客户端/├── web\/                       # Vue 3 前端应用/g' README.md
echo "  ✓ 已更新 README.md"
echo ""

# 5. 更新 docs/COMMERCIAL_APP_ARCHITECTURE.md
echo "📝 更新 docs/COMMERCIAL_APP_ARCHITECTURE.md..."
sed -i.bak 's/> 关联代码: \`web_client\/\`,/> 关联代码: \`web\/\`,/g' docs/COMMERCIAL_APP_ARCHITECTURE.md
echo "  ✓ 已更新 docs/COMMERCIAL_APP_ARCHITECTURE.md"
echo ""

# 6. 更新 docs/PROJECT_OPTIMIZATION_STATUS.md
echo "📝 更新 docs/PROJECT_OPTIMIZATION_STATUS.md..."
sed -i.bak 's/- Vue 3 + Vite + Pinia 重写 web_client/- Vue 3 + Vite + Pinia 前端应用已完成/g' docs/PROJECT_OPTIMIZATION_STATUS.md
echo "  ✓ 已更新 docs/PROJECT_OPTIMIZATION_STATUS.md"
echo ""

# 7. 更新 docs/TESTING_CHECKLIST.md
echo "📝 更新 docs/TESTING_CHECKLIST.md..."
sed -i.bak 's|URL: http://localhost:8000/web_client/|URL: http://localhost:5173|g' docs/TESTING_CHECKLIST.md
echo "  ✓ 已更新 docs/TESTING_CHECKLIST.md"
echo ""

# 8. 更新 docs/USER_GUIDE.md
echo "📝 更新 docs/USER_GUIDE.md..."
sed -i.bak 's|http://localhost:8000/web_client/|http://localhost:5173|g' docs/USER_GUIDE.md
echo "  ✓ 已更新 docs/USER_GUIDE.md"
echo ""

# 9. 清理备份文件
echo "🧹 清理 .bak 备份文件..."
find . -maxdepth 1 -name "*.bak" -delete
find docs -maxdepth 1 -name "*.bak" -delete 2>/dev/null
echo "  ✓ 已清理"
echo ""

echo "================================"
echo "✅ web_client 删除完成！"
echo "================================"
echo ""
echo "📋 变更摘要:"
echo "  • 删除: web_client/ 目录"
echo "  • 备份: $BACKUP_DIR"
echo "  • 更新: CLAUDE.md"
echo "  • 更新: README.md"
echo "  • 更新: docs/COMMERCIAL_APP_ARCHITECTURE.md"
echo "  • 更新: docs/PROJECT_OPTIMIZATION_STATUS.md"
echo "  • 更新: docs/TESTING_CHECKLIST.md"
echo "  • 更新: docs/USER_GUIDE.md"
echo ""
echo "🚀 新的前端目录: web/"
echo "🌐 新的访问地址: http://localhost:5173"
echo ""
echo "💡 启动前端:"
echo "  cd web && pnpm dev"
echo ""
