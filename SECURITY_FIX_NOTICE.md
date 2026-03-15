# 🔒 安全修复通知 - 重要

## API 密钥泄露处理

### 已采取的行动

1. **创建 .env.example 模板**
   - 将所有敏感配置迁移到 `.env.example` 模板文件
   - 所有真实值已替换为占位符说明

2. **更新 .gitignore**
   - 确保 `.env` 文件不会被提交到版本控制
   - 添加 `.env.local` 和 `.env.*.local` 到忽略列表

### 需要立即采取的行动

#### 1. 撤销泄露的飞书 API 密钥

⚠️ **以下密钥已泄露，需要立即撤销：**

```
FEISHU_APP_ID=cli_a9a2877d71789bc0
FEISHU_APP_SECRET=EG7wSiPIkalKsBl7Eh1OoaiUehcQmJkR
FEISHU_RECEIVE_ID=ou_c5fcc3c5532354f1c548a3a018f4f7d0
```

**撤销步骤：**
1. 登录飞书开放平台：https://open.feishu.cn/app
2. 找到应用 `cli_a9a2877d71789bc0`
3. 重新生成 `App Secret`
4. 更新生产环境的 `.env` 文件

#### 2. 从 Git 历史中删除敏感信息

⚠️ **注意：以下操作会重写 Git 历史，执行前请备份！**

```bash
# 方法 1：使用 git filter-repo（推荐）
pip install git-filter-repo
git filter-repo --invert-paths --path .env

# 方法 2：使用 git filter-branch（备选）
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch .env" \
  --prune-empty --tag-name-filter cat -- --all

# 强制推送到远程（⚠️ 谨慎操作）
git push origin --force --all
git push origin --force --tags
```

#### 3. 清理本地敏感文件

```bash
# 删除包含敏感信息的 .env 文件
rm .env

# 从 .env.example 复制新配置
cp .env.example .env

# 编辑 .env，填入新的密钥
nano .env
```

### 验证修复

```bash
# 确保 .env 不在 Git 跟踪中
git status

# 应该看到：
# .env
# 但不应该看到 .env.example（如果是新文件）

# 检查 .gitignore
cat .gitignore | grep .env
```

### 防止未来泄露

1. **安装 git-secrets**
   ```bash
   brew install git-secrets  # macOS
   # 或
   apt-get install git-secrets  # Ubuntu
   ```

2. **配置 git-secrets**
   ```bash
   cd /Users/jackwang/lelamp_runtime
   git-secrets --install
   git-secrets --register-azure
   git-secrets --add 'FEISHU_APP_ID.*=.*'
   git-secrets --add 'FEISHU_APP_SECRET.*=.*'
   git-secrets --add 'API_KEY.*=.*'
   git-secrets --add 'SECRET.*=.*'
   ```

3. **添加 pre-commit hook**
   ```bash
   # 创建 .git/hooks/pre-commit
   cat > .git/hooks/pre-commit << 'EOF'
   #!/bin/sh
   git-secrets --scan
   EOF

   chmod +x .git/hooks/pre-commit
   ```

### 联系方式

如果发现其他安全问题，请联系项目维护者。

---

**生成时间:** 2025-03-15
**修复版本:** 将在修复所有 P0 问题后创建新提交
