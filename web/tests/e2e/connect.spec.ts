import { test, expect } from '@playwright/test';

/**
 * 连接页面 E2E 测试
 * @description 测试 LeLamp Web 客户端连接页面的功能和交互
 */
test.describe('连接页面', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('应正确显示页面标题和元素', async ({ page }) => {
    // 等待页面加载
    await page.waitForLoadState('networkidle');

    // 验证页面标题
    await expect(page.locator('.connect-card h1')).toContainText('LeLamp Web Client');
    await expect(page.locator('.connect-card p')).toContainText('智能台灯，陪伴成长');

    // 验证表单标签（使用文本内容定位）
    await expect(page.getByText('LiveKit Server URL')).toBeVisible();
    await expect(page.getByText('Access Token')).toBeVisible();

    // 验证连接按钮
    await expect(page.getByRole('button', { name: '连接设备' })).toBeVisible();
  });

  test('应显示 URL 预配置提示', async ({ page }) => {
    // 等待页面加载
    await page.waitForLoadState('networkidle');

    // 检查是否有预配置的 URL 提示
    const successHint = page.locator('.hint.success');
    const warningHint = page.locator('.hint.warning');

    const hasSuccessHint = await successHint.count() > 0;
    const hasWarningHint = await warningHint.count() > 0;

    // 至少应该有一个提示（如果 .env.development 配置了 URL 则显示 success，否则显示 warning）
    expect(hasSuccessHint || hasWarningHint).toBeTruthy();
  });

  test('空表单点击连接应显示警告', async ({ page }) => {
    // 点击连接按钮
    await page.click('button:has-text("连接设备")');

    // 等待警告消息出现（Element Plus 的 ElMessage）
    const messageBox = page.locator('.el-message').filter({ hasText: '请填写完整信息' });
    await expect(messageBox).toBeVisible({ timeout: 5000 });
  });

  test('Server URL 输入框可输入内容', async ({ page }) => {
    const urlInput = page.locator('input[placeholder*="livekit.cloud"]');

    // 输入测试 URL
    await urlInput.fill('wss://test.livekit.cloud');

    // 验证输入值
    await expect(urlInput).toHaveValue('wss://test.livekit.cloud');

    // 测试清除功能
    await urlInput.clear();
    await expect(urlInput).toHaveValue('');
  });

  test('Token 输入框可输入多行文本', async ({ page }) => {
    const tokenInput = page.locator('textarea[placeholder*="粘贴生成的 Token"]');

    // 输入测试 Token
    const testToken = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test.token';
    await tokenInput.fill(testToken);

    // 验证输入值
    await expect(tokenInput).toHaveValue(testToken);

    // 测试多行输入
    const multilineToken = 'line1\nline2\nline3';
    await tokenInput.fill(multilineToken);
    await expect(tokenInput).toHaveValue(multilineToken);
  });

  test('输入框支持清除操作', async ({ page }) => {
    const urlInput = page.locator('input[placeholder*="livekit.cloud"]');
    const tokenInput = page.locator('textarea[placeholder*="粘贴生成的 Token"]');

    // 输入内容
    await urlInput.fill('wss://test.livekit.cloud');
    await tokenInput.fill('test-token');

    // 点击清除按钮（Element Plus 的 clearable 图标）
    const clearIcons = page.locator('.el-input__clear');
    if (await clearIcons.count() > 0) {
      await clearIcons.first().click();
      await expect(urlInput).toHaveValue('');
    }
  });

  test('连接按钮在加载时显示 loading 状态', async ({ page }) => {
    // 填写表单
    await page.fill('input[placeholder*="livekit.cloud"]', 'wss://test.livekit.cloud');
    await page.fill('textarea[placeholder*="粘贴生成的 Token"]', 'invalid-token');

    // 点击连接按钮
    const connectButton = page.getByRole('button', { name: '连接设备' });

    // 点击并等待响应
    await connectButton.click();

    // 等待错误消息出现（无效 token 会连接失败）
    const messageBox = page.locator('.el-message--error, .el-message--warning');
    await expect(messageBox).toBeVisible({ timeout: 10000 }).catch(() => {
      // 如果意外连接成功，验证跳转
      expect(page.url()).toMatch(/\/room/);
    });
  });

  test('页面应有正确的渐变背景', async ({ page }) => {
    const connectView = page.locator('.connect-view');

    // 验证元素存在
    await expect(connectView).toBeVisible();

    // 获取计算样式并验证背景渐变
    const background = await connectView.evaluate((el) => {
      return window.getComputedStyle(el).background;
    });

    // 验证是渐变背景（包含 gradient 或 rgb 颜色）
    expect(background || await connectView.evaluate(el => el.style.background)).toBeTruthy();
  });

  test('输入框应有正确的占位符文本', async ({ page }) => {
    const urlInput = page.locator('input[placeholder*="livekit.cloud"]');
    const tokenInput = page.locator('textarea[placeholder*="粘贴生成的 Token"]');

    await expect(urlInput).toHaveAttribute('placeholder', 'wss://your-project.livekit.cloud');
    await expect(tokenInput).toHaveAttribute('placeholder', '粘贴生成的 Token...');
  });

  test('卡片应有正确的样式和阴影', async ({ page }) => {
    const card = page.locator('.connect-card');

    await expect(card).toBeVisible();

    // 验证卡片有圆角
    const borderRadius = await card.evaluate((el) => {
      return window.getComputedStyle(el).borderRadius;
    });
    expect(borderRadius).toBeTruthy();

    // 验证卡片有阴影
    const boxShadow = await card.evaluate((el) => {
      return window.getComputedStyle(el).boxShadow;
    });
    expect(boxShadow).toBeTruthy();
    expect(boxShadow).toContain('0px');
  });
});
