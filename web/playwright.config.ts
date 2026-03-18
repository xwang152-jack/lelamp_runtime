import { defineConfig, devices } from '@playwright/test';

/**
 * LeLamp Web E2E 测试配置
 * @see https://playwright.dev/docs/test-configuration
 */
export default defineConfig({
  testDir: './tests/e2e',
  /* 并行运行测试文件 */
  fullyParallel: true,
  /* 在 CI 环境中失败时不重试 */
  forbidOnly: !!process.env.CI,
  /* 在 CI 环境中重试失败的测试 */
  retries: process.env.CI ? 2 : 0,
  /* 在 CI 中使用并行工作线程 */
  workers: process.env.CI ? 1 : undefined,
  /* 测试报告配置 */
  reporter: [
    ['html', { outputFolder: 'playwright-report' }],
    ['list'],
  ],
  /* 全局配置 */
  use: {
    /* 基础 URL - 用于测试中的导航 */
    baseURL: 'http://localhost:5173',
    /* 收集失败测试的追踪信息 */
    trace: 'on-first-retry',
    /* 截图配置 */
    screenshot: 'only-on-failure',
    /* 视频录制 */
    video: 'retain-on-failure',
    /* 操作超时时间 */
    actionTimeout: 10 * 1000,
    /* 导航超时时间 */
    navigationTimeout: 30 * 1000,
  },

  /* 测试项目配置 */
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
    // 可以添加更多浏览器
    // {
    //   name: 'firefox',
    //   use: { ...devices['Desktop Firefox'] },
    // },
    // {
    //   name: 'webkit',
    //   use: { ...devices['Desktop Safari'] },
    // },
  ],

  /* 开发服务器配置 - 测试前自动启动 */
  webServer: {
    command: 'npm run dev',
    url: 'http://localhost:5173',
    reuseExistingServer: !process.env.CI,
    timeout: 120 * 1000,
  },
});
