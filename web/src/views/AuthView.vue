<template>
  <div class="auth-view">
    <!-- Animated Background -->
    <div class="auth-background">
      <div class="bg-gradient"></div>
      <div class="floating-shapes">
        <div class="shape shape-1"></div>
        <div class="shape shape-2"></div>
        <div class="shape shape-3"></div>
      </div>
    </div>

    <!-- Main Container -->
    <div class="auth-container">
      <!-- Logo & Title -->
      <div class="auth-header">
        <div class="logo">
          <span class="logo-icon">🪔</span>
        </div>
        <h1 class="auth-title">LeLamp</h1>
        <p class="auth-subtitle">智能台灯控制系统</p>
      </div>

      <!-- Auth Card -->
      <div class="auth-card">
        <div class="card-decoration"></div>

        <!-- Mode Tabs -->
        <div class="auth-tabs">
          <button
            :class="['auth-tab', { 'active': mode === 'login' }]"
            @click="mode = 'login'"
          >
            登录
          </button>
          <button
            :class="['auth-tab', { 'active': mode === 'register' }]"
            @click="mode = 'register'"
          >
            注册
          </button>
        </div>

        <!-- Forms -->
        <div class="auth-forms">
          <transition name="slide-fade" mode="out-in">
            <LoginForm
              v-if="mode === 'login'"
              key="login"
              @success="handleAuthSuccess"
              @switch-mode="mode = $event"
            />
            <RegisterForm
              v-else
              key="register"
              @success="handleAuthSuccess"
              @switch-mode="mode = $event"
            />
          </transition>
        </div>
      </div>

      <!-- Footer -->
      <div class="auth-footer">
        <p class="footer-text">
          登录即表示同意我们的
          <a href="#" class="footer-link">服务条款</a>
          和
          <a href="#" class="footer-link">隐私政策</a>
        </p>
      </div>

      <!-- Skip Button -->
      <button
        v-if="!isAuthenticated"
        class="skip-btn"
        @click.stop="handleSkip"
      >
        跳过，稍后登录
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useAuthStore } from '@/stores'
import LoginForm from '@/components/auth/LoginForm.vue'
import RegisterForm from '@/components/auth/RegisterForm.vue'

const router = useRouter()
const route = useRoute()
const authStore = useAuthStore()

const mode = ref<'login' | 'register'>('login')

const isAuthenticated = computed(() => authStore.isAuthenticated)

onMounted(() => {
  // 如果已经登录，直接跳转
  if (authStore.isAuthenticated) {
    handleAuthSuccess()
  }
})

function handleAuthSuccess() {
  // 获取重定向路径
  const redirect = (route.query.redirect as string) || '/connect'

  ElMessage.success('欢迎回来！')

  // 延迟跳转以显示成功消息
  setTimeout(() => {
    router.replace(redirect)
  }, 500)
}

function handleSkip() {
  // 跳过登录，直接去目标页面或连接页面
  const redirect = (route.query.redirect as string) || '/connect'
  router.push(redirect)
}
</script>

<style lang="scss" scoped>
.auth-view {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: var(--lelamp-space-lg);
  position: relative;
  overflow: hidden;
}

/* === Background === */
.auth-background {
  position: fixed;
  inset: 0;
  z-index: 0;
}

.bg-gradient {
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, #FFF8F0 0%, #FFE5D9 50%, #FFF0E0 100%);
}

.floating-shapes {
  position: absolute;
  inset: 0;
  overflow: hidden;
  pointer-events: none;
}

.shape {
  position: absolute;
  border-radius: 50%;
  opacity: 0.5;
  animation: float 8s ease-in-out infinite;
}

.shape-1 {
  width: 300px;
  height: 300px;
  background: radial-gradient(circle, var(--lelamp-peach-light) 0%, transparent 70%);
  top: -100px;
  right: -50px;
  animation-delay: 0s;
}

.shape-2 {
  width: 200px;
  height: 200px;
  background: radial-gradient(circle, var(--lelamp-sunny-light) 0%, transparent 70%);
  bottom: 100px;
  left: -50px;
  animation-delay: 2s;
}

.shape-3 {
  width: 150px;
  height: 150px;
  background: radial-gradient(circle, var(--lelamp-mint-light) 0%, transparent 70%);
  bottom: -30px;
  right: 20%;
  animation-delay: 4s;
}

@keyframes float {
  0%, 100% { transform: translateY(0) rotate(0deg); }
  50% { transform: translateY(-20px) rotate(5deg); }
}

/* === Container === */
.auth-container {
  position: relative;
  z-index: 1;
  max-width: 420px;
  width: 100%;
}

/* === Header === */
.auth-header {
  text-align: center;
  margin-bottom: var(--lelamp-space-xl);
}

.logo {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 80px;
  height: 80px;
  background: linear-gradient(135deg, var(--lelamp-peach) 0%, var(--lelamp-coral) 100%);
  border-radius: 50%;
  box-shadow: 0 8px 24px rgba(255, 107, 138, 0.3);
  margin-bottom: var(--lelamp-space-md);
}

.logo-icon {
  font-size: 2.5rem;
}

.auth-title {
  font-family: var(--lelamp-font-display);
  font-size: 2rem;
  font-weight: 700;
  background: linear-gradient(135deg, var(--lelamp-peach) 0%, var(--lelamp-coral) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin: 0 0 var(--lelamp-space-xs);
}

.auth-subtitle {
  font-size: 1rem;
  color: var(--lelamp-text-secondary);
  margin: 0;
}

/* === Card === */
.auth-card {
  background: var(--lelamp-bg-white);
  border-radius: var(--lelamp-radius-xl);
  padding: var(--lelamp-space-xl);
  box-shadow: var(--lelamp-shadow-lg);
  position: relative;
  overflow: hidden;
}

.card-decoration {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 4px;
  background: linear-gradient(90deg, var(--lelamp-peach), var(--lelamp-sunny), var(--lelamp-mint), var(--lelamp-sky));
}

/* === Tabs === */
.auth-tabs {
  display: flex;
  gap: var(--lelamp-space-sm);
  margin-bottom: var(--lelamp-space-xl);
  padding: var(--lelamp-space-xs);
  background: var(--lelamp-bg-gray);
  border-radius: var(--lelamp-radius-lg);
}

.auth-tab {
  flex: 1;
  padding: var(--lelamp-space-sm) var(--lelamp-space-md);
  font-size: 0.938rem;
  font-weight: 600;
  font-family: var(--lelamp-font-body);
  color: var(--lelamp-text-secondary);
  background: transparent;
  border: none;
  border-radius: var(--lelamp-radius-md);
  cursor: pointer;
  transition: all var(--lelamp-transition-bounce);

  &:hover {
    color: var(--lelamp-text-primary);
  }

  &.active {
    background: var(--lelamp-bg-white);
    color: var(--lelamp-peach-dark);
    box-shadow: var(--lelamp-shadow-sm);
  }
}

/* === Forms === */
.auth-forms {
  min-height: 320px;
}

/* === Footer === */
.auth-footer {
  text-align: center;
  margin-top: var(--lelamp-space-lg);
}

.footer-text {
  font-size: 0.813rem;
  color: var(--lelamp-text-tertiary);
  margin: 0;
}

.footer-link {
  color: var(--lelamp-peach);
  text-decoration: none;

  &:hover {
    text-decoration: underline;
  }
}

/* === Skip Button === */
.skip-btn {
  display: block;
  width: 100%;
  margin-top: var(--lelamp-space-lg);
  padding: var(--lelamp-space-md);
  font-size: 0.875rem;
  font-weight: 600;
  color: var(--lelamp-text-secondary);
  background: transparent;
  border: 1px dashed rgba(0, 0, 0, 0.2);
  border-radius: var(--lelamp-radius-md);
  cursor: pointer;
  transition: all var(--lelamp-transition-normal);

  &:hover {
    border-color: var(--lelamp-peach);
    color: var(--lelamp-peach);
    background: rgba(255, 154, 118, 0.05);
  }
}

/* === Transitions === */
.slide-fade-enter-active {
  transition: all 0.3s ease-out;
}

.slide-fade-leave-active {
  transition: all 0.2s ease-in;
}

.slide-fade-enter-from {
  opacity: 0;
  transform: translateX(20px);
}

.slide-fade-leave-to {
  opacity: 0;
  transform: translateX(-20px);
}

/* === Responsive === */
@media (max-width: 480px) {
  .auth-view {
    padding: var(--lelamp-space-md);
  }

  .auth-container {
    max-width: 100%;
  }

  .logo {
    width: 64px;
    height: 64px;
  }

  .logo-icon {
    font-size: 2rem;
  }

  .auth-title {
    font-size: 1.5rem;
  }

  .auth-card {
    padding: var(--lelamp-space-lg);
  }
}
</style>
