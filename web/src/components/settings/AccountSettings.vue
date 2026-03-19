<template>
  <div class="account-settings">
    <!-- User Info Card -->
    <div class="user-card">
      <div class="user-avatar">
        <span class="avatar-text">{{ userInitial }}</span>
      </div>
      <div class="user-info">
        <h3 class="user-name">{{ authStore.user?.username }}</h3>
        <p class="user-email">{{ authStore.user?.email }}</p>
      </div>
      <div class="user-badges">
        <el-tag v-if="authStore.isAdmin" type="warning" size="small">
          <span class="badge-icon">👑</span>
          管理员
        </el-tag>
        <el-tag v-else type="info" size="small">免费用户</el-tag>
      </div>
    </div>

    <!-- Premium Features Notice -->
    <div class="premium-notice">
      <div class="notice-header">
        <span class="notice-icon">✨</span>
        <h4 class="notice-title">升级到高级版</h4>
      </div>
      <p class="notice-desc">解锁更多功能，享受更好的体验</p>
      <div class="feature-list">
        <div class="feature-item">
          <span class="feature-icon">🔧</span>
          <span>电机健康监控</span>
          <el-tag v-if="hasMotorHealth" type="success" size="small">已解锁</el-tag>
          <el-tag v-else type="info" size="small">高级功能</el-tag>
        </div>
        <div class="feature-item">
          <span class="feature-icon">🧠</span>
          <span>高级 AI 模型</span>
          <el-tag v-if="hasAdvancedAI" type="success" size="small">已解锁</el-tag>
          <el-tag v-else type="info" size="small">高级功能</el-tag>
        </div>
        <div class="feature-item">
          <span class="feature-icon">📦</span>
          <span>设备绑定</span>
          <el-tag v-if="hasDeviceBinding" type="success" size="small">已解锁</el-tag>
          <el-tag v-else type="info" size="small">需登录</el-tag>
        </div>
      </div>
    </div>

    <!-- Account Actions -->
    <div class="account-actions">
      <el-button
        type="primary"
        class="action-btn"
        @click="goToProfile"
      >
        <span class="btn-icon">👤</span>
        个人中心
      </el-button>
      <el-button
        type="primary"
        class="action-btn"
        @click="goToDevices"
      >
        <span class="btn-icon">📱</span>
        设备管理
      </el-button>
    </div>

    <!-- Logout Button -->
    <div class="logout-section">
      <el-button
        type="danger"
        plain
        class="logout-btn"
        @click="handleLogout"
      >
        <span class="btn-icon">🚪</span>
        退出登录
      </el-button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessageBox, ElMessage } from 'element-plus'
import { useAuthStore, useSubscriptionStore } from '@/stores'
import { getUserInitial } from '@/utils/device'

const router = useRouter()
const authStore = useAuthStore()
const subscriptionStore = useSubscriptionStore()

const userInitial = computed(() => {
  return getUserInitial(authStore.user?.username)
})

const hasMotorHealth = computed(() => subscriptionStore.hasMotorHealth)
const hasAdvancedAI = computed(() => subscriptionStore.hasAdvancedAI)
const hasDeviceBinding = computed(() => subscriptionStore.hasDeviceBinding)

function goToProfile() {
  router.push('/profile')
}

function goToDevices() {
  router.push('/devices')
}

async function handleLogout() {
  try {
    await ElMessageBox.confirm(
      '确定要退出登录吗？',
      '退出登录',
      {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )

    authStore.logout()
    ElMessage.success('已退出登录')
    router.push('/connect')
  } catch {
    // User cancelled
  }
}
</script>

<style lang="scss" scoped>
.account-settings {
  display: flex;
  flex-direction: column;
  gap: var(--lelamp-space-lg);
}

/* === User Card === */
.user-card {
  display: flex;
  align-items: center;
  gap: var(--lelamp-space-md);
  padding: var(--lelamp-space-lg);
  background: linear-gradient(135deg, rgba(255, 154, 118, 0.1), rgba(255, 217, 61, 0.1));
  border: 2px solid rgba(255, 154, 118, 0.2);
  border-radius: var(--lelamp-radius-xl);
}

.user-avatar {
  width: 64px;
  height: 64px;
  border-radius: 50%;
  background: linear-gradient(135deg, var(--lelamp-peach) 0%, var(--lelamp-coral) 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 4px 12px rgba(255, 107, 138, 0.3);
}

.avatar-text {
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--lelamp-bg-white);
}

.user-info {
  flex: 1;
}

.user-name {
  font-size: 1.125rem;
  font-weight: 700;
  color: var(--lelamp-text-primary);
  margin: 0 0 var(--lelamp-space-xs);
}

.user-email {
  font-size: 0.875rem;
  color: var(--lelamp-text-secondary);
  margin: 0;
}

.user-badges {
  display: flex;
  gap: var(--lelamp-space-xs);
}

.badge-icon {
  margin-right: var(--lelamp-space-xs);
}

/* === Premium Notice === */
.premium-notice {
  padding: var(--lelamp-space-lg);
  background: var(--lelamp-bg-white);
  border: 2px dashed rgba(255, 154, 118, 0.3);
  border-radius: var(--lelamp-radius-xl);
}

.notice-header {
  display: flex;
  align-items: center;
  gap: var(--lelamp-space-sm);
  margin-bottom: var(--lelamp-space-sm);
}

.notice-icon {
  font-size: 1.5rem;
}

.notice-title {
  font-size: 1rem;
  font-weight: 700;
  color: var(--lelamp-text-primary);
  margin: 0;
}

.notice-desc {
  font-size: 0.875rem;
  color: var(--lelamp-text-secondary);
  margin: 0 0 var(--lelamp-space-md);
}

.feature-list {
  display: flex;
  flex-direction: column;
  gap: var(--lelamp-space-sm);
}

.feature-item {
  display: flex;
  align-items: center;
  gap: var(--lelamp-space-sm);
  padding: var(--lelamp-space-sm);
  background: var(--lelamp-bg-gray);
  border-radius: var(--lelamp-radius-md);
  font-size: 0.875rem;
  color: var(--lelamp-text-primary);
}

.feature-icon {
  font-size: 1rem;
}

/* === Account Actions === */
.account-actions {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--lelamp-space-md);
}

.action-btn {
  height: 48px;
  font-size: 0.938rem;
  font-weight: 600;
  border-radius: var(--lelamp-radius-lg);
  background: linear-gradient(135deg, var(--lelamp-peach) 0%, var(--lelamp-sunny) 100%);
  border: none;
  box-shadow: 0 2px 8px rgba(255, 154, 118, 0.2);

  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(255, 154, 118, 0.3);
  }

  &:active {
    transform: translateY(0);
  }
}

.btn-icon {
  margin-right: var(--lelamp-space-xs);
  font-size: 1rem;
}

/* === Logout Section === */
.logout-section {
  padding-top: var(--lelamp-space-md);
  border-top: 1px dashed rgba(0, 0, 0, 0.1);
}

.logout-btn {
  width: 100%;
  height: 44px;
  font-size: 0.938rem;
  font-weight: 600;
  border-radius: var(--lelamp-radius-lg);
  border-color: var(--lelamp-coral);
  color: var(--lelamp-coral);

  &:hover {
    background: rgba(255, 107, 138, 0.1);
  }
}

@media (max-width: 480px) {
  .account-actions {
    grid-template-columns: 1fr;
  }

  .user-avatar {
    width: 48px;
    height: 48px;
  }

  .avatar-text {
    font-size: 1.125rem;
  }
}
</style>
