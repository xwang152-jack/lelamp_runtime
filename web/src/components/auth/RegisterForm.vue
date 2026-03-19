<template>
  <div class="register-form">
    <div class="form-header">
      <h2 class="form-title">创建账号</h2>
      <p class="form-subtitle">加入 LeLamp 智能台灯社区</p>
    </div>

    <el-form
      ref="formRef"
      :model="form"
      :rules="rules"
      label-position="top"
      size="large"
      @submit.prevent="handleSubmit"
    >
      <el-form-item prop="username">
        <el-input
          v-model="form.username"
          placeholder="用户名"
          :prefix-icon="User"
          :disabled="loading"
        />
      </el-form-item>

      <el-form-item prop="email">
        <el-input
          v-model="form.email"
          type="email"
          placeholder="邮箱地址"
          :prefix-icon="Message"
          :disabled="loading"
        />
      </el-form-item>

      <el-form-item prop="password">
        <el-input
          v-model="form.password"
          type="password"
          placeholder="密码（至少6位）"
          :prefix-icon="Lock"
          :disabled="loading"
          show-password
        />
        <!-- 密码强度指示器 -->
        <div v-if="form.password" class="password-strength">
          <div class="strength-bar">
            <div
              class="strength-fill"
              :class="strengthClass"
              :style="{ width: `${(passwordStrength.score / 4) * 100}%` }"
            ></div>
          </div>
          <span class="strength-text">{{ passwordStrength.feedback }}</span>
        </div>
      </el-form-item>

      <el-form-item prop="confirmPassword">
        <el-input
          v-model="form.confirmPassword"
          type="password"
          placeholder="确认密码"
          :prefix-icon="Lock"
          :disabled="loading"
          show-password
          @keyup.enter="handleSubmit"
        />
      </el-form-item>

      <el-form-item>
        <el-button
          type="primary"
          class="submit-btn"
          :loading="loading"
          @click="handleSubmit"
        >
          注册
        </el-button>
      </el-form-item>
    </el-form>

    <div class="form-footer">
      <span class="footer-text">已有账号？</span>
      <el-button link type="primary" @click="$emit('switch-mode', 'login')">
        立即登录
      </el-button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed } from 'vue'
import { ElMessage, type FormInstance, type FormRules } from 'element-plus'
import { User, Lock, Message } from '@element-plus/icons-vue'
import { useAuthStore } from '@/stores'
import { checkPasswordStrength } from '@/utils/security'

const emit = defineEmits<{
  (e: 'success'): void
  (e: 'switch-mode', mode: 'register' | 'login'): void
}>()

const authStore = useAuthStore()
const formRef = ref<FormInstance>()
const loading = ref(false)

const form = reactive({
  username: '',
  email: '',
  password: '',
  confirmPassword: ''
})

const validateConfirmPassword = (_rule: any, value: string, callback: any) => {
  if (value === '') {
    callback(new Error('请再次输入密码'))
  } else if (value !== form.password) {
    callback(new Error('两次输入的密码不一致'))
  } else {
    callback()
  }
}

const rules: FormRules = {
  username: [
    { required: true, message: '请输入用户名', trigger: 'blur' },
    { min: 3, max: 50, message: '用户名长度应为 3-50 个字符', trigger: 'blur' },
    { pattern: /^[a-zA-Z0-9_]+$/, message: '用户名只能包含字母、数字和下划线', trigger: 'blur' }
  ],
  email: [
    { required: true, message: '请输入邮箱地址', trigger: 'blur' },
    { type: 'email', message: '请输入有效的邮箱地址', trigger: 'blur' }
  ],
  password: [
    { required: true, message: '请输入密码', trigger: 'blur' },
    { min: 6, max: 100, message: '密码长度应为 6-100 个字符', trigger: 'blur' }
  ],
  confirmPassword: [
    { required: true, message: '请确认密码', trigger: 'blur' },
    { validator: validateConfirmPassword, trigger: 'blur' }
  ]
}

// 密码强度计算
const passwordStrength = computed(() => {
  return checkPasswordStrength(form.password)
})

const strengthClass = computed(() => {
  const score = passwordStrength.value.score
  if (score <= 1) return 'weak'
  if (score === 2) return 'fair'
  if (score === 3) return 'good'
  return 'strong'
})

async function handleSubmit() {
  if (!formRef.value) {
    ElMessage.error('表单未初始化，请刷新页面重试')
    return
  }

  try {
    const valid = await formRef.value.validate()
    if (!valid) {
      ElMessage.warning('请检查输入信息')
      return
    }
  } catch (error) {
    console.error('Form validation error:', error)
    ElMessage.error('表单验证失败')
    return
  }

  loading.value = true
  try {
    const result = await authStore.register(
      form.username,
      form.email,
      form.password
    )
    if (result.success) {
      ElMessage.success('注册成功')
      emit('success')
    } else {
      ElMessage.error(result.error || '注册失败')
    }
  } catch (e) {
    console.error('Register error:', e)
    ElMessage.error('注册失败，请稍后重试')
  } finally {
    loading.value = false
  }
}
</script>

<style lang="scss" scoped>
.register-form {
  width: 100%;
}

.form-header {
  text-align: center;
  margin-bottom: var(--lelamp-space-xl);
}

.form-title {
  font-family: var(--lelamp-font-display);
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--lelamp-text-primary);
  margin: 0 0 var(--lelamp-space-sm);
}

.form-subtitle {
  font-size: 0.938rem;
  color: var(--lelamp-text-secondary);
  margin: 0;
}

:deep(.el-form-item) {
  margin-bottom: var(--lelamp-space-md);
}

:deep(.el-input__wrapper) {
  border-radius: var(--lelamp-radius-lg);
  padding: var(--lelamp-space-md) var(--lelamp-space-lg);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);

  &:hover {
    box-shadow: 0 2px 12px rgba(255, 154, 118, 0.15);
  }

  &.is-focus {
    box-shadow: 0 0 0 3px rgba(255, 154, 118, 0.2);
  }
}

:deep(.el-input__inner) {
  font-size: 1rem;
}

.submit-btn {
  width: 100%;
  height: 48px;
  font-size: 1rem;
  font-weight: 700;
  border-radius: var(--lelamp-radius-lg);
  background: linear-gradient(135deg, var(--lelamp-peach) 0%, var(--lelamp-coral) 100%);
  border: none;
  box-shadow: 0 4px 16px rgba(255, 107, 138, 0.3);
  margin-top: var(--lelamp-space-sm);

  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(255, 107, 138, 0.4);
  }

  &:active {
    transform: translateY(0);
  }
}

.form-footer {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--lelamp-space-xs);
  margin-top: var(--lelamp-space-md);
}

.footer-text {
  font-size: 0.875rem;
  color: var(--lelamp-text-secondary);
}

:deep(.el-button--primary) {
  &:hover {
    background-color: var(--lelamp-peach);
    border-color: var(--lelamp-peach);
  }
}

/* === Password Strength === */
.password-strength {
  display: flex;
  align-items: center;
  gap: var(--lelamp-space-sm);
  margin-top: var(--lelamp-space-xs);
}

.strength-bar {
  flex: 1;
  height: 4px;
  background: var(--lelamp-bg-gray);
  border-radius: 2px;
  overflow: hidden;
}

.strength-fill {
  height: 100%;
  transition: all var(--lelamp-transition-normal);
  border-radius: 2px;

  &.weak {
    background: #ff4d4f;
  }

  &.fair {
    background: #faad14;
  }

  &.good {
    background: #52c41a;
  }

  &.strong {
    background: #1890ff;
  }
}

.strength-text {
  font-size: 0.75rem;
  color: var(--lelamp-text-secondary);
  white-space: nowrap;
}
</style>
