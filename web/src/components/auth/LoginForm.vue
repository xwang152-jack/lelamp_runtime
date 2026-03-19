<template>
  <div class="login-form">
    <div class="form-header">
      <h2 class="form-title">欢迎回来</h2>
      <p class="form-subtitle">登录以访问更多功能</p>
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

      <el-form-item prop="password">
        <el-input
          v-model="form.password"
          type="password"
          placeholder="密码"
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
          登录
        </el-button>
      </el-form-item>
    </el-form>

    <div class="form-footer">
      <span class="footer-text">还没有账号？</span>
      <el-button link type="primary" @click="$emit('switch-mode', 'register')">
        立即注册
      </el-button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive } from 'vue'
import { ElMessage, type FormInstance, type FormRules } from 'element-plus'
import { User, Lock } from '@element-plus/icons-vue'
import { useAuthStore } from '@/stores'

const emit = defineEmits<{
  (e: 'success'): void
  (e: 'switch-mode', mode: 'register' | 'login'): void
}>()

const authStore = useAuthStore()
const formRef = ref<FormInstance>()
const loading = ref(false)

const form = reactive({
  username: '',
  password: ''
})

const rules: FormRules = {
  username: [
    { required: true, message: '请输入用户名', trigger: 'blur' },
    { min: 3, max: 50, message: '用户名长度应为 3-50 个字符', trigger: 'blur' }
  ],
  password: [
    { required: true, message: '请输入密码', trigger: 'blur' },
    { min: 6, message: '密码长度至少为 6 个字符', trigger: 'blur' }
  ]
}

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
    const result = await authStore.login(form.username, form.password)
    if (result.success) {
      ElMessage.success('登录成功')
      emit('success')
    } else {
      ElMessage.error(result.error || '登录失败')
    }
  } catch (e) {
    console.error('Login error:', e)
    ElMessage.error('登录失败，请稍后重试')
  } finally {
    loading.value = false
  }
}
</script>

<style lang="scss" scoped>
.login-form {
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
  margin-bottom: var(--lelamp-space-lg);
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
  margin-top: var(--lelamp-space-lg);
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
</style>
