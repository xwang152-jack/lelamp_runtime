/**
 * 订阅状态管理 Store
 */

import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { SubscriptionTier, FeatureFlags, SubscriptionState } from '@/types/auth'

/**
 * 功能配置映射
 */
const FEATURE_CONFIG: Record<SubscriptionTier, FeatureFlags> = {
  free: {
    motorHealth: false,
    advancedAI: false,
    otaUpdates: false,
    deviceBinding: false,
    historyRecords: false
  },
  basic: {
    motorHealth: false,
    advancedAI: false,
    otaUpdates: true,
    deviceBinding: true,
    historyRecords: true
  },
  pro: {
    motorHealth: true,
    advancedAI: true,
    otaUpdates: true,
    deviceBinding: true,
    historyRecords: true
  }
}

export const useSubscriptionStore = defineStore('subscription', () => {
  // State
  const tier = ref<SubscriptionTier>('free')

  // Computed
  const isPremium = computed(() => tier.value !== 'free')

  const features = computed<FeatureFlags>(() => FEATURE_CONFIG[tier.value])

  const hasMotorHealth = computed(() => features.value.motorHealth)

  const hasAdvancedAI = computed(() => features.value.advancedAI)

  const hasOtaUpdates = computed(() => features.value.otaUpdates)

  const hasDeviceBinding = computed(() => features.value.deviceBinding)

  const hasHistoryRecords = computed(() => features.value.historyRecords)

  // Actions

  /**
   * 设置订阅层级
   */
  function setTier(newTier: SubscriptionTier) {
    tier.value = newTier
  }

  /**
   * 重置为免费用户
   */
  function resetToFree() {
    tier.value = 'free'
  }

  /**
   * 从用户数据加载订阅信息
   * 这里可以根据实际的用户属性来判断订阅层级
   */
  function loadFromUser(isAdmin: boolean) {
    // 管理员默认为 pro 用户
    if (isAdmin) {
      tier.value = 'pro'
    } else {
      tier.value = 'free'
    }
  }

  return {
    // State
    tier,

    // Computed
    isPremium,
    features,
    hasMotorHealth,
    hasAdvancedAI,
    hasOtaUpdates,
    hasDeviceBinding,
    hasHistoryRecords,

    // Actions
    setTier,
    resetToFree,
    loadFromUser
  }
})
