/**
 * src/services/auth.js
 * ====================
 * Authentication API client & local credential storage helpers.
 * Handles user registration, login, profile retrieval, and session storage.
 */

import apiClient from './api'

const TOKEN_KEY = 'diabetes_ai_token'
const USER_KEY  = 'diabetes_ai_user'

/**
 * Register a new user account.
 * POST /api/auth/register
 * @param {Object} data - { full_name, email, password }
 * @returns {Promise<{ access_token: string, token_type: string, user: Object }>}
 */
export async function registerUser({ full_name, email, password }) {
  const { data } = await apiClient.post('/auth/register', {
    full_name,
    email,
    password,
  })
  return data
}

/**
 * Authenticate existing user with email and password.
 * POST /api/auth/login
 * @param {Object} credentials - { email, password }
 * @returns {Promise<{ access_token: string, token_type: string, user: Object }>}
 */
export async function loginUser({ email, password }) {
  const { data } = await apiClient.post('/auth/login', {
    email,
    password,
  })
  return data
}

/**
 * Fetch authenticated user profile.
 * GET /api/auth/me
 * @returns {Promise<Object>}
 */
export async function fetchCurrentUser() {
  const { data } = await apiClient.get('/auth/me')
  return data
}

/**
 * Fetch enriched user profile with activity statistics.
 * GET /api/profile
 * @returns {Promise<Object>}
 */
export async function fetchUserProfile() {
  const { data } = await apiClient.get('/profile')
  return data
}

/**
 * Update authenticated user full name.
 * PUT /api/profile
 * @param {Object} data - { full_name }
 * @returns {Promise<Object>} updated user
 */
export async function updateUserProfile({ full_name }) {
  const { data } = await apiClient.put('/profile', { full_name })
  const storage = localStorage.getItem(TOKEN_KEY) ? localStorage : sessionStorage
  storage.setItem(USER_KEY, JSON.stringify(data))
  return data
}

/**
 * Change user password securely.
 * POST /api/auth/change-password
 * @param {Object} data - { current_password, new_password, confirm_password }
 * @returns {Promise<{ message: string }>}
 */
export async function changeUserPassword({ current_password, new_password, confirm_password }) {
  const { data } = await apiClient.post('/auth/change-password', {
    current_password,
    new_password,
    confirm_password,
  })
  return data
}

/**
 * Fetch authenticated user settings and preferences.
 * GET /api/settings
 * @returns {Promise<Object>}
 */
export async function fetchUserSettings() {
  const { data } = await apiClient.get('/settings')
  return data
}

/**
 * Update user settings and preferences in PostgreSQL.
 * PUT /api/settings
 * @param {Object} preferences - { theme, notify_assessment_ready, notify_security_alerts, notify_product_updates }
 * @returns {Promise<Object>} updated settings
 */
export async function updateUserSettings(preferences) {
  const { data } = await apiClient.put('/settings', preferences)
  return data
}

/**
 * Permanently delete authenticated user account and associated health records.
 * DELETE /api/account
 * @param {Object} payload - { password, confirmation: 'DELETE' }
 * @returns {Promise<{ message: string }>}
 */
export async function deleteUserAccount({ password, confirmation }) {
  const { data } = await apiClient.delete('/account', {
    data: { password, confirmation },
  })
  return data
}


/**
 * Store token and user object in local or session storage.
 */
export function storeAuth(token, user, rememberMe = true) {
  const storage = rememberMe ? localStorage : sessionStorage
  // Clear any existing opposite storage
  localStorage.removeItem(TOKEN_KEY)
  localStorage.removeItem(USER_KEY)
  sessionStorage.removeItem(TOKEN_KEY)
  sessionStorage.removeItem(USER_KEY)

  storage.setItem(TOKEN_KEY, token)
  if (user) {
    storage.setItem(USER_KEY, JSON.stringify(user))
  }
}

/**
 * Remove stored credentials upon logout.
 */
export function clearAuth() {
  localStorage.removeItem(TOKEN_KEY)
  localStorage.removeItem(USER_KEY)
  sessionStorage.removeItem(TOKEN_KEY)
  sessionStorage.removeItem(USER_KEY)
}

/**
 * Retrieve active JWT token if available.
 */
export function getStoredToken() {
  return localStorage.getItem(TOKEN_KEY) || sessionStorage.getItem(TOKEN_KEY)
}

/**
 * Retrieve cached user profile if available.
 */
export function getStoredUser() {
  const raw = localStorage.getItem(USER_KEY) || sessionStorage.getItem(USER_KEY)
  if (!raw) return null
  try {
    return JSON.parse(raw)
  } catch {
    return null
  }
}

