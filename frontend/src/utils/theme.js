/**
 * src/utils/theme.js
 * ==================
 * Application-wide appearance theme manager (Light / Dark / System).
 * Applies .dark class and data-theme to document.documentElement.
 */

const THEME_KEY = 'diabetes_ai_theme'

export function getStoredTheme() {
  return localStorage.getItem(THEME_KEY) || 'system'
}

export function applyTheme(theme) {
  const root = document.documentElement
  const mode = theme || getStoredTheme()

  localStorage.setItem(THEME_KEY, mode)
  root.setAttribute('data-theme', mode)

  if (mode === 'dark') {
    root.classList.add('dark')
  } else if (mode === 'light') {
    root.classList.remove('dark')
  } else {
    // System preference
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches
    if (prefersDark) {
      root.classList.add('dark')
    } else {
      root.classList.remove('dark')
    }
  }
}

// Initialize theme immediately upon script evaluation
if (typeof window !== 'undefined') {
  applyTheme(getStoredTheme())

  // Listen to OS scheme changes if system default is selected
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
    if (getStoredTheme() === 'system') {
      applyTheme('system')
    }
  })
}

