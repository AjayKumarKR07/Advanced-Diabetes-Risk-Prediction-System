/**
 * src/utils/theme.js
 * ==================
 * Application-wide appearance theme manager (Light / Dark / System).
 * Defaults to clean, high-contrast, professional Clinical White UI ('light').
 * Enforces clean white cards and typography across all clinical dashboards and workflows.
 */

const THEME_KEY = 'diabetes_ai_theme'

export function getStoredTheme() {
  const stored = localStorage.getItem(THEME_KEY)
  if (stored === 'dark') {
    return 'dark'
  }
  return 'light'
}

export function applyTheme(theme) {
  const root = document.documentElement
  const mode = theme || 'light'

  // Persist requested mode
  localStorage.setItem(THEME_KEY, mode)

  if (mode === 'dark') {
    root.classList.add('dark')
    root.classList.remove('light')
    root.setAttribute('data-theme', 'dark')
  } else {
    // Both 'light' and 'system' use the clean, professional Clinical White UI
    root.classList.remove('dark')
    root.classList.add('light')
    root.setAttribute('data-theme', 'light')
  }
}

// Initialize theme immediately upon script evaluation: Default to Clean Professional White UI
if (typeof window !== 'undefined') {
  // Reset any legacy dark/system mode from prior sessions so the user immediately gets the requested Clinical White UI
  const existing = localStorage.getItem(THEME_KEY)
  if (!existing || existing === 'dark' || existing === 'system') {
    localStorage.setItem(THEME_KEY, 'light')
  }
  applyTheme(getStoredTheme())

  // Listen to OS scheme changes ONLY if explicitly chosen 'system'
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
    if (localStorage.getItem(THEME_KEY) === 'system') {
      applyTheme('system')
    }
  })
}
