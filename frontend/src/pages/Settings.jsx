/**
 * src/pages/Settings.jsx
 * =======================
 * Complete User Account & Application Settings Management for DiabetesRisk AI (Phase 6.14).
 * 
 * Sections:
 * 1. Security Settings: Secure password change with strength analysis and re-authentication.
 * 2. Notification Preferences: Assessment, security, and product update toggles stored in PostgreSQL.
 * 3. Appearance Preferences: Light / Dark / System default theme selector synced with document root.
 * 4. Session Management: Token cryptographic parameters and session revocation.
 * 5. Danger Zone: Re-authenticated, permanent account deletion cascading user health records.
 */

import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  KeyRound,
  Shield,
  Eye,
  EyeOff,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Lock,
  LogOut,
  Info,
  Server,
  Cpu,
  Bell,
  Sun,
  Moon,
  Laptop,
  AlertTriangle,
  Trash2,
  X,
  ShieldAlert,
} from 'lucide-react'
import { useAuth } from '../context/AuthContext'
import {
  changeUserPassword,
  deleteUserAccount,
  fetchUserSettings,
  updateUserSettings,
} from '../services/auth'
import { applyTheme, getStoredTheme } from '../utils/theme'

export default function SettingsPage() {
  const navigate = useNavigate()
  const { user, logout } = useAuth()

  // Active Tab: 'security' | 'notifications' | 'appearance' | 'session' | 'danger'
  const [activeTab, setActiveTab] = useState('security')

  // ── 1. Change Password State ──────────────────────────────────────────────
  const [currentPassword, setCurrentPassword] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [showCurrentPassword, setShowCurrentPassword] = useState(false)
  const [showNewPassword, setShowNewPassword] = useState(false)
  const [showConfirmPassword, setShowConfirmPassword] = useState(false)
  const [loadingPassword, setLoadingPassword] = useState(false)
  const [passwordSuccess, setPasswordSuccess] = useState('')
  const [passwordError, setPasswordError] = useState('')

  // ── 2. Settings & Notifications State ─────────────────────────────────────
  const [settingsLoading, setSettingsLoading] = useState(true)
  const [settingsSaving, setSettingsSaving] = useState(false)
  const [settingsSuccess, setSettingsSuccess] = useState('')
  const [settingsError, setSettingsError] = useState('')

  const [themePreference, setThemePreference] = useState(getStoredTheme())
  const [notifyAssessmentReady, setNotifyAssessmentReady] = useState(true)
  const [notifySecurityAlerts, setNotifySecurityAlerts] = useState(true)
  const [notifyProductUpdates, setNotifyProductUpdates] = useState(false)

  // ── 3. Account Deletion Modal State ───────────────────────────────────────
  const [showDeleteModal, setShowDeleteModal] = useState(false)
  const [deletePassword, setDeletePassword] = useState('')
  const [deleteConfirmation, setDeleteConfirmation] = useState('')
  const [deleteLoading, setDeleteLoading] = useState(false)
  const [deleteError, setDeleteError] = useState('')

  // Load preferences from GET /api/settings on mount
  useEffect(() => {
    let isMounted = true
    setSettingsLoading(true)

    fetchUserSettings()
      .then((data) => {
        if (isMounted) {
          if (data.theme) {
            setThemePreference(data.theme)
            applyTheme(data.theme)
          }
          if (data.notify_assessment_ready !== undefined) {
            setNotifyAssessmentReady(data.notify_assessment_ready)
          }
          if (data.notify_security_alerts !== undefined) {
            setNotifySecurityAlerts(data.notify_security_alerts)
          }
          if (data.notify_product_updates !== undefined) {
            setNotifyProductUpdates(data.notify_product_updates)
          }
        }
      })
      .catch(() => {
        // Fallback to local theme if settings request fails
        if (isMounted) setThemePreference(getStoredTheme())
      })
      .finally(() => {
        if (isMounted) setSettingsLoading(false)
      })

    return () => {
      isMounted = false
    }
  }, [])

  // Calculate Password Strength Score (0 to 4)
  const calculatePasswordStrength = (pass) => {
    if (!pass) return 0
    let score = 0
    if (pass.length >= 8) score += 1
    if (/[A-Z]/.test(pass) && /[a-z]/.test(pass)) score += 1
    if (/[0-9]/.test(pass)) score += 1
    if (/[^A-Za-z0-9]/.test(pass)) score += 1
    return score
  }

  const pwStrength = calculatePasswordStrength(newPassword)
  const strengthLabels = ['Too Short', 'Weak', 'Fair', 'Good', 'Strong']
  const strengthColors = ['bg-slate-200', 'bg-rose-500', 'bg-amber-500', 'bg-blue-500', 'bg-emerald-500']

  // ── Handle Password Update ────────────────────────────────────────────────
  const handlePasswordSubmit = async (e) => {
    e.preventDefault()
    setPasswordError('')
    setPasswordSuccess('')

    if (!currentPassword) {
      setPasswordError('Please provide your current account password.')
      return
    }
    if (newPassword.length < 8) {
      setPasswordError('New password must be at least 8 characters long.')
      return
    }
    if (newPassword === currentPassword) {
      setPasswordError('New password cannot be the same as your current password.')
      return
    }
    if (newPassword !== confirmPassword) {
      setPasswordError('New password and confirmation password do not match.')
      return
    }

    setLoadingPassword(true)
    try {
      const res = await changeUserPassword({
        current_password: currentPassword,
        new_password: newPassword,
        confirm_password: confirmPassword,
      })
      setPasswordSuccess(res.message || 'Password updated successfully in PostgreSQL.')
      setCurrentPassword('')
      setNewPassword('')
      setConfirmPassword('')
    } catch (err) {
      setPasswordError(
        err.response?.data?.detail ||
          'Failed to update password. Please check your current password.'
      )
    } finally {
      setLoadingPassword(false)
    }
  }

  // ── Handle Notification Settings Save ─────────────────────────────────────
  const handleSaveNotifications = async () => {
    setSettingsSaving(true)
    setSettingsSuccess('')
    setSettingsError('')

    try {
      await updateUserSettings({
        notify_assessment_ready: notifyAssessmentReady,
        notify_security_alerts: notifySecurityAlerts,
        notify_product_updates: notifyProductUpdates,
      })
      setSettingsSuccess('Notification preferences saved successfully.')
    } catch (err) {
      setSettingsError(
        err.response?.data?.detail || 'Failed to save notification preferences.'
      )
    } finally {
      setSettingsSaving(false)
    }
  }

  // ── Handle Theme Selection ────────────────────────────────────────────────
  const handleThemeChange = async (theme) => {
    setThemePreference(theme)
    applyTheme(theme)

    try {
      await updateUserSettings({ theme })
      setSettingsSuccess(`Theme updated to ${theme.charAt(0).toUpperCase() + theme.slice(1)}.`)
    } catch {
      // Local theme already applied even if DB fails
    }
  }

  // ── Handle Logout ─────────────────────────────────────────────────────────
  const handleLogout = () => {
    logout()
    navigate('/login')
  }

  // ── Handle Account Deletion ───────────────────────────────────────────────
  const handleDeleteAccountSubmit = async (e) => {
    e.preventDefault()
    setDeleteError('')

    if (!deletePassword) {
      setDeleteError('Please enter your current account password to authorize deletion.')
      return
    }
    if (deleteConfirmation.trim().toUpperCase() !== 'DELETE') {
      setDeleteError('Confirmation token mismatch. You must type DELETE exactly.')
      return
    }

    setDeleteLoading(true)
    try {
      await deleteUserAccount({
        password: deletePassword,
        confirmation: 'DELETE',
      })
      logout()
      navigate('/login', {
        state: { message: 'Your account and all associated health records have been permanently deleted.' },
      })
    } catch (err) {
      setDeleteError(
        err.response?.data?.detail || 'Failed to delete account. Please verify your password.'
      )
      setDeleteLoading(false)
    }
  }

  return (
    <div className="p-4 sm:p-6 lg:p-8 max-w-4xl mx-auto space-y-6">
      {/* Page Title */}
      <div>
        <h1 className="text-2xl font-extrabold text-slate-900 tracking-tight">Account &amp; Security Settings</h1>
        <p className="text-xs sm:text-sm text-slate-500 mt-1">
          Configure security credentials, notifications, appearance themes, session details, and privacy controls.
        </p>
      </div>

      {/* Navigation Tabs */}
      <div className="flex items-center gap-1.5 p-1.5 bg-slate-100/90 rounded-2xl border border-slate-200/80 overflow-x-auto">
        <button
          onClick={() => setActiveTab('security')}
          className={`flex items-center gap-2 px-4 py-2.5 rounded-xl text-xs font-bold transition-all whitespace-nowrap ${
            activeTab === 'security'
              ? 'bg-white text-slate-900 shadow-xs'
              : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
          }`}
        >
          <KeyRound className="w-3.5 h-3.5 text-teal-600" />
          <span>Security &amp; Password</span>
        </button>

        <button
          onClick={() => setActiveTab('notifications')}
          className={`flex items-center gap-2 px-4 py-2.5 rounded-xl text-xs font-bold transition-all whitespace-nowrap ${
            activeTab === 'notifications'
              ? 'bg-white text-slate-900 shadow-xs'
              : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
          }`}
        >
          <Bell className="w-3.5 h-3.5 text-blue-600" />
          <span>Notifications</span>
        </button>

        <button
          onClick={() => setActiveTab('appearance')}
          className={`flex items-center gap-2 px-4 py-2.5 rounded-xl text-xs font-bold transition-all whitespace-nowrap ${
            activeTab === 'appearance'
              ? 'bg-white text-slate-900 shadow-xs'
              : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
          }`}
        >
          <Sun className="w-3.5 h-3.5 text-amber-500" />
          <span>Appearance</span>
        </button>

        <button
          onClick={() => setActiveTab('session')}
          className={`flex items-center gap-2 px-4 py-2.5 rounded-xl text-xs font-bold transition-all whitespace-nowrap ${
            activeTab === 'session'
              ? 'bg-white text-slate-900 shadow-xs'
              : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
          }`}
        >
          <Shield className="w-3.5 h-3.5 text-indigo-600" />
          <span>Session Info</span>
        </button>

        <button
          onClick={() => setActiveTab('danger')}
          className={`flex items-center gap-2 px-4 py-2.5 rounded-xl text-xs font-bold transition-all whitespace-nowrap ${
            activeTab === 'danger'
              ? 'bg-rose-50 text-rose-700 border border-rose-200'
              : 'text-rose-600 hover:text-rose-800 hover:bg-rose-50/50'
          }`}
        >
          <Trash2 className="w-3.5 h-3.5" />
          <span>Danger Zone</span>
        </button>
      </div>

      {/* Global Banners */}
      {settingsSuccess && (
        <div className="p-4 rounded-2xl bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs font-semibold flex items-center justify-between gap-3 animate-fadeIn">
          <div className="flex items-center gap-2.5">
            <CheckCircle2 className="w-5 h-5 text-emerald-600 flex-shrink-0" />
            <span>{settingsSuccess}</span>
          </div>
          <button onClick={() => setSettingsSuccess('')} className="text-emerald-700 hover:text-emerald-900">
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {settingsError && (
        <div className="p-4 rounded-2xl bg-rose-50 border border-rose-200 text-rose-800 text-xs font-semibold flex items-center justify-between gap-3 animate-fadeIn">
          <div className="flex items-center gap-2.5">
            <AlertCircle className="w-5 h-5 text-rose-600 flex-shrink-0" />
            <span>{settingsError}</span>
          </div>
          <button onClick={() => setSettingsError('')} className="text-rose-700 hover:text-rose-900">
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* TAB 1: SECURITY & PASSWORD                                          */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      {activeTab === 'security' && (
        <div className="space-y-6">
          <div className="bg-white rounded-3xl border border-slate-200/90 shadow-sm p-6 sm:p-8">
            <div className="pb-4 mb-6 border-b border-slate-100">
              <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
                <KeyRound className="w-5 h-5 text-teal-600" /> Change Account Password
              </h2>
              <p className="text-xs text-slate-500 mt-1">
                Passwords are salted and cryptographically hashed with 12-round Bcrypt before persisting to PostgreSQL.
              </p>
            </div>

            {passwordSuccess && (
              <div className="mb-6 p-4 rounded-2xl bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs font-semibold flex items-center gap-3">
                <CheckCircle2 className="w-5 h-5 text-emerald-600 flex-shrink-0" />
                <span>{passwordSuccess}</span>
              </div>
            )}

            {passwordError && (
              <div className="mb-6 p-4 rounded-2xl bg-rose-50 border border-rose-200 text-rose-800 text-xs font-semibold flex items-center gap-3">
                <AlertCircle className="w-5 h-5 text-rose-600 flex-shrink-0" />
                <span>{passwordError}</span>
              </div>
            )}

            <form onSubmit={handlePasswordSubmit} className="space-y-5 max-w-lg">
              {/* Current Password */}
              <div>
                <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5">
                  Current Password <span className="text-red-500">*</span>
                </label>
                <div className="relative">
                  <Lock className="absolute left-3.5 top-3 w-4 h-4 text-slate-400" />
                  <input
                    type={showCurrentPassword ? 'text' : 'password'}
                    value={currentPassword}
                    onChange={(e) => setCurrentPassword(e.target.value)}
                    placeholder="Enter current password"
                    className="w-full pl-10 pr-10 py-2.5 rounded-xl border border-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm font-semibold text-slate-800"
                  />
                  <button
                    type="button"
                    onClick={() => setShowCurrentPassword(!showCurrentPassword)}
                    className="absolute right-3.5 top-3 text-slate-400 hover:text-slate-600"
                  >
                    {showCurrentPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
              </div>

              {/* New Password */}
              <div>
                <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5">
                  New Password <span className="text-red-500">*</span>
                </label>
                <div className="relative">
                  <Lock className="absolute left-3.5 top-3 w-4 h-4 text-slate-400" />
                  <input
                    type={showNewPassword ? 'text' : 'password'}
                    value={newPassword}
                    onChange={(e) => setNewPassword(e.target.value)}
                    placeholder="At least 8 characters"
                    className="w-full pl-10 pr-10 py-2.5 rounded-xl border border-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm font-semibold text-slate-800"
                  />
                  <button
                    type="button"
                    onClick={() => setShowNewPassword(!showNewPassword)}
                    className="absolute right-3.5 top-3 text-slate-400 hover:text-slate-600"
                  >
                    {showNewPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>

                {/* Password Strength Meter */}
                {newPassword && (
                  <div className="mt-2 space-y-1">
                    <div className="flex items-center justify-between text-[11px]">
                      <span className="text-slate-500">Strength:</span>
                      <span className="font-bold text-slate-700">{strengthLabels[pwStrength]}</span>
                    </div>
                    <div className="w-full bg-slate-100 h-1.5 rounded-full overflow-hidden flex gap-0.5">
                      <div className={`h-full transition-all duration-300 ${pwStrength >= 1 ? strengthColors[pwStrength] : 'bg-transparent'}`} style={{ width: '25%' }} />
                      <div className={`h-full transition-all duration-300 ${pwStrength >= 2 ? strengthColors[pwStrength] : 'bg-transparent'}`} style={{ width: '25%' }} />
                      <div className={`h-full transition-all duration-300 ${pwStrength >= 3 ? strengthColors[pwStrength] : 'bg-transparent'}`} style={{ width: '25%' }} />
                      <div className={`h-full transition-all duration-300 ${pwStrength >= 4 ? strengthColors[pwStrength] : 'bg-transparent'}`} style={{ width: '25%' }} />
                    </div>
                  </div>
                )}
                <p className="text-[11px] text-slate-400 mt-1">Minimum 8 characters. Must differ from current password.</p>
              </div>

              {/* Confirm New Password */}
              <div>
                <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5">
                  Confirm New Password <span className="text-red-500">*</span>
                </label>
                <div className="relative">
                  <Lock className="absolute left-3.5 top-3 w-4 h-4 text-slate-400" />
                  <input
                    type={showConfirmPassword ? 'text' : 'password'}
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    placeholder="Re-enter your new password"
                    className="w-full pl-10 pr-10 py-2.5 rounded-xl border border-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm font-semibold text-slate-800"
                  />
                  <button
                    type="button"
                    onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                    className="absolute right-3.5 top-3 text-slate-400 hover:text-slate-600"
                  >
                    {showConfirmPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
              </div>

              <div className="pt-2">
                <button
                  type="submit"
                  disabled={loadingPassword}
                  className="px-6 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-700 text-white text-xs font-bold flex items-center gap-2 transition-all shadow-md shadow-blue-500/20 disabled:opacity-50"
                >
                  {loadingPassword ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span>Updating Password...</span>
                    </>
                  ) : (
                    'Update Password'
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* TAB 2: NOTIFICATION PREFERENCES                                     */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      {activeTab === 'notifications' && (
        <div className="bg-white rounded-3xl border border-slate-200/90 shadow-sm p-6 sm:p-8 space-y-6">
          <div className="pb-4 border-b border-slate-100">
            <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
              <Bell className="w-5 h-5 text-blue-600" /> Notification Preferences
            </h2>
            <p className="text-xs text-slate-500 mt-1">
              Select which clinical notifications and automated alerts you wish to receive. Preferences are saved to PostgreSQL.
            </p>
          </div>

          <div className="space-y-4">
            {/* Assessment Completion */}
            <div className="flex items-center justify-between p-4 rounded-2xl bg-slate-50 border border-slate-200/80">
              <div className="space-y-0.5 max-w-md">
                <p className="text-xs font-bold text-slate-900">Assessment Completion Notifications</p>
                <p className="text-[11px] text-slate-500">
                  Receive notification summaries whenever a 21-feature health assessment evaluation is processed.
                </p>
              </div>
              <label className="relative inline-flex items-center cursor-pointer flex-shrink-0">
                <input
                  type="checkbox"
                  checked={notifyAssessmentReady}
                  onChange={(e) => setNotifyAssessmentReady(e.target.checked)}
                  className="sr-only peer"
                />
                <div className="w-11 h-6 bg-slate-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
              </label>
            </div>

            {/* Security Alerts */}
            <div className="flex items-center justify-between p-4 rounded-2xl bg-slate-50 border border-slate-200/80">
              <div className="space-y-0.5 max-w-md">
                <p className="text-xs font-bold text-slate-900">Security &amp; Account Alerts</p>
                <p className="text-[11px] text-slate-500">
                  Critical security notifications regarding password changes, credential updates, or session activities.
                </p>
              </div>
              <label className="relative inline-flex items-center cursor-pointer flex-shrink-0">
                <input
                  type="checkbox"
                  checked={notifySecurityAlerts}
                  onChange={(e) => setNotifySecurityAlerts(e.target.checked)}
                  className="sr-only peer"
                />
                <div className="w-11 h-6 bg-slate-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
              </label>
            </div>

            {/* Product & Research Updates */}
            <div className="flex items-center justify-between p-4 rounded-2xl bg-slate-50 border border-slate-200/80">
              <div className="space-y-0.5 max-w-md">
                <p className="text-xs font-bold text-slate-900">Methodology &amp; Model Version Updates</p>
                <p className="text-[11px] text-slate-500">
                  Quarterly announcements regarding machine learning model releases and CDC BRFSS research findings.
                </p>
              </div>
              <label className="relative inline-flex items-center cursor-pointer flex-shrink-0">
                <input
                  type="checkbox"
                  checked={notifyProductUpdates}
                  onChange={(e) => setNotifyProductUpdates(e.target.checked)}
                  className="sr-only peer"
                />
                <div className="w-11 h-6 bg-slate-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
              </label>
            </div>
          </div>

          <div className="pt-2 flex items-center justify-between flex-wrap gap-4 border-t border-slate-100">
            <p className="text-[11px] text-slate-400">
              * Preferences are stored securely. External email dispatch is configured in production deployment.
            </p>
            <button
              type="button"
              onClick={handleSaveNotifications}
              disabled={settingsSaving}
              className="px-6 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-700 text-white text-xs font-bold flex items-center gap-2 transition-all shadow-md shadow-blue-500/20 disabled:opacity-50"
            >
              {settingsSaving ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Saving Preferences...</span>
                </>
              ) : (
                'Save Preferences'
              )}
            </button>
          </div>
        </div>
      )}

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* TAB 3: APPEARANCE PREFERENCES                                       */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      {activeTab === 'appearance' && (
        <div className="bg-white rounded-3xl border border-slate-200/90 shadow-sm p-6 sm:p-8 space-y-6">
          <div className="pb-4 border-b border-slate-100">
            <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
              <Sun className="w-5 h-5 text-amber-500" /> Interface Appearance Theme
            </h2>
            <p className="text-xs text-slate-500 mt-1">
              Select your preferred visual style. Setting is persisted to your account and applied instantly.
            </p>
          </div>

          <div className="grid sm:grid-cols-3 gap-4">
            {/* Light Mode Option */}
            <button
              type="button"
              onClick={() => handleThemeChange('light')}
              className={`p-5 rounded-2xl border text-left transition-all ${
                themePreference === 'light'
                  ? 'border-blue-600 ring-2 ring-blue-500/20 bg-blue-50/40'
                  : 'border-slate-200 hover:border-slate-300 bg-white'
              }`}
            >
              <div className="w-10 h-10 rounded-xl bg-amber-50 text-amber-600 flex items-center justify-center mb-3">
                <Sun className="w-5 h-5" />
              </div>
              <p className="text-xs font-bold text-slate-900">Light Theme</p>
              <p className="text-[11px] text-slate-500 mt-1">
                High-contrast clean clinical white interface with standard typography.
              </p>
              {themePreference === 'light' && (
                <span className="mt-3 inline-block text-[10px] font-bold text-blue-700 bg-blue-100 px-2 py-0.5 rounded-full">
                  Active
                </span>
              )}
            </button>

            {/* Dark Mode Option */}
            <button
              type="button"
              onClick={() => handleThemeChange('dark')}
              className={`p-5 rounded-2xl border text-left transition-all ${
                themePreference === 'dark'
                  ? 'border-blue-600 ring-2 ring-blue-500/20 bg-blue-50/40'
                  : 'border-slate-200 hover:border-slate-300 bg-white'
              }`}
            >
              <div className="w-10 h-10 rounded-xl bg-slate-900 text-slate-100 flex items-center justify-center mb-3">
                <Moon className="w-5 h-5" />
              </div>
              <p className="text-xs font-bold text-slate-900">Dark Theme</p>
              <p className="text-[11px] text-slate-500 mt-1">
                Low-glare dark slate background optimized for clinical and night environments.
              </p>
              {themePreference === 'dark' && (
                <span className="mt-3 inline-block text-[10px] font-bold text-blue-700 bg-blue-100 px-2 py-0.5 rounded-full">
                  Active
                </span>
              )}
            </button>

            {/* System Default Option */}
            <button
              type="button"
              onClick={() => handleThemeChange('system')}
              className={`p-5 rounded-2xl border text-left transition-all ${
                themePreference === 'system'
                  ? 'border-blue-600 ring-2 ring-blue-500/20 bg-blue-50/40'
                  : 'border-slate-200 hover:border-slate-300 bg-white'
              }`}
            >
              <div className="w-10 h-10 rounded-xl bg-teal-50 text-teal-600 flex items-center justify-center mb-3">
                <Laptop className="w-5 h-5" />
              </div>
              <p className="text-xs font-bold text-slate-900">System Default</p>
              <p className="text-[11px] text-slate-500 mt-1">
                Synchronizes automatically with your operating system light / dark preferences.
              </p>
              {themePreference === 'system' && (
                <span className="mt-3 inline-block text-[10px] font-bold text-blue-700 bg-blue-100 px-2 py-0.5 rounded-full">
                  Active
                </span>
              )}
            </button>
          </div>
        </div>
      )}

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* TAB 4: SESSION MANAGEMENT                                           */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      {activeTab === 'session' && (
        <div className="space-y-6">
          <div className="bg-white rounded-3xl border border-slate-200/90 shadow-sm p-6 sm:p-8 space-y-6">
            <div className="pb-4 border-b border-slate-100">
              <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
                <Shield className="w-5 h-5 text-indigo-600" /> Active Session &amp; Security Architecture
              </h2>
              <p className="text-xs text-slate-500 mt-1">
                Cryptographic parameters and authentication state governing your active connection.
              </p>
            </div>

            <div className="grid sm:grid-cols-2 gap-4 text-xs">
              <div className="p-4 rounded-2xl bg-slate-50 border border-slate-200/80 flex items-start gap-3">
                <Cpu className="w-4 h-4 text-indigo-600 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="font-bold text-slate-900">JWT Bearer Token (HS256)</p>
                  <p className="text-slate-500 text-[11px] mt-0.5">
                    Stateless token cryptographically signed with server HMAC secret. Verified on every protected endpoint.
                  </p>
                </div>
              </div>

              <div className="p-4 rounded-2xl bg-slate-50 border border-slate-200/80 flex items-start gap-3">
                <Server className="w-4 h-4 text-teal-600 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="font-bold text-slate-900">Session Validity Duration</p>
                  <p className="text-slate-500 text-[11px] mt-0.5">
                    Configured for 1,440 minutes (24 hours). Token automatically expires if left unused.
                  </p>
                </div>
              </div>
            </div>

            <div className="p-4 rounded-2xl bg-blue-50/60 border border-blue-200 text-xs text-blue-900 flex items-start gap-3">
              <Info className="w-4 h-4 text-blue-600 flex-shrink-0 mt-0.5" />
              <p className="leading-relaxed">
                <strong>Session Revocation Architecture:</strong> DiabetesRisk AI utilizes stateless JWTs. Signing out clears
                your credentials from browser storage. Immediate cross-device token revocation requires distributed token
                blacklisting (Redis cache), which is documented for future multi-tenant deployment.
              </p>
            </div>

            <div className="pt-2 flex items-center justify-between border-t border-slate-100">
              <div>
                <p className="text-xs font-bold text-slate-900">Revoke Session</p>
                <p className="text-[11px] text-slate-500">Sign out and purge local authorization tokens.</p>
              </div>
              <button
                type="button"
                onClick={handleLogout}
                className="px-5 py-2.5 rounded-xl border border-slate-300 hover:bg-slate-50 font-bold text-xs text-slate-700 flex items-center gap-1.5 transition-colors"
              >
                <LogOut className="w-3.5 h-3.5" />
                <span>Sign Out Now</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* TAB 5: DANGER ZONE (ACCOUNT DELETION)                               */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      {activeTab === 'danger' && (
        <div className="bg-white rounded-3xl border border-rose-300 shadow-sm p-6 sm:p-8 space-y-6">
          <div className="pb-4 border-b border-rose-100">
            <h2 className="text-base font-bold text-rose-800 flex items-center gap-2">
              <ShieldAlert className="w-5 h-5 text-rose-600" /> Danger Zone: Permanent Account Deletion
            </h2>
            <p className="text-xs text-slate-500 mt-1">
              Actions in this section are permanent and irreversible. Please proceed with caution.
            </p>
          </div>

          <div className="p-5 rounded-2xl bg-rose-50/80 border border-rose-200 space-y-3">
            <h3 className="text-xs font-bold text-rose-900 flex items-center gap-1.5">
              <AlertTriangle className="w-4 h-4 text-rose-600" />
              <span>Permanent Erasure of Personal Health Records</span>
            </h3>
            <p className="text-xs text-rose-800 leading-relaxed">
              Deleting your account permanently destroys your user profile, authentication credentials, and all
              associated health assessments and model prediction records stored in PostgreSQL.
            </p>
            <ul className="text-[11px] text-rose-700 list-disc list-inside space-y-1">
              <li>All 21 CDC health indicator entries will be permanently removed.</li>
              <li>Generated prediction results and probability records will be cascade deleted.</li>
              <li>You will immediately be logged out and cannot recover this data.</li>
            </ul>
          </div>

          <div className="flex items-center justify-between pt-2">
            <div>
              <p className="text-xs font-bold text-slate-900">Delete My Account</p>
              <p className="text-[11px] text-slate-400">Requires password verification and explicit typed confirmation.</p>
            </div>
            <button
              type="button"
              onClick={() => {
                setShowDeleteModal(true)
                setDeleteError('')
                setDeletePassword('')
                setDeleteConfirmation('')
              }}
              className="px-5 py-2.5 rounded-xl bg-rose-600 hover:bg-rose-700 text-white font-bold text-xs flex items-center gap-1.5 transition-colors shadow-sm shadow-rose-500/20"
            >
              <Trash2 className="w-3.5 h-3.5" />
              <span>Delete Account</span>
            </button>
          </div>
        </div>
      )}

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* ACCOUNT DELETION CONFIRMATION MODAL                                 */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      {showDeleteModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-xs animate-fadeIn">
          <div className="bg-white rounded-3xl border border-slate-200 shadow-2xl max-w-md w-full p-6 sm:p-8 space-y-5 animate-scaleUp">
            <div className="flex items-center justify-between pb-3 border-b border-slate-100">
              <div className="flex items-center gap-2 text-rose-700">
                <AlertTriangle className="w-5 h-5 text-rose-600" />
                <h3 className="font-extrabold text-base">Confirm Account Deletion</h3>
              </div>
              <button
                type="button"
                onClick={() => setShowDeleteModal(false)}
                className="text-slate-400 hover:text-slate-600"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <p className="text-xs text-slate-600 leading-relaxed">
              This action is <strong className="text-rose-700 font-bold">strictly permanent and irreversible</strong>.
              All your saved diabetes assessments, prediction probabilities, and personal account data in PostgreSQL will be purged immediately.
            </p>

            {deleteError && (
              <div className="p-3 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-center gap-2">
                <AlertCircle className="w-4 h-4 text-rose-600 flex-shrink-0" />
                <span>{deleteError}</span>
              </div>
            )}

            <form onSubmit={handleDeleteAccountSubmit} className="space-y-4">
              {/* Password Re-authentication */}
              <div>
                <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1">
                  Re-enter Password <span className="text-red-500">*</span>
                </label>
                <input
                  type="password"
                  value={deletePassword}
                  onChange={(e) => setDeletePassword(e.target.value)}
                  placeholder="Your account password"
                  className="w-full px-3.5 py-2.5 rounded-xl border border-slate-300 focus:outline-none focus:ring-2 focus:ring-rose-500 text-sm font-semibold text-slate-800"
                  autoFocus
                />
              </div>

              {/* Explicit Confirmation Token */}
              <div>
                <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1">
                  Type <span className="text-rose-700 font-mono font-bold">DELETE</span> to confirm <span className="text-red-500">*</span>
                </label>
                <input
                  type="text"
                  value={deleteConfirmation}
                  onChange={(e) => setDeleteConfirmation(e.target.value)}
                  placeholder="Type DELETE"
                  className="w-full px-3.5 py-2.5 rounded-xl border border-slate-300 focus:outline-none focus:ring-2 focus:ring-rose-500 text-sm font-semibold text-slate-800 uppercase"
                />
              </div>

              <div className="flex items-center justify-end gap-3 pt-3 border-t border-slate-100">
                <button
                  type="button"
                  onClick={() => setShowDeleteModal(false)}
                  disabled={deleteLoading}
                  className="px-4 py-2.5 rounded-xl border border-slate-200 text-slate-700 hover:bg-slate-50 text-xs font-semibold transition-colors"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={
                    deleteLoading ||
                    !deletePassword ||
                    deleteConfirmation.trim().toUpperCase() !== 'DELETE'
                  }
                  className="px-5 py-2.5 rounded-xl bg-rose-600 hover:bg-rose-700 text-white font-bold text-xs flex items-center gap-2 transition-all shadow-md shadow-rose-500/20 disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  {deleteLoading ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span>Deleting Account...</span>
                    </>
                  ) : (
                    <>
                      <Trash2 className="w-4 h-4" />
                      <span>Permanently Delete</span>
                    </>
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  )
}
