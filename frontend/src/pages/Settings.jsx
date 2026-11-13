/**
 * src/pages/Settings.jsx
 * =======================
 * Account & Security Settings Page for DiabetesRisk AI.
 * 
 * Sections:
 * 1. Account Information
 * 2. Change Password (Current Password verification + Bcrypt hashing via POST /api/auth/change-password)
 * 3. Session & Cryptographic Security Info
 * 4. Sign Out / Session Revocation
 */

import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Settings,
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
} from 'lucide-react'
import { useAuth } from '../context/AuthContext'
import { changeUserPassword } from '../services/auth'

export default function SettingsPage() {
  const navigate = useNavigate()
  const { user, logout } = useAuth()

  // Change Password state
  const [currentPassword, setCurrentPassword] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [showCurrentPassword, setShowCurrentPassword] = useState(false)
  const [showNewPassword, setShowNewPassword] = useState(false)
  const [loadingPassword, setLoadingPassword] = useState(false)
  const [passwordSuccess, setPasswordSuccess] = useState('')
  const [passwordError, setPasswordError] = useState('')

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
    if (newPassword !== confirmPassword) {
      setPasswordError('New password and confirmation password do not match.')
      return
    }

    setLoadingPassword(true)
    try {
      const res = await changeUserPassword({
        current_password: currentPassword,
        new_password: newPassword,
      })
      setPasswordSuccess(res.message || 'Password updated successfully.')
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

  const handleLogout = () => {
    logout()
    navigate('/login')
  }

  return (
    <div className="p-4 sm:p-6 lg:p-8 max-w-4xl mx-auto space-y-8">
      {/* Page Title */}
      <div>
        <h1 className="text-2xl font-extrabold text-slate-900 tracking-tight">Account &amp; Security Settings</h1>
        <p className="text-xs sm:text-sm text-slate-500 mt-1">
          Configure security credentials, manage password authentication, and inspect session encryption details.
        </p>
      </div>

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* 1. Account Summary Card                                             */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      <div className="bg-white rounded-3xl border border-slate-200/90 shadow-sm p-6 sm:p-8">
        <h2 className="text-base font-bold text-slate-900 flex items-center gap-2 mb-4 pb-3 border-b border-slate-100">
          <Settings className="w-5 h-5 text-blue-600" /> Account Overview
        </h2>

        <div className="grid sm:grid-cols-3 gap-4 text-xs">
          <div className="p-4 rounded-2xl bg-slate-50 border border-slate-200">
            <span className="text-slate-400 uppercase font-semibold">User ID</span>
            <p className="text-base font-extrabold text-slate-900 font-mono mt-1">#{user?.id || '—'}</p>
          </div>
          <div className="p-4 rounded-2xl bg-slate-50 border border-slate-200">
            <span className="text-slate-400 uppercase font-semibold">Registered Email</span>
            <p className="text-sm font-bold text-slate-900 truncate mt-1">{user?.email || '—'}</p>
          </div>
          <div className="p-4 rounded-2xl bg-slate-50 border border-slate-200">
            <span className="text-slate-400 uppercase font-semibold">Security Role</span>
            <p className="text-sm font-bold text-teal-700 mt-1">Authenticated Assessee</p>
          </div>
        </div>
      </div>

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* 2. Change Password Card                                             */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      <div className="bg-white rounded-3xl border border-slate-200/90 shadow-sm p-6 sm:p-8">
        <h2 className="text-base font-bold text-slate-900 flex items-center gap-2 mb-2">
          <KeyRound className="w-5 h-5 text-teal-600" /> Change Account Password
        </h2>
        <p className="text-xs text-slate-500 mb-6">
          Passwords are salted and cryptographically hashed with 12-round Bcrypt before persisting to PostgreSQL.
        </p>

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

        <form onSubmit={handlePasswordSubmit} className="space-y-4 max-w-lg">
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
                placeholder="••••••••"
                className="w-full pl-10 pr-10 py-2.5 rounded-xl border border-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
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
                className="w-full pl-10 pr-10 py-2.5 rounded-xl border border-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
              />
              <button
                type="button"
                onClick={() => setShowNewPassword(!showNewPassword)}
                className="absolute right-3.5 top-3 text-slate-400 hover:text-slate-600"
              >
                {showNewPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
            <p className="text-[11px] text-slate-400 mt-1">Minimum 8 characters required.</p>
          </div>

          {/* Confirm New Password */}
          <div>
            <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5">
              Confirm New Password <span className="text-red-500">*</span>
            </label>
            <div className="relative">
              <Lock className="absolute left-3.5 top-3 w-4 h-4 text-slate-400" />
              <input
                type={showNewPassword ? 'text' : 'password'}
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                placeholder="Confirm your new password"
                className="w-full pl-10 pr-4 py-2.5 rounded-xl border border-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
              />
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
                  Updating Password...
                </>
              ) : (
                'Update Password'
              )}
            </button>
          </div>
        </form>
      </div>

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* 3. Session & Security Information Card                             */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      <div className="bg-white rounded-3xl border border-slate-200/90 shadow-sm p-6 sm:p-8 space-y-4">
        <h2 className="text-base font-bold text-slate-900 flex items-center gap-2 pb-3 border-b border-slate-100">
          <Shield className="w-5 h-5 text-indigo-600" /> Session &amp; Cryptographic Architecture
        </h2>

        <div className="grid sm:grid-cols-2 gap-4 text-xs">
          <div className="p-4 rounded-2xl bg-slate-50 border border-slate-200 flex items-start gap-3">
            <Cpu className="w-4 h-4 text-indigo-600 flex-shrink-0 mt-0.5" />
            <div>
              <p className="font-bold text-slate-900">JWT Bearer Token (HS256)</p>
              <p className="text-slate-500 text-[11px] mt-0.5">
                Stateless authorization signed with environment secret key. Verified on every protected API call.
              </p>
            </div>
          </div>

          <div className="p-4 rounded-2xl bg-slate-50 border border-slate-200 flex items-start gap-3">
            <Server className="w-4 h-4 text-teal-600 flex-shrink-0 mt-0.5" />
            <div>
              <p className="font-bold text-slate-900">Session Validity Duration</p>
              <p className="text-slate-500 text-[11px] mt-0.5">
                Configured for 1,440 minutes (24 hours). Automatically expires if unused.
              </p>
            </div>
          </div>
        </div>

        <div className="p-4 rounded-2xl bg-blue-50/60 border border-blue-200 text-xs text-blue-900 flex items-start gap-3">
          <Info className="w-4 h-4 text-blue-600 flex-shrink-0 mt-0.5" />
          <p>
            Plaintext passwords and authentication secrets are strictly prohibited from logs, database persistence,
            and API responses. All database connections use connection pooling with automatic session rollback on exceptions.
          </p>
        </div>
      </div>

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* 4. Session Revocation / Log Out                                     */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      <div className="bg-white rounded-3xl border border-rose-200 shadow-sm p-6 sm:p-8">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
              <LogOut className="w-5 h-5 text-rose-500" /> Revoke Session &amp; Sign Out
            </h3>
            <p className="text-xs text-slate-500 mt-1 max-w-md">
              Clears your stored JWT token from browser storage and closes the authenticated session.
            </p>
          </div>
          <button
            type="button"
            onClick={handleLogout}
            className="px-5 py-2.5 rounded-xl border border-rose-200 text-rose-600 hover:bg-rose-50 font-bold text-xs transition-colors self-start sm:self-center shadow-xs"
          >
            Sign Out Now
          </button>
        </div>
      </div>
    </div>
  )
}

