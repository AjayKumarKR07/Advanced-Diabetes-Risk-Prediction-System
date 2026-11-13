/**
 * src/pages/Profile.jsx
 * =====================
 * User Profile Management Page for DiabetesRisk AI.
 * 
 * Features:
 * - View & update full name in PostgreSQL via PUT /api/auth/profile
 * - Read-only verified email display with security explanation
 * - Form validation (min 2 characters, sanitization)
 * - Real-time feedback banners (success, error, loading)
 * - Account baseline summary metrics
 */

import { useState } from 'react'
import {
  User,
  Mail,
  Calendar,
  ShieldCheck,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Edit2,
  X,
  Save,
  KeyRound,
} from 'lucide-react'
import { Link } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { updateUserProfile } from '../services/auth'

export default function Profile() {
  const { user, updateUser } = useAuth()

  const [isEditing, setIsEditing] = useState(false)
  const [fullName, setFullName] = useState(user?.full_name || '')
  const [loading, setLoading] = useState(false)
  const [successMessage, setSuccessMessage] = useState('')
  const [errorMessage, setErrorMessage] = useState('')

  const handleStartEdit = () => {
    setFullName(user?.full_name || '')
    setSuccessMessage('')
    setErrorMessage('')
    setIsEditing(true)
  }

  const handleCancel = () => {
    setFullName(user?.full_name || '')
    setErrorMessage('')
    setIsEditing(false)
  }

  const handleSave = async (e) => {
    e.preventDefault()
    setErrorMessage('')
    setSuccessMessage('')

    const trimmed = fullName.trim()
    if (trimmed.length < 2) {
      setErrorMessage('Full name must be at least 2 characters.')
      return
    }

    setLoading(true)
    try {
      const updated = await updateUserProfile({ full_name: trimmed })
      updateUser(updated)
      setSuccessMessage('Profile updated successfully.')
      setIsEditing(false)
    } catch (err) {
      setErrorMessage(
        err.response?.data?.detail ||
          'Failed to update profile. Please try again.'
      )
    } finally {
      setLoading(false)
    }
  }

  const memberSince = user?.created_at
    ? new Date(user.created_at).toLocaleDateString('en-US', {
        month: 'long',
        day: 'numeric',
        year: 'numeric',
      })
    : 'Active Session'

  return (
    <div className="p-4 sm:p-6 lg:p-8 max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-extrabold text-slate-900 tracking-tight">Account Profile</h1>
        <p className="text-xs sm:text-sm text-slate-500 mt-1">
          Manage your personal information, contact credentials, and account settings.
        </p>
      </div>

      {/* Success / Error Banners */}
      {successMessage && (
        <div className="p-4 rounded-2xl bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs font-semibold flex items-center gap-3">
          <CheckCircle2 className="w-5 h-5 text-emerald-600 flex-shrink-0" />
          <span>{successMessage}</span>
        </div>
      )}

      {errorMessage && (
        <div className="p-4 rounded-2xl bg-rose-50 border border-rose-200 text-rose-800 text-xs font-semibold flex items-center gap-3">
          <AlertCircle className="w-5 h-5 text-rose-600 flex-shrink-0" />
          <span>{errorMessage}</span>
        </div>
      )}

      {/* Profile Overview Card */}
      <div className="bg-white rounded-3xl border border-slate-200/90 shadow-sm p-6 sm:p-8">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-6 border-b border-slate-100 gap-4">
          <div className="flex items-center gap-4">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-tr from-teal-600 to-blue-600 text-white font-extrabold text-xl flex items-center justify-center shadow-md shadow-blue-500/20">
              {user?.full_name
                ? user.full_name
                    .split(' ')
                    .map((n) => n[0])
                    .join('')
                    .toUpperCase()
                    .slice(0, 2)
                : 'U'}
            </div>
            <div>
              <h2 className="text-lg font-bold text-slate-900">{user?.full_name}</h2>
              <p className="text-xs text-slate-500">{user?.email}</p>
              <div className="mt-1 flex items-center gap-2">
                <span className="inline-flex items-center gap-1 text-[10px] font-bold text-teal-700 bg-teal-50 border border-teal-200 px-2 py-0.5 rounded-full">
                  <ShieldCheck className="w-3 h-3 text-teal-600" /> Verified Account
                </span>
                <span className="text-[11px] text-slate-400">User ID #{user?.id}</span>
              </div>
            </div>
          </div>

          {!isEditing && (
            <button
              type="button"
              onClick={handleStartEdit}
              className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl border border-slate-200 hover:bg-slate-50 text-xs font-bold text-slate-700 transition-colors shadow-xs"
            >
              <Edit2 className="w-3.5 h-3.5 text-blue-600" />
              <span>Edit Profile</span>
            </button>
          )}
        </div>

        {/* Profile Details Form */}
        <form onSubmit={handleSave} className="pt-6 space-y-6">
          <div className="grid sm:grid-cols-2 gap-6">
            {/* Full Name */}
            <div>
              <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-2">
                Full Name
              </label>
              {isEditing ? (
                <div className="relative">
                  <User className="absolute left-3.5 top-3 w-4 h-4 text-slate-400" />
                  <input
                    type="text"
                    value={fullName}
                    onChange={(e) => setFullName(e.target.value)}
                    placeholder="Enter your full name"
                    className="w-full pl-10 pr-4 py-2.5 rounded-xl border border-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm font-semibold text-slate-800"
                    autoFocus
                  />
                </div>
              ) : (
                <div className="flex items-center gap-2 p-3 rounded-xl bg-slate-50 border border-slate-200/80 text-sm font-semibold text-slate-800">
                  <User className="w-4 h-4 text-slate-400" />
                  <span>{user?.full_name}</span>
                </div>
              )}
              <p className="text-[11px] text-slate-400 mt-1">
                Used in your personalized risk evaluations and assessment reports.
              </p>
            </div>

            {/* Email Address (Read-only) */}
            <div>
              <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-2">
                Registered Email Address
              </label>
              <div className="flex items-center gap-2 p-3 rounded-xl bg-slate-50 border border-slate-200/80 text-sm font-medium text-slate-600">
                <Mail className="w-4 h-4 text-slate-400" />
                <span className="truncate">{user?.email}</span>
              </div>
              <p className="text-[11px] text-slate-400 mt-1">
                Email serves as your primary sign-in credential and cannot be changed without re-verification.
              </p>
            </div>

            {/* Account Creation Date */}
            <div>
              <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-2">
                Account Member Since
              </label>
              <div className="flex items-center gap-2 p-3 rounded-xl bg-slate-50 border border-slate-200/80 text-sm font-medium text-slate-600">
                <Calendar className="w-4 h-4 text-slate-400" />
                <span>{memberSince}</span>
              </div>
            </div>

            {/* Security Profile Link */}
            <div>
              <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-2">
                Password &amp; Security
              </label>
              <div className="p-3 rounded-xl bg-slate-50 border border-slate-200/80 flex items-center justify-between text-xs">
                <div className="flex items-center gap-2 text-slate-700">
                  <KeyRound className="w-4 h-4 text-blue-600" />
                  <span>Password Hashed with 12-round Bcrypt</span>
                </div>
                <Link to="/settings" className="font-bold text-blue-600 hover:underline">
                  Manage →
                </Link>
              </div>
            </div>
          </div>

          {/* Action buttons during editing */}
          {isEditing && (
            <div className="flex items-center gap-3 pt-4 border-t border-slate-100">
              <button
                type="submit"
                disabled={loading}
                className="px-6 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-700 text-white text-xs font-bold flex items-center gap-2 transition-all shadow-md shadow-blue-500/20 disabled:opacity-50"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Saving...
                  </>
                ) : (
                  <>
                    <Save className="w-4 h-4" />
                    Save Changes
                  </>
                )}
              </button>
              <button
                type="button"
                onClick={handleCancel}
                disabled={loading}
                className="px-4 py-2.5 rounded-xl border border-slate-200 text-slate-700 hover:bg-slate-50 text-xs font-semibold flex items-center gap-1.5 transition-colors"
              >
                <X className="w-4 h-4" />
                Cancel
              </button>
            </div>
          )}
        </form>
      </div>

      {/* Account Security Information Card */}
      <div className="bg-white rounded-3xl border border-slate-200/90 shadow-sm p-6 sm:p-8 space-y-4">
        <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
          <ShieldCheck className="w-5 h-5 text-teal-600" /> Privacy &amp; Data Governance
        </h3>
        <p className="text-xs text-slate-600 leading-relaxed">
          Your personal health assessment records are strictly tied to your authenticated User ID in PostgreSQL.
          No healthcare responses or predicted risk scores are shared across accounts, indexed by search engines,
          or utilized for commercial advertising.
        </p>
      </div>
    </div>
  )
}

