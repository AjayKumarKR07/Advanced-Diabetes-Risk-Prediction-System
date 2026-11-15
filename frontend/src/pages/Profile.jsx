/**
 * src/pages/Profile.jsx
 * =====================
 * Professional User Profile Management Page for DiabetesRisk AI (Phase 6.14).
 * 
 * Features:
 * - Profile Overview: Dynamic avatar with initials fallback, user badges, email status.
 * - Personal Information: Full name editing, real-time validation, read-only email.
 * - Profile Picture: Client-side validated upload, preview, replace, remove, and size/format enforcement.
 * - Real Account Activity: Total completed assessments and latest evaluation date from PostgreSQL.
 * - AuthContext synchronization: updates immediately propagate to header, sidebar, and dashboard.
 */

import { useEffect, useRef, useState } from 'react'
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
  Camera,
  Trash2,
  Upload,
  Activity,
  Clock,
  ArrowRight,
  Info,
  Sparkles,
} from 'lucide-react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import {
  deleteUserAvatar,
  fetchUserProfile,
  updateUserProfile,
  uploadUserAvatar,
} from '../services/auth'

const MAX_AVATAR_SIZE_BYTES = 2 * 1024 * 1024 // 2MB
const ALLOWED_IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/webp']

export default function Profile() {
  const navigate = useNavigate()
  const { user, updateUser } = useAuth()
  const fileInputRef = useRef(null)

  // Enriched profile & activity state
  const [profileData, setProfileData] = useState(null)
  const [loadingProfile, setLoadingProfile] = useState(true)

  // Edit personal info state
  const [isEditing, setIsEditing] = useState(false)
  const [fullName, setFullName] = useState(user?.full_name || '')
  const [nameError, setNameError] = useState('')
  const [savingProfile, setSavingProfile] = useState(false)
  const [successBanner, setSuccessBanner] = useState('')
  const [errorBanner, setErrorBanner] = useState('')

  // Avatar state
  const avatarStorageKey = user?.id ? `diabetes_ai_avatar_${user.id}` : null
  const [avatarUrl, setAvatarUrl] = useState(() => {
    return user?.has_avatar ? '/api/profile/avatar' : (avatarStorageKey ? localStorage.getItem(avatarStorageKey) : null)
  })
  const [avatarPreview, setAvatarPreview] = useState(null)
  const [selectedFile, setSelectedFile] = useState(null)
  const [avatarError, setAvatarError] = useState('')
  const [uploadingAvatar, setUploadingAvatar] = useState(false)

  // Load enriched profile data from GET /api/profile
  useEffect(() => {
    let isMounted = true
    setLoadingProfile(true)

    fetchUserProfile()
      .then((data) => {
        if (isMounted) {
          setProfileData(data)
          setFullName(data.full_name)
          if (data.has_avatar) {
            setAvatarUrl(`/api/profile/avatar?t=${Date.now()}`)
          }
          updateUser(data)
        }
      })
      .catch((err) => {
        if (isMounted) {
          setErrorBanner(
            err.response?.data?.detail || 'Could not load complete profile statistics.'
          )
        }
      })
      .finally(() => {
        if (isMounted) setLoadingProfile(false)
      })

    return () => {
      isMounted = false
    }
  }, [])

  // Sync avatar if user id changes
  useEffect(() => {
    if (avatarStorageKey) {
      setAvatarUrl(localStorage.getItem(avatarStorageKey))
    }
  }, [avatarStorageKey])

  // Handle Edit Mode Toggle
  const handleStartEdit = () => {
    setFullName(profileData?.full_name || user?.full_name || '')
    setNameError('')
    setSuccessBanner('')
    setErrorBanner('')
    setIsEditing(true)
  }

  const handleCancelEdit = () => {
    setFullName(profileData?.full_name || user?.full_name || '')
    setNameError('')
    setIsEditing(false)
  }

  // Handle Full Name Save
  const handleSaveProfile = async (e) => {
    e.preventDefault()
    setNameError('')
    setSuccessBanner('')
    setErrorBanner('')

    const trimmed = fullName.trim()
    if (trimmed.length < 2) {
      setNameError('Full name must be at least 2 characters.')
      return
    }

    setSavingProfile(true)
    try {
      const updated = await updateUserProfile({ full_name: trimmed })
      updateUser(updated)
      setProfileData((prev) => ({ ...prev, ...updated }))
      setSuccessBanner('Your profile name has been updated and persisted to PostgreSQL.')
      setIsEditing(false)
    } catch (err) {
      setErrorBanner(
        err.response?.data?.detail || 'Failed to update profile. Please try again.'
      )
    } finally {
      setSavingProfile(false)
    }
  }

  // Handle Avatar Selection & Validation
  const handleFileChange = (e) => {
    setAvatarError('')
    const file = e.target.files?.[0]
    if (!file) return

    if (!ALLOWED_IMAGE_TYPES.includes(file.type)) {
      setAvatarError('Unsupported file type. Please upload a JPEG, PNG, or WebP image.')
      return
    }

    if (file.size > MAX_AVATAR_SIZE_BYTES) {
      setAvatarError('Image is too large. Maximum allowed size is 2MB.')
      return
    }

    setSelectedFile(file)
    const reader = new FileReader()
    reader.onload = () => {
      setAvatarPreview(reader.result)
    }
    reader.readAsDataURL(file)
  }

  // Save Selected Avatar to Server Storage
  const handleSaveAvatar = async () => {
    if (!selectedFile) return
    setUploadingAvatar(true)
    setAvatarError('')

    try {
      await uploadUserAvatar(selectedFile)
      const freshUrl = `/api/profile/avatar?t=${Date.now()}`
      setAvatarUrl(freshUrl)
      if (avatarStorageKey) {
        localStorage.setItem(avatarStorageKey, freshUrl)
      }
      updateUser({ ...user, has_avatar: true, avatar_url: '/api/profile/avatar' })
      setAvatarPreview(null)
      setSelectedFile(null)
      setSuccessBanner('Profile picture saved and persisted to server storage.')
    } catch (err) {
      setAvatarError(
        err.response?.data?.detail || 'Failed to upload avatar to server storage.'
      )
    } finally {
      setUploadingAvatar(false)
    }
  }

  const handleCancelAvatarPreview = () => {
    setAvatarPreview(null)
    setSelectedFile(null)
    setAvatarError('')
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  // Remove Avatar from Server Storage
  const handleRemoveAvatar = async () => {
    try {
      await deleteUserAvatar()
    } catch (err) {
      // Ignore if already deleted
    }

    if (avatarStorageKey) {
      localStorage.removeItem(avatarStorageKey)
    }
    setAvatarUrl(null)
    setAvatarPreview(null)
    setSelectedFile(null)
    setAvatarError('')
    if (fileInputRef.current) fileInputRef.current.value = ''
    updateUser({ ...user, has_avatar: false, avatar_url: null })
    setSuccessBanner('Profile picture removed from server. Initials avatar restored.')
  }

  // Computed initials
  const displayName = profileData?.full_name || user?.full_name || 'Patient'
  const initials = displayName
    .split(' ')
    .filter(Boolean)
    .map((n) => n[0])
    .join('')
    .toUpperCase()
    .slice(0, 2) || 'U'

  const memberSince = (profileData?.created_at || user?.created_at)
    ? new Date(profileData?.created_at || user?.created_at).toLocaleDateString('en-US', {
        month: 'long',
        day: 'numeric',
        year: 'numeric',
      })
    : 'Active Session'

  const latestAssessmentDate = profileData?.latest_assessment_date
    ? new Date(profileData.latest_assessment_date).toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      })
    : 'None recorded yet'

  return (
    <div className="p-4 sm:p-6 lg:p-8 max-w-4xl mx-auto space-y-6">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-extrabold text-slate-900 tracking-tight">Account Profile</h1>
          <p className="text-xs sm:text-sm text-slate-500 mt-1">
            Manage your personal clinical profile, avatar photo, and inspect verified account activity.
          </p>
        </div>
        <div className="flex items-center gap-2 self-start sm:self-auto">
          <Link
            to="/settings"
            className="px-4 py-2 rounded-xl border border-slate-200 hover:bg-slate-50 text-xs font-bold text-slate-700 flex items-center gap-1.5 transition-colors"
          >
            <KeyRound className="w-3.5 h-3.5 text-blue-600" />
            <span>Security Settings</span>
          </Link>
        </div>
      </div>

      {/* Success Notification Banner */}
      {successBanner && (
        <div className="p-4 rounded-2xl bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs font-semibold flex items-center justify-between gap-3 animate-fadeIn">
          <div className="flex items-center gap-2.5">
            <CheckCircle2 className="w-5 h-5 text-emerald-600 flex-shrink-0" />
            <span>{successBanner}</span>
          </div>
          <button
            onClick={() => setSuccessBanner('')}
            className="text-emerald-700 hover:text-emerald-900"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Error Notification Banner */}
      {errorBanner && (
        <div className="p-4 rounded-2xl bg-rose-50 border border-rose-200 text-rose-800 text-xs font-semibold flex items-center justify-between gap-3 animate-fadeIn">
          <div className="flex items-center gap-2.5">
            <AlertCircle className="w-5 h-5 text-rose-600 flex-shrink-0" />
            <span>{errorBanner}</span>
          </div>
          <button
            onClick={() => setErrorBanner('')}
            className="text-rose-700 hover:text-rose-900"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* 1. Profile Overview & Photo Management Card                         */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      <div className="bg-white rounded-3xl border border-slate-200/90 shadow-sm p-6 sm:p-8">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-6 pb-6 border-b border-slate-100">
          <div className="flex flex-col sm:flex-row items-center sm:items-start gap-5 text-center sm:text-left">
            {/* Avatar Display / Preview */}
            <div className="relative group">
              <div className="w-20 h-20 rounded-2xl overflow-hidden border-2 border-slate-200/80 shadow-md flex items-center justify-center bg-gradient-to-tr from-teal-600 to-blue-600 text-white font-extrabold text-2xl">
                {avatarPreview ? (
                  <img
                    src={avatarPreview}
                    alt="Preview Avatar"
                    className="w-full h-full object-cover"
                  />
                ) : avatarUrl ? (
                  <img
                    src={avatarUrl}
                    alt="User Avatar"
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <span>{initials}</span>
                )}
              </div>

              {/* Upload trigger button overlay */}
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="absolute -bottom-1 -right-1 p-1.5 rounded-xl bg-blue-600 text-white shadow-md hover:bg-blue-700 transition-colors"
                title="Change Avatar Image"
              >
                <Camera className="w-3.5 h-3.5" />
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/jpeg,image/png,image/webp"
                onChange={handleFileChange}
                className="hidden"
              />
            </div>

            {/* Profile Identity Details */}
            <div className="space-y-1">
              <div className="flex flex-wrap items-center justify-center sm:justify-start gap-2">
                <h2 className="text-xl font-bold text-slate-900">{displayName}</h2>
                <span className="inline-flex items-center gap-1 text-[11px] font-bold text-teal-700 bg-teal-50 border border-teal-200 px-2 py-0.5 rounded-full">
                  <ShieldCheck className="w-3 h-3 text-teal-600" />
                  {profileData?.account_status || 'Active Account'}
                </span>
              </div>
              <p className="text-xs text-slate-500">{user?.email}</p>
              <div className="pt-1 flex flex-wrap items-center justify-center sm:justify-start gap-x-3 gap-y-1 text-[11px] text-slate-400">
                <span>Account ID: <strong className="font-mono text-slate-600">#{user?.id}</strong></span>
                <span>•</span>
                <span>Member Since: <strong className="text-slate-600">{memberSince}</strong></span>
              </div>
            </div>
          </div>

          {/* Edit Profile Button */}
          {!isEditing && (
            <button
              type="button"
              onClick={handleStartEdit}
              className="inline-flex items-center justify-center gap-2 px-5 py-2.5 rounded-xl border border-slate-200 hover:bg-slate-50 text-xs font-bold text-slate-700 transition-colors shadow-xs self-center sm:self-auto"
            >
              <Edit2 className="w-3.5 h-3.5 text-blue-600" />
              <span>Edit Personal Info</span>
            </button>
          )}
        </div>

        {/* Avatar Actions & Validation Alerts */}
        {avatarPreview && (
          <div className="mt-4 p-4 rounded-2xl bg-blue-50/70 border border-blue-200 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-blue-900">
            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-blue-600 flex-shrink-0" />
              <span>New photo selected. Save changes to persist your new profile picture.</span>
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handleSaveAvatar}
                className="px-4 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-700 text-white font-bold text-xs shadow-xs"
              >
                Save Photo
              </button>
              <button
                type="button"
                onClick={handleCancelAvatarPreview}
                className="px-3 py-1.5 rounded-lg border border-slate-300 bg-white text-slate-700 font-semibold text-xs"
              >
                Cancel
              </button>
            </div>
          </div>
        )}

        {avatarError && (
          <div className="mt-4 p-3 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-rose-600 flex-shrink-0" />
            <span>{avatarError}</span>
          </div>
        )}

        {/* Avatar management helper row */}
        <div className="mt-4 pt-3 flex flex-wrap items-center justify-between gap-2 text-xs text-slate-500">
          <div className="flex items-center gap-2 text-[11px] text-slate-400">
            <Upload className="w-3.5 h-3.5" />
            <span>Supported formats: JPEG, PNG, WebP (Max 2MB). Sandboxed client storage.</span>
          </div>
          {avatarUrl && !avatarPreview && (
            <button
              type="button"
              onClick={handleRemoveAvatar}
              className="text-[11px] text-rose-600 hover:text-rose-700 font-semibold flex items-center gap-1"
            >
              <Trash2 className="w-3 h-3" />
              <span>Remove Photo</span>
            </button>
          )}
        </div>

        {/* ═════════════════════════════════════════════════════════════════ */}
        {/* 2. Personal Information Form                                      */}
        {/* ═════════════════════════════════════════════════════════════════ */}
        <form onSubmit={handleSaveProfile} className="pt-6 space-y-6">
          <div className="grid sm:grid-cols-2 gap-6">
            {/* Full Name */}
            <div>
              <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-2">
                Full Name <span className="text-red-500">*</span>
              </label>
              {isEditing ? (
                <div>
                  <div className="relative">
                    <User className="absolute left-3.5 top-3 w-4 h-4 text-slate-400" />
                    <input
                      type="text"
                      value={fullName}
                      onChange={(e) => {
                        setFullName(e.target.value)
                        if (nameError) setNameError('')
                      }}
                      placeholder="Enter your full name"
                      className={`w-full pl-10 pr-4 py-2.5 rounded-xl border ${
                        nameError ? 'border-rose-300 ring-1 ring-rose-300' : 'border-slate-300'
                      } focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm font-semibold text-slate-800`}
                      autoFocus
                    />
                  </div>
                  {nameError && (
                    <p className="text-xs text-rose-600 mt-1.5 flex items-center gap-1 font-medium">
                      <AlertCircle className="w-3.5 h-3.5" /> {nameError}
                    </p>
                  )}
                </div>
              ) : (
                <div className="flex items-center gap-2 p-3 rounded-xl bg-slate-50 border border-slate-200/80 text-sm font-semibold text-slate-800">
                  <User className="w-4 h-4 text-slate-400" />
                  <span>{displayName}</span>
                </div>
              )}
              <p className="text-[11px] text-slate-400 mt-1">
                Your full name appears on your official assessment records and PDF clinical reports.
              </p>
            </div>

            {/* Email Address (Read-only) */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider">
                  Registered Email
                </label>
                <span className="text-[10px] font-bold text-emerald-700 bg-emerald-50 border border-emerald-200 px-2 py-0.5 rounded-full">
                  Primary Sign-in
                </span>
              </div>
              <div className="flex items-center gap-2 p-3 rounded-xl bg-slate-50 border border-slate-200/80 text-sm font-medium text-slate-600">
                <Mail className="w-4 h-4 text-slate-400" />
                <span className="truncate">{user?.email}</span>
              </div>
              <p className="text-[11px] text-slate-400 mt-1">
                Email serves as your unique authentication identifier and cannot be modified directly.
              </p>
            </div>

            {/* Account Registration Date */}
            <div>
              <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-2">
                Account Registration Date
              </label>
              <div className="flex items-center gap-2 p-3 rounded-xl bg-slate-50 border border-slate-200/80 text-sm font-medium text-slate-600">
                <Calendar className="w-4 h-4 text-slate-400" />
                <span>{memberSince}</span>
              </div>
              <p className="text-[11px] text-slate-400 mt-1">Timestamp recorded in PostgreSQL upon initial registration.</p>
            </div>

            {/* Credential Status */}
            <div>
              <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-2">
                Security &amp; Password Policy
              </label>
              <div className="p-3 rounded-xl bg-slate-50 border border-slate-200/80 flex items-center justify-between text-xs">
                <div className="flex items-center gap-2 text-slate-700">
                  <KeyRound className="w-4 h-4 text-teal-600" />
                  <span className="font-medium">12-Round Salted Bcrypt</span>
                </div>
                <Link to="/settings" className="font-bold text-blue-600 hover:text-blue-800">
                  Change Password →
                </Link>
              </div>
              <p className="text-[11px] text-slate-400 mt-1">Passwords are salted and cryptographically hashed.</p>
            </div>
          </div>

          {/* Edit Mode Buttons */}
          {isEditing && (
            <div className="flex items-center gap-3 pt-4 border-t border-slate-100">
              <button
                type="submit"
                disabled={savingProfile}
                className="px-6 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-700 text-white text-xs font-bold flex items-center gap-2 transition-all shadow-md shadow-blue-500/20 disabled:opacity-50"
              >
                {savingProfile ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Saving Changes...</span>
                  </>
                ) : (
                  <>
                    <Save className="w-4 h-4" />
                    <span>Save Changes</span>
                  </>
                )}
              </button>
              <button
                type="button"
                onClick={handleCancelEdit}
                disabled={savingProfile}
                className="px-4 py-2.5 rounded-xl border border-slate-200 text-slate-700 hover:bg-slate-50 text-xs font-semibold flex items-center gap-1.5 transition-colors"
              >
                <X className="w-4 h-4" />
                <span>Cancel</span>
              </button>
            </div>
          )}
        </form>
      </div>

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* 3. Account Activity & Longitudinal Evaluations Card                 */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      <div className="bg-white rounded-3xl border border-slate-200/90 shadow-sm p-6 sm:p-8 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-100 gap-2">
          <div>
            <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
              <Activity className="w-5 h-5 text-blue-600" /> Verified Account Activity
            </h3>
            <p className="text-xs text-slate-500">Live statistics computed from your stored PostgreSQL evaluations.</p>
          </div>
          <button
            onClick={() => navigate('/history')}
            className="text-xs font-bold text-blue-600 hover:text-blue-800 flex items-center gap-1 self-start sm:self-auto"
          >
            <span>View Full History</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>

        <div className="grid sm:grid-cols-3 gap-4 pt-2">
          {/* Total Evaluations */}
          <div className="p-4 rounded-2xl bg-slate-50 border border-slate-200/80">
            <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">Completed Assessments</span>
            <p className="text-2xl font-extrabold text-slate-900 mt-1 font-mono">
              {loadingProfile ? (
                <Loader2 className="w-5 h-5 animate-spin text-slate-400" />
              ) : (
                profileData?.total_assessments ?? 0
              )}
            </p>
            <p className="text-[11px] text-slate-400 mt-1">Evaluations saved to your account</p>
          </div>

          {/* Most Recent Evaluation */}
          <div className="p-4 rounded-2xl bg-slate-50 border border-slate-200/80">
            <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">Most Recent Assessment</span>
            <p className="text-sm font-bold text-slate-900 mt-2 truncate">
              {loadingProfile ? (
                <Loader2 className="w-4 h-4 animate-spin text-slate-400" />
              ) : (
                latestAssessmentDate
              )}
            </p>
            <p className="text-[11px] text-slate-400 mt-1">Timestamp of latest submission</p>
          </div>

          {/* Current Session Security */}
          <div className="p-4 rounded-2xl bg-slate-50 border border-slate-200/80">
            <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">Session Security</span>
            <p className="text-sm font-bold text-emerald-700 mt-2 flex items-center gap-1.5">
              <ShieldCheck className="w-4 h-4 text-emerald-600" />
              <span>JWT Bearer Active</span>
            </p>
            <p className="text-[11px] text-slate-400 mt-1">Stateless 24-hour token validity</p>
          </div>
        </div>

        {/* Action Callout */}
        <div className="p-4 rounded-2xl bg-blue-50/60 border border-blue-200 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs text-blue-900">
          <div className="flex items-center gap-2">
            <Clock className="w-4 h-4 text-blue-600 flex-shrink-0" />
            <span>Ready to conduct another longitudinal metabolic evaluation?</span>
          </div>
          <button
            onClick={() => navigate('/assess')}
            className="px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-700 text-white font-bold text-xs shadow-xs self-start sm:self-auto transition-colors"
          >
            New Assessment
          </button>
        </div>
      </div>

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* 4. Privacy & Data Governance Card                                   */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      <div className="bg-white rounded-3xl border border-slate-200/90 shadow-sm p-6 sm:p-8 space-y-3">
        <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
          <ShieldCheck className="w-4 h-4 text-teal-600" /> Privacy &amp; Health Data Governance
        </h3>
        <p className="text-xs text-slate-600 leading-relaxed">
          Your personal health assessment responses and predictive model outputs are strictly associated with your
          authenticated account ID in PostgreSQL. Data is never aggregated for commercial marketing or shared with
          third parties without clinical authorization.
        </p>
      </div>
    </div>
  )
}
