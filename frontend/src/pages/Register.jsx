/**
 * src/pages/Register.jsx
 * ======================
 * User registration page for DiabetesRisk AI.
 * Includes:
 *  - Full Name, Email, Password, Confirm Password
 *  - Password visibility toggles
 *  - Inline validation and requirements checker
 *  - Terms of Service & Privacy Policy agreement
 *  - Duplicate email collision feedback
 *  - Cohesive healthcare technology aesthetic
 */

import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import {
  Stethoscope,
  User,
  Mail,
  Lock,
  Eye,
  EyeOff,
  ArrowRight,
  CheckCircle2,
  AlertCircle,
  RefreshCw,
  ShieldCheck,
  Sparkles,
} from 'lucide-react'
import { useAuth } from '../context/AuthContext'

const EMAIL_REGEX = /^[^\s@]+@[^\s@]+\.[^\s@]+$/

export default function Register() {
  const navigate = useNavigate()
  const { register } = useAuth()

  const [formData, setFormData] = useState({
    fullName: '',
    email: '',
    password: '',
    confirmPassword: '',
    agreeTerms: false,
  })

  const [showPassword, setShowPassword] = useState(false)
  const [showConfirmPassword, setShowConfirmPassword] = useState(false)
  const [errors, setErrors] = useState({})
  const [apiError, setApiError] = useState(null)
  const [loading, setLoading] = useState(false)
  const [success, setSuccess] = useState(false)

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target
    setFormData((prev) => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value,
    }))
    // Clear field-level error on change
    if (errors[name]) {
      setErrors((prev) => ({ ...prev, [name]: null }))
    }
    if (apiError) setApiError(null)
  }

  const validate = () => {
    const newErrors = {}

    if (!formData.fullName.trim()) {
      newErrors.fullName = 'Full name is required.'
    } else if (formData.fullName.trim().length < 2) {
      newErrors.fullName = 'Full name must be at least 2 characters.'
    }

    if (!formData.email.trim()) {
      newErrors.email = 'Email address is required.'
    } else if (!EMAIL_REGEX.test(formData.email.trim())) {
      newErrors.email = 'Please enter a valid email address.'
    }

    if (!formData.password) {
      newErrors.password = 'Password is required.'
    } else if (formData.password.length < 8) {
      newErrors.password = 'Password must be at least 8 characters long.'
    }

    if (!formData.confirmPassword) {
      newErrors.confirmPassword = 'Confirm your password.'
    } else if (formData.password !== formData.confirmPassword) {
      newErrors.confirmPassword = 'Passwords do not match.'
    }

    if (!formData.agreeTerms) {
      newErrors.agreeTerms = 'You must accept the terms to create an account.'
    }

    setErrors(newErrors)
    return Object.keys(newErrors).length === 0
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!validate() || loading) return

    setLoading(true)
    setApiError(null)

    try {
      await register(
        formData.fullName.trim(),
        formData.email.trim(),
        formData.password,
      )
      setSuccess(true)
      setTimeout(() => {
        navigate('/dashboard', { replace: true })
      }, 1000)
    } catch (err) {
      const detail = err.response?.data?.detail
      if (err.response?.status === 409) {
        setApiError('An account with this email address already exists. Please sign in instead.')
      } else if (typeof detail === 'string') {
        setApiError(detail)
      } else if (Array.isArray(detail)) {
        setApiError(detail.map((d) => d.msg).join(', '))
      } else {
        setApiError('Failed to create account. Please check your connection and try again.')
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-[85vh] flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8 bg-slate-50">
      <div className="max-w-md w-full">
        {/* Branding & Header */}
        <div className="text-center mb-8">
          <Link to="/" className="inline-flex items-center gap-2.5 mb-3 group">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-teal-600 to-blue-600 flex items-center justify-center text-white shadow-md shadow-blue-500/20 group-hover:scale-105 transition-transform duration-200">
              <Stethoscope className="w-5 h-5" />
            </div>
            <span className="font-extrabold text-slate-900 text-2xl tracking-tight">
              DiabetesRisk <span className="text-teal-600">AI</span>
            </span>
          </Link>
          <h1 className="text-2xl font-extrabold text-slate-900 tracking-tight">
            Create Your Research Account
          </h1>
          <p className="text-sm text-slate-500 mt-1 max-w-xs mx-auto">
            Access secure diabetes prediction tracking, assessment history, and model insights.
          </p>
        </div>

        {/* Form Card (Slate-Navy Theme) */}
        <div className="bg-[#162032] rounded-3xl shadow-2xl border border-slate-700/60 p-7 sm:p-9 relative overflow-hidden text-white">
          {/* Top accent gradient bar */}
          <div className="absolute top-0 inset-x-0 h-1.5 bg-gradient-to-r from-blue-600 via-teal-500 to-blue-600" />

          {/* Success Banner */}
          {success && (
            <div className="mb-6 p-4 rounded-xl bg-emerald-950/60 border border-emerald-800 text-emerald-200 flex items-center gap-3 text-sm animate-in fade-in">
              <CheckCircle2 className="w-5 h-5 text-emerald-400 flex-shrink-0" />
              <span>Account created successfully! Redirecting to your dashboard...</span>
            </div>
          )}

          {/* API Error Banner */}
          {apiError && (
            <div className="mb-6 p-4 rounded-xl bg-rose-950/60 border border-rose-800 text-rose-200 flex items-start gap-3 text-sm animate-in fade-in">
              <AlertCircle className="w-5 h-5 text-rose-400 flex-shrink-0 mt-0.5" />
              <div className="flex-1">
                <div className="font-semibold mb-0.5">Registration Error</div>
                <div className="text-xs text-rose-300">{apiError}</div>
              </div>
            </div>
          )}

          <form onSubmit={handleSubmit} noValidate className="space-y-4">
            {/* Full Name */}
            <div>
              <label htmlFor="fullName" className="block text-xs font-semibold uppercase tracking-wider text-slate-300 mb-1.5">
                Full Name
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-400">
                  <User className="w-4 h-4" />
                </div>
                <input
                  id="fullName"
                  name="fullName"
                  type="text"
                  autoComplete="name"
                  value={formData.fullName}
                  onChange={handleChange}
                  placeholder="Dr. Jane Smith"
                  className={`w-full rounded-xl border border-slate-700 bg-slate-900/80 px-4 py-2.5 text-sm text-white shadow-xs focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20 pl-10 transition-colors placeholder:text-slate-500 ${
                    errors.fullName ? 'border-rose-500 focus:border-rose-500 focus:ring-rose-500/20' : ''
                  }`}
                />
              </div>
              {errors.fullName && (
                <p className="mt-1 text-xs text-rose-400 flex items-center gap-1">
                  <AlertCircle className="w-3 h-3" /> {errors.fullName}
                </p>
              )}
            </div>

            {/* Email Address */}
            <div>
              <label htmlFor="email" className="block text-xs font-semibold uppercase tracking-wider text-slate-300 mb-1.5">
                Email Address
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-400">
                  <Mail className="w-4 h-4" />
                </div>
                <input
                  id="email"
                  name="email"
                  type="email"
                  autoComplete="email"
                  value={formData.email}
                  onChange={handleChange}
                  placeholder="name@institution.org"
                  className={`w-full rounded-xl border border-slate-700 bg-slate-900/80 px-4 py-2.5 text-sm text-white shadow-xs focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20 pl-10 transition-colors placeholder:text-slate-500 ${
                    errors.email ? 'border-rose-500 focus:border-rose-500 focus:ring-rose-500/20' : ''
                  }`}
                />
              </div>
              {errors.email && (
                <p className="mt-1 text-xs text-rose-400 flex items-center gap-1">
                  <AlertCircle className="w-3 h-3" /> {errors.email}
                </p>
              )}
            </div>

            {/* Password */}
            <div>
              <label htmlFor="password" className="block text-xs font-semibold uppercase tracking-wider text-slate-300 mb-1.5">
                Password
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-400">
                  <Lock className="w-4 h-4" />
                </div>
                <input
                  id="password"
                  name="password"
                  type={showPassword ? 'text' : 'password'}
                  autoComplete="new-password"
                  value={formData.password}
                  onChange={handleChange}
                  placeholder="Minimum 8 characters"
                  className={`w-full rounded-xl border border-slate-700 bg-slate-900/80 px-4 py-2.5 text-sm text-white shadow-xs focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20 pl-10 pr-10 transition-colors placeholder:text-slate-500 ${
                    errors.password ? 'border-rose-500 focus:border-rose-500 focus:ring-rose-500/20' : ''
                  }`}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute inset-y-0 right-0 pr-3 flex items-center text-slate-400 hover:text-slate-200 focus:outline-none"
                  aria-label={showPassword ? 'Hide password' : 'Show password'}
                >
                  {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
              {errors.password ? (
                <p className="mt-1 text-xs text-rose-400 flex items-center gap-1">
                  <AlertCircle className="w-3 h-3" /> {errors.password}
                </p>
              ) : (
                <p className="mt-1 text-[11px] text-slate-400">Must be at least 8 characters long.</p>
              )}
            </div>

            {/* Confirm Password */}
            <div>
              <label htmlFor="confirmPassword" className="block text-xs font-semibold uppercase tracking-wider text-slate-300 mb-1.5">
                Confirm Password
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-400">
                  <Lock className="w-4 h-4" />
                </div>
                <input
                  id="confirmPassword"
                  name="confirmPassword"
                  type={showConfirmPassword ? 'text' : 'password'}
                  autoComplete="new-password"
                  value={formData.confirmPassword}
                  onChange={handleChange}
                  placeholder="Re-enter your password"
                  className={`w-full rounded-xl border border-slate-700 bg-slate-900/80 px-4 py-2.5 text-sm text-white shadow-xs focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20 pl-10 pr-10 transition-colors placeholder:text-slate-500 ${
                    errors.confirmPassword ? 'border-rose-500 focus:border-rose-500 focus:ring-rose-500/20' : ''
                  }`}
                />
                <button
                  type="button"
                  onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                  className="absolute inset-y-0 right-0 pr-3 flex items-center text-slate-400 hover:text-slate-200 focus:outline-none"
                  aria-label={showConfirmPassword ? 'Hide password' : 'Show password'}
                >
                  {showConfirmPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
              {errors.confirmPassword && (
                <p className="mt-1 text-xs text-rose-400 flex items-center gap-1">
                  <AlertCircle className="w-3 h-3" /> {errors.confirmPassword}
                </p>
              )}
            </div>

            {/* Terms Checkbox */}
            <div className="pt-1">
              <label className="flex items-start gap-2.5 cursor-pointer">
                <input
                  type="checkbox"
                  name="agreeTerms"
                  checked={formData.agreeTerms}
                  onChange={handleChange}
                  className="mt-0.5 rounded border-slate-700 bg-slate-900 text-blue-600 focus:ring-blue-500"
                />
                <span className="text-xs text-slate-300 leading-snug">
                  I agree to the{' '}
                  <span className="text-blue-400 font-semibold underline underline-offset-2">
                    Terms &amp; Research Privacy Notice
                  </span>
                  . This platform is for educational and research evaluation only.
                </span>
              </label>
              {errors.agreeTerms && (
                <p className="mt-1 text-xs text-rose-400 flex items-center gap-1">
                  <AlertCircle className="w-3 h-3" /> {errors.agreeTerms}
                </p>
              )}
            </div>

            {/* Submit Button */}
            <div className="pt-2">
              <button
                type="submit"
                disabled={loading || success}
                className="w-full bg-blue-600 hover:bg-blue-500 text-white font-bold rounded-xl py-3 shadow-md shadow-blue-600/25 transition-all flex items-center justify-center gap-2 disabled:opacity-50"
              >
                {loading ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    <span>Creating Account...</span>
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4 text-teal-300" />
                    <span>Create Account</span>
                    <ArrowRight className="w-4 h-4" />
                  </>
                )}
              </button>
            </div>
          </form>

          {/* Research Safeguard Note */}
          <div className="mt-6 pt-5 border-t border-slate-700/60 flex items-center justify-center gap-2 text-[11px] text-slate-400">
            <ShieldCheck className="w-3.5 h-3.5 text-teal-400" />
            <span>Encrypted with bcrypt &amp; JWT · Never plaintext</span>
          </div>
        </div>

        {/* Footer Link */}
        <p className="text-center text-sm text-slate-600 mt-6">
          Already have an account?{' '}
          <Link
            to="/login"
            className="font-bold text-blue-600 hover:text-blue-700 hover:underline"
          >
            Sign In
          </Link>
        </p>
      </div>
    </div>
  )
}

