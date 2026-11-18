/**
 * src/pages/Register.jsx
 * ======================
 * Split-screen registration page for DiabetesRisk AI.
 * Left: Dark brand panel with features. Right: Clean white form with password strength meter.
 */

import { useState, useMemo } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import {
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
  Activity,
  BarChart3,
  ClipboardList,
  Heart,
} from 'lucide-react'
import { useAuth } from '../context/AuthContext'

const EMAIL_REGEX = /^[^\s@]+@[^\s@]+\.[^\s@]+$/

function getPasswordStrength(password) {
  if (!password) return { score: 0, label: '', color: '' }
  let score = 0
  if (password.length >= 8) score++
  if (/[a-z]/.test(password) && /[A-Z]/.test(password)) score++
  if (/[0-9]/.test(password)) score++
  if (/[^a-zA-Z0-9]/.test(password)) score++

  const levels = [
    { label: 'Too short', color: 'bg-slate-300', text: 'text-slate-400' },
    { label: 'Weak', color: 'bg-rose-500', text: 'text-rose-500' },
    { label: 'Fair', color: 'bg-amber-500', text: 'text-amber-500' },
    { label: 'Good', color: 'bg-blue-500', text: 'text-blue-500' },
    { label: 'Strong', color: 'bg-emerald-500', text: 'text-emerald-500' },
  ]
  return { score, ...levels[score] }
}

const BRAND_FEATURES = [
  { icon: ClipboardList, text: '21 CDC BRFSS health indicators analyzed' },
  { icon: BarChart3, text: 'AI-powered diabetes risk classification' },
  { icon: Heart, text: 'Longitudinal BMI & wellness tracking' },
  { icon: ShieldCheck, text: 'Encrypted data · JWT sessions · Private' },
]

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

  const strength = useMemo(() => getPasswordStrength(formData.password), [formData.password])

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target
    setFormData((prev) => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value,
    }))
    if (errors[name]) {
      setErrors((prev) => ({ ...prev, [name]: null }))
    }
    if (apiError) setApiError(null)
  }

  const validate = () => {
    const newErrors = {}
    if (!formData.fullName.trim()) newErrors.fullName = 'Full name is required.'
    else if (formData.fullName.trim().length < 2) newErrors.fullName = 'Full name must be at least 2 characters.'
    if (!formData.email.trim()) newErrors.email = 'Email address is required.'
    else if (!EMAIL_REGEX.test(formData.email.trim())) newErrors.email = 'Please enter a valid email address.'
    if (!formData.password) newErrors.password = 'Password is required.'
    else if (formData.password.length < 8) newErrors.password = 'Password must be at least 8 characters.'
    if (!formData.confirmPassword) newErrors.confirmPassword = 'Confirm your password.'
    else if (formData.password !== formData.confirmPassword) newErrors.confirmPassword = 'Passwords do not match.'
    if (!formData.agreeTerms) newErrors.agreeTerms = 'You must accept the terms to create an account.'
    setErrors(newErrors)
    return Object.keys(newErrors).length === 0
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!validate() || loading) return
    setLoading(true)
    setApiError(null)
    try {
      await register(formData.fullName.trim(), formData.email.trim(), formData.password)
      setSuccess(true)
      setTimeout(() => navigate('/dashboard', { replace: true }), 1000)
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
    <div className="min-h-screen flex bg-slate-50">
      {/* ── Left Brand Panel ── */}
      <div className="hidden lg:flex lg:w-5/12 xl:w-2/5 bg-gradient-to-br from-slate-900 via-slate-800 to-blue-950 flex-col justify-between p-10 relative overflow-hidden">
        {/* Background decoration */}
        <div className="absolute inset-0 opacity-10">
          <div className="absolute top-20 left-20 w-64 h-64 bg-blue-500 rounded-full blur-3xl" />
          <div className="absolute bottom-20 right-10 w-48 h-48 bg-teal-500 rounded-full blur-3xl" />
        </div>

        <div className="relative">
          {/* Brand logo */}
          <Link to="/" className="inline-flex items-center gap-3 mb-12 group">
            <div className="w-11 h-11 rounded-2xl bg-gradient-to-tr from-teal-500 to-blue-600 flex items-center justify-center shadow-lg group-hover:scale-105 transition-transform">
              <Activity className="w-6 h-6 text-white" />
            </div>
            <div>
              <span className="font-extrabold text-white text-xl tracking-tight">
                DiabetesRisk <span className="text-teal-400">AI</span>
              </span>
              <p className="text-[10px] uppercase tracking-widest text-slate-400 font-semibold mt-0.5">Clinical Analytics</p>
            </div>
          </Link>

          <h2 className="text-3xl font-extrabold text-white leading-tight mb-3">
            Join thousands<br />tracking their health.
          </h2>
          <p className="text-slate-400 text-sm leading-relaxed mb-10">
            Create a free account and get instant access to AI-powered diabetes risk screening powered by CDC BRFSS 2015 data.
          </p>

          <ul className="space-y-4">
            {BRAND_FEATURES.map(({ icon: Icon, text }) => (
              <li key={text} className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-white/10 flex items-center justify-center flex-shrink-0">
                  <Icon className="w-4 h-4 text-teal-400" />
                </div>
                <span className="text-sm text-slate-300">{text}</span>
              </li>
            ))}
          </ul>
        </div>

        <div className="relative">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-white/10 border border-white/15 backdrop-blur-sm">
            <ShieldCheck className="w-4 h-4 text-teal-400" />
            <span className="text-xs text-slate-300 font-medium">bcrypt encrypted · JWT sessions · Never plaintext</span>
          </div>
          <p className="text-[11px] text-slate-500 mt-4">
            For educational and research use only. Not a substitute for professional medical advice.
          </p>
        </div>
      </div>

      {/* ── Right Form Panel ── */}
      <div className="flex-1 flex items-center justify-center py-12 px-6 sm:px-10 lg:px-16">
        <div className="w-full max-w-md">

          {/* Mobile brand header */}
          <div className="flex items-center gap-3 mb-8 lg:hidden">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-teal-500 to-blue-600 flex items-center justify-center">
              <Activity className="w-5 h-5 text-white" />
            </div>
            <span className="font-extrabold text-slate-900 text-xl">DiabetesRisk <span className="text-teal-600">AI</span></span>
          </div>

          <div className="mb-8">
            <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">Create free account</h1>
            <p className="text-sm text-slate-500 mt-1">Already have an account?{' '}
              <Link to="/login" className="font-bold text-blue-600 hover:text-blue-700 hover:underline">Sign in</Link>
            </p>
          </div>

          {/* Success Banner */}
          {success && (
            <div className="mb-6 p-4 rounded-xl bg-emerald-50 border border-emerald-200 text-emerald-800 flex items-center gap-3 text-sm">
              <CheckCircle2 className="w-5 h-5 text-emerald-600 flex-shrink-0" />
              <span className="font-medium">Account created! Redirecting to dashboard...</span>
            </div>
          )}

          {/* API Error */}
          {apiError && (
            <div className="mb-6 p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 flex items-start gap-3 text-sm">
              <AlertCircle className="w-5 h-5 text-rose-600 flex-shrink-0 mt-0.5" />
              <div>
                <div className="font-semibold mb-0.5">Registration Error</div>
                <div className="text-xs text-rose-700">{apiError}</div>
              </div>
            </div>
          )}

          <form onSubmit={handleSubmit} noValidate className="space-y-5">
            {/* Full Name */}
            <div>
              <label htmlFor="fullName" className="block text-xs font-semibold uppercase tracking-wider text-slate-700 mb-1.5">
                Full Name
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-400">
                  <User className="w-4 h-4" />
                </div>
                <input
                  id="fullName" name="fullName" type="text" autoComplete="name"
                  value={formData.fullName} onChange={handleChange} placeholder="Your full name"
                  className={`w-full rounded-xl border bg-slate-50/70 px-4 py-2.5 text-sm text-slate-900 shadow-sm focus:bg-white focus:outline-none focus:ring-2 focus:ring-blue-500/20 pl-10 transition-colors placeholder:text-slate-400 ${errors.fullName ? 'border-rose-400 focus:border-rose-400' : 'border-slate-300 focus:border-blue-500'}`}
                />
              </div>
              {errors.fullName && <p className="mt-1 text-xs text-rose-600 flex items-center gap-1 font-medium"><AlertCircle className="w-3 h-3" /> {errors.fullName}</p>}
            </div>

            {/* Email */}
            <div>
              <label htmlFor="email" className="block text-xs font-semibold uppercase tracking-wider text-slate-700 mb-1.5">
                Email Address
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-400">
                  <Mail className="w-4 h-4" />
                </div>
                <input
                  id="email" name="email" type="email" autoComplete="email"
                  value={formData.email} onChange={handleChange} placeholder="name@example.com"
                  className={`w-full rounded-xl border bg-slate-50/70 px-4 py-2.5 text-sm text-slate-900 shadow-sm focus:bg-white focus:outline-none focus:ring-2 focus:ring-blue-500/20 pl-10 transition-colors placeholder:text-slate-400 ${errors.email ? 'border-rose-400 focus:border-rose-400' : 'border-slate-300 focus:border-blue-500'}`}
                />
              </div>
              {errors.email && <p className="mt-1 text-xs text-rose-600 flex items-center gap-1 font-medium"><AlertCircle className="w-3 h-3" /> {errors.email}</p>}
            </div>

            {/* Password + Strength Meter */}
            <div>
              <label htmlFor="password" className="block text-xs font-semibold uppercase tracking-wider text-slate-700 mb-1.5">
                Password
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-400">
                  <Lock className="w-4 h-4" />
                </div>
                <input
                  id="password" name="password" type={showPassword ? 'text' : 'password'}
                  autoComplete="new-password" value={formData.password} onChange={handleChange}
                  placeholder="Minimum 8 characters"
                  className={`w-full rounded-xl border bg-slate-50/70 px-4 py-2.5 text-sm text-slate-900 shadow-sm focus:bg-white focus:outline-none focus:ring-2 focus:ring-blue-500/20 pl-10 pr-10 transition-colors placeholder:text-slate-400 ${errors.password ? 'border-rose-400 focus:border-rose-400' : 'border-slate-300 focus:border-blue-500'}`}
                />
                <button type="button" onClick={() => setShowPassword(!showPassword)}
                  className="absolute inset-y-0 right-0 pr-3 flex items-center text-slate-400 hover:text-slate-600 focus:outline-none"
                  aria-label={showPassword ? 'Hide password' : 'Show password'}>
                  {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>

              {/* Password Strength Meter */}
              {formData.password && (
                <div className="mt-2 space-y-1">
                  <div className="flex gap-1">
                    {[1, 2, 3, 4].map((seg) => (
                      <div
                        key={seg}
                        className={`flex-1 h-1.5 rounded-full transition-all duration-300 ${
                          strength.score >= seg ? strength.color : 'bg-slate-200'
                        }`}
                      />
                    ))}
                  </div>
                  <p className={`text-[11px] font-semibold ${strength.text}`}>
                    {strength.label}
                    {strength.score < 4 && (
                      <span className="text-slate-400 font-normal ml-1">
                        — add {!(/[A-Z]/.test(formData.password)) ? 'uppercase, ' : ''}{!(/[0-9]/.test(formData.password)) ? 'numbers, ' : ''}{!(/[^a-zA-Z0-9]/.test(formData.password)) ? 'symbols' : ''}
                      </span>
                    )}
                  </p>
                </div>
              )}

              {errors.password && <p className="mt-1 text-xs text-rose-600 flex items-center gap-1 font-medium"><AlertCircle className="w-3 h-3" /> {errors.password}</p>}
            </div>

            {/* Confirm Password */}
            <div>
              <label htmlFor="confirmPassword" className="block text-xs font-semibold uppercase tracking-wider text-slate-700 mb-1.5">
                Confirm Password
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-400">
                  <Lock className="w-4 h-4" />
                </div>
                <input
                  id="confirmPassword" name="confirmPassword"
                  type={showConfirmPassword ? 'text' : 'password'}
                  autoComplete="new-password" value={formData.confirmPassword} onChange={handleChange}
                  placeholder="Re-enter your password"
                  className={`w-full rounded-xl border bg-slate-50/70 px-4 py-2.5 text-sm text-slate-900 shadow-sm focus:bg-white focus:outline-none focus:ring-2 focus:ring-blue-500/20 pl-10 pr-10 transition-colors placeholder:text-slate-400 ${errors.confirmPassword ? 'border-rose-400 focus:border-rose-400' : 'border-slate-300 focus:border-blue-500'}`}
                />
                <button type="button" onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                  className="absolute inset-y-0 right-0 pr-3 flex items-center text-slate-400 hover:text-slate-600 focus:outline-none"
                  aria-label={showConfirmPassword ? 'Hide password' : 'Show password'}>
                  {showConfirmPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
              {errors.confirmPassword && <p className="mt-1 text-xs text-rose-600 flex items-center gap-1 font-medium"><AlertCircle className="w-3 h-3" /> {errors.confirmPassword}</p>}
            </div>

            {/* Terms Checkbox */}
            <div>
              <label className="flex items-start gap-2.5 cursor-pointer">
                <input
                  type="checkbox" name="agreeTerms"
                  checked={formData.agreeTerms} onChange={handleChange}
                  className="mt-0.5 rounded border-slate-300 text-blue-600 focus:ring-blue-500"
                />
                <span className="text-xs text-slate-600 leading-snug">
                  I agree to the{' '}
                  <span className="text-blue-600 font-semibold underline underline-offset-2">Terms &amp; Research Privacy Notice</span>.
                  This platform is for educational and research evaluation only.
                </span>
              </label>
              {errors.agreeTerms && <p className="mt-1 text-xs text-rose-600 flex items-center gap-1 font-medium"><AlertCircle className="w-3 h-3" /> {errors.agreeTerms}</p>}
            </div>

            {/* Submit */}
            <button
              type="submit" disabled={loading || success}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-xl py-3 shadow-md shadow-blue-600/20 transition-all flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? (
                <><RefreshCw className="w-4 h-4 animate-spin" /><span>Creating Account...</span></>
              ) : (
                <><Sparkles className="w-4 h-4 text-teal-300" /><span>Create Free Account</span><ArrowRight className="w-4 h-4" /></>
              )}
            </button>
          </form>

          <div className="mt-6 flex items-center justify-center gap-2 text-[11px] text-slate-400">
            <ShieldCheck className="w-3.5 h-3.5 text-teal-500" />
            <span>Encrypted with bcrypt &amp; JWT · Never stored as plaintext</span>
          </div>
        </div>
      </div>
    </div>
  )
}
