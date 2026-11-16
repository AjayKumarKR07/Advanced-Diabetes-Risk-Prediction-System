/**
 * src/pages/Login.jsx
 * ===================
 * User login page for DiabetesRisk AI.
 * Includes:
 *  - Email & Password input fields with visibility toggle
 *  - Remember Me checkbox
 *  - Clarified Forgot Password link (research prototype note)
 *  - Inline & banner validation for invalid credentials (HTTP 401)
 *  - Prevention of duplicate/multiple submissions
 *  - Cohesive healthcare technology aesthetic
 */

import { useState } from 'react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import {
  Stethoscope,
  Mail,
  Lock,
  Eye,
  EyeOff,
  ArrowRight,
  AlertCircle,
  RefreshCw,
  ShieldCheck,
  LogIn,
  Info,
} from 'lucide-react'
import { useAuth } from '../context/AuthContext'

const EMAIL_REGEX = /^[^\s@]+@[^\s@]+\.[^\s@]+$/

export default function Login() {
  const navigate = useNavigate()
  const location = useLocation()
  const { login } = useAuth()

  // Where to send user after login
  const from = location.state?.from?.pathname || '/dashboard'

  const [formData, setFormData] = useState({
    email: '',
    password: '',
    rememberMe: true,
  })

  const [showPassword, setShowPassword] = useState(false)
  const [errors, setErrors] = useState({})
  const [apiError, setApiError] = useState(null)
  const [loading, setLoading] = useState(false)
  const [showForgotModal, setShowForgotModal] = useState(false)

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

    if (!formData.email.trim()) {
      newErrors.email = 'Email address is required.'
    } else if (!EMAIL_REGEX.test(formData.email.trim())) {
      newErrors.email = 'Please enter a valid email address.'
    }

    if (!formData.password) {
      newErrors.password = 'Password is required.'
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
      await login(
        formData.email.trim(),
        formData.password,
        formData.rememberMe,
      )
      navigate(from, { replace: true })
    } catch (err) {
      if (err.response?.status === 401) {
        setApiError('Invalid email address or password. Please verify your credentials and try again.')
      } else {
        const detail = err.response?.data?.detail
        setApiError(
          typeof detail === 'string'
            ? detail
            : 'Unable to connect to authentication service. Please check your backend.'
        )
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
            Sign In to Your Account
          </h1>
          <p className="text-sm text-slate-500 mt-1 max-w-xs mx-auto">
            Access your saved diabetes risk assessments and clinical prediction analytics.
          </p>
        </div>

        {/* Form Card (Slate-Navy Theme) */}
        <div className="bg-[#162032] rounded-3xl shadow-2xl border border-slate-700/60 p-7 sm:p-9 relative overflow-hidden text-white">
          {/* Top accent gradient bar */}
          <div className="absolute top-0 inset-x-0 h-1.5 bg-gradient-to-r from-blue-600 via-teal-500 to-blue-600" />

          {/* API Error Banner */}
          {apiError && (
            <div className="mb-6 p-4 rounded-xl bg-rose-950/60 border border-rose-800 text-rose-200 flex items-start gap-3 text-sm animate-in fade-in">
              <AlertCircle className="w-5 h-5 text-rose-400 flex-shrink-0 mt-0.5" />
              <div className="flex-1">
                <div className="font-semibold mb-0.5">Authentication Failed</div>
                <div className="text-xs text-rose-300">{apiError}</div>
              </div>
            </div>
          )}

          <form onSubmit={handleSubmit} noValidate className="space-y-4">
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
              <div className="flex items-center justify-between mb-1.5">
                <label htmlFor="password" className="block text-xs font-semibold uppercase tracking-wider text-slate-300">
                  Password
                </label>
                <button
                  type="button"
                  onClick={() => setShowForgotModal(true)}
                  className="text-xs text-blue-400 hover:text-blue-300 font-medium hover:underline"
                >
                  Forgot password?
                </button>
              </div>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-400">
                  <Lock className="w-4 h-4" />
                </div>
                <input
                  id="password"
                  name="password"
                  type={showPassword ? 'text' : 'password'}
                  autoComplete="current-password"
                  value={formData.password}
                  onChange={handleChange}
                  placeholder="Enter your password"
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
              {errors.password && (
                <p className="mt-1 text-xs text-rose-400 flex items-center gap-1">
                  <AlertCircle className="w-3 h-3" /> {errors.password}
                </p>
              )}
            </div>

            {/* Remember Me */}
            <div className="flex items-center justify-between pt-1">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  name="rememberMe"
                  checked={formData.rememberMe}
                  onChange={handleChange}
                  className="rounded border-slate-700 bg-slate-900 text-blue-600 focus:ring-blue-500"
                />
                <span className="text-xs text-slate-300 font-medium select-none">
                  Remember this device (24 hours)
                </span>
              </label>
            </div>

            {/* Submit Button */}
            <div className="pt-2">
              <button
                type="submit"
                disabled={loading}
                className="w-full bg-blue-600 hover:bg-blue-500 text-white font-bold rounded-xl py-3 shadow-md shadow-blue-600/25 transition-all flex items-center justify-center gap-2 disabled:opacity-50"
              >
                {loading ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    <span>Signing In...</span>
                  </>
                ) : (
                  <>
                    <LogIn className="w-4 h-4" />
                    <span>Sign In</span>
                    <ArrowRight className="w-4 h-4" />
                  </>
                )}
              </button>
            </div>
          </form>

          {/* Research Safeguard Note */}
          <div className="mt-6 pt-5 border-t border-slate-700/60 flex items-center justify-center gap-2 text-[11px] text-slate-400">
            <ShieldCheck className="w-3.5 h-3.5 text-teal-400" />
            <span>Secure session management · Research Prototype</span>
          </div>
        </div>

        {/* Footer Link */}
        <p className="text-center text-sm text-slate-600 mt-6">
          Don't have an account?{' '}
          <Link
            to="/register"
            className="font-bold text-blue-600 hover:text-blue-700 hover:underline"
          >
            Create an Account
          </Link>
        </p>
      </div>

      {/* Forgot Password Modal (Clearly Labelled Placeholder for Academic Prototype) */}
      {showForgotModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-xs animate-in fade-in">
          <div className="bg-white rounded-2xl max-w-sm w-full p-6 shadow-2xl border border-slate-100 relative">
            <div className="w-10 h-10 rounded-xl bg-blue-50 text-blue-600 flex items-center justify-center mb-3">
              <Info className="w-5 h-5" />
            </div>
            <h3 className="font-bold text-slate-900 text-lg mb-1.5">Password Recovery</h3>
            <p className="text-xs text-slate-500 leading-relaxed mb-5">
              This academic research prototype does not have an external SMTP email server configured.
              If you require a password reset, please register a new research account or contact the project team.
            </p>
            <button
              onClick={() => setShowForgotModal(false)}
              className="w-full btn-secondary py-2 text-xs font-semibold rounded-lg"
            >
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

