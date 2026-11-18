/**
 * src/pages/Login.jsx
 * ===================
 * Professional split-screen login page for DiabetesRisk AI.
 * Left: brand panel with feature highlights
 * Right: clean auth form with validation
 */

import { useState } from 'react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import {
  Activity,
  Mail,
  Lock,
  Eye,
  EyeOff,
  ArrowRight,
  AlertCircle,
  RefreshCw,
  ShieldCheck,
  LogIn,
  CheckCircle2,
  BarChart3,
  ClipboardList,
  Heart,
  TrendingUp,
  Info,
} from 'lucide-react'
import { useAuth } from '../context/AuthContext'

const EMAIL_REGEX = /^[^\s@]+@[^\s@]+\.[^\s@]+$/

const BRAND_FEATURES = [
  { icon: ClipboardList, text: '21-feature CDC BRFSS health assessment' },
  { icon: BarChart3,    text: 'AI-powered diabetes risk classification' },
  { icon: TrendingUp,   text: 'Longitudinal BMI & wellness tracking' },
  { icon: ShieldCheck,  text: 'Secure · encrypted · private data' },
]

export default function Login() {
  const navigate  = useNavigate()
  const location  = useLocation()
  const { login } = useAuth()

  const from = location.state?.from?.pathname || '/dashboard'

  const [formData, setFormData] = useState({ email: '', password: '', rememberMe: true })
  const [showPassword, setShowPassword]     = useState(false)
  const [errors, setErrors]                 = useState({})
  const [apiError, setApiError]             = useState(null)
  const [loading, setLoading]               = useState(false)
  const [showForgotModal, setShowForgotModal] = useState(false)

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target
    setFormData(prev => ({ ...prev, [name]: type === 'checkbox' ? checked : value }))
    if (errors[name]) setErrors(prev => ({ ...prev, [name]: null }))
    if (apiError) setApiError(null)
  }

  const validate = () => {
    const errs = {}
    if (!formData.email.trim())                    errs.email    = 'Email address is required.'
    else if (!EMAIL_REGEX.test(formData.email))    errs.email    = 'Please enter a valid email address.'
    if (!formData.password)                        errs.password = 'Password is required.'
    setErrors(errs)
    return Object.keys(errs).length === 0
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!validate() || loading) return
    setLoading(true); setApiError(null)
    try {
      await login(formData.email.trim(), formData.password, formData.rememberMe)
      navigate(from, { replace: true })
    } catch (err) {
      if (err.response?.status === 401) {
        setApiError('Invalid email or password. Please verify your credentials.')
      } else {
        const detail = err.response?.data?.detail
        setApiError(typeof detail === 'string' ? detail : 'Unable to connect. Please check your connection.')
      }
    } finally { setLoading(false) }
  }

  return (
    <div className="min-h-screen flex">

      {/* ── LEFT: Brand Panel ─────────────────────────────────────────────── */}
      <div className="hidden lg:flex lg:w-[45%] bg-gradient-to-br from-slate-900 via-blue-950 to-slate-900 flex-col justify-between p-12 relative overflow-hidden">
        {/* Grid overlay */}
        <div className="absolute inset-0 opacity-[0.04]" style={{ backgroundImage: 'linear-gradient(white 1px,transparent 1px),linear-gradient(90deg,white 1px,transparent 1px)', backgroundSize: '40px 40px' }} />
        {/* Glow */}
        <div className="absolute top-1/4 left-1/2 w-72 h-72 bg-blue-500 rounded-full blur-[120px] opacity-20 -translate-x-1/2 pointer-events-none" />

        {/* Brand */}
        <Link to="/" className="relative flex items-center gap-3 w-fit">
          <div className="w-11 h-11 rounded-2xl bg-gradient-to-tr from-teal-500 to-blue-600 flex items-center justify-center shadow-lg shadow-blue-600/30">
            <Activity className="w-6 h-6 text-white" />
          </div>
          <div>
            <div className="text-white font-extrabold text-xl tracking-tight leading-none">DiabetesRisk <span className="text-teal-300">AI</span></div>
            <div className="text-blue-300 text-[10px] font-semibold uppercase tracking-widest mt-0.5">Healthcare Analytics</div>
          </div>
        </Link>

        {/* Hero text */}
        <div className="relative space-y-6">
          <div>
            <h2 className="text-3xl font-extrabold text-white leading-tight mb-3">
              Your personal<br />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-teal-300 to-blue-400">health analytics</span><br />
              platform.
            </h2>
            <p className="text-slate-400 text-sm leading-relaxed max-w-xs">
              Track diabetes risk predictions, monitor health trends, and explore machine learning insights — all in one secure place.
            </p>
          </div>

          {/* Feature list */}
          <ul className="space-y-3">
            {BRAND_FEATURES.map(({ icon: Icon, text }) => (
              <li key={text} className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-white/10 flex items-center justify-center flex-shrink-0">
                  <Icon className="w-4 h-4 text-teal-300" />
                </div>
                <span className="text-slate-300 text-sm">{text}</span>
              </li>
            ))}
          </ul>
        </div>

        {/* Bottom disclaimer */}
        <div className="relative">
          <div className="inline-flex items-center gap-2 bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-xs text-slate-400">
            <ShieldCheck className="w-4 h-4 text-teal-400 flex-shrink-0" />
            Research & educational use only. Not clinical medical advice.
          </div>
        </div>
      </div>

      {/* ── RIGHT: Form Panel ─────────────────────────────────────────────── */}
      <div className="flex-1 flex items-center justify-center px-6 py-12 bg-white">
        <div className="w-full max-w-sm">

          {/* Mobile brand */}
          <Link to="/" className="lg:hidden flex items-center gap-2.5 mb-8 justify-center">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-teal-600 to-blue-600 flex items-center justify-center text-white shadow-md">
              <Activity className="w-5 h-5" />
            </div>
            <span className="font-extrabold text-slate-900 text-xl tracking-tight">DiabetesRisk <span className="text-teal-600">AI</span></span>
          </Link>

          {/* Heading */}
          <div className="mb-8">
            <h1 className="text-2xl font-extrabold text-slate-900 tracking-tight">Welcome back</h1>
            <p className="text-sm text-slate-500 mt-1">Sign in to access your health dashboard.</p>
          </div>

          {/* API Error */}
          {apiError && (
            <div className="mb-5 p-3.5 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 flex items-start gap-3 text-sm">
              <AlertCircle className="w-4 h-4 text-rose-600 flex-shrink-0 mt-0.5" />
              <div>
                <div className="font-semibold text-xs mb-0.5">Authentication Failed</div>
                <div className="text-xs text-rose-700">{apiError}</div>
              </div>
            </div>
          )}

          <form onSubmit={handleSubmit} noValidate className="space-y-4">
            {/* Email */}
            <div>
              <label htmlFor="email" className="block text-xs font-semibold text-slate-700 mb-1.5 uppercase tracking-wide">Email address</label>
              <div className="relative">
                <Mail className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400 pointer-events-none" />
                <input
                  id="email" name="email" type="email" autoComplete="email"
                  value={formData.email} onChange={handleChange}
                  placeholder="you@example.com"
                  className={`w-full pl-10 pr-4 py-2.5 rounded-xl border text-sm transition-all focus:outline-none focus:ring-2 ${
                    errors.email
                      ? 'border-rose-400 bg-rose-50 focus:border-rose-500 focus:ring-rose-500/20'
                      : 'border-slate-200 bg-slate-50/60 text-slate-900 focus:border-blue-500 focus:ring-blue-500/20 focus:bg-white'
                  }`}
                />
              </div>
              {errors.email && <p className="mt-1 text-xs text-rose-600 flex items-center gap-1"><AlertCircle className="w-3 h-3" />{errors.email}</p>}
            </div>

            {/* Password */}
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label htmlFor="password" className="text-xs font-semibold text-slate-700 uppercase tracking-wide">Password</label>
                <button type="button" onClick={() => setShowForgotModal(true)} className="text-xs text-blue-600 hover:text-blue-700 font-medium">Forgot password?</button>
              </div>
              <div className="relative">
                <Lock className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400 pointer-events-none" />
                <input
                  id="password" name="password" type={showPassword ? 'text' : 'password'} autoComplete="current-password"
                  value={formData.password} onChange={handleChange}
                  placeholder="Enter your password"
                  className={`w-full pl-10 pr-10 py-2.5 rounded-xl border text-sm transition-all focus:outline-none focus:ring-2 ${
                    errors.password
                      ? 'border-rose-400 bg-rose-50 focus:border-rose-500 focus:ring-rose-500/20'
                      : 'border-slate-200 bg-slate-50/60 text-slate-900 focus:border-blue-500 focus:ring-blue-500/20 focus:bg-white'
                  }`}
                />
                <button type="button" onClick={() => setShowPassword(!showPassword)} className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600">
                  {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
              {errors.password && <p className="mt-1 text-xs text-rose-600 flex items-center gap-1"><AlertCircle className="w-3 h-3" />{errors.password}</p>}
            </div>

            {/* Remember Me */}
            <label className="flex items-center gap-2 cursor-pointer">
              <input type="checkbox" name="rememberMe" checked={formData.rememberMe} onChange={handleChange} className="rounded border-slate-300 text-blue-600 focus:ring-blue-500" />
              <span className="text-xs text-slate-600 font-medium select-none">Remember this device (24 hours)</span>
            </label>

            {/* Submit */}
            <button
              type="submit" disabled={loading}
              className="w-full mt-2 bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-xl py-2.5 shadow-md shadow-blue-600/20 transition-all flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? <><RefreshCw className="w-4 h-4 animate-spin" /><span>Signing in...</span></> : <><LogIn className="w-4 h-4" /><span>Sign In</span><ArrowRight className="w-4 h-4" /></>}
            </button>
          </form>

          {/* Divider */}
          <div className="my-6 flex items-center gap-3">
            <div className="flex-1 h-px bg-slate-200" />
            <span className="text-xs text-slate-400 font-medium">secure session</span>
            <div className="flex-1 h-px bg-slate-200" />
          </div>

          {/* Trust badges */}
          <div className="flex items-center justify-center gap-4 text-[11px] text-slate-400 mb-6">
            <span className="flex items-center gap-1"><ShieldCheck className="w-3.5 h-3.5 text-teal-500" />bcrypt encrypted</span>
            <span className="flex items-center gap-1"><CheckCircle2 className="w-3.5 h-3.5 text-blue-500" />JWT sessions</span>
          </div>

          {/* Sign up link */}
          <p className="text-center text-sm text-slate-600">
            Don't have an account?{' '}
            <Link to="/register" className="font-bold text-blue-600 hover:text-blue-700 hover:underline">Create free account</Link>
          </p>
        </div>
      </div>

      {/* Forgot Password Modal */}
      {showForgotModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-xs">
          <div className="bg-white rounded-2xl max-w-sm w-full p-6 shadow-2xl border border-slate-100">
            <div className="w-10 h-10 rounded-xl bg-blue-50 text-blue-600 flex items-center justify-center mb-3">
              <Info className="w-5 h-5" />
            </div>
            <h3 className="font-bold text-slate-900 text-lg mb-1.5">Password Recovery</h3>
            <p className="text-xs text-slate-500 leading-relaxed mb-5">
              This research prototype does not have an email server configured.
              Please register a new account or contact the project team for access.
            </p>
            <button onClick={() => setShowForgotModal(false)} className="w-full py-2 text-xs font-semibold rounded-lg border border-slate-200 hover:bg-slate-50 transition-colors">
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
