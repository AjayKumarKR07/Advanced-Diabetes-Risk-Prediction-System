/**
 * src/pages/Landing.jsx
 * ======================
 * Phase 6.8 — Professional homepage redesign.
 * Premium healthcare AI landing page with:
 *  - Hero section with SVG dashboard illustration
 *  - Trust & project highlights with animated count-up
 *  - How It Works (3 structured steps)
 *  - Platform feature cards
 *  - Model overview & transparent Phase 5 metrics
 *  - Health awareness section (general risk factors)
 *  - Final Call to Action
 */

import { useEffect, useRef, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import {
  ArrowRight,
  Activity,
  Brain,
  ClipboardList,
  BarChart2,
  Database,
  Clock,
  ShieldCheck,
  Heart,
  TrendingUp,
  Users,
  CheckCircle2,
  ChevronRight,
  Zap,
  Lock,
  LineChart,
  Sparkles,
  LogIn,
  UserPlus,
  LayoutDashboard,
  Menu,
  X,
} from 'lucide-react'

// ─── Embedded professional top nav (Landing-only, no duplicate Navbar) ────────
function LandingNav() {
  const { isAuthenticated } = useAuth()
  const navigate = useNavigate()
  const [mobileOpen, setMobileOpen] = useState(false)

  return (
    <header className="absolute top-0 inset-x-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Brand */}
          <Link to="/" className="flex items-center gap-2.5 group">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-teal-500 to-blue-600 flex items-center justify-center shadow-md shadow-blue-600/30">
              <Activity className="w-5 h-5 text-white" />
            </div>
            <div className="flex flex-col leading-tight">
              <span className="text-sm font-extrabold text-white tracking-tight">DiabetesRisk <span className="text-teal-300">AI</span></span>
              <span className="text-[9px] text-blue-300 font-semibold uppercase tracking-widest">Healthcare Analytics</span>
            </div>
          </Link>

          {/* Desktop nav links */}
          <nav className="hidden md:flex items-center gap-1 text-sm font-medium">
            <Link to="/performance" className="text-slate-300 hover:text-white px-3.5 py-2 rounded-lg hover:bg-white/10 transition-colors">Model Performance</Link>
            <Link to="/about" className="text-slate-300 hover:text-white px-3.5 py-2 rounded-lg hover:bg-white/10 transition-colors">About &amp; Methodology</Link>
          </nav>

          {/* Desktop CTA */}
          <div className="hidden md:flex items-center gap-2">
            {isAuthenticated ? (
              <button
                onClick={() => navigate('/dashboard')}
                className="inline-flex items-center gap-2 bg-blue-600 hover:bg-blue-500 text-white font-semibold text-sm px-4 py-2 rounded-xl transition-all shadow-md shadow-blue-600/30"
              >
                <LayoutDashboard className="w-4 h-4" />
                Dashboard
              </button>
            ) : (
              <>
                <Link to="/login" className="inline-flex items-center gap-1.5 text-slate-300 hover:text-white text-sm font-medium px-4 py-2 rounded-xl hover:bg-white/10 transition-colors">
                  <LogIn className="w-4 h-4" /> Sign In
                </Link>
                <Link to="/register" className="inline-flex items-center gap-1.5 bg-blue-600 hover:bg-blue-500 text-white font-semibold text-sm px-4 py-2 rounded-xl transition-all shadow-md shadow-blue-600/30">
                  <UserPlus className="w-4 h-4" /> Get Started
                </Link>
              </>
            )}
          </div>

          {/* Mobile hamburger */}
          <button
            className="md:hidden p-2 rounded-lg text-slate-300 hover:text-white hover:bg-white/10 transition-colors"
            onClick={() => setMobileOpen(!mobileOpen)}
            aria-label="Toggle menu"
          >
            {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>

        {/* Mobile drawer */}
        {mobileOpen && (
          <div className="md:hidden bg-slate-900/95 backdrop-blur-md border border-white/10 rounded-2xl mb-4 p-4 space-y-1">
            <Link to="/performance" onClick={() => setMobileOpen(false)} className="flex items-center gap-2 text-slate-300 hover:text-white px-3 py-2.5 rounded-xl hover:bg-white/10 transition-colors text-sm font-medium">Model Performance</Link>
            <Link to="/about" onClick={() => setMobileOpen(false)} className="flex items-center gap-2 text-slate-300 hover:text-white px-3 py-2.5 rounded-xl hover:bg-white/10 transition-colors text-sm font-medium">About &amp; Methodology</Link>
            <div className="border-t border-white/10 pt-2 mt-1">
              {isAuthenticated ? (
                <button onClick={() => { setMobileOpen(false); navigate('/dashboard') }} className="w-full text-left flex items-center gap-2 text-blue-300 hover:text-white px-3 py-2.5 rounded-xl hover:bg-white/10 transition-colors text-sm font-semibold">
                  <LayoutDashboard className="w-4 h-4" /> Dashboard
                </button>
              ) : (
                <>
                  <Link to="/login" onClick={() => setMobileOpen(false)} className="flex items-center gap-2 text-slate-300 hover:text-white px-3 py-2.5 rounded-xl hover:bg-white/10 transition-colors text-sm font-medium"><LogIn className="w-4 h-4" /> Sign In</Link>
                  <Link to="/register" onClick={() => setMobileOpen(false)} className="flex items-center gap-2 text-blue-300 hover:text-white px-3 py-2.5 rounded-xl hover:bg-white/10 transition-colors text-sm font-semibold"><UserPlus className="w-4 h-4" /> Get Started</Link>
                </>
              )}
            </div>
          </div>
        )}
      </div>
    </header>
  )
}

// ─── Animated counter hook ────────────────────────────────────────────────────
function useCountUp(target, duration = 1800, start = false) {
  const [count, setCount] = useState(0)
  useEffect(() => {
    if (!start) return
    let startTime = null
    const numeric = parseInt(target.replace(/[^0-9]/g, ''), 10)
    const step = (timestamp) => {
      if (!startTime) startTime = timestamp
      const progress = Math.min((timestamp - startTime) / duration, 1)
      const eased = 1 - Math.pow(1 - progress, 3)
      setCount(Math.floor(eased * numeric))
      if (progress < 1) requestAnimationFrame(step)
    }
    requestAnimationFrame(step)
  }, [start, target, duration])
  return count
}

// ─── Intersection observer hook ───────────────────────────────────────────────
function useInView(options = {}) {
  const ref = useRef(null)
  const [inView, setInView] = useState(false)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const obs = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) { setInView(true); obs.disconnect() }
    }, { threshold: 0.2, ...options })
    obs.observe(el)
    return () => obs.disconnect()
  }, [options])
  return [ref, inView]
}

// ─── Hero SVG illustration ────────────────────────────────────────────────────
function HeroDashboardIllustration() {
  return (
    <svg
      viewBox="0 0 480 360"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className="w-full max-w-lg mx-auto drop-shadow-2xl"
      aria-hidden="true"
    >
      {/* Card background */}
      <rect x="10" y="10" width="460" height="340" rx="20" fill="white" fillOpacity="0.08" stroke="white" strokeOpacity="0.15" strokeWidth="1.5"/>

      {/* Top bar */}
      <rect x="10" y="10" width="460" height="52" rx="20" fill="white" fillOpacity="0.12"/>
      <rect x="10" y="42" width="460" height="20" fill="white" fillOpacity="0.06"/>
      <circle cx="38" cy="36" r="8" fill="#34d399" fillOpacity="0.9"/>
      <circle cx="62" cy="36" r="8" fill="#fbbf24" fillOpacity="0.9"/>
      <circle cx="86" cy="36" r="8" fill="#f87171" fillOpacity="0.8"/>
      <text x="118" y="41" fill="white" fillOpacity="0.85" fontSize="12" fontFamily="Inter,sans-serif" fontWeight="600">DiabetesRisk AI — Research Prototype (Sample Demo)</text>

      {/* Risk gauge card */}
      <rect x="30" y="78" width="130" height="118" rx="14" fill="white" fillOpacity="0.10" stroke="white" strokeOpacity="0.12" strokeWidth="1"/>
      <text x="95" y="100" textAnchor="middle" fill="white" fillOpacity="0.65" fontSize="8" fontFamily="Inter,sans-serif" fontWeight="700">SAMPLE PREDICTION</text>
      {/* Gauge arc */}
      <path d="M55 150 A40 40 0 0 1 135 150" stroke="#334155" strokeWidth="6" strokeLinecap="round"/>
      <path d="M55 150 A40 40 0 0 1 95 110" stroke="#34d399" strokeWidth="6" strokeLinecap="round"/>
      <circle cx="95" cy="150" r="22" fill="#0f172a" fillOpacity="0.6"/>
      <text x="95" y="148" textAnchor="middle" fill="#34d399" fontSize="10" fontFamily="Inter,sans-serif" fontWeight="700">NO DIABETES</text>
      <text x="95" y="162" textAnchor="middle" fill="white" fillOpacity="0.6" fontSize="7.5" fontFamily="Inter,sans-serif">Sample Output</text>

      {/* Probability bars card */}
      <rect x="176" y="78" width="140" height="118" rx="14" fill="white" fillOpacity="0.10" stroke="white" strokeOpacity="0.12" strokeWidth="1"/>
      <text x="246" y="100" textAnchor="middle" fill="white" fillOpacity="0.65" fontSize="8" fontFamily="Inter,sans-serif" fontWeight="700">SAMPLE PROBABILITIES</text>
      {/* Bar: No Diabetes */}
      <text x="186" y="118" fill="white" fillOpacity="0.75" fontSize="8" fontFamily="Inter,sans-serif">No Diabetes (Demo)</text>
      <rect x="186" y="121" width="110" height="7" rx="3.5" fill="white" fillOpacity="0.1"/>
      <rect x="186" y="121" width="82" height="7" rx="3.5" fill="#34d399"/>
      <text x="302" y="128" fill="#34d399" fontSize="8" fontFamily="Inter,sans-serif" fontWeight="600">74%</text>
      {/* Bar: Prediabetes */}
      <text x="186" y="145" fill="white" fillOpacity="0.75" fontSize="8" fontFamily="Inter,sans-serif">Prediabetes (Demo)</text>
      <rect x="186" y="148" width="110" height="7" rx="3.5" fill="white" fillOpacity="0.1"/>
      <rect x="186" y="148" width="17" height="7" rx="3.5" fill="#fbbf24"/>
      <text x="302" y="155" fill="#fbbf24" fontSize="8" fontFamily="Inter,sans-serif" fontWeight="600">16%</text>
      {/* Bar: Diabetes */}
      <text x="186" y="172" fill="white" fillOpacity="0.75" fontSize="8" fontFamily="Inter,sans-serif">Diabetes (Demo)</text>
      <rect x="186" y="175" width="110" height="7" rx="3.5" fill="white" fillOpacity="0.1"/>
      <rect x="186" y="175" width="11" height="7" rx="3.5" fill="#f87171"/>
      <text x="302" y="182" fill="#f87171" fontSize="8" fontFamily="Inter,sans-serif" fontWeight="600">10%</text>

      {/* BMI card */}
      <rect x="332" y="78" width="118" height="118" rx="14" fill="white" fillOpacity="0.10" stroke="white" strokeOpacity="0.12" strokeWidth="1"/>
      <text x="391" y="100" textAnchor="middle" fill="white" fillOpacity="0.65" fontSize="8" fontFamily="Inter,sans-serif" fontWeight="700">SAMPLE BMI INPUT</text>
      <circle cx="391" cy="148" r="30" fill="none" stroke="white" strokeOpacity="0.1" strokeWidth="9"/>
      <circle cx="391" cy="148" r="30" fill="none" stroke="#38bdf8" strokeWidth="9" strokeDasharray="135 54" strokeDashoffset="45" strokeLinecap="round"/>
      <text x="391" y="145" textAnchor="middle" fill="white" fontSize="13" fontFamily="Inter,sans-serif" fontWeight="700">27.5</text>
      <text x="391" y="160" textAnchor="middle" fill="white" fillOpacity="0.6" fontSize="7.5" fontFamily="Inter,sans-serif">Sample Metric</text>

      {/* Heart rate & Health indicators line */}
      <rect x="30" y="212" width="420" height="80" rx="14" fill="white" fillOpacity="0.08" stroke="white" strokeOpacity="0.12" strokeWidth="1"/>
      <text x="50" y="232" fill="white" fillOpacity="0.65" fontSize="8" fontFamily="Inter,sans-serif" fontWeight="700">SAMPLE HEALTH INDICATORS TREND (ILLUSTRATIVE DEMO)</text>
      {/* Sparkline */}
      <polyline
        points="50,272 80,260 110,268 140,248 170,255 200,242 230,250 260,238 290,246 320,234 350,242 380,230 410,238 440,228"
        stroke="#38bdf8" strokeWidth="2.5" fill="none" strokeLinecap="round" strokeLinejoin="round"
      />
      <polyline
        points="50,272 80,260 110,268 140,248 170,255 200,242 230,250 260,238 290,246 320,234 350,242 380,230 410,238 440,228 440,272"
        fill="url(#areaGrad)"
      />
      <defs>
        <linearGradient id="areaGrad" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stopColor="#38bdf8" stopOpacity="0.2"/>
          <stop offset="100%" stopColor="#38bdf8" stopOpacity="0"/>
        </linearGradient>
      </defs>
      {/* Dot highlight */}
      <circle cx="440" cy="228" r="4" fill="#34d399"/>
      <circle cx="440" cy="228" r="8" fill="#34d399" fillOpacity="0.35"/>

      {/* Feature tags */}
      <rect x="30" y="308" width="82" height="22" rx="11" fill="#1e40af" fillOpacity="0.7"/>
      <text x="71" y="323" textAnchor="middle" fill="white" fillOpacity="0.9" fontSize="8" fontFamily="Inter,sans-serif" fontWeight="600">21 Indicators</text>
      <rect x="122" y="308" width="94" height="22" rx="11" fill="#065f46" fillOpacity="0.7"/>
      <text x="169" y="323" textAnchor="middle" fill="white" fillOpacity="0.9" fontSize="8" fontFamily="Inter,sans-serif" fontWeight="600">253,680 Records</text>
      <rect x="226" y="308" width="88" height="22" rx="11" fill="#854d0e" fillOpacity="0.7"/>
      <text x="270" y="323" textAnchor="middle" fill="white" fillOpacity="0.9" fontSize="8" fontFamily="Inter,sans-serif" fontWeight="600">3 Target Classes</text>
      <rect x="324" y="308" width="92" height="22" rx="11" fill="#581c87" fillOpacity="0.7"/>
      <text x="370" y="323" textAnchor="middle" fill="white" fillOpacity="0.9" fontSize="8" fontFamily="Inter,sans-serif" fontWeight="600">AUC 0.7707</text>

      {/* Illustrative Demo Watermark */}
      <text x="240" y="344" textAnchor="middle" fill="white" fillOpacity="0.4" fontSize="7" fontFamily="Inter,sans-serif" letterSpacing="0.4">ILLUSTRATIVE DEMO VISUALIZATION — SYNTHETIC SAMPLE VALUES ONLY</text>
    </svg>
  )
}

// ─── Stats data (Strictly matching prompt: 253,680 records, 21 features, 3 classes, ML analysis) ─
const STATS = [
  { raw: '253680', display: '253,680', label: 'CDC Health Indicator Records', color: 'text-blue-600', isNumeric: true },
  { raw: '21',     display: '21',      label: 'Health-Related Input Features',  color: 'text-teal-600', isNumeric: true },
  { raw: '3',      display: '3',       label: 'Prediction Classes',            color: 'text-indigo-600', isNumeric: true },
  { raw: '100',    display: 'ML-Based',label: 'Machine Learning-Based Analysis',color: 'text-emerald-600', isNumeric: false },
]

function StatCard({ stat, animate }) {
  const count = useCountUp(stat.raw, 1600, animate && stat.isNumeric)
  const display = stat.isNumeric
    ? (animate ? count.toLocaleString() : stat.display)
    : stat.display

  return (
    <div className="text-center px-4 py-7 group">
      <div className={`text-3xl sm:text-4xl font-extrabold tracking-tight ${stat.color} mb-1.5 transition-transform duration-300 group-hover:scale-105`}>
        {display}
      </div>
      <div className="text-xs sm:text-sm text-slate-500 font-medium max-w-[180px] mx-auto leading-snug">
        {stat.label}
      </div>
    </div>
  )
}

// ─── How It Works (3 Steps) ───────────────────────────────────────────────────
const STEPS = [
  {
    num: '01',
    Icon: ClipboardList,
    title: 'Enter Health Information',
    desc: 'Provide your health indicators through a structured assessment form.',
    color: 'blue',
  },
  {
    num: '02',
    Icon: Brain,
    title: 'AI-Powered Analysis',
    desc: 'The trained machine learning model processes the provided health information.',
    color: 'teal',
  },
  {
    num: '03',
    Icon: BarChart2,
    title: 'View Prediction Results',
    desc: 'Explore the predicted classification and model-estimated class probabilities.',
    color: 'indigo',
  },
]

const STEP_COLORS = {
  blue:   { bg: 'bg-blue-50',   icon: 'text-blue-600',   border: 'border-blue-100',  num: 'text-blue-100'  },
  teal:   { bg: 'bg-teal-50',   icon: 'text-teal-600',   border: 'border-teal-100',  num: 'text-teal-100'  },
  indigo: { bg: 'bg-indigo-50', icon: 'text-indigo-600', border: 'border-indigo-100',num: 'text-indigo-100'},
}

// ─── Platform features (6 Cards strictly matching requirements) ───────────────
const FEATURES = [
  {
    Icon: ClipboardList,
    title: 'Health Assessment',
    desc: 'Interactive 21-parameter health questionnaire covering blood pressure, cholesterol, BMI, lifestyle habits, and general wellness.',
    color: 'blue',
  },
  {
    Icon: Brain,
    title: 'Machine Learning Prediction',
    desc: 'Multiclass Logistic Regression pipeline trained with balanced class weights to evaluate individual probabilities across all three categories.',
    color: 'teal',
  },
  {
    Icon: Clock,
    title: 'Prediction History',
    desc: 'Complete, paginated record of past health assessments and generated predictions securely stored with timestamps in PostgreSQL.',
    color: 'indigo',
  },
  {
    Icon: LineChart,
    title: 'Model Performance Analytics',
    desc: 'Rigorous Phase 5 evaluation metrics on held-out test data, including Macro F1 (0.4246), ROC-AUC (0.7707), and confusion matrix.',
    color: 'purple',
  },
  {
    Icon: TrendingUp,
    title: 'Data-Driven Insights',
    desc: 'Rich interactive Recharts visualizations displaying model-estimated class probability distributions, research risk categories, and benchmark profiles.',
    color: 'emerald',
  },
  {
    Icon: Lock,
    title: 'Secure Database Integration',
    desc: 'Robust PostgreSQL integration powered by SQLAlchemy 2.x ORM with environment-isolated credentials and migration tracking.',
    color: 'slate',
  },
]

const FEAT_COLORS = {
  blue:    { bg: 'bg-blue-50',    icon: 'text-blue-600',    ring: 'ring-blue-100'    },
  teal:    { bg: 'bg-teal-50',    icon: 'text-teal-600',    ring: 'ring-teal-100'    },
  indigo:  { bg: 'bg-indigo-50',  icon: 'text-indigo-600',  ring: 'ring-indigo-100'  },
  purple:  { bg: 'bg-purple-50',  icon: 'text-purple-600',  ring: 'ring-purple-100'  },
  emerald: { bg: 'bg-emerald-50', icon: 'text-emerald-600', ring: 'ring-emerald-100' },
  slate:   { bg: 'bg-slate-100',  icon: 'text-slate-600',   ring: 'ring-slate-200'   },
}

// ─── Health awareness factors ─────────────────────────────────────────────────
const HEALTH_FACTORS = [
  {
    label: 'Body Mass Index (BMI)',
    desc: 'A primary metabolic risk indicator. Higher BMI categories correlate strongly with elevated insulin resistance across population surveys.',
  },
  {
    label: 'Blood Pressure',
    desc: 'Hypertension frequently co-occurs with metabolic syndrome and vascular risk factors in adults at risk of type 2 diabetes.',
  },
  {
    label: 'Cholesterol Levels',
    desc: 'Elevated total and LDL cholesterol along with lipid dysregulation are well-documented comorbidities in population metabolic health studies.',
  },
  {
    label: 'Physical Activity',
    desc: 'Regular moderate or vigorous exercise promotes glucose uptake into skeletal muscles and enhances insulin sensitivity.',
  },
  {
    label: 'General Health Indicators',
    desc: 'Self-reported general, physical, and mental health metrics offer validated composite signals of overall systemic physiological wellness.',
  },
]

// ─── Main Landing page ────────────────────────────────────────────────────────
export default function Landing() {
  const [statsRef, statsInView] = useInView()
  const [stepsRef, stepsInView] = useInView()

  return (
    <div className="bg-white">

      {/* ── 1. HERO SECTION ───────────────────────────────────────────────── */}
      <section className="relative overflow-hidden bg-gradient-to-br from-slate-900 via-blue-950 to-slate-900">
        {/* Professional embedded nav — no duplicate top Navbar */}
        <LandingNav />

        {/* Subtle grid background */}
        <div
          className="absolute inset-0 opacity-[0.035]"
          style={{
            backgroundImage: 'linear-gradient(white 1px,transparent 1px),linear-gradient(90deg,white 1px,transparent 1px)',
            backgroundSize: '48px 48px',
          }}
        />
        {/* Gradient blur glow accents */}
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-blue-500 rounded-full blur-[128px] opacity-15 -translate-y-1/2 pointer-events-none" />
        <div className="absolute bottom-0 right-1/4 w-80 h-80 bg-teal-400 rounded-full blur-[128px] opacity-15 translate-y-1/2 pointer-events-none" />

        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-32 pb-20 lg:pt-36 lg:pb-28">
          <div className="grid lg:grid-cols-2 gap-12 lg:gap-14 items-center">
            {/* Left Column: Heading, Description, CTAs */}
            <div>
              {/* Pill badge */}
              <div className="inline-flex items-center gap-2 bg-blue-500/10 border border-blue-400/25 text-blue-300 rounded-full px-4 py-1.5 text-xs font-semibold tracking-wide mb-6 backdrop-blur-xs">
                <Sparkles className="w-3.5 h-3.5 text-teal-300" />
                <span>AI-Powered Health Analytics · Research Platform</span>
              </div>

              {/* Main Heading */}
              <h1 className="text-4xl sm:text-5xl lg:text-[3.25rem] font-extrabold text-white leading-[1.12] tracking-tight mb-6">
                Understand Your{' '}
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-teal-300 to-teal-400">
                  Diabetes Risk.
                </span>
                <br />
                Take Control of{' '}
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-teal-300 via-blue-400 to-indigo-300">
                  Your Health.
                </span>
              </h1>

              {/* Supporting Description */}
              <p className="text-base sm:text-lg text-slate-300 leading-relaxed mb-8 max-w-xl">
                An AI-powered diabetes risk prediction platform that analyzes health indicators using
                machine learning to provide research-based insights and support greater health awareness.
              </p>

              {/* CTA Buttons */}
              <div className="flex flex-wrap items-center gap-3.5 mb-8">
                <Link
                  to="/assess"
                  className="inline-flex items-center gap-2 bg-blue-600 hover:bg-blue-500 text-white font-semibold px-6 py-3.5 rounded-xl transition-all duration-200 hover:shadow-lg hover:shadow-blue-600/30 hover:-translate-y-0.5 text-sm sm:text-base"
                >
                  <span>Start Health Assessment</span>
                  <ArrowRight className="w-4 h-4" />
                </Link>
                <Link
                  to="/performance"
                  className="inline-flex items-center gap-2 bg-white/10 hover:bg-white/15 border border-white/20 text-white font-medium px-5 py-3.5 rounded-xl transition-all duration-200 backdrop-blur-xs text-sm sm:text-base hover:-translate-y-0.5"
                >
                  <Activity className="w-4 h-4 text-teal-300" />
                  <span>Explore Model Performance</span>
                </Link>
              </div>

              {/* Small Research Disclaimer */}
              <div className="inline-flex items-center gap-2 text-xs text-slate-400 bg-slate-800/60 border border-slate-700/60 rounded-lg px-3.5 py-2">
                <ShieldCheck className="w-4 h-4 text-teal-400 flex-shrink-0" />
                <span>Research and educational use only. Not a substitute for professional medical advice.</span>
              </div>
            </div>

            {/* Right Column: Custom SVG Dashboard Illustration */}
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-br from-blue-600/25 to-teal-500/25 rounded-3xl blur-2xl transform scale-95" />
              <div className="relative bg-gradient-to-br from-blue-950/70 via-slate-900/80 to-slate-950/90 border border-white/10 rounded-3xl p-3 sm:p-5 shadow-2xl backdrop-blur-sm">
                <HeroDashboardIllustration />
              </div>
              <p className="text-center text-[11px] text-slate-400 mt-3 font-mono tracking-wide">
                Illustrative demo interface with synthetic sample values — not actual patient data.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ── 2. TRUST & PROJECT HIGHLIGHTS ─────────────────────────────────── */}
      <section className="bg-white border-b border-slate-100 shadow-xs" ref={statsRef}>
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-2 lg:grid-cols-4 divide-y lg:divide-y-0 lg:divide-x divide-slate-100">
            {STATS.map((stat) => (
              <StatCard key={stat.label} stat={stat} animate={statsInView} />
            ))}
          </div>
        </div>
      </section>

      {/* ── 3. HOW IT WORKS ───────────────────────────────────────────────── */}
      <section className="bg-slate-50/80 py-20 lg:py-28" ref={stepsRef}>
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <span className="inline-block text-xs font-semibold tracking-widest text-blue-600 uppercase mb-3">
              Process Workflow
            </span>
            <h2 className="text-3xl sm:text-4xl font-extrabold text-slate-900 mb-4 tracking-tight">
              How It Works
            </h2>
            <p className="text-slate-500 max-w-xl mx-auto text-sm sm:text-base leading-relaxed">
              A transparent, three-step research analytics workflow designed to evaluate self-reported
              health indicators and compute model-estimated class probabilities.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-6 lg:gap-8 relative">
            {/* Desktop Connector Line */}
            <div className="hidden md:block absolute top-16 left-1/4 right-1/4 h-0.5 bg-gradient-to-r from-blue-200 via-teal-200 to-indigo-200 -z-0" />

            {STEPS.map(({ num, Icon, title, desc, color }, i) => {
              const c = STEP_COLORS[color]
              return (
                <div
                  key={num}
                  className="relative z-10 bg-white rounded-2xl border border-slate-100 shadow-sm p-7 hover:shadow-md transition-all duration-300 hover:-translate-y-1 flex flex-col justify-between"
                  style={{ animationDelay: `${i * 120}ms` }}
                >
                  <div>
                    {/* Step Number Watermark */}
                    <span className={`absolute top-5 right-5 text-4xl sm:text-5xl font-black ${c.num} select-none`}>
                      {num}
                    </span>
                    <div className={`w-12 h-12 ${c.bg} rounded-xl flex items-center justify-center mb-5`}>
                      <Icon className={`w-6 h-6 ${c.icon}`} />
                    </div>
                    <div className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">
                      Step {num}
                    </div>
                    <h3 className="font-bold text-slate-900 text-lg mb-2.5">{title}</h3>
                    <p className="text-slate-600 text-sm leading-relaxed">{desc}</p>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </section>

      {/* ── 4. PLATFORM FEATURES ──────────────────────────────────────────── */}
      <section className="bg-white py-20 lg:py-28">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <span className="inline-block text-xs font-semibold tracking-widest text-teal-600 uppercase mb-3">
              Capabilities
            </span>
            <h2 className="text-3xl sm:text-4xl font-extrabold text-slate-900 mb-4 tracking-tight">
              Platform Features
            </h2>
            <p className="text-slate-500 max-w-xl mx-auto text-sm sm:text-base leading-relaxed">
              Engineered end-to-end: from validated CDC epidemiological data ingestion to
              reproducible ML inference and persistent relational storage.
            </p>
          </div>

          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {FEATURES.map(({ Icon, title, desc, color }) => {
              const c = FEAT_COLORS[color]
              return (
                <div
                  key={title}
                  className="group relative bg-white border border-slate-100 rounded-2xl p-6 hover:border-slate-200 hover:shadow-lg transition-all duration-300 hover:-translate-y-1 flex flex-col justify-between"
                >
                  <div>
                    <div className={`w-11 h-11 ${c.bg} ring-4 ${c.ring} rounded-xl flex items-center justify-center mb-5 group-hover:scale-110 transition-transform duration-300`}>
                      <Icon className={`w-5 h-5 ${c.icon}`} />
                    </div>
                    <h3 className="font-bold text-slate-900 text-base mb-2 group-hover:text-blue-700 transition-colors">
                      {title}
                    </h3>
                    <p className="text-slate-500 text-xs sm:text-sm leading-relaxed">
                      {desc}
                    </p>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </section>

      {/* ── 5. MODEL OVERVIEW SECTION ─────────────────────────────────────── */}
      <section className="bg-gradient-to-br from-blue-950 via-slate-900 to-slate-950 py-20 lg:py-28 text-white relative overflow-hidden">
        <div
          className="absolute inset-0 opacity-[0.04]"
          style={{
            backgroundImage: 'radial-gradient(white 1px, transparent 1px)',
            backgroundSize: '32px 32px',
          }}
        />
        <div className="relative max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid lg:grid-cols-2 gap-12 lg:gap-14 items-center">
            {/* Left Column: Methodology Explanation */}
            <div>
              <span className="inline-block text-xs font-semibold tracking-widest text-teal-400 uppercase mb-3">
                Transparent Science
              </span>
              <h2 className="text-3xl sm:text-4xl font-extrabold mb-5 leading-tight tracking-tight">
                Model Overview &amp;<br />
                <span className="text-blue-400">Scientific Methodology</span>
              </h2>
              <p className="text-slate-300 leading-relaxed mb-6 text-sm sm:text-base">
                The underlying architecture is built on the official <strong>CDC BRFSS 2015</strong> survey
                data. To protect against feature-profile leakage, training used strict group-aware
                stratification. The candidate model was evaluated exactly once on a frozen held-out test
                partition of 50,736 rows.
              </p>

              {/* Metadata specs */}
              <div className="space-y-2.5 mb-8 text-sm">
                <div className="flex items-center gap-2 text-slate-300">
                  <CheckCircle2 className="w-4 h-4 text-teal-400 flex-shrink-0" />
                  <span><strong>Dataset:</strong> CDC BRFSS 2015 (253,680 total responses)</span>
                </div>
                <div className="flex items-center gap-2 text-slate-300">
                  <CheckCircle2 className="w-4 h-4 text-teal-400 flex-shrink-0" />
                  <span><strong>Algorithm:</strong> Multiclass Logistic Regression (balanced class weights)</span>
                </div>
                <div className="flex items-center gap-2 text-slate-300">
                  <CheckCircle2 className="w-4 h-4 text-teal-400 flex-shrink-0" />
                  <span><strong>Input Features:</strong> 21 standardized epidemiological health indicators</span>
                </div>
                <div className="flex items-center gap-2 text-slate-300">
                  <CheckCircle2 className="w-4 h-4 text-teal-400 flex-shrink-0" />
                  <span><strong>Classes:</strong> No Diabetes (0), Prediabetes (1), Diabetes (2)</span>
                </div>
              </div>

              <Link
                to="/performance"
                className="inline-flex items-center gap-2 bg-blue-600 hover:bg-blue-500 text-white font-semibold px-6 py-3 rounded-xl transition-all duration-200 hover:shadow-lg hover:shadow-blue-600/30 text-sm"
              >
                <span>View Model Performance</span>
                <ChevronRight className="w-4 h-4" />
              </Link>
            </div>

            {/* Right Column: Actual Phase 5 Metric Cards */}
            <div className="grid grid-cols-2 gap-3.5 sm:gap-4">
              {[
                { label: 'Evaluation Partition', value: '50,736 rows', icon: Database, color: 'blue' },
                { label: 'Selected Algorithm',   value: 'Logistic Regression', icon: Brain, color: 'teal' },
                { label: 'ROC-AUC (OvR Macro)',  value: '0.7707', icon: Activity, color: 'emerald' },
                { label: 'Macro F1-Score',       value: '0.4246', icon: TrendingUp, color: 'purple' },
                { label: 'Balanced Accuracy',    value: '51.72%', icon: Users, color: 'indigo' },
                { label: 'Overall Accuracy',     value: '64.33%', icon: LineChart, color: 'amber' },
              ].map(({ label, value, icon: Icon, color }) => (
                <div
                  key={label}
                  className="bg-white/5 border border-white/10 rounded-xl p-4 hover:bg-white/10 transition-colors"
                >
                  <Icon className={`w-5 h-5 mb-2.5 ${
                    color === 'blue'    ? 'text-blue-400'    :
                    color === 'teal'    ? 'text-teal-400'    :
                    color === 'emerald' ? 'text-emerald-400' :
                    color === 'purple'  ? 'text-purple-400'  :
                    color === 'indigo'  ? 'text-indigo-400'  :
                                         'text-amber-400'
                  }`} />
                  <div className="text-xs text-slate-400 mb-1">{label}</div>
                  <div className="text-base font-bold text-white tracking-tight">{value}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Classification Target Pills */}
          <div className="mt-14 border-t border-white/10 pt-10">
            <div className="text-center text-xs font-semibold uppercase tracking-wider text-slate-400 mb-4">
              3-Class Target Categorization
            </div>
            <div className="flex flex-wrap justify-center gap-3.5">
              {[
                { label: 'No Diabetes', desc: 'Class 0 · Non-Diabetic Population Profile', dot: '#34d399' },
                { label: 'Prediabetes', desc: 'Class 1 · Prediabetes Classification', dot: '#fbbf24' },
                { label: 'Diabetes',    desc: 'Class 2 · Diabetes Classification', dot: '#f87171' },
              ].map(({ label, desc, dot }) => (
                <div
                  key={label}
                  className="flex items-center gap-3 bg-white/5 border border-white/10 rounded-full px-5 py-2.5 backdrop-blur-xs"
                >
                  <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ background: dot }} />
                  <span className="text-sm font-semibold text-white">{label}</span>
                  <span className="text-xs text-slate-400 hidden sm:inline">— {desc}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* ── 6. HEALTH AWARENESS SECTION ───────────────────────────────────── */}
      <section className="bg-slate-50/70 py-20 lg:py-28">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid lg:grid-cols-2 gap-12 lg:gap-14 items-center">
            {/* Left */}
            <div>
              <span className="inline-block text-xs font-semibold tracking-widest text-indigo-600 uppercase mb-3">
                Health Education
              </span>
              <h2 className="text-3xl sm:text-4xl font-extrabold text-slate-900 mb-5 leading-tight tracking-tight">
                General Diabetes Risk Factors
              </h2>
              <p className="text-slate-600 leading-relaxed mb-6 text-sm sm:text-base">
                Understanding health indicators can empower individuals to engage proactively
                with their healthcare providers. Metabolic conditions frequently correlate with
                multiple lifestyle and biometric signals.
              </p>
              <div className="bg-blue-50 border border-blue-200/80 rounded-xl p-4 text-xs text-blue-900 leading-relaxed mb-6">
                <strong>Educational note:</strong> This analysis highlights statistical correlations identified
                in epidemiological research. It does not provide personalized diagnostic advice or treatment regimens.
              </div>
              <Link
                to="/about"
                className="inline-flex items-center gap-1.5 text-sm font-semibold text-blue-700 hover:text-blue-800 transition-colors"
              >
                <span>Read more about our research methodology</span>
                <ArrowRight className="w-4 h-4" />
              </Link>
            </div>

            {/* Right: Key Factors List */}
            <div className="space-y-3.5">
              {HEALTH_FACTORS.map(({ label, desc }) => (
                <div
                  key={label}
                  className="flex gap-4 items-start bg-white border border-slate-100 rounded-xl p-4.5 shadow-xs hover:shadow-sm transition-shadow"
                >
                  <div className="flex-shrink-0 w-8 h-8 bg-indigo-50 rounded-lg flex items-center justify-center text-indigo-600 mt-0.5">
                    <CheckCircle2 className="w-4 h-4" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-slate-900 text-sm mb-1">{label}</h3>
                    <p className="text-slate-500 text-xs sm:text-sm leading-relaxed">{desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* ── 7. FINAL CALL TO ACTION ───────────────────────────────────────── */}
      <section className="bg-white py-20 lg:py-28">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <div className="bg-gradient-to-br from-blue-600 via-blue-700 to-indigo-700 rounded-3xl px-8 py-14 sm:py-16 relative overflow-hidden shadow-2xl shadow-blue-900/20">
            {/* Decorative pattern */}
            <div
              className="absolute inset-0 opacity-10"
              style={{
                backgroundImage: 'radial-gradient(white 1px, transparent 1px)',
                backgroundSize: '24px 24px',
              }}
            />
            <div className="relative">
              <div className="inline-flex items-center gap-2 bg-white/15 rounded-full px-4 py-1.5 text-xs text-white font-medium mb-5 backdrop-blur-xs">
                <Heart className="w-3.5 h-3.5 text-pink-300" />
                <span>Accessible · Research Platform · No Registration Required</span>
              </div>
              <h2 className="text-3xl sm:text-4xl font-extrabold text-white mb-4 leading-tight tracking-tight">
                Start Understanding Your Health Indicators Today
              </h2>
              <p className="text-blue-100 max-w-xl mx-auto mb-8 text-sm sm:text-base leading-relaxed">
                Complete the health assessment and explore the insights generated by our
                research-based machine learning system.
              </p>
              <Link
                to="/assess"
                className="inline-flex items-center gap-2 bg-white hover:bg-blue-50 text-blue-700 font-bold px-8 py-3.5 rounded-xl transition-all duration-200 hover:-translate-y-0.5 shadow-lg shadow-blue-900/25 text-base"
              >
                <span>Start Your Assessment</span>
                <ArrowRight className="w-4 h-4" />
              </Link>
              <p className="text-blue-200/70 text-xs mt-5">
                Takes less than 2 minutes · Immediate probability feedback · Research prototype
              </p>
            </div>
          </div>
        </div>
      </section>

    </div>
  )
}
