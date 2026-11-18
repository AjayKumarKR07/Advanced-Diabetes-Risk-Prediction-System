/**
 * src/pages/PredictionResult.jsx
 * ================================
 * Professional AI Prediction Results & Health Assessment Report for DiabetesRisk AI.
 * Phase 6.12: Advanced Prediction Results, Detailed Health Analysis & Assessment Report.
 * 
 * Features:
 * - Advanced Prediction Summary (No Diabetes Class 0, Prediabetes Class 1, Diabetes Class 2)
 * - Strict Elimination of Contradictory Warning Badges
 * - Three Model-Estimated Class Probabilities with Donut Chart and Horizontal Bars
 * - Assessment Data Summary Card (21 Features: 14 binary, 7 ordinal/continuous)
 * - Complete Submitted Health Information in 5 Collapsible Categories
 * - Scientific Health Indicator Interpretation (non-diagnostic, non-causal)
 * - Model Performance & Research Information (Phase 5 frozen metrics on 50,736 test rows)
 * - Transparent Prediabetes Limitation Disclosure (Precision: 0.0307, Recall: 0.3089)
 * - Explainability Section (Multinomial Softmax & Argmax Decision Rule)
 * - Longitudinal Historical Comparison for Authenticated User
 * - Export Actions: Multi-Page PDF Download (jsPDF) & Clean Print (window.print())
 * - Resilient Direct Access & Page Refresh Recovery via Backend API
 */

import { useState, useEffect, useMemo } from 'react'
import { useLocation, useNavigate, Link, useSearchParams } from 'react-router-dom'
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts'
import {
  ArrowLeft,
  RefreshCw,
  Clock,
  LayoutDashboard,
  ShieldAlert,
  Database,
  Calendar,
  User,
  Mail,
  Info,
  CheckCircle2,
  AlertTriangle,
  AlertCircle,
  Flame,
  Download,
  Printer,
  ChevronDown,
  ChevronUp,
  FileText,
  Activity,
  Scale,
  Heart,
  Shield,
  HelpCircle,
  TrendingUp,
  TrendingDown,
  Minus,
  Layers,
  FileCheck,
} from 'lucide-react'
import DisclaimerBanner from '../components/DisclaimerBanner'
import FeatureContributionCard from '../components/FeatureContributionCard'
import { useAuth } from '../context/AuthContext'
import {
  fetchPredictions,
  fetchAssessments,
  fetchAssessment,
  fetchAssessmentPrediction,
} from '../services/api'
import { generateAssessmentReportPdf } from '../utils/generatePdfReport'
import {
  CLASS_LABELS,
  CLASS_COLORS,
  CDC_AGE_CATEGORIES,
  CDC_GENHLTH_OPTIONS,
  CDC_EDUCATION_OPTIONS,
  CDC_INCOME_OPTIONS,
  getBmiCategory,
} from '../utils/cdcFeatures'

export default function PredictionResult() {
  const location = useLocation()
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const { user } = useAuth()

  // State initialized from router navigation state (if coming from Assessment Form, History, or Dashboard)
  const [result, setResult] = useState(location.state?.result || null)
  const [inputs, setInputs] = useState(location.state?.inputs || null)
  const [personalInfo, setPersonalInfo] = useState(location.state?.personalInfo || null)

  // Loading & fallback states
  const [loading, setLoading] = useState(Boolean(searchParams.get('assessment_id') || searchParams.get('id') || !location.state?.result))
  const [error, setError] = useState('')
  const [pdfGenerating, setPdfGenerating] = useState(false)

  // Longitudinal history for comparison
  const [historicalData, setHistoricalData] = useState([])

  // Category collapsible toggles (all default expanded)
  const [expandedCategories, setExpandedCategories] = useState({
    personal: true,
    medical: true,
    lifestyle: true,
    general: true,
    demographics: true,
  })

  const toggleCategory = (key) => {
    setExpandedCategories((prev) => ({ ...prev, [key]: !prev[key] }))
  }

  const toggleAllCategories = () => {
    const allOpen = Object.values(expandedCategories).every(Boolean)
    const nextState = !allOpen
    setExpandedCategories({
      personal: nextState,
      medical: nextState,
      lifestyle: nextState,
      general: nextState,
      demographics: nextState,
    })
  }

  // ── Direct Access / Page Refresh Recovery & Data Enrichment ───────────────
  useEffect(() => {
    let isMounted = true

    async function loadData() {
      try {
        setError('')
        const targetId = searchParams.get('assessment_id') || searchParams.get('id')

        // 1. If explicit assessment_id is provided in URL query params
        if (targetId) {
          setLoading(true)
          const aid = parseInt(targetId, 10)
          const [assessRes, predRes] = await Promise.all([
            fetchAssessment(aid),
            fetchAssessmentPrediction(aid),
          ])

          if (isMounted) {
            setInputs(assessRes)
            setResult(predRes)
            setPersonalInfo({
              fullName: assessRes.user_name || user?.full_name || 'Patient Assessee',
              email: user?.email || 'N/A',
              rawAge: assessRes.Age ? `${assessRes.Age * 5 + 15}` : 'N/A',
              sex: assessRes.Sex,
              assessmentDate: new Date(assessRes.created_at).toLocaleDateString('en-US', {
                year: 'numeric',
                month: 'long',
                day: 'numeric',
              }),
            })
          }
        } else if (!result) {
          // 2. If result is missing (page refresh or direct URL navigation without query param)
          setLoading(true)
          const predsRes = await fetchPredictions(1, 10)
          const items = predsRes?.items || []

          if (items.length === 0) {
            if (isMounted) {
              setLoading(false)
              setError('No prior health assessments found on your account.')
            }
            return
          }

          const latestPred = items[0]
          if (isMounted) {
            setResult(latestPred)
          }

          // Fetch matching assessment data for 21 indicators
          const assessRes = await fetchAssessment(latestPred.assessment_id)
          if (isMounted) {
            setInputs(assessRes)
            setPersonalInfo({
              fullName: assessRes.user_name || user?.full_name || 'Patient Assessee',
              email: user?.email || 'N/A',
              rawAge: assessRes.Age ? `${assessRes.Age * 5 + 15}` : 'N/A',
              sex: assessRes.Sex,
              assessmentDate: new Date(assessRes.created_at).toLocaleDateString('en-US', {
                year: 'numeric',
                month: 'long',
                day: 'numeric',
              }),
            })
          }
        } else if (!inputs && result?.assessment_id) {
          // 3. Result exists from router state, but inputs missing (e.g. navigation from Dashboard)
          const assessRes = await fetchAssessment(result.assessment_id)
          if (isMounted) {
            setInputs(assessRes)
            if (!personalInfo) {
              setPersonalInfo({
                fullName: assessRes.user_name || user?.full_name || 'Patient Assessee',
                email: user?.email || 'N/A',
                rawAge: assessRes.Age ? `${assessRes.Age * 5 + 15}` : 'N/A',
                sex: assessRes.Sex,
                assessmentDate: new Date(assessRes.created_at).toLocaleDateString('en-US', {
                  year: 'numeric',
                  month: 'long',
                  day: 'numeric',
                }),
              })
            }
          }
        }

        // 4. Fetch history for longitudinal comparison
        const historyRes = await fetchPredictions(1, 10)
        const assessListRes = await fetchAssessments(1, 10)
        if (isMounted && historyRes?.items) {
          const assessMap = (assessListRes?.items || []).reduce((acc, a) => {
            acc[a.id] = a
            return acc
          }, {})

          const enriched = historyRes.items.map((p) => ({
            ...p,
            assessment: assessMap[p.assessment_id] || null,
          }))
          setHistoricalData(enriched)
        }
      } catch (err) {
        if (isMounted) {
          setError(
            err.response?.data?.detail ||
              'Failed to retrieve assessment details. You may not have permission to view this record.'
          )
        }
      } finally {
        if (isMounted) setLoading(false)
      }
    }

    loadData()
    return () => {
      isMounted = false
    }
  }, [location.search, user])

  // ── PDF Export Handler ───────────────────────────────────────────────────
  const handleDownloadPdf = () => {
    try {
      setPdfGenerating(true)
      generateAssessmentReportPdf({
        result,
        inputs: inputs || {},
        personalInfo: personalInfo || {
          fullName: user?.full_name || result?.user_name || 'Assessee',
          email: user?.email || 'N/A',
          assessmentDate: new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' }),
        },
        user,
        historicalStats: null,
      })
    } catch (err) {
      console.error('PDF export failed:', err)
      alert('Could not generate PDF report. Please check your browser settings.')
    } finally {
      setPdfGenerating(false)
    }
  }

  // ── Print Handler ────────────────────────────────────────────────────────
  const handlePrint = () => {
    window.print()
  }

  // ── Loading & Empty Fallback Views ───────────────────────────────────────
  if (loading) {
    return (
      <div className="min-h-[70vh] flex flex-col items-center justify-center px-4 py-16 text-center">
        <div className="w-14 h-14 rounded-2xl bg-blue-50 border border-blue-200 flex items-center justify-center text-blue-600 mb-4 animate-spin">
          <RefreshCw className="w-6 h-6" />
        </div>
        <h2 className="text-lg font-bold text-slate-800 mb-1">Loading Assessment Results...</h2>
        <p className="text-xs text-slate-500">Retrieving your encrypted assessment records and model outputs.</p>
      </div>
    )
  }

  if (!result || error) {
    return (
      <div className="min-h-[70vh] flex flex-col items-center justify-center px-4 py-16 text-center">
        <div className="w-16 h-16 rounded-2xl bg-blue-50 border border-blue-200 flex items-center justify-center text-blue-600 mb-4 shadow-sm">
          <Database className="w-8 h-8" />
        </div>
        <h2 className="text-xl font-bold text-slate-800 mb-2">No Active Assessment Result</h2>
        <p className="text-sm text-slate-500 max-w-md mb-6">
          {error || 'You have not submitted a health assessment yet. Complete an assessment or select an entry from your assessment history.'}
        </p>
        <div className="flex flex-wrap justify-center gap-3">
          <button onClick={() => navigate('/assess')} className="btn-primary">
            Start New Assessment
          </button>
          <button onClick={() => navigate('/history')} className="btn-secondary">
            View Assessment History
          </button>
        </div>
      </div>
    )
  }

  // ── Normalised Probability & Classification Values ────────────────────────
  const predClass = result.predicted_class ?? 0
  const predLabel = CLASS_LABELS[predClass] || 'Classification Result'
  const confidence = result.confidence ?? 0

  const p0 = result.probability_no_diabetes ?? result.probabilities?.no_diabetes ?? 0
  const p1 = result.probability_prediabetes ?? result.probabilities?.prediabetes ?? 0
  const p2 = result.probability_diabetes ?? result.probabilities?.diabetes ?? 0

  const predColor = CLASS_COLORS[predClass] || '#10b981'

  const displayName =
    personalInfo?.fullName ||
    result.user_name ||
    user?.full_name ||
    'Registered Health Assessee'

  const displayEmail = personalInfo?.email || user?.email || 'N/A'

  const assessmentDate =
    personalInfo?.assessmentDate ||
    (result.created_at
      ? new Date(result.created_at).toLocaleDateString('en-US', {
          year: 'numeric',
          month: 'long',
          day: 'numeric',
        })
      : new Date().toLocaleDateString('en-US', {
          year: 'numeric',
          month: 'long',
          day: 'numeric',
        }))

  const assessmentId = result.assessment_id || result.id || 'N/A'
  const predictionId = result.prediction_id || result.id || 'N/A'

  const pieData = [
    { name: 'No Diabetes (Class 0)', value: p0 },
    { name: 'Prediabetes (Class 1)', value: p1 },
    { name: 'Diabetes (Class 2)', value: p2 },
  ]
  const PIE_COLORS = ['#10b981', '#f59e0b', '#ef4444']

  const probBars = [
    {
      key: 'no_diabetes',
      label: 'No Diabetes (Class 0)',
      pct: (p0 * 100).toFixed(1),
      raw: p0.toFixed(4),
      value: p0,
      barClass: 'bg-emerald-500',
      textClass: 'text-emerald-700',
      bgClass: 'bg-emerald-50',
    },
    {
      key: 'prediabetes',
      label: 'Prediabetes (Class 1)',
      pct: (p1 * 100).toFixed(1),
      raw: p1.toFixed(4),
      value: p1,
      barClass: 'bg-amber-500',
      textClass: 'text-amber-700',
      bgClass: 'bg-amber-50',
    },
    {
      key: 'diabetes',
      label: 'Diabetes (Class 2)',
      pct: (p2 * 100).toFixed(1),
      raw: p2.toFixed(4),
      value: p2,
      barClass: 'bg-rose-500',
      textClass: 'text-rose-700',
      bgClass: 'bg-rose-50',
    },
  ]

  // Status icon based on predicted class
  const ResultIcon =
    predClass === 0 ? CheckCircle2 : predClass === 1 ? AlertTriangle : Flame

  // BMI Category calculation
  const bmiVal = inputs?.BMI !== undefined && inputs?.BMI !== null ? parseFloat(inputs.BMI) : null
  const bmiInfo = bmiVal ? getBmiCategory(bmiVal) : null

  // Longitudinal comparisons (previous assessments excluding the current one)
  const previousAssessments = historicalData.filter(
    (h) => h.assessment_id !== result.assessment_id && h.id !== result.prediction_id
  )

  return (
    <div className="py-8 px-4 sm:px-6 lg:px-8 max-w-5xl mx-auto space-y-8 print:bg-white print:py-0 print:px-0">
      <div className="space-y-8">
        
        {/* ═══════════════════════════════════════════════════════════════════ */}
        {/* TOP CONTROLS & BREADCRUMB (Hidden during Print)                      */}
        {/* ═══════════════════════════════════════════════════════════════════ */}
        <div className="flex items-center justify-between flex-wrap gap-4 print:hidden">
          <button
            type="button"
            onClick={() => navigate('/assess')}
            className="inline-flex items-center gap-2 text-sm font-semibold text-slate-600 hover:text-slate-900 transition-colors"
          >
            <ArrowLeft className="w-4 h-4" /> Start Another Assessment
          </button>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={handleDownloadPdf}
              disabled={pdfGenerating}
              className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-xs font-bold shadow-sm shadow-blue-500/20 transition-all disabled:opacity-50"
              title="Download publication-grade multi-page PDF assessment report"
            >
              <Download className="w-3.5 h-3.5 text-white" />
              <span>{pdfGenerating ? 'Generating PDF...' : 'Download PDF'}</span>
            </button>

            <button
              type="button"
              onClick={handlePrint}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-white hover:bg-slate-50 border border-slate-200 text-slate-700 text-xs font-semibold shadow-2xs transition-colors"
              title="Print assessment report using browser print"
            >
              <Printer className="w-3.5 h-3.5 text-slate-600" />
              <span>Print Report</span>
            </button>

            <div className="hidden sm:flex items-center gap-1.5 text-xs text-slate-400 font-mono pl-2 border-l border-slate-200">
              <span>Assessment #{assessmentId}</span>
              <span>•</span>
              <span>Prediction #{predictionId}</span>
            </div>
          </div>
        </div>

        {/* ═══════════════════════════════════════════════════════════════════ */}
        {/* SECTION 1: PRIMARY CLASSIFICATION HERO CARD (Advanced Summary)      */}
        {/* ═══════════════════════════════════════════════════════════════════ */}
        <div className="bg-white rounded-3xl border border-slate-200/90 shadow-sm p-6 sm:p-10 text-center relative overflow-hidden text-slate-900">
          {/* Top colored accent indicator line */}
          <div
            className="absolute top-0 left-0 right-0 h-2"
            style={{ backgroundColor: predColor }}
          />

          {/* User metadata row */}
          <div className="flex flex-wrap items-center justify-center gap-2.5 text-xs text-slate-600 mb-6">
            <span className="flex items-center gap-1.5 bg-slate-50 border border-slate-200 px-3 py-1 rounded-full">
              <User className="w-3.5 h-3.5 text-blue-600" />
              <strong className="text-slate-900">{displayName}</strong>
            </span>
            <span className="flex items-center gap-1.5 bg-slate-50 border border-slate-200 px-3 py-1 rounded-full">
              <Mail className="w-3.5 h-3.5 text-slate-500" />
              <span>{displayEmail}</span>
            </span>
            <span className="flex items-center gap-1.5 bg-slate-50 border border-slate-200 px-3 py-1 rounded-full">
              <Calendar className="w-3.5 h-3.5 text-slate-500" />
              <span>{assessmentDate}</span>
            </span>
            <span className="flex items-center gap-1.5 bg-blue-50 border border-blue-200 text-blue-700 px-3 py-1 rounded-full font-mono font-medium">
              <Database className="w-3 h-3 text-blue-600" />
              Assessment #{assessmentId} • Prediction #{predictionId}
            </span>
          </div>

          {/* Hero Classification Icon */}
          <div
            className="w-20 h-20 rounded-3xl mx-auto mb-4 flex items-center justify-center text-white shadow-lg transition-transform hover:scale-105"
            style={{ backgroundColor: predColor }}
          >
            <ResultIcon className="w-10 h-10" />
          </div>

          <div className="text-xs font-semibold tracking-wider text-slate-500 uppercase mb-1">
            Model Predicted Classification
          </div>

          <h1 className="text-3xl sm:text-4xl font-extrabold text-slate-900 mb-3 tracking-tight">
            {predLabel} (Class {predClass})
          </h1>

          {/* Harmonious Status Chip — Strictly aligned with Predicted Class (NO Contradictory Warning) */}
          <div className="flex justify-center mb-5">
            {predClass === 0 ? (
              <span className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-xs sm:text-sm font-semibold bg-emerald-50 text-emerald-800 border border-emerald-200">
                <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                <span>Class 0 — No Diabetes</span>
              </span>
            ) : predClass === 1 ? (
              <span className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-xs sm:text-sm font-semibold bg-amber-50 text-amber-900 border border-amber-200">
                <AlertTriangle className="w-4 h-4 text-amber-600" />
                <span>Class 1 — Prediabetes</span>
              </span>
            ) : (
              <span className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-xs sm:text-sm font-semibold bg-rose-50 text-rose-800 border border-rose-200">
                <Flame className="w-4 h-4 text-rose-600" />
                <span>Class 2 — Diabetes</span>
              </span>
            )}
          </div>

          {/* Primary Model Confidence Statement */}
          <p className="text-sm text-slate-600 max-w-xl mx-auto leading-relaxed">
            The model assigns a{' '}
            <strong className="text-slate-900 font-bold">
              {(confidence * 100).toFixed(1)}% posterior probability
            </strong>{' '}
            to <strong className="text-slate-900">{predLabel}</strong> based on your 21 reported clinical and lifestyle indicators.
          </p>

          {/* Pipeline & Artifact Metadata */}
          <div className="mt-4 pt-4 border-t border-slate-100 flex flex-wrap items-center justify-center gap-x-4 gap-y-1 text-[11px] text-slate-500 font-mono">
            <span>Model: Multinomial Logistic Regression</span>
            <span>•</span>
            <span>Artifact: {result.model_version || 'model_v1.0_phase5'}</span>
            <span>•</span>
            <span>Optimization: Balanced Class Weights</span>
          </div>

          <div className="mt-3">
            <span className="inline-block text-[11px] text-slate-500 bg-slate-50 border border-slate-200 px-3 py-1 rounded-md">
              Academic &amp; Research Demonstration — Not a Clinical Diagnostic Instrument
            </span>
          </div>
        </div>

        {/* ═══════════════════════════════════════════════════════════════════ */}
        {/* SECTION 2: DETAILED CLASS PROBABILITY ANALYSIS                      */}
        {/* ═══════════════════════════════════════════════════════════════════ */}
        <div className="grid md:grid-cols-2 gap-6">
          {/* Left Card: Horizontal Probability Bars */}
          <div className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-200/90 dark:border-slate-700 shadow-sm p-6 flex flex-col justify-between text-slate-900 dark:text-white transition-colors">
            <div>
              <div className="flex items-center justify-between mb-1">
                <h2 className="text-base font-bold text-slate-900 dark:text-white flex items-center gap-2">
                  <Info className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                  <span>Posterior Class Probabilities</span>
                </h2>
                <span className="text-[11px] font-mono text-slate-400 dark:text-slate-500">Sum = 100%</span>
              </div>
              <p className="text-xs text-slate-500 dark:text-slate-400 mb-5 leading-relaxed">
                Model-estimated probability distribution across the three target classes.
              </p>
            </div>

            <div className="space-y-5 my-2">
              {probBars.map(({ key, label, pct, raw, value, barClass, textClass, bgClass }) => (
                <div key={key}>
                  <div className="flex items-center justify-between text-xs mb-1.5">
                    <span className="font-semibold text-slate-700 dark:text-slate-200">{label}</span>
                    <div className="flex items-center gap-2">
                      <span className="text-[11px] font-mono text-slate-400 dark:text-slate-500">({raw})</span>
                      <span className={`font-bold ${textClass}`}>{pct}%</span>
                    </div>
                  </div>
                  <div className={`w-full rounded-full h-3 ${bgClass} dark:bg-slate-700/60`}>
                    <div
                      className={`h-3 rounded-full ${barClass} transition-all duration-700 ease-out`}
                      style={{ width: `${Math.max(4, value * 100)}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>

            <p className="text-[11px] text-slate-400 dark:text-slate-500 mt-4 pt-3 border-t border-slate-100 dark:border-slate-700/60 leading-relaxed">
              These values represent statistical model posteriors from population survey data, not clinically calibrated personal risk percentages.
            </p>
          </div>

          {/* Right Card: Donut Chart Distribution */}
          <div className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-200/90 dark:border-slate-700 shadow-sm p-6 flex flex-col justify-between text-slate-900 dark:text-white transition-colors">
            <div>
              <h2 className="text-base font-bold text-slate-900 dark:text-white mb-1 flex items-center gap-2">
                <Layers className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                <span>Probability Distribution</span>
              </h2>
              <p className="text-xs text-slate-500 dark:text-slate-400 mb-2">
                Relative share of model-estimated likelihood across classes.
              </p>
            </div>

            <div className="h-56 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    innerRadius={55}
                    outerRadius={85}
                    paddingAngle={3}
                    dataKey="value"
                  >
                    {PIE_COLORS.map((c, i) => (
                      <Cell key={i} fill={c} />
                    ))}
                  </Pie>
                  <Tooltip
                    formatter={(val) => [`${(val * 100).toFixed(1)}% (${val.toFixed(4)})`, 'Estimated Probability']}
                    contentStyle={{ borderRadius: '12px', border: '1px solid #334155', backgroundColor: '#1e293b', color: '#f8fafc', fontSize: '12px' }}
                  />
                  <Legend iconType="circle" wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <p className="text-[11px] text-center text-slate-400 dark:text-slate-500 mt-2">
              Probabilities are statistical estimates from survey data, not individualized diagnostic measurements.
            </p>
          </div>
        </div>

        {/* ═══════════════════════════════════════════════════════════════════ */}
        {/* SECTION 3: ASSESSMENT DATA SUMMARY CARD (Compact Metrics)           */}
        {/* ═══════════════════════════════════════════════════════════════════ */}
        <div className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-200/90 dark:border-slate-700 shadow-sm p-6 text-slate-900 dark:text-white transition-colors">
          <h2 className="text-base font-bold text-slate-900 dark:text-white mb-4 flex items-center gap-2">
            <FileCheck className="w-4 h-4 text-blue-600 dark:text-blue-400" />
            <span>Assessment Data Summary</span>
          </h2>

          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
            <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-900/50 border border-slate-100 dark:border-slate-700/60 text-center">
              <span className="block text-[11px] text-slate-400 dark:text-slate-400 font-medium">Total Indicators</span>
              <span className="text-base font-extrabold text-slate-900 dark:text-white">21 Features</span>
              <span className="block text-[10px] text-slate-400 dark:text-slate-500">100% Completed</span>
            </div>

            <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-900/50 border border-slate-100 dark:border-slate-700/60 text-center">
              <span className="block text-[11px] text-slate-400 dark:text-slate-400 font-medium">Feature Types</span>
              <span className="text-base font-extrabold text-slate-900 dark:text-white">14 Bin / 7 Ord</span>
              <span className="block text-[10px] text-slate-400 dark:text-slate-500">Standard BRFSS</span>
            </div>

            <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-900/50 border border-slate-100 dark:border-slate-700/60 text-center">
              <span className="block text-[11px] text-slate-400 dark:text-slate-400 font-medium">Body Mass Index</span>
              <span className="text-base font-extrabold text-slate-900 dark:text-white">
                {bmiVal ? `${bmiVal.toFixed(1)}` : 'N/A'}
              </span>
              <span className="block text-[10px] text-slate-500 dark:text-slate-400 truncate" title={bmiInfo?.label}>
                {bmiInfo?.label?.split(' ')[0] || 'kg/m²'}
              </span>
            </div>

            <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-900/50 border border-slate-100 dark:border-slate-700/60 text-center">
              <span className="block text-[11px] text-slate-400 dark:text-slate-400 font-medium">CDC Age Bracket</span>
              <span className="text-base font-extrabold text-slate-900 dark:text-white">
                {inputs?.Age ? `Cat ${inputs.Age}` : 'N/A'}
              </span>
              <span className="block text-[10px] text-slate-400 dark:text-slate-500 truncate">
                {inputs?.Age ? CDC_AGE_CATEGORIES[inputs.Age]?.bracket : 'Age bracket'}
              </span>
            </div>

            <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-900/50 border border-slate-100 dark:border-slate-700/60 text-center">
              <span className="block text-[11px] text-slate-400 dark:text-slate-400 font-medium">General Health</span>
              <span className="text-base font-extrabold text-slate-900 dark:text-white">
                {inputs?.GenHlth ? `${inputs.GenHlth}/5` : 'N/A'}
              </span>
              <span className="block text-[10px] text-slate-400 dark:text-slate-500 truncate">
                {inputs?.GenHlth ? CDC_GENHLTH_OPTIONS.find((o) => o.value === inputs.GenHlth)?.label?.split('—')[1]?.trim() : 'Self-rating'}
              </span>
            </div>

            <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-900/50 border border-slate-100 dark:border-slate-700/60 text-center">
              <span className="block text-[11px] text-slate-400 dark:text-slate-400 font-medium">Database Status</span>
              <span className="text-base font-extrabold text-emerald-600 dark:text-emerald-400">Persisted</span>
              <span className="block text-[10px] text-slate-400 dark:text-slate-500">Record #{assessmentId}</span>
            </div>
          </div>
        </div>

        {/* ═══════════════════════════════════════════════════════════════════ */}
        {/* SECTION 4: COMPLETE HEALTH ASSESSMENT DETAILS (21 Features)        */}
        {/* ═══════════════════════════════════════════════════════════════════ */}
        <div className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-200/90 dark:border-slate-700 shadow-sm p-6 sm:p-8 space-y-6 text-slate-900 dark:text-white transition-colors">
          <div className="flex items-center justify-between flex-wrap gap-3 pb-4 border-b border-slate-100 dark:border-slate-700">
            <div>
              <h2 className="text-lg font-bold text-slate-900 dark:text-white flex items-center gap-2">
                <Activity className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                <span>Your Submitted Health Information</span>
              </h2>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                Full breakdown of all 21 CDC BRFSS epidemiological indicators evaluated by the model.
              </p>
            </div>
            <button
              type="button"
              onClick={toggleAllCategories}
              className="text-xs font-semibold text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 transition-colors"
            >
              {Object.values(expandedCategories).every(Boolean) ? 'Collapse All' : 'Expand All'}
            </button>
          </div>

          <div className="space-y-4">
            {/* Category A: Personal Information */}
            <div className="border border-slate-200 dark:border-slate-700 rounded-xl overflow-hidden">
              <button
                type="button"
                onClick={() => toggleCategory('personal')}
                className="w-full flex items-center justify-between p-4 bg-slate-50/70 dark:bg-slate-900/60 hover:bg-slate-100 dark:hover:bg-slate-900 transition-colors text-left"
              >
                <div className="flex items-center gap-2.5">
                  <User className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                  <span className="text-sm font-bold text-slate-900 dark:text-white">A. Personal Information &amp; Demographics</span>
                </div>
                {expandedCategories.personal ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
              </button>

              {expandedCategories.personal && (
                <div className="p-4 grid sm:grid-cols-2 gap-4 text-xs bg-white dark:bg-slate-800">
                  <div className="p-3 rounded-lg bg-slate-50/60 dark:bg-slate-900/40 border border-slate-100 dark:border-slate-700/60">
                    <span className="text-slate-400 dark:text-slate-400 block mb-0.5">Full Name</span>
                    <span className="font-semibold text-slate-800 dark:text-white">{displayName}</span>
                  </div>
                  <div className="p-3 rounded-lg bg-slate-50/60 dark:bg-slate-900/40 border border-slate-100 dark:border-slate-700/60">
                    <span className="text-slate-400 dark:text-slate-400 block mb-0.5">Account Email</span>
                    <span className="font-semibold text-slate-800 dark:text-white">{displayEmail}</span>
                  </div>
                  <div className="p-3 rounded-lg bg-slate-50/60 dark:bg-slate-900/40 border border-slate-100 dark:border-slate-700/60">
                    <span className="text-slate-400 dark:text-slate-400 block mb-0.5">Chronological Age</span>
                    <span className="font-semibold text-slate-800 dark:text-white">
                      {personalInfo?.rawAge || (inputs?.Age ? `${inputs.Age * 5 + 15} years (derived)` : 'N/A')}
                    </span>
                  </div>
                  <div className="p-3 rounded-lg bg-slate-50/60 dark:bg-slate-900/40 border border-slate-100 dark:border-slate-700/60">
                    <span className="text-slate-400 dark:text-slate-400 block mb-0.5">CDC Age Category</span>
                    <span className="font-semibold text-slate-800 dark:text-white">
                      {inputs?.Age ? `Category ${inputs.Age}: ${CDC_AGE_CATEGORIES[inputs.Age]?.bracket || 'N/A'}` : 'N/A'}
                    </span>
                  </div>
                  <div className="p-3 rounded-lg bg-slate-50/60 dark:bg-slate-900/40 border border-slate-100 dark:border-slate-700/60">
                    <span className="text-slate-400 dark:text-slate-400 block mb-0.5">Biological Sex</span>
                    <span className="font-semibold text-slate-800 dark:text-white">
                      {inputs?.Sex === 1 ? 'Male (Value: 1)' : inputs?.Sex === 0 ? 'Female (Value: 0)' : 'N/A'}
                    </span>
                  </div>
                  <div className="p-3 rounded-lg bg-slate-50/60 dark:bg-slate-900/40 border border-slate-100 dark:border-slate-700/60">
                    <span className="text-slate-400 dark:text-slate-400 block mb-0.5">Assessment Date &amp; Time</span>
                    <span className="font-semibold text-slate-800 dark:text-white">{assessmentDate}</span>
                  </div>
                </div>
              )}
            </div>

            {/* Category B: Medical History */}
            <div className="border border-slate-200 dark:border-slate-700 rounded-xl overflow-hidden">
              <button
                type="button"
                onClick={() => toggleCategory('medical')}
                className="w-full flex items-center justify-between p-4 bg-slate-50/70 dark:bg-slate-900/60 hover:bg-slate-100 dark:hover:bg-slate-900 transition-colors text-left"
              >
                <div className="flex items-center gap-2.5">
                  <Heart className="w-4 h-4 text-rose-600 dark:text-rose-400" />
                  <span className="text-sm font-bold text-slate-900 dark:text-white">B. Medical History &amp; Vascular Indicators</span>
                </div>
                {expandedCategories.medical ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
              </button>

              {expandedCategories.medical && (
                <div className="p-4 grid sm:grid-cols-2 gap-3 text-xs bg-white dark:bg-slate-800">
                  <div className="p-3 rounded-lg bg-slate-50/60 dark:bg-slate-900/40 border border-slate-100 dark:border-slate-700/60 flex items-center justify-between">
                    <div>
                      <p className="font-semibold text-slate-800 dark:text-slate-200">High Blood Pressure (HighBP)</p>
                      <p className="text-[11px] text-slate-400 dark:text-slate-400">Ever diagnosed with hypertension</p>
                    </div>
                    <span className={`px-2.5 py-1 rounded-full font-bold ${inputs?.HighBP === 1 ? 'bg-amber-100 dark:bg-amber-950/60 text-amber-800 dark:text-amber-300' : 'bg-emerald-100 dark:bg-emerald-950/60 text-emerald-800 dark:text-emerald-300'}`}>
                      {inputs?.HighBP === 1 ? 'Yes (1)' : 'No (0)'}
                    </span>
                  </div>

                  <div className="p-3 rounded-lg bg-slate-50/60 dark:bg-slate-900/40 border border-slate-100 dark:border-slate-700/60 flex items-center justify-between">
                    <div>
                      <p className="font-semibold text-slate-800 dark:text-slate-200">High Blood Cholesterol (HighChol)</p>
                      <p className="text-[11px] text-slate-400 dark:text-slate-400">Ever diagnosed with high cholesterol</p>
                    </div>
                    <span className={`px-2.5 py-1 rounded-full font-bold ${inputs?.HighChol === 1 ? 'bg-amber-100 dark:bg-amber-950/60 text-amber-800 dark:text-amber-300' : 'bg-emerald-100 dark:bg-emerald-950/60 text-emerald-800 dark:text-emerald-300'}`}>
                      {inputs?.HighChol === 1 ? 'Yes (1)' : 'No (0)'}
                    </span>
                  </div>

                  <div className="p-3 rounded-lg bg-slate-50/60 dark:bg-slate-900/40 border border-slate-100 dark:border-slate-700/60 flex items-center justify-between">
                    <div>
                      <p className="font-semibold text-slate-800 dark:text-slate-200">Cholesterol Check (CholCheck)</p>
                      <p className="text-[11px] text-slate-400 dark:text-slate-400">Checked within the past 5 years</p>
                    </div>
                    <span className={`px-2.5 py-1 rounded-full font-bold ${inputs?.CholCheck === 1 ? 'bg-emerald-100 dark:bg-emerald-950/60 text-emerald-800 dark:text-emerald-300' : 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300'}`}>
                      {inputs?.CholCheck === 1 ? 'Yes (1)' : 'No (0)'}
                    </span>
                  </div>

                  <div className="p-3 rounded-lg bg-slate-50/60 dark:bg-slate-900/40 border border-slate-100 dark:border-slate-700/60 flex items-center justify-between">
                    <div>
                      <p className="font-semibold text-slate-800 dark:text-slate-200">History of Stroke (Stroke)</p>
                      <p className="text-[11px] text-slate-400 dark:text-slate-400">Ever diagnosed with a stroke</p>
                    </div>
                    <span className={`px-2.5 py-1 rounded-full font-bold ${inputs?.Stroke === 1 ? 'bg-rose-100 dark:bg-rose-950/60 text-rose-800 dark:text-rose-300' : 'bg-emerald-100 dark:bg-emerald-950/60 text-emerald-800 dark:text-emerald-300'}`}>
                      {inputs?.Stroke === 1 ? 'Yes (1)' : 'No (0)'}
                    </span>
                  </div>

                  <div className="p-3 rounded-lg bg-slate-50/60 dark:bg-slate-900/40 border border-slate-100 dark:border-slate-700/60 flex items-center justify-between sm:col-span-2">
                    <div>
                      <p className="font-semibold text-slate-800 dark:text-slate-200">Coronary Heart Disease or Myocardial Infarction (HeartDiseaseorAttack)</p>
                      <p className="text-[11px] text-slate-400 dark:text-slate-400">Reported history of CHD or heart attack</p>
                    </div>
                    <span className={`px-2.5 py-1 rounded-full font-bold ${inputs?.HeartDiseaseorAttack === 1 ? 'bg-rose-100 dark:bg-rose-950/60 text-rose-800 dark:text-rose-300' : 'bg-emerald-100 dark:bg-emerald-950/60 text-emerald-800 dark:text-emerald-300'}`}>
                      {inputs?.HeartDiseaseorAttack === 1 ? 'Yes (1)' : 'No (0)'}
                    </span>
                  </div>
                </div>
              )}
            </div>

            {/* Category C: Lifestyle & Body Metrics */}
            <div className="border border-slate-200 dark:border-slate-700 rounded-xl overflow-hidden">
              <button
                type="button"
                onClick={() => toggleCategory('lifestyle')}
                className="w-full flex items-center justify-between p-4 bg-slate-50/70 dark:bg-slate-900/60 hover:bg-slate-100 dark:hover:bg-slate-900 transition-colors text-left"
              >
                <div className="flex items-center gap-2.5">
                  <Scale className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
                  <span className="text-sm font-bold text-slate-900 dark:text-white">C. Lifestyle, Diet &amp; Body Metrics</span>
                </div>
                {expandedCategories.lifestyle ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
              </button>

              {expandedCategories.lifestyle && (
                <div className="p-4 space-y-4 bg-white dark:bg-slate-800 text-xs">
                  {/* BMI Feature with Detailed Interpretation & Clinical Disclaimer */}
                  <div className="p-4 rounded-xl bg-slate-50/80 dark:bg-slate-900/40 border border-slate-200 dark:border-slate-700">
                    <div className="flex items-center justify-between flex-wrap gap-2 mb-2">
                      <div>
                        <span className="font-bold text-slate-900 dark:text-white text-sm">Body Mass Index (BMI)</span>
                        <span className="text-slate-500 dark:text-slate-400 block text-[11px]">Continuous metric: weight in kg divided by height in meters squared</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-base font-extrabold text-slate-900 dark:text-white">{bmiVal ? bmiVal.toFixed(1) : '—'} kg/m²</span>
                        {bmiInfo && (
                          <span className="px-2.5 py-1 rounded-full text-[11px] font-bold bg-blue-100 dark:bg-blue-950/60 text-blue-800 dark:text-blue-300 border border-blue-200 dark:border-blue-800">
                            {bmiInfo.label}
                          </span>
                        )}
                      </div>
                    </div>
                    <p className="text-[11px] text-slate-500 dark:text-slate-400 italic leading-relaxed pt-2 border-t border-slate-200/80 dark:border-slate-700/60">
                      <strong>Scientific Disclaimer:</strong> BMI is an epidemiological screening index derived from population survey data. It does not measure body fat distribution, muscle mass composition, or individualized metabolic health.
                    </p>
                  </div>

                  <div className="grid sm:grid-cols-2 gap-3">
                    <div className="p-3 rounded-lg bg-slate-50/60 dark:bg-slate-900/40 border border-slate-100 dark:border-slate-700/60 flex items-center justify-between">
                      <div>
                        <p className="font-semibold text-slate-800 dark:text-slate-200">Lifetime Smoking (Smoker)</p>
                        <p className="text-[11px] text-slate-400 dark:text-slate-400">≥100 cigarettes in lifetime</p>
                      </div>
                      <span className={`px-2.5 py-1 rounded-full font-bold ${inputs?.Smoker === 1 ? 'bg-amber-100 dark:bg-amber-950/60 text-amber-800 dark:text-amber-300' : 'bg-emerald-100 dark:bg-emerald-950/60 text-emerald-800 dark:text-emerald-300'}`}>
                        {inputs?.Smoker === 1 ? 'Yes (1)' : 'No (0)'}
                      </span>
                    </div>

                    <div className="p-3 rounded-lg bg-slate-50/60 dark:bg-slate-900/40 border border-slate-100 dark:border-slate-700/60 flex items-center justify-between">
                      <div>
                        <p className="font-semibold text-slate-800 dark:text-slate-200">Physical Activity (PhysActivity)</p>
                        <p className="text-[11px] text-slate-400 dark:text-slate-400">Exercise in past 30 days outside work</p>
                      </div>
                      <span className={`px-2.5 py-1 rounded-full font-bold ${inputs?.PhysActivity === 1 ? 'bg-emerald-100 dark:bg-emerald-950/60 text-emerald-800 dark:text-emerald-300' : 'bg-amber-100 dark:bg-amber-950/60 text-amber-800 dark:text-amber-300'}`}>
                        {inputs?.PhysActivity === 1 ? 'Yes (1)' : 'No (0)'}
                      </span>
                    </div>

                    <div className="p-3 rounded-lg bg-slate-50/60 dark:bg-slate-900/40 border border-slate-100 dark:border-slate-700/60 flex items-center justify-between">
                      <div>
                        <p className="font-semibold text-slate-800 dark:text-slate-200">Daily Fruit Intake (Fruits)</p>
                        <p className="text-[11px] text-slate-400 dark:text-slate-400">Consumes fruit ≥1 time/day</p>
                      </div>
                      <span className={`px-2.5 py-1 rounded-full font-bold ${inputs?.Fruits === 1 ? 'bg-emerald-100 dark:bg-emerald-950/60 text-emerald-800 dark:text-emerald-300' : 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300'}`}>
                        {inputs?.Fruits === 1 ? 'Yes (1)' : 'No (0)'}
                      </span>
                    </div>

                    <div className="p-3 rounded-lg bg-slate-50/60 dark:bg-slate-900/40 border border-slate-100 dark:border-slate-700/60 flex items-center justify-between">
                      <div>
                        <p className="font-semibold text-slate-800 dark:text-slate-200">Daily Vegetable Intake (Veggies)</p>
                        <p className="text-[11px] text-slate-400 dark:text-slate-400">Consumes vegetables ≥1 time/day</p>
                      </div>
                      <span className={`px-2.5 py-1 rounded-full font-bold ${inputs?.Veggies === 1 ? 'bg-emerald-100 dark:bg-emerald-950/60 text-emerald-800 dark:text-emerald-300' : 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300'}`}>
                        {inputs?.Veggies === 1 ? 'Yes (1)' : 'No (0)'}
                      </span>
                    </div>

                    <div className="p-3 rounded-lg bg-slate-50/60 dark:bg-slate-900/40 border border-slate-100 dark:border-slate-700/60 flex items-center justify-between sm:col-span-2">
                      <div>
                        <p className="font-semibold text-slate-800 dark:text-slate-200">Heavy Alcohol Consumption (HvyAlcoholConsump)</p>
                        <p className="text-[11px] text-slate-400 dark:text-slate-400">Men &gt;14 drinks/week; Women &gt;7 drinks/week</p>
                      </div>
                      <span className={`px-2.5 py-1 rounded-full font-bold ${inputs?.HvyAlcoholConsump === 1 ? 'bg-rose-100 dark:bg-rose-950/60 text-rose-800 dark:text-rose-300' : 'bg-emerald-100 dark:bg-emerald-950/60 text-emerald-800 dark:text-emerald-300'}`}>
                        {inputs?.HvyAlcoholConsump === 1 ? 'Yes (1)' : 'No (0)'}
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Category D: General Health & Well-being */}
            <div className="border border-slate-200 dark:border-slate-700 rounded-xl overflow-hidden">
              <button
                type="button"
                onClick={() => toggleCategory('general')}
                className="w-full flex items-center justify-between p-4 bg-slate-50/70 dark:bg-slate-900/60 hover:bg-slate-100 dark:hover:bg-slate-900 transition-colors text-left"
              >
                <div className="flex items-center gap-2.5">
                  <Shield className="w-4 h-4 text-purple-600 dark:text-purple-400" />
                  <span className="text-sm font-bold text-slate-900 dark:text-white">D. General Health &amp; Physical Well-being</span>
                </div>
                {expandedCategories.general ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
              </button>

              {expandedCategories.general && (
                <div className="p-4 grid sm:grid-cols-2 gap-3 text-xs bg-white dark:bg-slate-800">
                  <div className="p-3 rounded-lg bg-slate-50/60 dark:bg-slate-900/40 border border-slate-100 dark:border-slate-700/60">
                    <span className="text-slate-400 dark:text-slate-400 block mb-0.5">Self-Rated General Health (GenHlth)</span>
                    <span className="font-bold text-slate-800 dark:text-white text-sm">
                      {inputs?.GenHlth ? CDC_GENHLTH_OPTIONS.find((o) => o.value === inputs.GenHlth)?.label || `Rating ${inputs.GenHlth}` : 'N/A'}
                    </span>
                    <span className="text-[11px] text-slate-400 dark:text-slate-500 block mt-0.5">1=Excellent to 5=Poor scale</span>
                  </div>

                  <div className="p-3 rounded-lg bg-slate-50/60 dark:bg-slate-900/40 border border-slate-100 dark:border-slate-700/60">
                    <span className="text-slate-400 dark:text-slate-400 block mb-0.5">Physical Illness or Injury Days (PhysHlth)</span>
                    <span className="font-bold text-slate-800 dark:text-white text-sm">
                      {inputs?.PhysHlth !== undefined ? `${inputs.PhysHlth} days` : 'N/A'}
                    </span>
                    <span className="text-[11px] text-slate-400 dark:text-slate-500 block mt-0.5">Reported out of the past 30 days</span>
                  </div>

                  <div className="p-3 rounded-lg bg-slate-50/60 dark:bg-slate-900/40 border border-slate-100 dark:border-slate-700/60">
                    <span className="text-slate-400 dark:text-slate-400 block mb-0.5">Poor Mental Health Days (MentHlth)</span>
                    <span className="font-bold text-slate-800 dark:text-white text-sm">
                      {inputs?.MentHlth !== undefined ? `${inputs.MentHlth} days` : 'N/A'}
                    </span>
                    <span className="text-[11px] text-slate-400 dark:text-slate-500 block mt-0.5">Stress, depression, emotional challenges in past 30 days</span>
                  </div>

                  <div className="p-3 rounded-lg bg-slate-50/60 dark:bg-slate-900/40 border border-slate-100 dark:border-slate-700/60 flex items-center justify-between">
                    <div>
                      <p className="font-semibold text-slate-800 dark:text-slate-200">Difficulty Walking (DiffWalk)</p>
                      <p className="text-[11px] text-slate-400 dark:text-slate-400">Serious difficulty walking or climbing stairs</p>
                    </div>
                    <span className={`px-2.5 py-1 rounded-full font-bold ${inputs?.DiffWalk === 1 ? 'bg-amber-100 dark:bg-amber-950/60 text-amber-800 dark:text-amber-300' : 'bg-emerald-100 dark:bg-emerald-950/60 text-emerald-800 dark:text-emerald-300'}`}>
                      {inputs?.DiffWalk === 1 ? 'Yes (1)' : 'No (0)'}
                    </span>
                  </div>
                </div>
              )}
            </div>

            {/* Category E: Healthcare Access & Demographics */}
            <div className="border border-slate-200 dark:border-slate-700 rounded-xl overflow-hidden">
              <button
                type="button"
                onClick={() => toggleCategory('demographics')}
                className="w-full flex items-center justify-between p-4 bg-slate-50/70 dark:bg-slate-900/60 hover:bg-slate-100 dark:hover:bg-slate-900 transition-colors text-left"
              >
                <div className="flex items-center gap-2.5">
                  <FileText className="w-4 h-4 text-teal-600 dark:text-teal-400" />
                  <span className="text-sm font-bold text-slate-900 dark:text-white">E. Healthcare Access &amp; Socioeconomic Indicators</span>
                </div>
                {expandedCategories.demographics ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
              </button>

              {expandedCategories.demographics && (
                <div className="p-4 grid sm:grid-cols-2 gap-3 text-xs bg-white dark:bg-slate-800">
                  <div className="p-3 rounded-lg bg-slate-50/60 dark:bg-slate-900/40 border border-slate-100 dark:border-slate-700/60 flex items-center justify-between">
                    <div>
                      <p className="font-semibold text-slate-800 dark:text-slate-200">Healthcare Coverage (AnyHealthcare)</p>
                      <p className="text-[11px] text-slate-400 dark:text-slate-400">Any health insurance coverage</p>
                    </div>
                    <span className={`px-2.5 py-1 rounded-full font-bold ${inputs?.AnyHealthcare === 1 ? 'bg-emerald-100 dark:bg-emerald-950/60 text-emerald-800 dark:text-emerald-300' : 'bg-rose-100 dark:bg-rose-950/60 text-rose-800 dark:text-rose-300'}`}>
                      {inputs?.AnyHealthcare === 1 ? 'Yes (1)' : 'No (0)'}
                    </span>
                  </div>

                  <div className="p-3 rounded-lg bg-slate-50/60 dark:bg-slate-900/40 border border-slate-100 dark:border-slate-700/60 flex items-center justify-between">
                    <div>
                      <p className="font-semibold text-slate-800 dark:text-slate-200">Cost Barrier to Doctor (NoDocbcCost)</p>
                      <p className="text-[11px] text-slate-400 dark:text-slate-400">Could not see doctor due to cost in past 12 mo</p>
                    </div>
                    <span className={`px-2.5 py-1 rounded-full font-bold ${inputs?.NoDocbcCost === 1 ? 'bg-amber-100 dark:bg-amber-950/60 text-amber-800 dark:text-amber-300' : 'bg-emerald-100 dark:bg-emerald-950/60 text-emerald-800 dark:text-emerald-300'}`}>
                      {inputs?.NoDocbcCost === 1 ? 'Yes (1)' : 'No (0)'}
                    </span>
                  </div>

                  <div className="p-3 rounded-lg bg-slate-50/60 dark:bg-slate-900/40 border border-slate-100 dark:border-slate-700/60">
                    <span className="text-slate-400 dark:text-slate-400 block mb-0.5">Highest Education Level Completed (Education)</span>
                    <span className="font-semibold text-slate-800 dark:text-white">
                      {inputs?.Education ? CDC_EDUCATION_OPTIONS.find((o) => o.value === inputs.Education)?.label || `Level ${inputs.Education}` : 'N/A'}
                    </span>
                  </div>

                  <div className="p-3 rounded-lg bg-slate-50/60 dark:bg-slate-900/40 border border-slate-100 dark:border-slate-700/60">
                    <span className="text-slate-400 dark:text-slate-400 block mb-0.5">Annual Household Income Bracket (Income)</span>
                    <span className="font-semibold text-slate-800 dark:text-white">
                      {inputs?.Income ? CDC_INCOME_OPTIONS.find((o) => o.value === inputs.Income)?.label || `Bracket ${inputs.Income}` : 'N/A'}
                    </span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* ═══════════════════════════════════════════════════════════════════ */}
        {/* SECTION 5: HEALTH INDICATOR INTERPRETATION                           */}
        {/* ═══════════════════════════════════════════════════════════════════ */}
        <div className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-200/90 dark:border-slate-700 shadow-sm p-6 sm:p-8 space-y-4">
          <h2 className="text-base font-bold text-slate-900 dark:text-white flex items-center gap-2">
            <HelpCircle className="w-5 h-5 text-blue-600 dark:text-blue-400" />
            <span>Understanding Your Health Indicators</span>
          </h2>
          <p className="text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
            The 21 variables evaluated in this assessment represent validated epidemiological risk correlates utilized in public health surveillance. The scientific mechanisms linking these clusters to metabolic status include:
          </p>

          <div className="grid sm:grid-cols-2 gap-4 text-xs">
            <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-900/50 border border-slate-200/80 dark:border-slate-700">
              <h3 className="font-bold text-slate-900 dark:text-white mb-1 flex items-center gap-1.5">
                <Heart className="w-3.5 h-3.5 text-rose-600 dark:text-rose-400" />
                <span>Cardiovascular &amp; Lipid Markers</span>
              </h3>
              <p className="text-slate-600 dark:text-slate-300 leading-relaxed">
                Hypertension and hypercholesterolemia frequently co-occur with metabolic syndrome and peripheral insulin resistance. Chronic microvascular resistance diminishes tissue perfusion, accelerating cardiometabolic strain.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-900/50 border border-slate-200/80 dark:border-slate-700">
              <h3 className="font-bold text-slate-900 dark:text-white mb-1 flex items-center gap-1.5">
                <Scale className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400" />
                <span>Metabolic Load &amp; Lifestyle Dynamics</span>
              </h3>
              <p className="text-slate-600 dark:text-slate-300 leading-relaxed">
                Regular physical activity promotes non-insulin-dependent glucose uptake into skeletal muscle. Elevated adiposity combined with low physical activity is correlated with chronic low-grade systemic inflammation.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-900/50 border border-slate-200/80 dark:border-slate-700">
              <h3 className="font-bold text-slate-900 dark:text-white mb-1 flex items-center gap-1.5">
                <Shield className="w-3.5 h-3.5 text-purple-600 dark:text-purple-400" />
                <span>Subjective Health &amp; Functional Mobility</span>
              </h3>
              <p className="text-slate-600 dark:text-slate-300 leading-relaxed">
                Self-rated general health and functional mobility limitations (such as difficulty walking) serve as integrative multi-system indicators strongly predictive of cumulative chronic disease burden in population studies.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-900/50 border border-slate-200/80 dark:border-slate-700">
              <h3 className="font-bold text-slate-900 dark:text-white mb-1 flex items-center gap-1.5">
                <User className="w-3.5 h-3.5 text-teal-600 dark:text-teal-400" />
                <span>Demographic &amp; Healthcare Determinants</span>
              </h3>
              <p className="text-slate-600 dark:text-slate-300 leading-relaxed">
                Chronological age reflects cumulative physiological exposure, while healthcare access and socioeconomic resources influence access to regular screening, nutritional support, and early preventative interventions.
              </p>
            </div>
          </div>

          <p className="text-[11px] text-slate-400 dark:text-slate-500 italic pt-2">
            * Note: These descriptions explain population-level epidemiological associations observed across hundreds of thousands of CDC survey participants. They do not claim direct individual causality or substitute for diagnostic clinical assessments.
          </p>
        </div>

        {/* ═══════════════════════════════════════════════════════════════════ */}
        {/* SECTION 6: MODEL PERFORMANCE & RESEARCH INFORMATION                  */}
        {/* ═══════════════════════════════════════════════════════════════════ */}
        <div className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-200/90 dark:border-slate-700 shadow-sm p-6 sm:p-8 space-y-6">
          <div>
            <h2 className="text-base font-bold text-slate-900 dark:text-white flex items-center gap-2">
              <ShieldAlert className="w-5 h-5 text-amber-500" />
              <span>Model Performance &amp; Research Information</span>
            </h2>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 leading-relaxed">
              This prediction was produced by a <strong className="text-slate-700 dark:text-slate-200">Multinomial Logistic Regression pipeline</strong> trained on 202,944 responses from the CDC Behavioral Risk Factor Surveillance System (BRFSS 2015, total 253,680 rows). All performance metrics below reflect frozen evaluation on the held-out test split of <strong className="text-slate-700 dark:text-slate-200">50,736 rows</strong>.
            </p>
          </div>

          {/* Global Evaluation Metrics */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
            <div className="p-3 rounded-xl bg-blue-50/50 dark:bg-blue-950/40 border border-blue-100 dark:border-blue-900/60 text-center">
              <span className="block text-[10px] text-slate-500 dark:text-slate-400 uppercase font-semibold">Macro F1</span>
              <span className="text-base font-extrabold text-blue-700 dark:text-blue-400">0.4246</span>
              <span className="block text-[10px] text-slate-400 dark:text-slate-500">Phase 5 Benchmark</span>
            </div>
            <div className="p-3 rounded-xl bg-blue-50/50 dark:bg-blue-950/40 border border-blue-100 dark:border-blue-900/60 text-center">
              <span className="block text-[10px] text-slate-500 dark:text-slate-400 uppercase font-semibold">Weighted F1</span>
              <span className="text-base font-extrabold text-blue-700 dark:text-blue-400">0.7187</span>
              <span className="block text-[10px] text-slate-400 dark:text-slate-500">Class Weighted</span>
            </div>
            <div className="p-3 rounded-xl bg-blue-50/50 dark:bg-blue-950/40 border border-blue-100 dark:border-blue-900/60 text-center">
              <span className="block text-[10px] text-slate-500 dark:text-slate-400 uppercase font-semibold">Balanced Acc</span>
              <span className="text-base font-extrabold text-blue-700 dark:text-blue-400">0.5172</span>
              <span className="block text-[10px] text-slate-400 dark:text-slate-500">Multi-class</span>
            </div>
            <div className="p-3 rounded-xl bg-blue-50/50 dark:bg-blue-950/40 border border-blue-100 dark:border-blue-900/60 text-center">
              <span className="block text-[10px] text-slate-500 dark:text-slate-400 uppercase font-semibold">Overall Acc</span>
              <span className="text-base font-extrabold text-blue-700 dark:text-blue-400">0.6433</span>
              <span className="block text-[10px] text-slate-400 dark:text-slate-500">Held-out Test</span>
            </div>
            <div className="p-3 rounded-xl bg-blue-50/50 dark:bg-blue-950/40 border border-blue-100 dark:border-blue-900/60 text-center">
              <span className="block text-[10px] text-slate-500 dark:text-slate-400 uppercase font-semibold">ROC-AUC Macro</span>
              <span className="text-base font-extrabold text-blue-700 dark:text-blue-400">0.7707</span>
              <span className="block text-[10px] text-slate-400 dark:text-slate-500">One-vs-Rest</span>
            </div>
            <div className="p-3 rounded-xl bg-blue-50/50 dark:bg-blue-950/40 border border-blue-100 dark:border-blue-900/60 text-center">
              <span className="block text-[10px] text-slate-500 dark:text-slate-400 uppercase font-semibold">ROC-AUC Wtd</span>
              <span className="text-base font-extrabold text-blue-700 dark:text-blue-400">0.8112</span>
              <span className="block text-[10px] text-slate-400 dark:text-slate-500">Class Weighted</span>
            </div>
          </div>

          {/* Class-wise Benchmark Table */}
          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left">
              <thead>
                <tr className="border-b border-slate-200 dark:border-slate-700 text-slate-400 dark:text-slate-400 font-semibold uppercase tracking-wider text-[10px]">
                  <th className="py-2.5 px-3">Target Class</th>
                  <th className="py-2.5 px-3">Precision</th>
                  <th className="py-2.5 px-3">Recall</th>
                  <th className="py-2.5 px-3">F1-Score</th>
                  <th className="py-2.5 px-3">Dataset Prevalence</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-700/60 text-slate-700 dark:text-slate-200">
                <tr className="hover:bg-slate-50/60 dark:hover:bg-slate-700/40">
                  <td className="py-2.5 px-3 font-semibold text-emerald-700 dark:text-emerald-400 flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-emerald-500" />
                    <span>Class 0: No Diabetes</span>
                  </td>
                  <td className="py-2.5 px-3 font-mono">0.8809</td>
                  <td className="py-2.5 px-3 font-mono">0.6974</td>
                  <td className="py-2.5 px-3 font-mono font-bold text-slate-900 dark:text-white">0.7785</td>
                  <td className="py-2.5 px-3 text-slate-500 dark:text-slate-400">84.7% (213,703 rows)</td>
                </tr>
                <tr className="hover:bg-slate-50/60 dark:hover:bg-slate-700/40 bg-amber-50/30 dark:bg-amber-950/20">
                  <td className="py-2.5 px-3 font-semibold text-amber-700 dark:text-amber-400 flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-amber-500" />
                    <span>Class 1: Prediabetes</span>
                  </td>
                  <td className="py-2.5 px-3 font-mono font-bold text-amber-900 dark:text-amber-300">0.0307</td>
                  <td className="py-2.5 px-3 font-mono font-bold text-amber-900 dark:text-amber-300">0.3089</td>
                  <td className="py-2.5 px-3 font-mono font-bold text-amber-900 dark:text-amber-300">0.0558</td>
                  <td className="py-2.5 px-3 text-amber-800 dark:text-amber-300 font-medium">~1.8% (4,631 rows — Rare)</td>
                </tr>
                <tr className="hover:bg-slate-50/60 dark:hover:bg-slate-700/40">
                  <td className="py-2.5 px-3 font-semibold text-rose-700 dark:text-rose-400 flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-rose-500" />
                    <span>Class 2: Diabetes</span>
                  </td>
                  <td className="py-2.5 px-3 font-mono">0.3621</td>
                  <td className="py-2.5 px-3 font-mono">0.5453</td>
                  <td className="py-2.5 px-3 font-mono font-bold text-slate-900 dark:text-white">0.4354</td>
                  <td className="py-2.5 px-3 text-slate-500 dark:text-slate-400">13.5% (35,346 rows)</td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* Highlighted Prediabetes Limitation Disclosure */}
          <div className="p-4 rounded-xl bg-amber-50/80 dark:bg-amber-950/50 border border-amber-200 dark:border-amber-800 text-amber-900 dark:text-amber-200 text-xs sm:text-sm">
            <div className="flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 text-amber-600 dark:text-amber-400 flex-shrink-0 mt-0.5" />
              <div>
                <p className="font-bold text-amber-950 dark:text-amber-100 mb-1">
                  Important Prediabetes Model Limitation Notice:
                </p>
                <p className="leading-relaxed text-amber-900/90 dark:text-amber-200 text-xs">
                  Due to extreme class imbalance in general population survey data (where prediabetes represents only ~1.8% of cases), Class 1 exhibits a precision of <strong>0.0307</strong> and recall of <strong>0.3089</strong> (F1: <strong>0.0558</strong>). The model was trained with balanced class weights to maximize sensitivity to early metabolic risk signals, but predictions in this bracket carry significant statistical uncertainty. Any elevated risk indication must be evaluated through formal clinical laboratory assays (such as Fasting Plasma Glucose or HbA1c).
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* ═══════════════════════════════════════════════════════════════════ */}
        {/* SECTION 7: EXPLAINABILITY (How to Interpret This Prediction)         */}
        {/* ═══════════════════════════════════════════════════════════════════ */}
        <div className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-200/90 dark:border-slate-700 shadow-sm p-6 sm:p-8 space-y-4">
          <h2 className="text-base font-bold text-slate-900 dark:text-white flex items-center gap-2">
            <Info className="w-5 h-5 text-blue-600 dark:text-blue-400" />
            <span>How to Interpret This Prediction</span>
          </h2>

          <div className="text-xs sm:text-sm text-slate-600 dark:text-slate-300 space-y-3 leading-relaxed">
            <p>
              <strong className="text-slate-900 dark:text-white">Mathematical Formulation:</strong> The deployed Multinomial Logistic Regression pipeline maps your 21 standardized survey indicators through learned coefficient vectors to compute three log-odds scores (logits). The model then applies the <em>softmax activation function</em>:
            </p>

            <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-900 font-mono text-center text-xs text-slate-800 dark:text-slate-200 border border-slate-200 dark:border-slate-700 overflow-x-auto">
              P(Class = k | X) = exp(β_k · X) / Σ [exp(β_j · X)] &nbsp;&nbsp;(for k ∈ &#123;0, 1, 2&#125;)
            </div>

            <p>
              <strong className="text-slate-900 dark:text-white">Argmax Decision Rule:</strong> The assigned categorical result (<strong className="text-slate-900 dark:text-white">{predLabel}</strong>) corresponds to the class holding the highest calculated posterior probability ({(confidence * 100).toFixed(1)}%).
            </p>

            <p>
              <strong className="text-slate-900 dark:text-white">Statistical Association vs. Clinical Causality:</strong> Machine learning algorithms identify correlational patterns across population survey cohorts. A higher probability score highlights elevated statistical resemblance to survey respondents who reported diagnosed diabetes, but it does <em>not</em> prove direct physiological causality or clinical disease presence in an individual patient.
            </p>
          </div>
        </div>

        {/* ═══════════════════════════════════════════════════════════════════ */}
        {/* SECTION 7B: AI FEATURE CONTRIBUTION BREAKDOWN (Phase 6.20 XAI)      */}
        {/* ═══════════════════════════════════════════════════════════════════ */}
        <FeatureContributionCard
          features={inputs}
          predictedClass={result?.predicted_class}
        />

        {/* ═══════════════════════════════════════════════════════════════════ */}
        {/* SECTION 8: LONGITUDINAL HISTORICAL COMPARISON                       */}
        {/* ═══════════════════════════════════════════════════════════════════ */}
        <div className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-200/90 dark:border-slate-700 shadow-sm p-6 sm:p-8 space-y-4">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <div>
              <h2 className="text-base font-bold text-slate-900 dark:text-white flex items-center gap-2">
                <Clock className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                <span>Compare Previous Assessments</span>
              </h2>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                Longitudinal progression of your health indicators and model classifications over time.
              </p>
            </div>
            <Link
              to="/history"
              className="text-xs font-semibold text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 flex items-center gap-1 transition-colors"
            >
              <span>Full History ({historicalData.length})</span>
            </Link>
          </div>

          {previousAssessments.length === 0 ? (
            <div className="p-6 rounded-xl bg-slate-50 dark:bg-slate-900/50 border border-slate-200 dark:border-slate-700 text-center text-xs text-slate-500 dark:text-slate-400">
              <Activity className="w-8 h-8 text-blue-400 mx-auto mb-2 opacity-60" />
              <p className="font-semibold text-slate-700 dark:text-slate-200">This is your first recorded health assessment.</p>
              <p className="mt-1 max-w-md mx-auto text-slate-500 dark:text-slate-400 leading-relaxed">
                As you complete future evaluations over time, this section will track changes in your submitted indicators, BMI trajectory, and model classification probabilities.
              </p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-xs text-left">
                <thead>
                  <tr className="border-b border-slate-200 dark:border-slate-700 text-slate-400 dark:text-slate-400 font-semibold uppercase tracking-wider text-[10px]">
                    <th className="py-2.5 px-3">Assessment Date</th>
                    <th className="py-2.5 px-3">Classification</th>
                    <th className="py-2.5 px-3">Confidence</th>
                    <th className="py-2.5 px-3">BMI Metric</th>
                    <th className="py-2.5 px-3">Gen Health</th>
                    <th className="py-2.5 px-3 text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 dark:divide-slate-700/60 text-slate-700 dark:text-slate-300">
                  {/* Current Assessment Row */}
                  <tr className="bg-blue-50/40 dark:bg-blue-950/40 font-semibold">
                    <td className="py-2.5 px-3">
                      <span className="flex items-center gap-1.5 text-blue-700 dark:text-blue-300">
                        <span className="w-2 h-2 rounded-full bg-blue-600 animate-pulse" />
                        <span>Current Assessment</span>
                      </span>
                    </td>
                    <td className="py-2.5 px-3 font-bold" style={{ color: predColor }}>
                      {predLabel}
                    </td>
                    <td className="py-2.5 px-3 font-mono">{(confidence * 100).toFixed(1)}%</td>
                    <td className="py-2.5 px-3 font-mono">
                      {bmiVal ? `${bmiVal.toFixed(1)} kg/m²` : '—'}
                    </td>
                    <td className="py-2.5 px-3">
                      {inputs?.GenHlth ? `Rating ${inputs.GenHlth}/5` : '—'}
                    </td>
                    <td className="py-2.5 px-3 text-right text-slate-400 dark:text-slate-400 font-normal">Active</td>
                  </tr>

                  {/* Previous Assessment Rows */}
                  {previousAssessments.slice(0, 4).map((prev) => {
                    const prevBmi = prev.assessment?.BMI
                    const bmiDelta = bmiVal && prevBmi ? (bmiVal - prevBmi).toFixed(1) : null
                    const pClass = prev.predicted_class ?? 0
                    const pColor = CLASS_COLORS[pClass] || '#64748b'

                    return (
                      <tr key={prev.id} className="hover:bg-slate-50/60 dark:hover:bg-slate-700/40">
                        <td className="py-2.5 px-3 text-slate-500 dark:text-slate-400">
                          {new Date(prev.created_at).toLocaleDateString('en-US', {
                            year: 'numeric',
                            month: 'short',
                            day: 'numeric',
                          })}
                        </td>
                        <td className="py-2.5 px-3 font-semibold" style={{ color: pColor }}>
                          {CLASS_LABELS[pClass] || prev.predicted_label}
                        </td>
                        <td className="py-2.5 px-3 font-mono">{(prev.confidence * 100).toFixed(1)}%</td>
                        <td className="py-2.5 px-3 font-mono">
                          {prevBmi ? (
                            <span className="flex items-center gap-1">
                              <span>{prevBmi.toFixed(1)}</span>
                              {bmiDelta && (
                                <span className={`text-[10px] ${parseFloat(bmiDelta) > 0 ? 'text-amber-600 dark:text-amber-400' : parseFloat(bmiDelta) < 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-slate-400'}`}>
                                  ({parseFloat(bmiDelta) > 0 ? `+${bmiDelta}` : bmiDelta})
                                </span>
                              )}
                            </span>
                          ) : (
                            '—'
                          )}
                        </td>
                        <td className="py-2.5 px-3 text-slate-500 dark:text-slate-400">
                          {prev.assessment?.GenHlth ? `Rating ${prev.assessment.GenHlth}/5` : '—'}
                        </td>
                        <td className="py-2.5 px-3 text-right">
                          <button
                            type="button"
                            onClick={() => {
                              setResult(prev)
                              setInputs(prev.assessment)
                              window.scrollTo({ top: 0, behavior: 'smooth' })
                            }}
                            className="text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 font-semibold"
                          >
                            Load Result
                          </button>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* ═══════════════════════════════════════════════════════════════════ */}
        {/* SECTION 9: FUNCTIONAL ACTIONS BAR                                   */}
        {/* ═══════════════════════════════════════════════════════════════════ */}
        <div className="flex flex-wrap items-center justify-center gap-3 pt-4 print:hidden">
          <button
            type="button"
            onClick={handleDownloadPdf}
            disabled={pdfGenerating}
            className="px-5 py-3 rounded-xl bg-blue-600 hover:bg-blue-700 text-white text-xs sm:text-sm font-bold flex items-center gap-2 transition-all shadow-md shadow-blue-500/20 disabled:opacity-50"
          >
            <Download className="w-4 h-4" />
            <span>{pdfGenerating ? 'Generating PDF...' : 'Download Report (PDF)'}</span>
          </button>

          <button
            type="button"
            onClick={handlePrint}
            className="px-5 py-3 rounded-xl bg-white dark:bg-slate-800 hover:bg-slate-50 dark:hover:bg-slate-700 border border-slate-300 dark:border-slate-700 text-slate-700 dark:text-slate-200 text-xs sm:text-sm font-semibold flex items-center gap-2 transition-all shadow-xs"
          >
            <Printer className="w-4 h-4 text-slate-600 dark:text-slate-400" />
            <span>Print Assessment Report</span>
          </button>

          <button
            type="button"
            onClick={() => navigate('/assess')}
            className="px-5 py-3 rounded-xl bg-white dark:bg-slate-800 hover:bg-slate-50 dark:hover:bg-slate-700 border border-slate-300 dark:border-slate-700 text-slate-700 dark:text-slate-200 text-xs sm:text-sm font-semibold flex items-center gap-2 transition-all shadow-xs"
          >
            <RefreshCw className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
            <span>Start New Assessment</span>
          </button>

          <button
            type="button"
            onClick={() => navigate('/history')}
            className="px-5 py-3 rounded-xl bg-white dark:bg-slate-800 hover:bg-slate-50 dark:hover:bg-slate-700 border border-slate-300 dark:border-slate-700 text-slate-700 dark:text-slate-200 text-xs sm:text-sm font-semibold flex items-center gap-2 transition-all shadow-xs"
          >
            <Clock className="w-4 h-4 text-purple-600 dark:text-purple-400" />
            <span>View History</span>
          </button>

          <button
            type="button"
            onClick={() => navigate('/dashboard')}
            className="px-5 py-3 rounded-xl bg-white dark:bg-slate-800 hover:bg-slate-50 dark:hover:bg-slate-700 border border-slate-300 dark:border-slate-700 text-slate-700 dark:text-slate-200 text-xs sm:text-sm font-semibold flex items-center gap-2 transition-all shadow-xs"
          >
            <LayoutDashboard className="w-4 h-4 text-slate-600 dark:text-slate-400" />
            <span>Dashboard</span>
          </button>
        </div>

        {/* ═══════════════════════════════════════════════════════════════════ */}
        {/* SECTION 10: MANDATORY RESEARCH & MEDICAL DISCLAIMER BANNER          */}
        {/* ═══════════════════════════════════════════════════════════════════ */}
        <DisclaimerBanner />
      </div>
    </div>
  )
}
