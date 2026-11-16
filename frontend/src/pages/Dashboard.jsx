/**
 * src/pages/Dashboard.jsx
 * ========================
 * Professional Personal Health Analytics Dashboard for DiabetesRisk AI (Phase 6.15).
 * 
 * Features:
 * 1. User welcome and account overview with quick actions
 * 2. Date range filter toolbar (Last 7d, 30d, 90d, 6m, All time, Custom range)
 * 3. 6 Personal analytics KPI cards with real database records
 * 4. Latest Assessment & Model Classification summary card
 * 5. Historical Prediction Class Distribution interactive Recharts Donut chart
 * 6. Longitudinal BMI Trajectory Line/Area chart with clinical screening thresholds
 * 7. Self-reported General Health rating trajectory chart (CDC scale 1-5)
 * 8. Key Health Indicator longitudinal comparison table (9 indicators)
 * 9. Assessment activity timeline with View Result and Download PDF actions
 * 10. Model Performance Research Benchmark (Phase 5 frozen evaluation metrics)
 * 11. Multi-page Downloadable Personal Health Analytics PDF Report
 * 12. Complete light/dark theme support & responsive layout
 */

import { useEffect, useState, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  Radar,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
  AreaChart,
  Area,
  LineChart,
  Line,
  CartesianGrid,
  ReferenceLine,
  PieChart,
  Pie,
  Legend,
} from 'recharts'
import {
  Activity,
  PlusCircle,
  Clock,
  Calendar,
  Database,
  ArrowRight,
  TrendingUp,
  TrendingDown,
  CheckCircle2,
  AlertTriangle,
  Flame,
  User as UserIcon,
  RefreshCw,
  Scale,
  PieChart as PieIcon,
  HelpCircle,
  Download,
  Filter,
  Eye,
  FileText,
  HeartPulse,
  Heart,
  Shield,
  Info,
  ChevronDown,
  ChevronUp,
  Minus,
  Sparkles,
} from 'lucide-react'
import {
  fetchAnalyticsOverview,
  fetchPredictionDistribution,
  fetchBmiTrends,
  fetchHealthTrends,
  fetchIndicatorComparison,
  fetchAssessmentTimeline,
  fetchAssessment,
  fetchAssessmentPrediction,
  fetchHealthInsights,
} from '../services/api'
import { useAuth } from '../context/AuthContext'
import MetricCard from '../components/MetricCard'
import RiskBadge from '../components/RiskBadge'
import DisclaimerBanner from '../components/DisclaimerBanner'
import HealthInsightsSection from '../components/HealthInsightsSection'
import {
  CLASS_LABELS,
  CLASS_COLORS,
  getBmiCategory,
  CDC_GENHLTH_OPTIONS,
} from '../utils/cdcFeatures'
import {
  generateAssessmentReportPdf,
  generatePersonalAnalyticsReportPdf,
} from '../utils/generatePdfReport'

// Frozen Phase 5 test-set metrics (CDC BRFSS held-out evaluation, 50,736 rows)
const PHASE5_METRICS = {
  accuracy: 0.6433,
  balanced_accuracy: 0.5172,
  macro_f1: 0.4246,
  weighted_f1: 0.7187,
  roc_auc_ovr_macro: 0.7707,
  roc_auc_ovr_weighted: 0.8112,
  cohen_kappa: 0.2695,
  matthews_cc: 0.2765,
}

const PHASE5_CLASS_METRICS = {
  0: { label: 'Class 0 — No Diabetes', precision: 0.9501, recall: 0.6606, f1: 0.7793, color: '#10b981' },
  1: { label: 'Class 1 — Prediabetes', precision: 0.0307, recall: 0.3089, f1: 0.0558, color: '#f59e0b' },
  2: { label: 'Class 2 — Diabetes', precision: 0.3520, recall: 0.5823, f1: 0.4387, color: '#ef4444' },
}

const MODEL_COMPARISON = [
  { model: 'Logistic Reg.', macro_f1: 0.4091, bal_acc: 0.5091, roc_auc: 0.7658 },
  { model: 'Decision Tree', macro_f1: 0.3227, bal_acc: 0.4451, roc_auc: 0.5992 },
  { model: 'Random Forest', macro_f1: 0.3819, bal_acc: 0.4756, roc_auc: 0.7501 },
  { model: 'XGBoost', macro_f1: 0.3985, bal_acc: 0.4961, roc_auc: 0.7549 },
]

const RADAR_DATA = [
  { metric: 'Accuracy', value: Math.round(PHASE5_METRICS.accuracy * 100) },
  { metric: 'Bal.Acc.', value: Math.round(PHASE5_METRICS.balanced_accuracy * 100) },
  { metric: 'Macro F1', value: Math.round(PHASE5_METRICS.macro_f1 * 100) },
  { metric: 'ROC-AUC', value: Math.round(PHASE5_METRICS.roc_auc_ovr_macro * 100) },
  { metric: 'Weighted F1', value: Math.round(PHASE5_METRICS.weighted_f1 * 100) },
]

// Date filter presets
const DATE_PRESETS = [
  { label: 'All Time', days: null },
  { label: 'Last 7 Days', days: 7 },
  { label: 'Last 30 Days', days: 30 },
  { label: 'Last 90 Days', days: 90 },
  { label: 'Last 6 Months', days: 180 },
  { label: 'Custom', days: 'custom' },
]

/**
 * Custom Tooltip for BMI Longitudinal Trend Chart
 */
function BmiCustomTooltip({ active, payload }) {
  if (active && payload && payload.length) {
    const data = payload[0].payload
    return (
      <div className="bg-white/95 dark:bg-slate-800/95 backdrop-blur-sm p-3 rounded-xl border border-slate-200 dark:border-slate-700 shadow-lg text-xs space-y-1">
        <p className="font-semibold text-slate-800 dark:text-slate-100">{data.full_date || data.fullDate}</p>
        <p className="text-slate-600 dark:text-slate-300">
          BMI: <span className="font-mono font-bold text-blue-600 dark:text-blue-400 text-sm">{data.bmi}</span>
        </p>
        <p className="text-[11px] font-medium text-slate-500 dark:text-slate-400">
          Category: <span className="font-semibold text-slate-700 dark:text-slate-200">{data.category}</span>
        </p>
      </div>
    )
  }
  return null
}

/**
 * Custom Tooltip for General Health Trend Chart
 */
function HealthCustomTooltip({ active, payload }) {
  if (active && payload && payload.length) {
    const data = payload[0].payload
    return (
      <div className="bg-white/95 dark:bg-slate-800/95 backdrop-blur-sm p-3 rounded-xl border border-slate-200 dark:border-slate-700 shadow-lg text-xs space-y-1">
        <p className="font-semibold text-slate-800 dark:text-slate-100">{data.full_date}</p>
        <p className="text-slate-600 dark:text-slate-300">
          Rating: <span className="font-bold text-teal-600 dark:text-teal-400 text-sm">{data.gen_hlth_label}</span>
        </p>
        <p className="text-[11px] text-slate-500 dark:text-slate-400">
          Physical Health: {data.phys_hlth} days | Mental Health: {data.ment_hlth} days
        </p>
      </div>
    )
  }
  return null
}

/**
 * Custom Tooltip for Prediction Distribution Donut Chart
 */
function DonutCustomTooltip({ active, payload }) {
  if (active && payload && payload.length) {
    const data = payload[0]
    return (
      <div className="bg-white/95 dark:bg-slate-800/95 backdrop-blur-sm p-3 rounded-xl border border-slate-200 dark:border-slate-700 shadow-lg text-xs space-y-1">
        <p className="font-bold" style={{ color: data.payload.fill }}>
          {data.name}
        </p>
        <p className="text-slate-700 dark:text-slate-200 font-mono">
          Count: <span className="font-bold">{data.value}</span> ({data.payload.percentage}%)
        </p>
      </div>
    )
  }
  return null
}

export default function Dashboard() {
  const navigate = useNavigate()
  const { user } = useAuth()

  // State: Analytics Data
  const [overview, setOverview] = useState(null)
  const [distribution, setDistribution] = useState(null)
  const [bmiTrends, setBmiTrends] = useState({ points: [], summary: {} })
  const [healthTrends, setHealthTrends] = useState({ points: [], summary: {} })
  const [indicators, setIndicators] = useState({ total_records: 0, records: [] })
  const [timeline, setTimeline] = useState({ total: 0, items: [] })
  const [insightsData, setInsightsData] = useState(null)

  // State: Loading & Refreshing
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [pdfGenerating, setPdfGenerating] = useState(false)
  const [downloadingAssessmentId, setDownloadingAssessmentId] = useState(null)

  // State: Date Filtering
  const [activePreset, setActivePreset] = useState('All Time')
  const [customStartDate, setCustomStartDate] = useState('')
  const [customEndDate, setCustomEndDate] = useState('')
  const [appliedStartDate, setAppliedStartDate] = useState(null)
  const [appliedEndDate, setAppliedEndDate] = useState(null)

  // State: Accordion expand/collapse for Indicators
  const [indicatorsExpanded, setIndicatorsExpanded] = useState(true)

  // Calculate start/end dates from preset
  const handlePresetSelect = (preset) => {
    setActivePreset(preset.label)
    if (preset.days === null) {
      setAppliedStartDate(null)
      setAppliedEndDate(null)
    } else if (preset.days === 'custom') {
      // Don't auto-apply custom until user clicks Apply
    } else {
      const end = new Date()
      const start = new Date()
      start.setDate(end.getDate() - preset.days)
      setAppliedStartDate(start.toISOString().split('T')[0])
      setAppliedEndDate(end.toISOString().split('T')[0])
    }
  }

  const handleApplyCustomDate = (e) => {
    e.preventDefault()
    if (!customStartDate) return
    setAppliedStartDate(customStartDate)
    setAppliedEndDate(customEndDate || null)
  }

  // Load all analytics data
  const loadData = () => {
    setRefreshing(true)
    Promise.all([
      fetchAnalyticsOverview(appliedStartDate, appliedEndDate).catch(() => null),
      fetchPredictionDistribution(appliedStartDate, appliedEndDate).catch(() => null),
      fetchBmiTrends(appliedStartDate, appliedEndDate).catch(() => ({ points: [], summary: {} })),
      fetchHealthTrends(appliedStartDate, appliedEndDate).catch(() => ({ points: [], summary: {} })),
      fetchIndicatorComparison(appliedStartDate, appliedEndDate).catch(() => ({ total_records: 0, records: [] })),
      fetchAssessmentTimeline(1, 10, appliedStartDate, appliedEndDate).catch(() => ({ total: 0, items: [] })),
      fetchHealthInsights().catch(() => null),
    ])
      .then(([ov, dist, bmi, hlth, ind, time, ins]) => {
        if (ov) setOverview(ov)
        if (dist) setDistribution(dist)
        if (bmi) setBmiTrends(bmi)
        if (hlth) setHealthTrends(hlth)
        if (ind) setIndicators(ind)
        if (time) setTimeline(time)
        if (ins) setInsightsData(ins)
      })
      .finally(() => {
        setLoading(false)
        setRefreshing(false)
      })
  }

  useEffect(() => {
    loadData()
  }, [appliedStartDate, appliedEndDate])

  // Donut chart formatted data
  const donutData = useMemo(() => {
    const counts = distribution?.counts || overview?.class_distribution || { '0': 0, '1': 0, '2': 0 }
    const total = (counts['0'] || 0) + (counts['1'] || 0) + (counts['2'] || 0)
    if (total === 0) return []
    return [
      {
        name: 'No Diabetes (Class 0)',
        value: counts['0'] || 0,
        percentage: total > 0 ? (((counts['0'] || 0) / total) * 100).toFixed(1) : '0',
        fill: '#10b981',
      },
      {
        name: 'Prediabetes (Class 1)',
        value: counts['1'] || 0,
        percentage: total > 0 ? (((counts['1'] || 0) / total) * 100).toFixed(1) : '0',
        fill: '#f59e0b',
      },
      {
        name: 'Diabetes (Class 2)',
        value: counts['2'] || 0,
        percentage: total > 0 ? (((counts['2'] || 0) / total) * 100).toFixed(1) : '0',
        fill: '#ef4444',
      },
    ]
  }, [distribution, overview])

  // Handler: Download Comprehensive Personal Analytics PDF
  const handleDownloadPersonalReport = () => {
    setPdfGenerating(true)
    try {
      generatePersonalAnalyticsReportPdf({
        user,
        overview,
        bmiTrends,
        healthTrends,
        indicators,
        timeline,
      })
    } catch (err) {
      console.error('Failed to generate personal analytics PDF report:', err)
    } finally {
      setPdfGenerating(false)
    }
  }

  // Handler: Download Single Assessment PDF from Timeline
  const handleDownloadSinglePdf = async (assessmentId) => {
    setDownloadingAssessmentId(assessmentId)
    try {
      const [inputs, pred] = await Promise.all([
        fetchAssessment(assessmentId),
        fetchAssessmentPrediction(assessmentId),
      ])
      generateAssessmentReportPdf({
        result: pred,
        inputs: inputs,
        personalInfo: {
          fullName: user?.full_name,
          email: user?.email,
          assessmentDate: new Date(inputs.created_at).toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'long',
            day: 'numeric',
          }),
        },
        user,
      })
    } catch (err) {
      console.error('Failed to download assessment report:', err)
    } finally {
      setDownloadingAssessmentId(null)
    }
  }

  const hasAssessments = overview && overview.total_assessments > 0

  return (
    <div className="py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto space-y-8">

        {/* ═══════════════════════════════════════════════════════════════════ */}
        {/* 1. Welcome & Personal Health Analytics Banner                       */}
        {/* ═══════════════════════════════════════════════════════════════════ */}
        <div className="bg-white dark:bg-slate-800 rounded-3xl border border-slate-200/90 dark:border-slate-700 shadow-sm p-6 sm:p-8 flex flex-col md:flex-row md:items-center justify-between gap-6 transition-colors">
          <div className="space-y-2">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-50 dark:bg-blue-900/40 border border-blue-200 dark:border-blue-700/60 text-blue-700 dark:text-blue-300 text-xs font-semibold">
              <Sparkles className="w-3.5 h-3.5" />
              <span>Personal Health Analytics Dashboard</span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 dark:text-white tracking-tight">
              Welcome back, {user?.full_name || 'Patient'}
            </h1>
            <p className="text-sm text-slate-600 dark:text-slate-300 max-w-2xl">
              Track your longitudinal metabolic health assessments, review model-predicted risk classifications,
              analyze BMI and well-being trajectories, or run a new health risk assessment.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <button
              onClick={() => navigate('/assess')}
              className="px-5 py-3 rounded-xl bg-blue-600 hover:bg-blue-700 text-white text-sm font-bold flex items-center gap-2 transition-all shadow-md shadow-blue-500/20"
            >
              <PlusCircle className="w-4 h-4" />
              New Health Assessment
            </button>
            <button
              onClick={handleDownloadPersonalReport}
              disabled={pdfGenerating || !hasAssessments}
              className="px-4 py-3 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 hover:bg-slate-50 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-200 text-sm font-semibold flex items-center gap-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              title="Download Comprehensive Personal Analytics PDF Report"
            >
              <Download className={`w-4 h-4 ${pdfGenerating ? 'animate-bounce' : ''}`} />
              <span>Analytics Report</span>
            </button>
            <button
              onClick={loadData}
              disabled={refreshing}
              className="p-3 rounded-xl border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors"
              title="Refresh Dashboard Data"
            >
              <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
            </button>
          </div>
        </div>

        {/* ═══════════════════════════════════════════════════════════════════ */}
        {/* 2. Date Range Filter Toolbar (Section 11)                           */}
        {/* ═══════════════════════════════════════════════════════════════════ */}
        <div className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-200/80 dark:border-slate-700 p-4 shadow-xs flex flex-col md:flex-row md:items-center justify-between gap-4 transition-colors">
          <div className="flex items-center gap-2 text-xs font-semibold text-slate-600 dark:text-slate-300">
            <Filter className="w-4 h-4 text-blue-600 dark:text-blue-400" />
            <span>Time Horizon:</span>
            <div className="flex flex-wrap gap-1.5 ml-1">
              {DATE_PRESETS.map((p) => {
                const isActive = activePreset === p.label
                return (
                  <button
                    key={p.label}
                    onClick={() => handlePresetSelect(p)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                      isActive
                        ? 'bg-blue-600 text-white shadow-xs'
                        : 'bg-slate-100 dark:bg-slate-700/60 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700'
                    }`}
                  >
                    {p.label}
                  </button>
                )
              })}
            </div>
          </div>

          {activePreset === 'Custom' && (
            <form onSubmit={handleApplyCustomDate} className="flex items-center gap-2 text-xs">
              <input
                type="date"
                value={customStartDate}
                onChange={(e) => setCustomStartDate(e.target.value)}
                className="px-2.5 py-1.5 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 text-slate-800 dark:text-slate-100 text-xs focus:ring-1 focus:ring-blue-500"
                required
              />
              <span className="text-slate-400">to</span>
              <input
                type="date"
                value={customEndDate}
                onChange={(e) => setCustomEndDate(e.target.value)}
                className="px-2.5 py-1.5 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 text-slate-800 dark:text-slate-100 text-xs focus:ring-1 focus:ring-blue-500"
              />
              <button
                type="submit"
                className="px-3 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-700 text-white text-xs font-bold transition-colors"
              >
                Apply
              </button>
            </form>
          )}

          {appliedStartDate && (
            <div className="text-[11px] text-slate-500 dark:text-slate-400">
              Showing records from <span className="font-semibold text-slate-700 dark:text-slate-200">{appliedStartDate}</span>
              {appliedEndDate && <> to <span className="font-semibold text-slate-700 dark:text-slate-200">{appliedEndDate}</span></>}
            </div>
          )}
        </div>

        {/* ═══════════════════════════════════════════════════════════════════ */}
        {/* 3. Personal Analytics Overview — 6 KPI Cards (Section 3)            */}
        {/* ═══════════════════════════════════════════════════════════════════ */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
          
          {/* Card 1: Total Assessments */}
          <div className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-200/80 dark:border-slate-700 p-5 shadow-xs flex flex-col justify-between">
            <div>
              <p className="text-[11px] font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                Total Assessments
              </p>
              <p className="text-2xl font-extrabold text-slate-900 dark:text-white mt-1">
                {overview?.total_assessments ?? 0}
              </p>
            </div>
            <div className="flex items-center justify-between mt-3 pt-3 border-t border-slate-100 dark:border-slate-700/60">
              <span className="text-[11px] text-slate-500 dark:text-slate-400">
                {overview?.assessments_this_month ? `+${overview.assessments_this_month} this month` : 'All time records'}
              </span>
              <Database className="w-4 h-4 text-blue-500" />
            </div>
          </div>

          {/* Card 2: Latest Predicted Class */}
          <div className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-200/80 dark:border-slate-700 p-5 shadow-xs flex flex-col justify-between">
            <div>
              <p className="text-[11px] font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                Latest Class
              </p>
              <p className="text-lg font-bold text-slate-900 dark:text-white mt-1 truncate">
                {overview?.latest_predicted_label || 'Awaiting Data'}
              </p>
              <div className="pt-1">
                {overview?.latest_predicted_class != null ? (
                  <RiskBadge predictedClass={overview.latest_predicted_class} />
                ) : (
                  <span className="text-[11px] text-slate-400">No evaluations</span>
                )}
              </div>
            </div>
            <div className="flex items-center justify-between mt-3 pt-3 border-t border-slate-100 dark:border-slate-700/60">
              <span className="text-[11px] text-slate-400 truncate">
                {overview?.latest_assessment_date || 'None recorded'}
              </span>
              <Activity className="w-4 h-4 text-teal-500" />
            </div>
          </div>

          {/* Card 3: Latest BMI */}
          <div className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-200/80 dark:border-slate-700 p-5 shadow-xs flex flex-col justify-between">
            <div>
              <p className="text-[11px] font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                Latest BMI
              </p>
              <p className="text-2xl font-extrabold text-slate-900 dark:text-white mt-1 font-mono">
                {overview?.latest_bmi != null ? overview.latest_bmi : '—'}
              </p>
              <p className="text-[11px] font-medium text-slate-600 dark:text-slate-300 mt-0.5 truncate">
                {overview?.latest_bmi_category || 'No BMI recorded'}
              </p>
            </div>
            <div className="flex items-center justify-between mt-3 pt-3 border-t border-slate-100 dark:border-slate-700/60">
              <span className="text-[10px] text-slate-400 italic">Screening classification</span>
              <Scale className="w-4 h-4 text-amber-500" />
            </div>
          </div>

          {/* Card 4: Assessment Activity */}
          <div className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-200/80 dark:border-slate-700 p-5 shadow-xs flex flex-col justify-between">
            <div>
              <p className="text-[11px] font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                Monthly Activity
              </p>
              <p className="text-2xl font-extrabold text-slate-900 dark:text-white mt-1">
                {overview?.assessments_this_month ?? 0}
              </p>
              <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5">
                Evaluations in {new Date().toLocaleString('en-US', { month: 'short' })}
              </p>
            </div>
            <div className="flex items-center justify-between mt-3 pt-3 border-t border-slate-100 dark:border-slate-700/60">
              <span className="text-[11px] text-slate-400">Activity cadence</span>
              <Calendar className="w-4 h-4 text-purple-500" />
            </div>
          </div>

          {/* Card 5: General Health (1 to 5) */}
          <div className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-200/80 dark:border-slate-700 p-5 shadow-xs flex flex-col justify-between">
            <div>
              <p className="text-[11px] font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                General Health
              </p>
              <p className="text-lg font-bold text-slate-900 dark:text-white mt-1 truncate">
                {overview?.latest_gen_hlth_label || 'Awaiting Data'}
              </p>
              <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5">
                Scale 1 (Best) to 5 (Poor)
              </p>
            </div>
            <div className="flex items-center justify-between mt-3 pt-3 border-t border-slate-100 dark:border-slate-700/60">
              <span className="text-[10px] text-slate-400 italic">Self-reported survey</span>
              <HeartPulse className="w-4 h-4 text-rose-500" />
            </div>
          </div>

          {/* Card 6: Model-Estimated Probability */}
          <div className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-200/80 dark:border-slate-700 p-5 shadow-xs flex flex-col justify-between">
            <div>
              <p className="text-[11px] font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                Model Probability
              </p>
              <p className="text-2xl font-extrabold text-slate-900 dark:text-white mt-1 font-mono">
                {overview?.latest_confidence != null
                  ? `${(overview.latest_confidence * 100).toFixed(1)}%`
                  : '—'}
              </p>
              <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5">
                Estimated class prob.
              </p>
            </div>
            <div className="flex items-center justify-between mt-3 pt-3 border-t border-slate-100 dark:border-slate-700/60">
              <span className="text-[10px] text-slate-400 italic">Non-diagnostic</span>
              <TrendingUp className="w-4 h-4 text-indigo-500" />
            </div>
          </div>

        </div>

        {/* ═══════════════════════════════════════════════════════════════════ */}
        {/* Onboarding State if 0 Assessments exist                             */}
        {/* ═══════════════════════════════════════════════════════════════════ */}
        {!hasAssessments && !loading && (
          <div className="bg-gradient-to-r from-blue-50 to-teal-50 dark:from-slate-800 dark:to-slate-800/80 rounded-3xl border border-blue-200/80 dark:border-slate-700 p-8 text-center space-y-4">
            <div className="w-16 h-16 rounded-2xl bg-blue-600 text-white flex items-center justify-center mx-auto shadow-md shadow-blue-500/20">
              <Heart className="w-8 h-8" />
            </div>
            <div className="max-w-xl mx-auto space-y-2">
              <h2 className="text-xl font-bold text-slate-900 dark:text-white">
                Begin Your Personalized Health Tracking
              </h2>
              <p className="text-sm text-slate-600 dark:text-slate-300">
                You haven't completed any health evaluations yet. Take your first assessment to calculate model predictions, track longitudinal BMI, and generate personal health analytics.
              </p>
            </div>
            <button
              onClick={() => navigate('/assess')}
              className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-blue-600 hover:bg-blue-700 text-white text-sm font-bold shadow-md shadow-blue-500/20 transition-all"
            >
              <PlusCircle className="w-4 h-4" /> Start First Assessment
            </button>
          </div>
        )}

        {/* ═══════════════════════════════════════════════════════════════════ */}
        {/* 4. Latest Assessment Summary (Section 9)                            */}
        {/* ═══════════════════════════════════════════════════════════════════ */}
        {hasAssessments && overview?.latest_assessment_id && (
          <div className="bg-white dark:bg-slate-800 rounded-3xl border border-slate-200/90 dark:border-slate-700 shadow-sm p-6 sm:p-8 space-y-6 transition-colors">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-100 dark:border-slate-700/60 pb-5">
              <div>
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-200 text-xs font-semibold mb-2">
                  <CheckCircle2 className="w-3.5 h-3.5 text-blue-600 dark:text-blue-400" />
                  <span>Latest Clinical Evaluation Summary</span>
                </div>
                <h2 className="text-xl font-bold text-slate-900 dark:text-white">
                  Assessment #{overview.latest_assessment_id} • Prediction #{overview.latest_prediction_id}
                </h2>
                <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                  Conducted on {overview.latest_assessment_date} • Saved in PostgreSQL
                </p>
              </div>

              <div className="flex items-center gap-3">
                <RiskBadge predictedClass={overview.latest_predicted_class} />
                <button
                  onClick={() => navigate(`/result?assessment_id=${overview.latest_assessment_id}`)}
                  className="px-4 py-2 rounded-xl bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 text-slate-800 dark:text-slate-100 text-xs font-bold flex items-center gap-1.5 transition-colors"
                >
                  <Eye className="w-3.5 h-3.5" /> View Result Report
                </button>
              </div>
            </div>

            {/* Probability Breakdown Progress Bars */}
            <div className="space-y-3">
              <div className="flex items-center justify-between text-xs font-semibold text-slate-700 dark:text-slate-200">
                <span>Model-Estimated Class Probabilities</span>
                <span className="text-[11px] text-slate-400 font-normal">Summing to 100%</span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {/* No Diabetes */}
                <div className="bg-slate-50 dark:bg-slate-900/50 rounded-xl p-4 border border-slate-100 dark:border-slate-700/60 space-y-2">
                  <div className="flex items-center justify-between text-xs font-semibold">
                    <span className="flex items-center gap-1.5 text-emerald-600 dark:text-emerald-400">
                      <span className="w-2.5 h-2.5 rounded-full bg-emerald-500"></span>
                      Class 0: No Diabetes
                    </span>
                    <span className="font-mono text-slate-900 dark:text-white font-bold">
                      {overview.latest_probabilities?.no_diabetes != null
                        ? `${(overview.latest_probabilities.no_diabetes * 100).toFixed(1)}%`
                        : '—'}
                    </span>
                  </div>
                  <div className="w-full bg-slate-200 dark:bg-slate-700 h-2 rounded-full overflow-hidden">
                    <div
                      className="bg-emerald-500 h-full rounded-full transition-all duration-500"
                      style={{
                        width: `${(overview.latest_probabilities?.no_diabetes || 0) * 100}%`,
                      }}
                    />
                  </div>
                </div>

                {/* Prediabetes */}
                <div className="bg-slate-50 dark:bg-slate-900/50 rounded-xl p-4 border border-slate-100 dark:border-slate-700/60 space-y-2">
                  <div className="flex items-center justify-between text-xs font-semibold">
                    <span className="flex items-center gap-1.5 text-amber-600 dark:text-amber-400">
                      <span className="w-2.5 h-2.5 rounded-full bg-amber-500"></span>
                      Class 1: Prediabetes
                    </span>
                    <span className="font-mono text-slate-900 dark:text-white font-bold">
                      {overview.latest_probabilities?.prediabetes != null
                        ? `${(overview.latest_probabilities.prediabetes * 100).toFixed(1)}%`
                        : '—'}
                    </span>
                  </div>
                  <div className="w-full bg-slate-200 dark:bg-slate-700 h-2 rounded-full overflow-hidden">
                    <div
                      className="bg-amber-500 h-full rounded-full transition-all duration-500"
                      style={{
                        width: `${(overview.latest_probabilities?.prediabetes || 0) * 100}%`,
                      }}
                    />
                  </div>
                </div>

                {/* Diabetes */}
                <div className="bg-slate-50 dark:bg-slate-900/50 rounded-xl p-4 border border-slate-100 dark:border-slate-700/60 space-y-2">
                  <div className="flex items-center justify-between text-xs font-semibold">
                    <span className="flex items-center gap-1.5 text-rose-600 dark:text-rose-400">
                      <span className="w-2.5 h-2.5 rounded-full bg-rose-500"></span>
                      Class 2: Diabetes
                    </span>
                    <span className="font-mono text-slate-900 dark:text-white font-bold">
                      {overview.latest_probabilities?.diabetes != null
                        ? `${(overview.latest_probabilities.diabetes * 100).toFixed(1)}%`
                        : '—'}
                    </span>
                  </div>
                  <div className="w-full bg-slate-200 dark:bg-slate-700 h-2 rounded-full overflow-hidden">
                    <div
                      className="bg-rose-500 h-full rounded-full transition-all duration-500"
                      style={{
                        width: `${(overview.latest_probabilities?.diabetes || 0) * 100}%`,
                      }}
                    />
                  </div>
                </div>
              </div>

              <div className="flex flex-wrap items-center justify-between text-[11px] text-slate-500 dark:text-slate-400 pt-1">
                <span>Submitted BMI: <strong className="text-slate-700 dark:text-slate-200">{overview.latest_bmi}</strong> ({overview.latest_bmi_category})</span>
                <span>Submitted General Health: <strong className="text-slate-700 dark:text-slate-200">{overview.latest_gen_hlth_label}</strong></span>
                <span className="italic">* Probabilities are mathematical estimates, not diagnostic findings.</span>
              </div>
            </div>
          </div>
        )}

        {/* ═══════════════════════════════════════════════════════════════════ */}
        {/* 5. Classification Distribution & Longitudinal BMI Trajectory        */}
        {/* ═══════════════════════════════════════════════════════════════════ */}
        <div className="grid lg:grid-cols-3 gap-6">

          {/* Left: Classification Distribution Donut Chart (Section 4) */}
          <div className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-200/90 dark:border-slate-700 shadow-sm p-6 flex flex-col justify-between transition-colors">
            <div>
              <div className="flex items-center gap-2 mb-2">
                <div className="w-8 h-8 rounded-lg bg-teal-50 dark:bg-teal-900/40 text-teal-600 dark:text-teal-300 flex items-center justify-center">
                  <PieIcon className="w-4 h-4" />
                </div>
                <div>
                  <h3 className="font-bold text-slate-900 dark:text-white text-sm">
                    Your Historical Model Classifications
                  </h3>
                  <p className="text-xs text-slate-500 dark:text-slate-400">
                    Distribution across saved predictions
                  </p>
                </div>
              </div>

              {donutData.length > 0 ? (
                <div>
                  <div className="h-56 w-full mt-2">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={donutData}
                          innerRadius={50}
                          outerRadius={80}
                          paddingAngle={4}
                          dataKey="value"
                        >
                          {donutData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.fill} />
                          ))}
                        </Pie>
                        <Tooltip content={<DonutCustomTooltip />} />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>

                  <div className="space-y-2 mt-2 pt-2 border-t border-slate-100 dark:border-slate-700/60">
                    {donutData.map((d) => (
                      <div key={d.name} className="flex items-center justify-between text-xs">
                        <span className="flex items-center gap-1.5 font-medium text-slate-700 dark:text-slate-300">
                          <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: d.fill }}></span>
                          {d.name.split(' (')[0]}
                        </span>
                        <span className="font-mono font-bold text-slate-800 dark:text-slate-200">
                          {d.value} <span className="font-normal text-slate-400">({d.percentage}%)</span>
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="py-12 px-4 text-center bg-slate-50/60 dark:bg-slate-900/40 rounded-xl border border-dashed border-slate-200 dark:border-slate-700 mt-4">
                  <PieIcon className="w-8 h-8 text-slate-300 dark:text-slate-600 mx-auto mb-2" />
                  <p className="text-xs font-semibold text-slate-600 dark:text-slate-300">No predictions recorded</p>
                  <p className="text-[11px] text-slate-400 mt-1">Complete your first assessment to populate classification breakdown.</p>
                </div>
              )}
            </div>

            <div className="border-t border-slate-100 dark:border-slate-700/60 pt-3 mt-4 text-[11px] text-slate-400">
              * Does not represent confirmed disease status or epidemiological prevalence.
            </div>
          </div>

          {/* Right: BMI Longitudinal Trend Line Chart (Section 5) */}
          <div className="lg:col-span-2 bg-white dark:bg-slate-800 rounded-2xl border border-slate-200/90 dark:border-slate-700 shadow-sm p-6 flex flex-col justify-between transition-colors">
            <div>
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-2">
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 rounded-lg bg-blue-50 dark:bg-blue-900/40 text-blue-600 dark:text-blue-300 flex items-center justify-center">
                    <Scale className="w-4 h-4" />
                  </div>
                  <div>
                    <h3 className="font-bold text-slate-900 dark:text-white text-sm">Your BMI History</h3>
                    <p className="text-xs text-slate-500 dark:text-slate-400">Chronological trajectory against clinical reference ranges</p>
                  </div>
                </div>

                {bmiTrends?.summary?.count >= 2 && (
                  <div className="flex items-center gap-2 text-xs">
                    <span className="font-semibold text-slate-600 dark:text-slate-300">
                      Latest: <strong className="font-mono text-slate-900 dark:text-white">{bmiTrends.summary.latest}</strong>
                    </span>
                    <span className="text-slate-400">•</span>
                    <span className="flex items-center gap-1 font-semibold">
                      Change:
                      {bmiTrends.summary.change > 0 ? (
                        <span className="text-rose-600 dark:text-rose-400 flex items-center font-mono">
                          +{bmiTrends.summary.change} <TrendingUp className="w-3.5 h-3.5" />
                        </span>
                      ) : bmiTrends.summary.change < 0 ? (
                        <span className="text-emerald-600 dark:text-emerald-400 flex items-center font-mono">
                          {bmiTrends.summary.change} <TrendingDown className="w-3.5 h-3.5" />
                        </span>
                      ) : (
                        <span className="text-slate-500 font-mono">0.0 (Stable)</span>
                      )}
                    </span>
                  </div>
                )}
              </div>

              {bmiTrends?.points?.length >= 2 ? (
                <div className="h-64 w-full mt-4">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={bmiTrends.points}
                      margin={{ top: 10, right: 20, left: -10, bottom: 0 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#cbd5e1" opacity={0.3} vertical={false} />
                      <XAxis
                        dataKey="date"
                        tick={{ fontSize: 11, fill: '#64748b' }}
                        stroke="#cbd5e1"
                      />
                      <YAxis
                        domain={['dataMin - 2', 'dataMax + 2']}
                        tick={{ fontSize: 11, fill: '#64748b' }}
                        stroke="#cbd5e1"
                      />
                      <Tooltip content={<BmiCustomTooltip />} />
                      <ReferenceLine
                        y={25.0}
                        stroke="#f59e0b"
                        strokeDasharray="4 4"
                        label={{ value: 'Overweight (25.0)', fill: '#f59e0b', fontSize: 10, position: 'insideTopRight' }}
                      />
                      <ReferenceLine
                        y={30.0}
                        stroke="#ef4444"
                        strokeDasharray="4 4"
                        label={{ value: 'Obese (30.0)', fill: '#ef4444', fontSize: 10, position: 'insideTopRight' }}
                      />
                      <ReferenceLine
                        y={18.5}
                        stroke="#10b981"
                        strokeDasharray="4 4"
                        label={{ value: 'Normal (18.5)', fill: '#10b981', fontSize: 10, position: 'insideBottomRight' }}
                      />
                      <Line
                        type="monotone"
                        dataKey="bmi"
                        stroke="#2563eb"
                        strokeWidth={2.5}
                        dot={{ r: 4, fill: '#1d4ed8', stroke: '#fff', strokeWidth: 2 }}
                        activeDot={{ r: 6, fill: '#2563eb', stroke: '#fff', strokeWidth: 2 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div className="py-12 px-4 text-center bg-slate-50/60 dark:bg-slate-900/40 rounded-xl border border-dashed border-slate-200 dark:border-slate-700 mt-4 flex flex-col items-center justify-center">
                  <Scale className="w-10 h-10 text-slate-300 dark:text-slate-600 mb-2" />
                  <p className="text-sm font-semibold text-slate-700 dark:text-slate-200">
                    Trajectory Requires At Least 2 Assessments
                  </p>
                  <p className="text-xs text-slate-500 dark:text-slate-400 max-w-md mt-1 mb-4">
                    Two or more evaluations are needed to plot your chronological BMI change against adult screening thresholds.
                  </p>
                  <button
                    onClick={() => navigate('/assess')}
                    className="inline-flex items-center gap-2 text-xs font-bold text-blue-600 dark:text-blue-400 hover:underline"
                  >
                    <PlusCircle className="w-3.5 h-3.5" /> Start Another Assessment
                  </button>
                </div>
              )}
            </div>

            <div className="flex flex-wrap items-center justify-between text-[11px] text-slate-400 border-t border-slate-100 dark:border-slate-700/60 pt-3 mt-4">
              <span>BMI categories are descriptive screening classifications, not individual diagnoses.</span>
              <span className="flex items-center gap-3">
                <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-emerald-500"></span> Normal (18.5–24.9)</span>
                <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-amber-500"></span> Overweight (25–29.9)</span>
                <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-rose-500"></span> Obese (≥30)</span>
              </span>
            </div>
          </div>

        </div>

        {/* ═══════════════════════════════════════════════════════════════════ */}
        {/* 6. General Health Trend Analysis (Section 6)                        */}
        {/* ═══════════════════════════════════════════════════════════════════ */}
        <div className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-200/90 dark:border-slate-700 shadow-sm p-6 space-y-4 transition-colors">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-purple-50 dark:bg-purple-900/40 text-purple-600 dark:text-purple-300 flex items-center justify-center">
                <HeartPulse className="w-4 h-4" />
              </div>
              <div>
                <h3 className="font-bold text-slate-900 dark:text-white text-sm">
                  General Health Trend (Self-Reported Survey)
                </h3>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  CDC BRFSS Ordinal Scale: 1 = Excellent, 2 = Very Good, 3 = Good, 4 = Fair, 5 = Poor
                </p>
              </div>
            </div>

            {healthTrends?.summary?.count >= 2 && (
              <div className="text-xs font-semibold text-slate-600 dark:text-slate-300">
                Latest: <span className="text-teal-600 dark:text-teal-400 font-bold">{healthTrends.summary.latest_label}</span>
                {healthTrends.summary.change !== null && (
                  <span className="ml-2 text-slate-400">
                    (Rating change: {healthTrends.summary.change > 0 ? `+${healthTrends.summary.change}` : healthTrends.summary.change})
                  </span>
                )}
              </div>
            )}
          </div>

          {healthTrends?.points?.length >= 2 ? (
            <div className="h-56 w-full mt-2">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart
                  data={healthTrends.points}
                  margin={{ top: 10, right: 20, left: -20, bottom: 0 }}
                >
                  <defs>
                    <linearGradient id="hlthGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#14b8a6" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#14b8a6" stopOpacity={0.0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#cbd5e1" opacity={0.3} vertical={false} />
                  <XAxis dataKey="date" tick={{ fontSize: 11, fill: '#64748b' }} stroke="#cbd5e1" />
                  <YAxis
                    domain={[1, 5]}
                    ticks={[1, 2, 3, 4, 5]}
                    tick={{ fontSize: 11, fill: '#64748b' }}
                    stroke="#cbd5e1"
                  />
                  <Tooltip content={<HealthCustomTooltip />} />
                  <Area
                    type="stepAfter"
                    dataKey="gen_hlth"
                    stroke="#0d9488"
                    strokeWidth={2}
                    fill="url(#hlthGradient)"
                    dot={{ r: 4, fill: '#0d9488', stroke: '#fff', strokeWidth: 2 }}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="py-8 px-4 text-center bg-slate-50/60 dark:bg-slate-900/40 rounded-xl border border-dashed border-slate-200 dark:border-slate-700">
              <p className="text-xs font-semibold text-slate-600 dark:text-slate-300">
                Requires at least 2 assessments to display general health trajectory.
              </p>
            </div>
          )}

          <div className="text-[11px] text-slate-400 border-t border-slate-100 dark:border-slate-700/60 pt-2">
            * Note: These values represent self-reported survey responses and must not be described as medically verified improvement or deterioration.
          </div>
        </div>

        {/* ═══════════════════════════════════════════════════════════════════ */}
        {/* 7. Key Health Indicator Comparison (Section 7)                      */}
        {/* ═══════════════════════════════════════════════════════════════════ */}
        <div className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-200/90 dark:border-slate-700 shadow-sm transition-colors overflow-hidden">
          <div
            onClick={() => setIndicatorsExpanded(!indicatorsExpanded)}
            className="p-6 flex items-center justify-between cursor-pointer hover:bg-slate-50/50 dark:hover:bg-slate-700/30 transition-colors"
          >
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-indigo-50 dark:bg-indigo-900/40 text-indigo-600 dark:text-indigo-300 flex items-center justify-center">
                <FileText className="w-4 h-4" />
              </div>
              <div>
                <h3 className="font-bold text-slate-900 dark:text-white text-sm">
                  Health Indicator Longitudinal Comparison
                </h3>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  Compare 9 core submitted indicators across your historical assessments
                </p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300">
                {indicators?.total_records || 0} Records
              </span>
              {indicatorsExpanded ? (
                <ChevronUp className="w-5 h-5 text-slate-400" />
              ) : (
                <ChevronDown className="w-5 h-5 text-slate-400" />
              )}
            </div>
          </div>

          {indicatorsExpanded && (
            <div className="px-6 pb-6 border-t border-slate-100 dark:border-slate-700/60">
              {indicators?.records?.length > 0 ? (
                <div className="overflow-x-auto mt-4">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="border-b border-slate-200 dark:border-slate-700 text-slate-500 dark:text-slate-400 font-semibold uppercase tracking-wider text-left">
                        <th className="pb-3 pl-2">ID</th>
                        <th className="pb-3">Date</th>
                        <th className="pb-3">BMI (Cont.)</th>
                        <th className="pb-3">High BP (Bin.)</th>
                        <th className="pb-3">High Chol (Bin.)</th>
                        <th className="pb-3">Phys Act (Bin.)</th>
                        <th className="pb-3">Smoker (Bin.)</th>
                        <th className="pb-3">Gen Health (Ord.)</th>
                        <th className="pb-3">Phys/Ment Days</th>
                        <th className="pb-3">Diff Walk (Bin.)</th>
                        <th className="pb-3 text-right pr-2">Predicted Class</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100 dark:divide-slate-700/60">
                      {indicators.records.map((r) => (
                        <tr key={r.assessment_id} className="hover:bg-slate-50/80 dark:hover:bg-slate-700/30 transition-colors">
                          <td className="py-3 pl-2 font-mono font-medium text-slate-500">#{r.assessment_id}</td>
                          <td className="py-3 text-slate-700 dark:text-slate-300 whitespace-nowrap">{r.date}</td>
                          <td className="py-3 font-mono font-bold text-slate-900 dark:text-white">
                            {r.bmi} <span className="font-normal text-[10px] text-slate-400">({r.bmi_category.split(' ')[0]})</span>
                          </td>
                          <td className="py-3">
                            <span className={`px-2 py-0.5 rounded-md text-[10px] font-bold ${
                              r.high_bp === 1 ? 'bg-amber-50 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300' : 'bg-slate-100 text-slate-600 dark:bg-slate-700 dark:text-slate-300'
                            }`}>
                              {r.high_bp === 1 ? 'Yes' : 'No'}
                            </span>
                          </td>
                          <td className="py-3">
                            <span className={`px-2 py-0.5 rounded-md text-[10px] font-bold ${
                              r.high_chol === 1 ? 'bg-amber-50 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300' : 'bg-slate-100 text-slate-600 dark:bg-slate-700 dark:text-slate-300'
                            }`}>
                              {r.high_chol === 1 ? 'Yes' : 'No'}
                            </span>
                          </td>
                          <td className="py-3">
                            <span className={`px-2 py-0.5 rounded-md text-[10px] font-bold ${
                              r.phys_activity === 1 ? 'bg-emerald-50 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300' : 'bg-slate-100 text-slate-600 dark:bg-slate-700 dark:text-slate-300'
                            }`}>
                              {r.phys_activity === 1 ? 'Active' : 'No'}
                            </span>
                          </td>
                          <td className="py-3">
                            <span className={`px-2 py-0.5 rounded-md text-[10px] font-bold ${
                              r.smoker === 1 ? 'bg-rose-50 text-rose-700 dark:bg-rose-900/30 dark:text-rose-300' : 'bg-slate-100 text-slate-600 dark:bg-slate-700 dark:text-slate-300'
                            }`}>
                              {r.smoker === 1 ? 'Yes' : 'No'}
                            </span>
                          </td>
                          <td className="py-3 text-slate-700 dark:text-slate-300">{r.gen_hlth_label}</td>
                          <td className="py-3 font-mono text-slate-600 dark:text-slate-400">
                            {r.phys_hlth}d / {r.ment_hlth}d
                          </td>
                          <td className="py-3">
                            <span className={`px-2 py-0.5 rounded-md text-[10px] font-bold ${
                              r.diff_walk === 1 ? 'bg-amber-50 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300' : 'bg-slate-100 text-slate-600 dark:bg-slate-700 dark:text-slate-300'
                            }`}>
                              {r.diff_walk === 1 ? 'Yes' : 'No'}
                            </span>
                          </td>
                          <td className="py-3 text-right pr-2">
                            {r.predicted_class != null ? (
                              <RiskBadge predictedClass={r.predicted_class} />
                            ) : (
                              <span className="text-slate-400">—</span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="text-xs text-slate-500 py-6 text-center">No assessments found to compare.</p>
              )}

              <p className="text-[11px] text-slate-400 mt-4 italic">
                * Features represent actual submitted values from CDC BRFSS survey instruments. No causal relationships are established.
              </p>
            </div>
          )}
        </div>

        {/* ═══════════════════════════════════════════════════════════════════ */}
        {/* 8. Assessment Activity Timeline (Section 8)                         */}
        {/* ═══════════════════════════════════════════════════════════════════ */}
        <div className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-200/90 dark:border-slate-700 shadow-sm p-6 space-y-4 transition-colors">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-teal-50 dark:bg-teal-900/40 text-teal-600 dark:text-teal-300 flex items-center justify-center">
                <Clock className="w-4 h-4" />
              </div>
              <div>
                <h3 className="font-bold text-slate-900 dark:text-white text-sm">
                  Assessment Activity Timeline
                </h3>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  Chronological log of completed evaluations with direct report viewing &amp; PDF download
                </p>
              </div>
            </div>

            <button
              onClick={() => navigate('/history')}
              className="text-xs font-bold text-blue-600 dark:text-blue-400 hover:underline flex items-center gap-1"
            >
              Full Assessment History <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>

          {timeline?.items?.length > 0 ? (
            <div className="space-y-3 mt-4">
              {timeline.items.map((item) => (
                <div
                  key={item.assessment_id}
                  className="bg-slate-50/70 dark:bg-slate-900/40 rounded-xl p-4 border border-slate-100 dark:border-slate-700/60 flex flex-col md:flex-row md:items-center justify-between gap-4 transition-colors"
                >
                  <div className="space-y-1">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="font-mono text-xs font-bold text-slate-700 dark:text-slate-300">
                        Assessment #{item.assessment_id}
                      </span>
                      <span className="text-slate-300 dark:text-slate-600">•</span>
                      <span className="text-xs text-slate-500 dark:text-slate-400">
                        {item.full_date}
                      </span>
                      {item.predicted_class != null && (
                        <RiskBadge predictedClass={item.predicted_class} />
                      )}
                    </div>

                    <div className="flex flex-wrap items-center gap-4 text-xs text-slate-600 dark:text-slate-300 pt-1">
                      <span>
                        BMI: <strong className="font-mono text-slate-800 dark:text-slate-100">{item.bmi}</strong> ({item.bmi_category})
                      </span>
                      <span>•</span>
                      <span>Health: <strong className="text-slate-800 dark:text-slate-100">{item.gen_hlth_label}</strong></span>
                      {item.confidence != null && (
                        <>
                          <span>•</span>
                          <span>
                            Model Probability: <strong className="font-mono text-blue-600 dark:text-blue-400">{(item.confidence * 100).toFixed(1)}%</strong>
                          </span>
                        </>
                      )}
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => navigate(`/result?assessment_id=${item.assessment_id}`)}
                      className="px-3 py-1.5 rounded-lg bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-200 text-xs font-semibold flex items-center gap-1.5 transition-colors"
                    >
                      <Eye className="w-3.5 h-3.5" /> View
                    </button>
                    <button
                      onClick={() => handleDownloadSinglePdf(item.assessment_id)}
                      disabled={downloadingAssessmentId === item.assessment_id}
                      className="px-3 py-1.5 rounded-lg bg-blue-50 dark:bg-blue-900/40 border border-blue-200 dark:border-blue-700/60 hover:bg-blue-100 dark:hover:bg-blue-900/60 text-blue-700 dark:text-blue-300 text-xs font-semibold flex items-center gap-1.5 transition-colors disabled:opacity-50"
                    >
                      <Download className={`w-3.5 h-3.5 ${downloadingAssessmentId === item.assessment_id ? 'animate-spin' : ''}`} />
                      PDF
                    </button>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="py-8 text-center bg-slate-50 dark:bg-slate-900/40 rounded-xl border border-dashed border-slate-200 dark:border-slate-700">
              <p className="text-xs font-semibold text-slate-600 dark:text-slate-300">No timeline entries found in selected window.</p>
            </div>
          )}
        </div>

        {/* ═══════════════════════════════════════════════════════════════════ */}
        {/* 9. Intelligent Health Insights & Recommendations (Phase 6.16)       */}
        {/* ═══════════════════════════════════════════════════════════════════ */}
        <HealthInsightsSection
          insightsData={insightsData}
          onStartAssessment={() => navigate('/assess')}
        />

        {/* ═══════════════════════════════════════════════════════════════════ */}
        {/* 10. Model Performance — Research Benchmark (Section 10)             */}
        {/* ═══════════════════════════════════════════════════════════════════ */}
        <div className="pt-6 border-t border-slate-200 dark:border-slate-700">
          <div className="mb-6 space-y-1">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-teal-50 dark:bg-teal-900/40 border border-teal-200 dark:border-teal-700 text-teal-700 dark:text-teal-300 text-xs font-semibold">
              <Shield className="w-3.5 h-3.5" />
              <span>Phase 5 Research Benchmark</span>
            </div>
            <h2 className="text-xl font-bold text-slate-900 dark:text-white tracking-tight">
              Model Performance — Research Benchmark
            </h2>
            <p className="text-xs text-slate-500 dark:text-slate-400 max-w-3xl">
              This section represents the frozen Phase 5 held-out evaluation on 50,736 CDC BRFSS 2015 test samples.
              These are population-level research metrics and NOT personal health predictions.
            </p>
          </div>

          {/* Key Frozen Global Metrics Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 mb-6">
            <MetricCard label="Macro F1" value={PHASE5_METRICS.macro_f1.toFixed(4)} color="amber" />
            <MetricCard label="Weighted F1" value={PHASE5_METRICS.weighted_f1.toFixed(4)} color="green" />
            <MetricCard label="Balanced Acc." value={`${(PHASE5_METRICS.balanced_accuracy * 100).toFixed(1)}%`} color="purple" />
            <MetricCard label="Accuracy" value={`${(PHASE5_METRICS.accuracy * 100).toFixed(1)}%`} color="blue" />
            <MetricCard label="ROC-AUC OvR" value={PHASE5_METRICS.roc_auc_ovr_macro.toFixed(4)} color="blue" />
            <MetricCard label="Weighted AUC" value={PHASE5_METRICS.roc_auc_ovr_weighted.toFixed(4)} color="slate" />
          </div>

          {/* Class-wise Metrics Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            {Object.entries(PHASE5_CLASS_METRICS).map(([cls, m]) => (
              <div
                key={cls}
                className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-200/90 dark:border-slate-700 p-5 shadow-xs space-y-2 transition-colors"
              >
                <div className="flex items-center justify-between">
                  <span className="font-bold text-xs" style={{ color: m.color }}>
                    {m.label}
                  </span>
                  <span className="text-[10px] font-mono text-slate-400">Class {cls}</span>
                </div>
                <div className="grid grid-cols-3 gap-2 pt-1 font-mono text-xs">
                  <div>
                    <span className="text-[10px] text-slate-400 block font-sans">Precision</span>
                    <strong className="text-slate-800 dark:text-slate-200">{m.precision.toFixed(4)}</strong>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-400 block font-sans">Recall</span>
                    <strong className="text-slate-800 dark:text-slate-200">{m.recall.toFixed(4)}</strong>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-400 block font-sans">F1-Score</span>
                    <strong className="text-slate-800 dark:text-slate-200">{m.f1.toFixed(4)}</strong>
                  </div>
                </div>
                {cls === '1' && (
                  <p className="text-[10px] text-amber-700 dark:text-amber-400 bg-amber-50 dark:bg-amber-950/40 p-2 rounded-lg mt-2">
                    * Known limitation: Class 1 exhibits low precision (0.0307) due to ~1.8% prediabetes prevalence in BRFSS survey data.
                  </p>
                )}
              </div>
            ))}
          </div>

          {/* Radar & Model Comparison Visualizations */}
          <div className="grid lg:grid-cols-2 gap-6">
            <div className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-200/90 dark:border-slate-700 shadow-sm p-6 transition-colors">
              <h3 className="font-bold text-slate-800 dark:text-slate-100 text-sm mb-3">
                Research Performance Profile (Radar)
              </h3>
              <div className="h-60 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart data={RADAR_DATA}>
                    <PolarGrid stroke="#e2e8f0" opacity={0.5} />
                    <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11, fill: '#64748b' }} />
                    <Radar name="Phase 5" dataKey="value" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.25} dot />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-200/90 dark:border-slate-700 shadow-sm p-6 transition-colors">
              <h3 className="font-bold text-slate-800 dark:text-slate-100 text-sm mb-3">
                Model Architecture Benchmark (Macro F1)
              </h3>
              <div className="h-60 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={MODEL_COMPARISON} layout="vertical" margin={{ left: 10 }}>
                    <XAxis type="number" domain={[0, 0.5]} tick={{ fontSize: 11 }} />
                    <YAxis type="category" dataKey="model" tick={{ fontSize: 11 }} width={90} />
                    <Tooltip formatter={(v) => v.toFixed(4)} />
                    <Bar dataKey="macro_f1" radius={[0, 6, 6, 0]}>
                      {MODEL_COMPARISON.map((_, i) => (
                        <Cell key={i} fill={i === 0 ? '#3b82f6' : '#94a3b8'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </div>

        {/* ═══════════════════════════════════════════════════════════════════ */}
        {/* 10. Mandatory Non-Diagnostic Research Disclaimers                   */}
        {/* ═══════════════════════════════════════════════════════════════════ */}
        <DisclaimerBanner />

      </div>
    </div>
  )
}
