/**
 * src/pages/Dashboard.jsx
 * ========================
 * User-Centric Dashboard for DiabetesRisk AI.
 * 
 * Features:
 * - Prominent "Start New Health Assessment" CTA
 * - Displays authenticated user's name and email greeting
 * - User-specific assessment metrics (total assessments, latest prediction, latest date, model confidence)
 * - Longitudinal BMI History Chart (Recharts) when >= 2 assessments exist with clinical reference thresholds
 * - Prediction Class Distribution breakdown for user's assessments
 * - User recent assessment summaries with direct navigation to /result
 * - Preserves Phase 5 test-set benchmark evaluation metrics (Macro F1 = 0.4246, ROC-AUC = 0.7707)
 * - Fully isolated: never exposes cross-user data
 * - Non-diagnostic research disclaimers
 */

import { useEffect, useState } from 'react'
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
  CartesianGrid,
  ReferenceLine,
} from 'recharts'
import {
  Activity,
  PlusCircle,
  Clock,
  Calendar,
  Database,
  ArrowRight,
  TrendingUp,
  CheckCircle2,
  AlertTriangle,
  Flame,
  User as UserIcon,
  RefreshCw,
  Scale,
  PieChart as PieIcon,
  HelpCircle,
} from 'lucide-react'
import { fetchStats, fetchPredictions, fetchAssessments } from '../services/api'
import { useAuth } from '../context/AuthContext'
import MetricCard from '../components/MetricCard'
import RiskBadge from '../components/RiskBadge'
import DisclaimerBanner from '../components/DisclaimerBanner'
import { CLASS_LABELS, CLASS_COLORS, getBmiCategory } from '../utils/cdcFeatures'

// Frozen Phase 5 test-set metrics (CDC BRFSS held-out evaluation, 50,736 rows)
const PHASE5_METRICS = {
  accuracy: 0.6433,
  balanced_accuracy: 0.5172,
  macro_f1: 0.4246,
  weighted_f1: 0.7187,
  roc_auc_ovr_macro: 0.7707,
  cohen_kappa: 0.2695,
  matthews_cc: 0.2765,
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

/**
 * Custom Tooltip for BMI Longitudinal Trend Chart
 */
function BmiCustomTooltip({ active, payload }) {
  if (active && payload && payload.length) {
    const data = payload[0].payload
    const bmiVal = data.bmi
    const info = getBmiCategory(bmiVal)
    return (
      <div className="bg-white/95 backdrop-blur-sm p-3 rounded-xl border border-slate-200 shadow-lg text-xs space-y-1">
        <p className="font-semibold text-slate-800">{data.fullDate}</p>
        <p className="text-slate-600">
          BMI: <span className="font-mono font-bold text-blue-600 text-sm">{bmiVal.toFixed(1)}</span>
        </p>
        <p className="text-[11px] font-medium text-slate-500">
          Category: <span className="font-semibold text-slate-700">{info?.category || 'Standard'}</span>
        </p>
      </div>
    )
  }
  return null
}

export default function Dashboard() {
  const navigate = useNavigate()
  const { user } = useAuth()

  const [dbStats, setDbStats] = useState(null)
  const [userPredictions, setUserPredictions] = useState([])
  const [userAssessments, setUserAssessments] = useState([])
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)

  const loadData = () => {
    setRefreshing(true)
    Promise.all([
      fetchStats().catch(() => null),
      fetchPredictions(1, 10).catch(() => ({ total: 0, items: [] })),
      fetchAssessments(1, 20).catch(() => ({ total: 0, items: [] })),
    ])
      .then(([stats, preds, assessments]) => {
        if (stats) setDbStats(stats)
        if (preds?.items) setUserPredictions(preds.items)
        if (assessments?.items) setUserAssessments(assessments.items)
      })
      .finally(() => {
        setLoading(false)
        setRefreshing(false)
      })
  }

  useEffect(() => {
    loadData()
  }, [])

  const latestPred = userPredictions.length > 0 ? userPredictions[0] : null
  const latestDate = latestPred
    ? new Date(latestPred.created_at).toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
      })
    : null

  // Prepare BMI longitudinal trajectory (chronological: oldest to newest)
  const bmiTimelineData = (userAssessments || [])
    .filter((a) => typeof a.BMI === 'number' && !isNaN(a.BMI))
    .sort((a, b) => new Date(a.created_at) - new Date(b.created_at))
    .map((a) => {
      const d = new Date(a.created_at)
      return {
        id: a.id,
        date: d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
        fullDate: d.toLocaleDateString('en-US', {
          month: 'short',
          day: 'numeric',
          year: 'numeric',
          hour: '2-digit',
          minute: '2-digit',
        }),
        bmi: parseFloat(a.BMI.toFixed(1)),
      }
    })

  // Calculate prediction class distribution
  const classDist = dbStats?.class_distribution || {}
  const countND = Number(classDist[0] || classDist['0'] || 0)
  const countPre = Number(classDist[1] || classDist['1'] || 0)
  const countDM = Number(classDist[2] || classDist['2'] || 0)
  const totalPreds = countND + countPre + countDM

  const pctND = totalPreds > 0 ? Math.round((countND / totalPreds) * 100) : 0
  const pctPre = totalPreds > 0 ? Math.round((countPre / totalPreds) * 100) : 0
  const pctDM = totalPreds > 0 ? Math.round((countDM / totalPreds) * 100) : 0

  return (
    <div className="py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Welcome & New Assessment Hero Banner */}
        <div className="bg-white rounded-3xl border border-slate-200/90 shadow-sm p-6 sm:p-8 flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div className="space-y-2">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-50 border border-blue-200 text-blue-700 text-xs font-semibold">
              <UserIcon className="w-3.5 h-3.5" />
              <span>Personal Health Profile</span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
              Welcome back, {user?.full_name || 'Patient'}
            </h1>
            <p className="text-sm text-slate-600 max-w-xl">
              Track your longitudinal metabolic health assessments, review model-predicted risk classifications,
              or run a new clinical evaluation.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <button
              onClick={() => navigate('/assess')}
              className="px-6 py-3.5 rounded-xl bg-blue-600 hover:bg-blue-700 text-white text-sm font-bold flex items-center gap-2 transition-all shadow-md shadow-blue-500/20"
            >
              <PlusCircle className="w-5 h-5" />
              New Health Assessment
            </button>
            <button
              onClick={loadData}
              disabled={refreshing}
              className="p-3 rounded-xl border border-slate-200 text-slate-600 hover:bg-slate-50 transition-colors"
              title="Refresh Stats"
            >
              <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
            </button>
          </div>
        </div>

        {/* User Specific Metrics Row */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Total Assessments */}
          <div className="bg-white rounded-2xl border border-slate-200/80 p-5 shadow-sm flex items-center justify-between">
            <div>
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Completed Assessments</p>
              <p className="text-3xl font-extrabold text-slate-900 mt-1">
                {dbStats?.total_assessments ?? userPredictions.length}
              </p>
              <p className="text-[11px] text-slate-400 mt-1">Associated with your account</p>
            </div>
            <div className="w-12 h-12 rounded-xl bg-blue-50 text-blue-600 flex items-center justify-center">
              <Database className="w-6 h-6" />
            </div>
          </div>

          {/* Latest Assessment Date */}
          <div className="bg-white rounded-2xl border border-slate-200/80 p-5 shadow-sm flex items-center justify-between">
            <div>
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Latest Evaluation</p>
              <p className="text-lg font-bold text-slate-900 mt-1 truncate">
                {latestDate || 'None recorded'}
              </p>
              <p className="text-[11px] text-slate-400 mt-1">
                {latestPred ? `Prediction #${latestPred.id}` : 'Complete first evaluation'}
              </p>
            </div>
            <div className="w-12 h-12 rounded-xl bg-teal-50 text-teal-600 flex items-center justify-center">
              <Calendar className="w-6 h-6" />
            </div>
          </div>

          {/* Latest Classification */}
          <div className="bg-white rounded-2xl border border-slate-200/80 p-5 shadow-sm flex items-center justify-between">
            <div className="space-y-1">
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Latest Predicted Class</p>
              <p className="text-lg font-bold text-slate-900">
                {latestPred ? latestPred.predicted_label : 'Awaiting Data'}
              </p>
              <div className="pt-0.5">
                {latestPred ? (
                  <RiskBadge predictedClass={latestPred.predicted_class} />
                ) : (
                  <span className="text-xs text-slate-400">No evaluations recorded</span>
                )}
              </div>
              {latestPred && (
                <p className="text-[11px] text-slate-500 font-medium pt-0.5">
                  Model-estimated probability: <strong className="font-mono text-slate-800">{(latestPred.confidence * 100).toFixed(1)}%</strong>
                </p>
              )}
            </div>
            <div
              className="w-12 h-12 rounded-xl flex items-center justify-center text-white flex-shrink-0"
              style={{
                backgroundColor: latestPred
                  ? CLASS_COLORS[latestPred.predicted_class]
                  : '#94a3b8',
              }}
            >
              {latestPred?.predicted_class === 0 ? (
                <CheckCircle2 className="w-6 h-6" />
              ) : latestPred?.predicted_class === 1 ? (
                <AlertTriangle className="w-6 h-6" />
              ) : latestPred?.predicted_class === 2 ? (
                <Flame className="w-6 h-6" />
              ) : (
                <Activity className="w-6 h-6" />
              )}
            </div>
          </div>

          {/* Latest Model Confidence / Probability */}
          <div className="bg-white rounded-2xl border border-slate-200/80 p-5 shadow-sm flex items-center justify-between">
            <div>
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Model-Estimated Probability</p>
              <p className="text-3xl font-extrabold text-slate-900 mt-1 font-mono">
                {latestPred ? `${(latestPred.confidence * 100).toFixed(1)}%` : '—'}
              </p>
              <p className="text-[11px] text-slate-400 mt-1">Model-estimated class probability</p>
            </div>
            <div className="w-12 h-12 rounded-xl bg-indigo-50 text-indigo-600 flex items-center justify-center">
              <TrendingUp className="w-6 h-6" />
            </div>
          </div>
        </div>

        <p className="text-[11px] text-slate-400 -mt-3 italic">
          * Probabilities are statistical model estimates and are not clinical diagnoses.
        </p>

        {/* ═══════════════════════════════════════════════════════════════════ */}
        {/* Longitudinal BMI Tracking & Classification Distribution             */}
        {/* ═══════════════════════════════════════════════════════════════════ */}
        <div className="grid lg:grid-cols-3 gap-6">
          {/* Left 2 Cols: Longitudinal BMI Trajectory Chart */}
          <div className="lg:col-span-2 bg-white rounded-2xl border border-slate-200/90 shadow-sm p-6 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 rounded-lg bg-blue-50 text-blue-600 flex items-center justify-center">
                    <Scale className="w-4 h-4" />
                  </div>
                  <div>
                    <h3 className="font-bold text-slate-900 text-sm">Longitudinal BMI Trajectory</h3>
                    <p className="text-xs text-slate-500">Chronological Body Mass Index across recorded assessments</p>
                  </div>
                </div>
                {bmiTimelineData.length >= 2 && (
                  <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-blue-50 text-blue-700 border border-blue-200">
                    {bmiTimelineData.length} records tracked
                  </span>
                )}
              </div>

              {bmiTimelineData.length >= 2 ? (
                <div className="h-64 w-full mt-4">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart
                      data={bmiTimelineData}
                      margin={{ top: 10, right: 20, left: -10, bottom: 0 }}
                    >
                      <defs>
                        <linearGradient id="bmiGradient" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                          <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                      <XAxis
                        dataKey="date"
                        tick={{ fontSize: 11, fill: '#64748b' }}
                        stroke="#cbd5e1"
                      />
                      <YAxis
                        domain={['dataMin - 3', 'dataMax + 3']}
                        tick={{ fontSize: 11, fill: '#64748b' }}
                        stroke="#cbd5e1"
                      />
                      <Tooltip content={<BmiCustomTooltip />} />
                      {/* Clinical Reference Lines */}
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
                      <Area
                        type="monotone"
                        dataKey="bmi"
                        stroke="#2563eb"
                        strokeWidth={2.5}
                        fillOpacity={1}
                        fill="url(#bmiGradient)"
                        dot={{ r: 4, fill: '#1d4ed8', stroke: '#fff', strokeWidth: 2 }}
                        activeDot={{ r: 6, fill: '#2563eb', stroke: '#fff', strokeWidth: 2 }}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div className="py-12 px-4 text-center bg-slate-50/60 rounded-xl border border-dashed border-slate-200 mt-4 flex flex-col items-center justify-center">
                  <Scale className="w-10 h-10 text-slate-300 mb-2" />
                  <p className="text-sm font-semibold text-slate-700">Longitudinal Trend Requires 2+ Assessments</p>
                  <p className="text-xs text-slate-500 max-w-md mt-1 mb-4">
                    At least two completed health assessments are required to calculate and plot your chronological BMI trajectory against clinical thresholds.
                  </p>
                  <button
                    onClick={() => navigate('/assess')}
                    className="inline-flex items-center gap-2 text-xs font-bold text-blue-600 hover:text-blue-800 bg-white border border-slate-200 px-3.5 py-2 rounded-lg shadow-xs hover:bg-slate-50 transition-all"
                  >
                    <PlusCircle className="w-3.5 h-3.5" /> Start Health Assessment
                  </button>
                </div>
              )}
            </div>

            <div className="flex flex-wrap items-center justify-between text-[11px] text-slate-400 border-t border-slate-100 pt-3 mt-4">
              <span>Thresholds based on WHO &amp; CDC clinical BMI standards.</span>
              <span className="flex items-center gap-3">
                <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-emerald-500"></span> Normal (18.5–24.9)</span>
                <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-amber-500"></span> Overweight (25–29.9)</span>
                <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-rose-500"></span> Obese (&ge;30)</span>
              </span>
            </div>
          </div>

          {/* Right Col: Prediction Class Distribution */}
          <div className="bg-white rounded-2xl border border-slate-200/90 shadow-sm p-6 flex flex-col justify-between">
            <div>
              <div className="flex items-center gap-2 mb-2">
                <div className="w-8 h-8 rounded-lg bg-teal-50 text-teal-600 flex items-center justify-center">
                  <PieIcon className="w-4 h-4" />
                </div>
                <div>
                  <h3 className="font-bold text-slate-900 text-sm">Classification Distribution</h3>
                  <p className="text-xs text-slate-500">Outcome breakdown across your assessments</p>
                </div>
              </div>

              {totalPreds > 0 ? (
                <div className="space-y-4 mt-5">
                  {/* Class 0: No Diabetes */}
                  <div>
                    <div className="flex items-center justify-between text-xs mb-1.5">
                      <span className="font-semibold text-slate-700 flex items-center gap-1.5">
                        <span className="w-2.5 h-2.5 rounded-full bg-emerald-500"></span>
                        No Diabetes
                      </span>
                      <span className="font-mono text-slate-500 font-bold">
                        {countND} <span className="font-normal text-slate-400">({pctND}%)</span>
                      </span>
                    </div>
                    <div className="w-full bg-slate-100 h-2.5 rounded-full overflow-hidden">
                      <div
                        className="bg-emerald-500 h-full rounded-full transition-all duration-500"
                        style={{ width: `${pctND}%` }}
                      />
                    </div>
                  </div>

                  {/* Class 1: Prediabetes */}
                  <div>
                    <div className="flex items-center justify-between text-xs mb-1.5">
                      <span className="font-semibold text-slate-700 flex items-center gap-1.5">
                        <span className="w-2.5 h-2.5 rounded-full bg-amber-500"></span>
                        Prediabetes
                      </span>
                      <span className="font-mono text-slate-500 font-bold">
                        {countPre} <span className="font-normal text-slate-400">({pctPre}%)</span>
                      </span>
                    </div>
                    <div className="w-full bg-slate-100 h-2.5 rounded-full overflow-hidden">
                      <div
                        className="bg-amber-500 h-full rounded-full transition-all duration-500"
                        style={{ width: `${pctPre}%` }}
                      />
                    </div>
                  </div>

                  {/* Class 2: Diabetes */}
                  <div>
                    <div className="flex items-center justify-between text-xs mb-1.5">
                      <span className="font-semibold text-slate-700 flex items-center gap-1.5">
                        <span className="w-2.5 h-2.5 rounded-full bg-rose-500"></span>
                        Diabetes
                      </span>
                      <span className="font-mono text-slate-500 font-bold">
                        {countDM} <span className="font-normal text-slate-400">({pctDM}%)</span>
                      </span>
                    </div>
                    <div className="w-full bg-slate-100 h-2.5 rounded-full overflow-hidden">
                      <div
                        className="bg-rose-500 h-full rounded-full transition-all duration-500"
                        style={{ width: `${pctDM}%` }}
                      />
                    </div>
                  </div>
                </div>
              ) : (
                <div className="py-8 px-4 text-center bg-slate-50/60 rounded-xl border border-dashed border-slate-200 mt-4">
                  <p className="text-xs font-semibold text-slate-600">No predictions recorded yet</p>
                  <p className="text-[11px] text-slate-400 mt-1">
                    Complete your first assessment to view class distribution.
                  </p>
                </div>
              )}
            </div>

            <div className="border-t border-slate-100 pt-3 mt-4 text-[11px] text-slate-400">
              Evaluated with frozen Phase 5 Logistic Regression model.
            </div>
          </div>
        </div>

        {/* User Recent Assessments Card */}
        <div className="bg-white rounded-2xl border border-slate-200/90 shadow-sm p-6">
          <div className="flex items-center justify-between mb-4 flex-wrap gap-2">
            <div>
              <h2 className="text-lg font-bold text-slate-900 flex items-center gap-2">
                <Clock className="w-5 h-5 text-blue-600" /> Your Recent Health Assessments
              </h2>
              <p className="text-xs text-slate-500">History records saved securely in PostgreSQL.</p>
            </div>
            <button
              onClick={() => navigate('/history')}
              className="text-xs font-bold text-blue-600 hover:text-blue-800 flex items-center gap-1"
            >
              View Full History <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>

          {userPredictions.length === 0 ? (
            <div className="text-center py-10 bg-slate-50/70 rounded-xl border border-dashed border-slate-200">
              <Activity className="w-8 h-8 mx-auto text-slate-300 mb-2" />
              <p className="text-sm font-semibold text-slate-700">No assessments recorded yet</p>
              <p className="text-xs text-slate-400 mt-0.5 mb-4">
                Start your first evaluation to see personalized predictions and historical tracking.
              </p>
              <button onClick={() => navigate('/assess')} className="btn-primary text-xs py-2 px-4">
                Start First Assessment
              </button>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-slate-100 text-slate-500 font-semibold uppercase tracking-wider text-left">
                    <th className="pb-3 pl-2">ID</th>
                    <th className="pb-3">Date</th>
                    <th className="pb-3">Predicted Class</th>
                    <th className="pb-3">Status Badge</th>
                    <th className="pb-3">Model Probability</th>
                    <th className="pb-3">Probabilities (ND / Pre / DM)</th>
                    <th className="pb-3 text-right pr-2">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {userPredictions.map((p) => {
                    const cls = p.predicted_class
                    const dateStr = new Date(p.created_at).toLocaleDateString('en-US', {
                      month: 'short',
                      day: 'numeric',
                      year: 'numeric',
                      hour: '2-digit',
                      minute: '2-digit',
                    })
                    return (
                      <tr key={p.id} className="hover:bg-slate-50/80 transition-colors">
                        <td className="py-3 pl-2 font-mono font-medium text-slate-500">#{p.id}</td>
                        <td className="py-3 text-slate-600 whitespace-nowrap">{dateStr}</td>
                        <td className="py-3">
                          <span
                            className="inline-flex items-center gap-1.5 font-bold"
                            style={{ color: CLASS_COLORS[cls] }}
                          >
                            <span
                              className="w-2 h-2 rounded-full"
                              style={{ backgroundColor: CLASS_COLORS[cls] }}
                            />
                            {CLASS_LABELS[cls]}
                          </span>
                        </td>
                        <td className="py-3">
                          <RiskBadge predictedClass={p.predicted_class} />
                        </td>
                        <td className="py-3 font-mono font-bold text-slate-800">
                          {(p.confidence * 100).toFixed(1)}%
                        </td>
                        <td className="py-3 text-slate-500 font-mono text-[11px]">
                          {(p.probability_no_diabetes * 100).toFixed(0)}% /{' '}
                          {(p.probability_prediabetes * 100).toFixed(0)}% /{' '}
                          {(p.probability_diabetes * 100).toFixed(0)}%
                        </td>
                        <td className="py-3 text-right pr-2">
                          <button
                            onClick={() =>
                              navigate(`/result?assessment_id=${p.assessment_id}`, {
                                state: {
                                  result: p,
                                  personalInfo: {
                                    fullName: user?.full_name,
                                    email: user?.email,
                                    assessmentDate: dateStr,
                                  },
                                },
                              })
                            }
                            className="text-blue-600 hover:text-blue-800 font-bold hover:underline"
                          >
                            View Result
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

        {/* Phase 5 Model Benchmark Section */}
        <div className="pt-4 border-t border-slate-200">
          <div className="mb-6">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-teal-50 border border-teal-200 text-teal-700 text-xs font-semibold mb-2">
              <Activity className="w-3.5 h-3.5" />
              <span>Phase 5 Validated Benchmarks</span>
            </div>
            <h2 className="text-xl font-bold text-slate-900 tracking-tight">
              Machine Learning Model Research Benchmarks
            </h2>
            <p className="text-xs text-slate-500 mt-1">
              One-time held-out evaluation on 50,736 CDC BRFSS test samples. Logistic Regression selected via group-aware cross-validation.
            </p>
          </div>

          {/* Key metrics grid */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-7 gap-3 mb-6">
            <MetricCard label="Accuracy" value={`${(PHASE5_METRICS.accuracy * 100).toFixed(1)}%`} color="blue" />
            <MetricCard label="Balanced Acc." value={`${(PHASE5_METRICS.balanced_accuracy * 100).toFixed(1)}%`} color="purple" />
            <MetricCard label="Macro F1" value={PHASE5_METRICS.macro_f1.toFixed(4)} color="amber" />
            <MetricCard label="Weighted F1" value={PHASE5_METRICS.weighted_f1.toFixed(4)} color="green" />
            <MetricCard label="ROC-AUC OvR" value={PHASE5_METRICS.roc_auc_ovr_macro.toFixed(4)} color="blue" />
            <MetricCard label="Cohen's κ" value={PHASE5_METRICS.cohen_kappa.toFixed(4)} color="slate" />
            <MetricCard label="MCC" value={PHASE5_METRICS.matthews_cc.toFixed(4)} color="slate" />
          </div>

          {/* Radar & Comparison Charts */}
          <div className="grid lg:grid-cols-2 gap-6">
            <div className="bg-white rounded-2xl border border-slate-200/90 shadow-sm p-6">
              <h3 className="font-bold text-slate-800 text-sm mb-3">Performance Profile (Radar)</h3>
              <div className="h-60 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart data={RADAR_DATA}>
                    <PolarGrid stroke="#e2e8f0" />
                    <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11, fill: '#64748b' }} />
                    <Radar name="Phase 5" dataKey="value" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.25} dot />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="bg-white rounded-2xl border border-slate-200/90 shadow-sm p-6">
              <h3 className="font-bold text-slate-800 text-sm mb-3">Model Comparison (Macro F1)</h3>
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

        <DisclaimerBanner />
      </div>
    </div>
  )
}
