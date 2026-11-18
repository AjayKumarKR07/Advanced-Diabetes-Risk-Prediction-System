/**
 * src/pages/Analytics.jsx
 * ========================
 * Personal Health Analytics Dashboard
 * Visualizes BMI trends, risk distribution, health indicators, and assessment timeline
 * using the existing /api/analytics/* endpoints.
 */

import { useState, useEffect } from 'react'
import {
  AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell,
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend
} from 'recharts'
import {
  TrendingUp, Activity, PieChart as PieIcon, BarChart2,
  Calendar, RefreshCw, AlertCircle, Info
} from 'lucide-react'
import {
  fetchAnalyticsOverview,
  fetchBmiTrends,
  fetchPredictionDistribution,
  fetchHealthTrends,
  fetchIndicatorComparison,
} from '../services/api'

const RISK_COLORS = {
  'No Diabetes': '#10b981',
  'Prediabetes':  '#f59e0b',
  'Diabetes':     '#ef4444',
}
const PIE_COLORS = ['#10b981', '#f59e0b', '#ef4444']

function SectionCard({ title, icon: Icon, children, className = '' }) {
  return (
    <div className={`bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden ${className}`}>
      <div className="flex items-center gap-2.5 px-5 py-4 border-b border-slate-100">
        <Icon className="w-4 h-4 text-blue-600" />
        <h2 className="text-sm font-bold text-slate-800">{title}</h2>
      </div>
      <div className="p-5">{children}</div>
    </div>
  )
}

function StatBadge({ label, value, color = 'blue' }) {
  const colors = {
    blue:  'bg-blue-50 text-blue-700 border-blue-100',
    green: 'bg-emerald-50 text-emerald-700 border-emerald-100',
    amber: 'bg-amber-50 text-amber-700 border-amber-100',
    red:   'bg-red-50 text-red-700 border-red-100',
  }
  return (
    <div className={`rounded-xl border px-4 py-3 text-center ${colors[color]}`}>
      <p className="text-xs font-medium opacity-70 mb-0.5">{label}</p>
      <p className="text-xl font-bold">{value ?? '—'}</p>
    </div>
  )
}

export default function Analytics() {
  const [overview, setOverview]       = useState(null)
  const [bmiData, setBmiData]         = useState([])
  const [distribution, setDist]       = useState([])
  const [healthTrends, setHealthTrends] = useState([])
  const [loading, setLoading]         = useState(true)
  const [error, setError]             = useState(null)
  const [refreshing, setRefreshing]   = useState(false)

  const load = async () => {
    try {
      setError(null)
      const [ov, bmi, dist, ht] = await Promise.all([
        fetchAnalyticsOverview(),
        fetchBmiTrends(),
        fetchPredictionDistribution(),
        fetchHealthTrends(),
      ])
      setOverview(ov)
      // BMI trend data
      setBmiData(
        (bmi?.bmi_trends || []).map((d) => ({
          date: d.date ? new Date(d.date).toLocaleDateString('en-IN', { month: 'short', day: 'numeric' }) : d.label,
          BMI: parseFloat(d.bmi ?? d.value ?? 0).toFixed(1),
        }))
      )
      // Prediction distribution for pie chart
      const rawDist = dist?.distribution || dist || {}
      setDist(
        Object.entries(rawDist).map(([name, value]) => ({ name, value: Number(value) }))
      )
      // General health / mental / physical trends
      setHealthTrends(
        (ht?.trends || []).map((d) => ({
          date: d.date ? new Date(d.date).toLocaleDateString('en-IN', { month: 'short', day: 'numeric' }) : d.label,
          'General Health': d.gen_hlth ?? d.general_health,
          'Mental Health':  d.ment_hlth ?? d.mental_health,
          'Physical Health': d.phys_hlth ?? d.physical_health,
        }))
      )
    } catch (e) {
      setError('Could not load analytics data. Make sure you have completed at least one assessment.')
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }

  useEffect(() => { load() }, [])

  const handleRefresh = () => { setRefreshing(true); load() }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="w-10 h-10 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
          <p className="text-sm text-slate-500">Loading analytics…</p>
        </div>
      </div>
    )
  }

  return (
    <div className="p-6 space-y-6 max-w-6xl mx-auto">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-slate-900">Health Analytics</h1>
          <p className="text-sm text-slate-500 mt-0.5">Your longitudinal health data visualized</p>
        </div>
        <button
          onClick={handleRefresh}
          disabled={refreshing}
          className="flex items-center gap-2 px-3 py-2 text-xs font-semibold text-slate-600 border border-slate-200 rounded-xl hover:bg-slate-50 transition-colors disabled:opacity-50"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${refreshing ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="flex items-center gap-3 p-4 bg-amber-50 border border-amber-200 rounded-xl">
          <AlertCircle className="w-5 h-5 text-amber-500 flex-shrink-0" />
          <p className="text-sm text-amber-700">{error}</p>
        </div>
      )}

      {/* Overview Stats */}
      {overview && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <StatBadge label="Total Assessments" value={overview.total_assessments} color="blue" />
          <StatBadge label="Avg BMI" value={overview.average_bmi?.toFixed(1)} color="green" />
          <StatBadge label="Latest Class" value={overview.latest_prediction_class ?? '—'} color="amber" />
          <StatBadge label="Avg Risk %" value={overview.average_probability != null ? `${(overview.average_probability * 100).toFixed(1)}%` : '—'} color="red" />
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* BMI Trend */}
        <SectionCard title="BMI Trend Over Time" icon={TrendingUp}>
          {bmiData.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <AreaChart data={bmiData}>
                <defs>
                  <linearGradient id="bmiGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor="#3b82f6" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="date" tick={{ fontSize: 11, fill: '#94a3b8' }} />
                <YAxis tick={{ fontSize: 11, fill: '#94a3b8' }} domain={['auto', 'auto']} />
                <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8 }} />
                <Area type="monotone" dataKey="BMI" stroke="#3b82f6" strokeWidth={2} fill="url(#bmiGrad)" dot={{ r: 4, fill: '#3b82f6' }} />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <EmptyChart message="No BMI data yet — complete an assessment to see your trend." />
          )}
          <div className="mt-3 flex items-start gap-2 p-2.5 bg-blue-50 rounded-lg">
            <Info className="w-3.5 h-3.5 text-blue-500 flex-shrink-0 mt-0.5" />
            <p className="text-[11px] text-blue-700">Healthy BMI range: 18.5–24.9. BMI is one of 21 CDC BRFSS indicators used in the model.</p>
          </div>
        </SectionCard>

        {/* Prediction Distribution */}
        <SectionCard title="Risk Class Distribution" icon={PieIcon}>
          {distribution.length > 0 ? (
            <div className="flex flex-col items-center">
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie
                    data={distribution}
                    cx="50%"
                    cy="50%"
                    innerRadius={55}
                    outerRadius={85}
                    paddingAngle={3}
                    dataKey="value"
                  >
                    {distribution.map((_, i) => (
                      <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8 }} />
                </PieChart>
              </ResponsiveContainer>
              <div className="flex flex-wrap justify-center gap-3 mt-2">
                {distribution.map((d, i) => (
                  <div key={d.name} className="flex items-center gap-1.5 text-xs text-slate-600">
                    <span className="w-2.5 h-2.5 rounded-full" style={{ background: PIE_COLORS[i % PIE_COLORS.length] }} />
                    <span className="font-medium">{d.name}</span>
                    <span className="text-slate-400">({d.value})</span>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <EmptyChart message="No prediction data yet. Run at least one assessment." />
          )}
        </SectionCard>

        {/* Health Trends */}
        <SectionCard title="Health Score Trends" icon={Activity} className="lg:col-span-2">
          {healthTrends.length > 0 ? (
            <ResponsiveContainer width="100%" height={240}>
              <LineChart data={healthTrends}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="date" tick={{ fontSize: 11, fill: '#94a3b8' }} />
                <YAxis tick={{ fontSize: 11, fill: '#94a3b8' }} domain={[1, 5]} />
                <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8 }} />
                <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 12 }} />
                <Line type="monotone" dataKey="General Health"  stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} />
                <Line type="monotone" dataKey="Mental Health"   stroke="#8b5cf6" strokeWidth={2} dot={{ r: 3 }} />
                <Line type="monotone" dataKey="Physical Health" stroke="#10b981" strokeWidth={2} dot={{ r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <EmptyChart message="Complete more assessments to see your health score trends over time." />
          )}
          <p className="text-[11px] text-slate-400 mt-2">Scale: 1 = Excellent, 5 = Poor (CDC BRFSS self-reported scale)</p>
        </SectionCard>
      </div>

      {/* Disclaimer */}
      <div className="flex items-start gap-3 p-4 bg-slate-50 border border-slate-200 rounded-xl">
        <AlertCircle className="w-4 h-4 text-slate-400 flex-shrink-0 mt-0.5" />
        <p className="text-xs text-slate-500">
          All analytics are derived from CDC BRFSS 2015 self-reported data processed by an academic Logistic Regression model (ROC-AUC 0.7707).
          This is a research prototype only — not a clinical diagnostic tool. Consult a healthcare professional for medical advice.
        </p>
      </div>
    </div>
  )
}

function EmptyChart({ message }) {
  return (
    <div className="h-48 flex flex-col items-center justify-center gap-2 text-center">
      <BarChart2 className="w-8 h-8 text-slate-200" />
      <p className="text-xs text-slate-400 max-w-48">{message}</p>
    </div>
  )
}
