/**
 * src/pages/Insights.jsx
 * =======================
 * AI-Powered Health Insights Page
 * Shows personalized insights, recommendations and historical insight summaries.
 */

import { useState, useEffect } from 'react'
import {
  Lightbulb, TrendingUp, TrendingDown, Minus,
  AlertTriangle, CheckCircle, Info, RefreshCw,
  Heart, Activity, Brain, Shield
} from 'lucide-react'
import { fetchInsightsSummary, fetchInsightsHistory } from '../services/api'

const CATEGORY_META = {
  bmi:        { icon: Activity, color: 'blue',   label: 'BMI & Weight' },
  blood:      { icon: Heart,    color: 'red',    label: 'Blood Pressure' },
  lifestyle:  { icon: Shield,   color: 'green',  label: 'Lifestyle' },
  mental:     { icon: Brain,    color: 'purple', label: 'Mental Health' },
  general:    { icon: Lightbulb,color: 'amber',  label: 'General Health' },
}

const SEVERITY_STYLES = {
  high:   'bg-red-50 border-red-200 text-red-700',
  medium: 'bg-amber-50 border-amber-200 text-amber-700',
  low:    'bg-emerald-50 border-emerald-200 text-emerald-700',
  info:   'bg-blue-50 border-blue-100 text-blue-700',
}

const TREND_ICON = {
  improving: <TrendingUp  className="w-4 h-4 text-emerald-500" />,
  worsening: <TrendingDown className="w-4 h-4 text-red-500" />,
  stable:    <Minus       className="w-4 h-4 text-slate-400" />,
}

function InsightCard({ insight }) {
  const severity = insight.severity || insight.level || 'info'
  const trend    = insight.trend || 'stable'
  return (
    <div className={`rounded-xl border p-4 ${SEVERITY_STYLES[severity] || SEVERITY_STYLES.info}`}>
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-start gap-3 flex-1">
          <div className="flex-shrink-0 mt-0.5">
            {severity === 'high'   && <AlertTriangle className="w-4 h-4 text-red-500" />}
            {severity === 'medium' && <AlertTriangle className="w-4 h-4 text-amber-500" />}
            {severity === 'low'    && <CheckCircle   className="w-4 h-4 text-emerald-500" />}
            {severity === 'info'   && <Info          className="w-4 h-4 text-blue-500" />}
          </div>
          <div className="flex-1">
            {insight.title && (
              <p className="text-xs font-bold mb-1">{insight.title}</p>
            )}
            <p className="text-xs leading-relaxed">{insight.message || insight.text || insight.description || JSON.stringify(insight)}</p>
            {insight.recommendation && (
              <div className="mt-2 pt-2 border-t border-current border-opacity-20">
                <p className="text-[11px] font-semibold mb-0.5">💡 Recommendation</p>
                <p className="text-[11px] opacity-90">{insight.recommendation}</p>
              </div>
            )}
          </div>
        </div>
        <div className="flex-shrink-0">
          {TREND_ICON[trend] || TREND_ICON.stable}
        </div>
      </div>
    </div>
  )
}

function HealthScoreRing({ score, label, color }) {
  const pct    = Math.min(Math.max(score || 0, 0), 100)
  const radius = 32
  const circ   = 2 * Math.PI * radius
  const dash   = (pct / 100) * circ
  const colors = { green: '#10b981', blue: '#3b82f6', amber: '#f59e0b', red: '#ef4444' }

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative w-20 h-20">
        <svg viewBox="0 0 80 80" className="rotate-[-90deg]">
          <circle cx="40" cy="40" r={radius} fill="none" stroke="#e2e8f0" strokeWidth="8" />
          <circle
            cx="40" cy="40" r={radius} fill="none"
            stroke={colors[color] || colors.blue} strokeWidth="8"
            strokeDasharray={`${dash} ${circ}`}
            strokeLinecap="round"
            style={{ transition: 'stroke-dasharray 0.6s ease' }}
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-sm font-bold text-slate-800">{pct}%</span>
        </div>
      </div>
      <p className="text-xs font-semibold text-slate-600 text-center">{label}</p>
    </div>
  )
}

export default function Insights() {
  const [summary, setSummary]   = useState(null)
  const [history, setHistory]   = useState([])
  const [loading, setLoading]   = useState(true)
  const [error, setError]       = useState(null)
  const [refreshing, setRefresh] = useState(false)

  const load = async () => {
    try {
      setError(null)
      const [sum, hist] = await Promise.all([
        fetchInsightsSummary().catch(() => null),
        fetchInsightsHistory().catch(() => []),
      ])
      setSummary(sum)
      setHistory(Array.isArray(hist) ? hist : hist?.insights || hist?.history || [])
    } catch {
      setError('Could not load health insights. Complete at least one assessment to generate insights.')
    } finally {
      setLoading(false)
      setRefresh(false)
    }
  }

  useEffect(() => { load() }, [])

  const handleRefresh = () => { setRefresh(true); load() }

  const insights = summary?.insights || summary?.recommendations || []
  const scores   = summary?.scores   || null

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="w-10 h-10 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
          <p className="text-sm text-slate-500">Loading insights…</p>
        </div>
      </div>
    )
  }

  return (
    <div className="p-6 space-y-6 max-w-4xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-slate-900">Health Insights</h1>
          <p className="text-sm text-slate-500 mt-0.5">AI-powered analysis of your health assessment data</p>
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

      {error && (
        <div className="flex items-center gap-3 p-4 bg-amber-50 border border-amber-200 rounded-xl">
          <AlertTriangle className="w-5 h-5 text-amber-500 flex-shrink-0" />
          <p className="text-sm text-amber-700">{error}</p>
        </div>
      )}

      {/* Health Score Rings */}
      {scores && (
        <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6">
          <h2 className="text-sm font-bold text-slate-800 mb-5">Your Health Scores</h2>
          <div className="flex flex-wrap justify-around gap-6">
            {scores.overall    != null && <HealthScoreRing score={scores.overall}    label="Overall Health"   color="blue"  />}
            {scores.lifestyle  != null && <HealthScoreRing score={scores.lifestyle}  label="Lifestyle Score"  color="green" />}
            {scores.metabolic  != null && <HealthScoreRing score={scores.metabolic}  label="Metabolic Health" color="amber" />}
            {scores.mental     != null && <HealthScoreRing score={scores.mental}     label="Mental Health"    color="purple" />}
          </div>
        </div>
      )}

      {/* Insights List */}
      {insights.length > 0 ? (
        <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
          <div className="flex items-center gap-2.5 px-5 py-4 border-b border-slate-100">
            <Lightbulb className="w-4 h-4 text-amber-500" />
            <h2 className="text-sm font-bold text-slate-800">Personalized Insights</h2>
            <span className="ml-auto text-[11px] bg-blue-50 text-blue-700 font-semibold px-2 py-0.5 rounded-full">
              {insights.length} insight{insights.length !== 1 ? 's' : ''}
            </span>
          </div>
          <div className="p-5 space-y-3">
            {insights.map((insight, i) => (
              <InsightCard key={i} insight={typeof insight === 'string' ? { message: insight } : insight} />
            ))}
          </div>
        </div>
      ) : (
        <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-10 text-center">
          <Lightbulb className="w-12 h-12 text-slate-200 mx-auto mb-3" />
          <p className="text-sm font-semibold text-slate-600">No insights yet</p>
          <p className="text-xs text-slate-400 mt-1 max-w-xs mx-auto">
            Complete at least one health assessment and insights will appear here automatically.
          </p>
        </div>
      )}

      {/* Insight History */}
      {history.length > 0 && (
        <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
          <div className="flex items-center gap-2.5 px-5 py-4 border-b border-slate-100">
            <Activity className="w-4 h-4 text-blue-600" />
            <h2 className="text-sm font-bold text-slate-800">Insight History</h2>
          </div>
          <div className="divide-y divide-slate-100">
            {history.slice(0, 8).map((item, i) => (
              <div key={i} className="px-5 py-3 flex items-start gap-3 hover:bg-slate-50 transition-colors">
                <div className="w-1.5 h-1.5 rounded-full bg-blue-400 mt-1.5 flex-shrink-0" />
                <div className="flex-1 min-w-0">
                  <p className="text-xs text-slate-700 leading-relaxed">
                    {typeof item === 'string' ? item : item.message || item.text || item.description || JSON.stringify(item)}
                  </p>
                  {item.created_at && (
                    <p className="text-[11px] text-slate-400 mt-0.5">
                      {new Date(item.created_at).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' })}
                    </p>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Disclaimer */}
      <div className="flex items-start gap-3 p-4 bg-slate-50 border border-slate-200 rounded-xl">
        <Info className="w-4 h-4 text-slate-400 flex-shrink-0 mt-0.5" />
        <p className="text-xs text-slate-500">
          Insights are generated from your self-reported CDC BRFSS indicators and academic model predictions.
          They are for educational awareness only — not clinical advice. Always consult a qualified healthcare professional.
        </p>
      </div>
    </div>
  )
}
