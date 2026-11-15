/**
 * src/components/HealthInsightsSection.jsx
 * ========================================
 * Intelligent Health Insights & Personalized Educational Recommendations Component (Phase 6.16 & Phase 6.16.1).
 * 
 * Features:
 * - Summary KPI cards (Total Insights, High Priority, Lifestyle Factors, Positive Habits)
 * - Category filter tabs (All, Metabolic, Cardiovascular, Lifestyle, Well-being, High Priority, Positive)
 * - Educational recommendation cards with expandable scientific rationale, evidence citations, rule triggers & limitations
 * - Longitudinal comparison matrix tracking indicator stability, neutral shifts, and survey response discrepancies
 * - Strict non-medical educational disclaimers
 * - Responsive light and dark theme support
 */

import { useState, useMemo } from 'react'
import {
  Lightbulb,
  ShieldAlert,
  CheckCircle2,
  Info,
  ChevronDown,
  ChevronUp,
  Activity,
  Heart,
  Scale,
  Sparkles,
  ArrowRight,
  TrendingUp,
  TrendingDown,
  Minus,
  HelpCircle,
  AlertTriangle,
  BookOpen,
} from 'lucide-react'

const PRIORITY_BADGES = {
  high: {
    label: 'High Priority',
    bg: 'bg-rose-50 dark:bg-rose-950/40 text-rose-700 dark:text-rose-300 border-rose-200 dark:border-rose-800',
    dot: 'bg-rose-500',
  },
  medium: {
    label: 'Moderate Priority',
    bg: 'bg-amber-50 dark:bg-amber-950/40 text-amber-700 dark:text-amber-300 border-amber-200 dark:border-amber-800',
    dot: 'bg-amber-500',
  },
  info: {
    label: 'Educational Info',
    bg: 'bg-blue-50 dark:bg-blue-950/40 text-blue-700 dark:text-blue-300 border-blue-200 dark:border-blue-800',
    dot: 'bg-blue-500',
  },
  positive: {
    label: 'Positive Habit',
    bg: 'bg-emerald-50 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-300 border-emerald-200 dark:border-emerald-800',
    dot: 'bg-emerald-500',
  },
}

export default function HealthInsightsSection({ insightsData, onStartAssessment }) {
  const [activeCategory, setActiveCategory] = useState('All')
  const [expandedCards, setExpandedCards] = useState({})
  const [showHistory, setShowHistory] = useState(false)

  const summary = insightsData?.summary || {}
  const recommendations = insightsData?.recommendations || []
  const comparisons = insightsData?.historical_comparisons || []
  const hasAssessments = (insightsData?.total_assessments || 0) > 0

  const toggleExpand = (id) => {
    setExpandedCards((prev) => ({ ...prev, [id]: !prev[id] }))
  }

  // Filter recommendations based on active category
  const filteredRecs = useMemo(() => {
    if (activeCategory === 'All') return recommendations
    if (activeCategory === 'High Priority') {
      return recommendations.filter((r) => r.priority === 'high')
    }
    if (activeCategory === 'Positive Factors') {
      return recommendations.filter((r) => r.priority === 'positive')
    }
    return recommendations.filter((r) => r.category === activeCategory)
  }, [recommendations, activeCategory])

  const categories = [
    'All',
    'High Priority',
    'Positive Factors',
    'Metabolic & Body',
    'Cardiovascular Health',
    'Physical Lifestyle',
    'General Well-being',
  ]

  if (!hasAssessments) {
    return (
      <div className="bg-white dark:bg-slate-800 rounded-3xl border border-slate-200/90 dark:border-slate-700 shadow-sm p-8 text-center space-y-4 transition-colors">
        <div className="w-14 h-14 rounded-2xl bg-indigo-50 dark:bg-indigo-900/40 text-indigo-600 dark:text-indigo-300 flex items-center justify-center mx-auto">
          <Lightbulb className="w-7 h-7" />
        </div>
        <div className="max-w-md mx-auto space-y-2">
          <h3 className="text-lg font-bold text-slate-900 dark:text-white">
            No Health Insights Available Yet
          </h3>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            Complete your first health assessment to generate personalized, explainable educational
            insights across your metabolic, cardiovascular, and lifestyle indicators.
          </p>
        </div>
        {onStartAssessment && (
          <button
            onClick={onStartAssessment}
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-700 text-white text-xs font-bold transition-all shadow-md shadow-blue-500/20"
          >
            Start First Assessment <ArrowRight className="w-3.5 h-3.5" />
          </button>
        )}
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* ── Section Header ────────────────────────────────────────────── */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-50 dark:bg-indigo-900/40 border border-indigo-200 dark:border-indigo-700 text-indigo-700 dark:text-indigo-300 text-xs font-semibold mb-2">
            <Sparkles className="w-3.5 h-3.5" />
            <span>Explainable Recommendation Engine</span>
          </div>
          <h2 className="text-xl font-bold text-slate-900 dark:text-white tracking-tight">
            Personalized Health Insights &amp; Educational Recommendations
          </h2>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 max-w-3xl">
            Transparent observations and educational suggestions generated from your self-reported CDC BRFSS survey indicators in Assessment #{insightsData?.assessment_id || '—'}.
          </p>
        </div>

        {comparisons.length > 0 && (
          <button
            onClick={() => setShowHistory(!showHistory)}
            className="px-4 py-2 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-200 text-xs font-semibold hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors flex items-center gap-1.5 self-start"
          >
            <Activity className="w-3.5 h-3.5 text-teal-600 dark:text-teal-400" />
            <span>{showHistory ? 'Hide Longitudinal Changes' : 'View Longitudinal Changes'}</span>
            {showHistory ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
          </button>
        )}
      </div>

      {/* ── Summary KPI Cards ─────────────────────────────────────────── */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        {/* Total Insights */}
        <div className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-200/80 dark:border-slate-700 p-4 shadow-xs">
          <p className="text-[11px] font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
            Total Insights
          </p>
          <p className="text-2xl font-extrabold text-slate-900 dark:text-white mt-1">
            {summary.total_insights ?? recommendations.length}
          </p>
          <p className="text-[10px] text-slate-400 mt-0.5">Across 9 health indicators</p>
        </div>

        {/* High Priority */}
        <div className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-200/80 dark:border-slate-700 p-4 shadow-xs">
          <p className="text-[11px] font-semibold text-rose-600 dark:text-rose-400 uppercase tracking-wider flex items-center gap-1">
            <ShieldAlert className="w-3.5 h-3.5" /> Areas for Doctor Review
          </p>
          <p className="text-2xl font-extrabold text-slate-900 dark:text-white mt-1">
            {summary.high_priority_count ?? 0}
          </p>
          <p className="text-[10px] text-slate-400 mt-0.5">Clinical consultation suggested</p>
        </div>

        {/* Moderate Priority */}
        <div className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-200/80 dark:border-slate-700 p-4 shadow-xs">
          <p className="text-[11px] font-semibold text-amber-600 dark:text-amber-400 uppercase tracking-wider flex items-center gap-1">
            <AlertTriangle className="w-3.5 h-3.5" /> Lifestyle Opportunities
          </p>
          <p className="text-2xl font-extrabold text-slate-900 dark:text-white mt-1">
            {summary.medium_priority_count ?? 0}
          </p>
          <p className="text-[10px] text-slate-400 mt-0.5">Actionable habit adjustments</p>
        </div>

        {/* Positive Habits */}
        <div className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-200/80 dark:border-slate-700 p-4 shadow-xs">
          <p className="text-[11px] font-semibold text-emerald-600 dark:text-emerald-400 uppercase tracking-wider flex items-center gap-1">
            <CheckCircle2 className="w-3.5 h-3.5" /> Positive Health Factors
          </p>
          <p className="text-2xl font-extrabold text-slate-900 dark:text-white mt-1">
            {summary.positive_habits_count ?? 0}
          </p>
          <p className="text-[10px] text-slate-400 mt-0.5">Protective lifestyle factors</p>
        </div>
      </div>

      {/* ── Longitudinal Comparison Matrix (Collapsible) ─────────────── */}
      {showHistory && comparisons.length > 0 && (
        <div className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-200/90 dark:border-slate-700 p-6 shadow-sm space-y-4 transition-colors">
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-teal-600 dark:text-teal-400" />
            <h3 className="font-bold text-slate-900 dark:text-white text-sm">
              Longitudinal Indicator Comparison (Latest vs. Previous Evaluation)
            </h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-slate-200 dark:border-slate-700 text-slate-500 dark:text-slate-400 font-semibold uppercase tracking-wider text-left">
                  <th className="pb-3 pl-2">Indicator</th>
                  <th className="pb-3">Previous Assessment</th>
                  <th className="pb-3">Latest Assessment</th>
                  <th className="pb-3">Shift Status</th>
                  <th className="pb-3 pr-2">Observation &amp; Lifestyle Context</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-700/60">
                {comparisons.map((c, idx) => {
                  let badgeClass = 'bg-slate-100 text-slate-600 dark:bg-slate-700 dark:text-slate-300'
                  let icon = <Minus className="w-3 h-3" />
                  let label = c.change_type.charAt(0).toUpperCase() + c.change_type.slice(1)

                  if (c.change_type === 'decreased') {
                    badgeClass = 'bg-blue-50 text-blue-700 dark:bg-blue-950/40 dark:text-blue-300 border border-blue-200 dark:border-blue-800'
                    icon = <TrendingDown className="w-3 h-3" />
                    label = 'Decreased'
                  } else if (c.change_type === 'increased') {
                    badgeClass = 'bg-indigo-50 text-indigo-700 dark:bg-indigo-950/40 dark:text-indigo-300 border border-indigo-200 dark:border-indigo-800'
                    icon = <TrendingUp className="w-3 h-3" />
                    label = 'Increased'
                  } else if (c.change_type === 'conflicting') {
                    badgeClass = 'bg-amber-50 text-amber-700 dark:bg-amber-950/40 dark:text-amber-300 border border-amber-200 dark:border-amber-800'
                    icon = <AlertTriangle className="w-3 h-3" />
                    label = 'Response Discrepancy'
                  } else if (c.change_type === 'improved') {
                    badgeClass = 'bg-emerald-50 text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-800'
                    icon = <TrendingUp className="w-3 h-3" />
                    label = 'Positive Shift'
                  } else if (c.change_type === 'declined') {
                    badgeClass = 'bg-amber-50 text-amber-700 dark:bg-amber-950/40 dark:text-amber-300 border border-amber-200 dark:border-amber-800'
                    icon = <TrendingDown className="w-3 h-3" />
                    label = 'Shift Noted'
                  } else if (c.change_type === 'stable') {
                    badgeClass = 'bg-slate-100 text-slate-600 dark:bg-slate-700 dark:text-slate-300 border border-slate-200 dark:border-slate-600'
                    icon = <Minus className="w-3 h-3" />
                    label = 'Relatively Stable'
                  }

                  return (
                    <tr key={idx} className="hover:bg-slate-50/80 dark:hover:bg-slate-700/30 transition-colors">
                      <td className="py-3 pl-2 font-semibold text-slate-800 dark:text-slate-200">{c.indicator}</td>
                      <td className="py-3 text-slate-500 font-mono">{c.previous_value}</td>
                      <td className="py-3 font-mono font-bold text-slate-900 dark:text-white">{c.current_value}</td>
                      <td className="py-3">
                        <span className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[10px] font-bold ${badgeClass}`}>
                          {icon}
                          {label}
                        </span>
                      </td>
                      <td className="py-3 text-slate-600 dark:text-slate-300 pr-2 max-w-md">{c.observation}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          <p className="text-[11px] text-slate-400 italic">
            * Changes reflect self-reported survey responses over time. They do not constitute objective clinical measurements or indicate disease progression or regression.
          </p>
        </div>
      )}

      {/* ── Category Filter Tabs ──────────────────────────────────────── */}
      <div className="flex flex-wrap gap-2 pt-2 border-t border-slate-200/80 dark:border-slate-700/60">
        {categories.map((cat) => {
          const isActive = activeCategory === cat
          return (
            <button
              key={cat}
              onClick={() => setActiveCategory(cat)}
              className={`px-3 py-1.5 rounded-xl text-xs font-semibold transition-all ${
                isActive
                  ? 'bg-indigo-600 text-white shadow-xs'
                  : 'bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-700'
              }`}
            >
              {cat}
            </button>
          )
        })}
      </div>

      {/* ── Recommendation Cards Grid ─────────────────────────────────── */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {filteredRecs.map((rec) => {
          const badge = PRIORITY_BADGES[rec.priority] || PRIORITY_BADGES.info
          const isExpanded = !!expandedCards[rec.id]

          return (
            <div
              key={rec.id}
              className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-200/90 dark:border-slate-700 shadow-xs p-5 flex flex-col justify-between transition-colors space-y-4"
            >
              <div className="space-y-3">
                {/* Header: Category & Priority */}
                <div className="flex items-center justify-between gap-2">
                  <span className="text-[11px] font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                    {rec.category}
                  </span>
                  <span
                    className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[10px] font-bold border ${badge.bg}`}
                  >
                    <span className={`w-1.5 h-1.5 rounded-full ${badge.dot}`}></span>
                    {badge.label}
                  </span>
                </div>

                {/* Observation */}
                <div>
                  <h4 className="text-sm font-bold text-slate-900 dark:text-white leading-snug">
                    {rec.observation}
                  </h4>
                  <div className="inline-block mt-1 px-2 py-0.5 rounded bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300 text-[11px] font-mono font-medium">
                    {rec.indicator_name}: {rec.indicator_value}
                  </div>
                </div>

                {/* Educational Suggestion */}
                <div className="bg-slate-50/80 dark:bg-slate-900/40 rounded-xl p-3.5 border border-slate-100 dark:border-slate-700/60 space-y-1">
                  <div className="flex items-center gap-1.5 text-xs font-semibold text-indigo-700 dark:text-indigo-300">
                    <Lightbulb className="w-3.5 h-3.5" />
                    <span>Educational Suggestion</span>
                  </div>
                  <p className="text-xs text-slate-700 dark:text-slate-300 leading-relaxed">
                    {rec.suggestion}
                  </p>
                </div>

                {/* Expandable Scientific Context, Evidence, Trigger & Limitations */}
                {isExpanded && (
                  <div className="pt-2 space-y-2.5 border-t border-slate-100 dark:border-slate-700/60 text-xs">
                    <div>
                      <span className="font-semibold text-slate-800 dark:text-slate-200 block">
                        Scientific Background:
                      </span>
                      <p className="text-slate-600 dark:text-slate-300 text-[11px] mt-0.5 leading-relaxed">
                        {rec.explanation}
                      </p>
                    </div>

                    {rec.evidence_reference && (
                      <div className="pt-1">
                        <span className="font-semibold text-slate-800 dark:text-slate-200 flex items-center gap-1 text-[11px]">
                          <BookOpen className="w-3 h-3 text-indigo-500" /> Authoritative Evidence Reference:
                        </span>
                        <p className="text-slate-500 dark:text-slate-400 text-[10px] mt-0.5 leading-relaxed italic">
                          {rec.evidence_reference}
                        </p>
                      </div>
                    )}

                    <div className="pt-1">
                      <span className="font-semibold text-slate-800 dark:text-slate-200 block text-[11px]">
                        Rule Trigger Condition:
                      </span>
                      <code className="text-[10px] text-slate-500 dark:text-slate-400 bg-slate-100 dark:bg-slate-700 px-2 py-1 rounded block mt-0.5">
                        {rec.reason}
                      </code>
                    </div>

                    {rec.limitations && (
                      <div className="pt-1">
                        <span className="font-semibold text-amber-700 dark:text-amber-400 flex items-center gap-1 text-[11px]">
                          <HelpCircle className="w-3 h-3" /> Survey Indicator Limitations:
                        </span>
                        <p className="text-slate-500 dark:text-slate-400 text-[10px] mt-0.5 leading-relaxed">
                          {rec.limitations}
                        </p>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Card Footer: Toggle & Date */}
              <div className="flex items-center justify-between pt-2 border-t border-slate-100 dark:border-slate-700/60 text-[11px]">
                <button
                  onClick={() => toggleExpand(rec.id)}
                  className="text-blue-600 dark:text-blue-400 font-bold hover:underline flex items-center gap-1"
                >
                  <span>{isExpanded ? 'Hide Scientific Rationale & References' : 'Why was this generated?'}</span>
                  {isExpanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                </button>
                <span className="text-slate-400">{rec.date_generated}</span>
              </div>
            </div>
          )
        })}
      </div>

      {filteredRecs.length === 0 && (
        <div className="text-center py-8 bg-slate-50 dark:bg-slate-900/40 rounded-2xl border border-dashed border-slate-200 dark:border-slate-700">
          <p className="text-xs font-semibold text-slate-600 dark:text-slate-300">
            No recommendations in category "{activeCategory}".
          </p>
        </div>
      )}

      {/* ── Academic & Safety Disclaimer Box ──────────────────────────── */}
      <div className="bg-slate-50 dark:bg-slate-800/80 rounded-2xl border border-slate-200 dark:border-slate-700 p-4 text-xs text-slate-500 dark:text-slate-400 space-y-1">
        <p className="font-bold text-slate-700 dark:text-slate-200 flex items-center gap-1.5">
          <Info className="w-4 h-4 text-indigo-500" /> Educational Insights Notice &amp; Safety Disclosures
        </p>
        <p className="text-[11px] leading-relaxed">
          {insightsData?.disclaimer ||
            'These educational insights are generated by a transparent rule-based algorithm from your self-reported survey responses. They are intended strictly for educational awareness and lifestyle reflection. They do NOT constitute medical advice, clinical diagnoses, or treatment plans. Never alter medication or therapy without consulting a licensed physician.'}
        </p>
      </div>
    </div>
  )
}
