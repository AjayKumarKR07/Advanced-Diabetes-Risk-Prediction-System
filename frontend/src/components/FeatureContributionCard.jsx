/**
 * FeatureContributionCard.jsx
 * Phase 6.20 — Explainable AI (XAI) feature contribution visualisation.
 *
 * Displays the top 5 health factors that drove the model's prediction using
 * horizontal bar charts (Recharts). Contributions are computed server-side
 * via logistic regression coefficients × scaled feature values.
 *
 * DISCLAIMER: Contributions are computed from logistic regression
 * coefficients for educational purposes only. This is NOT a clinical
 * diagnostic tool and does NOT imply causation.
 */

import React, { useEffect, useState } from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Cell,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'
import { fetchExplanation } from '../services/api'
import { AlertCircle, Loader2, Info } from 'lucide-react'

const CLASS_LABELS = {
  0: 'No Diabetes',
  1: 'Pre-Diabetes',
  2: 'Diabetes',
}

const CLASS_COLORS = {
  0: '#10b981',  // emerald
  1: '#f59e0b',  // amber
  2: '#ef4444',  // red
}

/** Custom tooltip shown on hover over a bar */
function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div className="bg-white border border-slate-200 rounded-xl shadow-lg px-3 py-2 text-sm">
      <p className="font-semibold text-slate-800">{d.label}</p>
      <p className="text-slate-500 text-xs mt-0.5">Your value: <span className="font-medium text-slate-700">{d.valueLabel}</span></p>
      <p className={`text-xs mt-0.5 font-medium ${d.direction === 'risk_increasing' ? 'text-red-600' : 'text-emerald-600'}`}>
        {d.direction === 'risk_increasing' ? '▲ Increases risk' : '▼ Reduces risk'}
      </p>
    </div>
  )
}

export default function FeatureContributionCard({ features, predictedClass }) {
  const [contributions, setContributions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!features || predictedClass === undefined || predictedClass === null) {
      setLoading(false)
      return
    }
    setLoading(true)
    setError(null)
    fetchExplanation(features, predictedClass)
      .then((data) => {
        setContributions(data)
      })
      .catch((err) => {
        console.error('[XAI] fetchExplanation error:', err)
        setError('Could not load feature contributions.')
      })
      .finally(() => setLoading(false))
  }, [features, predictedClass])

  const classLabel = CLASS_LABELS[predictedClass] ?? `Class ${predictedClass}`
  const classColor = CLASS_COLORS[predictedClass] ?? '#6366f1'

  /* ---- Loading state ---- */
  if (loading) {
    return (
      <div className="bg-white border border-slate-200 rounded-2xl shadow-sm p-6 flex items-center justify-center min-h-[180px]">
        <Loader2 className="w-6 h-6 text-blue-500 animate-spin mr-2" />
        <span className="text-slate-500 text-sm">Loading AI explanations…</span>
      </div>
    )
  }

  /* ---- Error / no-data state ---- */
  if (error || !contributions?.top_contributors?.length) {
    return (
      <div className="bg-white border border-slate-200 rounded-2xl shadow-sm p-6">
        <div className="flex items-center gap-2 text-slate-400">
          <AlertCircle className="w-5 h-5" />
          <p className="text-sm">{error ?? 'No contribution data available.'}</p>
        </div>
      </div>
    )
  }

  /* ---- Build chart data ---- */
  const chartData = contributions.top_contributors.map((c) => ({
    label: c.feature_label,
    valueLabel: c.value_label,
    magnitude: parseFloat(c.magnitude.toFixed(4)),
    direction: c.direction,
    contribution: c.direction === 'risk_increasing' ? c.magnitude : -c.magnitude,
  }))

  return (
    <div className="bg-white border border-slate-200 rounded-2xl shadow-sm p-6">
      {/* Header */}
      <div className="flex items-start justify-between mb-1">
        <div>
          <h3 className="text-base font-semibold text-slate-800">AI Explanation</h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Top 5 health factors for predicted class:{' '}
            <span className="font-medium" style={{ color: classColor }}>{classLabel}</span>
          </p>
        </div>
        <span className="text-xs bg-blue-50 text-blue-600 px-2 py-0.5 rounded-full border border-blue-100 font-medium">
          XAI
        </span>
      </div>

      {/* Chart */}
      <div className="mt-4" style={{ height: chartData.length * 52 + 20 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            layout="vertical"
            data={chartData}
            margin={{ top: 0, right: 16, bottom: 0, left: 10 }}
          >
            <XAxis
              type="number"
              tick={{ fontSize: 10, fill: '#94a3b8' }}
              axisLine={false}
              tickLine={false}
              domain={['auto', 'auto']}
              tickFormatter={(v) => v.toFixed(2)}
            />
            <YAxis
              type="category"
              dataKey="label"
              tick={{ fontSize: 11, fill: '#475569', fontWeight: 500 }}
              axisLine={false}
              tickLine={false}
              width={130}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: '#f8fafc' }} />
            <Bar dataKey="contribution" radius={[0, 6, 6, 0]} maxBarSize={22}>
              {chartData.map((entry, idx) => (
                <Cell
                  key={`cell-${idx}`}
                  fill={entry.direction === 'risk_increasing' ? '#fca5a5' : '#6ee7b7'}
                  stroke={entry.direction === 'risk_increasing' ? '#ef4444' : '#10b981'}
                  strokeWidth={1}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-4 mt-3">
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-sm bg-red-300 border border-red-500" />
          <span className="text-xs text-slate-500">Increases risk</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-sm bg-emerald-300 border border-emerald-500" />
          <span className="text-xs text-slate-500">Reduces risk</span>
        </div>
      </div>

      {/* Disclaimer */}
      <div className="mt-4 flex items-start gap-2 bg-slate-50 border border-slate-100 rounded-xl px-3 py-2">
        <Info className="w-3.5 h-3.5 text-slate-400 mt-0.5 shrink-0" />
        <p className="text-xs text-slate-400 leading-relaxed">
          Contributions are computed from logistic regression coefficients — for
          educational purposes only. This tool does <strong>not</strong> imply
          causation and is <strong>not</strong> a clinical diagnostic instrument.
        </p>
      </div>
    </div>
  )
}
