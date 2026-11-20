/**
 * src/pages/AuthModelPerformance.jsx
 * =====================================
 * Phase 6.17.1 — Authenticated Model Performance page.
 * Displays frozen Phase 5 Logistic Regression evaluation metrics inside the
 * AuthenticatedLayout (sidebar + header). Reuses MetricCard and DisclaimerBanner.
 *
 * FROZEN METRICS — DO NOT MODIFY.
 * Values sourced from Phase 5 final evaluation on CDC BRFSS 2015 test set.
 */

import { useEffect, useState } from 'react'
import { fetchModelInfo } from '../services/api'
import MetricCard from '../components/MetricCard'
import DisclaimerBanner from '../components/DisclaimerBanner'
import {
  Activity,
  Loader2,
  AlertTriangle,
  Database,
  Layers,
  TrendingUp,
  Info,
} from 'lucide-react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  Legend,
  PieChart,
  Pie,
} from 'recharts'

/* ── FROZEN Phase 5 metrics (spec §4) ── */
const GLOBAL_METRICS = {
  accuracy: 0.6433,
  balanced_accuracy: 0.5172,
  macro_f1: 0.4246,
  weighted_f1: 0.7187,
  roc_auc_macro: 0.7707,
  roc_auc_weighted: 0.8112,
}

const PER_CLASS = [
  {
    cls: 'No Diabetes',
    short: 'No DM',
    precision: 0.9501,
    recall: 0.6606,
    f1: 0.7793,
    support: 35346,
    color: '#10b981',
  },
  {
    cls: 'Prediabetes',
    short: 'Pre-DM',
    precision: 0.0307,
    recall: 0.3089,
    f1: 0.0558,
    support: 6236,
    color: '#f59e0b',
  },
  {
    cls: 'Diabetes',
    short: 'DM',
    precision: 0.3520,
    recall: 0.5823,
    f1: 0.4387,
    support: 9154,
    color: '#ef4444',
  },
]

const DISTRIBUTION_DATA = PER_CLASS.map((c) => ({
  name: c.short,
  value: c.support,
  color: c.color,
}))

const TOTAL_TEST = PER_CLASS.reduce((s, c) => s + c.support, 0) // 50,736

/* Custom Recharts tooltip */
function MetricTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-white border border-slate-200 rounded-xl shadow-lg px-3 py-2 text-xs">
      <p className="font-semibold text-slate-700 mb-1">{label}</p>
      {payload.map((p) => (
        <p key={p.name} style={{ color: p.fill ?? p.color }}>
          {p.name}: {typeof p.value === 'number' ? p.value.toFixed(4) : p.value}
        </p>
      ))}
    </div>
  )
}

export default function AuthModelPerformance() {
  const [liveInfo, setLiveInfo] = useState(null)
  const [liveLoading, setLiveLoading] = useState(true)
  const [liveError, setLiveError] = useState(null)

  useEffect(() => {
    fetchModelInfo()
      .then(setLiveInfo)
      .catch(() => setLiveError('Live model metadata unavailable — backend may be offline.'))
      .finally(() => setLiveLoading(false))
  }, [])

  const barData = PER_CLASS.map((c) => ({
    name: c.short,
    cls: c.cls,
    Precision: c.precision,
    Recall: c.recall,
    'F1-Score': c.f1,
    color: c.color,
  }))

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
      {/* Page Header */}
      <div className="flex items-start gap-3">
        <div className="w-10 h-10 bg-blue-50 rounded-xl flex items-center justify-center shrink-0">
          <Activity className="w-5 h-5 text-blue-600" />
        </div>
        <div>
          <h1 className="text-xl font-bold text-slate-900">Model Performance</h1>
          <p className="text-sm text-slate-500 mt-0.5">
            Frozen Phase 5 evaluation — CDC BRFSS 2015 test set (50,736 records).
            Logistic Regression selected via Phase 4.2 group-aware OOF benchmarking.
          </p>
        </div>
      </div>

      {/* Disclaimer */}
      <DisclaimerBanner />

      {/* Prediabetes Warning Banner */}
      <div className="bg-amber-50 border border-amber-200 rounded-xl px-4 py-3 flex items-start gap-3">
        <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
        <div className="text-xs text-amber-800">
          <p className="font-bold mb-0.5">Prediabetes Classification Limitation</p>
          <p>
            Class 1 (Prediabetes) achieves a Precision of <strong>0.0307</strong> and F1-Score of{' '}
            <strong>0.0558</strong> due to severe class imbalance (only ~2.5% of records). The model
            is configured with balanced class weights to maximize sensitivity, but individual
            predictions in this class carry <strong>substantial uncertainty</strong>. This model is{' '}
            <strong>not a clinical diagnostic tool</strong>.
          </p>
        </div>
      </div>

      {/* Dataset Info */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {[
          { icon: Database, label: 'Dataset', value: 'CDC BRFSS 2015' },
          { icon: Layers, label: 'Total Records', value: '253,680' },
          { icon: TrendingUp, label: 'Test Records', value: '50,736' },
          { icon: Activity, label: 'Features', value: '21 CDC Indicators' },
        ].map(({ icon: Icon, label, value }) => (
          <div
            key={label}
            className="bg-white border border-slate-200 rounded-xl px-4 py-3 flex items-center gap-3"
          >
            <div className="w-8 h-8 bg-blue-50 rounded-lg flex items-center justify-center shrink-0">
              <Icon className="w-4 h-4 text-blue-600" />
            </div>
            <div>
              <p className="text-[10px] text-slate-400 font-medium uppercase tracking-wide">{label}</p>
              <p className="text-sm font-bold text-slate-800">{value}</p>
            </div>
          </div>
        ))}
      </div>

      {/* Global Metrics KPI Cards */}
      <div>
        <h2 className="text-sm font-bold text-slate-700 mb-3 flex items-center gap-2">
          <span>Global Metrics</span>
          <span className="text-[10px] text-slate-400 font-normal">(Phase 5 frozen)</span>
        </h2>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          <MetricCard label="Accuracy"         value={`${(GLOBAL_METRICS.accuracy * 100).toFixed(1)}%`}        color="blue"   />
          <MetricCard label="Balanced Acc."    value={`${(GLOBAL_METRICS.balanced_accuracy * 100).toFixed(1)}%`} color="purple" />
          <MetricCard label="Macro F1"         value={GLOBAL_METRICS.macro_f1.toFixed(4)}                      color="amber"  />
          <MetricCard label="Weighted F1"      value={GLOBAL_METRICS.weighted_f1.toFixed(4)}                   color="green"  />
          <MetricCard label="ROC-AUC Macro"    value={GLOBAL_METRICS.roc_auc_macro.toFixed(4)}                 color="blue"   />
          <MetricCard label="ROC-AUC Weighted" value={GLOBAL_METRICS.roc_auc_weighted.toFixed(4)}              color="green"  />
        </div>
      </div>

      {/* Class-wise performance table */}
      <div className="bg-white border border-slate-200 rounded-2xl shadow-sm overflow-hidden">
        <div className="px-6 py-4 border-b border-slate-100">
          <h2 className="text-sm font-bold text-slate-800">Per-Class Performance</h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Phase 5 test set evaluation · Model: Multinomial Logistic Regression · Balanced class weights
          </p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-slate-50 text-slate-500 text-xs uppercase tracking-wide">
                <th className="px-6 py-3 text-left font-semibold">Class</th>
                <th className="px-6 py-3 text-right font-semibold">Precision</th>
                <th className="px-6 py-3 text-right font-semibold">Recall</th>
                <th className="px-6 py-3 text-right font-semibold">F1-Score</th>
                <th className="px-6 py-3 text-right font-semibold">Support</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-50">
              {PER_CLASS.map((c, i) => (
                <tr key={c.cls} className="hover:bg-slate-50 transition-colors">
                  <td className="px-6 py-3.5 font-semibold text-slate-800 flex items-center gap-2">
                    <span
                      className="w-3 h-3 rounded-full shrink-0"
                      style={{ background: c.color }}
                    />
                    {c.cls}
                    {i === 1 && (
                      <span className="text-[10px] bg-amber-100 text-amber-700 px-1.5 py-0.5 rounded-full border border-amber-200 font-bold">
                        Weak
                      </span>
                    )}
                  </td>
                  <td className={`px-6 py-3.5 text-right font-mono ${i === 1 ? 'text-amber-600 font-bold' : 'text-slate-700'}`}>
                    {c.precision.toFixed(4)}
                  </td>
                  <td className="px-6 py-3.5 text-right font-mono text-slate-700">
                    {c.recall.toFixed(4)}
                  </td>
                  <td className={`px-6 py-3.5 text-right font-mono font-bold ${i === 1 ? 'text-amber-600' : 'text-slate-800'}`}>
                    {c.f1.toFixed(4)}
                  </td>
                  <td className="px-6 py-3.5 text-right text-slate-500 font-mono">
                    {c.support.toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Charts row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Precision/Recall/F1 chart */}
        <div className="bg-white border border-slate-200 rounded-2xl shadow-sm p-5">
          <h3 className="text-sm font-bold text-slate-800 mb-4">Precision / Recall / F1 by Class</h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={barData} margin={{ left: -10 }}>
              <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#64748b' }} axisLine={false} tickLine={false} />
              <YAxis domain={[0, 1]} tick={{ fontSize: 10, fill: '#94a3b8' }} axisLine={false} tickLine={false} />
              <Tooltip content={<MetricTooltip />} />
              <Legend iconSize={10} wrapperStyle={{ fontSize: 11 }} />
              <Bar dataKey="Precision" fill="#93c5fd" radius={[3, 3, 0, 0]} />
              <Bar dataKey="Recall"    fill="#3b82f6" radius={[3, 3, 0, 0]} />
              <Bar dataKey="F1-Score"  fill="#1d4ed8" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Class distribution */}
        <div className="bg-white border border-slate-200 rounded-2xl shadow-sm p-5">
          <h3 className="text-sm font-bold text-slate-800 mb-4">Test-Set Class Distribution</h3>
          <ResponsiveContainer width="100%" height={180}>
            <PieChart>
              <Pie
                data={DISTRIBUTION_DATA}
                cx="50%"
                cy="50%"
                outerRadius={70}
                dataKey="value"
                nameKey="name"
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(1)}%`}
                labelLine={false}
              >
                {DISTRIBUTION_DATA.map((entry, i) => (
                  <Cell key={i} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip formatter={(v) => v.toLocaleString()} />
            </PieChart>
          </ResponsiveContainer>
          <div className="mt-2 flex items-center justify-center gap-4 flex-wrap">
            {DISTRIBUTION_DATA.map((d) => (
              <div key={d.name} className="flex items-center gap-1.5 text-xs text-slate-600">
                <span className="w-2.5 h-2.5 rounded-full" style={{ background: d.color }} />
                {d.name}: {d.value.toLocaleString()}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Live model info */}
      {!liveLoading && !liveError && liveInfo && (
        <div className="bg-white border border-slate-200 rounded-2xl shadow-sm p-5">
          <h3 className="text-sm font-bold text-slate-800 mb-3">
            Live Model Metadata
            <span className="ml-2 text-[10px] text-emerald-600 bg-emerald-50 border border-emerald-200 px-2 py-0.5 rounded-full font-bold">
              Backend Connected
            </span>
          </h3>
          {liveInfo.features && (
            <>
              <p className="text-xs text-slate-500 mb-2">
                Input features ({liveInfo.feature_count}):
              </p>
              <div className="flex flex-wrap gap-1.5">
                {liveInfo.features.map((f, i) => (
                  <span
                    key={f}
                    className="inline-flex items-center gap-1 bg-blue-50 text-blue-700 text-xs font-medium px-2 py-0.5 rounded-full border border-blue-100"
                  >
                    <span className="text-blue-300 font-mono text-[10px]">{i + 1}.</span> {f}
                  </span>
                ))}
              </div>
            </>
          )}
        </div>
      )}

      {liveError && (
        <div className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-xl px-4 py-2 flex items-center gap-2">
          <Info className="w-3.5 h-3.5 shrink-0" />
          {liveError}
        </div>
      )}

      {/* Scientific disclaimer */}
      <div className="bg-slate-50 border border-slate-200 rounded-xl px-5 py-4">
        <div className="flex items-start gap-2">
          <Info className="w-4 h-4 text-slate-400 shrink-0 mt-0.5" />
          <div className="text-xs text-slate-500 space-y-1">
            <p className="font-semibold text-slate-600">Scientific & Clinical Disclaimer</p>
            <p>
              All metrics shown are from the <strong>frozen Phase 5 one-time test-set evaluation</strong>.
              The model has not been retrained or fine-tuned. These values must not be interpreted as
              clinical validation or diagnostic reliability benchmarks.
            </p>
            <p>
              DiabetesRisk AI is an academic research prototype. It is <strong>not FDA-cleared</strong> and
              must not be used for clinical decision-making. Always consult a licensed physician for
              medical evaluation.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
