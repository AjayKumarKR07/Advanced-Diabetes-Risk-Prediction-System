/**
 * src/pages/ModelPerformance.jsx
 * ================================
 * Live model info from /api/model-info + static Phase 5 per-class breakdown.
 */

import { useEffect, useState } from 'react'
import { fetchModelInfo } from '../services/api'
import MetricCard from '../components/MetricCard'
import DisclaimerBanner from '../components/DisclaimerBanner'
import { Activity, Loader2 } from 'lucide-react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, Legend,
} from 'recharts'

const PER_CLASS = [
  { cls: 'No Diabetes (0)',  precision: 0.8671, recall: 0.7389, f1: 0.7980, support: 35346 },
  { cls: 'Prediabetes (1)', precision: 0.1548, recall: 0.3089, f1: 0.2064, support: 6236  },
  { cls: 'Diabetes (2)',    precision: 0.7232, recall: 0.5037, f1: 0.5940, support: 9154  },
]

const COLORS = ['#10b981', '#f59e0b', '#ef4444']

export default function ModelPerformance() {
  const [info, setInfo]       = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError]     = useState(null)

  useEffect(() => {
    fetchModelInfo()
      .then(setInfo)
      .catch(() => setError('Backend unreachable — API live metrics unavailable.'))
      .finally(() => setLoading(false))
  }, [])

  const ep = info?.evaluation_performance ?? {}

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
      <div className="mb-8 flex items-start gap-3">
        <div className="w-10 h-10 bg-blue-50 rounded-xl flex items-center justify-center flex-shrink-0">
          <Activity className="w-5 h-5 text-blue-600" />
        </div>
        <div>
          <h1 className="section-title">Model Performance</h1>
          <p className="section-subtitle">
            Phase 5 final evaluation on the frozen CDC test set (50,736 rows, one-time evaluation).
            Logistic Regression selected via Phase 4.2 group-aware OOF benchmarking.
          </p>
        </div>
      </div>

      {loading && (
        <div className="flex items-center gap-2 text-slate-500 mb-6">
          <Loader2 className="w-4 h-4 animate-spin" /> Loading live model info…
        </div>
      )}
      {error && (
        <div className="text-sm text-amber-700 bg-amber-50 border border-amber-200 rounded-lg px-4 py-2 mb-6">
          {error}
        </div>
      )}

      {/* Aggregate metrics */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4 mb-10">
        <MetricCard label="Test Accuracy"   value={`${((ep.test_accuracy    ?? 0.6433) * 100).toFixed(1)}%`} color="blue"   />
        <MetricCard label="Balanced Acc."   value={`${((ep.balanced_accuracy ?? 0.5172) * 100).toFixed(1)}%`} color="purple" />
        <MetricCard label="Macro F1"        value={(ep.macro_f1      ?? 0.4246).toFixed(4)}  color="amber"  />
        <MetricCard label="Weighted F1"     value={(ep.weighted_f1   ?? 0.7187).toFixed(4)}  color="green"  />
        <MetricCard label="ROC-AUC OvR"    value={(ep.roc_auc_ovr_macro ?? 0.7707).toFixed(4)} color="blue" />
        <MetricCard label="Prediabetes Rec." value={`${((ep.class1_prediabetes_recall ?? 0.3089) * 100).toFixed(1)}%`} color="amber" />
      </div>

      {/* Per-class table */}
      <div className="card mb-8 overflow-x-auto">
        <h2 className="font-semibold text-slate-800 mb-4">Per-Class Metrics (Phase 5 Test Set)</h2>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-100 text-slate-500 text-left">
              <th className="pb-2 font-medium">Class</th>
              <th className="pb-2 font-medium text-right">Precision</th>
              <th className="pb-2 font-medium text-right">Recall</th>
              <th className="pb-2 font-medium text-right">F1-Score</th>
              <th className="pb-2 font-medium text-right">Support</th>
            </tr>
          </thead>
          <tbody>
            {PER_CLASS.map(({ cls, precision, recall, f1, support }, i) => (
              <tr key={cls} className="border-b border-slate-50 hover:bg-slate-50">
                <td className="py-3 flex items-center gap-2 font-medium">
                  <span className="w-3 h-3 rounded-full" style={{ background: COLORS[i] }} />
                  {cls}
                </td>
                <td className="py-3 text-right">{precision.toFixed(4)}</td>
                <td className="py-3 text-right">{recall.toFixed(4)}</td>
                <td className="py-3 text-right font-semibold">{f1.toFixed(4)}</td>
                <td className="py-3 text-right text-slate-500">{support.toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Per-class bar chart */}
      <div className="card mb-8">
        <h2 className="font-semibold text-slate-800 mb-4">Precision / Recall / F1 by Class</h2>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={PER_CLASS.map(d => ({ ...d, name: d.cls.split(' ')[0] + ' ' + d.cls.split(' ')[1] }))}>
            <XAxis dataKey="name" tick={{ fontSize: 11 }} />
            <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
            <Tooltip formatter={(v) => v.toFixed(4)} />
            <Legend />
            <Bar dataKey="precision" name="Precision" fill="#93c5fd" radius={[4,4,0,0]} />
            <Bar dataKey="recall"    name="Recall"    fill="#3b82f6" radius={[4,4,0,0]} />
            <Bar dataKey="f1"        name="F1-Score"  fill="#1d4ed8" radius={[4,4,0,0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Features list */}
      {info?.features && (
        <div className="card mb-8">
          <h2 className="font-semibold text-slate-800 mb-3">Input Features ({info.feature_count})</h2>
          <div className="flex flex-wrap gap-2">
            {info.features.map((f, i) => (
              <span key={f} className="inline-flex items-center gap-1 bg-blue-50 text-blue-700 text-xs font-medium px-2.5 py-1 rounded-full">
                <span className="text-blue-300 font-mono">{i + 1}.</span> {f}
              </span>
            ))}
          </div>
        </div>
      )}

      <DisclaimerBanner />
    </div>
  )
}

