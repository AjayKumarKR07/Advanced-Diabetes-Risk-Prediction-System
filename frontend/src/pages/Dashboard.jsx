/**
 * src/pages/Dashboard.jsx
 * ========================
 * Project overview dashboard: Phase 5 test-set metrics, OOF performance,
 * class distribution charts using Recharts.
 */

import { useEffect, useState } from 'react'
import {
  RadarChart, PolarGrid, PolarAngleAxis, Radar, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, Tooltip, Cell, Legend,
  PieChart, Pie,
} from 'recharts'
import { fetchModelInfo } from '../services/api'
import MetricCard from '../components/MetricCard'
import DisclaimerBanner from '../components/DisclaimerBanner'
import { RefreshCw } from 'lucide-react'

// Phase 5 final test-set results (hardcoded from phase5_final_metrics.json)
const PHASE5_METRICS = {
  accuracy: 0.6433,
  balanced_accuracy: 0.5172,
  macro_f1: 0.4246,
  weighted_f1: 0.7187,
  roc_auc_ovr_macro: 0.7707,
  cohen_kappa: 0.2695,
  matthews_cc: 0.2765,
}

// Phase 4.2 OOF comparison data
const MODEL_COMPARISON = [
  { model: 'Logistic Reg.',   macro_f1: 0.4091, bal_acc: 0.5091, roc_auc: 0.7658 },
  { model: 'Decision Tree',  macro_f1: 0.3227, bal_acc: 0.4451, roc_auc: 0.5992 },
  { model: 'Random Forest',  macro_f1: 0.3819, bal_acc: 0.4756, roc_auc: 0.7501 },
  { model: 'XGBoost',        macro_f1: 0.3985, bal_acc: 0.4961, roc_auc: 0.7549 },
]

// Per-class metrics for the final test evaluation
const CLASS_METRICS = [
  { name: 'No Diabetes',  precision: 0.8671, recall: 0.7389, f1: 0.7980 },
  { name: 'Prediabetes',  precision: 0.1548, recall: 0.3089, f1: 0.2064 },
  { name: 'Diabetes',     precision: 0.7232, recall: 0.5037, f1: 0.5940 },
]

const RADAR_DATA = [
  { metric: 'Accuracy',      value: Math.round(PHASE5_METRICS.accuracy * 100) },
  { metric: 'Bal.Acc.',      value: Math.round(PHASE5_METRICS.balanced_accuracy * 100) },
  { metric: 'Macro F1',      value: Math.round(PHASE5_METRICS.macro_f1 * 100) },
  { metric: 'ROC-AUC',       value: Math.round(PHASE5_METRICS.roc_auc_ovr_macro * 100) },
  { metric: 'Weighted F1',   value: Math.round(PHASE5_METRICS.weighted_f1 * 100) },
]

const CLASS_COLORS = ['#10b981', '#f59e0b', '#ef4444']
const MODEL_COLORS = ['#3b82f6', '#94a3b8', '#94a3b8', '#94a3b8']

export default function Dashboard() {
  const [modelInfo, setModelInfo] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetchModelInfo()
      .then(setModelInfo)
      .catch(() => setError('Backend unreachable — showing cached Phase 5 metrics.'))
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
      <div className="mb-8">
        <h1 className="section-title">Project Dashboard</h1>
        <p className="section-subtitle">
          Phase 5 final evaluation results on the held-out CDC test set (50,736 rows, one-time evaluation).
          Logistic Regression selected from Phase 4.2 group-aware benchmarking.
        </p>
        {error && (
          <div className="mt-3 text-sm text-amber-700 bg-amber-50 border border-amber-200 rounded-lg px-4 py-2 flex items-center gap-2">
            <RefreshCw className="w-4 h-4" /> {error}
          </div>
        )}
      </div>

      {/* Key metrics row */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-7 gap-4 mb-10">
        <MetricCard label="Accuracy"      value={`${(PHASE5_METRICS.accuracy * 100).toFixed(1)}%`}        color="blue"   />
        <MetricCard label="Balanced Acc." value={`${(PHASE5_METRICS.balanced_accuracy * 100).toFixed(1)}%`} color="purple" />
        <MetricCard label="Macro F1"      value={PHASE5_METRICS.macro_f1.toFixed(4)}                      color="amber"  />
        <MetricCard label="Weighted F1"   value={PHASE5_METRICS.weighted_f1.toFixed(4)}                   color="green"  />
        <MetricCard label="ROC-AUC OvR"  value={PHASE5_METRICS.roc_auc_ovr_macro.toFixed(4)}             color="blue"   />
        <MetricCard label="Cohen's κ"    value={PHASE5_METRICS.cohen_kappa.toFixed(4)}                   color="slate"  />
        <MetricCard label="MCC"           value={PHASE5_METRICS.matthews_cc.toFixed(4)}                   color="slate"  />
      </div>

      {/* Charts row 1 */}
      <div className="grid lg:grid-cols-2 gap-6 mb-6">
        {/* Radar chart */}
        <div className="card">
          <h2 className="font-semibold text-slate-800 mb-4">Performance Profile (Radar)</h2>
          <ResponsiveContainer width="100%" height={260}>
            <RadarChart data={RADAR_DATA}>
              <PolarGrid stroke="#e2e8f0" />
              <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11, fill: '#64748b' }} />
              <Radar name="Phase 5" dataKey="value" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.25} dot />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        {/* Model comparison bar chart */}
        <div className="card">
          <h2 className="font-semibold text-slate-800 mb-4">Phase 4.2 OOF Model Comparison (Macro F1)</h2>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={MODEL_COMPARISON} layout="vertical" margin={{ left: 20 }}>
              <XAxis type="number" domain={[0, 0.5]} tick={{ fontSize: 11 }} />
              <YAxis type="category" dataKey="model" tick={{ fontSize: 11 }} width={90} />
              <Tooltip formatter={(v) => v.toFixed(4)} />
              <Bar dataKey="macro_f1" radius={[0, 6, 6, 0]}>
                {MODEL_COMPARISON.map((_, i) => (
                  <Cell key={i} fill={MODEL_COLORS[i]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Charts row 2 */}
      <div className="grid lg:grid-cols-2 gap-6 mb-10">
        {/* Per-class F1 bar */}
        <div className="card">
          <h2 className="font-semibold text-slate-800 mb-4">Per-Class F1 Score (Phase 5 Test Set)</h2>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={CLASS_METRICS}>
              <XAxis dataKey="name" tick={{ fontSize: 11 }} />
              <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v) => v.toFixed(4)} />
              <Legend />
              <Bar dataKey="precision" name="Precision" fill="#93c5fd" radius={[4,4,0,0]} />
              <Bar dataKey="recall"    name="Recall"    fill="#3b82f6" radius={[4,4,0,0]} />
              <Bar dataKey="f1"        name="F1"        fill="#1d4ed8" radius={[4,4,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Class distribution pie */}
        <div className="card">
          <h2 className="font-semibold text-slate-800 mb-4">Test-Set Class Distribution</h2>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie
                data={[
                  { name: 'No Diabetes',  value: 35346 },
                  { name: 'Prediabetes',  value: 6236  },
                  { name: 'Diabetes',     value: 9154  },
                ]}
                cx="50%" cy="50%" outerRadius={80}
                dataKey="value" label={({ name, percent }) => `${name} ${(percent * 100).toFixed(1)}%`}
                labelLine={false}
              >
                {CLASS_COLORS.map((color, i) => <Cell key={i} fill={color} />)}
              </Pie>
              <Tooltip formatter={(v) => v.toLocaleString()} />
            </PieChart>
          </ResponsiveContainer>
          <div className="flex justify-center gap-4 mt-2">
            {['No Diabetes', 'Prediabetes', 'Diabetes'].map((n, i) => (
              <span key={n} className="flex items-center gap-1 text-xs text-slate-500">
                <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ background: CLASS_COLORS[i] }} />
                {n}
              </span>
            ))}
          </div>
        </div>
      </div>

      <DisclaimerBanner />
    </div>
  )
}

