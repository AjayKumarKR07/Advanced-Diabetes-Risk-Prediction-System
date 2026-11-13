/**
 * src/pages/PredictionResult.jsx
 * ================================
 * Displays prediction output: class, risk tier, probability bars and pie chart.
 */

import { useLocation, useNavigate } from 'react-router-dom'
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend } from 'recharts'
import { ArrowLeft, RefreshCw } from 'lucide-react'
import RiskBadge from '../components/RiskBadge'
import ProbabilityBar from '../components/ProbabilityBar'
import DisclaimerBanner from '../components/DisclaimerBanner'

const CLASS_COLORS   = { 0: '#10b981', 1: '#f59e0b', 2: '#ef4444' }
const CLASS_NAMES    = { 0: 'No Diabetes', 1: 'Prediabetes', 2: 'Diabetes' }
const PROB_COLORS    = { no_diabetes: 'green', prediabetes: 'amber', diabetes: 'red' }
const PROB_LABELS    = { no_diabetes: 'No Diabetes (Class 0)', prediabetes: 'Prediabetes (Class 1)', diabetes: 'Diabetes (Class 2)' }
const PIE_COLORS     = ['#10b981', '#f59e0b', '#ef4444']

export default function PredictionResult() {
  const location = useLocation()
  const navigate  = useNavigate()
  const result    = location.state?.result

  if (!result) {
    return (
      <div className="max-w-2xl mx-auto px-4 py-24 text-center">
        <p className="text-slate-500 mb-6">No prediction result found. Please complete the health assessment form first.</p>
        <button onClick={() => navigate('/assess')} className="btn-primary">
          Go to Assessment
        </button>
      </div>
    )
  }

  const { predicted_class, predicted_label, probabilities, risk_tier, confidence, model_version } = result
  const predColor = CLASS_COLORS[predicted_class]

  const pieData = [
    { name: 'No Diabetes', value: probabilities.no_diabetes },
    { name: 'Prediabetes', value: probabilities.prediabetes },
    { name: 'Diabetes',    value: probabilities.diabetes    },
  ]

  return (
    <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
      <button
        onClick={() => navigate('/assess')}
        className="flex items-center gap-2 text-sm text-slate-500 hover:text-slate-800 mb-6 transition-colors"
      >
        <ArrowLeft className="w-4 h-4" /> Back to Form
      </button>

      <DisclaimerBanner className="mb-8" />

      {/* Primary result card */}
      <div className="card mb-6 text-center">
        <div
          className="w-20 h-20 rounded-full mx-auto mb-4 flex items-center justify-center text-white text-3xl font-extrabold"
          style={{ background: predColor }}
        >
          {predicted_class}
        </div>
        <h1 className="text-3xl font-extrabold text-slate-900 mb-2">{predicted_label}</h1>
        <div className="flex justify-center mb-3">
          <RiskBadge tier={risk_tier} />
        </div>
        <p className="text-sm text-slate-500">
          Model confidence: <strong>{(confidence * 100).toFixed(1)}%</strong> for class {predicted_class}
        </p>
        <p className="text-xs text-slate-400 mt-1">Model: {model_version}</p>
      </div>

      {/* Probabilities + pie */}
      <div className="grid sm:grid-cols-2 gap-6 mb-6">
        {/* Bars */}
        <div className="card">
          <h2 className="font-semibold text-slate-800 mb-4">Class Probabilities</h2>
          <div className="space-y-5">
            {Object.entries(probabilities).map(([key, val]) => (
              <ProbabilityBar
                key={key}
                label={PROB_LABELS[key]}
                value={val}
                color={PROB_COLORS[key]}
              />
            ))}
          </div>
        </div>

        {/* Pie */}
        <div className="card">
          <h2 className="font-semibold text-slate-800 mb-2">Distribution</h2>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie data={pieData} cx="50%" cy="50%" outerRadius={75} dataKey="value" label={({ percent }) => `${(percent * 100).toFixed(1)}%`}>
                {PIE_COLORS.map((c, i) => <Cell key={i} fill={c} />)}
              </Pie>
              <Tooltip formatter={(v) => `${(v * 100).toFixed(2)}%`} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Interpretation note */}
      <div className="card mb-6">
        <h2 className="font-semibold text-slate-800 mb-3">Interpretation Note</h2>
        <p className="text-sm text-slate-600 leading-relaxed">
          This model was trained on 202,944 CDC BRFSS 2015 survey responses using balanced class
          weighting and group-aware cross-validation. The model achieves a Macro F1 of&nbsp;
          <strong>0.4246</strong> and ROC-AUC of <strong>0.7707</strong> on the held-out test set.
        </p>
        <p className="text-sm text-slate-600 leading-relaxed mt-3">
          <strong>Class 1 (Prediabetes)</strong> has known low precision (0.155) and recall (0.309)
          due to class imbalance — the model intentionally over-recalls this class to favour
          sensitivity. <strong>Always consult a qualified healthcare professional.</strong>
        </p>
      </div>

      <div className="flex justify-center">
        <button
          onClick={() => navigate('/assess')}
          className="btn-secondary flex items-center gap-2"
        >
          <RefreshCw className="w-4 h-4" /> New Assessment
        </button>
      </div>
    </div>
  )
}

