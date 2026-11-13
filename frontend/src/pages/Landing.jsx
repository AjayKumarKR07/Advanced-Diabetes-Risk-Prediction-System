/**
 * src/pages/Landing.jsx
 * =====================
 * Hero landing page — project overview, quick stats, navigation CTA.
 */

import { Link } from 'react-router-dom'
import { Activity, ArrowRight, Database, FlaskConical, ShieldCheck, BarChart2 } from 'lucide-react'
import DisclaimerBanner from '../components/DisclaimerBanner'

const QUICK_STATS = [
  { label: 'Training Rows',    value: '202,944', color: 'text-blue-600' },
  { label: 'Test Set Rows',    value: '50,736',  color: 'text-blue-600' },
  { label: 'ROC-AUC (OvR)',   value: '0.7707',  color: 'text-emerald-600' },
  { label: 'Macro F1-Score',  value: '0.4246',  color: 'text-amber-600' },
  { label: 'Balanced Acc.',   value: '51.7%',   color: 'text-purple-600' },
  { label: 'CDC Features',    value: '21',      color: 'text-blue-600' },
]

const FEATURES = [
  {
    Icon: Database,
    title: 'CDC BRFSS 2015 Dataset',
    desc: '253,680 survey responses. Group-aware validation to prevent feature-profile leakage.',
  },
  {
    Icon: FlaskConical,
    title: 'Group-Aware ML Pipeline',
    desc: 'StratifiedGroupKFold with deterministic profile hashing ensures zero group overlap across folds.',
  },
  {
    Icon: ShieldCheck,
    title: 'Frozen Test Set Evaluation',
    desc: 'Single one-time evaluation on a held-out test set. Test data never touched during development.',
  },
  {
    Icon: BarChart2,
    title: '3-Class Prediction',
    desc: 'Classifies as No Diabetes · Prediabetes · Diabetes with calibrated class-balanced probabilities.',
  },
]

export default function Landing() {
  return (
    <div className="min-h-screen">
      {/* Hero */}
      <section className="relative overflow-hidden bg-gradient-to-br from-blue-900 via-blue-800 to-blue-700 text-white">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,_rgba(255,255,255,0.08)_0%,_transparent_60%)]" />
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-28 relative">
          <div className="max-w-3xl">
            <div className="inline-flex items-center gap-2 bg-white/10 backdrop-blur border border-white/20 rounded-full px-4 py-1.5 text-sm font-medium mb-6">
              <Activity className="w-4 h-4" />
              Research Prototype — Not a Clinical Tool
            </div>
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold leading-tight mb-6">
              Advanced Diabetes<br />
              <span className="text-blue-200">Risk Prediction System</span>
            </h1>
            <p className="text-lg text-blue-100 leading-relaxed mb-8 max-w-2xl">
              A group-aware machine learning pipeline trained on the CDC BRFSS 2015 health
              indicators to classify diabetes risk across three classes — No Diabetes,
              Prediabetes, and Diabetes.
            </p>
            <div className="flex flex-wrap gap-4">
              <Link to="/assess" className="inline-flex items-center gap-2 bg-white text-blue-700 font-semibold px-6 py-3 rounded-xl hover:bg-blue-50 transition-colors">
                Start Assessment <ArrowRight className="w-4 h-4" />
              </Link>
              <Link to="/dashboard" className="inline-flex items-center gap-2 bg-white/10 hover:bg-white/20 border border-white/30 font-medium px-6 py-3 rounded-xl transition-colors">
                View Dashboard
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Quick Stats */}
      <section className="bg-white border-b border-slate-100">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-6 text-center">
            {QUICK_STATS.map(({ label, value, color }) => (
              <div key={label}>
                <div className={`text-2xl font-extrabold ${color}`}>{value}</div>
                <div className="text-xs text-slate-500 font-medium mt-0.5">{label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <h2 className="text-3xl font-bold text-slate-900 mb-3 text-center">What This System Does</h2>
        <p className="text-slate-500 text-center mb-12 max-w-xl mx-auto">
          A complete research pipeline from raw CDC survey data to a served prediction API.
        </p>
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {FEATURES.map(({ Icon, title, desc }) => (
            <div key={title} className="card hover:shadow-md transition-shadow">
              <div className="w-10 h-10 bg-blue-50 rounded-xl flex items-center justify-center mb-4">
                <Icon className="w-5 h-5 text-blue-600" />
              </div>
              <h3 className="font-semibold text-slate-900 mb-2">{title}</h3>
              <p className="text-sm text-slate-500 leading-relaxed">{desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Disclaimer */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-16">
        <DisclaimerBanner />
      </div>
    </div>
  )
}

