/**
 * src/pages/AboutMethodology.jsx
 * =================================
 * Project methodology, pipeline phases, limitations, and disclaimer.
 */

import { ExternalLink, FlaskConical, Database, GitBranch, ShieldAlert, Info } from 'lucide-react'
import DisclaimerBanner from '../components/DisclaimerBanner'

const PHASES = [
  {
    num: '1–2',
    title: 'Data Validation & EDA',
    desc: 'CDC BRFSS 2015 dataset loaded (253,680 rows × 22 columns). Zero missing values confirmed. Class imbalance documented: No Diabetes (84.2%), Prediabetes (1.8%), Diabetes (13.9%).',
    color: 'bg-blue-50 border-blue-200',
  },
  {
    num: '3',
    title: 'Preprocessing & Split',
    desc: 'StratifiedShuffleSplit (80/20, seed=42). Preprocessing pipeline: 14 binary passthrough, 1 continuous StandardScaler (BMI), 2 count passthrough (MentHlth, PhysHlth), 4 ordinal OrdinalEncoder. 253,680 total rows.',
    color: 'bg-indigo-50 border-indigo-200',
  },
  {
    num: '4.1',
    title: 'Baseline Benchmarking',
    desc: 'StratifiedKFold (5-fold). Identified 13.58% feature-profile overlap between train and validation — methodological limitation flagged.',
    color: 'bg-amber-50 border-amber-200',
  },
  {
    num: '4.2',
    title: 'Group-Aware Benchmarking',
    desc: 'StratifiedGroupKFold (5-fold) with deterministic murmur3-equivalent profile hashing. Zero group overlap verified. 4 models benchmarked. Logistic Regression selected (highest Macro F1, best Prediabetes recall).',
    color: 'bg-emerald-50 border-emerald-200',
  },
  {
    num: '4.3',
    title: 'Integrity Audit',
    desc: 'Independent verification of group isolation, OOF prediction integrity, metric consistency, and duplicate-profile handling. All checks passed.',
    color: 'bg-slate-50 border-slate-200',
  },
  {
    num: '5',
    title: 'Final Test-Set Evaluation',
    desc: 'One-time evaluation on frozen CDC test set (50,736 rows). Logistic Regression: Macro F1 0.4246, Balanced Acc. 51.72%, ROC-AUC 0.7707. Test set never used in development.',
    color: 'bg-purple-50 border-purple-200',
  },
  {
    num: '5.1–5.2',
    title: 'Governance & Reproducibility Audit',
    desc: 'Clinical language corrected. Split identity forensically resolved (column-order hash difference between Phase 3 DF and Phase 4.2 DF). All findings documented in reports.',
    color: 'bg-rose-50 border-rose-200',
  },
  {
    num: '6',
    title: 'Application Development',
    desc: 'FastAPI backend (Python 3.11+) + React/Vite/Tailwind frontend. Frozen Phase 4.2 LogisticRegression artifact used for inference. No retraining or test-set access during this phase.',
    color: 'bg-cyan-50 border-cyan-200',
  },
]

const LIMITATIONS = [
  'Class 1 (Prediabetes) is severely underrepresented (1.8% of data). Precision is 0.155 and recall 0.309 — many misclassifications are expected.',
  'The model uses self-reported survey data which is subject to recall bias and reporting errors.',
  'The CDC BRFSS 2015 data is from a specific US population and time period. Generalization to other populations or time periods is unknown.',
  'Macro F1 of 0.4246 reflects the difficulty of the 3-class imbalanced problem — this is not a high-accuracy clinical tool.',
  'Zero test-set degradation cannot be claimed without further evaluation. OOF and test metrics differ due to different split methods.',
  'This prototype does not account for medication, lab values, family history, or temporal clinical trends.',
]

export default function AboutMethodology() {
  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
      <div className="mb-8 flex items-start gap-3">
        <div className="w-10 h-10 bg-blue-50 rounded-xl flex items-center justify-center flex-shrink-0">
          <Info className="w-5 h-5 text-blue-600" />
        </div>
        <div>
          <h1 className="section-title">About &amp; Methodology</h1>
          <p className="section-subtitle">
            Research prototype pipeline — CDC BRFSS 2015 three-class diabetes risk classification.
            All development stages, design decisions, and limitations are documented here.
          </p>
        </div>
      </div>

      <DisclaimerBanner className="mb-8" />

      {/* Dataset */}
      <div className="card mb-6">
        <div className="flex items-center gap-2 mb-3">
          <Database className="w-5 h-5 text-blue-600" />
          <h2 className="font-semibold text-slate-800">Dataset</h2>
        </div>
        <p className="text-sm text-slate-600 leading-relaxed">
          <strong>CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015</strong> — a nationwide
          US telephone survey collecting health-related risk behaviors. This version contains
          253,680 responses with 21 health indicator features and a 3-class diabetes target:
          0 (No Diabetes), 1 (Prediabetes), 2 (Diabetes).
        </p>
        <a
          href="https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset"
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1.5 text-sm text-blue-600 hover:text-blue-700 mt-3 font-medium"
        >
          Kaggle: CDC Diabetes Health Indicators Dataset
          <ExternalLink className="w-3.5 h-3.5" />
        </a>
      </div>

      {/* Pipeline phases */}
      <div className="mb-6">
        <div className="flex items-center gap-2 mb-4">
          <FlaskConical className="w-5 h-5 text-blue-600" />
          <h2 className="font-semibold text-slate-800 text-lg">Pipeline Phases</h2>
        </div>
        <div className="space-y-3">
          {PHASES.map(({ num, title, desc, color }) => (
            <div key={num} className={`rounded-xl border ${color} px-5 py-4`}>
              <div className="flex items-start gap-3">
                <span className="bg-white border border-slate-200 rounded-lg px-2 py-0.5 text-xs font-mono font-bold text-slate-600 flex-shrink-0 mt-0.5">
                  Phase {num}
                </span>
                <div>
                  <div className="font-semibold text-slate-800 text-sm mb-1">{title}</div>
                  <div className="text-xs text-slate-600 leading-relaxed">{desc}</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Model selection */}
      <div className="card mb-6">
        <div className="flex items-center gap-2 mb-3">
          <GitBranch className="w-5 h-5 text-blue-600" />
          <h2 className="font-semibold text-slate-800">Model Selection Rationale</h2>
        </div>
        <p className="text-sm text-slate-600 leading-relaxed mb-3">
          Logistic Regression was selected over Decision Tree, Random Forest, and XGBoost based on
          Phase 4.2 group-aware OOF results:
        </p>
        <ul className="text-sm text-slate-600 space-y-1 list-disc list-inside">
          <li>Highest Macro F1 (0.4091) among all benchmarked models</li>
          <li>Highest Prediabetes recall (most important for population-level screening)</li>
          <li>Superior interpretability for a research prototype context</li>
          <li>Competitive ROC-AUC (0.7658) — only marginally below XGBoost</li>
          <li>Balanced class weighting applied to address severe class imbalance</li>
        </ul>
      </div>

      {/* Limitations */}
      <div className="card mb-8">
        <div className="flex items-center gap-2 mb-3">
          <ShieldAlert className="w-5 h-5 text-amber-500" />
          <h2 className="font-semibold text-slate-800">Known Limitations</h2>
        </div>
        <ul className="space-y-2.5">
          {LIMITATIONS.map((lim, i) => (
            <li key={i} className="flex gap-3 text-sm text-slate-600">
              <span className="flex-shrink-0 w-5 h-5 bg-amber-100 text-amber-700 rounded-full flex items-center justify-center text-xs font-bold">{i + 1}</span>
              {lim}
            </li>
          ))}
        </ul>
      </div>

      {/* Repository */}
      <div className="card">
        <h2 className="font-semibold text-slate-800 mb-2">Source Code</h2>
        <a
          href="https://github.com/AjayKumarKR07/Advanced-Diabetes-Risk-Prediction-System"
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-2 text-sm text-blue-600 hover:text-blue-700 font-medium"
        >
          github.com/AjayKumarKR07/Advanced-Diabetes-Risk-Prediction-System
          <ExternalLink className="w-3.5 h-3.5" />
        </a>
        <p className="text-xs text-slate-400 mt-2">
          Full pipeline code, audit reports, model artifacts, and documentation.
        </p>
      </div>
    </div>
  )
}

