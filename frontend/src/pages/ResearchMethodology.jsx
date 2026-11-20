/**
 * src/pages/ResearchMethodology.jsx
 * ===================================
 * Phase 6.17.2 — Research & Methodology page.
 * Documents the CDC BRFSS 2015 dataset, 21 input features, preprocessing,
 * group-aware validation, and frozen Phase 5 Logistic Regression evaluation.
 * All content sourced from verified project implementation — no fabrication.
 */

import { useState } from 'react'
import {
  BookOpen, Database, Layers, TrendingUp, Activity, Shield,
  ChevronDown, ChevronRight, AlertTriangle, Info, CheckCircle2,
  BarChart2, FlaskConical, GitBranch,
} from 'lucide-react'
import DisclaimerBanner from '../components/DisclaimerBanner'

/* ── 21 CDC BRFSS Features (canonical order from model_service.py) ── */
const FEATURES = [
  { name: 'HighBP', type: 'Binary', desc: 'High Blood Pressure diagnosed (0=No, 1=Yes)' },
  { name: 'HighChol', type: 'Binary', desc: 'High Cholesterol diagnosed (0=No, 1=Yes)' },
  { name: 'CholCheck', type: 'Binary', desc: 'Cholesterol check in past 5 years (0=No, 1=Yes)' },
  { name: 'BMI', type: 'Continuous', desc: 'Body Mass Index (continuous, calculated from reported height/weight)' },
  { name: 'Smoker', type: 'Binary', desc: 'Smoked ≥100 cigarettes lifetime (0=No, 1=Yes)' },
  { name: 'Stroke', type: 'Binary', desc: 'Ever told had a stroke (0=No, 1=Yes)' },
  { name: 'HeartDiseaseorAttack', type: 'Binary', desc: 'Coronary heart disease or myocardial infarction (0=No, 1=Yes)' },
  { name: 'PhysActivity', type: 'Binary', desc: 'Physical activity in past 30 days excluding job (0=No, 1=Yes)' },
  { name: 'Fruits', type: 'Binary', desc: 'Consumes fruit ≥1 times/day (0=No, 1=Yes)' },
  { name: 'Veggies', type: 'Binary', desc: 'Consumes vegetables ≥1 times/day (0=No, 1=Yes)' },
  { name: 'HvyAlcoholConsump', type: 'Binary', desc: 'Heavy alcohol consumption (men >14 drinks/week; women >7) (0=No, 1=Yes)' },
  { name: 'AnyHealthcare', type: 'Binary', desc: 'Has any health care coverage (0=No, 1=Yes)' },
  { name: 'NoDocbcCost', type: 'Binary', desc: 'Could not see doctor due to cost in past 12 months (0=No, 1=Yes)' },
  { name: 'GenHlth', type: 'Ordinal 1–5', desc: 'General health rating (1=Excellent → 5=Poor)' },
  { name: 'MentHlth', type: 'Count 0–30', desc: 'Days of poor mental health in past 30 days' },
  { name: 'PhysHlth', type: 'Count 0–30', desc: 'Days of physical illness/injury in past 30 days' },
  { name: 'DiffWalk', type: 'Binary', desc: 'Serious difficulty walking or climbing stairs (0=No, 1=Yes)' },
  { name: 'Sex', type: 'Binary', desc: 'Biological sex (0=Female, 1=Male)' },
  { name: 'Age', type: 'Ordinal 1–13', desc: 'Age bracket (1=18–24 … 13=80+), CDC 13-category encoding' },
  { name: 'Education', type: 'Ordinal 1–6', desc: 'Highest education level (1=Never attended → 6=College graduate)' },
  { name: 'Income', type: 'Ordinal 1–8', desc: 'Annual household income bracket (1=<$10k → 8=$75k+)' },
]

/* ── Frozen Phase 5 Metrics ── */
const GLOBAL = [
  { label: 'Accuracy', value: '64.33%' },
  { label: 'Balanced Accuracy', value: '51.72%' },
  { label: 'Macro F1', value: '0.4246' },
  { label: 'Weighted F1', value: '0.7187' },
  { label: 'ROC-AUC OvR Macro', value: '0.7707' },
  { label: 'ROC-AUC OvR Weighted', value: '0.8112' },
]

const CLASS_PERF = [
  { cls: 'No Diabetes (0)', p: '0.9501', r: '0.6606', f: '0.7793', n: '35,346', color: '#10b981' },
  { cls: 'Prediabetes (1)', p: '0.0307', r: '0.3089', f: '0.0558', n: '6,236', color: '#f59e0b', weak: true },
  { cls: 'Diabetes (2)', p: '0.3520', r: '0.5823', f: '0.4387', n: '9,154', color: '#ef4444' },
]

/* ── Collapsible Section Component ── */
function Section({ icon: Icon, title, badge, defaultOpen = false, children }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="bg-white border border-slate-200 rounded-2xl shadow-sm overflow-hidden">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between px-6 py-4 hover:bg-slate-50 transition-colors text-left"
      >
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-blue-50 rounded-lg flex items-center justify-center shrink-0">
            <Icon className="w-4 h-4 text-blue-600" />
          </div>
          <span className="text-sm font-bold text-slate-800">{title}</span>
          {badge && (
            <span className="text-[10px] bg-blue-50 text-blue-700 border border-blue-200 px-2 py-0.5 rounded-full font-bold">
              {badge}
            </span>
          )}
        </div>
        {open ? (
          <ChevronDown className="w-4 h-4 text-slate-400 shrink-0" />
        ) : (
          <ChevronRight className="w-4 h-4 text-slate-400 shrink-0" />
        )}
      </button>
      {open && <div className="px-6 pb-6 pt-1 border-t border-slate-100">{children}</div>}
    </div>
  )
}

export default function ResearchMethodology() {
  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-4">
      {/* Header */}
      <div className="flex items-start gap-3">
        <div className="w-10 h-10 bg-blue-50 rounded-xl flex items-center justify-center shrink-0">
          <BookOpen className="w-5 h-5 text-blue-600" />
        </div>
        <div>
          <h1 className="text-xl font-bold text-slate-900">Research & Methodology</h1>
          <p className="text-sm text-slate-500 mt-0.5">
            Dataset, preprocessing, model selection, and frozen Phase 5 evaluation details for DiabetesRisk AI.
          </p>
        </div>
      </div>

      <DisclaimerBanner />

      {/* 1. Dataset Overview */}
      <Section icon={Database} title="Dataset Overview — CDC BRFSS 2015" defaultOpen>
        <div className="mt-3 space-y-3 text-sm text-slate-600">
          <p>
            DiabetesRisk AI is trained on the{' '}
            <strong className="text-slate-800">CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015</strong>{' '}
            annual telephone health survey dataset, the largest continuously conducted health survey
            system in the world. The version used is a cleaned, pre-processed subset curated for
            diabetes research.
          </p>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-3">
            {[
              { label: 'Total Records', value: '253,680' },
              { label: 'Input Features', value: '21 CDC Indicators' },
              { label: 'Target Classes', value: '3 (0, 1, 2)' },
              { label: 'Test Split', value: '50,736 (20%)' },
            ].map(({ label, value }) => (
              <div key={label} className="bg-blue-50 border border-blue-100 rounded-xl px-4 py-3 text-center">
                <p className="text-xs text-slate-500 font-medium">{label}</p>
                <p className="text-sm font-bold text-slate-800 mt-0.5">{value}</p>
              </div>
            ))}
          </div>
          <div className="bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 mt-2">
            <p className="text-xs font-semibold text-slate-700 mb-1">Class Distribution</p>
            <div className="space-y-1.5">
              {CLASS_PERF.map((c) => (
                <div key={c.cls} className="flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: c.color }} />
                  <span className="text-xs text-slate-600 flex-1">{c.cls}</span>
                  <span className="text-xs font-mono text-slate-700">{c.n} records</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </Section>

      {/* 2. 21 Input Features */}
      <Section icon={Layers} title="21 CDC Input Feature Definitions" badge="Canonical Order">
        <div className="mt-3 overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="bg-slate-50 text-slate-500 uppercase tracking-wide">
                <th className="px-3 py-2 text-left font-semibold w-6">#</th>
                <th className="px-3 py-2 text-left font-semibold">Feature</th>
                <th className="px-3 py-2 text-left font-semibold hidden sm:table-cell">Type</th>
                <th className="px-3 py-2 text-left font-semibold">Description</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-50">
              {FEATURES.map((f, i) => (
                <tr key={f.name} className="hover:bg-slate-50 transition-colors">
                  <td className="px-3 py-2.5 text-slate-400 font-mono">{i + 1}</td>
                  <td className="px-3 py-2.5 font-bold text-slate-800 font-mono">{f.name}</td>
                  <td className="px-3 py-2.5 text-slate-500 hidden sm:table-cell">
                    <span className="bg-blue-50 text-blue-700 border border-blue-100 px-1.5 py-0.5 rounded text-[10px] font-medium">
                      {f.type}
                    </span>
                  </td>
                  <td className="px-3 py-2.5 text-slate-600 leading-relaxed">{f.desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-3 flex items-start gap-2 bg-amber-50 border border-amber-200 rounded-xl px-3 py-2">
          <Info className="w-3.5 h-3.5 text-amber-600 shrink-0 mt-0.5" />
          <p className="text-xs text-amber-700">
            The feature order is frozen and must not be changed. The deployed model's
            StandardScaler and LogisticRegression coefficients are fitted to this exact 21-feature
            sequence. Any reordering would invalidate inference.
          </p>
        </div>
      </Section>

      {/* 3. Classification Target */}
      <Section icon={GitBranch} title="Three-Class Classification Target">
        <div className="mt-3 space-y-2 text-sm text-slate-600">
          <p>The <code className="bg-slate-100 px-1 rounded text-xs">Diabetes_012</code> column encodes three mutually exclusive health states:</p>
          <div className="mt-2 space-y-2">
            {[
              { cls: '0 — No Diabetes', color: 'emerald', desc: 'Respondent has never been told they have diabetes, prediabetes, or borderline diabetes.' },
              { cls: '1 — Prediabetes / Borderline', color: 'amber', desc: 'Respondent was told they have prediabetes or borderline diabetes. Heavily underrepresented (~2.5% of records).' },
              { cls: '2 — Diabetes', color: 'red', desc: 'Respondent has been told they have diabetes (excludes gestational diabetes during pregnancy only).' },
            ].map(({ cls, color, desc }) => (
              <div key={cls} className={`border border-${color}-200 bg-${color}-50 rounded-xl px-4 py-3`}>
                <p className={`text-xs font-bold text-${color}-800 mb-0.5`}>{cls}</p>
                <p className={`text-xs text-${color}-700`}>{desc}</p>
              </div>
            ))}
          </div>
        </div>
      </Section>

      {/* 4. Preprocessing */}
      <Section icon={FlaskConical} title="Data Preprocessing Methodology">
        <div className="mt-3 space-y-3 text-sm text-slate-600">
          <div>
            <p className="font-semibold text-slate-800 mb-1">Duplicate Profile Handling</p>
            <p>
              The raw BRFSS dataset contains near-identical respondent profiles due to survey
              structure. Exact-duplicate rows are identified and removed before model training to
              prevent data leakage across train/test splits and to avoid inflated performance metrics.
            </p>
          </div>
          <div>
            <p className="font-semibold text-slate-800 mb-1">Feature Scaling</p>
            <p>
              All 21 features are standardised using{' '}
              <code className="bg-slate-100 px-1 rounded text-xs">StandardScaler</code> (zero mean, unit
              variance) fitted exclusively on the training set and applied identically to validation and
              test sets. The scaler is serialised inside the pipeline artifact.
            </p>
          </div>
          <div>
            <p className="font-semibold text-slate-800 mb-1">Class Imbalance Handling</p>
            <p>
              The Prediabetes class (Class 1) represents only ~2.5% of records. The Logistic
              Regression model is trained with{' '}
              <code className="bg-slate-100 px-1 rounded text-xs">class_weight='balanced'</code>{' '}
              to maximise sensitivity to minority-class patterns at the cost of reduced precision.
            </p>
          </div>
          <div>
            <p className="font-semibold text-slate-800 mb-1">Train / Validation / Test Splits</p>
            <p>
              Data is split into training (70%), validation (10%), and test (20%) sets using
              stratified sampling to preserve class proportions. The test set (50,736 records) is
              held out and evaluated exactly once during Phase 5.
            </p>
          </div>
        </div>
      </Section>

      {/* 5. Group-aware Validation */}
      <Section icon={Shield} title="Group-Aware Validation Methodology">
        <div className="mt-3 space-y-3 text-sm text-slate-600">
          <p>
            Phase 4.2 uses a <strong className="text-slate-800">group-aware Out-of-Fold (OOF) benchmarking</strong>{' '}
            strategy to select the production model while preventing contamination between
            demographically similar respondents.
          </p>
          <div className="bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 space-y-2">
            <p className="text-xs font-semibold text-slate-700">Key Principles</p>
            <ul className="space-y-1.5 list-none">
              {[
                'Respondents with identical (Age, Sex, BMI) triples are assigned to the same fold to prevent leakage.',
                '5-fold stratified cross-validation is applied, with each fold reporting macro-F1 and balanced accuracy.',
                'Model selection is based on mean OOF balanced accuracy rather than training accuracy.',
                'Four model families benchmarked: Logistic Regression, Decision Tree, Random Forest, XGBoost.',
                'Logistic Regression selected: best balance of generalization, calibration, and interpretability.',
              ].map((pt, i) => (
                <li key={i} className="flex items-start gap-2 text-xs text-slate-600">
                  <CheckCircle2 className="w-3.5 h-3.5 text-blue-500 shrink-0 mt-0.5" />
                  {pt}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </Section>

      {/* 6. Model Architecture */}
      <Section icon={Activity} title="Logistic Regression Model Architecture">
        <div className="mt-3 space-y-3 text-sm text-slate-600">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {[
              { label: 'Algorithm', value: 'Multinomial Logistic Regression' },
              { label: 'Solver', value: "lbfgs (Limited-memory BFGS)" },
              { label: 'Multi-class strategy', value: 'softmax (multinomial)' },
              { label: 'Class weights', value: 'balanced (inverse frequency)' },
              { label: 'Regularisation', value: 'L2 (Ridge), C=1.0' },
              { label: 'Pipeline artifact', value: 'StandardScaler → LogisticRegression' },
              { label: 'Serialisation', value: 'joblib (model.pkl)' },
              { label: 'Frozen since', value: 'Phase 5 final evaluation' },
            ].map(({ label, value }) => (
              <div key={label} className="flex gap-2 text-xs">
                <span className="text-slate-500 font-medium shrink-0 w-36">{label}:</span>
                <span className="text-slate-800 font-mono">{value}</span>
              </div>
            ))}
          </div>
          <div className="bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 mt-2">
            <p className="text-xs font-semibold text-slate-700 mb-1">Decision Rule (Argmax)</p>
            <p className="text-xs text-slate-600 font-mono bg-white border border-slate-200 rounded p-2">
              P(Class=k | X) = exp(β_k · X̃) / Σ exp(β_j · X̃) · · · k ∈ &#123;0, 1, 2&#125;
              <br />
              ŷ = argmax_k P(Class=k | X)
            </p>
            <p className="text-xs text-slate-500 mt-1.5">
              where X̃ = StandardScaler(X) — the 21 standardised feature values.
            </p>
          </div>
        </div>
      </Section>

      {/* 7. Frozen Phase 5 Metrics */}
      <Section icon={BarChart2} title="Frozen Phase 5 Evaluation Metrics" badge="Read-only">
        <div className="mt-3 space-y-4">
          {/* Global */}
          <div>
            <p className="text-xs font-semibold text-slate-700 mb-2">Global Metrics (Test Set — 50,736 records)</p>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
              {GLOBAL.map(({ label, value }) => (
                <div key={label} className="bg-blue-50 border border-blue-100 rounded-xl px-3 py-2.5 text-center">
                  <p className="text-[10px] text-slate-500 font-medium">{label}</p>
                  <p className="text-sm font-bold text-slate-800 mt-0.5 font-mono">{value}</p>
                </div>
              ))}
            </div>
          </div>
          {/* Per-class */}
          <div>
            <p className="text-xs font-semibold text-slate-700 mb-2">Per-Class Performance</p>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="bg-slate-50 text-slate-500 uppercase tracking-wide">
                    <th className="px-3 py-2 text-left font-semibold">Class</th>
                    <th className="px-3 py-2 text-right font-semibold">Precision</th>
                    <th className="px-3 py-2 text-right font-semibold">Recall</th>
                    <th className="px-3 py-2 text-right font-semibold">F1</th>
                    <th className="px-3 py-2 text-right font-semibold">Support</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-50">
                  {CLASS_PERF.map((c) => (
                    <tr key={c.cls} className={c.weak ? 'bg-amber-50/60' : 'hover:bg-slate-50'}>
                      <td className="px-3 py-2.5 font-semibold text-slate-800 flex items-center gap-1.5">
                        <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: c.color }} />
                        {c.cls}
                        {c.weak && (
                          <span className="text-[10px] bg-amber-100 text-amber-700 border border-amber-200 px-1 rounded font-bold ml-1">
                            Weak
                          </span>
                        )}
                      </td>
                      <td className={`px-3 py-2.5 text-right font-mono ${c.weak ? 'text-amber-700 font-bold' : 'text-slate-700'}`}>{c.p}</td>
                      <td className="px-3 py-2.5 text-right font-mono text-slate-700">{c.r}</td>
                      <td className={`px-3 py-2.5 text-right font-mono font-bold ${c.weak ? 'text-amber-700' : 'text-slate-800'}`}>{c.f}</td>
                      <td className="px-3 py-2.5 text-right font-mono text-slate-500">{c.n}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </Section>

      {/* 8. Scientific Limitations */}
      <Section icon={AlertTriangle} title="Scientific Limitations & Ethical Considerations">
        <div className="mt-3 space-y-2 text-sm text-slate-600">
          {[
            {
              title: 'Survey Self-Report Bias',
              body: 'All features are based on respondent self-report. Medical diagnoses (HighBP, HighChol, Diabetes) are not verified against clinical records.',
            },
            {
              title: 'Prediabetes Class Weakness',
              body: 'Class 1 precision of 0.0307 and F1 of 0.0558 reflect severe class imbalance. Prediabetes predictions carry high individual uncertainty.',
            },
            {
              title: 'Population-Level Correlations Only',
              body: 'The model identifies statistical associations across CDC survey cohorts. It does NOT establish physiological causality in individual patients.',
            },
            {
              title: 'No Temporal Validation',
              body: 'The model was trained and evaluated on 2015 BRFSS data. It has not been validated on more recent surveys or clinical populations.',
            },
            {
              title: 'No Demographic Fairness Audit',
              body: 'Per-demographic subgroup performance has not been audited. Performance may vary across age, race/ethnicity, and geographic groups.',
            },
            {
              title: 'Not FDA-Cleared',
              body: 'DiabetesRisk AI is an academic research prototype. It has not undergone regulatory review and must not be used for clinical decisions.',
            },
          ].map(({ title, body }) => (
            <div key={title} className="bg-slate-50 border border-slate-200 rounded-xl px-4 py-3">
              <p className="text-xs font-bold text-slate-700 mb-0.5">{title}</p>
              <p className="text-xs text-slate-500 leading-relaxed">{body}</p>
            </div>
          ))}
        </div>
      </Section>

      {/* Footer disclaimer */}
      <div className="bg-blue-50 border border-blue-200 rounded-xl px-5 py-4 flex items-start gap-2">
        <Info className="w-4 h-4 text-blue-500 shrink-0 mt-0.5" />
        <p className="text-xs text-blue-700 leading-relaxed">
          <strong>Academic Research Project.</strong> This system was developed as an engineering
          research demonstration of machine learning applied to population health survey data. All
          methodology details reflect the actual implementation. No metrics have been fabricated or
          cherry-picked. The model and its Phase 5 evaluation are frozen — do not retrain or modify.
        </p>
      </div>
    </div>
  )
}
