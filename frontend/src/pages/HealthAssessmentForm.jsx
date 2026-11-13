/**
 * src/pages/HealthAssessmentForm.jsx
 * ====================================
 * 21-field CDC input form with range-validated controls.
 * Navigates to PredictionResult with the API response on submit.
 */

import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { predict } from '../services/api'
import { ClipboardList, ChevronDown, Loader2, AlertCircle } from 'lucide-react'
import DisclaimerBanner from '../components/DisclaimerBanner'

// ── Field definitions ─────────────────────────────────────────────────────

const BINARY_FIELDS = [
  { key: 'HighBP',               label: 'High Blood Pressure',          desc: 'Diagnosed with high blood pressure' },
  { key: 'HighChol',             label: 'High Cholesterol',             desc: 'Diagnosed with high cholesterol' },
  { key: 'CholCheck',            label: 'Cholesterol Checked',          desc: 'Cholesterol checked in past 5 years' },
  { key: 'Smoker',               label: 'Smoker',                       desc: 'Smoked at least 100 cigarettes in lifetime' },
  { key: 'Stroke',               label: 'Stroke History',               desc: 'Ever told you had a stroke' },
  { key: 'HeartDiseaseorAttack', label: 'Heart Disease / Attack',       desc: 'Coronary heart disease or heart attack' },
  { key: 'PhysActivity',         label: 'Physical Activity',            desc: 'Physical activity in past 30 days (not job-related)' },
  { key: 'Fruits',               label: 'Fruits Daily',                 desc: 'Consume fruit 1+ times per day' },
  { key: 'Veggies',              label: 'Vegetables Daily',             desc: 'Consume vegetables 1+ times per day' },
  { key: 'HvyAlcoholConsump',    label: 'Heavy Alcohol',                desc: 'Heavy alcohol consumer (men >14/wk, women >7/wk)' },
  { key: 'AnyHealthcare',        label: 'Health Coverage',              desc: 'Have any health care coverage' },
  { key: 'NoDocbcCost',          label: 'Avoided Doctor (Cost)',        desc: 'Could not see doctor due to cost in past 12 months' },
  { key: 'DiffWalk',             label: 'Difficulty Walking',           desc: 'Serious difficulty walking or climbing stairs' },
  { key: 'Sex',                  label: 'Sex',                          desc: '0 = Female, 1 = Male' },
]

const SELECT_FIELDS = [
  {
    key: 'GenHlth', label: 'General Health (1–5)', desc: 'Self-rated general health',
    options: [
      { value: 1, label: '1 — Excellent' },
      { value: 2, label: '2 — Very Good' },
      { value: 3, label: '3 — Good' },
      { value: 4, label: '4 — Fair' },
      { value: 5, label: '5 — Poor' },
    ],
  },
  {
    key: 'Age', label: 'Age Category (1–13)', desc: 'Age bracket',
    options: [
      { value: 1,  label: '1 — 18–24' },
      { value: 2,  label: '2 — 25–29' },
      { value: 3,  label: '3 — 30–34' },
      { value: 4,  label: '4 — 35–39' },
      { value: 5,  label: '5 — 40–44' },
      { value: 6,  label: '6 — 45–49' },
      { value: 7,  label: '7 — 50–54' },
      { value: 8,  label: '8 — 55–59' },
      { value: 9,  label: '9 — 60–64' },
      { value: 10, label: '10 — 65–69' },
      { value: 11, label: '11 — 70–74' },
      { value: 12, label: '12 — 75–79' },
      { value: 13, label: '13 — 80 or older' },
    ],
  },
  {
    key: 'Education', label: 'Education Level (1–6)', desc: 'Highest education completed',
    options: [
      { value: 1, label: '1 — Never attended / Kindergarten' },
      { value: 2, label: '2 — Grades 1–8 (Elementary)' },
      { value: 3, label: '3 — Grades 9–11 (Some high school)' },
      { value: 4, label: '4 — Grade 12 / GED' },
      { value: 5, label: '5 — Some college / technical school' },
      { value: 6, label: '6 — College graduate (4+ years)' },
    ],
  },
  {
    key: 'Income', label: 'Income Level (1–8)', desc: 'Annual household income category',
    options: [
      { value: 1, label: '1 — Less than $10,000' },
      { value: 2, label: '2 — $10,000 – $14,999' },
      { value: 3, label: '3 — $15,000 – $19,999' },
      { value: 4, label: '4 — $20,000 – $24,999' },
      { value: 5, label: '5 — $25,000 – $34,999' },
      { value: 6, label: '6 — $35,000 – $49,999' },
      { value: 7, label: '7 — $50,000 – $74,999' },
      { value: 8, label: '8 — $75,000 or more' },
    ],
  },
]

// ── Default form state ─────────────────────────────────────────────────────

const DEFAULT_STATE = {
  HighBP: 0, HighChol: 0, CholCheck: 1, BMI: 25.0,
  Smoker: 0, Stroke: 0, HeartDiseaseorAttack: 0, PhysActivity: 1,
  Fruits: 1, Veggies: 1, HvyAlcoholConsump: 0, AnyHealthcare: 1,
  NoDocbcCost: 0, GenHlth: 3, MentHlth: 0, PhysHlth: 0,
  DiffWalk: 0, Sex: 0, Age: 5, Education: 5, Income: 5,
}

// ── Component ──────────────────────────────────────────────────────────────

function BinaryToggle({ label, desc, value, onChange }) {
  return (
    <div className="flex items-center justify-between p-4 rounded-xl border border-slate-200 hover:border-blue-200 hover:bg-blue-50/30 transition-colors">
      <div>
        <div className="text-sm font-medium text-slate-700">{label}</div>
        <div className="text-xs text-slate-400 mt-0.5">{desc}</div>
      </div>
      <button
        type="button"
        onClick={() => onChange(value === 0 ? 1 : 0)}
        className={`relative w-14 h-7 rounded-full transition-colors duration-200 focus:outline-none ${
          value === 1 ? 'bg-blue-600' : 'bg-slate-200'
        }`}
      >
        <span
          className={`absolute top-1 w-5 h-5 bg-white rounded-full shadow transition-transform duration-200 ${
            value === 1 ? 'translate-x-7' : 'translate-x-1'
          }`}
        />
      </button>
    </div>
  )
}

export default function HealthAssessmentForm() {
  const navigate = useNavigate()
  const [form, setForm] = useState({ ...DEFAULT_STATE })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const setBinary = (key, val) => setForm(f => ({ ...f, [key]: val }))
  const setVal    = (key, val) => setForm(f => ({ ...f, [key]: val }))

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    try {
      const payload = { ...form, BMI: parseFloat(form.BMI) }
      const result = await predict(payload)
      navigate('/result', { state: { result, inputs: form } })
    } catch (err) {
      const detail = err.response?.data?.message || err.message || 'Unknown error'
      setError(`Prediction failed: ${detail}`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
      <div className="mb-8 flex items-start gap-3">
        <div className="w-10 h-10 bg-blue-50 rounded-xl flex items-center justify-center flex-shrink-0">
          <ClipboardList className="w-5 h-5 text-blue-600" />
        </div>
        <div>
          <h1 className="section-title">Health Risk Assessment</h1>
          <p className="section-subtitle">
            Complete all 21 CDC BRFSS health indicators below.
            All fields are required. This prototype uses a Logistic Regression model
            trained with group-aware cross-validation.
          </p>
        </div>
      </div>

      <DisclaimerBanner className="mb-8" />

      <form onSubmit={handleSubmit} className="space-y-8">
        {/* BMI */}
        <div className="card">
          <h2 className="font-semibold text-slate-800 mb-4">Body Measurements</h2>
          <div>
            <label className="label">BMI (Body Mass Index) — 10.0 to 100.0</label>
            <div className="flex items-center gap-4">
              <input
                type="range" min="10" max="100" step="0.5"
                value={form.BMI}
                onChange={e => setVal('BMI', parseFloat(e.target.value))}
                className="flex-1 accent-blue-600"
              />
              <div className="w-20 text-center font-bold text-blue-700 text-lg">
                {parseFloat(form.BMI).toFixed(1)}
              </div>
            </div>
            <div className="flex justify-between text-xs text-slate-400 mt-1">
              <span>10 (Very low)</span>
              <span>55 (Obese III)</span>
              <span>100 (Max)</span>
            </div>
          </div>
        </div>

        {/* Count fields */}
        <div className="card">
          <h2 className="font-semibold text-slate-800 mb-4">Health Days (Last 30 Days)</h2>
          <div className="grid sm:grid-cols-2 gap-6">
            {[
              { key: 'MentHlth', label: 'Poor Mental Health Days (0–30)' },
              { key: 'PhysHlth', label: 'Poor Physical Health Days (0–30)' },
            ].map(({ key, label }) => (
              <div key={key}>
                <label className="label">{label}</label>
                <div className="flex items-center gap-4">
                  <input
                    type="range" min="0" max="30" step="1"
                    value={form[key]}
                    onChange={e => setVal(key, parseInt(e.target.value))}
                    className="flex-1 accent-blue-600"
                  />
                  <span className="w-10 text-center font-bold text-slate-700">{form[key]}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Select fields */}
        <div className="card">
          <h2 className="font-semibold text-slate-800 mb-4">Demographic &amp; Lifestyle Categories</h2>
          <div className="grid sm:grid-cols-2 gap-5">
            {SELECT_FIELDS.map(({ key, label, desc, options }) => (
              <div key={key}>
                <label className="label">{label}</label>
                <div className="relative">
                  <select
                    value={form[key]}
                    onChange={e => setVal(key, parseInt(e.target.value))}
                    className="input-field pr-10 appearance-none"
                  >
                    {options.map(o => (
                      <option key={o.value} value={o.value}>{o.label}</option>
                    ))}
                  </select>
                  <ChevronDown className="absolute right-3 top-3.5 w-4 h-4 text-slate-400 pointer-events-none" />
                </div>
                <p className="text-xs text-slate-400 mt-1">{desc}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Binary toggles */}
        <div className="card">
          <h2 className="font-semibold text-slate-800 mb-4">Health Conditions &amp; Behaviours</h2>
          <div className="grid sm:grid-cols-2 gap-3">
            {BINARY_FIELDS.map(({ key, label, desc }) => (
              <BinaryToggle
                key={key}
                label={label}
                desc={desc}
                value={form[key]}
                onChange={val => setBinary(key, val)}
              />
            ))}
          </div>
        </div>

        {error && (
          <div className="flex items-start gap-3 bg-red-50 border border-red-200 rounded-xl px-4 py-3 text-sm text-red-700">
            <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
            {error}
          </div>
        )}

        <div className="flex justify-end">
          <button
            type="submit"
            disabled={loading}
            className="btn-primary flex items-center gap-2 text-base px-8 py-4"
          >
            {loading ? (
              <><Loader2 className="w-5 h-5 animate-spin" /> Analysing…</>
            ) : (
              'Get Risk Assessment →'
            )}
          </button>
        </div>
      </form>
    </div>
  )
}

