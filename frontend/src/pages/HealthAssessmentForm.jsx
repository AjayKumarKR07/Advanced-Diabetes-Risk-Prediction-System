/**
 * src/pages/HealthAssessmentForm.jsx
 * ====================================
 * Professional Multi-Section Health Assessment Form for DiabetesRisk AI.
 * 
 * Features:
 * - 6-step guided wizard (Personal Info, Medical History, Lifestyle, General Health, Demographics, Review)
 * - Strict step-by-step form validation preventing forward navigation on invalid fields
 * - Unselected / empty fields trigger inline errors and red border highlighting
 * - Clicking "Review & Submit" directly checks all fields; prevents skipping and enforces "fill all fields"
 * - Step 6 submission pre-check: halts submission and redirects to incomplete step if any field is missing
 * - Stepper indicators show checkmarks ONLY when all questions in that step are actually answered
 * - Live error clearing on input change when a field becomes valid
 * - Backward navigation without validation, preserving all entered values
 * - Duplicate submission prevention (disabled button + loading state)
 * - Automatic conversion of raw chronological age to CDC BRFSS 13-category ordinal
 * - Prefills authenticated user profile (Full Name, Email)
 * - Transitions seamlessly to /result with model-estimated risk probabilities
 */

import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  User,
  Heart,
  Activity,
  Smile,
  Shield,
  FileCheck2,
  ChevronRight,
  ChevronLeft,
  CheckCircle2,
  AlertCircle,
  AlertTriangle,
  Loader2,
  Calendar,
  Mail,
  Info,
  Scale,
} from 'lucide-react'
import { predict } from '../services/api'
import { useAuth } from '../context/AuthContext'
import DisclaimerBanner from '../components/DisclaimerBanner'
import {
  convertAgeToCdcCategory,
  CDC_AGE_CATEGORIES,
  CDC_GENHLTH_OPTIONS,
  CDC_EDUCATION_OPTIONS,
  CDC_INCOME_OPTIONS,
  getBmiCategory,
} from '../utils/cdcFeatures'

export default function HealthAssessmentForm() {
  const navigate = useNavigate()
  const { user } = useAuth()

  // ── Step & Submission State ────────────────────────────────────────────────
  const [currentStep, setCurrentStep] = useState(1)
  const [errors, setErrors] = useState({})
  const [validationError, setValidationError] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [apiError, setApiError] = useState('')

  // ── Form State ─────────────────────────────────────────────────────────────
  // Personal Information
  const [personal, setPersonal] = useState({
    fullName: user?.full_name || '',
    email: user?.email || '',
    rawAge: '',
    sex: null, // null = unselected, 0 = Female, 1 = Male
    assessmentDate: new Date().toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    }),
  })

  // 21 CDC features state - initialized to null/empty so every question must be explicitly answered
  const [form, setForm] = useState({
    HighBP: null,
    HighChol: null,
    CholCheck: null,
    BMI: '',
    Smoker: null,
    Stroke: null,
    HeartDiseaseorAttack: null,
    PhysActivity: null,
    Fruits: null,
    Veggies: null,
    HvyAlcoholConsump: null,
    AnyHealthcare: null,
    NoDocbcCost: null,
    GenHlth: null,
    MentHlth: '',
    PhysHlth: '',
    DiffWalk: null,
    Sex: null,
    Age: null,
    Education: '',
    Income: '',
  })

  // Synchronize user profile if loaded asynchronously
  useEffect(() => {
    if (user) {
      setPersonal((prev) => ({
        ...prev,
        fullName: prev.fullName || user.full_name || '',
        email: user.email || '',
      }))
    }
  }, [user])

  // Synchronize sex and converted age into the 21-feature state when valid
  useEffect(() => {
    const cdcAge = convertAgeToCdcCategory(personal.rawAge)
    if (cdcAge !== null && (personal.sex === 0 || personal.sex === 1)) {
      setForm((prev) => ({
        ...prev,
        Age: cdcAge,
        Sex: parseInt(personal.sex, 10),
      }))
    }
  }, [personal.rawAge, personal.sex])

  // ── Step Completion Checker ────────────────────────────────────────────────
  const isStepComplete = (stepNum) => {
    if (stepNum === 1) {
      const name = (personal.fullName || '').trim()
      const rawAgeStr = String(personal.rawAge || '').trim()
      const ageNum = parseInt(rawAgeStr, 10)
      return (
        name.length >= 2 &&
        /^\d+$/.test(rawAgeStr) &&
        ageNum >= 18 &&
        ageNum <= 120 &&
        (personal.sex === 0 || personal.sex === 1)
      )
    }
    if (stepNum === 2) {
      return (
        (form.HighBP === 0 || form.HighBP === 1) &&
        (form.HighChol === 0 || form.HighChol === 1) &&
        (form.CholCheck === 0 || form.CholCheck === 1) &&
        (form.Stroke === 0 || form.Stroke === 1) &&
        (form.HeartDiseaseorAttack === 0 || form.HeartDiseaseorAttack === 1)
      )
    }
    if (stepNum === 3) {
      const bmiVal = parseFloat(form.BMI)
      return (
        !isNaN(bmiVal) &&
        bmiVal >= 10.0 &&
        bmiVal <= 100.0 &&
        (form.Smoker === 0 || form.Smoker === 1) &&
        (form.PhysActivity === 0 || form.PhysActivity === 1) &&
        (form.Fruits === 0 || form.Fruits === 1) &&
        (form.Veggies === 0 || form.Veggies === 1) &&
        (form.HvyAlcoholConsump === 0 || form.HvyAlcoholConsump === 1)
      )
    }
    if (stepNum === 4) {
      const genHlth = parseInt(form.GenHlth, 10)
      const physStr = String(form.PhysHlth === undefined || form.PhysHlth === null ? '' : form.PhysHlth).trim()
      const mentStr = String(form.MentHlth === undefined || form.MentHlth === null ? '' : form.MentHlth).trim()
      const physVal = parseInt(physStr, 10)
      const mentVal = parseInt(mentStr, 10)
      return (
        !isNaN(genHlth) && genHlth >= 1 && genHlth <= 5 &&
        physStr !== '' && /^\d+$/.test(physStr) && physVal >= 0 && physVal <= 30 &&
        mentStr !== '' && /^\d+$/.test(mentStr) && mentVal >= 0 && mentVal <= 30 &&
        (form.DiffWalk === 0 || form.DiffWalk === 1)
      )
    }
    if (stepNum === 5) {
      const edu = parseInt(form.Education, 10)
      const inc = parseInt(form.Income, 10)
      return (
        (form.AnyHealthcare === 0 || form.AnyHealthcare === 1) &&
        (form.NoDocbcCost === 0 || form.NoDocbcCost === 1) &&
        !isNaN(edu) && edu >= 1 && edu <= 6 &&
        !isNaN(inc) && inc >= 1 && inc <= 8
      )
    }
    return false
  }

  // ── Error Scrolling & Focus Helper ─────────────────────────────────────────
  const scrollToFirstError = (stepErrors) => {
    const errorKeys = Object.keys(stepErrors)
    if (errorKeys.length === 0) return
    const firstKey = errorKeys[0]
    setTimeout(() => {
      const el = document.getElementById(`field-${firstKey}`)
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'center' })
        if (typeof el.focus === 'function' && el.tagName !== 'DIV') {
          el.focus()
        } else {
          const innerInput = el.querySelector('input, select, button')
          innerInput?.focus?.()
        }
      }
    }, 60)
  }

  // ── Input Change Handlers with Live Error Clearing ────────────────────────
  const handlePersonalChange = (field, val) => {
    setPersonal((prev) => ({ ...prev, [field]: val }))

    setErrors((prev) => {
      if (!prev[field]) return prev
      const next = { ...prev }

      if (field === 'fullName') {
        const trimmed = (val || '').trim()
        if (trimmed.length >= 2) delete next.fullName
      } else if (field === 'rawAge') {
        const trimmed = String(val || '').trim()
        if (/^\d+$/.test(trimmed)) {
          const num = parseInt(trimmed, 10)
          if (num >= 18 && num <= 120) delete next.rawAge
        }
      } else if (field === 'sex') {
        if (val === 0 || val === 1) delete next.sex
      }

      return next
    })

    setValidationError('')
  }

  const setVal = (key, val) => {
    setForm((f) => ({ ...f, [key]: val }))

    setErrors((prev) => {
      if (!prev[key]) return prev
      const next = { ...prev }

      if (key === 'BMI') {
        const num = parseFloat(val)
        if (!isNaN(num) && num >= 10.0 && num <= 100.0) delete next.BMI
      } else if (key === 'PhysHlth' || key === 'MentHlth') {
        const num = parseInt(val, 10)
        if (!isNaN(num) && num >= 0 && num <= 30) delete next[key]
      } else {
        delete next[key]
      }

      return next
    })

    setValidationError('')
  }

  // ── Step Validation Logic ──────────────────────────────────────────────────
  const validateStep = (stepNumber) => {
    const stepErrors = {}

    // STEP 1: Personal Information
    if (stepNumber === 1) {
      const name = (personal.fullName || '').trim()
      if (!name) {
        stepErrors.fullName = 'Please enter your full name.'
      } else if (name.length < 2) {
        stepErrors.fullName = 'Full name must be at least 2 characters.'
      }

      const rawAgeStr = String(personal.rawAge || '').trim()
      if (!rawAgeStr) {
        stepErrors.rawAge = 'Please enter your age.'
      } else if (!/^\d+$/.test(rawAgeStr)) {
        stepErrors.rawAge = 'Age must be a whole integer number without decimals or text.'
      } else {
        const ageNum = parseInt(rawAgeStr, 10)
        if (ageNum < 18 || ageNum > 120) {
          stepErrors.rawAge = 'Age must be between 18 and 120 years (CDC BRFSS research standard is 18+).'
        } else {
          const cdcAge = convertAgeToCdcCategory(rawAgeStr)
          if (!cdcAge) {
            stepErrors.rawAge = 'Could not map chronological age to a valid CDC age bracket.'
          }
        }
      }

      if (personal.sex !== 0 && personal.sex !== 1) {
        stepErrors.sex = 'Please select your biological sex (Female or Male).'
      }
    }

    // STEP 2: Medical History
    if (stepNumber === 2) {
      const binaryFields = [
        { key: 'HighBP', label: 'High Blood Pressure' },
        { key: 'HighChol', label: 'High Blood Cholesterol' },
        { key: 'CholCheck', label: 'Cholesterol Check in Past 5 Years' },
        { key: 'Stroke', label: 'History of Stroke' },
        { key: 'HeartDiseaseorAttack', label: 'Coronary Heart Disease' },
      ]
      for (const f of binaryFields) {
        if (form[f.key] !== 0 && form[f.key] !== 1) {
          stepErrors[f.key] = `Please select Yes or No for ${f.label}.`
        }
      }
    }

    // STEP 3: Lifestyle & Physical Activity
    if (stepNumber === 3) {
      const bmiStr = String(form.BMI === undefined || form.BMI === null ? '' : form.BMI).trim()
      if (!bmiStr) {
        stepErrors.BMI = 'Please enter your Body Mass Index (BMI).'
      } else {
        const bmiVal = parseFloat(bmiStr)
        if (isNaN(bmiVal)) {
          stepErrors.BMI = 'BMI must be a valid numeric value.'
        } else if (bmiVal < 10.0 || bmiVal > 100.0) {
          stepErrors.BMI = 'BMI must be a realistic numeric value between 10.0 and 100.0 kg/m².'
        }
      }

      const binaryFields = [
        { key: 'Smoker', label: 'Smoking History (100+ Cigarettes)' },
        { key: 'PhysActivity', label: 'Physical Activity in Past 30 Days' },
        { key: 'Fruits', label: 'Daily Fruit Consumption' },
        { key: 'Veggies', label: 'Daily Vegetable Consumption' },
        { key: 'HvyAlcoholConsump', label: 'Heavy Alcohol Consumption' },
      ]
      for (const f of binaryFields) {
        if (form[f.key] !== 0 && form[f.key] !== 1) {
          stepErrors[f.key] = `Please select Yes or No for ${f.label}.`
        }
      }
    }

    // STEP 4: General Health & Functional Status
    if (stepNumber === 4) {
      const genHlth = parseInt(form.GenHlth, 10)
      if (isNaN(genHlth) || genHlth < 1 || genHlth > 5) {
        stepErrors.GenHlth = 'Please select your general health self-rating (1 to 5).'
      }

      const physStr = String(form.PhysHlth === undefined || form.PhysHlth === null ? '' : form.PhysHlth).trim()
      if (!physStr || !/^\d+$/.test(physStr)) {
        stepErrors.PhysHlth = 'Poor physical health days must be a whole integer between 0 and 30.'
      } else {
        const physVal = parseInt(physStr, 10)
        if (physVal < 0 || physVal > 30) {
          stepErrors.PhysHlth = 'Poor physical health days must be between 0 and 30.'
        }
      }

      const mentStr = String(form.MentHlth === undefined || form.MentHlth === null ? '' : form.MentHlth).trim()
      if (!mentStr || !/^\d+$/.test(mentStr)) {
        stepErrors.MentHlth = 'Poor mental health days must be a whole integer between 0 and 30.'
      } else {
        const mentVal = parseInt(mentStr, 10)
        if (mentVal < 0 || mentVal > 30) {
          stepErrors.MentHlth = 'Poor mental health days must be between 0 and 30.'
        }
      }

      if (form.DiffWalk !== 0 && form.DiffWalk !== 1) {
        stepErrors.DiffWalk = 'Please select Yes or No for difficulty walking or climbing stairs.'
      }
    }

    // STEP 5: Demographics & Healthcare Access
    if (stepNumber === 5) {
      if (form.AnyHealthcare !== 0 && form.AnyHealthcare !== 1) {
        stepErrors.AnyHealthcare = 'Please select Yes or No for health care coverage.'
      }
      if (form.NoDocbcCost !== 0 && form.NoDocbcCost !== 1) {
        stepErrors.NoDocbcCost = 'Please select Yes or No for avoiding doctor due to cost.'
      }

      const edu = parseInt(form.Education, 10)
      if (isNaN(edu) || edu < 1 || edu > 6) {
        stepErrors.Education = 'Please select your highest education level completed (1 to 6).'
      }

      const inc = parseInt(form.Income, 10)
      if (isNaN(inc) || inc < 1 || inc > 8) {
        stepErrors.Income = 'Please select your annual household income bracket (1 to 8).'
      }
    }

    setErrors(stepErrors)

    if (Object.keys(stepErrors).length > 0) {
      const firstMsg = Object.values(stepErrors)[0]
      setValidationError(firstMsg)
      scrollToFirstError(stepErrors)
      return false
    }

    setValidationError('')
    return true
  }

  // Validate all steps before final submission
  const validateAllSteps = () => {
    for (let s = 1; s <= 5; s++) {
      if (!validateStep(s)) {
        setCurrentStep(s)
        setValidationError('Please fill all fields across all steps before submitting your assessment.')
        return false
      }
    }
    return true
  }

  // ── Navigation Handlers ────────────────────────────────────────────────────
  const handleNext = () => {
    if (validateStep(currentStep)) {
      setErrors({})
      setValidationError('')
      setCurrentStep((s) => Math.min(6, s + 1))
      window.scrollTo({ top: 0, behavior: 'smooth' })
    }
  }

  const handlePrev = () => {
    setErrors({})
    setValidationError('')
    setCurrentStep((s) => Math.max(1, s - 1))
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  // Stepper Click Handler: enforces completing all preceding steps before advancing to Step 6
  const handleStepClick = (targetStep) => {
    if (targetStep < currentStep) {
      // Allow moving backward without validation
      setErrors({})
      setValidationError('')
      setCurrentStep(targetStep)
      window.scrollTo({ top: 0, behavior: 'smooth' })
      return
    }

    if (targetStep === currentStep) return

    // Validate every step from 1 up to targetStep - 1
    for (let s = 1; s < targetStep; s++) {
      if (!validateStep(s)) {
        setCurrentStep(s)
        setValidationError(
          targetStep === 6
            ? 'Please fill all fields across all steps before proceeding to Review & Submit.'
            : 'Please fill all required fields in this step before advancing.'
        )
        return
      }
    }

    setErrors({})
    setValidationError('')
    setCurrentStep(targetStep)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  // ── Form Submission ────────────────────────────────────────────────────────
  const handleSubmit = async (e) => {
    if (e) e.preventDefault()
    if (submitting) return // Prevent duplicate form submissions

    // Validate ALL steps from 1 to 5 before allowing submission
    for (let s = 1; s <= 5; s++) {
      const isValid = validateStep(s)
      if (!isValid) {
        setCurrentStep(s)
        setValidationError('Please fill all fields across all steps before submitting your assessment.')
        return
      }
    }

    setSubmitting(true)
    setApiError('')

    try {
      const cdcAge = convertAgeToCdcCategory(personal.rawAge)
      const payload = {
        HighBP: parseInt(form.HighBP, 10),
        HighChol: parseInt(form.HighChol, 10),
        CholCheck: parseInt(form.CholCheck, 10),
        BMI: parseFloat(form.BMI),
        Smoker: parseInt(form.Smoker, 10),
        Stroke: parseInt(form.Stroke, 10),
        HeartDiseaseorAttack: parseInt(form.HeartDiseaseorAttack, 10),
        PhysActivity: parseInt(form.PhysActivity, 10),
        Fruits: parseInt(form.Fruits, 10),
        Veggies: parseInt(form.Veggies, 10),
        HvyAlcoholConsump: parseInt(form.HvyAlcoholConsump, 10),
        AnyHealthcare: parseInt(form.AnyHealthcare, 10),
        NoDocbcCost: parseInt(form.NoDocbcCost, 10),
        GenHlth: parseInt(form.GenHlth, 10),
        MentHlth: parseInt(form.MentHlth, 10),
        PhysHlth: parseInt(form.PhysHlth, 10),
        DiffWalk: parseInt(form.DiffWalk, 10),
        Sex: parseInt(personal.sex, 10),
        Age: cdcAge,
        Education: parseInt(form.Education, 10),
        Income: parseInt(form.Income, 10),
      }

      // API client automatically attaches JWT bearer token from storage
      const result = await predict(payload)

      // Navigate to result view with model output and inputs
      navigate('/result', {
        state: {
          result,
          inputs: payload,
          personalInfo: {
            ...personal,
            cdcAgeCategory: cdcAge,
            cdcAgeLabel: CDC_AGE_CATEGORIES[cdcAge]?.bracket,
          },
        },
      })
    } catch (err) {
      const detail =
        err.response?.data?.detail ||
        err.response?.data?.message ||
        err.message ||
        'Failed to connect to the prediction engine.'
      setApiError(typeof detail === 'string' ? detail : JSON.stringify(detail))
    } finally {
      setSubmitting(false)
    }
  }

  // ── Step Navigation Metadata ───────────────────────────────────────────────
  const STEPS = [
    { num: 1, title: 'Personal Info', icon: User },
    { num: 2, title: 'Medical History', icon: Heart },
    { num: 3, title: 'Lifestyle & Body', icon: Activity },
    { num: 4, title: 'General Health', icon: Smile },
    { num: 5, title: 'Demographics', icon: Shield },
    { num: 6, title: 'Review & Submit', icon: FileCheck2 },
  ]

  const calculatedAgeCat = convertAgeToCdcCategory(personal.rawAge)
  const bmiInfo =
    form.BMI === '' || isNaN(parseFloat(form.BMI))
      ? { label: 'Enter BMI (10.0–100.0)', color: 'slate' }
      : getBmiCategory(form.BMI)

  const getToggleBtnClass = (fieldKey, optVal) => {
    const isSelected = form[fieldKey] === optVal
    const hasError = errors[fieldKey]
    if (isSelected) {
      return optVal === 1
        ? 'bg-blue-600 text-white shadow-sm font-bold ring-2 ring-blue-500/20'
        : 'bg-slate-800 text-white shadow-sm font-bold ring-2 ring-slate-700/20'
    }
    if (hasError) {
      return 'bg-red-50 dark:bg-red-950/40 text-red-700 dark:text-red-300 border border-red-300 dark:border-red-800 hover:bg-red-100 font-semibold'
    }
    return 'bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-700 font-semibold shadow-2xs'
  }

  const getSexBtnClass = (optVal) => {
    const isSelected = personal.sex === optVal
    const hasError = errors.sex
    if (isSelected) {
      return 'bg-blue-600 text-white shadow-sm font-bold ring-2 ring-blue-500/20'
    }
    if (hasError) {
      return 'bg-red-50 dark:bg-red-950/40 text-red-700 dark:text-red-300 border border-red-300 dark:border-red-800 hover:bg-red-100 font-semibold'
    }
    return 'bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-700 font-semibold shadow-2xs'
  }

  const getGenHlthBtnClass = (optValue) => {
    const isSelected = form.GenHlth === optValue
    const hasError = errors.GenHlth
    if (isSelected) {
      return 'bg-blue-600 text-white border-blue-600 shadow-md ring-2 ring-blue-500/20 font-bold'
    }
    if (hasError) {
      return 'bg-red-50 dark:bg-red-950/40 text-red-700 dark:text-red-300 border border-red-300 dark:border-red-800 hover:bg-red-100 font-semibold'
    }
    return 'bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-700 font-semibold shadow-2xs'
  }

  return (
    <div className="py-8 px-4 sm:px-6 lg:px-8 max-w-4xl mx-auto space-y-6">
      <div>
        {/* Header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-blue-50 dark:bg-blue-900/40 border border-blue-200 dark:border-blue-700/60 text-blue-700 dark:text-blue-300 text-xs font-semibold mb-3">
            <Activity className="w-3.5 h-3.5 text-blue-600 dark:text-blue-400" />
            <span>CDC BRFSS Health Indicator Protocol</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 dark:text-white tracking-tight">New Health Assessment</h1>
          <p className="text-sm text-slate-600 dark:text-slate-300 mt-2 max-w-xl mx-auto">
            Provide the 21 health and demographic indicators evaluated by our research machine learning model.
            All responses remain private and confidential.
          </p>
        </div>

        {/* Stepper Progress Bar */}
        <div className="bg-white dark:bg-slate-800 rounded-3xl border border-slate-200/90 dark:border-slate-700 shadow-sm p-4 sm:p-6 mb-6 text-slate-900 dark:text-white transition-colors">
          <div className="grid grid-cols-6 gap-2 text-center">
            {STEPS.map((s) => {
              const Icon = s.icon
              const isDone = isStepComplete(s.num)
              const isCurrent = currentStep === s.num
              return (
                <button
                  key={s.num}
                  type="button"
                  onClick={() => handleStepClick(s.num)}
                  className={`flex flex-col items-center group transition-colors ${
                    isCurrent
                      ? 'text-blue-600 dark:text-blue-400 font-bold'
                      : isDone
                      ? 'text-teal-600 dark:text-teal-400 font-semibold'
                      : 'text-slate-400 dark:text-slate-500 font-medium'
                  }`}
                  title={`Step ${s.num}: ${s.title}`}
                >
                  <div
                    className={`w-9 h-9 sm:w-10 sm:h-10 rounded-xl flex items-center justify-center text-sm font-bold transition-all mb-1.5 ${
                      isCurrent
                        ? 'bg-blue-600 text-white shadow-md shadow-blue-500/20 ring-4 ring-blue-50 dark:ring-blue-950/60'
                        : isDone
                        ? 'bg-teal-50 dark:bg-teal-950/60 text-teal-700 dark:text-teal-300 border border-teal-200 dark:border-teal-800'
                        : 'bg-slate-100 dark:bg-slate-700/60 text-slate-500 dark:text-slate-400 border border-slate-200 dark:border-slate-600'
                    }`}
                  >
                    {isDone ? <CheckCircle2 className="w-5 h-5 text-teal-600 dark:text-teal-400" /> : <Icon className="w-4 h-4 sm:w-5 sm:h-5" />}
                  </div>
                  <span className="text-xs hidden md:inline truncate max-w-[100px]">{s.title}</span>
                </button>
              )
            })}
          </div>

          <div className="w-full bg-slate-100 dark:bg-slate-700/60 h-2 rounded-full mt-4 overflow-hidden">
            <div
              className="bg-gradient-to-r from-blue-600 to-teal-500 h-full rounded-full transition-all duration-300"
              style={{ width: `${((currentStep - 1) / (STEPS.length - 1)) * 100}%` }}
            />
          </div>
          <div className="flex items-center justify-between mt-2 text-xs text-slate-500 dark:text-slate-400">
            <span className="font-semibold">Step {currentStep} of {STEPS.length} · {Math.round(((currentStep - 1) / (STEPS.length - 1)) * 100)}% complete</span>
            <span className="italic">~{Math.max(1, STEPS.length - currentStep)} min remaining</span>
          </div>
        </div>

        {/* Top Validation Error Banner */}
        {validationError && (
          <div className="mb-6 p-4 rounded-2xl bg-red-50 border border-red-200 flex items-start gap-3 text-red-700 text-sm animate-shake shadow-xs">
            <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5 text-red-600" />
            <div>
              <p className="font-bold text-red-800">Incomplete Assessment</p>
              <p className="text-xs text-red-700 mt-0.5 font-medium">{validationError}</p>
            </div>
          </div>
        )}

        {/* API Error Banner */}
        {apiError && (
          <div className="mb-6 p-4 rounded-2xl bg-rose-50 border border-rose-200 flex items-start gap-3 text-rose-800 text-sm">
            <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5 text-rose-600" />
            <div>
              <p className="font-semibold">Assessment Inference Failed</p>
              <p className="text-xs text-rose-700 mt-0.5">{apiError}</p>
            </div>
          </div>
        )}

        {/* Step Forms */}
        <div className="bg-white dark:bg-slate-800 rounded-3xl border border-slate-200/90 dark:border-slate-700 shadow-sm p-6 sm:p-8 mb-6 text-slate-900 dark:text-white transition-colors">
          {/* ═════════════════════════════════════════════════════════════════ */}
          {/* STEP 1: Personal Information                                      */}
          {/* ═════════════════════════════════════════════════════════════════ */}
          {currentStep === 1 && (
            <div className="space-y-6">
              <div className="border-b border-slate-100 dark:border-slate-700 pb-4">
                <h2 className="text-xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
                  <User className="w-5 h-5 text-blue-600 dark:text-blue-400" /> Personal Information
                </h2>
                <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                  Demographic baseline. Your age will be automatically mapped to the CDC BRFSS model category.
                </p>
              </div>

              <div className="grid sm:grid-cols-2 gap-5">
                {/* Full Name */}
                <div id="field-fullName">
                  <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 uppercase tracking-wider mb-1.5">
                    Full Name <span className="text-red-500">*</span>
                  </label>
                  <div className="relative">
                    <User className={`absolute left-3.5 top-3.5 w-4 h-4 ${errors.fullName ? 'text-red-500' : 'text-slate-400 dark:text-slate-500'}`} />
                    <input
                      type="text"
                      value={personal.fullName}
                      onChange={(e) => handlePersonalChange('fullName', e.target.value)}
                      placeholder="Jane Doe"
                      className={`w-full pl-10 pr-4 py-2.5 rounded-xl border text-sm transition-all focus:outline-none ${
                        errors.fullName
                          ? 'border-red-500 bg-red-50/20 text-red-900 dark:text-red-300 focus:ring-2 focus:ring-red-300'
                          : 'border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900/80 focus:ring-2 focus:ring-blue-500 text-slate-800 dark:text-white placeholder-slate-400 dark:placeholder-slate-500'
                      }`}
                    />
                  </div>
                  {errors.fullName ? (
                    <p id="err-fullName" className="text-xs font-semibold text-red-600 mt-1.5 flex items-center gap-1">
                      <AlertCircle className="w-3.5 h-3.5 flex-shrink-0" />
                      <span>{errors.fullName}</span>
                    </p>
                  ) : (
                    <p className="text-xs text-slate-400 dark:text-slate-500 mt-1">Prefilled from your account profile; editable for this assessment.</p>
                  )}
                </div>

                {/* Email (Read-only) */}
                <div>
                  <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 uppercase tracking-wider mb-1.5">
                    Account Email (Verified)
                  </label>
                  <div className="relative">
                    <Mail className="absolute left-3.5 top-3.5 w-4 h-4 text-slate-400 dark:text-slate-500" />
                    <input
                      type="email"
                      value={personal.email}
                      readOnly
                      disabled
                      className="w-full pl-10 pr-4 py-2.5 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900/50 text-slate-500 dark:text-slate-400 text-sm cursor-not-allowed"
                    />
                  </div>
                  <p className="text-xs text-slate-400 dark:text-slate-500 mt-1">Linked directly to your authenticated user account.</p>
                </div>

                {/* Age (Years) with Live CDC Bracket Conversion */}
                <div id="field-rawAge">
                  <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 uppercase tracking-wider mb-1.5">
                    Age (Years) <span className="text-red-500">*</span>
                  </label>
                  <input
                    type="text"
                    inputMode="numeric"
                    value={personal.rawAge}
                    onChange={(e) => handlePersonalChange('rawAge', e.target.value)}
                    placeholder="e.g. 45"
                    className={`w-full px-4 py-2.5 rounded-xl border text-sm font-semibold transition-all focus:outline-none ${
                      errors.rawAge
                        ? 'border-red-500 bg-red-50/20 text-red-900 dark:text-red-300 focus:ring-2 focus:ring-red-300'
                        : 'border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900/80 focus:ring-2 focus:ring-blue-500 text-slate-800 dark:text-white placeholder-slate-400 dark:placeholder-slate-500'
                    }`}
                  />
                  {errors.rawAge && (
                    <p id="err-rawAge" className="text-xs font-semibold text-red-600 mt-1.5 flex items-center gap-1">
                      <AlertCircle className="w-3.5 h-3.5 flex-shrink-0" />
                      <span>{errors.rawAge}</span>
                    </p>
                  )}

                  <div className="mt-2 p-2.5 rounded-lg bg-blue-50/70 dark:bg-blue-950/40 border border-blue-200 dark:border-blue-800 text-xs text-blue-800 dark:text-blue-300 flex items-center justify-between">
                    <span className="font-medium">CDC Model Bracket:</span>
                    <span className="font-bold bg-white dark:bg-slate-800 px-2.5 py-0.5 rounded border border-blue-200 dark:border-blue-700 text-blue-700 dark:text-blue-300">
                      {calculatedAgeCat ? CDC_AGE_CATEGORIES[calculatedAgeCat]?.bracket : 'Enter age (18–120)'}
                    </span>
                  </div>
                  <p className="text-xs text-slate-400 dark:text-slate-500 mt-1">
                    CDC BRFSS uses 13 discrete age brackets. Our system automatically converts chronological age.
                  </p>
                </div>

                {/* Biological Sex */}
                <div id="field-sex">
                  <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 uppercase tracking-wider mb-1.5">
                    Sex <span className="text-red-500">*</span>
                  </label>
                  <div className="grid grid-cols-2 gap-3 mt-1">
                    {[
                      { val: 0, label: 'Female' },
                      { val: 1, label: 'Male' },
                    ].map((opt) => (
                      <button
                        key={opt.val}
                        type="button"
                        onClick={() => handlePersonalChange('sex', opt.val)}
                        className={`py-2.5 px-4 rounded-xl border text-sm transition-all ${getSexBtnClass(opt.val)}`}
                      >
                        {opt.label}
                      </button>
                    ))}
                  </div>
                  {errors.sex ? (
                    <p id="err-sex" className="text-xs font-semibold text-red-600 mt-1.5 flex items-center gap-1">
                      <AlertCircle className="w-3.5 h-3.5 flex-shrink-0" />
                      <span>{errors.sex}</span>
                    </p>
                  ) : (
                    <p className="text-xs text-slate-400 dark:text-slate-500 mt-2">Encoded according to CDC BRFSS specification (0 = Female, 1 = Male).</p>
                  )}
                </div>
              </div>

              {/* Assessment Timestamp */}
              <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-900/50 border border-slate-200/80 dark:border-slate-700 flex items-center justify-between text-xs text-slate-600 dark:text-slate-300">
                <span className="flex items-center gap-2">
                  <Calendar className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                  <strong>Assessment Timestamp:</strong>
                </span>
                <span className="font-mono bg-white dark:bg-slate-800 px-3 py-1 rounded-md border border-slate-200 dark:border-slate-700 font-medium text-slate-800 dark:text-slate-200">
                  {personal.assessmentDate}
                </span>
              </div>
            </div>
          )}

          {/* ═════════════════════════════════════════════════════════════════ */}
          {/* STEP 2: Medical History                                           */}
          {/* ═════════════════════════════════════════════════════════════════ */}
          {currentStep === 2 && (
            <div className="space-y-6">
              <div className="border-b border-slate-100 dark:border-slate-700 pb-4">
                <h2 className="text-xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
                  <Heart className="w-5 h-5 text-red-500" /> Medical &amp; Clinical History
                </h2>
                <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                  Cardiovascular and metabolic clinical indicators diagnosed by a healthcare provider.
                </p>
              </div>

              <div className="space-y-4">
                {[
                  {
                    key: 'HighBP',
                    label: 'High Blood Pressure (Hypertension)',
                    desc: 'Have you ever been told by a doctor, nurse, or healthcare professional that you have high blood pressure?',
                  },
                  {
                    key: 'HighChol',
                    label: 'High Blood Cholesterol',
                    desc: 'Have you ever been told by a doctor or health professional that your blood cholesterol is high?',
                  },
                  {
                    key: 'CholCheck',
                    label: 'Cholesterol Check in Past 5 Years',
                    desc: 'Has your blood cholesterol been checked by a healthcare provider within the last 5 years?',
                  },
                  {
                    key: 'Stroke',
                    label: 'History of Stroke',
                    desc: 'Have you ever been formally diagnosed with a stroke (cerebrovascular accident)?',
                  },
                  {
                    key: 'HeartDiseaseorAttack',
                    label: 'Coronary Heart Disease or Myocardial Infarction',
                    desc: 'Have you ever had coronary heart disease (CHD), angina, or a heart attack?',
                  },
                ].map((item) => (
                  <div
                    key={item.key}
                    id={`field-${item.key}`}
                    className={`p-4 rounded-xl border transition-all flex flex-col sm:flex-row sm:items-center justify-between gap-4 ${
                      errors[item.key]
                        ? 'border-red-400 bg-red-50/20 ring-1 ring-red-400'
                        : 'border-slate-200/80 dark:border-slate-700 bg-slate-50/70 dark:bg-slate-900/40 hover:border-blue-200 dark:hover:border-blue-800'
                    }`}
                  >
                    <div className="flex-1">
                      <p className="text-sm font-semibold text-slate-800 dark:text-slate-200">
                        {item.label} <span className="text-red-500">*</span>
                      </p>
                      <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">{item.desc}</p>
                      {errors[item.key] && (
                        <p className="text-xs font-semibold text-red-600 mt-1 flex items-center gap-1">
                          <AlertCircle className="w-3.5 h-3.5" />
                          <span>{errors[item.key]}</span>
                        </p>
                      )}
                    </div>
                    <div className="flex gap-2 self-end sm:self-center">
                      {[
                        { val: 0, label: 'No' },
                        { val: 1, label: 'Yes' },
                      ].map((opt) => (
                        <button
                          key={opt.val}
                          type="button"
                          onClick={() => setVal(item.key, opt.val)}
                          className={`w-16 py-2 rounded-lg text-xs transition-all ${getToggleBtnClass(item.key, opt.val)}`}
                        >
                          {opt.label}
                        </button>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* ═════════════════════════════════════════════════════════════════ */}
          {/* STEP 3: Lifestyle & Physical Activity                             */}
          {/* ═════════════════════════════════════════════════════════════════ */}
          {currentStep === 3 && (
            <div className="space-y-6">
              <div className="border-b border-slate-100 dark:border-slate-700 pb-4">
                <h2 className="text-xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
                  <Activity className="w-5 h-5 text-teal-600 dark:text-teal-400" /> Lifestyle, Nutrition &amp; Physical Activity
                </h2>
                <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                  Anthropometric and behavioral factors that significantly influence diabetes risk.
                </p>
              </div>

              {/* BMI Card */}
              <div
                id="field-BMI"
                className={`p-5 rounded-2xl border transition-all ${
                  errors.BMI
                    ? 'border-red-400 bg-red-50/20 ring-1 ring-red-400'
                    : 'border-slate-200/80 dark:border-slate-700 bg-slate-50/70 dark:bg-slate-900/40'
                }`}
              >
                <div className="flex items-center justify-between flex-wrap gap-2 mb-2">
                  <label className="text-sm font-bold text-slate-800 dark:text-slate-200 flex items-center gap-2">
                    <Scale className="w-4 h-4 text-blue-600 dark:text-blue-400" /> Body Mass Index (BMI, kg/m²) <span className="text-red-500">*</span>
                  </label>
                  <span
                    className={`text-xs font-bold px-3 py-1 rounded-full border ${
                      bmiInfo.color === 'emerald'
                        ? 'bg-emerald-50 dark:bg-emerald-950/60 text-emerald-700 dark:text-emerald-300 border-emerald-200 dark:border-emerald-800'
                        : bmiInfo.color === 'amber'
                        ? 'bg-amber-50 dark:bg-amber-950/60 text-amber-700 dark:text-amber-300 border-amber-200 dark:border-amber-800'
                        : bmiInfo.color === 'blue'
                        ? 'bg-blue-50 dark:bg-blue-950/60 text-blue-700 dark:text-blue-300 border-blue-200 dark:border-blue-800'
                        : bmiInfo.color === 'rose'
                        ? 'bg-rose-50 dark:bg-rose-950/60 text-rose-700 dark:text-rose-300 border-rose-200 dark:border-rose-800'
                        : 'bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 border-slate-200 dark:border-slate-700'
                    }`}
                  >
                    {bmiInfo.label}
                  </span>
                </div>

                <div className="grid sm:grid-cols-3 gap-4 items-center mt-3">
                  <div className="sm:col-span-2">
                    <input
                      type="range"
                      min="10.0"
                      max="60.0"
                      step="0.1"
                      value={form.BMI === '' || isNaN(parseFloat(form.BMI)) ? 25.0 : parseFloat(form.BMI)}
                      onChange={(e) => setVal('BMI', parseFloat(e.target.value))}
                      className="w-full accent-blue-600"
                    />
                    <div className="flex justify-between text-[11px] text-slate-400 dark:text-slate-500 mt-1">
                      <span>10.0 (Min)</span>
                      <span>18.5 (Normal)</span>
                      <span>25.0 (Overweight)</span>
                      <span>30.0 (Obese)</span>
                      <span>60.0+ (Severe)</span>
                    </div>
                  </div>
                  <div>
                    <input
                      type="number"
                      min="10.0"
                      max="100.0"
                      step="0.1"
                      placeholder="e.g. 25.0"
                      value={form.BMI}
                      onChange={(e) => setVal('BMI', e.target.value === '' ? '' : parseFloat(e.target.value))}
                      className={`w-full px-3 py-2 rounded-xl border text-center text-lg font-bold transition-all focus:outline-none ${
                        errors.BMI
                          ? 'border-red-500 bg-red-50 dark:bg-red-950/40 text-red-900 dark:text-red-300 focus:ring-2 focus:ring-red-300'
                          : 'border-slate-200 dark:border-slate-700 text-blue-700 dark:text-blue-400 bg-white dark:bg-slate-900/80 focus:ring-2 focus:ring-blue-500'
                      }`}
                    />
                  </div>
                </div>
                {errors.BMI && (
                  <p id="err-BMI" className="text-xs font-semibold text-red-600 mt-2 flex items-center gap-1">
                    <AlertCircle className="w-3.5 h-3.5 flex-shrink-0" />
                    <span>{errors.BMI}</span>
                  </p>
                )}
              </div>

              {/* Behavioral Questions */}
              <div className="space-y-4">
                {[
                  {
                    key: 'Smoker',
                    label: 'Smoking History (100+ Cigarettes Lifetime)',
                    desc: 'Have you smoked at least 100 cigarettes (5 packs) in your entire life? (Includes former smokers).',
                  },
                  {
                    key: 'PhysActivity',
                    label: 'Physical Activity in Past 30 Days',
                    desc: 'During the past month, other than your regular job, did you participate in any physical activities or exercises such as running, calisthenics, golf, gardening, or walking?',
                  },
                  {
                    key: 'Fruits',
                    label: 'Fruit Consumption Daily',
                    desc: 'Do you consume fruit (fresh, frozen, or 100% pure juice) 1 or more times per day?',
                  },
                  {
                    key: 'Veggies',
                    label: 'Vegetable Consumption Daily',
                    desc: 'Do you consume vegetables (including green salads and cooked vegetables) 1 or more times per day?',
                  },
                  {
                    key: 'HvyAlcoholConsump',
                    label: 'Heavy Alcohol Consumption',
                    desc: 'Adult men having more than 14 drinks/week; adult women having more than 7 drinks/week.',
                  },
                ].map((item) => (
                  <div
                    key={item.key}
                    id={`field-${item.key}`}
                    className={`p-4 rounded-xl border transition-all flex flex-col sm:flex-row sm:items-center justify-between gap-4 ${
                      errors[item.key]
                        ? 'border-red-400 bg-red-50/20 ring-1 ring-red-400'
                        : 'border-slate-200/80 dark:border-slate-700 bg-slate-50/70 dark:bg-slate-900/40 hover:border-blue-200 dark:hover:border-blue-800'
                    }`}
                  >
                    <div className="flex-1">
                      <p className="text-sm font-semibold text-slate-800 dark:text-slate-200">
                        {item.label} <span className="text-red-500">*</span>
                      </p>
                      <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">{item.desc}</p>
                      {errors[item.key] && (
                        <p className="text-xs font-semibold text-red-600 mt-1 flex items-center gap-1">
                          <AlertCircle className="w-3.5 h-3.5" />
                          <span>{errors[item.key]}</span>
                        </p>
                      )}
                    </div>
                    <div className="flex gap-2 self-end sm:self-center">
                      {[
                        { val: 0, label: 'No' },
                        { val: 1, label: 'Yes' },
                      ].map((opt) => (
                        <button
                          key={opt.val}
                          type="button"
                          onClick={() => setVal(item.key, opt.val)}
                          className={`w-16 py-2 rounded-lg text-xs transition-all ${getToggleBtnClass(item.key, opt.val)}`}
                        >
                          {opt.label}
                        </button>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* ═════════════════════════════════════════════════════════════════ */}
          {/* STEP 4: General Health & Functional Status                         */}
          {/* ═════════════════════════════════════════════════════════════════ */}
          {currentStep === 4 && (
            <div className="space-y-6">
              <div className="border-b border-slate-100 dark:border-slate-700 pb-4">
                <h2 className="text-xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
                  <Smile className="w-5 h-5 text-amber-500" /> General Health &amp; Functional Status
                </h2>
                <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                  Self-rated general wellness and days of impaired physical or mental health in the past 30 days.
                </p>
              </div>

              {/* General Health Dropdown */}
              <div id="field-GenHlth">
                <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 uppercase tracking-wider mb-1.5">
                  General Health Self-Assessment <span className="text-red-500">*</span>
                </label>
                <div className="grid sm:grid-cols-5 gap-2.5">
                  {CDC_GENHLTH_OPTIONS.map((opt) => (
                    <button
                      key={opt.value}
                      type="button"
                      onClick={() => setVal('GenHlth', opt.value)}
                      className={`p-3 rounded-xl border text-center transition-all ${getGenHlthBtnClass(opt.value)}`}
                    >
                      <div className="text-xs font-bold">{opt.label.split('—')[1]?.trim()}</div>
                      <div className={`text-[10px] mt-1 ${form.GenHlth === opt.value ? 'text-blue-100' : 'text-slate-400 dark:text-slate-400'}`}>
                        {opt.desc}
                      </div>
                    </button>
                  ))}
                </div>
                {errors.GenHlth && (
                  <p className="text-xs font-semibold text-red-600 mt-1.5 flex items-center gap-1">
                    <AlertCircle className="w-3.5 h-3.5" />
                    <span>{errors.GenHlth}</span>
                  </p>
                )}
              </div>

              {/* Health Days Sliders */}
              <div className="grid sm:grid-cols-2 gap-6 pt-2">
                {/* Physical Health Days */}
                <div
                  id="field-PhysHlth"
                  className={`p-4 rounded-xl border transition-all ${
                    errors.PhysHlth ? 'border-red-400 bg-red-50/20 ring-1 ring-red-400' : 'border-slate-200/80 dark:border-slate-700 bg-slate-50/70 dark:bg-slate-900/40'
                  }`}
                >
                  <div className="flex justify-between items-center mb-1">
                    <label className="text-xs font-bold text-slate-700 dark:text-slate-300 uppercase tracking-wider">
                      Poor Physical Health Days (0–30) <span className="text-red-500">*</span>
                    </label>
                    <span className="text-sm font-extrabold text-blue-600 dark:text-blue-400 font-mono">
                      {form.PhysHlth === '' ? 'Unset' : `${form.PhysHlth} days`}
                    </span>
                  </div>
                  <p className="text-xs text-slate-500 dark:text-slate-400 mb-3">
                    Number of days during the past 30 days when physical health (illness, injury) was not good.
                  </p>
                  <div className="flex items-center gap-3">
                    <input
                      type="range"
                      min="0"
                      max="30"
                      step="1"
                      value={form.PhysHlth === '' ? 0 : form.PhysHlth}
                      onChange={(e) => setVal('PhysHlth', parseInt(e.target.value, 10))}
                      className="w-full accent-blue-600"
                    />
                    <input
                      type="number"
                      min="0"
                      max="30"
                      placeholder="0"
                      value={form.PhysHlth}
                      onChange={(e) => setVal('PhysHlth', e.target.value === '' ? '' : parseInt(e.target.value, 10))}
                      className={`w-16 px-2 py-1 rounded-lg border text-center font-bold text-sm bg-white dark:bg-slate-900/80 ${
                        errors.PhysHlth ? 'border-red-500 text-red-700' : 'border-slate-300 dark:border-slate-700 text-slate-800 dark:text-white'
                      }`}
                    />
                  </div>
                  <div className="flex gap-2 mt-2">
                    {[0, 5, 15, 30].map((preset) => (
                      <button
                        key={preset}
                        type="button"
                        onClick={() => setVal('PhysHlth', preset)}
                        className={`text-[10px] px-2 py-0.5 rounded border ${
                          form.PhysHlth === preset
                            ? 'bg-blue-600 text-white border-blue-600'
                            : 'bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-700'
                        }`}
                      >
                        {preset === 0 ? '0 (None)' : `${preset}d`}
                      </button>
                    ))}
                  </div>
                  {errors.PhysHlth && (
                    <p className="text-xs font-semibold text-red-600 mt-2 flex items-center gap-1">
                      <AlertCircle className="w-3.5 h-3.5" />
                      <span>{errors.PhysHlth}</span>
                    </p>
                  )}
                </div>

                {/* Mental Health Days */}
                <div
                  id="field-MentHlth"
                  className={`p-4 rounded-xl border transition-all ${
                    errors.MentHlth ? 'border-red-400 bg-red-50/20 ring-1 ring-red-400' : 'border-slate-200/80 dark:border-slate-700 bg-slate-50/70 dark:bg-slate-900/40'
                  }`}
                >
                  <div className="flex justify-between items-center mb-1">
                    <label className="text-xs font-bold text-slate-700 dark:text-slate-300 uppercase tracking-wider">
                      Poor Mental Health Days (0–30) <span className="text-red-500">*</span>
                    </label>
                    <span className="text-sm font-extrabold text-purple-600 dark:text-purple-400 font-mono">
                      {form.MentHlth === '' ? 'Unset' : `${form.MentHlth} days`}
                    </span>
                  </div>
                  <p className="text-xs text-slate-500 dark:text-slate-400 mb-3">
                    Number of days during the past 30 days when mental health (stress, depression) was not good.
                  </p>
                  <div className="flex items-center gap-3">
                    <input
                      type="range"
                      min="0"
                      max="30"
                      step="1"
                      value={form.MentHlth === '' ? 0 : form.MentHlth}
                      onChange={(e) => setVal('MentHlth', parseInt(e.target.value, 10))}
                      className="w-full accent-purple-600"
                    />
                    <input
                      type="number"
                      min="0"
                      max="30"
                      placeholder="0"
                      value={form.MentHlth}
                      onChange={(e) => setVal('MentHlth', e.target.value === '' ? '' : parseInt(e.target.value, 10))}
                      className={`w-16 px-2 py-1 rounded-lg border text-center font-bold text-sm bg-white dark:bg-slate-900/80 ${
                        errors.MentHlth ? 'border-red-500 text-red-700' : 'border-slate-300 dark:border-slate-700 text-slate-800 dark:text-white'
                      }`}
                    />
                  </div>
                  <div className="flex gap-2 mt-2">
                    {[0, 5, 15, 30].map((preset) => (
                      <button
                        key={preset}
                        type="button"
                        onClick={() => setVal('MentHlth', preset)}
                        className={`text-[10px] px-2 py-0.5 rounded border ${
                          form.MentHlth === preset
                            ? 'bg-purple-600 text-white border-purple-600'
                            : 'bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-700'
                        }`}
                      >
                        {preset === 0 ? '0 (None)' : `${preset}d`}
                      </button>
                    ))}
                  </div>
                  {errors.MentHlth && (
                    <p className="text-xs font-semibold text-red-600 mt-2 flex items-center gap-1">
                      <AlertCircle className="w-3.5 h-3.5" />
                      <span>{errors.MentHlth}</span>
                    </p>
                  )}
                </div>
              </div>

              {/* Difficulty Walking */}
              <div
                id="field-DiffWalk"
                className={`p-4 rounded-xl border transition-all flex flex-col sm:flex-row sm:items-center justify-between gap-4 ${
                  errors.DiffWalk
                    ? 'border-red-400 bg-red-50/20 ring-1 ring-red-400'
                    : 'border-slate-200/80 dark:border-slate-700 bg-slate-50/70 dark:bg-slate-900/40 hover:border-blue-200 dark:hover:border-blue-800'
                }`}
              >
                <div className="flex-1">
                  <p className="text-sm font-semibold text-slate-800 dark:text-slate-200">
                    Difficulty Walking or Climbing Stairs <span className="text-red-500">*</span>
                  </p>
                  <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                    Do you have serious difficulty walking or climbing stairs due to health issues?
                  </p>
                  {errors.DiffWalk && (
                    <p className="text-xs font-semibold text-red-600 mt-1 flex items-center gap-1">
                      <AlertCircle className="w-3.5 h-3.5" />
                      <span>{errors.DiffWalk}</span>
                    </p>
                  )}
                </div>
                <div className="flex gap-2 self-end sm:self-center">
                  {[
                    { val: 0, label: 'No' },
                    { val: 1, label: 'Yes' },
                  ].map((opt) => (
                    <button
                      key={opt.val}
                      type="button"
                      onClick={() => setVal('DiffWalk', opt.val)}
                      className={`w-16 py-2 rounded-lg text-xs transition-all ${getToggleBtnClass('DiffWalk', opt.val)}`}
                    >
                      {opt.label}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* ═════════════════════════════════════════════════════════════════ */}
          {/* STEP 5: Demographics & Healthcare Access                          */}
          {/* ═════════════════════════════════════════════════════════════════ */}
          {currentStep === 5 && (
            <div className="space-y-6">
              <div className="border-b border-slate-100 dark:border-slate-700 pb-4">
                <h2 className="text-xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
                  <Shield className="w-5 h-5 text-indigo-600 dark:text-indigo-400" /> Demographics &amp; Healthcare Access
                </h2>
                <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                  Socioeconomic and insurance access indicators associated with diabetes screening frequency.
                </p>
              </div>

              {/* Healthcare Coverage */}
              <div className="space-y-4">
                {[
                  {
                    key: 'AnyHealthcare',
                    label: 'Health Care Coverage',
                    desc: 'Do you have any health care coverage (health insurance, prepaid plans such as HMOs, or government plans like Medicare/Medicaid)?',
                  },
                  {
                    key: 'NoDocbcCost',
                    label: 'Avoided Doctor Due to Cost in Past 12 Months',
                    desc: 'Was there a time in the past 12 months when you needed to see a doctor but could not because of cost?',
                  },
                ].map((item) => (
                  <div
                    key={item.key}
                    id={`field-${item.key}`}
                    className={`p-4 rounded-xl border transition-all flex flex-col sm:flex-row sm:items-center justify-between gap-4 ${
                      errors[item.key]
                        ? 'border-red-400 bg-red-50/20 ring-1 ring-red-400'
                        : 'border-slate-200/80 dark:border-slate-700 bg-slate-50/70 dark:bg-slate-900/40 hover:border-blue-200 dark:hover:border-blue-800'
                    }`}
                  >
                    <div className="flex-1">
                      <p className="text-sm font-semibold text-slate-800 dark:text-slate-200">
                        {item.label} <span className="text-red-500">*</span>
                      </p>
                      <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">{item.desc}</p>
                      {errors[item.key] && (
                        <p className="text-xs font-semibold text-red-600 mt-1 flex items-center gap-1">
                          <AlertCircle className="w-3.5 h-3.5" />
                          <span>{errors[item.key]}</span>
                        </p>
                      )}
                    </div>
                    <div className="flex gap-2 self-end sm:self-center">
                      {[
                        { val: 0, label: 'No' },
                        { val: 1, label: 'Yes' },
                      ].map((opt) => (
                        <button
                          key={opt.val}
                          type="button"
                          onClick={() => setVal(item.key, opt.val)}
                          className={`w-16 py-2 rounded-lg text-xs transition-all ${getToggleBtnClass(item.key, opt.val)}`}
                        >
                          {opt.label}
                        </button>
                      ))}
                    </div>
                  </div>
                ))}
              </div>

              {/* Education & Income Selectors */}
              <div className="grid sm:grid-cols-2 gap-5 pt-2">
                {/* Education */}
                <div id="field-Education">
                  <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 uppercase tracking-wider mb-1.5">
                    Highest Education Level Completed <span className="text-red-500">*</span>
                  </label>
                  <select
                    value={form.Education}
                    onChange={(e) => setVal('Education', e.target.value === '' ? '' : parseInt(e.target.value, 10))}
                    className={`w-full px-3.5 py-2.5 rounded-xl border text-sm text-slate-800 dark:text-white focus:outline-none transition-all ${
                      errors.Education
                        ? 'border-red-500 bg-red-50/20 text-red-900 dark:text-red-300 focus:ring-2 focus:ring-red-300'
                        : 'border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900/80 focus:ring-2 focus:ring-blue-500'
                    }`}
                  >
                    <option value="" className="dark:bg-slate-800 dark:text-white">-- Select Highest Education Level Completed --</option>
                    {CDC_EDUCATION_OPTIONS.map((opt) => (
                      <option key={opt.value} value={opt.value} className="dark:bg-slate-800 dark:text-white">
                        {opt.label}
                      </option>
                    ))}
                  </select>
                  {errors.Education ? (
                    <p className="text-xs font-semibold text-red-600 mt-1.5 flex items-center gap-1">
                      <AlertCircle className="w-3.5 h-3.5" />
                      <span>{errors.Education}</span>
                    </p>
                  ) : (
                    <p className="text-xs text-slate-400 dark:text-slate-500 mt-1">Standard 6-level CDC education bracket.</p>
                  )}
                </div>

                {/* Income */}
                <div id="field-Income">
                  <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 uppercase tracking-wider mb-1.5">
                    Annual Household Income Bracket <span className="text-red-500">*</span>
                  </label>
                  <select
                    value={form.Income}
                    onChange={(e) => setVal('Income', e.target.value === '' ? '' : parseInt(e.target.value, 10))}
                    className={`w-full px-3.5 py-2.5 rounded-xl border text-sm text-slate-800 dark:text-white focus:outline-none transition-all ${
                      errors.Income
                        ? 'border-red-500 bg-red-50/20 text-red-900 dark:text-red-300 focus:ring-2 focus:ring-red-300'
                        : 'border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900/80 focus:ring-2 focus:ring-blue-500'
                    }`}
                  >
                    <option value="" className="dark:bg-slate-800 dark:text-white">-- Select Annual Household Income Bracket --</option>
                    {CDC_INCOME_OPTIONS.map((opt) => (
                      <option key={opt.value} value={opt.value} className="dark:bg-slate-800 dark:text-white">
                        {opt.label}
                      </option>
                    ))}
                  </select>
                  {errors.Income ? (
                    <p className="text-xs font-semibold text-red-600 mt-1.5 flex items-center gap-1">
                      <AlertCircle className="w-3.5 h-3.5" />
                      <span>{errors.Income}</span>
                    </p>
                  ) : (
                    <p className="text-xs text-slate-400 dark:text-slate-500 mt-1">Standard 8-level CDC household income bracket.</p>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* ═════════════════════════════════════════════════════════════════ */}
          {/* STEP 6: Review Your Answers & Final Submit                        */}
          {/* ═════════════════════════════════════════════════════════════════ */}
          {currentStep === 6 && (
            <div className="space-y-6">
              <div className="border-b border-slate-100 dark:border-slate-700 pb-4">
                <h2 className="text-xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
                  <FileCheck2 className="w-5 h-5 text-teal-600 dark:text-teal-400" /> Review Your Health Assessment
                </h2>
                <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                  Please review all 21 parameters below. You can navigate back to modify any values before submitting.
                </p>
              </div>

              {/* Review Grid */}
              <div className="grid sm:grid-cols-2 gap-4 text-xs">
                {/* Personal */}
                <div className="bg-slate-50/70 dark:bg-slate-900/50 rounded-2xl p-4 sm:p-5 border border-slate-100 dark:border-slate-700/60 text-slate-900 dark:text-white transition-colors">
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-bold text-slate-800 dark:text-slate-200 flex items-center gap-1.5">
                      <User className="w-3.5 h-3.5 text-blue-600 dark:text-blue-400" /> Personal Baseline
                    </span>
                    <button
                      type="button"
                      onClick={() => handleStepClick(1)}
                      className="text-blue-600 dark:text-blue-400 hover:underline font-semibold"
                    >
                      Edit
                    </button>
                  </div>
                  <div className="space-y-1.5 text-slate-600 dark:text-slate-300">
                    <div className="flex justify-between">
                      <span>Full Name:</span>
                      <strong className={personal.fullName?.trim() ? 'text-slate-900 dark:text-white' : 'text-red-600'}>
                        {personal.fullName?.trim() || 'Missing (Required)'}
                      </strong>
                    </div>
                    <div className="flex justify-between">
                      <span>Email:</span>
                      <span className="text-slate-700 dark:text-slate-200">{personal.email}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Age:</span>
                      <strong className={personal.rawAge ? 'text-slate-900 dark:text-white' : 'text-red-600'}>
                        {personal.rawAge ? `${personal.rawAge} yrs (${calculatedAgeCat ? CDC_AGE_CATEGORIES[calculatedAgeCat]?.bracket : 'Cat ' + form.Age})` : 'Missing (Required)'}
                      </strong>
                    </div>
                    <div className="flex justify-between">
                      <span>Sex:</span>
                      <strong className={personal.sex !== null ? 'text-slate-900 dark:text-white' : 'text-red-600'}>
                        {personal.sex === 1 ? 'Male (1)' : personal.sex === 0 ? 'Female (0)' : 'Unselected (Required)'}
                      </strong>
                    </div>
                    <div className="flex justify-between">
                      <span>Date:</span>
                      <span>{personal.assessmentDate}</span>
                    </div>
                  </div>
                </div>

                {/* Medical History */}
                <div className="bg-slate-50/70 dark:bg-slate-900/50 rounded-2xl p-4 sm:p-5 border border-slate-100 dark:border-slate-700/60 text-slate-900 dark:text-white transition-colors">
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-bold text-slate-800 dark:text-slate-200 flex items-center gap-1.5">
                      <Heart className="w-3.5 h-3.5 text-red-500" /> Clinical Indicators
                    </span>
                    <button
                      type="button"
                      onClick={() => handleStepClick(2)}
                      className="text-blue-600 dark:text-blue-400 hover:underline font-semibold"
                    >
                      Edit
                    </button>
                  </div>
                  <div className="space-y-1.5 text-slate-600 dark:text-slate-300">
                    <div className="flex justify-between">
                      <span>High Blood Pressure:</span>
                      <span className={form.HighBP === null ? 'text-red-600 font-bold' : form.HighBP ? 'text-red-600 font-bold' : 'text-slate-700 dark:text-slate-200'}>
                        {form.HighBP === null ? 'Unanswered (Required)' : form.HighBP ? 'Yes (1)' : 'No (0)'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>High Cholesterol:</span>
                      <span className={form.HighChol === null ? 'text-red-600 font-bold' : form.HighChol ? 'text-red-600 font-bold' : 'text-slate-700 dark:text-slate-200'}>
                        {form.HighChol === null ? 'Unanswered (Required)' : form.HighChol ? 'Yes (1)' : 'No (0)'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Cholesterol Checked (5 yrs):</span>
                      <span className={form.CholCheck === null ? 'text-red-600 font-bold' : 'text-slate-700 dark:text-slate-200'}>
                        {form.CholCheck === null ? 'Unanswered (Required)' : form.CholCheck ? 'Yes (1)' : 'No (0)'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>History of Stroke:</span>
                      <span className={form.Stroke === null ? 'text-red-600 font-bold' : 'text-slate-700 dark:text-slate-200'}>
                        {form.Stroke === null ? 'Unanswered (Required)' : form.Stroke ? 'Yes (1)' : 'No (0)'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Coronary Heart Disease / MI:</span>
                      <span className={form.HeartDiseaseorAttack === null ? 'text-red-600 font-bold' : 'text-slate-700 dark:text-slate-200'}>
                        {form.HeartDiseaseorAttack === null ? 'Unanswered (Required)' : form.HeartDiseaseorAttack ? 'Yes (1)' : 'No (0)'}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Lifestyle */}
                <div className="bg-slate-50/70 dark:bg-slate-900/50 rounded-2xl p-4 sm:p-5 border border-slate-100 dark:border-slate-700/60 text-slate-900 dark:text-white transition-colors">
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-bold text-slate-800 dark:text-slate-200 flex items-center gap-1.5">
                      <Activity className="w-3.5 h-3.5 text-teal-600 dark:text-teal-400" /> Lifestyle &amp; Body
                    </span>
                    <button
                      type="button"
                      onClick={() => handleStepClick(3)}
                      className="text-blue-600 dark:text-blue-400 hover:underline font-semibold"
                    >
                      Edit
                    </button>
                  </div>
                  <div className="space-y-1.5 text-slate-600 dark:text-slate-300">
                    <div className="flex justify-between">
                      <span>BMI:</span>
                      <strong className={form.BMI !== '' ? 'text-slate-900 dark:text-white' : 'text-red-600 font-bold'}>
                        {form.BMI !== '' ? `${form.BMI} kg/m² (${bmiInfo.label})` : 'Missing (Required)'}
                      </strong>
                    </div>
                    <div className="flex justify-between">
                      <span>Smoker (100+ lifetime):</span>
                      <span className={form.Smoker === null ? 'text-red-600 font-bold' : 'text-slate-700 dark:text-slate-200'}>
                        {form.Smoker === null ? 'Unanswered (Required)' : form.Smoker ? 'Yes (1)' : 'No (0)'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Physical Activity (30d):</span>
                      <span className={form.PhysActivity === null ? 'text-red-600 font-bold' : 'text-slate-700 dark:text-slate-200'}>
                        {form.PhysActivity === null ? 'Unanswered (Required)' : form.PhysActivity ? 'Yes (1)' : 'No (0)'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Fruits / Veggies Daily:</span>
                      <span className={form.Fruits === null || form.Veggies === null ? 'text-red-600 font-bold' : 'text-slate-700 dark:text-slate-200'}>
                        {form.Fruits === null ? 'Fruits: ?' : form.Fruits ? 'Fruits' : 'No Fruit'} / {form.Veggies === null ? 'Veggies: ?' : form.Veggies ? 'Veggies' : 'No Veggies'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Heavy Alcohol Consumption:</span>
                      <span className={form.HvyAlcoholConsump === null ? 'text-red-600 font-bold' : 'text-slate-700 dark:text-slate-200'}>
                        {form.HvyAlcoholConsump === null ? 'Unanswered (Required)' : form.HvyAlcoholConsump ? 'Yes (1)' : 'No (0)'}
                      </span>
                    </div>
                  </div>
                </div>

                {/* General Health & Demographics */}
                <div className="bg-slate-50/70 dark:bg-slate-900/50 rounded-2xl p-4 sm:p-5 border border-slate-100 dark:border-slate-700/60 text-slate-900 dark:text-white transition-colors">
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-bold text-slate-800 dark:text-slate-200 flex items-center gap-1.5">
                      <Smile className="w-3.5 h-3.5 text-amber-500" /> General &amp; Demographics
                    </span>
                    <button
                      type="button"
                      onClick={() => handleStepClick(4)}
                      className="text-blue-600 dark:text-blue-400 hover:underline font-semibold"
                    >
                      Edit
                    </button>
                  </div>
                  <div className="space-y-1.5 text-slate-600 dark:text-slate-300">
                    <div className="flex justify-between">
                      <span>General Health Self-Rating:</span>
                      <strong className={form.GenHlth ? 'text-slate-900 dark:text-white' : 'text-red-600 font-bold'}>
                        {form.GenHlth ? `Level ${form.GenHlth} (1=Exc, 5=Poor)` : 'Unanswered (Required)'}
                      </strong>
                    </div>
                    <div className="flex justify-between">
                      <span>Unhealthy Days (30d):</span>
                      <span className={form.PhysHlth === '' || form.MentHlth === '' ? 'text-red-600 font-bold' : 'text-slate-700 dark:text-slate-200'}>
                        Phys: {form.PhysHlth === '' ? '?' : `${form.PhysHlth}d`} | Ment: {form.MentHlth === '' ? '?' : `${form.MentHlth}d`}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Difficulty Walking:</span>
                      <span className={form.DiffWalk === null ? 'text-red-600 font-bold' : 'text-slate-700 dark:text-slate-200'}>
                        {form.DiffWalk === null ? 'Unanswered (Required)' : form.DiffWalk ? 'Yes (1)' : 'No (0)'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Health Care Coverage:</span>
                      <span className={form.AnyHealthcare === null ? 'text-red-600 font-bold' : 'text-slate-700 dark:text-slate-200'}>
                        {form.AnyHealthcare === null ? 'Unanswered (Required)' : form.AnyHealthcare ? 'Covered (1)' : 'Uninsured (0)'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Education &amp; Income:</span>
                      <span className={!form.Education || !form.Income ? 'text-red-600 font-bold' : 'text-slate-700 dark:text-slate-200'}>
                        Edu: {form.Education ? `${form.Education} / 6` : '?'} | Inc: {form.Income ? `${form.Income} / 8` : '?'}
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Research Protocol Notice */}
              <div className="p-4 rounded-xl bg-blue-50/70 dark:bg-blue-950/40 border border-blue-200 dark:border-blue-800 text-xs text-blue-800 dark:text-blue-300 flex items-start gap-3">
                <Info className="w-4 h-4 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-0.5" />
                <p>
                  Upon submission, your responses will be processed by our frozen Phase 5 Logistic Regression model
                  evaluated on 202,944 CDC BRFSS records. Assessment results will be saved to your personal history.
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Stepper Controls */}
        <div className="flex items-center justify-between gap-4">
          {currentStep > 1 ? (
            <button
              type="button"
              onClick={handlePrev}
              disabled={submitting}
              className="px-5 py-2.5 rounded-xl border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-200 bg-white dark:bg-slate-800 hover:bg-slate-50 dark:hover:bg-slate-700 text-sm font-semibold flex items-center gap-2 transition-colors shadow-2xs disabled:opacity-50"
            >
              <ChevronLeft className="w-4 h-4" /> Previous
            </button>
          ) : (
            <div />
          )}

          {currentStep < 6 ? (
            <button
              type="button"
              onClick={handleNext}
              className="px-6 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-sm font-bold flex items-center gap-2 transition-all shadow-sm shadow-blue-500/20"
            >
              Next Step <ChevronRight className="w-4 h-4" />
            </button>
          ) : (
            <button
              type="button"
              onClick={handleSubmit}
              disabled={submitting}
              className="px-8 py-3 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-sm font-bold flex items-center gap-2 transition-all shadow-sm shadow-blue-500/20 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {submitting ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Analyzing Indicators...
                </>
              ) : (
                <>
                  Submit Health Assessment <ChevronRight className="w-4 h-4" />
                </>
              )}
            </button>
          )}
        </div>

        {/* Disclaimer Footer */}
        <div className="mt-12">
          <DisclaimerBanner />
        </div>
      </div>
    </div>
  )
}
