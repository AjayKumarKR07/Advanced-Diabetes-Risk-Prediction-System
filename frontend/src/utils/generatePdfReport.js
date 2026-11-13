/**
 * src/utils/generatePdfReport.js
 * ===============================
 * Generates a publication-grade, multi-page PDF Assessment Report for DiabetesRisk AI
 * using jsPDF.
 * 
 * Includes:
 * 1. DiabetesRisk AI branding and metadata
 * 2. Assessment ID, Prediction ID, and Timestamp
 * 3. User details (Full name, verified email, chronological age, sex)
 * 4. Model-predicted classification and confidence
 * 5. All three model-estimated class probabilities (ND, Pre, DM)
 * 6. Complete 21 CDC BRFSS submitted indicators with readable labels
 * 7. Frozen Phase 5 model evaluation metrics (Macro F1 = 0.4246, ROC-AUC = 0.7707)
 * 8. Prediabetes limitation disclosure (Precision: 0.0307, Recall: 0.3089)
 * 9. Comprehensive academic research and medical disclaimers
 * 10. Multi-page pagination footer ("Page X of Y")
 */

import { jsPDF } from 'jspdf'
import {
  CLASS_LABELS,
  CDC_AGE_CATEGORIES,
  CDC_GENHLTH_OPTIONS,
  CDC_EDUCATION_OPTIONS,
  CDC_INCOME_OPTIONS,
  getBmiCategory,
} from './cdcFeatures'

export function generateAssessmentReportPdf({
  result,
  inputs,
  personalInfo,
  user,
}) {
  const doc = new jsPDF({
    orientation: 'portrait',
    unit: 'mm',
    format: 'a4',
  })

  const pageWidth = doc.internal.pageSize.getWidth() // 210mm
  const pageHeight = doc.internal.pageSize.getHeight() // 297mm
  const margin = 14
  const contentWidth = pageWidth - margin * 2 // 182mm

  const displayName = personalInfo?.fullName || result?.user_name || user?.full_name || 'Assessee'
  const email = personalInfo?.email || user?.email || 'N/A'
  const assessmentDate =
    personalInfo?.assessmentDate ||
    (result?.created_at ? new Date(result.created_at).toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' }) : new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' }))

  const predClass = result?.predicted_class ?? 0
  const predLabel = CLASS_LABELS[predClass] || 'No Diabetes'
  const confidence = result?.confidence ? (result.confidence * 100).toFixed(1) : '—'

  const p0 = result?.probability_no_diabetes ?? result?.probabilities?.no_diabetes ?? 0
  const p1 = result?.probability_prediabetes ?? result?.probabilities?.prediabetes ?? 0
  const p2 = result?.probability_diabetes ?? result?.probabilities?.diabetes ?? 0

  const assessmentId = result?.assessment_id || result?.id || '—'
  const predictionId = result?.prediction_id || result?.id || '—'

  // Helper colors
  const colorNavy = [15, 23, 42]     // #0f172a
  const colorBlue = [37, 99, 235]     // #2563eb
  const colorTeal = [13, 148, 136]    // #0d9488
  const colorSlate = [100, 116, 139] // #64748b
  const colorLightBg = [248, 250, 252] // #f8fafc
  const colorBorder = [226, 232, 240] // #e2e8f0

  // ═════════════════════════════════════════════════════════════════════════
  // PAGE 1: Executive Summary & Predictions
  // ═════════════════════════════════════════════════════════════════════════

  // Header Banner
  doc.setFillColor(...colorBlue)
  doc.rect(0, 0, pageWidth, 5, 'F')

  // Brand Name & Subtitle
  doc.setFont('helvetica', 'bold')
  doc.setFontSize(18)
  doc.setTextColor(...colorNavy)
  doc.text('DiabetesRisk AI', margin, 18)

  doc.setFont('helvetica', 'normal')
  doc.setFontSize(9)
  doc.setTextColor(...colorTeal)
  doc.text('ADVANCED HEALTHCARE ANALYTICS & PREDICTION SYSTEM', margin, 23)

  // Document Title (Right Aligned)
  doc.setFont('helvetica', 'bold')
  doc.setFontSize(12)
  doc.setTextColor(...colorNavy)
  doc.text('CLINICAL ASSESSMENT REPORT', pageWidth - margin, 18, { align: 'right' })

  doc.setFont('helvetica', 'normal')
  doc.setFontSize(8)
  doc.setTextColor(...colorSlate)
  doc.text(`Generated: ${new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}`, pageWidth - margin, 23, { align: 'right' })

  // Divider Line
  doc.setDrawColor(...colorBorder)
  doc.setLineWidth(0.5)
  doc.line(margin, 27, pageWidth - margin, 27)

  // Patient & Evaluation Metadata Card
  doc.setFillColor(...colorLightBg)
  doc.setDrawColor(...colorBorder)
  doc.roundedRect(margin, 31, contentWidth, 24, 2, 2, 'FD')

  doc.setFont('helvetica', 'bold')
  doc.setFontSize(9)
  doc.setTextColor(...colorNavy)
  doc.text('Account Holder:', margin + 4, 38)
  doc.text('Assessment Date:', margin + 4, 44)
  doc.text('Record Identifiers:', margin + 4, 50)

  doc.setFont('helvetica', 'normal')
  doc.setTextColor(51, 65, 85)
  doc.text(`${displayName} (${email})`, margin + 35, 38)
  doc.text(`${assessmentDate}`, margin + 35, 44)
  doc.text(`Assessment #${assessmentId}  |  Prediction #${predictionId}  |  Model: Logistic Regression`, margin + 35, 50)

  // Primary Predicted Classification Hero Box
  let classBg = [236, 253, 245] // emerald-50
  let classBorder = [16, 185, 129] // emerald-500
  let classText = [5, 150, 105] // emerald-600

  if (predClass === 1) {
    classBg = [255, 251, 235] // amber-50
    classBorder = [245, 158, 11] // amber-500
    classText = [180, 83, 9] // amber-700
  } else if (predClass === 2) {
    classBg = [254, 242, 242] // rose-50
    classBorder = [239, 68, 68] // rose-500
    classText = [185, 28, 28] // rose-700
  }

  doc.setFillColor(...classBg)
  doc.setDrawColor(...classBorder)
  doc.setLineWidth(0.8)
  doc.roundedRect(margin, 60, contentWidth, 34, 3, 3, 'FD')

  doc.setFont('helvetica', 'bold')
  doc.setFontSize(8)
  doc.setTextColor(...colorSlate)
  doc.text('MODEL-PREDICTED RESEARCH CLASSIFICATION', margin + 6, 68)

  doc.setFont('helvetica', 'bold')
  doc.setFontSize(20)
  doc.setTextColor(...classText)
  doc.text(predLabel, margin + 6, 78)

  doc.setFont('helvetica', 'normal')
  doc.setFontSize(9)
  doc.setTextColor(71, 85, 105)
  doc.text(
    `Class ${predClass}  |  Model Confidence: ${confidence}% posterior probability  |  Algorithm: Multinomial Logistic Regression`,
    margin + 6,
    86
  )

  // Probabilities Section Title
  doc.setFont('helvetica', 'bold')
  doc.setFontSize(11)
  doc.setTextColor(...colorNavy)
  doc.text('Model-Estimated Probability Distribution Across Classes', margin, 103)

  doc.setFont('helvetica', 'normal')
  doc.setFontSize(8)
  doc.setTextColor(...colorSlate)
  doc.text(
    'Posterior probabilities computed via softmax over 21 CDC health indicators. Sums to 100%. (Not clinical risk percentages).',
    margin,
    108
  )

  // 3 Probability Boxes
  const boxWidth = (contentWidth - 6) / 3
  const probData = [
    { title: 'Class 0: No Diabetes', prob: p0, color: [16, 185, 129], bg: [240, 253, 244] },
    { title: 'Class 1: Prediabetes', prob: p1, color: [245, 158, 11], bg: [255, 251, 235] },
    { title: 'Class 2: Diabetes', prob: p2, color: [239, 68, 68], bg: [254, 242, 242] },
  ]

  probData.forEach((item, idx) => {
    const x = margin + idx * (boxWidth + 3)
    doc.setFillColor(...item.bg)
    doc.setDrawColor(...colorBorder)
    doc.setLineWidth(0.5)
    doc.roundedRect(x, 113, boxWidth, 22, 2, 2, 'FD')

    doc.setFont('helvetica', 'bold')
    doc.setFontSize(8)
    doc.setTextColor(71, 85, 105)
    doc.text(item.title, x + 4, 120)

    doc.setFont('helvetica', 'bold')
    doc.setFontSize(14)
    doc.setTextColor(...item.color)
    doc.text(`${(item.prob * 100).toFixed(1)}%`, x + 4, 129)
  })

  // Submitted Features Section Header
  doc.setFont('helvetica', 'bold')
  doc.setFontSize(11)
  doc.setTextColor(...colorNavy)
  doc.text('Submitted Health Information (CDC BRFSS Indicators)', margin, 144)

  doc.setFont('helvetica', 'normal')
  doc.setFontSize(8)
  doc.setTextColor(...colorSlate)
  doc.text('Exact values submitted and processed by the inference engine.', margin, 149)

  // Table Category A: Personal Baseline & Demographics
  let y = 154
  doc.setFillColor(...colorNavy)
  doc.rect(margin, y, contentWidth, 6, 'F')
  doc.setFont('helvetica', 'bold')
  doc.setFontSize(8)
  doc.setTextColor(255, 255, 255)
  doc.text('A. Personal Demographics Baseline', margin + 3, y + 4.2)

  y += 6
  const ageCat = inputs?.Age || personalInfo?.cdcAgeCategory
  const bmiVal = inputs?.BMI !== undefined && inputs?.BMI !== null ? Number(inputs.BMI).toFixed(1) : '—'
  const bmiInfo = inputs?.BMI ? getBmiCategory(inputs.BMI) : { label: 'Unspecified' }

  const personalRows = [
    ['Full Name', displayName, 'Account Email', email],
    [
      'Age (Chronological)',
      personalInfo?.rawAge ? `${personalInfo.rawAge} years` : 'Reported in Category',
      'CDC Age Bracket',
      ageCat ? `${CDC_AGE_CATEGORIES[ageCat]?.bracket || 'Category ' + ageCat}` : 'Category ' + inputs?.Age,
    ],
    [
      'Biological Sex',
      inputs?.Sex === 1 ? 'Male (1)' : inputs?.Sex === 0 ? 'Female (0)' : 'Unspecified',
      'Assessment Date',
      assessmentDate,
    ],
  ]

  personalRows.forEach((row, i) => {
    doc.setFillColor(i % 2 === 0 ? 255 : 248, i % 2 === 0 ? 255 : 250, i % 2 === 0 ? 255 : 252)
    doc.rect(margin, y, contentWidth, 6, 'F')
    doc.setDrawColor(...colorBorder)
    doc.line(margin, y + 6, margin + contentWidth, y + 6)

    doc.setFont('helvetica', 'bold')
    doc.setFontSize(7.5)
    doc.setTextColor(71, 85, 105)
    doc.text(row[0], margin + 2, y + 4.2)
    doc.setFont('helvetica', 'normal')
    doc.setTextColor(15, 23, 42)
    doc.text(String(row[1]), margin + 45, y + 4.2)

    doc.setFont('helvetica', 'bold')
    doc.setTextColor(71, 85, 105)
    doc.text(row[2], margin + 95, y + 4.2)
    doc.setFont('helvetica', 'normal')
    doc.setTextColor(15, 23, 42)
    doc.text(String(row[3]), margin + 140, y + 4.2)
    y += 6
  })

  // Table Category B: Medical & Clinical History
  y += 4
  doc.setFillColor(...colorNavy)
  doc.rect(margin, y, contentWidth, 6, 'F')
  doc.setFont('helvetica', 'bold')
  doc.setFontSize(8)
  doc.setTextColor(255, 255, 255)
  doc.text('B. Medical & Clinical History (Diagnosed by Healthcare Professional)', margin + 3, y + 4.2)

  y += 6
  const clinicalRows = [
    ['High Blood Pressure (HighBP)', inputs?.HighBP === 1 ? 'Yes (Diagnosed)' : 'No', 'High Blood Cholesterol (HighChol)', inputs?.HighChol === 1 ? 'Yes (Diagnosed)' : 'No'],
    ['Cholesterol Check 5 Yrs (CholCheck)', inputs?.CholCheck === 1 ? 'Yes (Screened)' : 'No', 'History of Stroke (Stroke)', inputs?.Stroke === 1 ? 'Yes (Diagnosed)' : 'No'],
    ['Coronary Heart Disease / MI', inputs?.HeartDiseaseorAttack === 1 ? 'Yes (Diagnosed)' : 'No', 'Difficulty Walking (DiffWalk)', inputs?.DiffWalk === 1 ? 'Yes (Mobility Impairment)' : 'No'],
  ]

  clinicalRows.forEach((row, i) => {
    doc.setFillColor(i % 2 === 0 ? 255 : 248, i % 2 === 0 ? 255 : 250, i % 2 === 0 ? 255 : 252)
    doc.rect(margin, y, contentWidth, 6, 'F')
    doc.setDrawColor(...colorBorder)
    doc.line(margin, y + 6, margin + contentWidth, y + 6)

    doc.setFont('helvetica', 'bold')
    doc.setFontSize(7.5)
    doc.setTextColor(71, 85, 105)
    doc.text(row[0], margin + 2, y + 4.2)
    doc.setFont('helvetica', 'normal')
    doc.setTextColor(row[1].startsWith('Yes') ? 185 : 15, row[1].startsWith('Yes') ? 28 : 23, row[1].startsWith('Yes') ? 28 : 42)
    doc.text(String(row[1]), margin + 55, y + 4.2)

    doc.setFont('helvetica', 'bold')
    doc.setTextColor(71, 85, 105)
    doc.text(row[2], margin + 95, y + 4.2)
    doc.setFont('helvetica', 'normal')
    doc.setTextColor(row[3].startsWith('Yes') ? 185 : 15, row[3].startsWith('Yes') ? 28 : 23, row[3].startsWith('Yes') ? 28 : 42)
    doc.text(String(row[3]), margin + 150, y + 4.2)
    y += 6
  })

  // Table Category C: Lifestyle & Anthropometrics
  y += 4
  doc.setFillColor(...colorNavy)
  doc.rect(margin, y, contentWidth, 6, 'F')
  doc.setFont('helvetica', 'bold')
  doc.setFontSize(8)
  doc.setTextColor(255, 255, 255)
  doc.text('C. Lifestyle & Anthropometric Indicators', margin + 3, y + 4.2)

  y += 6
  const lifestyleRows = [
    ['Body Mass Index (BMI)', `${bmiVal} kg/m² (${bmiInfo.label})`, 'Smoking History (100+ Cigs)', inputs?.Smoker === 1 ? 'Yes (Smoker/Former)' : 'No'],
    ['Physical Activity (30 Days)', inputs?.PhysActivity === 1 ? 'Yes (Active)' : 'No (Sedentary)', 'Daily Fruit Consumption', inputs?.Fruits === 1 ? 'Yes (>= 1/day)' : 'No'],
    ['Daily Vegetable Consumption', inputs?.Veggies === 1 ? 'Yes (>= 1/day)' : 'No', 'Heavy Alcohol Consumption', inputs?.HvyAlcoholConsump === 1 ? 'Yes (Heavy)' : 'No'],
  ]

  lifestyleRows.forEach((row, i) => {
    doc.setFillColor(i % 2 === 0 ? 255 : 248, i % 2 === 0 ? 255 : 250, i % 2 === 0 ? 255 : 252)
    doc.rect(margin, y, contentWidth, 6, 'F')
    doc.setDrawColor(...colorBorder)
    doc.line(margin, y + 6, margin + contentWidth, y + 6)

    doc.setFont('helvetica', 'bold')
    doc.setFontSize(7.5)
    doc.setTextColor(71, 85, 105)
    doc.text(row[0], margin + 2, y + 4.2)
    doc.setFont('helvetica', 'normal')
    doc.setTextColor(15, 23, 42)
    doc.text(String(row[1]), margin + 45, y + 4.2)

    doc.setFont('helvetica', 'bold')
    doc.setTextColor(71, 85, 105)
    doc.text(row[2], margin + 95, y + 4.2)
    doc.setFont('helvetica', 'normal')
    doc.setTextColor(15, 23, 42)
    doc.text(String(row[3]), margin + 145, y + 4.2)
    y += 6
  })

  // Page 1 Footer
  doc.setFont('helvetica', 'normal')
  doc.setFontSize(7.5)
  doc.setTextColor(...colorSlate)
  doc.text('DiabetesRisk AI • Clinical Analytics & Research Assessment • Page 1 of 2', margin, pageHeight - 8)
  doc.text('Strictly Academic / Non-Diagnostic Tool', pageWidth - margin, pageHeight - 8, { align: 'right' })

  // ═════════════════════════════════════════════════════════════════════════
  // PAGE 2: General Health, Demographics & Scientific Information
  // ═════════════════════════════════════════════════════════════════
  doc.addPage()

  // Header Banner
  doc.setFillColor(...colorBlue)
  doc.rect(0, 0, pageWidth, 5, 'F')

  doc.setFont('helvetica', 'bold')
  doc.setFontSize(14)
  doc.setTextColor(...colorNavy)
  doc.text('Detailed Indicators & Research Benchmarks', margin, 16)

  doc.setFont('helvetica', 'normal')
  doc.setFontSize(8)
  doc.setTextColor(...colorSlate)
  doc.text(`Record #${assessmentId} — Continued Information`, pageWidth - margin, 16, { align: 'right' })

  doc.setDrawColor(...colorBorder)
  doc.line(margin, 20, pageWidth - margin, 20)

  // Category D: General Health & Functional Status
  y = 25
  doc.setFillColor(...colorNavy)
  doc.rect(margin, y, contentWidth, 6, 'F')
  doc.setFont('helvetica', 'bold')
  doc.setFontSize(8)
  doc.setTextColor(255, 255, 255)
  doc.text('D. General Health & Functional Impairment (Self-Reported Past 30 Days)', margin + 3, y + 4.2)

  y += 6
  const genOpt = CDC_GENHLTH_OPTIONS.find((o) => o.value === inputs?.GenHlth)
  const healthRows = [
    [
      'General Health Rating (GenHlth)',
      genOpt ? genOpt.label : `Level ${inputs?.GenHlth} of 5`,
      'Difficulty Walking (DiffWalk)',
      inputs?.DiffWalk === 1 ? 'Yes (Reported difficulty)' : 'No',
    ],
    [
      'Poor Physical Health Days (0–30)',
      `${inputs?.PhysHlth ?? 0} days in past month`,
      'Poor Mental Health Days (0–30)',
      `${inputs?.MentHlth ?? 0} days in past month`,
    ],
  ]

  healthRows.forEach((row, i) => {
    doc.setFillColor(i % 2 === 0 ? 255 : 248, i % 2 === 0 ? 255 : 250, i % 2 === 0 ? 255 : 252)
    doc.rect(margin, y, contentWidth, 6, 'F')
    doc.setDrawColor(...colorBorder)
    doc.line(margin, y + 6, margin + contentWidth, y + 6)

    doc.setFont('helvetica', 'bold')
    doc.setFontSize(7.5)
    doc.setTextColor(71, 85, 105)
    doc.text(row[0], margin + 2, y + 4.2)
    doc.setFont('helvetica', 'normal')
    doc.setTextColor(15, 23, 42)
    doc.text(String(row[1]), margin + 50, y + 4.2)

    doc.setFont('helvetica', 'bold')
    doc.setTextColor(71, 85, 105)
    doc.text(row[2], margin + 95, y + 4.2)
    doc.setFont('helvetica', 'normal')
    doc.setTextColor(15, 23, 42)
    doc.text(String(row[3]), margin + 145, y + 4.2)
    y += 6
  })

  // Category E: Healthcare Coverage & Demographics
  y += 4
  doc.setFillColor(...colorNavy)
  doc.rect(margin, y, contentWidth, 6, 'F')
  doc.setFont('helvetica', 'bold')
  doc.setFontSize(8)
  doc.setTextColor(255, 255, 255)
  doc.text('E. Healthcare Access & Socioeconomic Demographics', margin + 3, y + 4.2)

  y += 6
  const eduOpt = CDC_EDUCATION_OPTIONS.find((o) => o.value === inputs?.Education)
  const incOpt = CDC_INCOME_OPTIONS.find((o) => o.value === inputs?.Income)

  const accessRows = [
    [
      'Healthcare Coverage (AnyHealthcare)',
      inputs?.AnyHealthcare === 1 ? 'Yes (Covered/Insured)' : 'No (Uninsured)',
      'Doctor Avoided Due to Cost (Past 12m)',
      inputs?.NoDocbcCost === 1 ? 'Yes (Cost barrier)' : 'No',
    ],
    [
      'Education Level (Education)',
      eduOpt ? eduOpt.label : `Bracket ${inputs?.Education} / 6`,
      'Household Income Bracket (Income)',
      incOpt ? incOpt.label : `Bracket ${inputs?.Income} / 8`,
    ],
  ]

  accessRows.forEach((row, i) => {
    doc.setFillColor(i % 2 === 0 ? 255 : 248, i % 2 === 0 ? 255 : 250, i % 2 === 0 ? 255 : 252)
    doc.rect(margin, y, contentWidth, 6, 'F')
    doc.setDrawColor(...colorBorder)
    doc.line(margin, y + 6, margin + contentWidth, y + 6)

    doc.setFont('helvetica', 'bold')
    doc.setFontSize(7.5)
    doc.setTextColor(71, 85, 105)
    doc.text(row[0], margin + 2, y + 4.2)
    doc.setFont('helvetica', 'normal')
    doc.setTextColor(15, 23, 42)
    doc.text(String(row[1]), margin + 55, y + 4.2)

    doc.setFont('helvetica', 'bold')
    doc.setTextColor(71, 85, 105)
    doc.text(row[2], margin + 95, y + 4.2)
    doc.setFont('helvetica', 'normal')
    doc.setTextColor(15, 23, 42)
    doc.text(String(row[3]), margin + 150, y + 4.2)
    y += 6
  })

  // Model Performance & Research Information
  y += 6
  doc.setFont('helvetica', 'bold')
  doc.setFontSize(11)
  doc.setTextColor(...colorNavy)
  doc.text('Model Performance & Scientific Validation (Phase 5 Held-Out Evaluation)', margin, y)

  y += 5
  doc.setFont('helvetica', 'normal')
  doc.setFontSize(8)
  doc.setTextColor(...colorSlate)
  doc.text(
    'Frozen research benchmarks evaluated on 50,736 held-out test rows from the CDC BRFSS 2015 dataset.',
    margin,
    y
  )

  y += 5
  // Global metrics 4-card grid
  const gBoxWidth = (contentWidth - 6) / 4
  const gMetrics = [
    { label: 'Macro F1', val: '0.4246', desc: 'Primary metric' },
    { label: 'ROC-AUC OvR', val: '0.7707', desc: 'Macro discrimination' },
    { label: 'Balanced Accuracy', val: '51.72%', desc: 'Class-balanced' },
    { label: 'Overall Accuracy', val: '64.33%', desc: '50,736 test rows' },
  ]

  gMetrics.forEach((m, idx) => {
    const x = margin + idx * (gBoxWidth + 2)
    doc.setFillColor(...colorLightBg)
    doc.setDrawColor(...colorBorder)
    doc.roundedRect(x, y, gBoxWidth, 16, 1.5, 1.5, 'FD')

    doc.setFont('helvetica', 'bold')
    doc.setFontSize(7)
    doc.setTextColor(...colorSlate)
    doc.text(m.label, x + 3, y + 5)

    doc.setFont('helvetica', 'bold')
    doc.setFontSize(11)
    doc.setTextColor(...colorBlue)
    doc.text(m.val, x + 3, y + 11)

    doc.setFont('helvetica', 'normal')
    doc.setFontSize(6.5)
    doc.setTextColor(100, 116, 139)
    doc.text(m.desc, x + 3, y + 14.5)
  })

  // Class-wise Table
  y += 21
  doc.setFillColor(241, 245, 249)
  doc.rect(margin, y, contentWidth, 5, 'F')
  doc.setFont('helvetica', 'bold')
  doc.setFontSize(7.5)
  doc.setTextColor(30, 41, 59)
  doc.text('Target Class', margin + 3, y + 3.8)
  doc.text('Precision', margin + 60, y + 3.8)
  doc.text('Recall (Sensitivity)', margin + 100, y + 3.8)
  doc.text('F1-Score', margin + 145, y + 3.8)

  y += 5
  const classMetricRows = [
    ['Class 0: No Diabetes', '0.8809 (88.1%)', '0.6974 (69.7%)', '0.7785'],
    ['Class 1: Prediabetes (Weak Minority Class)', '0.0307 (3.1%)', '0.3089 (30.9%)', '0.0558'],
    ['Class 2: Diabetes', '0.3621 (36.2%)', '0.5453 (54.5%)', '0.4354'],
  ]

  classMetricRows.forEach((row, i) => {
    doc.setFillColor(i % 2 === 0 ? 255 : 248, i % 2 === 0 ? 255 : 250, i % 2 === 0 ? 255 : 252)
    doc.rect(margin, y, contentWidth, 5, 'F')
    doc.setDrawColor(...colorBorder)
    doc.line(margin, y + 5, margin + contentWidth, y + 5)

    doc.setFont('helvetica', i === 1 ? 'bold' : 'normal')
    doc.setFontSize(7.5)
    doc.setTextColor(i === 1 ? 180 : 15, i === 1 ? 83 : 23, i === 1 ? 9 : 42)
    doc.text(row[0], margin + 3, y + 3.8)

    doc.setFont('helvetica', 'normal')
    doc.setTextColor(15, 23, 42)
    doc.text(row[1], margin + 60, y + 3.8)
    doc.text(row[2], margin + 100, y + 3.8)
    doc.text(row[3], margin + 145, y + 3.8)
    y += 5
  })

  // Prediabetes Model Limitation Disclosure Box
  y += 4
  doc.setFillColor(255, 251, 235)
  doc.setDrawColor(245, 158, 11)
  doc.setLineWidth(0.6)
  doc.roundedRect(margin, y, contentWidth, 18, 2, 2, 'FD')

  doc.setFont('helvetica', 'bold')
  doc.setFontSize(7.5)
  doc.setTextColor(180, 83, 9)
  doc.text('Important Prediabetes (Class 1) Classification Limitation Notice:', margin + 4, y + 5)

  doc.setFont('helvetica', 'normal')
  doc.setFontSize(7)
  doc.setTextColor(146, 64, 14)
  const limText =
    'Due to heavy class imbalance in the general survey population (~1.8% prediabetes representation), Class 1 achieves a precision of 0.0307 and recall of 0.3089. The model is intentionally configured with balanced class weights to maintain sensitivity to early metabolic vulnerability, but individual predictions in Class 1 carry substantial uncertainty.'
  const splitLim = doc.splitTextToSize(limText, contentWidth - 8)
  doc.text(splitLim, margin + 4, y + 9.5)

  // Full Medical & Research Disclaimer Box
  y += 22
  doc.setFillColor(...colorLightBg)
  doc.setDrawColor(...colorBorder)
  doc.roundedRect(margin, y, contentWidth, 24, 2, 2, 'FD')

  doc.setFont('helvetica', 'bold')
  doc.setFontSize(7.5)
  doc.setTextColor(...colorNavy)
  doc.text('Mandatory Research Demonstration & Academic Disclaimer:', margin + 4, y + 5)

  doc.setFont('helvetica', 'normal')
  doc.setFontSize(6.8)
  doc.setTextColor(...colorSlate)
  const discText =
    'This system is an academic research engineering demonstration. It is not an FDA-cleared, CE-marked, or clinically validated diagnostic medical device. Model-estimated probabilities are statistical survey approximations and must never be interpreted as clinical risk percentages or confirmed medical diagnoses. Do not make healthcare or medication decisions based on this report. Always consult a licensed physician or endocrinologist for clinical laboratory testing (HbA1c or Fasting Plasma Glucose).'
  const splitDisc = doc.splitTextToSize(discText, contentWidth - 8)
  doc.text(splitDisc, margin + 4, y + 9.5)

  // Page 2 Footer
  doc.setFont('helvetica', 'normal')
  doc.setFontSize(7.5)
  doc.setTextColor(...colorSlate)
  doc.text('DiabetesRisk AI • Clinical Analytics & Research Assessment • Page 2 of 2', margin, pageHeight - 8)
  doc.text('Strictly Academic / Non-Diagnostic Tool', pageWidth - margin, pageHeight - 8, { align: 'right' })

  // Trigger browser download
  const filename = `DiabetesRisk_AI_Assessment_${assessmentId}.pdf`
  doc.save(filename)
}

