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

import {
  CLASS_LABELS,
  CDC_AGE_CATEGORIES,
  CDC_GENHLTH_OPTIONS,
  CDC_EDUCATION_OPTIONS,
  CDC_INCOME_OPTIONS,
  getBmiCategory,
} from './cdcFeatures'

export async function generateAssessmentReportPdf({
  result,
  inputs,
  personalInfo,
  user,
  historicalStats = null,
}) {
  const { jsPDF } = await import('jspdf')
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

/**
 * Generates a comprehensive Personal Health Analytics PDF Report (Phase 6.15)
 * covering all 11 required sections:
 * 1. DiabetesRisk AI branding
 * 2. User name and report generation date
 * 3. Total assessment count
 * 4. Latest assessment summary
 * 5. Historical predicted-class distribution
 * 6. BMI history
 * 7. General Health history
 * 8. Selected health indicator comparisons
 * 9. Assessment timeline
 * 10. Research model performance benchmark
 * 11. Research limitations and medical disclaimer
 */
export async function generatePersonalAnalyticsReportPdf({
  user,
  overview,
  bmiTrends,
  healthTrends,
  indicators,
  timeline,
}) {
  const { jsPDF } = await import('jspdf')
  const doc = new jsPDF({
    orientation: 'portrait',
    unit: 'mm',
    format: 'a4',
  })

  const pageWidth = doc.internal.pageSize.getWidth() // 210mm
  const pageHeight = doc.internal.pageSize.getHeight() // 297mm
  const margin = 14
  const contentWidth = pageWidth - margin * 2 // 182mm

  const userName = user?.full_name || 'Patient'
  const userEmail = user?.email || 'N/A'
  const reportDate = new Date().toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })

  // Palette
  const colorNavy = [15, 23, 42]
  const colorBlue = [37, 99, 235]
  const colorTeal = [13, 148, 136]
  const colorSlate = [100, 116, 139]
  const colorLightBg = [248, 250, 252]
  const colorBorder = [226, 232, 240]

  // Helper for pagination
  const totalPages = 3
  const addFooter = (pageNum) => {
    doc.setFont('helvetica', 'normal')
    doc.setFontSize(7.5)
    doc.setTextColor(...colorSlate)
    doc.text(
      `DiabetesRisk AI • Personal Health Analytics Report • Page ${pageNum} of ${totalPages}`,
      margin,
      pageHeight - 8
    )
    doc.text('Academic Research Prototype • Non-Diagnostic', pageWidth - margin, pageHeight - 8, {
      align: 'right',
    })
  }

  // ═════════════════════════════════════════════════════════════════════════
  // PAGE 1: Overview, Latest Assessment, Distribution
  // ═════════════════════════════════════════════════════════════════════════

  // Top accent banner
  doc.setFillColor(...colorBlue)
  doc.rect(0, 0, pageWidth, 5, 'F')

  // Header Title
  doc.setFont('helvetica', 'bold')
  doc.setFontSize(18)
  doc.setTextColor(...colorNavy)
  doc.text('DiabetesRisk AI', margin, 18)

  doc.setFont('helvetica', 'normal')
  doc.setFontSize(9)
  doc.setTextColor(...colorTeal)
  doc.text('COMPREHENSIVE PERSONAL HEALTH ANALYTICS REPORT', margin, 23)

  doc.setFont('helvetica', 'bold')
  doc.setFontSize(11)
  doc.setTextColor(...colorNavy)
  doc.text('LONGITUDINAL EVALUATION', pageWidth - margin, 18, { align: 'right' })

  doc.setFont('helvetica', 'normal')
  doc.setFontSize(8)
  doc.setTextColor(...colorSlate)
  doc.text(`Generated: ${reportDate}`, pageWidth - margin, 23, { align: 'right' })

  doc.setDrawColor(...colorBorder)
  doc.setLineWidth(0.5)
  doc.line(margin, 27, pageWidth - margin, 27)

  // Account & Metadata Box
  doc.setFillColor(...colorLightBg)
  doc.setDrawColor(...colorBorder)
  doc.roundedRect(margin, 31, contentWidth, 22, 2, 2, 'FD')

  doc.setFont('helvetica', 'bold')
  doc.setFontSize(8.5)
  doc.setTextColor(...colorNavy)
  doc.text('Account Holder:', margin + 4, 38)
  doc.text('Email Address:', margin + 4, 44)
  doc.text('Total Assessments:', margin + 4, 49)

  doc.setFont('helvetica', 'normal')
  doc.setTextColor(51, 65, 85)
  doc.text(userName, margin + 35, 38)
  doc.text(userEmail, margin + 35, 44)
  doc.text(`${overview?.total_assessments ?? 0} completed evaluations`, margin + 35, 49)

  doc.setFont('helvetica', 'bold')
  doc.setTextColor(...colorNavy)
  doc.text('Monthly Activity:', margin + 100, 38)
  doc.text('Evaluation Window:', margin + 100, 44)

  doc.setFont('helvetica', 'normal')
  doc.setTextColor(51, 65, 85)
  doc.text(`${overview?.assessments_this_month ?? 0} assessments this month`, margin + 132, 38)
  doc.text(
    overview?.filter_applied?.start_date
      ? `${overview.filter_applied.start_date.substring(0, 10)} to ${overview.filter_applied.end_date?.substring(0, 10) || 'Now'}`
      : 'All Time Records',
    margin + 132,
    44
  )

  // Section 2: Key Analytics Overview (KPI Cards)
  let y = 60
  doc.setFont('helvetica', 'bold')
  doc.setFontSize(11)
  doc.setTextColor(...colorNavy)
  doc.text('1. Personal Health Analytics Overview', margin, y)

  y += 5
  const cardW = (contentWidth - 9) / 4
  const cardH = 26

  // KPI 1: Completed Assessments
  doc.setFillColor(255, 255, 255)
  doc.setDrawColor(...colorBorder)
  doc.roundedRect(margin, y, cardW, cardH, 2, 2, 'FD')
  doc.setFont('helvetica', 'normal')
  doc.setFontSize(7)
  doc.setTextColor(...colorSlate)
  doc.text('TOTAL ASSESSMENTS', margin + 3, y + 6)
  doc.setFont('helvetica', 'bold')
  doc.setFontSize(14)
  doc.setTextColor(...colorNavy)
  doc.text(String(overview?.total_assessments ?? 0), margin + 3, y + 16)
  doc.setFont('helvetica', 'normal')
  doc.setFontSize(6.5)
  doc.setTextColor(...colorSlate)
  doc.text('Saved in PostgreSQL', margin + 3, y + 22)

  // KPI 2: Latest Classification
  const kpi2X = margin + cardW + 3
  doc.roundedRect(kpi2X, y, cardW, cardH, 2, 2, 'FD')
  doc.setFontSize(7)
  doc.setTextColor(...colorSlate)
  doc.text('LATEST CLASSIFICATION', kpi2X + 3, y + 6)
  doc.setFont('helvetica', 'bold')
  doc.setFontSize(10)
  const predCls = overview?.latest_predicted_class
  const clsColor = predCls === 2 ? [225, 29, 72] : predCls === 1 ? [217, 119, 6] : [16, 185, 129]
  doc.setTextColor(...clsColor)
  doc.text(overview?.latest_predicted_label || 'None Recorded', kpi2X + 3, y + 15)
  doc.setFont('helvetica', 'normal')
  doc.setFontSize(6.5)
  doc.setTextColor(...colorSlate)
  doc.text(overview?.latest_assessment_date || 'Awaiting evaluation', kpi2X + 3, y + 22)

  // KPI 3: Latest BMI
  const kpi3X = margin + (cardW + 3) * 2
  doc.roundedRect(kpi3X, y, cardW, cardH, 2, 2, 'FD')
  doc.setFontSize(7)
  doc.setTextColor(...colorSlate)
  doc.text('LATEST BMI', kpi3X + 3, y + 6)
  doc.setFont('helvetica', 'bold')
  doc.setFontSize(13)
  doc.setTextColor(...colorNavy)
  doc.text(overview?.latest_bmi ? String(overview.latest_bmi) : '—', kpi3X + 3, y + 16)
  doc.setFont('helvetica', 'normal')
  doc.setFontSize(6.5)
  doc.setTextColor(...colorSlate)
  doc.text(overview?.latest_bmi_category || 'Screening metric', kpi3X + 3, y + 22)

  // KPI 4: Latest Confidence
  const kpi4X = margin + (cardW + 3) * 3
  doc.roundedRect(kpi4X, y, cardW, cardH, 2, 2, 'FD')
  doc.setFontSize(7)
  doc.setTextColor(...colorSlate)
  doc.text('MODEL PROBABILITY', kpi4X + 3, y + 6)
  doc.setFont('helvetica', 'bold')
  doc.setFontSize(13)
  doc.setTextColor(...colorNavy)
  doc.text(
    overview?.latest_confidence ? `${(overview.latest_confidence * 100).toFixed(1)}%` : '—',
    kpi4X + 3,
    y + 16
  )
  doc.setFont('helvetica', 'normal')
  doc.setFontSize(6.5)
  doc.setTextColor(...colorSlate)
  doc.text('Model-estimated class prob.', kpi4X + 3, y + 22)

  // Section 3: Latest Assessment Summary
  y += 33
  doc.setFont('helvetica', 'bold')
  doc.setFontSize(11)
  doc.setTextColor(...colorNavy)
  doc.text('2. Latest Assessment Summary', margin, y)

  y += 5
  doc.setFillColor(...colorLightBg)
  doc.setDrawColor(...colorBorder)
  doc.roundedRect(margin, y, contentWidth, 38, 2, 2, 'FD')

  if (overview?.latest_assessment_id) {
    doc.setFont('helvetica', 'bold')
    doc.setFontSize(8.5)
    doc.setTextColor(...colorNavy)
    doc.text(`Assessment #${overview.latest_assessment_id} (Prediction #${overview.latest_prediction_id || '—'})`, margin + 4, y + 7)
    doc.setFont('helvetica', 'normal')
    doc.setFontSize(8)
    doc.setTextColor(...colorSlate)
    doc.text(`Conducted: ${overview.latest_assessment_date || 'N/A'}`, margin + 110, y + 7)

    // Class probabilities table
    const probs = overview.latest_probabilities || {}
    const pND = probs.no_diabetes != null ? `${(probs.no_diabetes * 100).toFixed(1)}%` : '—'
    const pPre = probs.prediabetes != null ? `${(probs.prediabetes * 100).toFixed(1)}%` : '—'
    const pDM = probs.diabetes != null ? `${(probs.diabetes * 100).toFixed(1)}%` : '—'

    doc.setFont('helvetica', 'bold')
    doc.setFontSize(8)
    doc.setTextColor(...colorNavy)
    doc.text('Model Posterior Probabilities:', margin + 4, y + 16)

    doc.setFont('helvetica', 'normal')
    doc.setFontSize(8)
    doc.setTextColor(16, 185, 129)
    doc.text(`• Class 0 (No Diabetes): ${pND}`, margin + 6, y + 23)
    doc.setTextColor(217, 119, 6)
    doc.text(`• Class 1 (Prediabetes): ${pPre}`, margin + 65, y + 23)
    doc.setTextColor(225, 29, 72)
    doc.text(`• Class 2 (Diabetes): ${pDM}`, margin + 125, y + 23)

    doc.setFont('helvetica', 'bold')
    doc.setFontSize(8)
    doc.setTextColor(...colorNavy)
    doc.text('Reported Indicators at Evaluation:', margin + 4, y + 31)

    doc.setFont('helvetica', 'normal')
    doc.setTextColor(51, 65, 85)
    doc.text(`BMI: ${overview.latest_bmi} (${overview.latest_bmi_category})`, margin + 65, y + 31)
    doc.text(`General Health: ${overview.latest_gen_hlth_label || 'N/A'}`, margin + 125, y + 31)
  } else {
    doc.setFont('helvetica', 'normal')
    doc.setFontSize(8.5)
    doc.setTextColor(...colorSlate)
    doc.text('No completed assessments found for this user account. Start an assessment to see results.', margin + 6, y + 20)
  }

  // Section 4: Historical Prediction Distribution
  y += 45
  doc.setFont('helvetica', 'bold')
  doc.setFontSize(11)
  doc.setTextColor(...colorNavy)
  doc.text('3. Historical Model Classifications Distribution', margin, y)

  y += 5
  doc.setFillColor(255, 255, 255)
  doc.setDrawColor(...colorBorder)
  doc.roundedRect(margin, y, contentWidth, 36, 2, 2, 'FD')

  const dist = overview?.class_distribution || { '0': 0, '1': 0, '2': 0 }
  const c0 = dist['0'] || 0
  const c1 = dist['1'] || 0
  const c2 = dist['2'] || 0
  const totalD = c0 + c1 + c2
  const pct0 = totalD > 0 ? ((c0 / totalD) * 100).toFixed(1) : '0.0'
  const pct1 = totalD > 0 ? ((c1 / totalD) * 100).toFixed(1) : '0.0'
  const pct2 = totalD > 0 ? ((c2 / totalD) * 100).toFixed(1) : '0.0'

  doc.setFont('helvetica', 'bold')
  doc.setFontSize(8)
  doc.setTextColor(...colorNavy)
  doc.text(`Total Saved Predictions: ${totalD}`, margin + 4, y + 7)

  // Distribution row 1: Class 0
  doc.setFont('helvetica', 'normal')
  doc.setFontSize(8)
  doc.setTextColor(16, 185, 129)
  doc.text(`• Class 0 — No Diabetes:`, margin + 6, y + 16)
  doc.setFont('helvetica', 'bold')
  doc.setTextColor(...colorNavy)
  doc.text(`${c0} evaluations (${pct0}%)`, margin + 50, y + 16)

  // Distribution row 2: Class 1
  doc.setFont('helvetica', 'normal')
  doc.setTextColor(217, 119, 6)
  doc.text(`• Class 1 — Prediabetes:`, margin + 6, y + 23)
  doc.setFont('helvetica', 'bold')
  doc.setTextColor(...colorNavy)
  doc.text(`${c1} evaluations (${pct1}%)`, margin + 50, y + 23)

  // Distribution row 3: Class 2
  doc.setFont('helvetica', 'normal')
  doc.setTextColor(225, 29, 72)
  doc.text(`• Class 2 — Diabetes:`, margin + 6, y + 30)
  doc.setFont('helvetica', 'bold')
  doc.setTextColor(...colorNavy)
  doc.text(`${c2} evaluations (${pct2}%)`, margin + 50, y + 30)

  doc.setFont('helvetica', 'normal')
  doc.setFontSize(6.5)
  doc.setTextColor(...colorSlate)
  doc.text(
    '* Scientific Note: Historical classifications reflect mathematical outputs of the frozen Logistic Regression model across self-reported survey indicators and do NOT represent clinical disease prevalence or confirmed medical diagnosis.',
    margin + 90,
    y + 16,
    { maxWidth: contentWidth - 94 }
  )

  addFooter(1)

  // ═════════════════════════════════════════════════════════════════════════
  // PAGE 2: BMI Trends, General Health Trends, Indicator Comparison
  // ═════════════════════════════════════════════════════════════════════════
  doc.addPage()

  doc.setFillColor(...colorBlue)
  doc.rect(0, 0, pageWidth, 5, 'F')

  doc.setFont('helvetica', 'bold')
  doc.setFontSize(14)
  doc.setTextColor(...colorNavy)
  doc.text('Longitudinal Trends & Indicator Comparison', margin, 18)

  doc.setFont('helvetica', 'normal')
  doc.setFontSize(8)
  doc.setTextColor(...colorSlate)
  doc.text('DiabetesRisk AI • Personal Health Trajectory Analysis', margin, 23)

  doc.setDrawColor(...colorBorder)
  doc.setLineWidth(0.5)
  doc.line(margin, 26, pageWidth - margin, 26)

  // Section 5: BMI Trend Summary
  y = 32
  doc.setFont('helvetica', 'bold')
  doc.setFontSize(11)
  doc.setTextColor(...colorNavy)
  doc.text('4. Longitudinal BMI History & Trajectory', margin, y)

  y += 5
  doc.setFillColor(...colorLightBg)
  doc.setDrawColor(...colorBorder)
  doc.roundedRect(margin, y, contentWidth, 32, 2, 2, 'FD')

  const bmiSum = bmiTrends?.summary || overview?.bmi_summary || {}
  doc.setFont('helvetica', 'bold')
  doc.setFontSize(8)
  doc.setTextColor(...colorNavy)
  doc.text(`Latest BMI: ${bmiSum.latest ?? '—'} (${bmiSum.category || 'N/A'})`, margin + 4, y + 7)
  doc.text(`Previous BMI: ${bmiSum.previous ?? '—'}`, margin + 80, y + 7)
  doc.text(
    `Change: ${bmiSum.change != null ? (bmiSum.change > 0 ? `+${bmiSum.change}` : String(bmiSum.change)) : 'N/A (Requires ≥ 2 assessments)'}`,
    margin + 130,
    y + 7
  )

  doc.setFont('helvetica', 'normal')
  doc.setFontSize(7)
  doc.setTextColor(...colorSlate)
  doc.text('Recent BMI Data Points (Oldest to Newest):', margin + 4, y + 14)

  const pts = bmiTrends?.points || []
  let ptStr = pts.length > 0
    ? pts.slice(-6).map((p) => `${p.date}: ${p.bmi}`).join('  |  ')
    : 'No longitudinal data recorded yet.'
  doc.setFont('helvetica', 'normal')
  doc.setFontSize(7.5)
  doc.setTextColor(51, 65, 85)
  doc.text(ptStr, margin + 4, y + 20)

  doc.setFont('helvetica', 'normal')
  doc.setFontSize(6.5)
  doc.setTextColor(...colorSlate)
  doc.text(
    'Reference screening ranges: Underweight (<18.5) • Normal (18.5–24.9) • Overweight (25.0–29.9) • Obesity (≥30.0). BMI is a screening classification, not an individual diagnosis.',
    margin + 4,
    y + 27
  )

  // Section 6: General Health Trends
  y += 38
  doc.setFont('helvetica', 'bold')
  doc.setFontSize(11)
  doc.setTextColor(...colorNavy)
  doc.text('5. General Health & Well-being Trends (Self-Reported)', margin, y)

  y += 5
  doc.setFillColor(255, 255, 255)
  doc.setDrawColor(...colorBorder)
  doc.roundedRect(margin, y, contentWidth, 28, 2, 2, 'FD')

  const hlthSum = healthTrends?.summary || overview?.gen_hlth_summary || {}
  doc.setFont('helvetica', 'bold')
  doc.setFontSize(8)
  doc.setTextColor(...colorNavy)
  doc.text(`Latest GenHlth: ${hlthSum.latest_label || hlthSum.latest || '—'}`, margin + 4, y + 7)
  doc.text(`Previous Rating: ${hlthSum.previous ?? '—'}`, margin + 80, y + 7)
  doc.text(
    `Rating Change: ${hlthSum.change != null ? (hlthSum.change > 0 ? `+${hlthSum.change}` : String(hlthSum.change)) : 'N/A'}`,
    margin + 130,
    y + 7
  )

  doc.setFont('helvetica', 'normal')
  doc.setFontSize(6.8)
  doc.setTextColor(...colorSlate)
  doc.text(
    'CDC Scale: 1 = Excellent, 2 = Very Good, 3 = Good, 4 = Fair, 5 = Poor. These values represent self-reported survey responses and must not be described as medically verified improvement or deterioration.',
    margin + 4,
    y + 15,
    { maxWidth: contentWidth - 8 }
  )

  // Section 7: Key Health Indicator Comparison Table
  y += 35
  doc.setFont('helvetica', 'bold')
  doc.setFontSize(11)
  doc.setTextColor(...colorNavy)
  doc.text('6. Key Health Indicator Comparison (Recent Assessments)', margin, y)

  y += 5
  // Table headers
  const colW = [16, 22, 16, 16, 16, 16, 18, 18, 18, 26]
  const colHeaders = ['ID', 'Date', 'BMI', 'HighBP', 'HighChol', 'PhysAct', 'Smoker', 'GenHlth', 'DiffWalk', 'Class']

  doc.setFillColor(241, 245, 249)
  doc.rect(margin, y, contentWidth, 6, 'F')
  doc.setFont('helvetica', 'bold')
  doc.setFontSize(6.5)
  doc.setTextColor(...colorNavy)

  let curX = margin
  colHeaders.forEach((h, i) => {
    doc.text(h, curX + 2, y + 4.2)
    curX += colW[i]
  })

  y += 6
  const recs = (indicators?.records || []).slice(0, 6)
  if (recs.length === 0) {
    doc.setFont('helvetica', 'normal')
    doc.setFontSize(7.5)
    doc.setTextColor(...colorSlate)
    doc.text('No historical assessments to compare.', margin + 4, y + 8)
    y += 12
  } else {
    recs.forEach((r, idx) => {
      const rowY = y + idx * 6.5
      if (idx % 2 === 1) {
        doc.setFillColor(250, 250, 250)
        doc.rect(margin, rowY, contentWidth, 6.5, 'F')
      }
      doc.setFont('helvetica', 'normal')
      doc.setFontSize(6.5)
      doc.setTextColor(51, 65, 85)

      let rx = margin
      doc.text(`#${r.assessment_id}`, rx + 2, rowY + 4.5); rx += colW[0]
      doc.text(r.date.substring(0, 10), rx + 2, rowY + 4.5); rx += colW[1]
      doc.text(String(r.bmi), rx + 2, rowY + 4.5); rx += colW[2]
      doc.text(r.high_bp === 1 ? 'Yes' : 'No', rx + 2, rowY + 4.5); rx += colW[3]
      doc.text(r.high_chol === 1 ? 'Yes' : 'No', rx + 2, rowY + 4.5); rx += colW[4]
      doc.text(r.phys_activity === 1 ? 'Yes' : 'No', rx + 2, rowY + 4.5); rx += colW[5]
      doc.text(r.smoker === 1 ? 'Yes' : 'No', rx + 2, rowY + 4.5); rx += colW[6]
      doc.text(String(r.gen_hlth), rx + 2, rowY + 4.5); rx += colW[7]
      doc.text(r.diff_walk === 1 ? 'Yes' : 'No', rx + 2, rowY + 4.5); rx += colW[8]
      doc.text(r.predicted_label || '—', rx + 2, rowY + 4.5); rx += colW[9]
    })
    y += recs.length * 6.5 + 4
  }

  addFooter(2)

  // ═════════════════════════════════════════════════════════════════════════
  // PAGE 3: Assessment Timeline, Research Model Benchmark & Disclaimers
  // ═════════════════════════════════════════════════════════════════════════
  doc.addPage()

  doc.setFillColor(...colorBlue)
  doc.rect(0, 0, pageWidth, 5, 'F')

  doc.setFont('helvetica', 'bold')
  doc.setFontSize(14)
  doc.setTextColor(...colorNavy)
  doc.text('Assessment Activity & Model Research Benchmarks', margin, 18)

  doc.setFont('helvetica', 'normal')
  doc.setFontSize(8)
  doc.setTextColor(...colorSlate)
  doc.text('DiabetesRisk AI • Phase 5 Model Evaluation & Disclaimers', margin, 23)

  doc.setDrawColor(...colorBorder)
  doc.setLineWidth(0.5)
  doc.line(margin, 26, pageWidth - margin, 26)

  // Section 8: Assessment Activity Timeline
  y = 32
  doc.setFont('helvetica', 'bold')
  doc.setFontSize(11)
  doc.setTextColor(...colorNavy)
  doc.text('7. Assessment Activity Timeline (Recent Evaluations)', margin, y)

  y += 5
  const timelineItems = (timeline?.items || []).slice(0, 5)
  if (timelineItems.length === 0) {
    doc.setFont('helvetica', 'normal')
    doc.setFontSize(8)
    doc.setTextColor(...colorSlate)
    doc.text('No assessment activity recorded yet.', margin + 4, y + 8)
    y += 14
  } else {
    timelineItems.forEach((t) => {
      doc.setFillColor(...colorLightBg)
      doc.setDrawColor(...colorBorder)
      doc.roundedRect(margin, y, contentWidth, 12, 1.5, 1.5, 'FD')

      doc.setFont('helvetica', 'bold')
      doc.setFontSize(7.5)
      doc.setTextColor(...colorNavy)
      doc.text(`Assessment #${t.assessment_id}`, margin + 4, y + 5)
      doc.setFont('helvetica', 'normal')
      doc.setTextColor(...colorSlate)
      doc.text(`${t.date}`, margin + 35, y + 5)

      const tCls = t.predicted_class
      const tColor = tCls === 2 ? [225, 29, 72] : tCls === 1 ? [217, 119, 6] : [16, 185, 129]
      doc.setFont('helvetica', 'bold')
      doc.setTextColor(...tColor)
      doc.text(`${t.predicted_label || 'Evaluated'}`, margin + 70, y + 5)

      doc.setFont('helvetica', 'normal')
      doc.setTextColor(51, 65, 85)
      doc.text(`Model Probability: ${t.confidence ? `${(t.confidence * 100).toFixed(1)}%` : '—'}`, margin + 115, y + 5)
      doc.text(`BMI: ${t.bmi} | GenHlth: ${t.gen_hlth}`, margin + 4, y + 9.5)

      y += 14
    })
  }

  // Section 9: Model Performance — Research Benchmark (Phase 5 Frozen Metrics)
  y += 2
  doc.setFont('helvetica', 'bold')
  doc.setFontSize(11)
  doc.setTextColor(...colorNavy)
  doc.text('8. Model Performance — Research Benchmark', margin, y)

  y += 4
  doc.setFont('helvetica', 'normal')
  doc.setFontSize(6.8)
  doc.setTextColor(...colorSlate)
  doc.text(
    'Frozen Phase 5 held-out test evaluation (CDC BRFSS 2015, 50,736 rows). This represents population-level research metrics and NOT personal health diagnoses.',
    margin,
    y
  )

  y += 4
  doc.setFillColor(...colorLightBg)
  doc.setDrawColor(...colorBorder)
  doc.roundedRect(margin, y, contentWidth, 38, 2, 2, 'FD')

  // Global metrics row
  doc.setFont('helvetica', 'bold')
  doc.setFontSize(7.5)
  doc.setTextColor(...colorNavy)
  doc.text('Frozen Global Evaluation Benchmarks:', margin + 4, y + 6)

  doc.setFont('helvetica', 'normal')
  doc.setFontSize(7)
  doc.setTextColor(51, 65, 85)
  doc.text('• Macro F1: 0.4246', margin + 6, y + 13)
  doc.text('• Weighted F1: 0.7187', margin + 50, y + 13)
  doc.text('• Balanced Accuracy: 0.5172', margin + 95, y + 13)
  doc.text('• Accuracy: 0.6433', margin + 140, y + 13)
  doc.text('• ROC-AUC OvR Macro: 0.7707', margin + 6, y + 19)
  doc.text('• ROC-AUC OvR Weighted: 0.8112', margin + 65, y + 19)

  // Class-wise metrics
  doc.setFont('helvetica', 'bold')
  doc.setFontSize(7.5)
  doc.setTextColor(...colorNavy)
  doc.text('Class-Wise Metrics (Precision / Recall / F1):', margin + 4, y + 26)

  doc.setFont('helvetica', 'normal')
  doc.setFontSize(7)
  doc.setTextColor(16, 185, 129)
  doc.text('Class 0 (No Diabetes): P 0.9501 | R 0.6606 | F1 0.7793', margin + 6, y + 33)
  doc.setTextColor(217, 119, 6)
  doc.text('Class 1 (Prediabetes): P 0.0307 | R 0.3089 | F1 0.0558', margin + 70, y + 33)
  doc.setTextColor(225, 29, 72)
  doc.text('Class 2 (Diabetes): P 0.3520 | R 0.5823 | F1 0.4387', margin + 132, y + 33)

  // Disclaimers & Limitations
  y += 42
  doc.setFillColor(254, 243, 199) // amber-100
  doc.setDrawColor(245, 158, 11) // amber-500
  doc.roundedRect(margin, y, contentWidth, 18, 1.5, 1.5, 'FD')

  doc.setFont('helvetica', 'bold')
  doc.setFontSize(7)
  doc.setTextColor(180, 83, 9)
  doc.text('Prediabetes (Class 1) Research Limitation Disclosure:', margin + 3, y + 4.5)

  doc.setFont('helvetica', 'normal')
  doc.setFontSize(6.5)
  doc.setTextColor(146, 64, 14)
  const lim =
    'Due to heavy class imbalance in the general survey population (~1.8% prediabetes representation), Class 1 achieves a precision of 0.0307 and recall of 0.3089. The model is intentionally configured with balanced class weights to maximize sensitivity to early metabolic risk, but individual predictions in Class 1 carry substantial uncertainty.'
  doc.text(doc.splitTextToSize(lim, contentWidth - 6), margin + 3, y + 8.5)

  y += 21
  doc.setFillColor(...colorLightBg)
  doc.setDrawColor(...colorBorder)
  doc.roundedRect(margin, y, contentWidth, 22, 1.5, 1.5, 'FD')

  doc.setFont('helvetica', 'bold')
  doc.setFontSize(7)
  doc.setTextColor(...colorNavy)
  doc.text('Mandatory Academic Research & Medical Disclaimer:', margin + 3, y + 4.5)

  doc.setFont('helvetica', 'normal')
  doc.setFontSize(6.2)
  doc.setTextColor(...colorSlate)
  const disc =
    'DiabetesRisk AI is an academic research engineering demonstration. It is not an FDA-cleared diagnostic tool. Model-estimated probabilities are statistical survey approximations and must never be interpreted as clinical risk percentages or confirmed medical diagnoses. Do not make healthcare or medication decisions based on this report. Always consult a licensed physician or endocrinologist for clinical laboratory testing (HbA1c or Fasting Plasma Glucose).'
  doc.text(doc.splitTextToSize(disc, contentWidth - 6), margin + 3, y + 8.5)

  addFooter(3)



  // ═══════════════════════════════════════════════════════════════
  // SECTION 11: XAI FEATURE DIRECTION SUMMARY
  // ═══════════════════════════════════════════════════════════════
  if (y > pageHeight - 60) { doc.addPage(); y = margin; addHeader(); }

  y += 8
  doc.setFillColor(...colorNavy)
  doc.rect(margin, y, contentWidth, 7, 'F')
  doc.setFont('helvetica', 'bold')
  doc.setFontSize(8)
  doc.setTextColor(255, 255, 255)
  doc.text('Section 11 — AI Explanation: Feature Direction Summary', margin + 3, y + 4.5)
  y += 11

  doc.setFont('helvetica', 'normal')
  doc.setFontSize(6.5)
  doc.setTextColor(...colorSlate)
  doc.text(
    'The following health factors are associated with the predicted class based on logistic regression coefficient analysis. This is for educational purposes only — correlation, not causation.',
    margin, y, { maxWidth: contentWidth }
  )
  y += 10

  const xaiByClass = {
    0: [
      ['BMI', 'Lower values associated with Class 0'],
      ['High Blood Pressure', 'Absent in Class 0 group'],
      ['General Health', 'Better self-reported health'],
      ['Physical Activity', 'Regular exercise present'],
      ['Income Level', 'Higher income bracket'],
    ],
    1: [
      ['BMI', 'Elevated BMI associated with pre-diabetes'],
      ['High Blood Pressure', 'Often co-present'],
      ['Age', 'Middle-age bracket more common'],
      ['Physical Activity', 'Reduced activity level'],
      ['Fruits/Vegetables', 'Lower dietary quality'],
    ],
    2: [
      ['BMI', 'High BMI strongly associated with Class 2'],
      ['High Blood Pressure', 'Very high co-occurrence'],
      ['General Health', 'Poor self-reported health'],
      ['Age', 'Older age groups predominant'],
      ['Difficulty Walking', 'Often co-reported'],
    ],
  }

  const pClass = result?.predicted_class ?? 0
  const xaiRows = xaiByClass[pClass] || xaiByClass[0]

  // Table header
  doc.setFillColor(...colorLightBg)
  doc.setDrawColor(...colorBorder)
  doc.rect(margin, y, contentWidth, 6, 'FD')
  doc.setFont('helvetica', 'bold')
  doc.setFontSize(6.5)
  doc.setTextColor(...colorNavy)
  doc.text('Health Factor', margin + 3, y + 4)
  doc.text('Contribution Direction', margin + 70, y + 4)
  y += 6

  xaiRows.forEach(([factor, direction], idx) => {
    const rowBg = idx % 2 === 0 ? [255, 255, 255] : [248, 250, 252]
    doc.setFillColor(...rowBg)
    doc.rect(margin, y, contentWidth, 5.5, 'F')
    doc.setFont('helvetica', 'normal')
    doc.setFontSize(6.2)
    doc.setTextColor(...colorSlate)
    doc.text(factor, margin + 3, y + 3.5)
    doc.text(direction, margin + 70, y + 3.5)
    y += 5.5
  })

  y += 4
  doc.setFont('helvetica', 'italic')
  doc.setFontSize(5.8)
  doc.setTextColor(150, 150, 150)
  doc.text('These directions are derived from logistic regression coefficients and represent population-level statistical associations, not individual-level causal factors.', margin, y, { maxWidth: contentWidth })
  y += 8

  // ═══════════════════════════════════════════════════════════════
  // SECTION 12: LONGITUDINAL HEALTH SUMMARY
  // ═══════════════════════════════════════════════════════════════
  if (y > pageHeight - 55) { doc.addPage(); y = margin; addHeader(); }

  y += 4
  doc.setFillColor(...colorNavy)
  doc.rect(margin, y, contentWidth, 7, 'F')
  doc.setFont('helvetica', 'bold')
  doc.setFontSize(8)
  doc.setTextColor(255, 255, 255)
  doc.text('Section 12 — Longitudinal Health Summary', margin + 3, y + 4.5)
  y += 11

  if (historicalStats) {
    const statsRows = [
      ['Total Assessments Completed', String(historicalStats.totalAssessments ?? 'N/A')],
      ['Average BMI (All Assessments)', historicalStats.avgBmi ? historicalStats.avgBmi.toFixed(1) : 'N/A'],
      ['Most Recent Risk Class', historicalStats.latestClass ?? 'N/A'],
      ['Trend', historicalStats.trend ?? 'Insufficient data'],
    ]
    statsRows.forEach(([label, value], idx) => {
      const rowBg = idx % 2 === 0 ? [255, 255, 255] : [248, 250, 252]
      doc.setFillColor(...rowBg)
      doc.setDrawColor(...colorBorder)
      doc.rect(margin, y, contentWidth, 5.5, 'FD')
      doc.setFont('helvetica', 'bold')
      doc.setFontSize(6.2)
      doc.setTextColor(...colorNavy)
      doc.text(label, margin + 3, y + 3.5)
      doc.setFont('helvetica', 'normal')
      doc.setTextColor(...colorSlate)
      doc.text(value, margin + 90, y + 3.5)
      y += 5.5
    })
  } else {
    doc.setFillColor(...colorLightBg)
    doc.setDrawColor(...colorBorder)
    doc.roundedRect(margin, y, contentWidth, 12, 1.5, 1.5, 'FD')
    doc.setFont('helvetica', 'normal')
    doc.setFontSize(6.5)
    doc.setTextColor(...colorSlate)
    doc.text(
      'Complete additional assessments over time to unlock longitudinal health tracking and trend analysis.',
      margin + 3, y + 7, { maxWidth: contentWidth - 6 }
    )
    y += 14
  }

  y += 6

  // ═══════════════════════════════════════════════════════════════
  // SECTION 13: ACTIONABLE HEALTH EDUCATION NOTES
  // ═══════════════════════════════════════════════════════════════
  if (y > pageHeight - 60) { doc.addPage(); y = margin; addHeader(); }

  doc.setFillColor(...colorNavy)
  doc.rect(margin, y, contentWidth, 7, 'F')
  doc.setFont('helvetica', 'bold')
  doc.setFontSize(8)
  doc.setTextColor(255, 255, 255)
  doc.text('Section 13 — Actionable Health Education Notes', margin + 3, y + 4.5)
  y += 11

  const actionableNotes = {
    0: [
      'Maintain your current healthy lifestyle habits — they are associated with lower metabolic risk.',
      'Schedule routine annual check-ups including fasting glucose and HbA1c if you are over 45 or have family history.',
      'Continue regular physical activity (≥150 min/week moderate aerobic exercise per CDC guidelines).',
      'Sustain a balanced diet rich in vegetables, whole grains, and lean proteins.',
    ],
    1: [
      'Pre-diabetes indicators detected — lifestyle intervention is highly effective at this stage.',
      'Aim for at least 150 minutes of moderate aerobic activity per week (brisk walking, cycling).',
      'Reduce refined carbohydrate and added sugar intake; increase dietary fibre.',
      'Consult your physician for a formal fasting plasma glucose or HbA1c test to confirm metabolic status.',
      'Even a 5–7% reduction in body weight can significantly reduce progression risk (DPP Trial evidence).',
    ],
    2: [
      'Diabetes-associated risk profile detected — please consult a physician for clinical evaluation.',
      'Request formal laboratory testing: HbA1c (goal: <7.0% if treated) and Fasting Plasma Glucose.',
      'Work with a registered dietitian on a structured meal plan emphasising glycaemic index management.',
      'Engage in supervised physical activity appropriate to your fitness level.',
      'Monitor blood pressure and cholesterol as cardiovascular risk is elevated in this population group.',
      'IMPORTANT: This model output is NOT a diagnosis. Only a licensed clinician can diagnose diabetes.',
    ],
  }

  const notes = actionableNotes[pClass] || actionableNotes[0]

  notes.forEach((note, idx) => {
    if (y > pageHeight - 20) { doc.addPage(); y = margin; addHeader(); }
    const rowBg = idx % 2 === 0 ? [255, 255, 255] : [248, 250, 252]
    doc.setFillColor(...rowBg)
    doc.setDrawColor(...colorBorder)
    const noteLines = doc.splitTextToSize(`${idx + 1}. ${note}`, contentWidth - 6)
    const noteH = noteLines.length * 4 + 3
    doc.rect(margin, y, contentWidth, noteH, 'FD')
    doc.setFont('helvetica', 'normal')
    doc.setFontSize(6.2)
    doc.setTextColor(...colorSlate)
    doc.text(noteLines, margin + 3, y + 4)
    y += noteH + 1
  })

  y += 4
  doc.setFont('helvetica', 'italic')
  doc.setFontSize(5.8)
  doc.setTextColor(150, 150, 150)
  doc.text(
    'Educational guidance only. DiabetesRisk AI is not an FDA-cleared medical device. Always consult a licensed healthcare provider before making health decisions.',
    margin, y, { maxWidth: contentWidth }
  )
  y += 8

  // Save PDF
  const filename = `DiabetesRisk_AI_Personal_Analytics_${user?.id || 'Report'}.pdf`
  doc.save(filename)
}


