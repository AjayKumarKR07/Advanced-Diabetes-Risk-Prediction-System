/**
 * src/utils/cdcFeatures.js
 * =========================
 * Canonical CDC BRFSS 2015 21-feature encodings, option mappings,
 * and age category conversion logic for DiabetesRisk AI.
 */

/**
 * Converts actual chronological age (18–120) to the exact CDC BRFSS 13-category ordinal.
 * Returns null if outside the 18–120 adult range.
 */
export function convertAgeToCdcCategory(rawAge) {
  const age = parseInt(rawAge, 10)
  if (isNaN(age) || age < 18 || age > 120) return null
  if (age <= 24) return 1
  if (age <= 29) return 2
  if (age <= 34) return 3
  if (age <= 39) return 4
  if (age <= 44) return 5
  if (age <= 49) return 6
  if (age <= 54) return 7
  if (age <= 59) return 8
  if (age <= 64) return 9
  if (age <= 69) return 10
  if (age <= 74) return 11
  if (age <= 79) return 12
  return 13 // 80 or older
}

export const CDC_AGE_CATEGORIES = {
  1: { bracket: '18–24 years', desc: 'Age 18 to 24' },
  2: { bracket: '25–29 years', desc: 'Age 25 to 29' },
  3: { bracket: '30–34 years', desc: 'Age 30 to 34' },
  4: { bracket: '35–39 years', desc: 'Age 35 to 39' },
  5: { bracket: '40–44 years', desc: 'Age 40 to 44' },
  6: { bracket: '45–49 years', desc: 'Age 45 to 49' },
  7: { bracket: '50–54 years', desc: 'Age 50 to 54' },
  8: { bracket: '55–59 years', desc: 'Age 55 to 59' },
  9: { bracket: '60–64 years', desc: 'Age 60 to 64' },
  10: { bracket: '65–69 years', desc: 'Age 65 to 69' },
  11: { bracket: '70–74 years', desc: 'Age 70 to 74' },
  12: { bracket: '75–79 years', desc: 'Age 75 to 79' },
  13: { bracket: '80+ years',   desc: 'Age 80 or older' },
}

export const CDC_GENHLTH_OPTIONS = [
  { value: 1, label: '1 — Excellent', desc: 'Energetic, highly healthy' },
  { value: 2, label: '2 — Very Good', desc: 'Minor or no health complaints' },
  { value: 3, label: '3 — Good', desc: 'Average health status' },
  { value: 4, label: '4 — Fair', desc: 'Frequent fatigue or manageable condition' },
  { value: 5, label: '5 — Poor', desc: 'Significant chronic health limitations' },
]

export const CDC_EDUCATION_OPTIONS = [
  { value: 1, label: '1 — Never attended / Kindergarten' },
  { value: 2, label: '2 — Grades 1–8 (Elementary)' },
  { value: 3, label: '3 — Grades 9–11 (Some high school)' },
  { value: 4, label: '4 — Grade 12 or GED (High school graduate)' },
  { value: 5, label: '5 — College 1–3 years (Some college / technical school)' },
  { value: 6, label: '6 — College 4+ years (College graduate)' },
]

export const CDC_INCOME_OPTIONS = [
  { value: 1, label: '1 — Less than $10,000' },
  { value: 2, label: '2 — $10,000 to less than $15,000' },
  { value: 3, label: '3 — $15,000 to less than $20,000' },
  { value: 4, label: '4 — $20,000 to less than $25,000' },
  { value: 5, label: '5 — $25,000 to less than $35,000' },
  { value: 6, label: '6 — $35,000 to less than $50,000' },
  { value: 7, label: '7 — $50,000 to less than $75,000' },
  { value: 8, label: '8 — $75,000 or more' },
]

export function getBmiCategory(bmi) {
  const num = parseFloat(bmi)
  if (isNaN(num)) return { label: 'Invalid', color: 'slate' }
  if (num < 18.5) return { label: 'Underweight (<18.5)', color: 'blue' }
  if (num < 25.0) return { label: 'Normal Weight (18.5–24.9)', color: 'emerald' }
  if (num < 30.0) return { label: 'Overweight (25.0–29.9)', color: 'amber' }
  if (num < 35.0) return { label: 'Obese Class I (30.0–34.9)', color: 'orange' }
  return { label: 'Obese Class II/III (≥35.0)', color: 'rose' }
}

export const CLASS_LABELS = {
  0: 'No Diabetes',
  1: 'Prediabetes',
  2: 'Diabetes',
}

export const CLASS_COLORS = {
  0: '#10b981', // emerald
  1: '#f59e0b', // amber
  2: '#ef4444', // red
}

