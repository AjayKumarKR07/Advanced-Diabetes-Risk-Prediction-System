/**
 * src/components/RiskBadge.jsx
 * ============================
 * Standardized predicted-class status badge for DiabetesRisk AI.
 * Phase 6.13.1: Strictly aligned with actual model-predicted classes (0, 1, 2).
 * 
 * Class 0:
 * - Label: No Diabetes
 * - Badge: Class 0 — No Diabetes
 * - Color: Green (Emerald)
 * 
 * Class 1:
 * - Label: Prediabetes
 * - Badge: Class 1 — Prediabetes
 * - Color: Amber
 * 
 * Class 2:
 * - Label: Diabetes
 * - Badge: Class 2 — Diabetes
 * - Color: Red (Rose)
 */

import { CheckCircle2, AlertTriangle, Flame } from 'lucide-react'

const CLASS_CONFIG = {
  0: {
    label: 'Class 0 — No Diabetes',
    className: 'bg-emerald-50 text-emerald-800 border-emerald-200',
    Icon: CheckCircle2,
    iconColor: 'text-emerald-600',
  },
  1: {
    label: 'Class 1 — Prediabetes',
    className: 'bg-amber-50 text-amber-800 border-amber-200',
    Icon: AlertTriangle,
    iconColor: 'text-amber-600',
  },
  2: {
    label: 'Class 2 — Diabetes',
    className: 'bg-rose-50 text-rose-800 border-rose-200',
    Icon: Flame,
    iconColor: 'text-rose-600',
  },
}

export default function RiskBadge({ predictedClass, tier, className = '' }) {
  // Resolve class from integer/string predictedClass or legacy tier string
  let cls = 0
  if (predictedClass !== undefined && predictedClass !== null) {
    cls = parseInt(predictedClass, 10)
    if (isNaN(cls) || cls < 0 || cls > 2) cls = 0
  } else if (typeof tier === 'string') {
    const t = tier.toLowerCase()
    if (t.includes('prediabetes')) cls = 1
    else if (t.includes('diabetes') && !t.includes('no diabetes')) cls = 2
    else cls = 0
  }

  const config = CLASS_CONFIG[cls] || CLASS_CONFIG[0]
  const { label, className: badgeStyle, Icon, iconColor } = config

  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold border ${badgeStyle} ${className}`}
    >
      <Icon className={`w-3.5 h-3.5 ${iconColor}`} />
      <span>{label}</span>
    </span>
  )
}
