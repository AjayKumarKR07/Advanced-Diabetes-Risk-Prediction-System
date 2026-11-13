/**
 * src/components/RiskBadge.jsx
 * ============================
 * Color-coded badge for risk tier display.
 */

import { ShieldCheck, AlertTriangle, AlertCircle } from 'lucide-react'

const TIER_CONFIG = {
  'Low / Baseline Risk': {
    className: 'risk-low',
    Icon: ShieldCheck,
  },
  'Elevated Risk (Prediabetes Warning)': {
    className: 'risk-elevated',
    Icon: AlertTriangle,
  },
  'High Risk (Diabetes Indication)': {
    className: 'risk-high',
    Icon: AlertCircle,
  },
}

export default function RiskBadge({ tier, className = '' }) {
  const config = TIER_CONFIG[tier] || TIER_CONFIG['Low / Baseline Risk']
  const { className: tierClass, Icon } = config

  return (
    <span
      className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-semibold border ${tierClass} ${className}`}
    >
      <Icon className="w-4 h-4" />
      {tier}
    </span>
  )
}

