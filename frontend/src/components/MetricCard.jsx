/**
 * src/components/MetricCard.jsx
 * ==============================
 * Single metric display card for dashboards and performance pages.
 */

export default function MetricCard({ label, value, subtitle, color = 'blue', className = '' }) {
  const colorMap = {
    blue:    'text-blue-600',
    green:   'text-emerald-600',
    amber:   'text-amber-600',
    red:     'text-red-600',
    slate:   'text-slate-600',
    purple:  'text-purple-600',
  }

  return (
    <div className={`card text-center ${className}`}>
      <div className={`text-3xl font-extrabold ${colorMap[color] || colorMap.blue} mb-1`}>
        {value}
      </div>
      <div className="text-sm font-semibold text-slate-700">{label}</div>
      {subtitle && <div className="text-xs text-slate-400 mt-0.5">{subtitle}</div>}
    </div>
  )
}

