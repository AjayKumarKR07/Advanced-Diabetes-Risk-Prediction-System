/**
 * src/components/MetricCard.jsx
 * ==============================
 * Single metric display card for dashboards and performance pages.
 */

export default function MetricCard({ label, value, subtitle, color = 'blue', className = '' }) {
  const colorMap = {
    blue:    'text-blue-600 dark:text-blue-400',
    green:   'text-emerald-600 dark:text-emerald-400',
    amber:   'text-amber-600 dark:text-amber-400',
    red:     'text-rose-600 dark:text-rose-400',
    slate:   'text-slate-600 dark:text-slate-300',
    purple:  'text-purple-600 dark:text-purple-400',
  }

  return (
    <div className={`bg-white dark:bg-slate-800 rounded-2xl border border-slate-200/80 dark:border-slate-700 p-5 shadow-xs text-center text-slate-900 dark:text-white transition-colors ${className}`}>
      <div className={`text-3xl font-extrabold ${colorMap[color] || colorMap.blue} mb-1`}>
        {value}
      </div>
      <div className="text-sm font-semibold text-slate-700 dark:text-slate-200">{label}</div>
      {subtitle && <div className="text-xs text-slate-400 dark:text-slate-400 mt-0.5">{subtitle}</div>}
    </div>
  )
}

