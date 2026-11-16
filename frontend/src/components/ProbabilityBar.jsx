/**
 * src/components/ProbabilityBar.jsx
 * ==================================
 * Animated horizontal probability bar for a single class.
 */

export default function ProbabilityBar({ label, value, color, className = '' }) {
  const pct = Math.round(value * 100)

  const colorMap = {
    green:  { bar: 'bg-emerald-500', text: 'text-emerald-700 dark:text-emerald-400', bg: 'bg-emerald-50 dark:bg-emerald-950/60' },
    amber:  { bar: 'bg-amber-500',   text: 'text-amber-700 dark:text-amber-400',   bg: 'bg-amber-50 dark:bg-amber-950/60' },
    red:    { bar: 'bg-red-500',     text: 'text-red-700 dark:text-red-400',     bg: 'bg-red-50 dark:bg-red-950/60' },
    blue:   { bar: 'bg-blue-500',    text: 'text-blue-700 dark:text-blue-400',    bg: 'bg-blue-50 dark:bg-blue-950/60' },
  }

  const { bar, text, bg } = colorMap[color] || colorMap.blue

  return (
    <div className={`${className}`}>
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-sm font-medium text-slate-700 dark:text-slate-300">{label}</span>
        <span className={`text-sm font-bold ${text}`}>{pct}%</span>
      </div>
      <div className={`w-full rounded-full h-3 ${bg}`}>
        <div
          className={`h-3 rounded-full ${bar} transition-all duration-700 ease-out`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

