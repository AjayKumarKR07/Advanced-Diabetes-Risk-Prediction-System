/**
 * src/components/DisclaimerBanner.jsx
 * =====================================
 * Research prototype disclaimer — always visible.
 */

import { AlertTriangle } from 'lucide-react'

export default function DisclaimerBanner({ className = '' }) {
  return (
    <div className={`flex gap-3 items-start bg-amber-50 dark:bg-amber-950/60 border border-amber-200 dark:border-amber-800 rounded-xl px-4 py-3 text-sm text-amber-800 dark:text-amber-200 ${className}`}>
      <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0 text-amber-500 dark:text-amber-400" />
      <p>
        <strong>RESEARCH PROTOTYPE.</strong> This tool is for academic and educational purposes only.
        It is <strong>not a clinical diagnostic instrument</strong> and must not substitute for
        consultation with a qualified healthcare provider.
        Any outputs should not be used to make medical decisions.
      </p>
    </div>
  )
}

