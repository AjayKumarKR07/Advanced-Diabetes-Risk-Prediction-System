/**
 * src/components/Footer.jsx
 * =========================
 * Professional healthcare technology footer.
 * Features:
 *  - Project branding (DiabetesRisk AI) & description
 *  - Quick navigation links
 *  - Technology stack badges
 *  - GitHub repository link
 *  - Prominent academic & educational disclaimer
 */

import { Link } from 'react-router-dom'
import { Stethoscope, Github, ExternalLink, ShieldCheck } from 'lucide-react'

const FOOTER_NAV = [
  { label: 'Home', to: '/' },
  { label: 'Dashboard', to: '/dashboard' },
  { label: 'Health Assessment', to: '/assess' },
  { label: 'Model Performance', to: '/performance' },
  { label: 'Prediction History', to: '/history' },
  { label: 'About & Methodology', to: '/about' },
]

const TECH_STACK = [
  'React 18',
  'FastAPI',
  'Python 3.14',
  'Scikit-learn',
  'PostgreSQL',
  'Tailwind CSS',
  'SQLAlchemy 2.x',
  'Vite',
]

export default function Footer() {
  return (
    <footer className="bg-slate-900 text-slate-300 border-t border-slate-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-10 lg:gap-12 mb-12">
          {/* Col 1: Brand & Description */}
          <div className="md:col-span-2">
            <Link to="/" className="inline-flex items-center gap-2.5 mb-4 group">
              <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-blue-600 to-teal-500 flex items-center justify-center text-white shadow-md shadow-blue-500/20 group-hover:scale-105 transition-transform">
                <Stethoscope className="w-5 h-5" />
              </div>
              <span className="font-extrabold text-white text-xl tracking-tight">
                DiabetesRisk <span className="text-teal-400">AI</span>
              </span>
            </Link>
            <p className="text-sm text-slate-400 leading-relaxed mb-6 max-w-md">
              An advanced AI-powered diabetes risk prediction platform designed for research,
              educational demonstration, and population health awareness. Powered by a group-aware
              machine learning pipeline trained on CDC BRFSS 2015 health indicator data.
            </p>
            <div className="flex items-center gap-3">
              <a
                href="https://github.com/AjayKumarKR07/Advanced-Diabetes-Risk-Prediction-System"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 text-xs font-medium text-slate-300 bg-slate-800 hover:bg-slate-700 hover:text-white px-3.5 py-2 rounded-lg border border-slate-700 transition-colors"
              >
                <Github className="w-4 h-4" />
                <span>GitHub Repository</span>
                <ExternalLink className="w-3 h-3 text-slate-400" />
              </a>
            </div>
          </div>

          {/* Col 2: Navigation */}
          <div>
            <h4 className="text-white font-semibold text-xs uppercase tracking-wider mb-4">
              Navigation
            </h4>
            <ul className="space-y-2.5 text-sm">
              {FOOTER_NAV.map(({ label, to }) => (
                <li key={to}>
                  <Link
                    to={to}
                    className="text-slate-400 hover:text-teal-300 transition-colors inline-flex items-center gap-1.5"
                  >
                    <span>{label}</span>
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Col 3: Technology Stack */}
          <div>
            <h4 className="text-white font-semibold text-xs uppercase tracking-wider mb-4">
              Technology Stack
            </h4>
            <div className="flex flex-wrap gap-1.5">
              {TECH_STACK.map((tech) => (
                <span
                  key={tech}
                  className="px-2.5 py-1 bg-slate-800/90 text-slate-300 text-xs rounded-md border border-slate-700/60 font-mono"
                >
                  {tech}
                </span>
              ))}
            </div>
            <div className="mt-5 p-3 rounded-lg bg-slate-800/50 border border-slate-700/50">
              <div className="flex items-center gap-2 text-xs text-teal-400 font-semibold mb-1">
                <ShieldCheck className="w-3.5 h-3.5" />
                <span>CDC BRFSS 2015 Pipeline</span>
              </div>
              <p className="text-[11px] text-slate-400 leading-normal">
                Frozen held-out evaluation · StratifiedGroupKFold · Leakage-safe validation
              </p>
            </div>
          </div>
        </div>

        {/* Disclaimer & Bottom Bar */}
        <div className="border-t border-slate-800 pt-8 mt-4">
          <div className="bg-slate-800/40 border border-slate-700/40 rounded-xl p-4 mb-6">
            <p className="text-xs text-slate-400 leading-relaxed text-center max-w-4xl mx-auto">
              <strong className="text-slate-300">Important Disclaimer:</strong> This platform is developed
              strictly for educational and research purposes only. Predictions are not medical diagnoses,
              do not substitute for laboratory screening (such as HbA1c or fasting glucose), and must not
              be used for clinical decision-making. Always consult qualified healthcare professionals for
              medical advice, evaluation, and treatment.
            </p>
          </div>
          <div className="flex flex-col sm:flex-row items-center justify-between text-xs text-slate-500 gap-3">
            <p>© {new Date().getFullYear()} DiabetesRisk AI · Advanced Diabetes Risk Prediction System</p>
            <p className="text-slate-500">Powered by CDC BRFSS 2015 · Logistic Regression · ROC-AUC 0.7707</p>
          </div>
        </div>
      </div>
    </footer>
  )
}

