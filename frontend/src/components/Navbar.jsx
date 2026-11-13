/**
 * src/components/Navbar.jsx
 * =========================
 * Responsive top navigation bar with active route highlighting.
 */

import { NavLink } from 'react-router-dom'
import { Activity, BarChart2, ClipboardList, Info, Home, Stethoscope } from 'lucide-react'

const NAV_ITEMS = [
  { to: '/',             label: 'Home',        Icon: Home },
  { to: '/dashboard',   label: 'Dashboard',   Icon: BarChart2 },
  { to: '/assess',      label: 'Assessment',  Icon: ClipboardList },
  { to: '/performance', label: 'Model',       Icon: Activity },
  { to: '/about',       label: 'About',       Icon: Info },
]

export default function Navbar() {
  return (
    <nav className="fixed top-0 inset-x-0 z-50 bg-white/80 backdrop-blur-md border-b border-slate-200 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <NavLink to="/" className="flex items-center gap-2 font-bold text-blue-700">
            <Stethoscope className="w-6 h-6" />
            <span className="hidden sm:block text-sm">Diabetes Risk Predictor</span>
          </NavLink>

          {/* Navigation links */}
          <div className="flex items-center gap-1">
            {NAV_ITEMS.map(({ to, label, Icon }) => (
              <NavLink
                key={to}
                to={to}
                end={to === '/'}
                className={({ isActive }) =>
                  `flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium transition-colors duration-150 ${
                    isActive
                      ? 'bg-blue-50 text-blue-700'
                      : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                  }`
                }
              >
                <Icon className="w-4 h-4" />
                <span className="hidden md:block">{label}</span>
              </NavLink>
            ))}
          </div>
        </div>
      </div>
    </nav>
  )
}

