/**
 * src/components/Navbar.jsx
 * =========================
 * Professional sticky navigation bar with:
 *  - Healthcare/AI-inspired brand logo & "DiabetesRisk AI" typography
 *  - Desktop navigation items with active indicators
 *  - Authentication state integration (User profile, Logout, Sign In / Register buttons)
 *  - Fully functional responsive mobile navigation drawer
 *  - Consistent styling across all application routes
 */

import { useState } from 'react'
import { NavLink, Link, useLocation } from 'react-router-dom'
import {
  Stethoscope,
  BarChart2,
  ClipboardList,
  Activity,
  Clock,
  Info,
  Home,
  Menu,
  X,
  ArrowRight,
  Sparkles,
  LogOut,
  User,
} from 'lucide-react'
import { useAuth } from '../context/AuthContext'

const NAV_ITEMS = [
  { to: '/',             label: 'Home',              Icon: Home          },
  { to: '/dashboard',   label: 'Dashboard',         Icon: BarChart2     },
  { to: '/assess',      label: 'Health Assessment', Icon: ClipboardList  },
  { to: '/performance', label: 'Model Performance', Icon: Activity      },
  { to: '/history',     label: 'History',           Icon: Clock         },
  { to: '/about',       label: 'About',             Icon: Info          },
]

export default function Navbar() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const location = useLocation()
  const { user, isAuthenticated, logout } = useAuth()

  const closeMenu = () => setMobileMenuOpen(false)

  const handleLogout = () => {
    logout()
    closeMenu()
  }

  // Get user's initials for avatar
  const getInitials = (name) => {
    if (!name) return 'U'
    const parts = name.trim().split(/\s+/)
    if (parts.length === 1) return parts[0].charAt(0).toUpperCase()
    return (parts[0].charAt(0) + parts[parts.length - 1].charAt(0)).toUpperCase()
  }

  return (
    <nav className="fixed top-0 inset-x-0 z-50 bg-white/90 dark:bg-slate-900/90 backdrop-blur-md border-b border-slate-200/80 dark:border-slate-800 shadow-xs transition-all">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Left: Brand Logo & Title */}
          <NavLink
            to="/"
            onClick={closeMenu}
            className="flex items-center gap-2.5 group"
          >
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-blue-600 via-blue-500 to-teal-400 flex items-center justify-center text-white shadow-sm shadow-blue-500/30 group-hover:scale-105 transition-transform duration-200">
              <Stethoscope className="w-5 h-5" />
            </div>
            <div className="flex flex-col">
              <span className="font-extrabold text-slate-900 dark:text-white text-lg tracking-tight leading-none">
                DiabetesRisk <span className="text-teal-600 dark:text-teal-400">AI</span>
              </span>
              <span className="text-[10px] text-slate-400 font-medium tracking-wide">
                HEALTHCARE ANALYTICS
              </span>
            </div>
          </NavLink>

          {/* Center: Desktop Navigation Links */}
          <div className="hidden lg:flex items-center gap-1">
            {NAV_ITEMS.map(({ to, label, Icon }) => (
              <NavLink
                key={to}
                to={to}
                end={to === '/'}
                className={({ isActive }) =>
                  `flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium transition-all duration-150 ${
                    isActive
                      ? 'bg-blue-50 dark:bg-blue-950/60 text-blue-700 dark:text-blue-300 font-semibold shadow-xs'
                      : 'text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white hover:bg-slate-100/80 dark:hover:bg-slate-800'
                  }`
                }
              >
                <Icon className="w-4 h-4" />
                <span>{label}</span>
              </NavLink>
            ))}
          </div>

          {/* Right: Auth controls & Mobile menu toggle */}
          <div className="flex items-center gap-3">
            {isAuthenticated ? (
              <div className="hidden sm:flex items-center gap-3">
                {/* User badge */}
                <div className="flex items-center gap-2 bg-slate-100/90 dark:bg-slate-800 text-slate-800 dark:text-slate-200 rounded-full pl-1.5 pr-3 py-1 text-xs font-semibold border border-slate-200/80 dark:border-slate-700">
                  <div className="w-6 h-6 rounded-full bg-blue-600 text-white flex items-center justify-center text-[10px] font-bold">
                    {getInitials(user?.full_name)}
                  </div>
                  <span className="max-w-[120px] truncate">{user?.full_name?.split(' ')[0]}</span>
                </div>

                {/* Logout Button */}
                <button
                  type="button"
                  onClick={handleLogout}
                  className="flex items-center gap-1.5 text-xs font-semibold text-slate-600 dark:text-slate-300 hover:text-red-600 dark:hover:text-red-400 bg-white dark:bg-slate-800 hover:bg-red-50 dark:hover:bg-red-950/40 border border-slate-200 dark:border-slate-700 hover:border-red-200 dark:hover:border-red-800 px-3 py-2 rounded-xl transition-all"
                  title="Sign out of account"
                >
                  <LogOut className="w-3.5 h-3.5" />
                  <span>Logout</span>
                </button>
              </div>
            ) : (
              <div className="hidden sm:flex items-center gap-2.5">
                <Link
                  to="/login"
                  className="text-sm font-semibold text-slate-700 dark:text-slate-300 hover:text-blue-700 dark:hover:text-blue-400 px-3.5 py-2 rounded-xl transition-colors hover:bg-slate-100/70 dark:hover:bg-slate-800"
                >
                  Sign In
                </Link>
                <Link
                  to="/register"
                  className="inline-flex items-center gap-1.5 bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white font-semibold px-4 py-2 rounded-xl text-sm transition-all duration-200 shadow-sm shadow-blue-600/25 hover:shadow-md hover:shadow-blue-600/35 hover:-translate-y-0.5"
                >
                  <Sparkles className="w-3.5 h-3.5 text-teal-300" />
                  <span>Create Account</span>
                  <ArrowRight className="w-3.5 h-3.5 ml-0.5" />
                </Link>
              </div>
            )}

            {/* Mobile Hamburger Button */}
            <button
              type="button"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="lg:hidden p-2 rounded-lg text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500/20"
              aria-label="Toggle navigation menu"
              aria-expanded={mobileMenuOpen}
            >
              {mobileMenuOpen ? (
                <X className="w-6 h-6" />
              ) : (
                <Menu className="w-6 h-6" />
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Navigation Dropdown / Drawer */}
      {mobileMenuOpen && (
        <div className="lg:hidden bg-white border-b border-slate-200 px-4 pt-3 pb-6 shadow-lg animate-in slide-in-from-top duration-200">
          {isAuthenticated && (
            <div className="mb-4 pb-3 border-b border-slate-100 flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-full bg-blue-600 text-white flex items-center justify-center text-xs font-bold">
                  {getInitials(user?.full_name)}
                </div>
                <div>
                  <div className="text-xs font-bold text-slate-900">{user?.full_name}</div>
                  <div className="text-[11px] text-slate-400 truncate max-w-[180px]">{user?.email}</div>
                </div>
              </div>
              <button
                onClick={handleLogout}
                className="text-xs font-semibold text-red-600 hover:text-red-700 bg-red-50 px-2.5 py-1.5 rounded-lg flex items-center gap-1"
              >
                <LogOut className="w-3 h-3" />
                <span>Logout</span>
              </button>
            </div>
          )}

          <div className="space-y-1">
            {NAV_ITEMS.map(({ to, label, Icon }) => {
              const isActive =
                to === '/'
                  ? location.pathname === '/'
                  : location.pathname.startsWith(to)

              return (
                <NavLink
                  key={to}
                  to={to}
                  end={to === '/'}
                  onClick={closeMenu}
                  className={`flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm font-medium transition-colors ${
                    isActive
                      ? 'bg-blue-50 text-blue-700 font-semibold'
                      : 'text-slate-700 hover:bg-slate-50 hover:text-slate-900'
                  }`}
                >
                  <Icon className={`w-4 h-4 ${isActive ? 'text-blue-600' : 'text-slate-400'}`} />
                  <span>{label}</span>
                </NavLink>
              )
            })}
          </div>

          <div className="mt-4 pt-4 border-t border-slate-100 space-y-2">
            {!isAuthenticated ? (
              <>
                <Link
                  to="/login"
                  onClick={closeMenu}
                  className="w-full flex items-center justify-center gap-2 bg-slate-100 hover:bg-slate-200 text-slate-800 font-semibold px-4 py-2.5 rounded-xl text-sm transition-colors"
                >
                  <User className="w-4 h-4 text-slate-500" />
                  <span>Sign In</span>
                </Link>
                <Link
                  to="/register"
                  onClick={closeMenu}
                  className="w-full flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 text-white font-semibold px-4 py-2.5 rounded-xl text-sm shadow-sm transition-colors"
                >
                  <Sparkles className="w-4 h-4 text-teal-300" />
                  <span>Create Account</span>
                  <ArrowRight className="w-4 h-4" />
                </Link>
              </>
            ) : (
              <Link
                to="/assess"
                onClick={closeMenu}
                className="w-full flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 text-white font-semibold px-4 py-2.5 rounded-xl text-sm shadow-sm transition-colors"
              >
                <Sparkles className="w-4 h-4 text-teal-300" />
                <span>Start Health Assessment</span>
                <ArrowRight className="w-4 h-4" />
              </Link>
            )}
          </div>
        </div>
      )}
    </nav>
  )
}
