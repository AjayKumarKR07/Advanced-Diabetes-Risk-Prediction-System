/**
 * src/components/AuthenticatedLayout.jsx
 * =======================================
 * Persistent Authenticated Application Layout for DiabetesRisk AI.
 * 
 * Features:
 * - Persistent desktop left sidebar with collapsible mode (icons + labels or compact icons)
 * - Exact ordered sidebar navigation:
 *     1. New Assessment — /assess
 *     2. Dashboard — /dashboard
 *     3. History — /history
 *     4. Profile — /profile
 *     5. Settings — /settings
 *     + Logout at bottom
 * - Responsive mobile navigation drawer with backdrop
 * - Top sticky header with dynamic page title, notification bell, and user avatar dropdown
 * - Seamless integration with AuthContext
 */

import { useState, useRef, useEffect } from 'react'
import { Link, useLocation, useNavigate, Outlet } from 'react-router-dom'
import {
  Activity,
  PlusCircle,
  LayoutDashboard,
  Clock,
  User,
  Settings,
  LogOut,
  Menu,
  X,
  ChevronLeft,
  ChevronRight,
  Bell,
  ChevronDown,
  ShieldCheck,
  CheckCircle2,
  ExternalLink,
} from 'lucide-react'
import { useAuth } from '../context/AuthContext'
import AIChatWidget from './AIChatWidget'

export default function AuthenticatedLayout({ children }) {
  const location = useLocation()
  const navigate = useNavigate()
  const { user, logout } = useAuth()

  // Sidebar states
  const [collapsed, setCollapsed] = useState(false)
  const [mobileOpen, setMobileOpen] = useState(false)

  // Header dropdown states
  const [profileDropdownOpen, setProfileDropdownOpen] = useState(false)
  const [notificationsOpen, setNotificationsOpen] = useState(false)

  const dropdownRef = useRef(null)
  const notifRef = useRef(null)

  // Close mobile drawer on route change
  useEffect(() => {
    setMobileOpen(false)
    setProfileDropdownOpen(false)
    setNotificationsOpen(false)
  }, [location.pathname])

  // Close dropdowns on outside click
  useEffect(() => {
    function handleClickOutside(e) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
        setProfileDropdownOpen(false)
      }
      if (notifRef.current && !notifRef.current.contains(e.target)) {
        setNotificationsOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const handleLogout = () => {
    logout()
    navigate('/login')
  }

  // Exact ordered navigation items required by Phase 6.11 specification
  const NAV_ITEMS = [
    { name: 'New Assessment', path: '/assess', icon: PlusCircle, badge: 'New' },
    { name: 'Dashboard', path: '/dashboard', icon: LayoutDashboard },
    { name: 'History', path: '/history', icon: Clock },
    { name: 'Profile', path: '/profile', icon: User },
    { name: 'Settings', path: '/settings', icon: Settings },
  ]

  // Page title mapping
  const PAGE_TITLES = {
    '/assess': 'New Health Assessment',
    '/dashboard': 'Patient Dashboard',
    '/history': 'Assessment History',
    '/profile': 'User Profile Management',
    '/settings': 'Account & Security Settings',
    '/result': 'Assessment Results & Analysis',
  }
  const currentTitle = PAGE_TITLES[location.pathname] || 'DiabetesRisk AI'

  // User initials & server-backed avatar with fallback
  const [avatarError, setAvatarError] = useState(false)
  const avatarUrl = !avatarError && (user?.has_avatar || user?.avatar_url
    ? '/api/profile/avatar'
    : (user?.id ? localStorage.getItem(`diabetes_ai_avatar_${user.id}`) : null))
  const initials = user?.full_name
    ? user.full_name
        .split(' ')
        .map((n) => n[0])
        .join('')
        .toUpperCase()
        .slice(0, 2)
    : 'DR'

  return (
    <div className="min-h-screen bg-slate-50 flex">
      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* DESKTOP SIDEBAR                                                     */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      <aside
        className={`hidden md:flex flex-col bg-white border-r border-slate-200/90 z-20 sticky top-0 h-screen transition-all duration-300 ease-in-out ${
          collapsed ? 'w-20' : 'w-64'
        }`}
      >
        {/* Brand / Logo */}
        <div className="h-16 flex items-center justify-between px-4 border-b border-slate-100 flex-shrink-0">
          <Link to="/dashboard" className="flex items-center gap-2.5 overflow-hidden">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-teal-600 to-blue-600 flex items-center justify-center text-white flex-shrink-0 shadow-sm shadow-teal-500/20">
              <Activity className="w-5 h-5" />
            </div>
            {!collapsed && (
              <div className="flex flex-col">
                <span className="font-extrabold text-slate-900 text-sm tracking-tight">DiabetesRisk AI</span>
                <span className="text-[10px] text-teal-600 font-semibold uppercase tracking-wider">Clinical Analytics</span>
              </div>
            )}
          </Link>
          <button
            type="button"
            onClick={() => setCollapsed(!collapsed)}
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-colors"
            title={collapsed ? 'Expand Sidebar' : 'Collapse Sidebar'}
          >
            {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
          </button>
        </div>

        {/* Navigation Menu */}
        <div className="flex-1 py-6 px-3 space-y-1.5 overflow-y-auto">
          {!collapsed && (
            <div className="px-3 pb-2 text-[11px] font-bold uppercase tracking-wider text-slate-400">
              Application Menu
            </div>
          )}
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon
            const isActive = location.pathname === item.path
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-semibold transition-all ${
                  isActive
                    ? 'bg-blue-50 text-blue-700 shadow-sm border border-blue-100/80 font-bold'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100/70'
                } ${collapsed ? 'justify-center px-0' : ''}`}
                title={collapsed ? item.name : undefined}
              >
                <Icon className={`w-5 h-5 flex-shrink-0 ${isActive ? 'text-blue-600' : 'text-slate-400'}`} />
                {!collapsed && (
                  <span className="truncate flex-1">{item.name}</span>
                )}
                {!collapsed && item.badge && !isActive && (
                  <span className="text-[10px] bg-teal-50 text-teal-700 font-bold px-2 py-0.5 rounded-full border border-teal-200">
                    {item.badge}
                  </span>
                )}
              </Link>
            )
          })}
        </div>

        {/* Public Homepage Link */}
        <div className="px-3 py-2 border-t border-slate-100">
          <Link
            to="/"
            className={`flex items-center gap-3 px-3 py-2 rounded-xl text-xs font-semibold text-slate-500 hover:text-slate-900 hover:bg-slate-50 transition-colors ${
              collapsed ? 'justify-center px-0' : ''
            }`}
            title={collapsed ? 'Public Home' : undefined}
          >
            <ExternalLink className="w-4 h-4 text-slate-400 flex-shrink-0" />
            {!collapsed && <span>Public Portal</span>}
          </Link>
        </div>

        {/* Bottom User & Logout Area */}
        <div className="p-3 border-t border-slate-100 bg-slate-50/50 flex-shrink-0">
          {!collapsed && user && (
            <div className="flex items-center gap-3 px-2 py-2 mb-2 rounded-xl bg-white border border-slate-200/80 shadow-xs">
              <div className="w-9 h-9 rounded-xl bg-blue-600 text-white font-bold text-xs flex items-center justify-center flex-shrink-0 overflow-hidden">
                {avatarUrl ? (
                  <img
                    src={avatarUrl}
                    alt="Avatar"
                    className="w-full h-full object-cover"
                    onError={() => setAvatarError(true)}
                  />
                ) : (
                  initials
                )}
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-xs font-bold text-slate-900 truncate">{user.full_name}</p>
                <p className="text-[10px] text-slate-400 truncate">{user.email}</p>
              </div>
            </div>
          )}

          <button
            type="button"
            onClick={handleLogout}
            className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-semibold text-rose-600 hover:bg-rose-50 transition-colors ${
              collapsed ? 'justify-center px-0' : ''
            }`}
            title={collapsed ? 'Log Out' : undefined}
          >
            <LogOut className="w-5 h-5 flex-shrink-0 text-rose-500" />
            {!collapsed && <span>Log Out</span>}
          </button>
        </div>
      </aside>

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* MOBILE DRAWER BACKDROP & SIDEBAR                                    */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      {mobileOpen && (
        <div
          className="fixed inset-0 z-40 bg-slate-900/50 backdrop-blur-xs md:hidden"
          onClick={() => setMobileOpen(false)}
        />
      )}

      <aside
        className={`fixed top-0 bottom-0 left-0 w-72 bg-white z-50 shadow-2xl transform transition-transform duration-300 ease-in-out md:hidden flex flex-col ${
          mobileOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <div className="h-16 flex items-center justify-between px-5 border-b border-slate-100">
          <Link to="/dashboard" className="flex items-center gap-2.5">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-teal-600 to-blue-600 flex items-center justify-center text-white shadow-sm">
              <Activity className="w-5 h-5" />
            </div>
            <div>
              <span className="font-extrabold text-slate-900 text-sm">DiabetesRisk AI</span>
              <span className="block text-[10px] text-teal-600 font-semibold uppercase">Clinical Portal</span>
            </div>
          </Link>
          <button
            type="button"
            onClick={() => setMobileOpen(false)}
            className="p-2 rounded-xl text-slate-400 hover:text-slate-600 hover:bg-slate-100"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="flex-1 py-6 px-4 space-y-1.5 overflow-y-auto">
          <div className="px-3 pb-2 text-[11px] font-bold uppercase tracking-wider text-slate-400">
            Navigation Menu
          </div>
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon
            const isActive = location.pathname === item.path
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`flex items-center gap-3 px-3.5 py-3 rounded-xl text-sm font-semibold transition-all ${
                  isActive
                    ? 'bg-blue-50 text-blue-700 font-bold border border-blue-100'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
                }`}
              >
                <Icon className={`w-5 h-5 ${isActive ? 'text-blue-600' : 'text-slate-400'}`} />
                <span className="flex-1">{item.name}</span>
                {item.badge && !isActive && (
                  <span className="text-[10px] bg-teal-50 text-teal-700 font-bold px-2 py-0.5 rounded-full border border-teal-200">
                    {item.badge}
                  </span>
                )}
              </Link>
            )
          })}

          <div className="pt-4 border-t border-slate-100">
            <Link
              to="/"
              className="flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-xs font-semibold text-slate-500 hover:text-slate-900"
            >
              <ExternalLink className="w-4 h-4 text-slate-400" />
              <span>Public Homepage</span>
            </Link>
          </div>
        </div>

        <div className="p-4 border-t border-slate-100 bg-slate-50">
          {user && (
            <div className="flex items-center gap-3 px-2 py-2 mb-3 rounded-xl bg-white border border-slate-200">
              <div className="w-9 h-9 rounded-xl bg-blue-600 text-white font-bold text-xs flex items-center justify-center overflow-hidden">
                {avatarUrl ? (
                  <img
                    src={avatarUrl}
                    alt="Avatar"
                    className="w-full h-full object-cover"
                    onError={() => setAvatarError(true)}
                  />
                ) : (
                  initials
                )}
              </div>
              <div className="min-w-0 flex-1">
                <p className="text-xs font-bold text-slate-900 truncate">{user.full_name}</p>
                <p className="text-[10px] text-slate-400 truncate">{user.email}</p>
              </div>
            </div>
          )}
          <button
            type="button"
            onClick={handleLogout}
            className="w-full flex items-center justify-center gap-2 py-2.5 rounded-xl text-sm font-semibold text-rose-600 bg-white border border-rose-200 hover:bg-rose-50 transition-colors"
          >
            <LogOut className="w-4 h-4 text-rose-500" />
            <span>Log Out</span>
          </button>
        </div>
      </aside>

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* MAIN VIEWPORT AREA                                                  */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top Header */}
        <header className="h-16 bg-white/95 backdrop-blur-sm border-b border-slate-200/90 sticky top-0 z-30 px-4 sm:px-6 lg:px-8 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={() => setMobileOpen(true)}
              className="md:hidden p-2 rounded-xl text-slate-600 hover:bg-slate-100"
            >
              <Menu className="w-5 h-5" />
            </button>
            <h1 className="text-base sm:text-lg font-bold text-slate-900 tracking-tight flex items-center gap-2">
              <span>{currentTitle}</span>
            </h1>
          </div>

          {/* Right Header Controls */}
          <div className="flex items-center gap-3">
            {/* Notifications Menu */}
            <div className="relative" ref={notifRef}>
              <button
                type="button"
                onClick={() => setNotificationsOpen(!notificationsOpen)}
                className="p-2 rounded-xl text-slate-500 hover:text-slate-700 hover:bg-slate-100 relative transition-colors"
                title="Notifications"
              >
                <Bell className="w-4 h-4" />
                <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-blue-600 rounded-full" />
              </button>

              {notificationsOpen && (
                <div className="absolute right-0 mt-2 w-80 bg-white rounded-2xl border border-slate-200 shadow-xl py-3 z-50 animate-in fade-in zoom-in-95">
                  <div className="px-4 py-2 border-b border-slate-100 flex items-center justify-between">
                    <span className="text-xs font-bold text-slate-900">Notifications</span>
                    <span className="text-[10px] bg-blue-50 text-blue-700 font-semibold px-2 py-0.5 rounded-full">
                      1 Unread
                    </span>
                  </div>
                  <div className="p-3">
                    <div className="p-3 rounded-xl bg-blue-50/50 border border-blue-100 flex items-start gap-2.5">
                      <ShieldCheck className="w-4 h-4 text-blue-600 flex-shrink-0 mt-0.5" />
                      <div className="text-xs text-slate-700">
                        <p className="font-semibold text-slate-900">System Ready</p>
                        <p className="text-[11px] text-slate-500 mt-0.5">
                          Logistic Regression model pipeline (Phase 5 frozen artifact) active and verified.
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Profile Dropdown */}
            <div className="relative" ref={dropdownRef}>
              <button
                type="button"
                onClick={() => setProfileDropdownOpen(!profileDropdownOpen)}
                className="flex items-center gap-2 p-1.5 pr-2.5 rounded-xl border border-slate-200 hover:bg-slate-50 transition-colors"
              >
                <div className="w-7 h-7 rounded-lg bg-gradient-to-tr from-teal-600 to-blue-600 text-white font-bold text-xs flex items-center justify-center overflow-hidden">
                  {avatarUrl ? (
                    <img
                      src={avatarUrl}
                      alt="Avatar"
                      className="w-full h-full object-cover"
                      onError={() => setAvatarError(true)}
                    />
                  ) : (
                    initials
                  )}
                </div>
                <span className="text-xs font-bold text-slate-800 hidden sm:inline max-w-[120px] truncate">
                  {user?.full_name?.split(' ')[0] || 'Account'}
                </span>
                <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
              </button>

              {profileDropdownOpen && (
                <div className="absolute right-0 mt-2 w-56 bg-white rounded-2xl border border-slate-200 shadow-xl py-2 z-50 animate-in fade-in zoom-in-95">
                  <div className="px-4 py-2.5 border-b border-slate-100">
                    <p className="text-xs font-bold text-slate-900 truncate">{user?.full_name}</p>
                    <p className="text-[11px] text-slate-400 truncate">{user?.email}</p>
                  </div>

                  <div className="py-1">
                    <Link
                      to="/profile"
                      className="flex items-center gap-2.5 px-4 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50 hover:text-blue-600"
                    >
                      <User className="w-4 h-4 text-slate-400" />
                      <span>My Profile</span>
                    </Link>
                    <Link
                      to="/settings"
                      className="flex items-center gap-2.5 px-4 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50 hover:text-blue-600"
                    >
                      <Settings className="w-4 h-4 text-slate-400" />
                      <span>Settings &amp; Security</span>
                    </Link>
                    <Link
                      to="/assess"
                      className="flex items-center gap-2.5 px-4 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50 hover:text-blue-600"
                    >
                      <PlusCircle className="w-4 h-4 text-slate-400" />
                      <span>New Assessment</span>
                    </Link>
                  </div>

                  <div className="pt-1 border-t border-slate-100">
                    <button
                      type="button"
                      onClick={handleLogout}
                      className="w-full flex items-center gap-2.5 px-4 py-2 text-xs font-semibold text-rose-600 hover:bg-rose-50"
                    >
                      <LogOut className="w-4 h-4 text-rose-500" />
                      <span>Sign Out</span>
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </header>

        {/* Dynamic Page Content */}
        <main className="flex-1">
        {children || <Outlet />}
        </main>
      </div>
      {/* AI Health Assistant floating chat widget */}
      <AIChatWidget />
    </div>
  )
}
