/**
 * src/pages/Notifications.jsx
 * ============================
 * Phase 6.17.1 — Full Notifications page.
 * Displays user health notifications with read/unread state,
 * mark-all-read, and clear empty states.
 * Uses the existing fetchNotifications() API and localStorage read-state
 * from NotificationBell.jsx for consistency.
 */

import { useState, useEffect, useCallback } from 'react'
import {
  Bell,
  CheckCheck,
  AlertTriangle,
  Info,
  Calendar,
  Sparkles,
  RefreshCw,
  Loader2,
  Clock,
  Activity,
} from 'lucide-react'
import { Link } from 'react-router-dom'
import { fetchNotifications } from '../services/api'

const STORAGE_KEY = 'diabetes_ai_notifs_read'

function readStoredIds() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]')
  } catch {
    return []
  }
}

function saveStoredIds(ids) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(ids))
  } catch {}
}

function getNotifStyle(type, category) {
  if (type === 'assessment' || category === 'assessment') {
    return {
      icon: <Activity className="w-4 h-4 text-blue-600" />,
      border: 'border-l-blue-600',
      badge: 'bg-blue-50 text-blue-700 border-blue-200',
      label: 'Assessment',
    }
  }

  switch (type) {
    case 'reminder':
      return {
        icon: <Calendar className="w-4 h-4 text-indigo-500" />,
        border: 'border-l-indigo-500',
        badge: 'bg-indigo-50 text-indigo-700 border-indigo-200',
        label: 'Reminder',
      }
    case 'alert':
    case 'urgent':
      return {
        icon: <AlertTriangle className="w-4 h-4 text-amber-500" />,
        border: 'border-l-amber-500',
        badge: 'bg-amber-50 text-amber-700 border-amber-200',
        label: 'Notice',
      }
    case 'success':
      return {
        icon: <CheckCheck className="w-4 h-4 text-emerald-500" />,
        border: 'border-l-emerald-500',
        badge: 'bg-emerald-50 text-emerald-700 border-emerald-200',
        label: 'Milestone',
      }
    case 'ai':
    case 'feature':
      return {
        icon: <Sparkles className="w-4 h-4 text-purple-500" />,
        border: 'border-l-purple-500',
        badge: 'bg-purple-50 text-purple-700 border-purple-200',
        label: 'Research',
      }
    default:
      return {
        icon: <Info className="w-4 h-4 text-slate-500" />,
        border: 'border-l-slate-400',
        badge: 'bg-slate-50 text-slate-700 border-slate-200',
        label: 'Information',
      }
  }
}

function formatRelativeTime(isoString) {
  if (!isoString) return ''
  try {
    const d = new Date(isoString)
    const now = new Date()
    const diffMs = now - d
    const diffMin = Math.floor(diffMs / 60000)
    if (diffMin < 1) return 'Just now'
    if (diffMin < 60) return `${diffMin}m ago`
    const diffHr = Math.floor(diffMin / 60)
    if (diffHr < 24) return `${diffHr}h ago`
    const diffDay = Math.floor(diffHr / 24)
    if (diffDay === 1) return 'Yesterday'
    return `${diffDay} days ago`
  } catch {
    return ''
  }
}

function formatAbsoluteTime(isoString) {
  if (!isoString) return ''
  try {
    return new Date(isoString).toLocaleString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
  } catch {
    return ''
  }
}

export default function Notifications() {
  const [notifications, setNotifications] = useState([])
  const [readIds, setReadIds] = useState(readStoredIds)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [filter, setFilter] = useState('all') // 'all' | 'unread'

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await fetchNotifications()
      setNotifications(data?.notifications ?? [])
    } catch (err) {
      console.error('[Notifications] fetch error:', err)
      setError('Could not load notifications. Please try again.')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    load()
  }, [load])

  function markAllRead() {
    const allIds = notifications.map((n) => n.id)
    setReadIds(allIds)
    saveStoredIds(allIds)
  }

  function markOneRead(id) {
    if (readIds.includes(id)) return
    const updated = [...readIds, id]
    setReadIds(updated)
    saveStoredIds(updated)
  }

  const unreadCount = notifications.filter((n) => !readIds.includes(n.id)).length
  const assessmentCount = notifications.filter((n) => n.category === 'assessment' || n.type === 'assessment').length

  const displayedNotifications =
    filter === 'unread'
      ? notifications.filter((n) => !readIds.includes(n.id))
      : filter === 'assessment'
      ? notifications.filter((n) => n.category === 'assessment' || n.type === 'assessment')
      : notifications

  return (
    <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
      {/* Page Header */}
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-start gap-3">
          <div className="w-10 h-10 bg-blue-50 rounded-xl flex items-center justify-center shrink-0">
            <Bell className="w-5 h-5 text-blue-600" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-slate-900 flex items-center gap-2">
              Notifications
              {unreadCount > 0 && (
                <span className="text-xs bg-red-500 text-white font-bold px-2 py-0.5 rounded-full">
                  {unreadCount}
                </span>
              )}
            </h1>
            <p className="text-sm text-slate-500 mt-0.5">
              Health reminders, assessment alerts, and system updates.
            </p>
          </div>
        </div>
        <button
          onClick={load}
          disabled={loading}
          title="Refresh notifications"
          className="p-2 rounded-xl text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition-colors disabled:opacity-50"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Toolbar */}
      <div className="bg-white border border-slate-200 rounded-2xl px-4 py-3 flex items-center justify-between gap-3">
        {/* Filter tabs */}
        <div className="flex gap-1">
          {[
            { id: 'all', label: `All (${notifications.length})` },
            { id: 'assessment', label: `Assessments (${assessmentCount})` },
            { id: 'unread', label: `Unread (${unreadCount})` },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setFilter(tab.id)}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors capitalize ${
                filter === tab.id
                  ? 'bg-blue-50 text-blue-700 border border-blue-200'
                  : 'text-slate-500 hover:text-slate-700 hover:bg-slate-50'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {unreadCount > 0 && (
          <button
            onClick={markAllRead}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold text-emerald-700 bg-emerald-50 border border-emerald-200 hover:bg-emerald-100 transition-colors"
          >
            <CheckCheck className="w-3.5 h-3.5" />
            Mark all read
          </button>
        )}
      </div>

      {/* Loading */}
      {loading && (
        <div className="flex items-center gap-2 text-slate-500 py-6">
          <Loader2 className="w-4 h-4 animate-spin" />
          <span className="text-sm">Loading notifications…</span>
        </div>
      )}

      {/* Error */}
      {error && !loading && (
        <div className="flex items-center gap-2 bg-red-50 border border-red-200 text-red-700 rounded-xl px-4 py-3 text-sm">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          {error}
        </div>
      )}

      {/* Empty state */}
      {!loading && !error && displayedNotifications.length === 0 && (
        <div className="bg-white border border-slate-200 rounded-2xl py-16 text-center">
          <CheckCheck className="w-10 h-10 text-emerald-300 mx-auto mb-3" />
          <p className="text-sm font-semibold text-slate-700">
            {filter === 'unread' ? 'No unread notifications' : 'All caught up!'}
          </p>
          <p className="text-xs text-slate-400 mt-1">
            {filter === 'unread'
              ? 'Switch to "All" to see your read notifications.'
              : 'Health reminders and alerts will appear here.'}
          </p>
          {filter === 'unread' && (
            <button
              onClick={() => setFilter('all')}
              className="mt-4 text-xs text-blue-600 hover:underline font-medium"
            >
              View all notifications
            </button>
          )}
        </div>
      )}

      {/* Notification list */}
      {!loading && !error && displayedNotifications.length > 0 && (
        <div className="bg-white border border-slate-200 rounded-2xl overflow-hidden divide-y divide-slate-50">
          {displayedNotifications.map((notif) => {
            const isRead = readIds.includes(notif.id)
            const style = getNotifStyle(notif.type, notif.category)

            return (
              <div
                key={notif.id}
                onClick={() => markOneRead(notif.id)}
                className={`flex gap-4 px-5 py-4 border-l-4 ${style.border} cursor-pointer transition-colors ${
                  isRead ? 'bg-white opacity-70 hover:opacity-100' : 'bg-slate-50/60 hover:bg-white'
                }`}
              >
                {/* Icon */}
                <div className="mt-0.5 shrink-0">{style.icon}</div>

                {/* Content */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex items-center gap-2 flex-wrap">
                      <p className={`text-sm text-slate-800 ${!isRead ? 'font-bold' : 'font-semibold'}`}>
                        {notif.title}
                        {!isRead && (
                          <span className="ml-1.5 inline-block w-1.5 h-1.5 rounded-full bg-blue-500 align-middle" />
                        )}
                      </p>
                      <span className={`text-[10px] font-semibold px-1.5 py-0.5 rounded-full border ${style.badge}`}>
                        {style.label}
                      </span>
                    </div>
                    <span className="text-[11px] text-slate-400 shrink-0 flex items-center gap-1 mt-0.5">
                      <Clock className="w-3 h-3" />
                      <span title={formatAbsoluteTime(notif.created_at)}>
                        {formatRelativeTime(notif.created_at)}
                      </span>
                    </span>
                  </div>

                  <p className="text-xs text-slate-500 mt-1 leading-relaxed">
                    {notif.message}
                  </p>

                  {notif.action_url && (
                    <Link
                      to={notif.action_url}
                      onClick={(e) => e.stopPropagation()}
                      className="mt-2 inline-flex items-center gap-1 text-xs font-semibold text-blue-600 hover:text-blue-800 underline"
                    >
                      {notif.action_label ?? 'View details →'}
                    </Link>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* Footer note */}
      {!loading && notifications.length > 0 && (
        <p className="text-center text-xs text-slate-400 pb-2">
          Notifications are based on your health assessment history. Not clinical advice. —{' '}
          <button
            onClick={() => { markAllRead() }}
            className="text-blue-500 hover:underline"
          >
            Mark all read
          </button>
        </p>
      )}
    </div>
  )
}
