/**
 * NotificationBell.jsx
 * Phase 6.20 — Notification & Assessment Reminder Center
 *
 * Bell icon with unread badge, polling every 5 minutes, and a dropdown
 * showing health reminders with mark-all-read persistence in localStorage.
 */

import React, { useCallback, useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { Bell, X, CheckCheck, AlertTriangle, Info, Calendar, Sparkles, Activity } from 'lucide-react'
import { fetchNotifications } from '../services/api'

const STORAGE_KEY = 'diabetes_ai_notifs_read'
const POLL_INTERVAL_MS = 5 * 60 * 1000 // 5 minutes

/** Map notification type → icon + colour scheme */
function getNotifStyle(type) {
  switch (type) {
    case 'assessment':
      return {
        icon: <Activity className="w-4 h-4 text-blue-600" />,
        border: 'border-l-blue-600',
        bg: 'bg-blue-50/70',
      }
    case 'alert':
    case 'urgent':
      return {
        icon: <AlertTriangle className="w-4 h-4 text-amber-500" />,
        border: 'border-l-amber-500',
        bg: 'bg-amber-50',
      }
    case 'reminder':
      return {
        icon: <Calendar className="w-4 h-4 text-indigo-500" />,
        border: 'border-l-indigo-500',
        bg: 'bg-indigo-50',
      }
    case 'success':
      return {
        icon: <CheckCheck className="w-4 h-4 text-emerald-500" />,
        border: 'border-l-emerald-500',
        bg: 'bg-emerald-50',
      }
    case 'ai':
    case 'feature':
      return {
        icon: <Sparkles className="w-4 h-4 text-purple-500" />,
        border: 'border-l-purple-500',
        bg: 'bg-purple-50',
      }
    default:
      return {
        icon: <Info className="w-4 h-4 text-slate-500" />,
        border: 'border-l-slate-400',
        bg: 'bg-slate-50',
      }
  }
}

function readNotifIds() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]')
  } catch {
    return []
  }
}

function saveNotifIds(ids) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(ids))
  } catch {}
}

function formatTime(isoString) {
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
    return `${diffDay}d ago`
  } catch {
    return ''
  }
}

export default function NotificationBell() {
  const [open, setOpen] = useState(false)
  const [notifications, setNotifications] = useState([])
  const [readIds, setReadIds] = useState(readNotifIds)
  const [loading, setLoading] = useState(false)
  const dropdownRef = useRef(null)
  const intervalRef = useRef(null)

  /* ---- Fetch notifications ---- */
  const loadNotifications = useCallback(async () => {
    try {
      setLoading(true)
      const data = await fetchNotifications()
      setNotifications(data?.notifications ?? [])
    } catch (err) {
      console.error('[NotificationBell] fetch error:', err)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    loadNotifications()
    intervalRef.current = setInterval(loadNotifications, POLL_INTERVAL_MS)
    return () => clearInterval(intervalRef.current)
  }, [loadNotifications])

  /* ---- Close on outside click ---- */
  useEffect(() => {
    function handleClickOutside(e) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
        setOpen(false)
      }
    }
    if (open) document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [open])

  /* ---- Computed unread count ---- */
  const unreadCount = notifications.filter((n) => !readIds.includes(n.id)).length

  /* ---- Mark all read ---- */
  function markAllRead() {
    const allIds = notifications.map((n) => n.id)
    setReadIds(allIds)
    saveNotifIds(allIds)
  }

  return (
    <div className="relative" ref={dropdownRef}>
      {/* Bell button */}
      <button
        onClick={() => setOpen((v) => !v)}
        className="relative p-2 rounded-xl text-slate-500 hover:text-slate-700 hover:bg-slate-100 transition-colors"
        aria-label="Notifications"
      >
        <Bell className="w-5 h-5" />
        {unreadCount > 0 && (
          <span className="absolute -top-0.5 -right-0.5 flex items-center justify-center w-4 h-4 text-[10px] font-bold text-white bg-red-500 rounded-full ring-2 ring-white animate-pulse">
            {unreadCount > 9 ? '9+' : unreadCount}
          </span>
        )}
      </button>

      {/* Dropdown */}
      {open && (
        <div className="absolute right-0 top-full mt-2 w-80 bg-white border border-slate-200 rounded-2xl shadow-xl z-50 overflow-hidden">
          {/* Header */}
          <div className="flex items-center justify-between px-4 py-3 border-b border-slate-100">
            <div className="flex items-center gap-2">
              <Bell className="w-4 h-4 text-slate-600" />
              <span className="text-sm font-semibold text-slate-700">Notifications</span>
              {unreadCount > 0 && (
                <span className="text-xs bg-red-100 text-red-600 px-1.5 py-0.5 rounded-full font-medium">
                  {unreadCount} new
                </span>
              )}
            </div>
            <div className="flex items-center gap-1">
              {unreadCount > 0 && (
                <button
                  onClick={markAllRead}
                  className="text-xs text-blue-600 hover:text-blue-800 px-2 py-1 rounded-lg hover:bg-blue-50 transition-colors"
                >
                  Mark all read
                </button>
              )}
              <button
                onClick={() => setOpen(false)}
                className="p-1 text-slate-400 hover:text-slate-600 rounded-lg hover:bg-slate-100 transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Notification list */}
          <div className="max-h-[380px] overflow-y-auto divide-y divide-slate-50">
            {loading && notifications.length === 0 ? (
              <div className="px-4 py-6 text-center text-sm text-slate-400">
                Loading…
              </div>
            ) : notifications.length === 0 ? (
              <div className="px-4 py-8 text-center">
                <CheckCheck className="w-8 h-8 text-emerald-300 mx-auto mb-2" />
                <p className="text-sm font-medium text-slate-600">All caught up!</p>
                <p className="text-xs text-slate-400 mt-0.5">No new notifications.</p>
              </div>
            ) : (
              notifications.map((notif) => {
                const isRead = readIds.includes(notif.id)
                const style = getNotifStyle(notif.type, notif.category)
                return (
                  <div
                    key={notif.id}
                    className={`px-4 py-3 border-l-4 ${style.border} ${isRead ? 'opacity-60' : style.bg} transition-opacity`}
                  >
                    <div className="flex items-start gap-2">
                      <div className="mt-0.5 shrink-0">{style.icon}</div>
                      <div className="flex-1 min-w-0">
                        <p className={`text-xs font-semibold text-slate-700 ${!isRead ? 'font-bold' : ''}`}>
                          {notif.title}
                        </p>
                        <p className="text-xs text-slate-500 mt-0.5 leading-relaxed line-clamp-2">
                          {notif.message}
                        </p>
                        <div className="flex items-center justify-between mt-1.5">
                          <span className="text-[10px] text-slate-400">
                            {formatTime(notif.created_at)}
                          </span>
                          {notif.action_url && (
                            <Link
                              to={notif.action_url}
                              onClick={() => setOpen(false)}
                              className="text-[10px] text-blue-600 hover:text-blue-800 font-medium underline"
                            >
                              {notif.action_label ?? 'View →'}
                            </Link>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                )
              })
            )}
          </div>

          {/* Footer */}
          {notifications.length > 0 && (
            <div className="px-4 py-2.5 border-t border-slate-100 bg-slate-50">
              <p className="text-[10px] text-slate-400 text-center">
                Health reminders refresh every 5 minutes. Not clinical advice.
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
