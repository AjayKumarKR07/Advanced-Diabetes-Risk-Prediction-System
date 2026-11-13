/**
 * src/pages/History.jsx
 * ======================
 * Advanced Assessment History Management for DiabetesRisk AI.
 * Phase 6.13: View, Download PDF, and Secure Delete for Every Assessment.
 * 
 * Features:
 * - View Action: Opens the full assessment and predictions on /result?assessment_id={id}
 * - Download Action: Generates and downloads DiabetesRisk_AI_Assessment_{id}.pdf using jsPDF
 * - Delete Action: Interactive confirmation modal with secure backend deletion
 * - Real-time Stats Updates: Class distribution and total assessment counters refresh on delete
 * - Responsive Layout: Clean desktop table and mobile-friendly assessment cards
 * - Scientific Correction: Strictly removed unvalidated Risk Tier column; shows true predicted classes
 * - Complete User Isolation: Enforces authenticated user ownership on frontend and backend
 */

import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Clock,
  ChevronLeft,
  ChevronRight,
  RefreshCw,
  AlertCircle,
  Database,
  Eye,
  Download,
  Trash2,
  X,
  User,
  Heart,
  Activity,
  Smile,
  Shield,
  FileText,
  Calendar,
  AlertTriangle,
  CheckCircle2,
  ExternalLink,
} from 'lucide-react'
import {
  fetchPredictions,
  fetchStats,
  fetchAssessment,
  deleteAssessment,
} from '../services/api'
import { useAuth } from '../context/AuthContext'
import { generateAssessmentReportPdf } from '../utils/generatePdfReport'
import {
  CLASS_LABELS,
  CLASS_COLORS,
  CDC_AGE_CATEGORIES,
  CDC_GENHLTH_OPTIONS,
  CDC_EDUCATION_OPTIONS,
  CDC_INCOME_OPTIONS,
  getBmiCategory,
} from '../utils/cdcFeatures'

function ProbPill({ label, value, color }) {
  const colorMap = {
    green: 'bg-emerald-50 text-emerald-700 border-emerald-200',
    amber: 'bg-amber-50 text-amber-700 border-amber-200',
    red: 'bg-rose-50 text-rose-700 border-rose-200',
  }
  return (
    <span
      className={`inline-block border rounded px-2 py-0.5 text-[11px] font-mono font-medium ${colorMap[color] || colorMap.green}`}
    >
      {label} {(value * 100).toFixed(0)}%
    </span>
  )
}

export default function History() {
  const navigate = useNavigate()
  const { user } = useAuth()

  const [page, setPage] = useState(1)
  const [data, setData] = useState(null)
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [successMessage, setSuccessMessage] = useState('')

  // Detail Modal State (Quick Indicators View)
  const [selectedPrediction, setSelectedPrediction] = useState(null)
  const [selectedAssessment, setSelectedAssessment] = useState(null)
  const [loadingDetail, setLoadingDetail] = useState(false)
  const [detailError, setDetailError] = useState('')

  // Delete Modal State
  const [deletingRecord, setDeletingRecord] = useState(null)
  const [isDeleting, setIsDeleting] = useState(false)
  const [deleteError, setDeleteError] = useState('')

  // PDF Download State (tracks which prediction ID is currently generating)
  const [downloadingId, setDownloadingId] = useState(null)

  const PAGE_SIZE = 10

  // ── Load History & Stats ──────────────────────────────────────────────────
  const loadHistory = (p = page) => {
    setLoading(true)
    setError(null)
    Promise.all([
      fetchPredictions(p, PAGE_SIZE),
      fetchStats(),
    ])
      .then(([pred, st]) => {
        setData(pred)
        setStats(st)
        // If the current page is empty after a deletion, go back one page if possible
        if (pred.items.length === 0 && p > 1) {
          setPage(p - 1)
        }
      })
      .catch((err) => {
        setError(
          err.response?.data?.detail ||
            'Could not load assessment history. Please ensure the backend is running.'
        )
      })
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    loadHistory(page)
  }, [page])

  // ── 1. View Action ────────────────────────────────────────────────────────
  const handleViewAssessment = (pred) => {
    navigate(`/result?assessment_id=${pred.assessment_id}`, {
      state: {
        result: pred,
        personalInfo: {
          fullName: pred.user_name || user?.full_name || 'Assessee',
          email: user?.email,
          assessmentDate: new Date(pred.created_at).toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'long',
            day: 'numeric',
          }),
        },
      },
    })
  }

  // ── Quick Detail Modal View ───────────────────────────────────────────────
  const handleOpenDetail = async (pred) => {
    setSelectedPrediction(pred)
    setSelectedAssessment(null)
    setDetailError('')
    setLoadingDetail(true)

    try {
      const assessData = await fetchAssessment(pred.assessment_id)
      setSelectedAssessment(assessData)
    } catch (err) {
      setDetailError(
        err.response?.data?.detail || 'Failed to fetch detailed assessment indicators.'
      )
    } finally {
      setLoadingDetail(false)
    }
  }

  const handleCloseDetail = () => {
    setSelectedPrediction(null)
    setSelectedAssessment(null)
    setDetailError('')
  }

  // ── 2. Download Assessment PDF Action ─────────────────────────────────────
  const handleDownloadPdf = async (pred) => {
    try {
      setDownloadingId(pred.id)
      setError(null)

      // Fetch the actual saved 21 indicators for this specific assessment
      const assessData = await fetchAssessment(pred.assessment_id)

      const formattedDate = new Date(pred.created_at).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
      })

      generateAssessmentReportPdf({
        result: pred,
        inputs: assessData,
        personalInfo: {
          fullName: pred.user_name || user?.full_name || 'Assessee',
          email: user?.email || 'N/A',
          rawAge: assessData.Age ? `${assessData.Age * 5 + 15}` : 'N/A',
          sex: assessData.Sex,
          assessmentDate: formattedDate,
        },
        user,
      })
    } catch (err) {
      console.error('PDF download error:', err)
      setError(
        err.response?.data?.detail ||
          'Failed to generate PDF report for this assessment. Please try again.'
      )
    } finally {
      setDownloadingId(null)
    }
  }

  // ── 3. Delete Action & Modal ──────────────────────────────────────────────
  const handleOpenDeleteModal = (pred) => {
    setDeletingRecord(pred)
    setDeleteError('')
  }

  const handleCloseDeleteModal = () => {
    if (isDeleting) return
    setDeletingRecord(null)
    setDeleteError('')
  }

  const handleConfirmDelete = async () => {
    if (!deletingRecord || isDeleting) return

    setIsDeleting(true)
    setDeleteError('')

    try {
      await deleteAssessment(deletingRecord.assessment_id)
      setSuccessMessage(
        `Assessment #${deletingRecord.assessment_id} and its associated prediction record were permanently deleted.`
      )
      setDeletingRecord(null)
      // Refresh both history table and dashboard counters
      loadHistory(page)
      // Auto-clear success message after 5 seconds
      setTimeout(() => setSuccessMessage(''), 5000)
    } catch (err) {
      setDeleteError(
        err.response?.data?.detail ||
          'Failed to delete assessment. You may not have permission to delete this record.'
      )
    } finally {
      setIsDeleting(false)
    }
  }

  const totalPages = data ? Math.ceil(data.total / PAGE_SIZE) : 1

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 via-blue-50/20 to-slate-100 py-10 px-4 sm:px-6 lg:px-8">
      <div className="max-w-6xl mx-auto space-y-6">
        
        {/* ═════════════════════════════════════════════════════════════════ */}
        {/* PAGE HEADER                                                       */}
        {/* ═════════════════════════════════════════════════════════════════ */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-50 border border-blue-200 text-blue-700 text-xs font-semibold mb-2">
              <Clock className="w-3.5 h-3.5" />
              <span>PostgreSQL Longitudinal Storage</span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
              Assessment History
            </h1>
            <p className="text-xs sm:text-sm text-slate-500 mt-1">
              All prior diabetes risk evaluations associated with your account ({user?.email}).
            </p>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={() => navigate('/assess')}
              className="px-4 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-700 text-white text-xs font-bold transition-all shadow-sm flex items-center gap-1.5"
            >
              <span>+ New Assessment</span>
            </button>
            <button
              onClick={() => loadHistory(page)}
              className="p-2.5 rounded-xl border border-slate-200 bg-white text-slate-600 hover:bg-slate-50 transition-colors shadow-sm"
              title="Refresh History"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* ═════════════════════════════════════════════════════════════════ */}
        {/* STATS SUMMARY BAR                                                 */}
        {/* ═════════════════════════════════════════════════════════════════ */}
        {stats && (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="bg-white rounded-xl border border-slate-200 p-3.5 shadow-sm">
              <span className="text-[11px] text-slate-400 uppercase font-semibold">Total Assessed</span>
              <p className="text-xl font-bold text-slate-900 mt-0.5">{stats.total_assessments}</p>
            </div>
            {[
              { key: 0, label: 'No Diabetes' },
              { key: 1, label: 'Prediabetes' },
              { key: 2, label: 'Diabetes' },
            ].map(({ key, label }) => (
              <div key={key} className="bg-white rounded-xl border border-slate-200 p-3.5 shadow-sm">
                <span className="text-[11px] text-slate-400 uppercase font-semibold flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full" style={{ backgroundColor: CLASS_COLORS[key] }} />
                  {label}
                </span>
                <p className="text-xl font-bold text-slate-900 mt-0.5">
                  {stats.class_distribution?.[key] ?? 0}
                </p>
              </div>
            ))}
          </div>
        )}

        {/* ═════════════════════════════════════════════════════════════════ */}
        {/* SUCCESS NOTIFICATION BANNER                                       */}
        {/* ═════════════════════════════════════════════════════════════════ */}
        {successMessage && (
          <div className="p-4 rounded-xl bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs flex items-center justify-between gap-3 animate-in fade-in duration-200">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-600 flex-shrink-0" />
              <span className="font-semibold">{successMessage}</span>
            </div>
            <button
              type="button"
              onClick={() => setSuccessMessage('')}
              className="p-1 rounded-md text-emerald-600 hover:bg-emerald-100"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        )}

        {/* ═════════════════════════════════════════════════════════════════ */}
        {/* ERROR ALERT BANNER                                                */}
        {/* ═════════════════════════════════════════════════════════════════ */}
        {error && (
          <div className="p-4 rounded-xl bg-red-50 border border-red-200 text-red-700 text-xs flex items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-4 h-4 flex-shrink-0" />
              <span>{error}</span>
            </div>
            <button
              type="button"
              onClick={() => setError(null)}
              className="p-1 rounded-md text-red-600 hover:bg-red-100"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        )}

        {/* ═════════════════════════════════════════════════════════════════ */}
        {/* MAIN RECORDS TABLE CARD                                           */}
        {/* ═════════════════════════════════════════════════════════════════ */}
        <div className="bg-white rounded-2xl border border-slate-200/90 shadow-sm overflow-hidden">
          {loading && !error && (
            <div className="py-16 text-center text-slate-400 flex flex-col items-center justify-center">
              <RefreshCw className="w-6 h-6 animate-spin text-blue-600 mb-2" />
              <p className="text-xs">Loading records from database...</p>
            </div>
          )}

          {!loading && !error && data && (
            <>
              {data.items.length === 0 ? (
                <div className="py-16 text-center text-slate-400">
                  <Database className="w-10 h-10 mx-auto text-slate-300 mb-2" />
                  <p className="text-sm font-semibold text-slate-700">No assessment records found</p>
                  <p className="text-xs text-slate-400 mt-0.5 mb-4">
                    Submit your first health assessment to begin tracking your records.
                  </p>
                  <button onClick={() => navigate('/assess')} className="btn-primary text-xs py-2 px-4">
                    Take Health Assessment
                  </button>
                </div>
              ) : (
                <>
                  {/* Desktop Table View (Hidden on mobile) */}
                  <div className="hidden md:block overflow-x-auto">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="border-b border-slate-100 bg-slate-50/70 text-slate-500 font-semibold uppercase tracking-wider text-left">
                          <th className="py-3 px-4">Assessment ID</th>
                          <th className="py-3 px-4">Date &amp; Time</th>
                          <th className="py-3 px-4">User Name</th>
                          <th className="py-3 px-4">Predicted Class</th>
                          <th className="py-3 px-4">Confidence</th>
                          <th className="py-3 px-4">Probabilities</th>
                          <th className="py-3 px-4 text-right">Actions</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100">
                        {data.items.map((pred) => {
                          const cls = pred.predicted_class
                          const dateFormatted = new Date(pred.created_at).toLocaleString('en-US', {
                            month: 'short',
                            day: 'numeric',
                            year: 'numeric',
                            hour: '2-digit',
                            minute: '2-digit',
                          })
                          const userName = pred.user_name || user?.full_name || 'Assessee'
                          const isRowDownloading = downloadingId === pred.id

                          return (
                            <tr key={pred.id} className="hover:bg-slate-50/80 transition-colors">
                              <td className="py-3 px-4 font-mono font-medium text-slate-700">
                                <span>A#{pred.assessment_id}</span>
                                <span className="text-[10px] text-slate-400 block font-normal">
                                  Pred #{pred.id}
                                </span>
                              </td>
                              <td className="py-3 px-4 text-slate-600 whitespace-nowrap">{dateFormatted}</td>
                              <td className="py-3 px-4 font-medium text-slate-800">{userName}</td>
                              <td className="py-3 px-4">
                                <span
                                  className="inline-flex items-center gap-1.5 font-bold px-2.5 py-1 rounded-full text-[11px]"
                                  style={{
                                    backgroundColor: `${CLASS_COLORS[cls]}15`,
                                    color: CLASS_COLORS[cls],
                                    border: `1px solid ${CLASS_COLORS[cls]}30`,
                                  }}
                                >
                                  <span
                                    className="w-2 h-2 rounded-full"
                                    style={{ backgroundColor: CLASS_COLORS[cls] }}
                                  />
                                  {CLASS_LABELS[cls]} (Class {cls})
                                </span>
                              </td>
                              <td className="py-3 px-4 font-mono font-bold text-slate-800">
                                {(pred.confidence * 100).toFixed(1)}%
                              </td>
                              <td className="py-3 px-4">
                                <div className="flex flex-wrap gap-1">
                                  <ProbPill label="ND" value={pred.probability_no_diabetes} color="green" />
                                  <ProbPill label="Pre" value={pred.probability_prediabetes} color="amber" />
                                  <ProbPill label="DM" value={pred.probability_diabetes} color="red" />
                                </div>
                              </td>
                              <td className="py-3 px-4 text-right whitespace-nowrap">
                                <div className="flex items-center justify-end gap-1.5">
                                  {/* Action 1: View Assessment */}
                                  <button
                                    type="button"
                                    onClick={() => handleViewAssessment(pred)}
                                    className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg border border-slate-200 text-slate-700 hover:bg-blue-50 hover:text-blue-700 hover:border-blue-200 font-semibold transition-colors shadow-2xs"
                                    title="View full assessment report and analysis"
                                  >
                                    <Eye className="w-3.5 h-3.5 text-blue-600" />
                                    <span>View</span>
                                  </button>

                                  {/* Action 2: Download Assessment PDF */}
                                  <button
                                    type="button"
                                    onClick={() => handleDownloadPdf(pred)}
                                    disabled={isRowDownloading}
                                    className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg border border-slate-200 text-slate-700 hover:bg-slate-100 font-semibold transition-colors disabled:opacity-50 shadow-2xs"
                                    title={`Download PDF report for Assessment #${pred.assessment_id}`}
                                  >
                                    {isRowDownloading ? (
                                      <RefreshCw className="w-3.5 h-3.5 animate-spin text-blue-600" />
                                    ) : (
                                      <Download className="w-3.5 h-3.5 text-slate-600" />
                                    )}
                                    <span>{isRowDownloading ? 'PDF...' : 'PDF'}</span>
                                  </button>

                                  {/* Action 3: Delete Assessment */}
                                  <button
                                    type="button"
                                    onClick={() => handleOpenDeleteModal(pred)}
                                    className="inline-flex items-center gap-1 p-1.5 rounded-lg border border-slate-200 text-slate-500 hover:text-rose-600 hover:bg-rose-50 hover:border-rose-200 font-semibold transition-colors shadow-2xs"
                                    title={`Delete Assessment #${pred.assessment_id} permanently`}
                                  >
                                    <Trash2 className="w-3.5 h-3.5" />
                                  </button>
                                </div>
                              </td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>

                  {/* Mobile Card View (Shown on screens < md) */}
                  <div className="md:hidden divide-y divide-slate-100">
                    {data.items.map((pred) => {
                      const cls = pred.predicted_class
                      const dateFormatted = new Date(pred.created_at).toLocaleString('en-US', {
                        month: 'short',
                        day: 'numeric',
                        year: 'numeric',
                        hour: '2-digit',
                        minute: '2-digit',
                      })
                      const userName = pred.user_name || user?.full_name || 'Assessee'
                      const isRowDownloading = downloadingId === pred.id

                      return (
                        <div key={pred.id} className="p-4 space-y-3">
                          <div className="flex items-center justify-between">
                            <span className="font-mono text-xs font-bold text-slate-700">
                              Assessment #{pred.assessment_id}
                              <span className="text-[10px] text-slate-400 block font-normal">
                                Prediction #{pred.id}
                              </span>
                            </span>
                            <span className="text-[11px] text-slate-400">{dateFormatted}</span>
                          </div>

                          <div className="flex items-center justify-between flex-wrap gap-2">
                            <span
                              className="inline-flex items-center gap-1.5 font-bold px-2.5 py-1 rounded-full text-xs"
                              style={{
                                backgroundColor: `${CLASS_COLORS[cls]}15`,
                                color: CLASS_COLORS[cls],
                                border: `1px solid ${CLASS_COLORS[cls]}30`,
                              }}
                            >
                              <span
                                className="w-2 h-2 rounded-full"
                                style={{ backgroundColor: CLASS_COLORS[cls] }}
                              />
                              {CLASS_LABELS[cls]} (Class {cls})
                            </span>
                            <span className="text-xs font-mono font-bold text-slate-800">
                              Confidence: {(pred.confidence * 100).toFixed(1)}%
                            </span>
                          </div>

                          <div className="flex flex-wrap gap-1 pt-1">
                            <ProbPill label="ND" value={pred.probability_no_diabetes} color="green" />
                            <ProbPill label="Pre" value={pred.probability_prediabetes} color="amber" />
                            <ProbPill label="DM" value={pred.probability_diabetes} color="red" />
                          </div>

                          {/* Mobile Action Buttons */}
                          <div className="grid grid-cols-3 gap-2 pt-2 border-t border-slate-100">
                            <button
                              type="button"
                              onClick={() => handleViewAssessment(pred)}
                              className="flex items-center justify-center gap-1 py-2 px-2 rounded-xl bg-blue-50 text-blue-700 border border-blue-200 text-xs font-bold"
                            >
                              <Eye className="w-3.5 h-3.5" />
                              <span>View</span>
                            </button>
                            <button
                              type="button"
                              onClick={() => handleDownloadPdf(pred)}
                              disabled={isRowDownloading}
                              className="flex items-center justify-center gap-1 py-2 px-2 rounded-xl bg-slate-50 text-slate-700 border border-slate-200 text-xs font-semibold disabled:opacity-50"
                            >
                              {isRowDownloading ? (
                                <RefreshCw className="w-3.5 h-3.5 animate-spin text-blue-600" />
                              ) : (
                                <Download className="w-3.5 h-3.5 text-slate-600" />
                              )}
                              <span>{isRowDownloading ? 'PDF...' : 'PDF'}</span>
                            </button>
                            <button
                              type="button"
                              onClick={() => handleOpenDeleteModal(pred)}
                              className="flex items-center justify-center gap-1 py-2 px-2 rounded-xl bg-rose-50 text-rose-700 border border-rose-200 text-xs font-semibold"
                            >
                              <Trash2 className="w-3.5 h-3.5 text-rose-600" />
                              <span>Delete</span>
                            </button>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </>
              )}

              {/* Pagination controls */}
              {totalPages > 1 && (
                <div className="p-4 border-t border-slate-100 flex items-center justify-between flex-wrap gap-2 text-xs text-slate-600">
                  <span>
                    Showing {(page - 1) * PAGE_SIZE + 1}–{Math.min(page * PAGE_SIZE, data.total)} of{' '}
                    <strong>{data.total}</strong> records
                  </span>
                  <div className="flex items-center gap-1">
                    <button
                      onClick={() => setPage((p) => Math.max(1, p - 1))}
                      disabled={page === 1}
                      className="p-1.5 rounded border border-slate-200 disabled:opacity-30 disabled:cursor-not-allowed hover:bg-slate-50"
                      title="Previous Page"
                    >
                      <ChevronLeft className="w-4 h-4" />
                    </button>
                    <span className="px-3 font-semibold text-slate-800">
                      {page} / {totalPages}
                    </span>
                    <button
                      onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                      disabled={page === totalPages}
                      className="p-1.5 rounded border border-slate-200 disabled:opacity-30 disabled:cursor-not-allowed hover:bg-slate-50"
                      title="Next Page"
                    >
                      <ChevronRight className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        {/* ═════════════════════════════════════════════════════════════════ */}
        {/* DELETE CONFIRMATION MODAL                                         */}
        {/* ═════════════════════════════════════════════════════════════════ */}
        {deletingRecord && (
          <div className="fixed inset-0 z-50 bg-slate-900/60 backdrop-blur-xs flex items-center justify-center p-4 overflow-y-auto">
            <div className="bg-white rounded-3xl border border-slate-200 shadow-2xl max-w-md w-full p-6 space-y-4 animate-in fade-in zoom-in-95 duration-200">
              <div className="w-12 h-12 rounded-2xl bg-rose-50 border border-rose-200 flex items-center justify-center text-rose-600 mx-auto">
                <AlertTriangle className="w-6 h-6" />
              </div>

              <div className="text-center">
                <h3 className="text-lg font-bold text-slate-900">Delete Assessment?</h3>
                <p className="text-xs text-slate-500 mt-2 leading-relaxed">
                  Are you sure you want to permanently delete assessment{' '}
                  <strong className="text-slate-800">#{deletingRecord.assessment_id}</strong>{' '}
                  and its associated prediction record? This action cannot be undone.
                </p>
              </div>

              {deleteError && (
                <div className="p-3 rounded-xl bg-rose-50 border border-rose-200 text-rose-700 text-xs">
                  {deleteError}
                </div>
              )}

              <div className="grid grid-cols-2 gap-3 pt-2">
                <button
                  type="button"
                  onClick={handleCloseDeleteModal}
                  disabled={isDeleting}
                  className="py-2.5 px-4 rounded-xl border border-slate-200 text-slate-700 hover:bg-slate-50 font-semibold text-xs transition-colors disabled:opacity-50"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={handleConfirmDelete}
                  disabled={isDeleting}
                  className="py-2.5 px-4 rounded-xl bg-rose-600 hover:bg-rose-700 text-white font-bold text-xs transition-colors shadow-sm disabled:opacity-50 flex items-center justify-center gap-1.5"
                >
                  {isDeleting ? (
                    <>
                      <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                      <span>Deleting...</span>
                    </>
                  ) : (
                    <span>Delete Assessment</span>
                  )}
                </button>
              </div>
            </div>
          </div>
        )}

      </div>
    </div>
  )
}
