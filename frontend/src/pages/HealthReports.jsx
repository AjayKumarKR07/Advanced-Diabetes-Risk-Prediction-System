/**
 * src/pages/HealthReports.jsx
 * ============================
 * Phase 6.17.1 — Health Reports page.
 * Allows authenticated users to generate and download personal health analytics
 * PDF reports based on their actual assessment records.
 * Reuses the existing generatePdfReport.js utility (lazy jsPDF import).
 */

import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import {
  FileText,
  Download,
  Loader2,
  CheckCircle2,
  AlertCircle,
  Clock,
  PlusCircle,
  Info,
  RefreshCw,
} from 'lucide-react'
import { useAuth } from '../context/AuthContext'
import { fetchAssessments, fetchPredictions } from '../services/api'
import { generateAssessmentReportPdf } from '../utils/generatePdfReport'
import DisclaimerBanner from '../components/DisclaimerBanner'

export default function HealthReports() {
  const { user } = useAuth()

  const [assessments, setAssessments] = useState([])
  const [predictions, setPredictions] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [generatingId, setGeneratingId] = useState(null)
  const [lastGenerated, setLastGenerated] = useState(null)
  const [genError, setGenError] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    Promise.all([
      fetchAssessments(1, 10).catch(() => ({ items: [], total: 0 })),
      fetchPredictions(1, 10).catch(() => ({ items: [], total: 0 })),
    ])
      .then(([assessData, predData]) => {
        const aItems = Array.isArray(assessData) ? assessData : assessData?.items ?? []
        const pItems = Array.isArray(predData) ? predData : predData?.items ?? []
        setAssessments(aItems)
        setPredictions(pItems)
      })
      .catch(() => setError('Could not load your assessment records.'))
      .finally(() => setLoading(false))
  }, [])

  const CLASS_LABELS = { 0: 'No Diabetes', 1: 'Pre-Diabetes', 2: 'Diabetes' }
  const CLASS_COLORS = {
    0: 'bg-emerald-50 text-emerald-700 border-emerald-200',
    1: 'bg-amber-50 text-amber-700 border-amber-200',
    2: 'bg-red-50 text-red-700 border-red-200',
  }

  async function handleGenerateLatest() {
    if (!assessments.length || !predictions.length) return
    setGeneratingId('latest')
    setGenError(null)
    try {
      const latestAssessment = assessments[0]
      const latestPrediction = predictions.find(
        (p) => p.assessment_id === latestAssessment?.id
      ) ?? predictions[0]

      await generateAssessmentReportPdf({
        result: latestPrediction,
        inputs: latestAssessment?.input_features ?? latestAssessment ?? {},
        personalInfo: {
          fullName: user?.full_name ?? 'User',
          email: user?.email ?? 'N/A',
          assessmentDate: new Date(
            latestAssessment?.created_at ?? Date.now()
          ).toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'long',
            day: 'numeric',
          }),
        },
        user,
        historicalStats: null,
      })
      setLastGenerated(new Date())
    } catch (err) {
      console.error('[HealthReports] PDF error:', err)
      setGenError('PDF generation failed. Please try again.')
    } finally {
      setGeneratingId(null)
    }
  }

  async function handleGenerateForAssessment(assessment, prediction) {
    setGeneratingId(assessment.id)
    setGenError(null)
    try {
      await generateAssessmentReportPdf({
        result: prediction,
        inputs: assessment?.input_features ?? assessment ?? {},
        personalInfo: {
          fullName: user?.full_name ?? 'User',
          email: user?.email ?? 'N/A',
          assessmentDate: new Date(
            assessment?.created_at ?? Date.now()
          ).toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'long',
            day: 'numeric',
          }),
        },
        user,
        historicalStats: null,
      })
      setLastGenerated(new Date())
    } catch (err) {
      console.error('[HealthReports] PDF error:', err)
      setGenError('PDF generation failed. Please try again.')
    } finally {
      setGeneratingId(null)
    }
  }

  const hasData = assessments.length > 0

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
      {/* Page Header */}
      <div className="flex items-start gap-3">
        <div className="w-10 h-10 bg-blue-50 rounded-xl flex items-center justify-center flex-shrink-0">
          <FileText className="w-5 h-5 text-blue-600" />
        </div>
        <div>
          <h1 className="text-xl font-bold text-slate-900">Health Reports</h1>
          <p className="text-sm text-slate-500 mt-0.5">
            Generate and download clinical summary PDF reports based on your actual assessment records.
          </p>
        </div>
      </div>

      {/* Disclaimer */}
      <DisclaimerBanner />

      {/* Loading */}
      {loading && (
        <div className="flex items-center gap-2 text-slate-500 py-6">
          <Loader2 className="w-4 h-4 animate-spin" />
          <span className="text-sm">Loading your assessment records…</span>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="flex items-center gap-2 bg-red-50 border border-red-200 text-red-700 rounded-xl px-4 py-3 text-sm">
          <AlertCircle className="w-4 h-4 shrink-0" />
          {error}
        </div>
      )}

      {/* Gen Error */}
      {genError && (
        <div className="flex items-center gap-2 bg-amber-50 border border-amber-200 text-amber-700 rounded-xl px-4 py-3 text-sm">
          <AlertCircle className="w-4 h-4 shrink-0" />
          {genError}
        </div>
      )}

      {/* Success banner */}
      {lastGenerated && !genError && (
        <div className="flex items-center gap-2 bg-emerald-50 border border-emerald-200 text-emerald-700 rounded-xl px-4 py-3 text-sm">
          <CheckCircle2 className="w-4 h-4 shrink-0" />
          Report generated and downloaded at {lastGenerated.toLocaleTimeString()}
        </div>
      )}

      {!loading && !error && (
        <>
          {/* Quick Generate Card */}
          <div className="bg-white border border-slate-200 rounded-2xl shadow-sm p-6">
            <div className="flex items-start justify-between gap-4">
              <div>
                <h2 className="text-base font-semibold text-slate-800 flex items-center gap-2">
                  <Download className="w-4 h-4 text-blue-600" />
                  Generate Personal Health Report
                </h2>
                <p className="text-xs text-slate-500 mt-1">
                  Creates a multi-page PDF containing your latest assessment, risk probability,
                  feature contribution analysis, and actionable health education notes.
                </p>
              </div>
            </div>

            <div className="mt-4 flex flex-col sm:flex-row items-start sm:items-center gap-3">
              <button
                onClick={handleGenerateLatest}
                disabled={!hasData || generatingId === 'latest'}
                className={`flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-semibold transition-all ${
                  hasData
                    ? 'bg-blue-600 text-white hover:bg-blue-700 shadow-sm'
                    : 'bg-slate-100 text-slate-400 cursor-not-allowed'
                }`}
              >
                {generatingId === 'latest' ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Download className="w-4 h-4" />
                )}
                {generatingId === 'latest' ? 'Generating…' : 'Download Latest Report'}
              </button>

              {!hasData && (
                <Link
                  to="/assess"
                  className="flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-semibold text-teal-700 bg-teal-50 border border-teal-200 hover:bg-teal-100 transition-colors"
                >
                  <PlusCircle className="w-4 h-4" />
                  Complete Your First Assessment
                </Link>
              )}
            </div>

            {hasData && (
              <div className="mt-3 flex items-center gap-1.5 text-xs text-slate-400">
                <Info className="w-3 h-3" />
                Report will be downloaded as a PDF file. No data leaves your device.
              </div>
            )}
          </div>

          {/* Assessment-Specific Reports */}
          <div className="bg-white border border-slate-200 rounded-2xl shadow-sm overflow-hidden">
            <div className="px-6 py-4 border-b border-slate-100 flex items-center justify-between">
              <h2 className="text-base font-semibold text-slate-800 flex items-center gap-2">
                <Clock className="w-4 h-4 text-slate-500" />
                Assessment Records ({assessments.length})
              </h2>
              <Link
                to="/history"
                className="text-xs text-blue-600 hover:text-blue-800 font-medium"
              >
                View All →
              </Link>
            </div>

            {assessments.length === 0 ? (
              <div className="px-6 py-12 text-center">
                <FileText className="w-10 h-10 text-slate-200 mx-auto mb-3" />
                <p className="text-sm font-medium text-slate-600">No assessments yet</p>
                <p className="text-xs text-slate-400 mt-1 mb-4">
                  Complete a health assessment to generate your first report.
                </p>
                <Link
                  to="/assess"
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-blue-600 text-white text-sm font-semibold hover:bg-blue-700 transition-colors"
                >
                  <PlusCircle className="w-4 h-4" />
                  New Assessment
                </Link>
              </div>
            ) : (
              <div className="divide-y divide-slate-50">
                {assessments.slice(0, 8).map((assessment) => {
                  const prediction = predictions.find(
                    (p) => p.assessment_id === assessment.id
                  ) ?? null
                  const cls = prediction?.predicted_class ?? null
                  const isGen = generatingId === assessment.id

                  return (
                    <div
                      key={assessment.id}
                      className="px-6 py-4 flex items-center justify-between gap-4 hover:bg-slate-50 transition-colors"
                    >
                      <div className="flex items-center gap-3 min-w-0">
                        <div className="w-8 h-8 rounded-lg bg-blue-50 flex items-center justify-center shrink-0">
                          <FileText className="w-4 h-4 text-blue-500" />
                        </div>
                        <div className="min-w-0">
                          <p className="text-sm font-semibold text-slate-800 truncate">
                            Assessment #{assessment.id}
                          </p>
                          <p className="text-xs text-slate-400 flex items-center gap-1 mt-0.5">
                            <Clock className="w-3 h-3" />
                            {new Date(assessment.created_at).toLocaleDateString('en-US', {
                              year: 'numeric',
                              month: 'short',
                              day: 'numeric',
                            })}
                          </p>
                        </div>
                      </div>

                      <div className="flex items-center gap-3 shrink-0">
                        {cls !== null && (
                          <span
                            className={`text-[11px] font-semibold px-2.5 py-1 rounded-full border ${
                              CLASS_COLORS[cls] ?? 'bg-slate-50 text-slate-600 border-slate-200'
                            }`}
                          >
                            {CLASS_LABELS[cls] ?? `Class ${cls}`}
                          </span>
                        )}
                        <button
                          onClick={() =>
                            handleGenerateForAssessment(assessment, prediction ?? {})
                          }
                          disabled={isGen}
                          title="Download PDF report for this assessment"
                          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold text-blue-600 border border-blue-200 hover:bg-blue-50 transition-colors disabled:opacity-50"
                        >
                          {isGen ? (
                            <Loader2 className="w-3 h-3 animate-spin" />
                          ) : (
                            <Download className="w-3 h-3" />
                          )}
                          {isGen ? 'Generating…' : 'PDF'}
                        </button>
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </div>

          {/* Info Card */}
          <div className="bg-slate-50 border border-slate-200 rounded-xl px-5 py-4">
            <div className="flex items-start gap-2">
              <Info className="w-4 h-4 text-slate-400 shrink-0 mt-0.5" />
              <div className="text-xs text-slate-500 space-y-1">
                <p className="font-semibold text-slate-600">About Health Reports</p>
                <p>
                  Reports are generated entirely in your browser using your stored assessment data.
                  They include risk probability scores, feature contribution analysis (XAI),
                  longitudinal health notes, and actionable education guidelines.
                </p>
                <p>
                  <strong className="text-slate-600">These reports are for educational purposes only</strong> and
                  do not constitute a medical diagnosis. Always consult a licensed healthcare
                  provider for clinical evaluation.
                </p>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
