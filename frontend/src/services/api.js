/**
 * src/services/api.js
 * ===================
 * Axios-based API client for the FastAPI backend.
 * All requests go to /api (proxied to localhost:8000 in dev).
 * Phase 6.6: adds stats, assessments, and predictions history endpoints.
 */

import axios from 'axios'

const BASE_URL = '/api'

const apiClient = axios.create({
  baseURL: BASE_URL,
  timeout: 15000,
  headers: { 'Content-Type': 'application/json' },
})

// Automatically attach JWT bearer token if user is signed in
apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('diabetes_ai_token') || sessionStorage.getItem('diabetes_ai_token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
}, (error) => Promise.reject(error))

/**
 * GET /api/health
 * Returns { status, timestamp, model_loaded, database_connected, version }
 */
export async function fetchHealth() {
  const { data } = await apiClient.get('/health')
  return data
}

/**
 * GET /api/model-info
 * Returns model metadata, feature list, class mapping, evaluation performance
 */
export async function fetchModelInfo() {
  const { data } = await apiClient.get('/model-info')
  return data
}

/**
 * POST /api/predict
 * @param {Object} features - 21 CDC feature fields
 * Returns PredictAndSaveResponse: assessment_id, prediction_id, predicted_class,
 * predicted_label, risk_tier, confidence, probability_no_diabetes,
 * probability_prediabetes, probability_diabetes, model_version, disclaimer
 */
export async function predict(features) {
  const { data } = await apiClient.post('/predict', features)
  return data
}

/**
 * GET /api/stats
 * Returns { total_assessments, total_predictions, class_distribution, recent_assessments }
 */
export async function fetchStats() {
  const { data } = await apiClient.get('/stats')
  return data
}

/**
 * GET /api/assessments?page=1&page_size=20
 * Returns { total, page, page_size, items: Assessment[] }
 */
export async function fetchAssessments(page = 1, pageSize = 20) {
  const { data } = await apiClient.get('/assessments', {
    params: { page, page_size: pageSize },
  })
  return data
}

/**
 * GET /api/assessments/:id
 */
export async function fetchAssessment(id) {
  const { data } = await apiClient.get(`/assessments/${id}`)
  return data
}

/**
 * GET /api/assessments/:id/prediction
 */
export async function fetchAssessmentPrediction(id) {
  const { data } = await apiClient.get(`/assessments/${id}/prediction`)
  return data
}

/**
 * DELETE /api/assessments/:id
 * Permanently deletes the specified assessment and its associated prediction.
 */
export async function deleteAssessment(id) {
  const { data } = await apiClient.delete(`/assessments/${id}`)
  return data
}

/**
 * GET /api/predictions?page=1&page_size=20
 * Returns { total, page, page_size, items: Prediction[] }
 */
export async function fetchPredictions(page = 1, pageSize = 20) {
  const { data } = await apiClient.get('/predictions', {
    params: { page, page_size: pageSize },
  })
  return data
}

/**
 * GET /api/predictions/:id
 */
export async function fetchPrediction(id) {
  const { data } = await apiClient.get(`/predictions/${id}`)
  return data
}

export default apiClient
