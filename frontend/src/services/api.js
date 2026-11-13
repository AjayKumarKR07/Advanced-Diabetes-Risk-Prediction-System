/**
 * src/services/api.js
 * ===================
 * Axios-based API client for the FastAPI backend.
 * All requests go to /api (proxied to localhost:8000 in dev).
 */

import axios from 'axios'

const BASE_URL = '/api'

const apiClient = axios.create({
  baseURL: BASE_URL,
  timeout: 15000,
  headers: { 'Content-Type': 'application/json' },
})

/**
 * GET /api/health
 * Returns { status, timestamp, model_loaded, version }
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
 * Returns prediction, probabilities, risk tier, confidence
 */
export async function predict(features) {
  const { data } = await apiClient.post('/predict', features)
  return data
}

export default apiClient

