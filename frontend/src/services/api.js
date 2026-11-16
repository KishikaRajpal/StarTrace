// services/api.js - Axios API client for StarTrace backend

import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 120000,  // 2 minutes for analysis (BLS + CNN can take time on CPU)
})

// Request interceptor for logging
api.interceptors.request.use(config => {
  return config
})

// Response interceptor for error normalisation
api.interceptors.response.use(
  res => res,
  err => {
    const msg = err.response?.data?.detail || err.message || 'Network error'
    return Promise.reject(new Error(msg))
  }
)

// --- Analysis endpoints ---

export const runAnalysisByTic = (ticId, minPeriod = 0.5, maxPeriod = 15.0) => {
  const form = new FormData()
  form.append('tic_id', String(ticId))
  form.append('min_period', String(minPeriod))
  form.append('max_period', String(maxPeriod))
  return api.post('/analysis/run', form)
}

export const uploadCSVAnalysis = (file, minPeriod = 0.5, maxPeriod = 15.0) => {
  const form = new FormData()
  form.append('file', file)
  form.append('min_period', String(minPeriod))
  form.append('max_period', String(maxPeriod))
  return api.post('/analysis/upload', form)
}

export const getAnalysis = (analysisId) =>
  api.get(`/analysis/${analysisId}`)

export const getAnalysisVisualization = (analysisId) =>
  api.get(`/analysis/${analysisId}/visualization`)

export const listAnalyses = (params = {}) =>
  api.get('/analysis', { params })

export const deleteAnalysis = (analysisId) =>
  api.delete(`/analysis/${analysisId}`)

export const getDashboardStats = () =>
  api.get('/analysis/dashboard')

// --- Model endpoints ---

export const getModelStatus = () =>
  api.get('/model/status')

export const getModelMetrics = () =>
  api.get('/model/metrics')

export const getModelArchitecture = () =>
  api.get('/model/architecture')

// --- Health & misc ---

export const getHealth = () =>
  api.get('/health')

export const getSampleDatasets = () =>
  api.get('/samples')

// --- Report endpoints ---

export const getCSVReportUrl = (analysisId) =>
  `/api/report/${analysisId}/csv`

export const getPDFReportUrl = (analysisId) =>
  `/api/report/${analysisId}/pdf`

export default api
