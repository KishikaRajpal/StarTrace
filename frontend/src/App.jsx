import React from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import LandingPage from './pages/LandingPage'
import DashboardLayout from './layouts/DashboardLayout'
import Overview from './pages/Overview'
import NewAnalysis from './pages/NewAnalysis'
import AnalysisResult from './pages/AnalysisResult'
import AnalysisHistory from './pages/AnalysisHistory'
import ModelInfo from './pages/ModelInfo'
import AboutPage from './pages/AboutPage'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/dashboard" element={<DashboardLayout />}>
          <Route index element={<Overview />} />
          <Route path="analyze" element={<NewAnalysis />} />
          <Route path="result/:analysisId" element={<AnalysisResult />} />
          <Route path="history" element={<AnalysisHistory />} />
          <Route path="model" element={<ModelInfo />} />
          <Route path="about" element={<AboutPage />} />
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  )
}
