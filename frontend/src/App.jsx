/**
 * src/App.jsx
 * ===========
 * Root application with React Router routes and Navbar layout.
 */

import { Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import Landing from './pages/Landing'
import Dashboard from './pages/Dashboard'
import HealthAssessmentForm from './pages/HealthAssessmentForm'
import PredictionResult from './pages/PredictionResult'
import ModelPerformance from './pages/ModelPerformance'
import AboutMethodology from './pages/AboutMethodology'

export default function App() {
  return (
    <div className="min-h-screen bg-slate-50">
      <Navbar />
      {/* Top padding to clear fixed navbar */}
      <main className="pt-16">
        <Routes>
          <Route path="/"           element={<Landing />} />
          <Route path="/dashboard"  element={<Dashboard />} />
          <Route path="/assess"     element={<HealthAssessmentForm />} />
          <Route path="/result"     element={<PredictionResult />} />
          <Route path="/performance" element={<ModelPerformance />} />
          <Route path="/about"      element={<AboutMethodology />} />
          {/* 404 fallback */}
          <Route path="*" element={
            <div className="max-w-xl mx-auto text-center py-32 px-4">
              <div className="text-6xl font-extrabold text-slate-200 mb-4">404</div>
              <p className="text-slate-500 mb-6">Page not found.</p>
              <a href="/" className="btn-primary inline-block">Go Home</a>
            </div>
          } />
        </Routes>
      </main>
    </div>
  )
}

